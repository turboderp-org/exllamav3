from __future__ import annotations
from typing_extensions import override
import torch

from ..cache import Cache
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, TransformerBlock, Attention, GatedMLP
from ..modules.arch_specific.dflash import (
    DFlashInputLayer,
    DFlashRing,
    DFlashRingAttention,
    dflash_ring_slots,
)
from ..modules.attn import prepare_for_attn
from ..util.device_copy import to_device
import weakref

from ..util.tensor import get_for_device
import os

# TODO: Support DFlash models trained in Speculators (includes lm_head for speculator with limited vocabulary?)

class DFlashConfig(Config):
    arch_string = "DFlashDraftModel"

    # Offset from the checkpoint's target_layer_ids to exllamav3 export indices (which denote the
    # OUTPUT of layer j). The original DFlash release needs +1 (determined empirically); variants
    # whose reference uses hidden_states[i + 1] (output of layer i) use raw ids
    tap_shift = 1

    def __init__(
        self,
        directory: str,
        model_classes: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            directory,
            model_classes or {"text": DFlashModel},
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        # self.num_target_layers = self.read_cfg(int, "num_target_layers", no_default)
        self.layer_types = self.read_cfg(list, "layer_types", ["full_attention"] * self.num_hidden_layers)
        self.sliding_window = self.read_cfg(int, "sliding_window", 2048)

        # DFlash. Config keys live under dflash_config-> in the original release, at the top
        # level in later ones (MuseGlimmerAssistant)
        self.mask_token_id = self.read_cfg(int, ["dflash_config->mask_token_id", "mask_token_id"], no_default)
        self.target_layer_ids = self.read_cfg(list, ["dflash_config->target_layer_ids", "target_layer_ids"], no_default)
        # The offset is per checkpoint and not derivable from the config: gemma4-31b-it-dflash
        # wants +1 (2.4-2.9 vs 0.3 accepted/round), gemma4-26b-a4b-it-dflash wants 0 (3.2 vs
        # 0.8), same trainer version. A checkpoint (or its quantized config.json) can pin it with
        # "tap_shift" under dflash_config or at the top level
        self.tap_shift = self.read_cfg(int, ["dflash_config->tap_shift", "tap_shift"], self.tap_shift)
        self.target_layer_ids = [i + self.tap_shift for i in self.target_layer_ids]
        assert len(set(self.target_layer_ids)) == len(self.target_layer_ids), \
            "DFlash target_layer_ids must be unique"
        self.block_size = self.read_cfg(int, ["block_size", "dflash_config->block_size"], no_default)

        # --- variant switches -------------------------------------------------------------
        # These cover checkpoints whose drafter is not the plain z-lab one. MiMo-V2.6-Flash-RL's
        # `dflash/` is the first: 5 sliding-window layers with gpt-oss style learned sinks, a
        # value scale, a learned mask embedding that is NOT in the target's embedding table, and
        # `is_causal: false`. All four switches default to the previous behaviour, so the
        # original z-lab checkpoints load and run exactly as before.

        # Learned per-q-head attention sinks (extra logit column dropped after the softmax),
        # stored as layers.N.self_attn.attention_sink_bias. Declared by the config or simply
        # detected in the checkpoint
        self.attention_sink_bias = self.read_cfg(
            bool,
            ["dflash_config->attention_sink_bias", "attention_sink_bias", "add_swa_attention_sink_bias"],
            None,
        )
        if self.attention_sink_bias is None:
            self.attention_sink_bias = self.stc.has_tensor("layers.0.self_attn.attention_sink_bias")

        # V is scaled before the attention output; attention is linear in V (the sink column
        # carries no value), so this folds exactly into o_proj
        self.attention_value_scale = self.read_cfg(
            float, ["dflash_config->attention_value_scale", "attention_value_scale"], None
        ) or 1.0

        # `is_causal: false` with a sliding window. causal = False alone is not enough: the
        # kernels read a bare int window as (left, 0), which re-imposes causality inside the
        # drafted block. The bidirectional block is a window of (sliding_window, block_size - 1)
        self.bidirectional_block = self.read_cfg(
            bool, ["dflash_config->bidirectional_block", "bidirectional_block"], None
        )
        if self.bidirectional_block is None:
            self.bidirectional_block = not self.read_cfg(bool, "is_causal", True)

        # Learned mask embedding. The original DFlash release reuses the target's embedding row
        # for mask_token_id; checkpoints that ship their own vector (MiMo) carry it as a
        # "mask_embedding" tensor in the draft directory
        self.key_mask_embedding = "mask_embedding" if self.stc.has_tensor("mask_embedding") else None

        # Sliding-window draft layers keep a fixed per-slot ring instead of a paged cache
        # sized for the whole context. Config key, else EXL3_DFLASH_RING (0 restores the paged
        # cache for A/B). Only engaged when *every* layer is sliding: the ring rewrites the
        # block table for the whole forward pass, so a mixed drafter would have to carry two
        self.draft_ring = self.read_cfg(
            bool, ["dflash_config->draft_ring", "draft_ring"], None
        )
        if self.draft_ring is None:
            self.draft_ring = os.environ.get("EXL3_DFLASH_RING", "1") != "0"
        self.draft_ring = self.draft_ring and all(t == "sliding_attention" for t in self.layer_types)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)

        # Vision placeholders
        self.vision = None


def dflash_update_kv_from_target(
    model: Model,
    target_hidden: list,
    cache: Cache,
    params: dict,
    lengths: list[int] = None,
):
    """
    Update a DFlash-style draft's K/V cache with hidden states extracted from the target model.
    Shared by every drafter built on the DFlash encoder (fc + hidden_norm over concatenated taps)
    with plain GQA attention layers: model.input_layer, model.attn_modules and
    model.config.target_layer_ids are what it reads.

    params:
        "block_table": torch.Tensor
        "cache_seqlens": torch.Tensor
    """

    # Target states arrive in layer execution order. Reorder them only when the checkpoint's
    # projection expects a different target_layer_ids order.
    target_layer_ids = model.config.target_layer_ids
    if target_layer_ids != sorted(target_layer_ids):
        source_idx = {layer_id: idx for idx, layer_id in enumerate(sorted(target_layer_ids))}
        target_hidden = [target_hidden[source_idx[layer_id]] for layer_id in target_layer_ids]

    # May update a few redundant tokens when batching, but we'd never draft longer than the cache length
    if lengths is not None:
        max_length = max(lengths)
        target_hidden = [t[:, :max_length] for t in target_hidden]

    # Ensure all state snapshots are on the same device
    device = model.input_layer.device
    for i in range(len(target_hidden)):
        target_hidden[i] = to_device(target_hidden[i], device)

    # Projection concatenated states to hidden size, once
    target_hidden = torch.cat(target_hidden, dim = -1)
    target_hidden = model.input_layer.proj.forward(target_hidden, {}, out_dtype = torch.half)
    target_hidden = model.input_layer.norm.forward(target_hidden, {}, out_dtype = torch.half)

    bsz, target_seqlen, dim = target_hidden.shape
    params["target_hidden_cc"] = target_hidden

    # Ring drafters address their own fixed buffer, not the target's page table. The ring
    # views are the same for every layer, so build them once; RoPE still uses the absolute
    # positions in cache_seqlens
    ring = getattr(model, "dflash_ring", None)
    if ring is not None:
        ring_slots = dflash_ring_slots(params, bsz)
        ring_positions = [int(p) for p in params["cache_seqlens"].flatten().tolist()]

    # Update KV layers
    for layer in model.attn_modules:
        block_table = get_for_device(params, "block_table", layer.device) if ring is None else None
        cache_seqlens = get_for_device(params, "cache_seqlens", layer.device)
        target_hidden = get_for_device(params, "target_hidden_cc", layer.device)

        # k/v project
        k = layer.k_proj.forward(target_hidden, params)
        v = layer.v_proj.forward(target_hidden, params)
        k = k.view(bsz, target_seqlen, layer.num_kv_heads, layer.head_dim)
        v = v.view(bsz, target_seqlen, layer.num_kv_heads, layer.head_dim)

        # Apply rope and norm to k
        k, _ = layer.rope.apply(
            k, None,
            0,
            cache_seqlens,
            None,
            True,
            layer.k_norm_tensor,
            None,
            layer.norm_eps,
            layer.norm_constant_bias,
            None,
        )

        # Write k, v rows to the paged cache; quantized caches quantize them in place rather
        # than dequantizing/requantizing full layers
        if ring is None:
            cache.update_layer_direct(layer.layer_idx, cache_seqlens, block_table, k, v, target_seqlen, 0)
        else:
            for t, c, r_bt, r_sl in ring.write_views(ring_slots, ring_positions, target_seqlen, layer.device):
                cache.update_layer_direct(
                    layer.layer_idx, r_sl, r_bt,
                    k[:, t : t + c].contiguous(), v[:, t : t + c].contiguous(), c, 0
                )

    if ring is not None:
        ring.note_write(ring_slots, ring_positions, target_seqlen)


class DFlashModel(Model):
    config_class = DFlashConfig

    # Encoder tensor keys; overridden by variants with a different namespace
    key_fc = "fc"
    key_fc_norm = "hidden_norm"

    def __init__(
        self,
        config: DFlashConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.input_layer = DFlashInputLayer(
            config = config,
            key = self.key_fc,
            key_norm = self.key_fc_norm,
            hidden_size = config.hidden_size,
            target_state_size = config.hidden_size * len(config.target_layer_ids),
            mask_token_id = config.mask_token_id,
            rms_norm_eps = config.rms_norm_eps,
            native_draft_len = config.block_size,
            key_mask_embedding = config.key_mask_embedding,
            qmap = "target_hidden",
        )
        self.modules += [self.input_layer]

        self.first_block_idx = len(self.modules)
        self.attn_modules = []

        # One ring shared by every sliding-window layer (they share window and block size), so
        # the block table is computed once per forward in prepare_inputs()
        self.dflash_ring = DFlashRing(config.sliding_window, config.block_size) \
            if config.draft_ring else None

        for idx in range(config.num_hidden_layers):
            is_swa = config.layer_types[idx] == "sliding_attention"
            attn_cls = DFlashRingAttention if (is_swa and self.dflash_ring is not None) else Attention
            attn_extra = {"ring": self.dflash_ring} if attn_cls is DFlashRingAttention else {}

            attn = attn_cls(
                config = config,
                key = f"layers.{idx}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                head_dim = config.head_dim,
                num_q_heads = config.num_q_heads,
                num_kv_heads = config.num_kv_heads,
                rope_settings = config.rope_settings,
                key_q = "q_proj",
                key_k = "k_proj",
                key_v = "v_proj",
                key_o = "o_proj",
                qmap = "block.attn",
                sliding_window = config.sliding_window if is_swa else -1,
                # `is_causal: false`: the drafted block attends to itself in both directions.
                # With no window that follows from causal = False; with one it has to be an
                # explicit right bound, or the window's implied (left, 0) re-imposes causality
                window_right = (config.block_size - 1) if (is_swa and config.bidirectional_block) else 0,
                key_sinks = "attention_sink_bias" if config.attention_sink_bias else None,
                q_norm = RMSNorm(
                    config = config,
                    key = f"layers.{idx}.self_attn.q_norm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                k_norm = RMSNorm(
                    config = config,
                    key = f"layers.{idx}.self_attn.k_norm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                out_dtype = torch.float,
                **attn_extra,
            )
            # attention_value_scale multiplies V before the attention output; fold into o_proj
            attn.o_proj.weight_scale = config.attention_value_scale
            self.attn_modules.append(attn)

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = attn,
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                    ),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        self.modules += [
            RMSNorm(
                config = config,
                key = f"norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            )
        ]

        self.logit_layer_idx = None
        self.caps.update({
            "uncalibrated_quantize": True,
            "supports_tp": False,
            "attach_target": True,
            "dflash_draft": True,
            "dflash_ring": self.dflash_ring is not None,
            "default_draft_size": config.block_size - 1,
            "autosplit_load_fwd": False,
        })

        self.attached_model = None

        self.draft_verifier_params.update({
            "export_state_layers": set(config.target_layer_ids),
        })


    def attach_to(self, target):
        self.attached_model = weakref.ref(target)
        self.input_layer.attached_model = weakref.ref(target)


    def update_kv_from_target(
        self,
        target_hidden: list,
        cache: Cache,
        params: dict,
        lengths: list[int] = None,
    ):
        dflash_update_kv_from_target(self, target_hidden, cache, params, lengths)


    def sample_from_state(
        self,
        state: torch.Tensor,
        params: dict
    ) -> torch.Tensor:
        if not self.attached_model().loaded_tp:
            ll = self.attached_model().logit_layer_idx
            lm = self.attached_model().modules[ll]
            logits = lm.prepare_for_device(state, params)
            logits = lm.forward(logits, params)
            logits = logits[..., :self.attached_model().config.vocab_size]
            if params.get("export_draft_conf"):
                # Per-position confidence for the generator's draft truncation: the argmax logit
                # value separates converged from degenerate block positions far better than any
                # distribution-shape statistic (the softcapped head is near-flat either way)
                conf, ids = torch.max(logits, dim = -1)
                params["draft_conf"] = conf
                return ids
            return torch.argmax(logits, dim = -1)
        else:
            state = self.attached_model().tp_producer.send(state)
            argmax = self.attached_model().tp_dispatch_lm_head_argmax((state, {}))
            return argmax


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # The draft block attends to itself bidirectionally; causality on the sliding-window
        # layers is expressed through their window (left sw, right 0) instead
        params["causal"] = False
        if self.dflash_ring is not None and params.get("attn_mode") == "flash_attn":
            # Every layer is a ring layer (the config only enables the ring when they all are),
            # so rewrite the page addressing for the whole pass: RoPE keeps the absolute
            # positions, the kernels get the ring's rotated block table and in-ring past length
            bsz = input_ids.shape[0]
            cache_seqlens = params["cache_seqlens"]
            if params.get("positions") is None and params.get("position_ids") is None:
                params["positions"] = cache_seqlens
            slots = dflash_ring_slots(params, bsz)
            positions = [int(p) for p in cache_seqlens.flatten().tolist()]
            device = self.attn_modules[0].device
            bt, sl = self.dflash_ring.read_view(slots, positions, self.config.block_size, device)
            params["block_table"] = bt
            params["cache_seqlens"] = sl
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError()


    @classmethod
    @override
    def get_additional_compiled_tensors(cls, config: DFlashConfig) -> dict:
        # The fc norm is stored in DFlashInputLayer but doesn't match the fc module-key prefix
        tensors = dict(config.stc.list_tensors(prefix = cls.key_fc_norm))
        if config.key_mask_embedding:
            tensors.update(config.stc.list_tensors(prefix = config.key_mask_embedding))
        return tensors
