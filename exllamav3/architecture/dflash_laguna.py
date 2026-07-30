from __future__ import annotations
from typing_extensions import override
import torch

from ..cache import Cache
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, TransformerBlock, Attention, GatedMLP
from ..modules.arch_specific.dflash import DFlashInputLayer
from ..modules.attn import prepare_for_attn
from ..modules.module import no_p2p_copy
import weakref

from ..util.tensor import get_for_device

# DFlash speculator for Laguna targets (Poolside's Laguna-XS-2.1-DFlash). Differences from the
# original z-lab DFlash drafter (dflash.py): Laguna-style attention layers (fused qkv, QK-norm,
# per-head softplus output gate, uniform sliding-window), per-tap RMSNorms on the captured
# target hidden states before the fc projection, causal drafting (dflash_config.causal), and
# target hidden states taken at the RAW dflash_config.target_layer_ids ("output of layer i",
# cross-confirmed by eagle_aux_hidden_state_layer_ids = ids + 1 under vLLM's capture-point
# indexing), unlike dflash.py, which shifts the ids one layer deeper to match what its
# supported checkpoints were evidently trained on.

class DFlashLagunaConfig(Config):
    arch_string = "DFlashLagunaForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": DFlashLagunaModel},
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Attention output gate: o *= softplus(g_proj(x)), one gate value per head
        self.assert_cfg(str, "gating", "per-head")

        # MLP params
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers: uniform attention flavor only (matches the vLLM reference restriction)
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.layer_types = self.read_cfg(list, "layer_types", ["full_attention"] * self.num_hidden_layers)
        assert len(set(self.layer_types)) == 1, \
            "DFlashLaguna: drafter requires a uniform layer_types"
        self.is_swa = self.layer_types[0] == "sliding_attention"
        self.sliding_window = self.read_cfg(int, "sliding_window", 2048)

        # The draft shares the target's lm_head; a reduced draft vocabulary is not supported
        draft_vocab_size = self.read_cfg(int, "draft_vocab_size", self.vocab_size)
        assert draft_vocab_size == self.vocab_size, \
            "DFlashLaguna: draft_vocab_size must match the model vocab_size"

        # DFlash. target_layer_ids are used RAW: id i means the output of target layer i
        # (0-based). eagle_aux_hidden_state_layer_ids, when present, is the same set of taps in
        # vLLM's capture-point indexing (input of layer k = output of layer k - 1) and is
        # cross-checked here
        self.mask_token_id = self.read_cfg(int, "dflash_config->mask_token_id", no_default)
        self.target_layer_ids = self.read_cfg(list, "dflash_config->target_layer_ids", no_default)
        eagle_ids = self.read_cfg(list, "eagle_aux_hidden_state_layer_ids", None)
        if eagle_ids is not None:
            assert eagle_ids == [i + 1 for i in self.target_layer_ids], \
                "DFlashLaguna: eagle_aux_hidden_state_layer_ids inconsistent with dflash_config.target_layer_ids"
        self.block_size = self.read_cfg(int, "dflash_config->block_size", no_default)
        self.dflash_causal = self.read_cfg(bool, "dflash_config->causal", True)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)

        # Vision placeholders
        self.vision = None


class DFlashLagunaModel(Model):
    config_class = DFlashLagunaConfig

    def __init__(
        self,
        config: DFlashLagunaConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.input_layer = DFlashInputLayer(
            config = config,
            key = "fc",
            key_norm = "hidden_norm",
            key_aux_norms = "aux_hidden_norms",
            num_aux_norms = len(config.target_layer_ids),
            hidden_size = config.hidden_size,
            target_state_size = config.hidden_size * len(config.target_layer_ids),
            mask_token_id = config.mask_token_id,
            rms_norm_eps = config.rms_norm_eps,
            native_draft_len = config.block_size,
            qmap = "target_hidden",
        )
        self.modules += [self.input_layer]

        self.first_block_idx = len(self.modules)
        self.attn_modules = []
        self.attn_norms = []

        for idx in range(config.num_hidden_layers):
            attn = Attention(
                config = config,
                key = f"layers.{idx}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                head_dim = config.head_dim,
                num_q_heads = config.num_q_heads,
                num_kv_heads = config.num_kv_heads,
                rope_settings = config.rope_settings,
                key_fused_qkv = "qkv_proj",
                key_o = "o_proj",
                key_g = "g_proj",
                gate_softplus = True,
                qmap = "block.attn",
                sliding_window = config.sliding_window if config.is_swa else -1,
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
            )
            self.attn_modules.append(attn)

            attn_norm = RMSNorm(
                config = config,
                key = f"layers.{idx}.input_layernorm",
                rms_norm_eps = config.rms_norm_eps,
            )
            self.attn_norms.append(attn_norm)

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"layers.{idx}",
                    layer_idx = idx,
                    attn_norm = attn_norm,
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
            "default_draft_size": config.block_size - 1,
            "autosplit_load_fwd": False,
        })

        self.attached_model = None

        # Raw target layer ids: exported at the output of target layer i (see module comment)
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
        """
        Update K/V cache with hidden states extracted from target model

        params:
            "block_table": torch.Tensor
            "cache_seqlens": torch.Tensor
        """

        # May update a few redundant tokens when batching, but we'd never draft longer than the cache length
        if lengths is not None:
            max_length = max(lengths)
            target_hidden = [t[:, :max_length] for t in target_hidden]

        # Ensure all state snapshots are on the same device
        device = self.input_layer.device
        for i in range(len(target_hidden)):
            if target_hidden[i].device != device:
                if no_p2p_copy:
                    target_hidden[i] = target_hidden[i].cpu().to(device)
                else:
                    target_hidden[i] = target_hidden[i].to(device)

        # Per-tap norm, concatenate, project to hidden size, norm -- once
        target_hidden = [
            norm.forward(t, {}, out_dtype = torch.half)
            for norm, t in zip(self.input_layer.aux_norms, target_hidden)
        ]
        target_hidden = torch.cat(target_hidden, dim = -1)
        target_hidden = self.input_layer.proj.forward(target_hidden, {}, out_dtype = torch.half)
        target_hidden = self.input_layer.norm.forward(target_hidden, {}, out_dtype = torch.half)

        bsz, target_seqlen, dim = target_hidden.shape
        params["target_hidden_cc"] = target_hidden

        # Update KV layers
        for layer, attn_norm in zip(self.attn_modules, self.attn_norms):
            block_table = get_for_device(params, "block_table", layer.device)
            cache_seqlens = get_for_device(params, "cache_seqlens", layer.device)
            target_hidden = get_for_device(params, "target_hidden_cc", layer.device)

            # Each layer's input_layernorm applies to the shared context features before its
            # K/V projection (vLLM DFlashLaguna convention; z-lab drafters project them raw)
            target_hidden = attn_norm.forward(target_hidden, params, out_dtype = torch.half)

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
                False,
            )

            # Write k, v rows to the paged cache; quantized caches quantize them in place rather
            # than dequantizing/requantizing full layers
            cache.update_layer_direct(layer.layer_idx, cache_seqlens, block_table, k, v, target_seqlen, 0)


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
        # Laguna DFlash drafts causally (dflash_config.causal); the original bidirectional
        # block-attention mode is expressed by clearing the causal flag like dflash.py does
        if not self.config.dflash_causal:
            params["causal"] = False
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError()


    @staticmethod
    @override
    def get_additional_compiled_tensors(config: DFlashLagunaConfig) -> dict:
        # "hidden_norm" and "aux_hidden_norms.*" are stored in DFlashInputLayer but don't match
        # the "fc" prefix
        tensors = {}
        tensors.update(config.stc.list_tensors(prefix = "hidden_norm"))
        tensors.update(config.stc.list_tensors(prefix = "aux_hidden_norms"))
        return tensors
