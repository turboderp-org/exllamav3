from __future__ import annotations
from typing_extensions import override
import torch
import weakref

from ..cache import Cache
from ..model.config import no_default
from ..model.model import Model
from ..modules import RMSNorm, Attention, GatedMLP
from ..modules.arch_specific.dflash import DFlashInputLayer
from ..modules.arch_specific.dflash2 import DFlash2Block, DFlash2DynConv, DFlash2Selector
from ..modules.attn import prepare_for_attn
from .dflash import DFlashConfig, dflash_update_kv_from_target
from ..util.tensor import get_for_device
from ..util.device_copy import to_device

# DFlash2 draft model: the DFlash encoder (fc + hidden_norm over the concatenated target taps)
# and SWA/GQA layers, with grouped dynamic convolutions around every attention/MLP sublayer and
# a top-k candidate selector in place of row-wise argmax.
#
# Conventions:
#   - Block input = [anchor, mask x (block_size-1)]; each remaining row predicts
#     its own position and the selector walks candidates starting from the anchor.
#   - Taps: reference reads HF hidden_states[target_layer_ids[i] + 1] = output
#     of target layer id  =>  tap_shift 0 (exl3 export index = layer output).
#   - Draft cache ctx K/V come from the shared DFlash K/V update (fc+hidden_norm
#     projected tap stream); per-round writes at the new cache position overwrite
#     the transient noise-block K/V, reproducing the reference's crop semantics.
#   - Proposals are the greedy selector path (T-independent), verified by the
#     stock accept-while-match rule — trivially lossless at every temperature
#     (per-position output marginal equals the target distribution).


class DFlash2Config(DFlashConfig):

    arch_string = "DFlash2DraftModel"

    # Reference extract uses hidden_states[id + 1] == output of layer id
    tap_shift = 0

    def __init__(
        self,
        directory: str,
        model_classes: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            directory,
            model_classes or {"text": DFlash2Model},
            **kwargs
        )

        self.conv_kernel_size = self.read_cfg(
            int, ["dflash_config->conv_kernel_size", "conv_kernel_size"], 2)
        self.conv_group_size = self.read_cfg(
            int, ["dflash_config->conv_group_size", "conv_group_size"], 16)
        self.selector_rank = self.read_cfg(
            int, ["dflash_config->selector_rank", "selector_rank"], no_default)
        self.selector_top_k = self.read_cfg(
            int, ["dflash_config->selector_top_k", "selector_top_k"], no_default)
        self.input_embedding_scale = float(self.read_cfg(
            [float, int], ["dflash_config->input_embedding_scale", "input_embedding_scale"], 1.0))
        self.output_multiplier = float(self.read_cfg(
            [float, int], ["dflash_config->output_multiplier", "output_multiplier"], 1.0))
        self.final_logit_softcapping = float(self.read_cfg(
            [float, int], ["dflash_config->final_logit_softcapping", "final_logit_softcapping"], 0.0))
        assert 0 < self.conv_kernel_size <= self.block_size, \
            "DFlash2 conv_kernel_size must be positive and no larger than block_size"
        assert self.conv_group_size > 0 and self.hidden_size % self.conv_group_size == 0, \
            "DFlash2 hidden_size must be divisible by a positive conv_group_size"
        assert self.selector_rank > 0, \
            "DFlash2 selector_rank must be positive"
        assert 0 < self.selector_top_k <= self.vocab_size, \
            "DFlash2 selector_top_k must be between 1 and vocab_size"


class DFlash2Model(Model):
    config_class = DFlash2Config

    # Encoder tensor keys
    key_fc = "fc"
    key_fc_norm = "hidden_norm"

    def __init__(
        self,
        config: DFlash2Config,
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
            qmap = "target_hidden",
            input_embedding_scale = config.input_embedding_scale,
        )
        self.modules += [self.input_layer]

        self.first_block_idx = len(self.modules)
        self.attn_modules = []

        for idx in range(config.num_hidden_layers):
            is_swa = config.layer_types[idx] == "sliding_attention"

            attn = Attention(
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

            def dynconv(name: str, qmap: str):
                return DFlash2DynConv(
                    config = config,
                    key = f"layers.{idx}.{name}",
                    hidden_size = config.hidden_size,
                    kernel_size = config.conv_kernel_size,
                    group_size = config.conv_group_size,
                    qmap = qmap,
                )

            self.modules += [
                DFlash2Block(
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
                    attn_conv = dynconv("attention_conv", "block.attn"),
                    mlp_conv = dynconv("mlp_conv", "block.mlp"),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        self.selector = DFlash2Selector(
            config = config,
            key = "candidate_selector",
            vocab_size = config.vocab_size,
            hidden_size = config.hidden_size,
            rank = config.selector_rank,
            top_k = config.selector_top_k,
        )

        self.modules += [
            RMSNorm(
                config = config,
                key = f"norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            # Identity in the module walk; part of the list so loading, autosplit and compilation
            # account for its tensors. sample_from_state runs its walk after the target's head
            self.selector,
        ]

        # Logits come from the attached target's head
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

        self.draft_verifier_params.update({
            "export_state_layers": set(config.target_layer_ids),
        })


    def attach_to(self, target):
        if target.loaded_tp:
            raise NotImplementedError(
                "DFlash2 does not support tensor-parallel targets because the selector needs top-k logits"
            )
        if target.config.vocab_size != self.config.vocab_size:
            raise ValueError(
                f"DFlash2 vocabulary size {self.config.vocab_size} does not match "
                f"target vocabulary size {target.config.vocab_size}"
            )
        if target.config.hidden_size != self.config.hidden_size:
            raise ValueError(
                f"DFlash2 hidden size {self.config.hidden_size} does not match "
                f"target hidden size {target.config.hidden_size}"
            )
        if not 0 <= self.config.mask_token_id < target.config.vocab_size:
            raise ValueError("DFlash2 mask_token_id is outside the target vocabulary")
        if target.logit_layer_idx is None:
            raise ValueError("DFlash2 target has no compatible LM head")
        if any(not 0 <= layer_id < target.config.num_hidden_layers
               for layer_id in self.config.target_layer_ids):
            raise ValueError("DFlash2 target_layer_ids contains a layer outside the target model")
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
        """Target lm_head over all block rows, then the selector walk over
        rows 1.. (rows predict their own position; row 0 is the anchor).
        Returns (bsz, block) ids [anchor, path...]; the generator crops the
        anchor. The selector is greedy; sampling remains lossless because the
        target verifier still samples normally and accepts only exact matches."""
        target = self.attached_model()
        lm = target.modules[target.logit_layer_idx]
        logits = lm.prepare_for_device(state.half(), params)
        logits = lm.forward(logits, params)

        dev = self.selector.device
        # The generator stages the block ids in pinned memory: upload without a host sync
        anchor = get_for_device(params, "dflash2_anchor_ids", dev)[:, -1]
        export_conf = params.get("export_draft_conf", False)
        # [anchor, path...] and [0, score...] straight from the selector, in the block layout
        # the generator consumes. The head's padded width, the output multiplier and the
        # softcap (Gemma-class targets) are handled inside the selector's top-k. The state
        # (draft's last block) and the logits (target's head) may sit on other devices than
        # the selector: to_device bounces pairs with a broken peer path through the host
        out, confidence = self.selector.walk_block(
            to_device(state[:, 1:], dev), to_device(logits[:, 1:], dev), anchor,
            return_confidence = export_conf,
            vocab_size = target.config.vocab_size,
            scale = self.config.output_multiplier,
            softcap = self.config.final_logit_softcapping,
        )
        if export_conf:
            params["draft_conf"] = confidence
        return out


    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long


    def default_load_params(self, max_chunk_size):
        return {}


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        assert input_ids.shape[-1] == 1, \
            "DFlash2 expects one verified anchor token per draft block"
        params["dflash2_anchor_ids"] = input_ids
        # The draft block attends to itself bidirectionally; causality on the sliding-window
        # layers is expressed through their window (left sw, right 0) instead
        params["causal"] = False
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError()


    @classmethod
    @override
    def get_additional_compiled_tensors(cls, config: DFlash2Config) -> dict:
        # Encoder norm (stored in DFlashInputLayer under a key that doesn't match the fc module
        # prefix), conv base kernels and selector codebooks: raw tensors outside any Linear
        tensors = dict(config.stc.list_tensors(prefix = cls.key_fc_norm))
        tensors.update(config.stc.list_tensors(prefix = "candidate_selector."))
        for idx in range(config.num_hidden_layers):
            for conv in ("attention_conv", "mlp_conv"):
                tensors.update(config.stc.list_tensors(
                    prefix = f"layers.{idx}.{conv}.base_kernel"))
        return tensors
