from __future__ import annotations
from typing_extensions import override
import torch

from .dflash import DFlashConfig, DFlashModel
from ..model.config import no_default
from ..modules.arch_specific.dflash2 import (
    DFlash2Block, DFlash2DynConv, DFlash2Selector, _DFlash2Norm,
)

# DFlash2 draft model: a DFlash backbone with grouped dynamic convolutions around
# every attention/MLP sublayer and a top-k candidate selector in place of row-wise argmax.
#
# Conventions:
#   - Block input = [anchor, mask x (block_size-1)]; each remaining row predicts
#     its own position and the selector walks candidates starting from the anchor.
#   - Taps: reference reads HF hidden_states[target_layer_ids[i] + 1] = output
#     of target layer id  =>  tap_shift 0 (exl3 export index = layer output).
#   - Draft cache ctx K/V come from update_kv_from_target (fc+hidden_norm
#     projected tap stream), inherited unchanged from the DFlash1 backbone;
#     per-round writes at the new cache position overwrite the transient
#     noise-block K/V, reproducing the reference's crop semantics.
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

        # HF/DFlash defines the window as a token count including the query, while attention
        # backends take an inclusive maximum distance to the left.
        if self.sliding_window > 0:
            self.sliding_window -= 1

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
        # A top-level setting takes precedence over the older nested override.
        self.draft_causal = self.read_cfg(
            bool, ["is_causal", "dflash_config->causal"], False)

        assert 0 < self.conv_kernel_size <= self.block_size, \
            "DFlash2 conv_kernel_size must be positive and no larger than block_size"
        assert self.conv_group_size > 0 and self.hidden_size % self.conv_group_size == 0, \
            "DFlash2 hidden_size must be divisible by a positive conv_group_size"
        assert self.selector_rank > 0, \
            "DFlash2 selector_rank must be positive"
        assert 0 < self.selector_top_k <= self.vocab_size, \
            "DFlash2 selector_top_k must be between 1 and vocab_size"


class DFlash2Model(DFlashModel):
    """DFlash1 backbone rebuilt with dynconv-wrapped blocks + selector head."""

    config_class = DFlash2Config

    def __init__(
        self,
        config: DFlash2Config,
        **kwargs
    ):
        # Build the DFlash1 backbone (input layer, plain blocks, final norm,
        # caps, attach/update_kv_from_target machinery)
        super().__init__(config, **kwargs)
        self.input_layer.input_embedding_scale = config.input_embedding_scale

        # Swap each TransformerBlock for a conv-wrapped DFlash2Block reusing
        # the same attn/mlp modules (attn_modules already reference them, so
        # update_kv_from_target keeps working unchanged). Norms are replaced
        # with torch-only bf16-capable norms (residual stream is bf16).
        for idx in range(config.num_hidden_layers):
            old = self.modules[self.first_block_idx + idx]
            self.modules[self.first_block_idx + idx] = DFlash2Block(
                config = config,
                key = f"layers.{idx}",
                layer_idx = idx,
                attn = old.attn,
                mlp = old.mlp,
                attn_norm = _DFlash2Norm(
                    config = config,
                    key = f"layers.{idx}.input_layernorm",
                    eps = config.rms_norm_eps,
                ),
                mlp_norm = _DFlash2Norm(
                    config = config,
                    key = f"layers.{idx}.post_attention_layernorm",
                    eps = config.rms_norm_eps,
                ),
                attn_conv = DFlash2DynConv(
                    config = config,
                    key = f"layers.{idx}.attention_conv",
                    hidden_size = config.hidden_size,
                    kernel_size = config.conv_kernel_size,
                    group_size = config.conv_group_size,
                ),
                mlp_conv = DFlash2DynConv(
                    config = config,
                    key = f"layers.{idx}.mlp_conv",
                    hidden_size = config.hidden_size,
                    kernel_size = config.conv_kernel_size,
                    group_size = config.conv_group_size,
                ),
            )

        self.modules[-1] = _DFlash2Norm(
            config = config,
            key = "norm",
            eps = config.rms_norm_eps,
        )

        self.selector = DFlash2Selector(
            config = config,
            key = "candidate_selector",
            vocab_size = config.vocab_size,
            hidden_size = config.hidden_size,
            rank = config.selector_rank,
            top_k = config.selector_top_k,
        )
        self.modules += [self.selector]
        self.caps.update({
            "max_draft_size": config.block_size - 1,
            "required_draft_size": config.block_size - 1,
            "supports_dynamic_draft": False,
        })

    @override
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
        super().attach_to(target)

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        assert input_ids.shape[-1] == 1, \
            "DFlash2 expects one verified anchor token per draft block"
        params["dflash2_anchor_ids"] = input_ids
        if self.config.draft_causal:
            params.pop("non_causal_spans", None)
        else:
            # The synthetic block attends bidirectionally within itself and through the configured
            # left window into cached context. Setting the span here covers forward and prefill.
            params["non_causal_spans"] = [(0, self.config.block_size, True)]
        prepared = super().prepare_inputs(input_ids, params)
        # DFlashModel defaults to bilateral attention; restore the checkpoint's causal override.
        params["causal"] = self.config.draft_causal
        return prepared

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
        if self.attached_model().loaded_tp:
            raise NotImplementedError(
                "DFlash2Model does not yet support tensor-parallel targets"
            )
        ll = self.attached_model().logit_layer_idx
        lm = self.attached_model().modules[ll]
        logits = lm.prepare_for_device(state.half(), params)
        logits = lm.forward(logits, params)
        logits = logits[..., :self.attached_model().config.vocab_size]
        if self.config.output_multiplier != 1.0:
            logits = logits * self.config.output_multiplier
        if self.config.final_logit_softcapping > 0.0:
            softcap = self.config.final_logit_softcapping
            logits = torch.tanh(logits / softcap) * softcap

        dev = self.selector.device
        anchor = params["dflash2_anchor_ids"][:, -1].to(dev)
        path = self.selector.walk(
            state[:, 1:].to(dev), logits[:, 1:].to(dev).float(), anchor
        )
        out = torch.empty(
            (path.shape[0], path.shape[1] + 1),
            dtype = torch.long, device = dev)
        out[:, 0] = anchor
        out[:, 1:] = path
        return out

    @classmethod
    @override
    def get_additional_compiled_tensors(cls, config: DFlash2Config) -> dict:
        # Backbone fc norm (DFlash1) + conv base kernels + selector codebooks
        tensors = dict(config.stc.list_tensors(prefix = cls.key_fc_norm))
        tensors.update(config.stc.list_tensors(prefix = "candidate_selector."))
        for idx in range(config.num_hidden_layers):
            for conv in ("attention_conv", "mlp_conv"):
                tensors.update(config.stc.list_tensors(
                    prefix = f"layers.{idx}.{conv}.base_kernel"))
        return tensors
