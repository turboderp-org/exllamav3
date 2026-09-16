from __future__ import annotations
from typing_extensions import override
import torch
import weakref

from .dflash import DFlashConfig, DFlashModel
from ..model.config import no_default
from ..modules import RMSNorm, Attention, GatedMLP
from ..modules.transformer import TransformerBlock
from ..modules.arch_specific.dflash2 import GroupedDynamicCausalConv, CandidateSelector
from ..util.tensor import to2


class DFlash2Config(DFlashConfig):
    """
    DFlash 2 draft model (block diffusion drafter with grouped dynamic convolution and a
    candidate path selector). Checkpoint arch string: DFlash2DraftModel.

    Adds over DFlash v1 (all under dflash_config-> or top level, mirroring the reference
    z-lab/dflash implementation):
      - conv_kernel_size, conv_group_size: per-sublayer grouped dynamic depthwise convolution
      - selector_rank, selector_top_k: candidate path selector (bilinear adjacent-pair scoring)
    """

    arch_string = "DFlash2DraftModel"

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

        # DFlash 2 additions
        self.conv_kernel_size = self.read_cfg(int, ["dflash_config->conv_kernel_size", "conv_kernel_size"], no_default)
        self.conv_group_size = self.read_cfg(int, ["dflash_config->conv_group_size", "conv_group_size"], no_default)
        self.selector_rank = self.read_cfg(int, ["dflash_config->selector_rank", "selector_rank"], no_default)
        self.selector_top_k = self.read_cfg(int, ["dflash_config->selector_top_k", "selector_top_k"], no_default)


class DFlash2Block(TransformerBlock):
    """
    TransformerBlock with DFlash 2 grouped dynamic convolutions wrapped around the
    attention and MLP sublayers (prepare before the sublayer, finish after). The
    residual/norm/export-state semantics are identical to TransformerBlock.
    """

    def __init__(
        self,
        *args,
        attention_conv: GroupedDynamicCausalConv | None = None,
        mlp_conv: GroupedDynamicCausalConv | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attention_conv = attention_conv
        self.mlp_conv = mlp_conv
        self.register_submodule(self.attention_conv)
        self.register_submodule(self.mlp_conv)


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        export_state = params.get("export_state_layers")
        export_state = export_state and self.layer_idx in export_state and params.get("layer_instance", 0) == 0

        y_resid = None  # pending attn output whose residual add is folded into the MLP input norm

        if self.attn:
            assert not self.attn_hc, "DFlash2Block does not support hyperconnections"
            if self.attn_norm:
                y = self.attn_norm.forward(x, params, out_dtype = torch.half)
            else:
                y = x.half()
            attn_dyn = None
            if self.attention_conv is not None:
                y, attn_dyn = self.attention_conv.prepare(y, params)
            y = self.attn.forward(y, params)
            if params.get("prefill") and not export_state:
                return x
            if self.attention_conv is not None:
                y = self.attention_conv.finish(y, attn_dyn, params)
            if self.attn_post_norm:
                self.attn_post_norm.forward(y, params, residual = x)
            elif self.mlp is not None and self.mlp_norm is not None and self.mlp_norm.can_fuse_residual(x, y):
                y_resid = y
            else:
                x += y

        if self.mlp:
            assert not self.mlp_hc, "DFlash2Block does not support hyperconnections"
            params["residual"] = x
            if y_resid is not None:
                y = self.mlp_norm.forward(y_resid, params, out_dtype = torch.half, residual_in = x)
            elif self.mlp_norm:
                y = self.mlp_norm.forward(x, params, out_dtype = torch.half)
            else:
                y = x.half()
            mlp_dyn = None
            if self.mlp_conv is not None:
                y, mlp_dyn = self.mlp_conv.prepare(y, params)
            y = self.mlp.forward(y, params)
            if self.mlp_conv is not None:
                y = self.mlp_conv.finish(y, mlp_dyn, params)
            if self.mlp_post_norm:
                self.mlp_post_norm.forward(y, params, residual = x)
            else:
                x += y

        if export_state:
            s = params.get("export_states")
            if not s:
                s = params["export_states"] = []
            if x.dtype == torch.half:
                s.append(x.clamp_(-65504.0, 65504.0))
            else:
                x_ = x.half()
                x_.clamp_(-65504.0, 65504.0)
                s.append(x_)

        if self.layer_scalar_f is not None:
            x *= self.layer_scalar_f

        return to2(x, out_dtype, self.out_dtype)


class DFlash2Model(DFlashModel):
    """
    DFlash 2 draft. v1 backbone (fc + SWA layers + shared target head) plus:
      - GroupedDynamicCausalConv wrapped around each attention and MLP sublayer
      - CandidateSelector producing the drafted token path (greedy walk or sampled walk
        with q over the per-position candidate lists, for the lossless verify)
    Conv and selector parameters stay in fp16 (small, uncalibrated); the backbone Linears
    quantize normally.
    """

    config_class = DFlash2Config

    def __init__(
        self,
        config: DFlash2Config,
        **kwargs
    ):
        # Build the v1 backbone by running the parent init against a model class that
        # builds DFlash2Blocks. The parent hardcodes TransformerBlock, so construct the
        # module list manually here instead (keeps DFlashModel untouched).
        super().__init__(config, **kwargs)

        # The parent built plain TransformerBlocks; rebuild as DFlash2Blocks with convs.
        # Replace the block modules in-place at the same module-list positions so the
        # loader's per-module layout (first_block_idx .. last_kv_module_idx) is preserved.
        from ..modules.arch_specific.dflash import DFlashInputLayer

        # Rebuild blocks: un-register parent blocks and append DFlash2Blocks in identical order
        # (module list order is the load order)
        parent_blocks = self.modules[self.first_block_idx : self.last_kv_module_idx + 1]
        assert len(parent_blocks) == config.num_hidden_layers

        new_blocks = []
        for idx in range(config.num_hidden_layers):
            parent_block = parent_blocks[idx]
            is_swa = config.layer_types[idx] == "sliding_attention"
            attn_conv = GroupedDynamicCausalConv(
                config = config,
                key = f"layers.{idx}.attention_conv",
                hidden_size = config.hidden_size,
                kernel_size = config.conv_kernel_size,
                group_size = config.conv_group_size,
            )
            mlp_conv = GroupedDynamicCausalConv(
                config = config,
                key = f"layers.{idx}.mlp_conv",
                hidden_size = config.hidden_size,
                kernel_size = config.conv_kernel_size,
                group_size = config.conv_group_size,
            )
            block = DFlash2Block(
                config = config,
                key = f"layers.{idx}",
                layer_idx = idx,
                attn_norm = parent_block.attn_norm,
                attn = parent_block.attn,
                mlp_norm = parent_block.mlp_norm,
                mlp = parent_block.mlp,
                attention_conv = attn_conv,
                mlp_conv = mlp_conv,
            )
            new_blocks.append(block)

        self.modules[self.first_block_idx : self.last_kv_module_idx + 1] = new_blocks
        self.attn_modules = [b.attn for b in new_blocks]

        self.candidate_selector = CandidateSelector(
            config = config,
            key = "candidate_selector",
            vocab_size = config.vocab_size,
            hidden_size = config.hidden_size,
            rank = config.selector_rank,
            top_k = config.selector_top_k,
        )
        self.modules.append(self.candidate_selector)

        self.caps.update({
            "dflash2_draft": True,
        })


    def propose(
        self,
        out_state: torch.Tensor,
        anchor_ids: torch.Tensor,
        params: dict,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Run the candidate selector over the draft block states.

        out_state: (batch, block_size, hidden) full block states; position 0 is the
        anchor token position and positions 1: are the mask positions whose predictions
        form the draft. anchor_ids: (batch,) the last verified token ids. Returns
        (path tokens (batch, block_size - 1), candidates (batch, block_size - 1, k),
        q (batch, block_size - 1, k) or None for greedy).
        """
        assert not self.loaded_tp, "DFlash2 drafting is not supported with tensor parallelism"
        target = self.attached_model()
        hidden = out_state[:, 1:, :].contiguous()  # EXL3 lm_head requires contiguous input
        ll = target.logit_layer_idx
        lm = target.modules[ll]
        logits = lm.prepare_for_device(hidden, params)
        logits = lm.forward(logits, params)
        logits = logits[..., : target.config.vocab_size]
        return self.candidate_selector.select(hidden, logits, anchor_ids, temperature)
