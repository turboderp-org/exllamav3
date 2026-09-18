from __future__ import annotations
from typing_extensions import override
import torch

from .dflash import DFlashConfig, DFlashModel
from ..model.config import no_default
from ..modules.transformer import TransformerBlock
from ..modules.arch_specific.dflash2 import GroupedDynamicCausalConv, CandidateSelector
from ..util.tensor import to2


def _tap_permutation_for(target_layer_ids: list[int]) -> list[int] | None:
    """
    Permutation mapping ascending-order exported taps to the listed
    target_layer_ids order: result[i] = taps[permutation[i]]. None when the
    listing is already ascending (pass-through). Pure function so the mapping
    is unit-testable without constructing a model.

    export_states are appended in TransformerBlock walk order (ascending
    layer_idx). _tap_permutation_for maps that list onto
    config.target_layer_ids.
    """
    ascending = sorted(range(len(target_layer_ids)), key = lambda i: target_layer_ids[i])
    if ascending == list(range(len(target_layer_ids))):
        return None
    return [ascending.index(i) for i in range(len(target_layer_ids))]


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
        assert 0 < self.conv_kernel_size <= self.block_size, \
            "DFlash2 conv_kernel_size must be positive and no larger than block_size"
        assert self.conv_group_size > 0 and self.hidden_size % self.conv_group_size == 0, \
            "DFlash2 hidden_size must be divisible by a positive conv_group_size"
        assert self.selector_rank > 0, \
            "DFlash2 selector_rank must be positive"
        assert 0 < self.selector_top_k <= self.vocab_size, \
            "DFlash2 selector_top_k must be between 1 and vocab_size"

        # Checkpoint logit scalings (mirror the reference dflash implementation;
        # output_multiplier/softcapping apply to the draft logits in propose()).
        # input_embedding_scale is read for schema compatibility but not applied:
        # the reference scales its mask embeddings inside its own forward, and
        # porting that means touching DFlashInputLayer.forward shared with v1 —
        # deferred until a real checkpoint needing it is available to validate.
        self.input_embedding_scale = float(self.read_cfg(
            [float, int], ["dflash_config->input_embedding_scale", "input_embedding_scale"], 1.0))
        self.output_multiplier = float(self.read_cfg(
            [float, int], ["dflash_config->output_multiplier", "output_multiplier"], 1.0))
        self.final_logit_softcapping = float(self.read_cfg(
            [float, int], ["dflash_config->final_logit_softcapping", "final_logit_softcapping"], 0.0))


class DFlash2Block(TransformerBlock):
    """
    TransformerBlock with DFlash 2 grouped dynamic convolutions wrapped around the
    attention and MLP sublayers via the parent's _pre/_post hooks. All residual,
    norm, export-state, and prefill semantics live in TransformerBlock.forward.
    """

    def __init__(
        self,
        *args,
        attention_conv: GroupedDynamicCausalConv | None = None,
        mlp_conv: GroupedDynamicCausalConv | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Self-build the convs from the checkpoint geometry when the caller
        # only passes the shared TransformerBlock kwargs (i.e. construction
        # via DFlashModel's block_cls hook). Explicit convs still win.
        if attention_conv is None or mlp_conv is None:
            assert kwargs.get("config") is not None and kwargs.get("key") is not None, \
                "DFlash2Block needs config= and key= to build its convs"
            config, key = kwargs["config"], kwargs["key"]
            conv_kwargs = dict(
                hidden_size = config.hidden_size,
                kernel_size = config.conv_kernel_size,
                group_size = config.conv_group_size,
            )
            if attention_conv is None:
                attention_conv = GroupedDynamicCausalConv(
                    config = config, key = f"{key}.attention_conv", **conv_kwargs)
            if mlp_conv is None:
                mlp_conv = GroupedDynamicCausalConv(
                    config = config, key = f"{key}.mlp_conv", **conv_kwargs)
        self.attention_conv = attention_conv
        self.mlp_conv = mlp_conv
        self.register_submodule(self.attention_conv)
        self.register_submodule(self.mlp_conv)
        assert not self.attn_hc, "DFlash2Block does not support hyperconnections"
        assert not self.mlp_hc, "DFlash2Block does not support hyperconnections"


    @override
    def _pre_attn(self, y, params):
        if self.attention_conv is None:
            return y, None
        return self.attention_conv.prepare(y, params)


    @override
    def _post_attn(self, y, ctx, params):
        if ctx is None:
            return y
        return self.attention_conv.finish(y, ctx, params)


    @override
    def _pre_mlp(self, y, params):
        if self.mlp_conv is None:
            return y, None
        return self.mlp_conv.prepare(y, params)


    @override
    def _post_mlp(self, y, ctx, params):
        if ctx is None:
            return y
        return self.mlp_conv.finish(y, ctx, params)


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

    # Built by DFlashModel.__init__ via its block_cls hook: blocks arrive as
    # conv-wrapped DFlash2Blocks, so no teardown/rebuild here.
    block_cls = DFlash2Block

    def __init__(
        self,
        config: DFlash2Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)

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

        # Tap permutation for update_kv_from_target, computed once (see
        # _tap_permutation_for). None when the listing is already ascending
        # (pass-through, no per-round work).
        self._tap_permutation = _tap_permutation_for(config.target_layer_ids)


    @override
    def attach_to(self, target):
        if target.loaded_tp:
            raise NotImplementedError(
                "DFlash2 drafting does not support tensor-parallel targets because "
                "the selector needs full top-k logits"
            )
        super().attach_to(target)


    @override
    def update_kv_from_target(
        self,
        target_hidden: list,
        cache,
        params: dict,
        lengths: list[int] | None = None,
    ):
        """
        Reorder the exported taps into config.target_layer_ids order before the v1
        fc projection, using the permutation computed at construction.
        """
        ids = self.config.target_layer_ids
        assert len(target_hidden) == len(ids), \
            f"DFlash2 exported {len(target_hidden)} taps for {len(ids)} target_layer_ids"
        if self._tap_permutation is not None:
            target_hidden = [target_hidden[k] for k in self._tap_permutation]
        return super().update_kv_from_target(target_hidden, cache, params, lengths)


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
        q (batch, block_size - 1, k) or None for greedy). When params carries
        "export_draft_conf", also sets params["draft_conf"] to
        [anchor_placeholder, per-position scores...] (the generator crops the anchor).
        """
        assert not self.loaded_tp, "DFlash2 drafting is not supported with tensor parallelism"
        target = self.attached_model()
        if target.loaded_tp:
            raise NotImplementedError(
                "DFlash2 drafting does not support tensor-parallel targets because "
                "the selector needs full top-k logits"
            )
        hidden = out_state[:, 1:, :].contiguous()  # EXL3 lm_head requires contiguous input
        ll = target.logit_layer_idx
        lm = target.modules[ll]
        logits = lm.prepare_for_device(hidden, params)
        logits = lm.forward(logits, params)
        logits = logits[..., : target.config.vocab_size]
        if self.config.output_multiplier != 1.0:
            logits = logits * self.config.output_multiplier
        if self.config.final_logit_softcapping > 0.0:
            softcap = self.config.final_logit_softcapping
            logits = torch.tanh(logits / softcap) * softcap
        export_conf = params.get("export_draft_conf", False)
        path, candidates, q, conf = self.candidate_selector.select(
            hidden, logits, anchor_ids, temperature, return_confidence = export_conf,
        )
        if export_conf and conf is not None:
            # The generator's draft_conf crop reads [:batch, 1:], so lead with a zero
            # anchor placeholder (position 0 is the verified anchor, not a draft).
            params["draft_conf"] = torch.cat(
                (torch.zeros_like(conf[:, :1]), conf), dim = 1
            )
        return path, candidates, q
