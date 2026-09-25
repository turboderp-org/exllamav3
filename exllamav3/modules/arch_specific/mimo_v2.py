from __future__ import annotations
from typing_extensions import override
import torch
from .. import Module
from ...model import Config
from ...util.tensor import get_for_device


class MiMoV2VisionReorder(Module):
    """
    Token re-serialization between MiMo-ViT blocks. The windowed blocks alternate between
    row-major and column-major token order so their sliding window covers both spatial axes:
    the tower permutes the hidden states at spatial-merge-unit granularity when the order
    changes, and the rotary table travels with the tokens. params[index_key] holds the unit
    gather index (built in MiMoV2VisionModel.prepare_inputs from the grid); absent = identity.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        index_key: str,
        spatial_merge_unit: int,
    ):
        super().__init__(config, key, None)
        self.module_name = "MiMoV2VisionReorder"
        self.index_key = index_key
        self.spatial_merge_unit = spatial_merge_unit

    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def weights_numel(self):
        return 0

    @override
    def forward(
        self,
        x: torch.Tensor,
        params,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        index = get_for_device(params, self.index_key, x.device, None)
        if index is None:
            return x
        bsz, seq_len, dim = x.shape
        unit = self.spatial_merge_unit
        x = x.reshape(bsz * seq_len // unit, unit, dim)[index].reshape(bsz, seq_len, dim)
        inv_freq = params.get("inv_freq")
        if inv_freq is not None:
            # New tensor object: the per-device cache keys on identity, so nothing stale is hit
            f = inv_freq.shape[-1]
            index_f = index.to(inv_freq.device)
            params["inv_freq"] = inv_freq.reshape(bsz * seq_len // unit, unit, f)[index_f].reshape(bsz, seq_len, f)
        return x
