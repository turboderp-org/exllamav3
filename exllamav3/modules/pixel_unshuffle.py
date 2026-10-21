from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config
from . import Module
from ..util.tensor import to2


class PixelUnshuffle(Module):
    """
    Merges each (factor x factor) block of a (rows, cols) token grid into one token by channel
    concatenation, in the InternVL / LFM2-VL ordering: the merged channel vector runs over
    (block row offset, block column offset, channel). Leading prefix tokens (cls/registers) are dropped
    first. legacy_order keeps InternVL's v1 output, which leaves the merged grid transposed.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        factor: int,
        num_prefix_tokens: int = 0,
        legacy_order: bool = False,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "PixelUnshuffle"
        self.factor = factor
        self.num_prefix_tokens = num_prefix_tokens
        self.legacy_order = legacy_order
        self.out_dtype = out_dtype


    @override
    def optimizer_targets(self):
        return []


    @override
    def weights_numel(self):
        return 0


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        rows, cols = params["grid_hw"]
        f = self.factor
        n, _, c = x.shape
        x = x[:, self.num_prefix_tokens:, :].reshape(n, rows, cols, c)
        x = x.reshape(n, rows, cols // f, c * f)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(n, cols // f, rows // f, c * f * f)
        if not self.legacy_order:
            x = x.permute(0, 2, 1, 3).contiguous()
        y = x.reshape(n, -1, c * f * f)
        return to2(y, out_dtype, self.out_dtype)
