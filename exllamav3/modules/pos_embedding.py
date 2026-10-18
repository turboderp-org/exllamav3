from __future__ import annotations
from typing_extensions import override
import torch
from torch import nn
import torch.nn.functional as F
from ..model.config import Config
from . import Module
from ..util.tensor import to2
from ..model.model_tp_alloc import TPAllocation

class PosEmbedding(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        vocab_size: int | None = None,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
        base_grid: int | None = None,
        antialias: bool = False,
    ):
        """
        Learned absolute position table added to the sequence. With base_grid, the table is a square
        (base_grid x base_grid) grid of 2D positions that is bilinearly resampled to the (rows, cols) grid
        in params["grid_hw"] when the input grid differs (NaFlex-style vision towers); antialias selects the
        reference's filtered downsampling.
        """
        super().__init__(config, key, None)
        assert qmap is None, "No quant scheme for PosEmbedding"
        self.module_name = "PosEmbedding"

        self.hidden_size = hidden_size
        self.out_dtype = out_dtype
        self.vocab_size = vocab_size
        self.base_grid = base_grid
        self.antialias = antialias
        self.embedding = None
        self._numel = vocab_size * hidden_size if vocab_size is not None else None

    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        weight = self.config.stc.get_tensor(self.key + ".weight", self.device, float2half = True)
        self.vocab_size = weight.shape[0]
        self._numel = weight.numel()
        # Built around the loaded weight: constructing on the meta device and swapping the weight in
        # runs the default initializer there, and the first meta-device op in a process imports
        # a large part of torch's Python decomposition machinery
        self.embedding = nn.Embedding(*weight.shape, _weight = weight, _freeze = True)

    @override
    def unload(self):
        self.device = None
        self.embedding = None

    @override
    def get_tensors(self):
       return {
            f"{self.key}.weight": self.embedding.weight.data.contiguous()
        }

    @override
    def weights_numel(self):
        # TODO: Figure out what to do while vocab_size is None (for quantizing etc.)
        return self._numel

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if self.base_grid is not None:
            x += self.sample_grid(*params["grid_hw"]).to(x.dtype)
        else:
            pos_start = 0
            pos_end = x.shape[1]
            emb_slice = self.embedding.weight.data[pos_start:pos_end]
            x += emb_slice

        # TODO: Support position offset and position IDs for chunking (and GPT2 etc.)

        return to2(x, out_dtype, self.out_dtype)

    def sample_grid(self, rows: int, cols: int) -> torch.Tensor:
        """(rows * cols, hidden) positions for a (rows, cols) grid, resampled from the square table in fp32"""
        table = self.embedding.weight.data
        side = self.base_grid
        if rows == side and cols == side:
            return table
        grid = table.float().view(1, side, side, self.hidden_size).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size = (rows, cols), mode = "bilinear", align_corners = False, antialias = self.antialias)
        return grid.permute(0, 2, 3, 1).reshape(rows * cols, self.hidden_size)


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        return []
