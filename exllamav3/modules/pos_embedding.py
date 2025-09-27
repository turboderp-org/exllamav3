from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..model.config import Config
from . import Module
from ..ext import exllamav3_ext as ext
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
    ):
        super().__init__(config, key, None)
        assert qmap is None, "No quant scheme for PosEmbedding"
        self.module_name = "PosEmbedding"

        self.hidden_size = hidden_size
        self.out_dtype = out_dtype
        self.vocab_size = vocab_size
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
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            device = "meta"
        )
        self.embedding.weight = nn.Parameter(weight)

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

        pos_start = 0
        pos_end = x.shape[1]
        emb_slice = self.embedding.weight.data[pos_start:pos_end]
        x += emb_slice

        # TODO: Support position offset and position IDs for chunking (and GPT2 etc.)

        return to2(x, out_dtype, self.out_dtype)

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        return []