from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..models import Config
from ..util.tensor import to2
from . import Module

class Embedding(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        vocab_size: int,
        hidden_size: int,
        out_dtype: torch.dtype | None = torch.float,
        qmap: str | None = None,
        normalize: bool = False
    ):
        super().__init__(config, key, None)
        assert qmap is None, "No quant scheme for Embedding"

        self.key = key
        self.embedding = None
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.out_dtype = out_dtype
        self._numel = vocab_size * hidden_size
        self.normalize = normalize

        self.caps.update({
            "prefer_cpu": True,
        })

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        weight = self.config.stc.get_tensor(self.key + ".weight", self.device, float2half = True)
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
        return self._numel
        
    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        x = self.embedding.forward(x)
        x = to2(x, out_dtype, self.out_dtype)
        if self.normalize:
            x *= x.shape[-1] ** 0.5
        return x