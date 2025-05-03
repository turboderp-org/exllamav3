from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..models import Config
from . import Module
from ..ext import exllamav3_ext as ext

class RMSNorm(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        rms_norm_eps: float,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
        constant_bias: float = 0.0
    ):
        super().__init__(config, key, None)
        assert qmap is None, "No quant scheme for RMSNorm"
        self.module_name = "RMSNorm"

        self.weight = None
        self.rms_norm_eps = rms_norm_eps
        self.out_dtype = out_dtype
        self._numel = None
        self.constant_bias = constant_bias

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        weight = self.config.stc.get_tensor(f"{self.key}.weight", self.device, float2half = True)
        self._numel = weight.numel()
        self.weight = nn.Parameter(weight, requires_grad = False)

    @override
    def unload(self):
        self.device = None
        self.weight = None

    @override
    def get_tensors(self):
        return {
            f"{self.key}.weight": self.weight.data
        }

    def forward_torch(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim = -1, keepdim = True) + self.rms_norm_eps
        x = x * torch.rsqrt(var)
        x = x.to(dtype)
        x = x * self.weight if self.constant_bias == 0.0 else x * (self.weight + self.constant_bias)
        x = x.to(out_dtype or self.out_dtype)
        return x

    @override
    def weights_numel(self):
        return self._numel

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        # TODO: Evalute whether a specialized kernel would be preferable for Q/K norms

        x_shape = x.shape
        x = x.view(-1, x.size(-1))
        y = torch.empty_like(x, dtype = out_dtype or self.out_dtype)
        ext.rms_norm(x, self.weight, y, self.rms_norm_eps, self.constant_bias)
        return y.view(x_shape)