from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..models import Config
from . import Module
from ..ext import exllamav3_ext as ext

class LayerNorm(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        layernorm_eps: float,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        assert qmap is None, "No quant scheme for LayerNorm"
        self.module_name = "LayerNorm"

        self.weight = None
        self.weight_f = None
        self.bias = None
        self.bias_f = None
        self.layernorm_eps = layernorm_eps
        self.out_dtype = out_dtype
        self._numel = None

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        weight = self.config.stc.get_tensor(f"{self.key}.weight", self.device, float2half = True)
        bias = self.config.stc.get_tensor(f"{self.key}.bias", self.device, optional = True, float2half = True)
        self._numel = weight.numel() + (bias.numel() if bias is not None else 0)
        self.weight = weight
        self.weight_f = None
        self.bias = bias
        self.bias_f = None

    @override
    def unload(self):
        self.device = None
        self.weight = None
        self.weight_f = None
        self.bias = None
        self.bias_f = None

    @override
    def get_tensors(self):
        t = {}
        t[f"{self.key}.weight"] = self.weight.contiguous()
        if self.bias is not None:
            t[f"{self.key}.bias"] = self.bias.contiguous()
        return t

    def _weight_f(self):
        if self.weight_f is None:
            self.weight_f = self.weight.to(torch.float)
        return self.weight_f

    def _bias_f(self):
        if self.bias is None:
            return None
        if self.bias_f is None:
            self.bias_f = self.bias.to(torch.float)
        return self.bias_f

    def forward_torch(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        w, b = (self._weight_f(), self._bias_f()) if x.dtype == torch.float else (self.weight, self.bias)
        x = F.layer_norm(x, x.shape[-1:], w, b, eps = self.layernorm_eps)
        return x.to(out_dtype or self.out_dtype)

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
        w, b = (self._weight_f(), self._bias_f()) if x.dtype == torch.float else (self.weight, self.bias)
        x = F.layer_norm(x, x.shape[-1:], w, b, eps = self.layernorm_eps)
        return x.to(out_dtype or self.out_dtype)