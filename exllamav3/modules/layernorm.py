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
        weight = self.config.stc.get_tensor(f"{self.key}.weight", self.device)
        bias = self.config.stc.get_tensor(f"{self.key}.bias", self.device, optional = True)
        if weight.dtype == torch.float: weight = weight.to(torch.half)
        if bias is not None and bias.dtype == torch.float: bias = bias.to(torch.half)
        self._numel = weight.numel() + (bias.numel() if bias is not None else 0)
        self.weight = weight
        self.weight_f = weight.to(torch.float)
        self.bias = bias
        self.bias_f = bias.to(torch.float) if self.bias is not None else None

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

    def forward_torch(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        w, b = (self.weight_f, self.bias_f) if x.dtype == torch.float else (self.weight, self.bias)
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
        w, b = (self.weight_f, self.bias_f) if x.dtype == torch.float else (self.weight, self.bias)
        x = F.layer_norm(x, x.shape[-1:], w, b, eps = self.layernorm_eps)
        return x.to(out_dtype or self.out_dtype)