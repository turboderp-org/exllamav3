from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..model.config import Config
from . import Module
from ..ext import exllamav3_ext as ext
from ..model.model_tp_alloc import TPAllocation

class LayerNorm(Module):

    def __init__(
        self,
        config: Config | None,
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
    def optimizer_targets(self):
        return []

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
        d = w.dim()
        if d == 1:
            x = F.layer_norm(x, x.shape[-1:], w, b, eps = self.layernorm_eps)
        else:
            x = F.layer_norm(x, x.shape[-1:], None, None, eps = self.layernorm_eps)
            x *= w
            if b is not None:
                x += b
        return x.to(out_dtype or self.out_dtype)

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        stc = self.config.stc
        storage = sum(stc.get_tensor_sizes(self.key))
        overhead = storage // 2 * (self.out_dtype or torch.half).itemsize
        tpa = TPAllocation(
            key = self.key,
            storage_per_device = storage,
            overhead_per_device = overhead,
        )
        return [tpa]

    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."
        return {
            "cls": LayerNorm,
            "kwargs": {
                "key": self.key,
                "layernorm_eps": self.layernorm_eps,
                "out_dtype": self.out_dtype,
            },
            "weight": producer.send(self.weight),
            "bias": producer.send(self.bias),
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        device = local_context["device"]
        module = LayerNorm(
            config = None,
            **exported["kwargs"],
        )
        module.device = device
        w = consumer.recv(exported["weight"], cuda = True)
        module.weight = nn.Parameter(w)
        return module

    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        consumer = local_context["consumer"]
        device = local_context["device"]
        first, last = split
        module = LayerNorm(
            config = None,
            **exported["kwargs"],
        )
        module.device = device

        w = consumer.recv(exported["weight"], cuda = True)
        if w.dim() == 2 and w.shape[0] > 1:
            w = w[first : last, :]
        module.weight = nn.Parameter(w.contiguous())

        b = consumer.recv(exported["bias"], cuda = True)
        if b is not None:
            if b.dim() == 2 and b.shape[0] > 1:
                b = b[first : last, :]
            module.bias = nn.Parameter(b.contiguous())

        return module