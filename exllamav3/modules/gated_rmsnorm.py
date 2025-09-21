from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..model.config import Config
from . import Module
from ..ext import exllamav3_ext as ext
from ..model.model_tp_alloc import TPAllocation

class GatedRMSNorm(Module):

    def __init__(
        self,
        config: Config | None,
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
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        weight = self.config.stc.get_tensor(f"{self.key}.weight", self.device, allow_bf16 = True)
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

    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
        gate: torch.Tensor = None,
    ) -> torch.Tensor:
        from fla.modules.fused_norm_gate import rms_norm_gated
        x = rms_norm_gated(
            x = x,
            g = gate,
            weight = self.weight,
            bias = None,
            activation = "silu",
            eps = self.rms_norm_eps
        )
        x = x.to(out_dtype or self.out_dtype)
        return x

    @override
    def weights_numel(self):
        return self._numel

    # @override
    # def forward_torch(
    #     self,
    #     x: torch.Tensor,
    #     params,
    #     out_dtype: torch.dtype | None = None,
    #     gate: torch.Tensor = None,
    # ) -> torch.Tensor:
    #     return self.forward_torch(x, params, out_dtype, gate)

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
            "cls": GatedRMSNorm,
            "kwargs": {
                "key": self.key,
                "rms_norm_eps": self.rms_norm_eps,
                "out_dtype": self.out_dtype,
                "constant_bias": self.constant_bias,
            },
            "weight": producer.send(self.weight),
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        device = local_context["device"]
        module = GatedRMSNorm(
            config = None,
            **exported["kwargs"],
        )
        module.device = device
        w = consumer.recv(exported["weight"], cuda = True)
        module.weight = nn.Parameter(w)
        torch.cuda.synchronize()
        return module

    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        consumer = local_context["consumer"]
        device = local_context["device"]
        first, last = split
        module = GatedRMSNorm(
            config = None,
            **exported["kwargs"],
        )
        module.device = device

        w = consumer.recv(exported["weight"], cuda = True)
        if w.dim() == 2:
            w = w[first : last, :]
        module.weight = nn.Parameter(w.to(module.device).contiguous())

        return module