from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..model.config import Config
from . import Module
from ..ext import exllamav3_ext as ext
from ..model.model_tp_alloc import TPAllocation

class Conv(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
        flat: bool = False,
        reshape2d: bool = False,
    ):
        super().__init__(config, key, None)
        assert qmap is None, "No quant scheme for Conv"
        self.module_name = "Conv"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = None
        self.bias = None
        self.out_dtype = out_dtype
        self._numel = None
        self.flat = flat
        self.reshape2d = reshape2d

        self.dims = len(kernel_size)
        assert self.dims in [2, 3], "Convolution must be 2D or 3D"

        self.flat = flat
        self.dim = in_channels
        for d in self.kernel_size: self.dim *= d


    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        weight = self.config.stc.get_tensor(f"{self.key}.weight", self.device, float2half = True)
        bias = self.config.stc.get_tensor(f"{self.key}.bias", self.device, optional = True, float2half = True)
        self._numel = weight.numel() + (bias.numel() if bias is not None else 0)
        self.weight = weight
        self.bias = bias


    @override
    def unload(self):
        self.device = None
        self.weight = None
        self.bias = None


    @override
    def get_tensors(self):
        t = {}
        t[f"{self.key}.weight"] = self.weight.contiguous()
        if self.bias is not None:
            t[f"{self.key}.bias"] = self.bias.contiguous()
        return t


    @override
    def weights_numel(self):
        return self._numel


    @override
    def forward(
        self,
        x: torch.Tensor,
        params,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        if self.dims == 2:
            if self.reshape2d:
                x = x.view(-1, self.kernel_size[0], self.kernel_size[1], x.shape[-1]).permute(0, 3, 1, 2)
            y = F.conv2d(x.to(self.weight.dtype), self.weight, self.bias, self.kernel_size)
            y = y.flatten(2).permute(0, 2, 1).contiguous()

        elif self.dims == 3:
            bsz, seqlen, dim = x.shape

            if self.flat:
                x_flat = x.view(-1, self.dim)
                w_flat = self.weight.view(self.weight.shape[0], -1).T.contiguous()
                out_dtype = out_dtype or self.out_dtype
                assert x_flat.dtype == torch.half
                assert w_flat.dtype == torch.half
                y = torch.empty((x_flat.shape[0], w_flat.shape[1]), dtype = out_dtype, device = x_flat.device)
                ext.hgemm(x_flat, w_flat, y)
                y = y.view(bsz, seqlen, self.out_channels)
                if self.bias is not None:
                    y += self.bias
            else:
                # Extremely inefficient for Qwen3
                x = x.view(-1, self.in_channels, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2])
                y = F.conv3d(x, self.weight, self.bias, self.kernel_size)
                y = y.view(bsz, seqlen, self.out_channels)

        return y


    @override
    def optimizer_targets(self):
        raise NotImplementedError()


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        raise NotImplementedError("TP not implemented for Conv layer.")