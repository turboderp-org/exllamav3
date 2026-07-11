from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...util.tensor import to2
from ...model.config import Config
from ...modules import Module


class Step3_7Downsampler(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        key_1: str,
        key_2: str,
        hidden_size: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        bias: bool = True,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "Step3_7Downsampler"
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        self.out_dtype = out_dtype
        self._numel = 0

        self.key_1 = key_1
        self.key_2 = key_2
        self.weight_1 = None
        self.bias_1 = None
        self.weight_2 = None
        self.bias_2 = None


    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        self.weight_1 = self.config.stc.get_tensor(f"{self.key_1}.weight", device, float2half = True)
        self.bias_1 = self.config.stc.get_tensor(f"{self.key_1}.bias", device, optional = not self.has_bias, float2half = True)
        self.weight_2 = self.config.stc.get_tensor(f"{self.key_2}.weight", device, float2half = True)
        self.bias_2 = self.config.stc.get_tensor(f"{self.key_2}.bias", device, optional = not self.has_bias, float2half = True)
        self._numel = self.weight_1.numel() + (self.bias_1.numel() if self.bias_1 is not None else 0)
        self._numel += self.weight_2.numel() + (self.bias_2.numel() if self.bias_2 is not None else 0)


    @override
    def unload(self):
        self.device = None
        self.weight_1 = None
        self.bias_1 = None
        self.weight_2 = None
        self.bias_2 = None


    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        bsz, seqlen, dim = x.shape

        hw = int(seqlen ** 0.5)
        y = x.transpose(1, 2).view(bsz, dim, hw, hw).half()

        y = F.conv2d(
            y,
            self.weight_1,
            self.bias_1,
            stride = self.stride,
            padding = self.padding,
        )
        y = F.conv2d(
            y,
            self.weight_2,
            self.bias_2,
            stride = self.stride,
            padding = self.padding,
        )

        y = y.view(bsz, -1, y.shape[-1] ** 2).transpose(1, 2).contiguous()
        return to2(y, out_dtype, self.out_dtype)


    @override
    def get_tensors(self):
        return {
            k: v.contiguous() for k, v in [
                (f"{self.key_1}.weight", self.weight_1),
                (f"{self.key_1}.bias", self.bias_1),
                (f"{self.key_2}.weight", self.weight_2),
                (f"{self.key_2}.bias", self.bias_2),
            ] if k is not None and v is not None
        }


    @override
    def weights_numel(self):
        return self._numel


    @override
    def optimizer_targets(self):
        raise NotImplementedError()


class Step3_7PosEmbedding(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        base_grid: int,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "Step3_7PosEmbedding"
        self.qmap = qmap
        self.key = key

        self.hidden_size = hidden_size
        self.base_grid = base_grid
        self.out_dtype = out_dtype

        self.weight = None


    @override
    def weights_numel(self):
        return self.base_grid ** 2 * self.hidden_size


    def optimizer_targets(self):
        raise NotImplementedError()


    def get_tensors(self):
        return {self.key: self.weight.contiguous()}


    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        self.weight = self.config.stc.get_tensor(self.key, self.device, allow_bf16 = True)


    @override
    def unload(self):
        self.device = None
        self.weight = None


    def sample_abs_posemb(self, grid_h: int, grid_w: int):
        if self.base_grid == grid_h and self.base_grid == grid_w:
            return self.weight.unsqueeze(0)
        return F.interpolate(
            self.weight.reshape(1, self.base_grid, self.base_grid, -1).permute(0, 3, 1, 2).contiguous(),
            size = (grid_h, grid_w),
            mode = "bilinear", align_corners = False
        ).permute(0, 2, 3, 1).reshape(-1, self.hidden_size).unsqueeze(0)


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ):
        pos_emb = self.sample_abs_posemb(*params["grid_hw"])
        x += pos_emb
        return to2(x, out_dtype, self.out_dtype)
