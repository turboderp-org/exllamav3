from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ...ext import exllamav3_ext as ext
from ...util.tensor import to2
from ...util import first_not_none

class LinearFP16:

    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        full_in_features: int | None = None,
        full_out_features: int | None = None,
        first_in_feature: int | None = None,
        first_out_feature: int | None = None,
        out_dtype: torch.dtype | None = None
    ):
        self.weight = weight
        if bias is not None and bias.dtype == torch.float: bias = bias.to(torch.half)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.swap_device = None
        self.full_in_features = full_in_features
        self.full_out_features = full_out_features
        self.first_in_feature = first_in_feature
        self.first_out_feature = first_out_feature
        self.out_dtype = out_dtype

        if self.weight.shape[0] == full_in_features and self.weight.shape[0] != in_features:
            self.weight = self.weight[first_in_feature : first_in_feature + in_features, :]
        if self.weight.shape[1] == full_out_features and self.weight.shape[1] != out_features:
            self.weight = self.weight[:, first_out_feature : first_out_feature + out_features]
            if bias is not None:
                self.bias = self.bias[..., first_out_feature : first_out_feature + out_features]

        if in_features != full_in_features or out_features != full_out_features:
            w = torch.empty(self.weight.shape, dtype = self.weight.dtype, device = self.weight.device)
            w.copy_(self.weight)
            self.weight = w

    def get_tensors(self, key: str):
        t = {}
        t[f"{key}.weight"] = self.weight.T.contiguous()
        if self.bias is not None:
            t[f"{key}.bias"] = self.bias
        return t

    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.view(-1, x.shape[-1])
        dtype = first_not_none(out_dtype, self.out_dtype, torch.half)
        y = torch.zeros(
            (x.shape[0], self.out_features),
            dtype = dtype,
            device = x.device
        )
        if dtype == x.dtype:
            torch.matmul(x, self.weight, out = y)
        else:
            ext.hgemm(x, self.weight, y)
        if self.bias is not None:
            y += self.bias
        y = y.view(out_shape)
        return y

    def get_weight_tensor(self) -> torch.Tensor:
        return self.weight

    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.bias

    def set_weight(self, w: torch.Tensor):
        self.weight = w.half()

    # Swap tensors to CPU (to free some space while quantizing)
    def swap_cpu(self):
        if self.swap_device is not None:
            return
        self.swap_device = self.weight.device
        self.weight = self.weight.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

    def unswap_cpu(self):
        if self.swap_device is None:
            return
        self.weight = self.weight.to(self.swap_device)
        if self.bias is not None:
            self.bias = self.bias.to(self.swap_device)
        self.swap_device = None


class LinearFP16_torch:

    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ):
        self.nn_linear = nn.Linear(
            in_features,
            out_features,
            bias is not None,
            device = "meta"
        )
        self.nn_linear.weight = nn.Parameter(weight, requires_grad = False)
        if bias is not None:
            self.nn_linear.bias = nn.Parameter(bias, requires_grad = False)

    def get_tensors(self, key: str):
        t = {}
        t[f"{key}.weight"] = self.nn_linear.weight.data.contiguous()
        if self.nn_linear.bias is not None:
            t[f"{key}.bias"] = self.nn_linear.bias.data.contiguous()
        return t

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        x = self.nn_linear.forward(x)
        return to2(x, out_dtype)

    def get_weight_tensor(self) -> torch.Tensor:
        return self.nn_linear.weight.data.T

    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.nn_linear.bias.data if self.nn_linear.bias is not None else None

    def set_weight(self, w):
        self.nn_linear.weight = nn.Parameter(w.T.half())
