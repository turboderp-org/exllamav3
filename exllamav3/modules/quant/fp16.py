from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ...ext import exllamav3_ext as ext
from ...util.tensor import to2

class LinearFP16:

    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ):
        if weight.dtype == torch.float: weight = weight.to(torch.half)
        if bias is not None and bias.dtype == torch.float: bias = bias.to(torch.half)
        self.weight = weight.T.contiguous()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.swap_device = None

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
        bsz, seqlen, dim = x.shape
        x = x.view(-1, x.shape[-1])
        y = torch.zeros((x.shape[0], self.out_features), dtype = torch.half, device = x.device)
        ext.hgemm(x, self.weight, y)
        if self.bias is not None:
            y += self.bias
        y = y.view(bsz, seqlen, self.out_features)
        return to2(y, out_dtype)

    def get_weight_tensor(self) -> torch.Tensor:
        return self.weight

    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.bias

    def set_weight(self, w: torch.Tensor):
        self.weight = w.half()

    # Swap tensors to CPU (to free some space while quantizing)
    def swap_cpu(self):
        assert self.swap_device is None
        self.swap_device = self.weight.device
        self.weight = self.weight.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

    def unswap_cpu(self):
        assert self.swap_device is not None
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
