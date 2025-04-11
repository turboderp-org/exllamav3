from __future__ import annotations
import torch
from ...models.config import Config
from ...util.tensor import to2
import math
from .exl3_lib.quantize import preapply_had_l, preapply_had_r, had_k, had_n, tensor_core_perm, tensor_core_perm_i
from ...ext import exllamav3_ext as ext

class LinearEXL3:

    def __init__(
        self,
        config: Config,
        in_features: int,
        out_features: int,
        scale: torch.Tensor | None = None,
        su: torch.Tensor | None = None,
        sv: torch.Tensor | None = None,
        suh: torch.Tensor | None = None,
        svh: torch.Tensor | None = None,
        trellis: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ):
        assert scale is None, "scale is no longer used"
        assert su is not None or suh is not None, "either su (packed) or suh (unpacked) is required"
        assert sv is not None or svh is not None, "either sv (packed) or svh (unpacked) is required"
        assert trellis is not None, "trellis is required"
        if su is not None: assert su.dtype == torch.int16, "su is wrong datatype"
        if sv is not None: assert sv.dtype == torch.int16, "sv is wrong datatype"
        if suh is not None: assert suh.dtype == torch.half, "suh is wrong datatype"
        if svh is not None: assert svh.dtype == torch.half, "svh is wrong datatype"
        assert trellis.dtype == torch.int16, "trellis is wrong datatype"
        assert len(trellis.shape) == 3, "trellis must have dim = 3"

        if bias is not None and bias.dtype == torch.float: bias = bias.to(torch.half)

        # self.scale = scale.item()
        self.su = su
        self.sv = sv
        self.suh = suh
        self.svh = svh
        self.trellis = trellis
        self.K = trellis.shape[-1] // 16
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias


    def get_tensors(self, key: str):
        t = {}
        # t[f"{key}.scale"] = torch.tensor([self.scale], dtype = torch.float, device = self.su.device)
        if self.su is not None: t[f"{key}.su"] = self.su.contiguous()
        if self.suh is not None: t[f"{key}.suh"] = self.suh.contiguous()
        if self.sv is not None: t[f"{key}.sv"] = self.sv.contiguous()
        if self.svh is not None: t[f"{key}.svh"] = self.svh.contiguous()
        t[f"{key}.trellis"] = self.trellis.contiguous()
        if self.bias is not None: t[f"{key}.bias"] = self.bias.contiguous()
        return t


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        out_shape = x.shape[:-1] + (self.out_features,)
        input_dtype = x.dtype
        x = x.view(-1, self.in_features)

        torch_mode = params.get("reconstruct", x.shape[0] >= 32)

        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, self.suh, None, 1.0)

        if torch_mode:
            y = torch.empty((x.shape[0], self.out_features), dtype = torch.half, device = x.device)
            w = self.get_inner_weight_tensor()
            ext.hgemm(xh, w, y)
            ext.had_r_128(y, y, None, self.sv, 1.0)
            if self.sv is None:
                # TODO: Fuse out scales into had kernel
                y *= self.svh
        else:
            y = torch.empty((x.shape[0], self.out_features), dtype = torch.half, device = x.device)
            ext.exl3_gemm(xh, self.trellis, y, self.sv, -1)
            # ext.exl3_gemm(xh, self.trellis, y, None, -1)
            if self.sv is None:
                # TODO: Fuse out scales into GEMM kernel
                ext.had_r_128(y, y, None, None, 1.0)
                y *= self.svh

        x = y.view(out_shape)

        if self.bias is not None:
            x = x.to(self.bias.dtype)
            x += self.bias

        return to2(x, out_dtype, input_dtype)


    def unpack_bf(self, bitfield: torch.Tensor):
        # TODO: Maybe custom kernel for this. Only used for full reconstruct though, not during inference
        bitfield = bitfield.view(torch.uint16).to(torch.int)
        masks = (1 << torch.arange(16)).to(bitfield.device)
        expanded = (bitfield.unsqueeze(-1) & masks) > 0
        expanded = expanded.flatten()
        expanded = torch.where(expanded, torch.tensor(-1.0, dtype = torch.float16), torch.tensor(1.0, dtype = torch.float16))
        return expanded.contiguous()


    def get_weight_tensor(self):
        # suh = self.unpack_bf(self.su).unsqueeze(1)
        suh = self.unpack_bf(self.su).unsqueeze(1) if self.su else self.suh.unsqueeze(1)
        svh = self.unpack_bf(self.sv).unsqueeze(0) if self.sv else self.svh.unsqueeze(0)
        w = self.get_inner_weight_tensor()
        w = preapply_had_l(w, had_k)
        w *= suh
        w = preapply_had_r(w, had_n)
        w *= svh
        # w *= self.scale
        return w


    def get_inner_weight_tensor(self):
        w = torch.zeros((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
        ext.reconstruct(w, self.trellis, self.K)
        return w


    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.bias
