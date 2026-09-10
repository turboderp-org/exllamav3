from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..model.config import Config
from . import Module
from ..ext import exllamav3_ext as ext
from ..model.model_tp_alloc import TPAllocation

class RMSNorm(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        rms_norm_eps: float,
        tensor_weight_suffix: bool = True,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
        constant_bias: float = 0.0,
        constant_scale: float = 1.0,
        span_heads: bool = False,
        unweighted: bool = False,
        groups: int = 1
    ):
        super().__init__(config, key, None)
        assert qmap is None, "No quant scheme for RMSNorm"
        self.module_name = "RMSNorm"

        self.weight = None
        self.rms_norm_eps = rms_norm_eps
        self.out_dtype = out_dtype
        self._numel = None
        self.constant_bias = constant_bias
        self.constant_scale = constant_scale
        self.span_heads = span_heads
        self.unweighted = unweighted
        # Grouped norm (PLE/hyper-connection stream stacks): the weight holds `groups` rows of
        # (numel / groups) channels, selected per input row by (row % groups); the rms itself is
        # per row as always. Input (..., groups, dim) flattens to rows cycling the groups
        self.groups = groups
        assert groups == 1 or not span_heads

        self.tensor_key = f"{self.key}.weight" if tensor_weight_suffix else f"{self.key}"

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        if not self.unweighted:
            weight = self.config.stc.get_tensor(self.tensor_key, self.device, float2half = True, allow_bf16 = True)
            self._numel = weight.numel()
            self.weight = nn.Parameter(weight, requires_grad = False)
        else:
            self._numel = 0

    @override
    def unload(self):
        self.device = None
        self.weight = None

    @override
    def get_tensors(self):
        return {} if self.unweighted else {
            self.tensor_key: self.weight.data
        }

    def _rms_weight(self) -> torch.Tensor | None:
        # Effective weight table with constant_bias/scale folded in, float32.
        # None when unweighted (matches the kernel's null-weight path).
        if self.weight is None or self.unweighted:
            return None
        w = self.weight.float()
        if self.constant_bias != 0.0:
            w = w + self.constant_bias
        if self.constant_scale != 1.0:
            w = w * self.constant_scale
        return w

    def forward_torch(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
        residual: torch.Tensor | None = None,
        residual_in: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.groups == 1 or not self.span_heads
        assert residual is None or residual_in is None
        dtype_out = out_dtype or self.out_dtype or x.dtype
        eps = self.rms_norm_eps

        # Fused pre-norm residual: residual_in += x (in place), y = norm(residual_in).
        # Matches RES_IN in norm.cu: the sum is rounded back to the residual dtype
        # before the variance pass, so norm the rounded-back values.
        if residual_in is not None:
            assert not self.span_heads  # rms_norm_res_in hardcodes span_heads=False
            assert self.groups == 1  # rms_norm_res_in takes no w_groups
            r_sum = residual_in.float() + x.float()
            residual_in.copy_(r_sum.to(residual_in.dtype))
            h = residual_in.float()
            w_eff = self._rms_weight()
            if w_eff is None:
                var = h.pow(2).mean(dim = -1, keepdim = True) + eps
                h = h * torch.rsqrt(var) * self.constant_scale
            else:
                h = F.rms_norm(h, (h.shape[-1],), w_eff, eps)
            return h.to(dtype_out)

        if self.span_heads:
            assert self.groups == 1
            h = x.flatten(-2).float()
            res_flat = residual.flatten(-2) if residual is not None else None
        else:
            h = x.float()
            if self.groups > 1:
                assert x.shape[-2] == self.groups
            res_flat = residual

        w_eff = self._rms_weight()
        if w_eff is None:
            var = h.pow(2).mean(dim = -1, keepdim = True) + eps
            h = h * torch.rsqrt(var) * self.constant_scale
        elif self.groups > 1:
            var = h.pow(2).mean(dim = -1, keepdim = True) + eps
            s = h * torch.rsqrt(var)
            h_shape = s.shape
            d = h_shape[-1]
            s2 = s.reshape(-1, d)
            wtab = w_eff.reshape(self.groups, d)
            h = (s2 * wtab[torch.arange(s2.shape[0], device = s.device) % self.groups]).view(h_shape)
        else:
            h = F.rms_norm(h, (h.shape[-1],), w_eff, eps)
        n = h.to(dtype_out)
        if res_flat is None:
            return n.view_as(x) if self.span_heads else n
        out_f = res_flat.float() + n.float()
        out_f = out_f.to(residual.dtype)
        if self.span_heads:
            residual.copy_(out_f.view_as(residual))
        else:
            residual.copy_(out_f)
        return residual

    def forward_cpu(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
        residual: torch.Tensor | None = None,
        residual_in: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = self._rms_norm_cpu_ext(x, out_dtype, residual, residual_in)
        if y is None:
            y = self.forward_torch(x, params, out_dtype = out_dtype, residual = residual,
                                   residual_in = residual_in)
        return y

    def _rms_norm_cpu_ext(
        self,
        x: torch.Tensor,
        out_dtype: torch.dtype | None,
        residual: torch.Tensor | None,
        residual_in: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Native fused CPU kernel; None when unsupported (caller falls back)."""
        ok_dt = (torch.float, torch.half, torch.bfloat16)
        if not x.is_contiguous() or x.dtype not in ok_dt:
            return None
        if self.weight is not None and (
            not self.weight.is_cpu or not self.weight.is_contiguous() or
            self.weight.dtype not in ok_dt
        ):
            return None
        dtype_out = out_dtype or self.out_dtype or x.dtype
        if dtype_out not in ok_dt:
            return None
        eps = self.rms_norm_eps
        bias = self.constant_bias
        scale = self.constant_scale
        if residual_in is not None:
            if residual is not None or self.span_heads or self.groups != 1:
                return None
            r = residual_in
            if not r.is_contiguous() or r.shape != x.shape or r.dtype not in ok_dt:
                return None
            y = torch.empty_like(x, dtype = dtype_out)
            ext.rms_norm_res_in_cpu(x, self.weight, y, r, eps, bias, scale)
            return y
        if self.span_heads:
            if self.groups != 1:
                return None
            xf = x.flatten(-2)
            if residual is None:
                y = torch.empty_like(x, dtype = dtype_out)
                ext.rms_norm_cpu(xf, self.weight, y.flatten(-2), eps, bias, scale, False, 1)
                return y
            if (not residual.is_contiguous() or residual.shape != x.shape or
                    residual.dtype not in ok_dt):
                return None
            ext.rms_norm_cpu(xf, self.weight, residual.flatten(-2), eps, bias, scale, True, 1)
            return residual
        if self.groups > 1 and x.shape[-2] != self.groups:
            return None
        if residual is None:
            y = torch.empty_like(x, dtype = dtype_out)
            ext.rms_norm_cpu(x, self.weight, y, eps, bias, scale, False, self.groups)
            return y
        if (not residual.is_contiguous() or residual.shape != x.shape or
                residual.dtype not in ok_dt):
            return None
        ext.rms_norm_cpu(x, self.weight, residual, eps, bias, scale, True, self.groups)
        return residual

    @override
    def weights_numel(self):
        return self._numel

    def can_fuse_residual(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        """
        True if forward() can be called with residual_in = x for new sublayer output y
        (fused x += y; norm(x))
        """
        if x.is_cpu and y.is_cpu:
            return x.shape == y.shape
        return (
            not self.span_heads and self.groups == 1 and
            x.dtype in (torch.half, torch.float) and
            y.dtype in (torch.half, torch.float) and
            x.is_contiguous() and y.is_contiguous() and
            x.shape[-1] % 4 == 0
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        params,
        out_dtype: torch.dtype | None = None,
        residual: torch.Tensor | None = None,
        residual_in: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = out_dtype or self.out_dtype

        # CPU tensors (vision towers): native fused kernel, torch fallback off-path
        if x.is_cpu:
            y = self.forward_cpu(x, params, out_dtype = out_dtype, residual = residual, residual_in = residual_in)

        # Fused pre-norm residual: residual_in += x (in place), y = norm(residual_in)
        elif residual_in is not None:
            x_2d = x.view(-1, x.shape[-1])
            r_2d = residual_in.view(-1, residual_in.shape[-1])
            y_2d = torch.empty_like(x_2d, dtype = dtype)
            ext.rms_norm_res_in(
                x_2d,
                self.weight,
                y_2d,
                r_2d,
                self.rms_norm_eps,
                self.constant_bias,
                self.constant_scale,
            )
            y = y_2d.view(x.shape)

        # ext.rms_norm expects row-major 2D inputs for the standard
        # per-channel RMSNorm path. Flattening keeps results consistent across
        # chunk/prefill shapes (e.g. [B, T, C] vs [B*T, C]).
        # TODO: Weight tensor is always contiguous, so this could be handled more efficiently in the extension
        elif not self.span_heads and x.dim() > 2:
            # Grouped norm: the flattened rows cycle the groups (input (..., groups, dim)), the
            # kernel selects the weight row by (row % groups)
            if self.groups > 1:
                assert x.shape[-2] == self.groups
            x_2d = x.reshape(-1, x.shape[-1]).contiguous()
            y_2d = torch.empty_like(x_2d, dtype = dtype) if residual is None else residual.view_as(x_2d)
            ext.rms_norm(
                x_2d,
                self.weight,
                y_2d,
                self.rms_norm_eps,
                self.constant_bias,
                self.constant_scale,
                self.span_heads,
                residual is not None,
                self.groups,
            )
            y = y_2d.view_as(x)
        else:
            assert self.groups == 1
            y = torch.empty_like(x, dtype = dtype) if residual is None else residual.view_as(x)
            ext.rms_norm(
                x,
                self.weight,
                y,
                self.rms_norm_eps,
                self.constant_bias,
                self.constant_scale,
                self.span_heads,
                residual is not None,
            )

        if self.key in params.get("export_state_norm_keys", ()):
            states = params.get("export_states")
            if states is None:
                states = params["export_states"] = []
            states.append(y.half())

        return y

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        if self.unweighted:
            return []
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
            "cls": RMSNorm,
            "kwargs": {
                "key": self.key,
                "rms_norm_eps": self.rms_norm_eps,
                "out_dtype": self.out_dtype,
                "constant_bias": self.constant_bias,
                "span_heads": self.span_heads,
                "constant_scale": self.constant_scale,
                "unweighted": self.unweighted,
                "groups": self.groups,
            },
            "weight": producer.send(self.weight),
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        device = local_context["device"]
        module = RMSNorm(
            config = None,
            **exported["kwargs"],
        )
        module.device = device
        w = consumer.recv(exported["weight"], cuda = True)
        module.weight = nn.Parameter(w) if w is not None else None
        # span_heads is preserved via kwargs
        torch.cuda.synchronize()
        return module

    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        consumer = local_context["consumer"]
        device = local_context["device"]
        first, last = split
        module = RMSNorm(
            config = None,
            **exported["kwargs"],
        )
        module.device = device

        w = consumer.recv(exported["weight"], cuda = True)
        if w is not None:
            if w.dim() == 2:
                w = w[first : last, :]
            elif w.dim() == 1 and module.span_heads:
                # 1D weight tensor (e.g., span_heads=True norms)
                # split contains element indices
                w = w[first : last]
            module.weight = nn.Parameter(w.to(module.device).contiguous())
        # span_heads is preserved via kwargs

        return module
