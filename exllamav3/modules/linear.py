from __future__ import annotations
from functools import cached_property
from typing_extensions import override
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from ..model.config import Config
from . import Module
from .quant import LinearFP16, LinearFP16_torch, LinearEXL3
from .quant.exl3_lib import quantize_exl3
from ..ext import exllamav3_ext as ext
from ..conversion.allocation import allocate_linear
from ..util.memory import free_mem
from ..model.model_tp_alloc import TPAllocation


class Linear(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        in_features: int,
        out_features: int,
        qmap: str | None = None,
        alt_key: str | None = None,
        qbits_key: str = "bits",
        qbits_mod_key: str = "",
        fkey : str | None = None,
        frange: tuple[int, int] | None = None,
        fidx: int | None = None,
        caps: dict = None,
        softcap: float = 0.0,
        pad_to: int = 128,
        full_in_features: int | None = None,
        full_out_features: int | None = None,
        first_in_feature: int | None = None,
        first_out_feature: int | None = None,
        out_dtype: torch.dtype | None = None,
        allow_input_padding: bool = False,
        post_scale: float = 1.0,
        transposed_load: bool = True
    ):
        super().__init__(config, key, qmap)

        self.alt_key = alt_key
        self.in_features_unpadded = in_features
        self.in_features = (in_features + pad_to - 1) // pad_to * pad_to
        self.out_features_unpadded = out_features
        self.out_features = (out_features + pad_to - 1) // pad_to * pad_to
        self.full_in_features = full_in_features if full_in_features is not None else self.in_features
        self.full_out_features = full_out_features if full_out_features is not None else self.out_features
        self.first_in_feature = first_in_feature if first_in_feature is not None else 0
        self.first_out_feature = first_out_feature if first_out_feature is not None else 0
        self.inner = None
        self.qbits_key = qbits_key
        self.qbits_mod_key = qbits_mod_key
        self.fkey = fkey
        self.frange = frange
        self.fidx = fidx
        self.quant_type = None
        self.softcap = softcap
        self.is_sliced = self.in_features < self.full_in_features or self.out_features < self.full_out_features
        self.out_dtype = out_dtype
        self.post_scale = post_scale
        self.transposed_load = transposed_load

        assert self.in_features_unpadded == self.in_features or allow_input_padding, \
            f"Input padding is not allowed for {self.key}, in_dim: {self.in_features_unpadded}, pad_to: {pad_to}"

        if caps is not None:
            self.caps.update(caps)


    @override
    def optimizer_targets(self):
        return [self.key]


    def pad_out(self, w: torch.Tensor | None) -> torch.Tensor | None:
        if w is None or self.out_features == self.out_features_unpadded and self.in_features == self.in_features_unpadded:
            return w
        if w.dim() == 2:
            padded = torch.zeros((self.in_features, self.out_features), dtype = w.dtype, device = w.device)
            if self.out_features != self.out_features_unpadded:
                padded[:, :w.shape[1]] = w
            elif self.in_features != self.in_features_unpadded:
                padded[:w.shape[0], :] = w
        else:
            assert w.dim() == 1
            padded = torch.zeros((self.out_features,), dtype = w.dtype, device = w.device)
            padded[:w.shape[0]] = w
        return padded.contiguous()


    def apply_fp8_scales_(self, weight: torch.Tensor, scale_inv: torch.Tensor):
        ws = weight.shape
        ss = scale_inv.shape
        assert len(ws) == len(ss) == 2
        assert all(w == s * 128 for w, s in zip(ws, ss))
        weight = weight.view(ss[0], 128, ss[1], 128)
        scale_inv = scale_inv.view(ss[0], 1, ss[1], 1)
        weight = weight.float() * scale_inv
        return weight.view(ws).half()


    def load_fp16(self, key: str) -> bool:

        if self.config.stc.has_tensor_group(key, ["weight"]):

            self.used_alt_key = key == self.alt_key
            dev = "cpu" if self.is_sliced else self.device
            pad1 = (self.out_features,) if not self.is_sliced else None
            pad2 = (self.in_features, self.out_features) if not self.is_sliced else None
            scale_inv = self.config.stc.get_tensor(key + ".weight_scale_inv", dev, transpose = self.transposed_load, optional = True, no_defer = True)
            weight = self.config.stc.get_tensor(key + ".weight", dev, float2half = True, transpose = self.transposed_load, pad_to = pad2, no_defer = scale_inv is not None)
            bias = self.config.stc.get_tensor(key + ".bias", dev, float2half = True, optional = True, pad_to = pad1)
            if scale_inv is not None:
                weight = self.apply_fp8_scales_(weight, scale_inv)
            self.inner = LinearFP16(
                self.in_features,
                self.out_features,
                weight,
                bias,
                self.full_in_features,
                self.full_out_features,
                self.first_in_feature,
                self.first_out_feature,
                self.out_dtype,
                key = self.key
            )
            if self.is_sliced:
                self.inner.swap_device = self.device
                self.inner.unswap_cpu()
            self.quant_type = "fp16"
            return True

        # Special dumb loading mode for Qwen3VLMoE
        elif self.fkey and self.config.stc.has_tensor(self.fkey) and self.fidx is not None:

            # Load and split fused weight along first dim
            weight = self.config.stc.get_tensor(
                self.fkey,
                self.device,
                transpose = self.transposed_load,
                no_defer = True,
                fidx = self.fidx
            )
            if self.frange is not None:
                weight = weight[self.frange[0] : self.frange[1]].contiguous()
            weight = self.pad_out(weight)
            self.inner = LinearFP16(
                self.in_features,
                self.out_features,
                weight.T.contiguous(),
                None,
                self.full_in_features,
                self.full_out_features,
                self.first_in_feature,
                self.first_out_feature,
                out_dtype = self.out_dtype,
                key = self.key
            )
            self.quant_type = "fp16"
            return True

        elif self.fkey and self.config.stc.has_tensor_group(self.fkey, ["weight"]) and self.frange is not None:

            weight = self.config.stc.get_tensor(
                self.fkey + ".weight",
                self.device,
                no_defer = True
            )
            weight = weight[self.frange[0] : self.frange[1]].contiguous()
            weight = self.pad_out(weight)
            bias = self.config.stc.get_tensor(
                self.fkey + ".bias",
                self.device,
                optional = True,
                no_defer = True
            )
            if bias is not None:
                bias = bias[self.frange[0] : self.frange[1]].contiguous()
                bias = self.pad_out(bias)
            self.inner = LinearFP16(
                self.in_features,
                self.out_features,
                weight.T.contiguous(),
                bias,
                self.full_in_features,
                self.full_out_features,
                self.first_in_feature,
                self.first_out_feature,
                out_dtype = self.out_dtype,
                key = self.key
            )
            self.quant_type = "fp16"
            return True

        return False


    def is_exl3_storage(self, key: str):
        return self.config.stc.has_tensor_group(
            key,
            [["sv", "svh"], ["su", "suh"], "trellis"]
        )

    def load_exl3(self, key: str) -> bool:
        if not self.is_exl3_storage(key):
            return False
        self.used_alt_key = key == self.alt_key
        scale = self.config.stc.get_tensor(key + ".scale", self.device, optional = True)
        su = self.config.stc.get_tensor(key + ".su", self.device, optional = True, no_defer = True)
        suh = self.config.stc.get_tensor(key + ".suh", self.device, optional = True)
        sv = self.config.stc.get_tensor(key + ".sv", self.device, optional = True, no_defer = True)
        svh = self.config.stc.get_tensor(key + ".svh", self.device, optional = True)
        trellis = self.config.stc.get_tensor(key + ".trellis", self.device)
        # TODO: We technically don't need to load these unless we need to save the tensors later
        mcg = self.config.stc.get_tensor(key + ".mcg", "cpu", optional = True)
        mul1 = self.config.stc.get_tensor(key + ".mul1", "cpu", optional = True)
        bias = self.config.stc.get_tensor(key + ".bias", self.device, optional = True)
        self.inner = LinearEXL3(
            self.config,
            self.in_features,
            self.out_features,
            scale,
            su,
            sv,
            suh,
            svh,
            trellis,
            mcg,
            mul1,
            bias,
            self.out_dtype,
            key = self.key
        )
        self.quant_type = "exl3"
        return True


    @override
    def can_defer_load(self):
        if self.is_sliced: return False
        return super().can_defer_load()


    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        keys = [self.key]
        if self.alt_key:
            keys += [self.alt_key]
        if any(self.load_exl3(k) for k in keys): return
        if any(self.load_fp16(k) for k in keys): return
        raise ValueError(f"No tensors found for {self.key} matching supported quantization format.")


    @override
    def unload(self):
        if self.inner is not None:
            self.inner.unload()
        self.device = None
        self.inner = None


    @override
    def get_tensors(self):
        if self.device:
            return self.inner.get_tensors(self.key)
        else:
            return {}


    def convert_exl3(
        self,
        H_data: dict,
        quant_args: dict,
        progress_str: str | None = None,
        return_weight_q: bool = False,
        verbose: bool = False,
        save_reg: str = None,
        override_swap_device: torch.device | None = None
    ):
        assert isinstance(self.inner, LinearFP16), \
            "Inner layer is already quant type"

        # Destroy original layer here to save VRAM, we only need weights
        swap_to_device = self.inner.swap_device  # in case weights are swapped to CPU
        if swap_to_device is not None and override_swap_device is not None:
            swap_to_device = override_swap_device

        orig_weight = self.inner.get_weight_tensor().float()
        orig_bias = self.inner.get_bias_tensor()
        self.inner = None

        weight_q, proxy_err, out_tensors = quantize_exl3(
            orig_weight,
            H_data,
            quant_args,
            return_weight_q,
            progress_str,
            verbose,
            swap_to_device,
            save_reg = save_reg
        )

        self.inner = LinearEXL3(
            self.config,
            self.in_features,
            self.out_features,
            out_tensors.get("scale"),
            out_tensors.get("su"),
            out_tensors.get("sv"),
            out_tensors.get("suh"),
            out_tensors.get("svh"),
            out_tensors.get("trellis"),
            out_tensors.get("mcg"),
            out_tensors.get("mul1"),
            orig_bias,
            self.out_dtype,
            key = self.key
        )

        if return_weight_q:
            return proxy_err, weight_q
        else:
            return proxy_err


    def capture_H(self, x: torch.Tensor, params: dict):
        if self.qmap not in params["capture"]:
            params["capture"][self.qmap] = {
                "H": torch.zeros(self.in_features, self.in_features, dtype = torch.float32, device = self.device),
                "first_key": self.key,
                "count": 0,
                "finalized": False,
                "num_total": 0,
                "inf_nan": torch.zeros(2, dtype = torch.long, device = self.device),
            }

        params["capture"][self.qmap]["num_total"] += x.numel()
        ext.count_inf_nan(x, params["capture"][self.qmap]["inf_nan"])

        if params["capture"][self.qmap]["first_key"] == self.key:
            rows = np.prod(x.shape[:-1])
            dim = x.shape[-1]
            x = x.view((rows, dim)).to(torch.float, copy = True)  # TODO: Why copy here?

            params["capture"][self.qmap]["H"].addmm_(x.T, x)
            params["capture"][self.qmap]["count"] += rows


    @override
    def weights_numel(self):
        return self.in_features * self.out_features


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        if "capture" in params and self.qmap:
            self.capture_H(x, params)

        x = self.inner.forward(x, params, out_dtype)
        if self.softcap != 0.0:
            ext.softcap(x, x, self.softcap)
        if self.post_scale != 1.0:
            x *= self.post_scale
        return x


    def allocate_q(self, quant_args: dict, surplus_bits: int):
        return allocate_linear(
            quant_args[self.qbits_key],
            surplus_bits,
            self
        )


    def quant_format_id(self):
        # alt_key is only used when loading unquantized model
        if self.is_exl3_storage(self.key):
            return "exl3"
        else:
            return None


    @cached_property
    def _storage_size(self):
        # alt_key is only used when loading unquantized model
        if self.is_exl3_storage(self.key):
            return sum(self.config.stc.get_tensor_sizes(prefix = self.key))
        else:
            return 2 * self.in_features * self.out_features
    def storage_size(self):
        return self._storage_size


    def recons_size(self):
        return 2 * self.in_features * self.out_features


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        storage = 0
        storage += self.storage_size()
        overhead_d = self.out_features * (self.out_dtype or torch.half).itemsize
        overhead_s = self.out_features * (self.out_dtype or torch.half).itemsize
        recons = self.recons_size()
        tpa = TPAllocation(
            key = self.key,
            channel_width = 128,
            channel_unit = "channels",
            storage_per_device = 0,
            storage_to_split = storage,
            overhead_per_device = overhead_d,
            overhead_to_split = overhead_s,
            recons_temp = recons,
            channels_to_split = self.out_features // 128,
            limit_key = "linear"
        )
        return [tpa]


    def tp_export(self, plan, producer):
        assert self.device is not None and self.inner is not None, "Cannot export module for TP before loading."
        return {
            "cls": Linear,
            "kwargs": {
                "key": self.key,
                "in_features": self.in_features,
                "out_features": self.out_features,
                "out_dtype": self.out_dtype,
                "alt_key": self.alt_key,
                "caps": self.caps,
                "softcap": self.softcap,
                "full_in_features": self.full_in_features,
                "full_out_features": self.full_out_features,
                "first_in_feature": self.first_in_feature,
                "first_out_feature": self.first_out_feature,
                "post_scale": self.post_scale,
            },
            "inner": self.inner.tp_export(plan, producer),
            "device": self.device
        }


    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        device = local_context["device"]
        module = Linear(
            config = None,
            **exported["kwargs"],
        )
        module.device = device
        module.inner = exported["inner"]["cls"].tp_import_split(local_context, exported["inner"], plan, split)
        module.quant_type = module.inner.quant_type
        module.out_features = module.inner.out_features
        return module

    @staticmethod
    def tp_import(local_context, exported, plan):
        key = exported["kwargs"]["key"]
        first, last, unit = plan[key] if key in plan else (None, None, "channels")
        assert unit == "channels"
        split = (True, first, last) if first is not None else None
        return Linear.tp_import_split(local_context, exported, plan, split)
