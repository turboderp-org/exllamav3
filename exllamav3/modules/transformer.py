from __future__ import annotations
from typing_extensions import override
import torch
from ..util.tensor import to2, get_for_device
from ..model.config import Config
from . import Module, RMSNorm, LayerNorm, Attention, GatedDeltaNet, GatedMLP, MLP, BlockSparseMLP, Linear
from ..util import profile_opt

class TransformerBlock(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int | None = None,
        ve_gate: Linear | None = None,
        resid_lambda: float | None = None,
        x0_lambda: float | None = None,
        attn_norm: RMSNorm | LayerNorm | None = None,
        attn: Attention | GatedDeltaNet | None = None,
        attn_post_norm: RMSNorm | LayerNorm | None = None,
        mlp_norm: RMSNorm | LayerNorm | None = None,
        mlp: MLP | GatedMLP | BlockSparseMLP | None = None,
        mlp_post_norm: RMSNorm | LayerNorm | None = None,
        backout_extract: bool = False,
        backout_lambda: float | None = None,
        qmap: str | None = None,
        qbits_key: str = "bits",
        out_dtype: torch.dtype = None
    ):
        super().__init__(config, key, None)

        self.layer_idx = layer_idx
        self.ve_gate = ve_gate
        self.resid_lambda = resid_lambda
        self.x0_lambda = x0_lambda
        self.attn_norm = attn_norm
        self.attn = attn
        self.attn_post_norm = attn_post_norm
        self.mlp_norm = mlp_norm
        self.mlp = mlp
        self.mlp_post_norm = mlp_post_norm
        self.backout_extract = backout_extract
        self.backout_lambda = backout_lambda
        self.qbits_key = qbits_key
        self.out_dtype = out_dtype

        self.register_submodule(self.ve_gate)
        self.register_submodule(self.attn_norm)
        self.register_submodule(self.attn)
        self.register_submodule(self.attn_post_norm)
        self.register_submodule(self.mlp_norm)
        self.register_submodule(self.mlp)
        self.register_submodule(self.mlp_post_norm)

        self.num_slices = mlp.num_slices if mlp else 1


    @override
    def optimizer_targets(self):
        a = self.attn.optimizer_targets() if self.attn else []
        m = self.mlp.optimizer_targets() if self.mlp else []
        return [a, m]


    def _apply_resid_lambda(self, x: torch.Tensor, params: dict):
        if self.layer_idx == 0:
            x0 = x.clone()
            params["_nc_x0"] = x0
            if "quant_preserve" in params:
                params["quant_preserve"]["_nc_x0"] = x0
        else:
            x0 = get_for_device(params, "_nc_x0", self.device)
        return self.resid_lambda * x + self.x0_lambda * x0


    def _extract_backout(self, x: torch.Tensor, params: dict):
        params["_nc_x_backout"] = x.clone()
        if "quant_preserve" in params:
            params["quant_preserve"]["_nc_x_backout"] = params["_nc_x_backout"]
        return x


    def _apply_backout(self, x: torch.Tensor, params: dict):
        xmid = get_for_device(params, "_nc_x_backout", self.device)
        if xmid is None:
            return x
        return x - self.backout_lambda * xmid


    def _compute_ve_addend(self, x: torch.Tensor, params: dict):
        ve = params[f"_nc_ve.{self.layer_idx}"].to(self.device)  # already on device, except while loading model
        y = x[..., :self.ve_gate.in_features].half()
        g = self.ve_gate.forward(y, params)
        g.sigmoid_()
        g *= 3
        params[f"_nc_ve.{self.layer_idx}"] = g.unsqueeze(-1) * ve
        return x


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if self.resid_lambda is not None:
            x = self._apply_resid_lambda(x, params)

        if self.backout_extract:
            x = self._extract_backout(x, params)

        if self.ve_gate:
            x = self._compute_ve_addend(x, params)

        if self.attn:
            if self.attn_norm:
                y = self.attn_norm.forward(x, params, out_dtype = torch.half)
            else:
                y = x.half()
            y = self.attn.forward(y, params)
            if params.get("prefill"): return x
            if self.attn_post_norm:
                y = self.attn_post_norm.forward(y, params)
            x += y

        if self.mlp:
            if self.mlp_norm:
                y = self.mlp_norm.forward(x, params, out_dtype = torch.half)
            else:
                y = x.half()
            y = self.mlp.forward(y, params)
            if self.mlp_post_norm:
                y = self.mlp_post_norm.forward(y, params)
            x += y

        if self.backout_lambda is not None:
            x = self._apply_backout(x, params)

        return to2(x, out_dtype, self.out_dtype)


    def get_name(self):
        name = super().get_name()
        if not self.attn and not self.mlp:
            name += " (no-op)"
        return name


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": TransformerBlock,
            "kwargs": {
                "key": self.key,
                "out_dtype": self.out_dtype,
            },
            **{name: _export(getattr(self, name, None)) for name in (
                "attn_norm",
                "attn",
                "attn_post_norm",
                "mlp_norm",
                "mlp",
                "mlp_post_norm",
            )},
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan):
        device = local_context["device"]

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        module = TransformerBlock(
            config = None,
            **exported["kwargs"],
            attn_norm = _import("attn_norm"),
            attn = _import("attn"),
            attn_post_norm = _import("attn_post_norm"),
            mlp_norm = _import("mlp_norm"),
            mlp = _import("mlp"),
            mlp_post_norm = _import("mlp_post_norm"),
        )
        module.device = device
        return module


class ParallelDecoderBlock(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        input_norm: RMSNorm | LayerNorm | None = None,
        attn: Attention | None = None,
        mlp: MLP | GatedMLP | None = None,
        qmap: str | None = None,
        qbits_key: str = "bits",
        out_dtype: torch.dtype = None
    ):
        super().__init__(config, key, None)

        self.input_norm = input_norm
        self.attn = attn
        self.mlp = mlp
        self.qbits_key = qbits_key
        self.out_dtype = out_dtype

        self.register_submodule(self.input_norm)
        self.register_submodule(self.attn)
        self.register_submodule(self.mlp)

        self.num_slices = mlp.num_slices if mlp else 1

        self.tp_reduce = False


    @override
    def optimizer_targets(self):
        a = self.attn.optimizer_targets() if self.attn else []
        m = self.mlp.optimizer_targets() if self.mlp else []
        return [a, m]


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        y = self.input_norm.forward(x, params, out_dtype = torch.half)
        y1 = self.attn.forward(y, params)
        if not params.get("prefill"):
            y2 = self.mlp.forward(y, params)
            y1 += y2

            if self.tp_reduce:
                params["backend"].all_reduce(y1)

            x += y1

        return to2(x, out_dtype, self.out_dtype)



    def get_name(self):
        name = super().get_name()
        if not self.attn and not self.mlp:
            name += " (no-op)"
        return name


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": ParallelDecoderBlock,
            "kwargs": {
                "key": self.key,
                "out_dtype": self.out_dtype,
            },
            **{name: _export(getattr(self, name, None)) for name in (
                "input_norm",
                "attn",
                "mlp",
            )},
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan):
        device = local_context["device"]

        def _import(name, **kwargs):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan, **kwargs) \
                if exported.get(name) else None

        module = ParallelDecoderBlock(
            config = None,
            **exported["kwargs"],
            input_norm = _import("input_norm"),
            attn = _import("attn", skip_reduction = True),
            mlp = _import("mlp", skip_reduction = True),
        )
        module.device = device

        # Use single reduction for sum of mlp and attn
        module.tp_reduce = True
        return module
