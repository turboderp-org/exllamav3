from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..util.tensor import to2
from ..model.config import Config
from . import Module, RMSNorm, LayerNorm, Attention, GatedDeltaNet, GatedMLP, MLP, BlockSparseMLP
from ..conversion.allocation import allocate_transformer
from ..util import profile_opt
import torch.distributed as dist

class TransformerBlock(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        attn_norm: RMSNorm | LayerNorm | None = None,
        attn: Attention | GatedDeltaNet | None = None,
        attn_post_norm: RMSNorm | LayerNorm | None = None,
        mlp_norm: RMSNorm | LayerNorm | None = None,
        mlp: MLP | GatedMLP | BlockSparseMLP | None = None,
        mlp_post_norm: RMSNorm | LayerNorm | None = None,
        qmap: str | None = None,
        qbits_key: str = "bits",
        out_dtype: torch.dtype = None
    ):
        super().__init__(config, key, None)

        self.attn_norm = attn_norm
        self.attn = attn
        self.attn_post_norm = attn_post_norm
        self.mlp_norm = mlp_norm
        self.mlp = mlp
        self.mlp_post_norm = mlp_post_norm
        self.qbits_key = qbits_key
        self.out_dtype = out_dtype

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


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

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

        return to2(x, out_dtype, self.out_dtype)


    def allocate_q(self, quant_args: dict, surplus_bits: int):

        if not self.attn and not self.mlp:
            return {}, surplus_bits

        g = self.mlp.gates if any(isinstance(self.mlp, x) for x in [GatedMLP, BlockSparseMLP]) else None
        u = self.mlp.ups if self.mlp else None
        d = self.mlp.downs if self.mlp else None
        if self.mlp and isinstance(self.mlp, BlockSparseMLP) and self.mlp.shared_experts:
            g = g + self.mlp.shared_experts.gates
            u = u + self.mlp.shared_experts.ups
            d = d + self.mlp.shared_experts.downs

        q, k, v, o = None, None, None, None
        qkvz = None
        if self.attn:
            if isinstance(self.attn, Attention):
                q = self.attn.q_proj
                k = self.attn.k_proj
                v = self.attn.v_proj
                o = self.attn.o_proj
            elif isinstance(self.attn, GatedDeltaNet):
                qkvz = self.attn.qkvz_proj
                o = self.attn.o_proj

        return allocate_transformer(
            quant_args[self.qbits_key],
            surplus_bits,
            q, k, v, o, g, u, d, qkvz
        )

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


    def allocate_q(self, quant_args: dict, surplus_bits: int):
        if self.attn:
            assert isinstance(self.attn, Attention)

        if not self.attn and not self.mlp:
            return {}, surplus_bits

        return allocate_transformer(
            quant_args[self.qbits_key],
            surplus_bits,
            self.attn.q_proj if self.attn else None,
            self.attn.k_proj if self.attn else None,
            self.attn.v_proj if self.attn else None,
            self.attn.o_proj if self.attn else None,
            self.mlp.gates if any(isinstance(self.mlp, x) for x in [GatedMLP, BlockSparseMLP]) else None,
            self.mlp.ups if self.mlp else None,
            self.mlp.downs if self.mlp else None,
        )


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