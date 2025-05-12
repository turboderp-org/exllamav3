from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..util.tensor import to2
from ..models import Config
from . import Module, RMSNorm, LayerNorm, Attention, GatedMLP, MLP, BlockSparseMLP
from ..conversion.allocation import allocate_transformer

class TransformerBlock(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        attn_norm: RMSNorm | None = None,
        attn: Attention | None = None,
        attn_post_norm: RMSNorm | None = None,
        mlp_norm: RMSNorm | LayerNorm | None = None,
        mlp: MLP | GatedMLP | BlockSparseMLP | None = None,
        mlp_post_norm: RMSNorm | None = None,
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
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if self.attn:
            if self.attn_norm:
                y = self.attn_norm.forward(x, params, out_dtype = torch.half)
            y = self.attn.forward(y, params)
            if params.get("prefill"): return x
            if self.attn_post_norm:
                y = self.attn_post_norm.forward(y, params)
            x += y

        if self.mlp:
            if self.mlp_norm:
                y = self.mlp_norm.forward(x, params, out_dtype = torch.half)
            y = self.mlp.forward(y, params)
            if self.mlp_post_norm:
                y = self.mlp_post_norm.forward(y, params)
            x += y

        return to2(x, out_dtype, self.out_dtype)


    def allocate_q(self, quant_args: dict, surplus_bits: int):
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


class ParallelDecoderBlock(Module):

    def __init__(
        self,
        config: Config,
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


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        y = self.input_norm.forward(x, params, out_dtype = torch.half)
        y1 = self.attn.forward(y, params)
        x += y1
        if params.get("prefill"): return x
        y2 = self.mlp.forward(y, params)
        x += y2

        return to2(x, out_dtype, self.out_dtype)


    def allocate_q(self, quant_args: dict, surplus_bits: int):
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