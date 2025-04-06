from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..models import Config
from ..util.tensor import to2
from . import Module, Linear
from ..ext import exllamav3_ext as ext

class MLP(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        key_up: str | None = None,
        key_down: str | None = None,
        key_fused_gate_up: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype = None
    ):
        super().__init__(config, key, None)
        assert key_fused_gate_up is None

        self.out_dtype = out_dtype

        self.up = Linear(config, f"{key}.{key_up}", hidden_size, intermediate_size, qmap = qmap + ".up")
        self.down = Linear(config, f"{key}.{key_down}", intermediate_size, hidden_size, qmap = qmap + ".down")

        self.register_submodule(self.up)
        self.register_submodule(self.down)

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        x = self.up.forward(x, params)
        x = F.silu(x)
        x = self.down.forward(x, params)

        return to2(x, out_dtype, self.out_dtype)


class GatedMLP(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        key_up: str | None = None,
        key_gate: str | None = None,
        key_down: str | None = None,
        key_fused_gate_up: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype = None,
        activation_fn: str = "silu"
    ):
        super().__init__(config, key, None)

        self.out_dtype = out_dtype
        self.activation_fn = activation_fn

        if key_fused_gate_up:
            fkey = f"{key}.{key_fused_gate_up}"
            frange_gate = (0, intermediate_size)
            frange_up = (intermediate_size, 2 * intermediate_size)
        else:
            fkey, frange_gate, frange_up = None, None, None

        self.gate = Linear(config, f"{key}.{key_gate}", hidden_size, intermediate_size, qmap = qmap + ".up", fkey = fkey, frange = frange_gate)
        self.up = Linear(config, f"{key}.{key_up}", hidden_size, intermediate_size, qmap = qmap + ".up", fkey = fkey, frange = frange_up)
        self.down = Linear(config, f"{key}.{key_down}", intermediate_size, hidden_size, qmap = qmap + ".down")

        self.register_submodule(self.up)
        self.register_submodule(self.gate)
        self.register_submodule(self.down)

        match activation_fn:
            case "silu": self.activation_fn_call = ext.silu_mul
            case "gelu": self.activation_fn_call = ext.gelu_mul

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        g = self.gate.forward(x, params)
        x = self.up.forward(x, params)
        self.activation_fn_call(g, x, x)
        x = self.down.forward(x, params)

        return to2(x, out_dtype, self.out_dtype)
