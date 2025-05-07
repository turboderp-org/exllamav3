from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..models import Config
from ..util.tensor import to2
from . import Module, Linear
from ..ext import exllamav3_ext as ext
from ..constants import MAX_MLP_INTERMEDIATE

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
        activation_fn: str = "silu",
        intermediate_split_size: int | None = MAX_MLP_INTERMEDIATE,
        interm_dtype: torch.dtype = None,
    ):
        super().__init__(config, key, None)

        self.out_dtype = out_dtype
        self.interm_dtype = interm_dtype
        self.activation_fn = activation_fn
        self.intermediate_size = intermediate_size

        if key_fused_gate_up:
            assert not intermediate_split_size or intermediate_size <= intermediate_split_size, \
                "Cannot combine fused gate/up layers with MLP slicing"
            fkey = f"{key}.{key_fused_gate_up}"
            frange_gate = (0, intermediate_size)
            frange_up = (intermediate_size, 2 * intermediate_size)
        else:
            fkey, frange_gate, frange_up = None, None, None

        if intermediate_split_size and intermediate_size > intermediate_split_size:
            num_slices = (intermediate_size + intermediate_split_size - 1) // intermediate_split_size
            interm_slice = intermediate_size // num_slices // 128 * 128
            interm_split = [interm_slice for _ in range(num_slices)]
            interm_split[-1] += intermediate_size - sum(interm_split)
            self.num_slices = num_slices
        else:
            interm_split = [intermediate_size]
            self.num_slices = 1

        self.gates = []
        self.ups = []
        self.downs = []

        a = 0
        for idx, sp in enumerate(interm_split):
            b = a + sp

            if self.num_slices > 1:
                s_key_g = f"{key}.{key_gate}.slice.{idx}"
                s_key_u = f"{key}.{key_up}.slice.{idx}"
                s_key_d = f"{key}.{key_down}.slice.{idx}"
                a_key_g = f"{key}.{key_gate}"
                a_key_u = f"{key}.{key_up}"
                a_key_d = f"{key}.{key_down}"
            else:
                s_key_g = f"{key}.{key_gate}"
                s_key_u = f"{key}.{key_up}"
                s_key_d = f"{key}.{key_down}"
                a_key_g = None
                a_key_u = None
                a_key_d = None

            gate = Linear(
                config = config,
                key = s_key_g,
                in_features = hidden_size,
                out_features = b - a,
                full_in_features = hidden_size,
                full_out_features = intermediate_size,
                first_in_feature = 0,
                first_out_feature = a,
                qmap = qmap + ".input",
                fkey = fkey,
                frange = frange_gate,
                alt_key = a_key_g,
                out_dtype = self.interm_dtype
            )
            up = Linear(
                config = config,
                key = s_key_u,
                in_features = hidden_size,
                out_features = b - a,
                full_in_features = hidden_size,
                full_out_features = intermediate_size,
                first_in_feature = 0,
                first_out_feature = a,
                qmap = qmap + ".input",
                fkey = fkey,
                frange = frange_up,
                alt_key = a_key_u,
                out_dtype = self.interm_dtype
            )
            down = Linear(
                config = config,
                key = s_key_d,
                in_features = b - a,
                out_features = hidden_size,
                full_in_features = intermediate_size,
                full_out_features = hidden_size,
                first_in_feature = a,
                first_out_feature = 0,
                qmap = qmap + ".down",
                alt_key = a_key_d,
                out_dtype = self.out_dtype,
                allow_input_padding = True,
            )

            self.ups.append(up)
            self.gates.append(gate)
            self.downs.append(down)

            self.register_submodule(up)
            self.register_submodule(gate)
            self.register_submodule(down)

            a = b

        match activation_fn:
            case "silu": self.activation_fn_call = ext.silu_mul
            case "gelu": self.activation_fn_call = ext.gelu_mul


    @override
    def can_defer_load(self):
        if self.num_slices > 1: return False
        return super().can_defer_load()


    @override
    def load(self, device: torch.Device, load_slice: int | None = None, **kwargs):
        if load_slice is None:
            super().load(device, **kwargs)
        else:
            self.gates[load_slice].load(device, **kwargs)
            self.ups[load_slice].load(device, **kwargs)
            self.downs[load_slice].load(device, **kwargs)


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        qs = params.get("q_mlp_slice")
        r = [qs] if qs is not None else range(0, self.num_slices)
        d = None

        for s in r:
            g = self.gates[s].forward(x, params)
            u = self.ups[s].forward(x, params)
            self.activation_fn_call(g, u, u)
            d_ = self.downs[s].forward(u, params)
            if d is None: d = d_
            else: d += d_
            del d_

        return to2(d, out_dtype, self.out_dtype)
