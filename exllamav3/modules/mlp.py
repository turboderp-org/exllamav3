from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..model.config import Config
from ..util.tensor import to2
from . import Module, Linear
from ..ext import exllamav3_ext as ext
from ..constants import MAX_MLP_INTERMEDIATE
from ..model.model_tp_alloc import TPAllocation
import torch.distributed as dist
from .multilinear import MultiLinear

class MLP(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        out_size: int | None = None,
        key_up: str | None = None,
        key_down: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype = None,
        activation_fn: str = "silu",
        intermediate_split_size: int | None = MAX_MLP_INTERMEDIATE,
        interm_dtype: torch.dtype = None,
        pad_to = 128,
        ups: list[Linear | Module] = None,
        downs: list[Linear | Module] = None,
    ):
        super().__init__(config, key, None)

        self.hidden_size = hidden_size
        self.pad_to = pad_to
        self.out_dtype = out_dtype
        self.interm_dtype = interm_dtype
        self.activation_fn = activation_fn
        self.intermediate_size = intermediate_size
        self.intermediate_split_size = intermediate_split_size
        self.out_size = out_size or hidden_size

        fkey, frange_up = None, None

        if ups is not None:
            assert downs is not None and len(downs) == len(ups)
            self.num_slices = len(ups)
            self.ups = ups
            self.downs = downs

        else:
            if intermediate_split_size and intermediate_size > intermediate_split_size:
                num_slices = (intermediate_size + intermediate_split_size - 1) // intermediate_split_size
                interm_slice = intermediate_size // num_slices // 128 * 128
                interm_split = [interm_slice for _ in range(num_slices)]
                interm_split[-1] += intermediate_size - sum(interm_split)
                self.num_slices = num_slices
            else:
                interm_split = [intermediate_size]
                self.num_slices = 1

            self.ups = []
            self.downs = []

            a = 0
            for idx, sp in enumerate(interm_split):
                b = a + sp

                if self.num_slices > 1:
                    s_key_u = f"{key}.{key_up}.slice.{idx}"
                    s_key_d = f"{key}.{key_down}.slice.{idx}"
                    a_key_u = f"{key}.{key_up}"
                    a_key_d = f"{key}.{key_down}"
                else:
                    s_key_u = f"{key}.{key_up}"
                    s_key_d = f"{key}.{key_down}"
                    a_key_u = None
                    a_key_d = None

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
                    out_dtype = self.interm_dtype,
                    qbits_mod_key = "u",
                    pad_to = pad_to
                )
                down = Linear(
                    config = config,
                    key = s_key_d,
                    in_features = b - a,
                    out_features = self.out_size,
                    full_in_features = intermediate_size,
                    full_out_features = self.out_size,
                    first_in_feature = a,
                    first_out_feature = 0,
                    qmap = qmap + ".down",
                    alt_key = a_key_d,
                    out_dtype = self.out_dtype,
                    allow_input_padding = True,
                    qbits_mod_key = "d",
                    pad_to = pad_to
                )

                self.ups.append(up)
                self.downs.append(down)

                self.register_submodule(up)
                self.register_submodule(down)

                a = b

        self.activation_fn = activation_fn

        match activation_fn:
            case "silu": self.activation_fn_call = F.silu
            case "gelu": self.activation_fn_call = lambda x: F.gelu(x, approximate = "tanh")
            case "relu2": self.activation_fn_call = lambda x: torch.square(F.relu(x))

        self.tp_reduce = False


    @override
    def can_defer_load(self):
        if self.num_slices > 1: return False
        return super().can_defer_load()


    @override
    def load(self, device: torch.Device, load_slice: int | None = None, **kwargs):
        if load_slice is None:
            super().load(device, **kwargs)
        else:
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
            u = self.ups[s].forward(x, params)
            # TODO: mixed precision activation kernel?
            a = self.activation_fn_call(u)
            d_ = self.downs[s].forward(a, params)
            if d is None: d = d_
            else: d += d_
            del d_

        if self.tp_reduce:
            params["backend"].all_reduce(d)

        return to2(d, out_dtype, self.out_dtype)


    def make_tp_allocation(self) -> list[TPAllocation]:
        storage = 0
        ims = min(self.intermediate_size, self.intermediate_split_size)
        for u in self.ups: storage += u.storage_size()
        for d in self.downs: storage += d.storage_size()
        overhead_d = self.hidden_size * (self.out_dtype or torch.half).itemsize
        overhead_s = 2 * ims * (self.interm_dtype or torch.half).itemsize
        if self.interm_dtype != torch.half:
            overhead_s += ims * torch.half.itemsize
        recons = max(
            self.ups[0].recons_size(),
            self.downs[0].recons_size()
        )
        slice = self.num_slices > 1
        tpa = TPAllocation(
            key = self.key,
            channel_width = 1 if slice else 128,
            channel_unit = "slices" if slice else "channels",
            storage_per_device = 0,
            storage_to_split = storage,
            overhead_per_device = overhead_d,
            overhead_to_split = overhead_s,
            recons_temp = recons,
            channels_to_split = self.num_slices if slice else self.intermediate_size // 128,
            limit_key = "mlp"
        )
        return [tpa]


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."
        assert self.pad_to == 128, "Cannot export module for TP unless pad_to == 128."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": MLP,
            "kwargs": {
                "key": self.key,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "activation_fn": self.activation_fn,
                "out_dtype": self.out_dtype,
                "interm_dtype": self.interm_dtype,
                "intermediate_split_size": self.intermediate_split_size,
            },
            "ups": [_export(self.ups[i]) for i in range(self.num_slices)],
            "downs": [_export(self.downs[i]) for i in range(self.num_slices)],
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        key = exported["kwargs"]["key"]
        device = local_context["device"]
        first, last = plan[key]
        num_slices = len(exported["ups"])

        if first >= last:
            num_slices = 0

        def _import_i_split(name, i, split):
            nonlocal exported, plan
            return exported[name][i]["cls"].tp_import_split(local_context, exported[name][i], plan, split) \
                if exported.get(name) else None

        def _import_i(name, i):
            nonlocal exported, plan
            return exported[name][i]["cls"].tp_import(local_context, exported[name][i], plan) \
                if exported.get(name) else None

        if num_slices == 1:
            u_split = (True, first, last)
            d_split = (False, first, last)
            module = MLP(
                config = None,
                **exported["kwargs"],
                ups = [_import_i_split("ups", 0, u_split)],
                downs = [_import_i_split("downs", 0, d_split)],
            )

        elif num_slices > 1:
            module = MLP(
                config = None,
                **exported["kwargs"],
                ups = [_import_i("ups", i) for i in range(first, last)],
                downs = [_import_i("downs", i) for i in range(first, last)],
            )

        else:
            module = MLP(
                config = None,
                **exported["kwargs"],
                ups = [],
                downs = [],
            )

        module.device = device
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True
        torch.cuda.synchronize()
        return module


class GatedMLP(Module):

    def __init__(
        self,
        config: Config | None,
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
        pad_to = 128,
        gates: list[Linear | Module] = None,
        ups: list[Linear | Module] = None,
        downs: list[Linear | Module] = None,
    ):
        super().__init__(config, key, None)

        self.out_dtype = out_dtype
        self.interm_dtype = interm_dtype
        self.activation_fn = activation_fn
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.intermediate_split_size = intermediate_split_size
        self.pad_to = pad_to

        if key_fused_gate_up:
            assert not intermediate_split_size or intermediate_size <= intermediate_split_size, \
                "Cannot combine fused gate/up layers with MLP slicing"
            fkey = f"{key}.{key_fused_gate_up}"
            frange_gate = (0, intermediate_size)
            frange_up = (intermediate_size, 2 * intermediate_size)
        else:
            fkey, frange_gate, frange_up = None, None, None

        if gates is not None:
            assert ups is not None and len(ups) == len(gates)
            assert downs is not None and len(downs) == len(gates)
            self.num_slices = len(gates)
            self.gates = gates
            self.ups = ups
            self.downs = downs

        else:
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
                    out_dtype = self.interm_dtype,
                    qbits_mod_key = "g",
                    pad_to = pad_to
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
                    out_dtype = self.interm_dtype,
                    qbits_mod_key = "u",
                    pad_to = pad_to
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
                    qbits_mod_key = "d",
                    pad_to = pad_to
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
            case "relu2": self.activation_fn_call = ext.relu2_mul
            case "gelu": self.activation_fn_call = ext.gelu_mul

        self.tp_reduce = False
        self.multi_gu: list[MultiLinear | None] = [None] * self.num_slices


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

        self.load_local(device, load_slice or 0, **kwargs)


    def load_local(self, device: torch.Device, load_slice: int, **kwargs):
        # Test if gate and up proj can be fused
        # TODO: Appears to be broken for MLPs using padding on gate/up. Disable until that's sorted out
        pass
        # if (
        #     device != torch.device("cpu") and
        #     self.gates[load_slice].quant_type == "exl3" and
        #     self.ups[load_slice].quant_type == "exl3" and
        #     self.gates[load_slice].out_features == self.ups[load_slice].out_features and
        #     self.gates[load_slice].inner.K == self.ups[load_slice].inner.K and
        #     self.gates[load_slice].inner.bias is None and
        #     self.ups[load_slice].inner.bias is None
        # ):
        #     self.multi_gu[load_slice] = MultiLinear(self.device, [self.gates[load_slice], self.ups[load_slice]])


    @override
    def unload(self):
        super().unload()

        for i in range(self.num_slices):
            if self.multi_gu[i] is not None:
                self.multi_gu[i].unload()
                self.multi_gu[i] = None


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        bsz, q_len, dim = x.shape

        if self.num_slices == 0:
            d = torch.zeros_like(x, dtype = self.out_dtype)
            if self.tp_reduce:
                params["backend"].all_reduce(d, False)
        else:
            qs = params.get("q_mlp_slice")
            r = [qs] if qs is not None else range(0, self.num_slices)
            d = None

            for s in r:

                if self.multi_gu[s] is None or bsz * q_len > 32:
                    g = self.gates[s].forward(x, params)
                    u = self.ups[s].forward(x, params)

                else:
                    x = x.view(1, bsz * q_len, dim)
                    guh = torch.empty((2, bsz * q_len, self.multi_gu[s].out_features), dtype = self.interm_dtype, device = x.device)
                    gu = torch.empty((2, bsz * q_len, self.multi_gu[s].out_features), dtype = self.interm_dtype, device = x.device)
                    ext.exl3_mgemm(
                        x,
                        self.multi_gu[s].ptrs_trellis,
                        gu,
                        self.multi_gu[s].ptrs_suh,
                        guh,
                        self.multi_gu[s].ptrs_svh,
                        None,
                        None,
                        self.multi_gu[s].K,
                        -1,
                        self.multi_gu[s].mcg_mult,
                        self.multi_gu[s].mul1_mult,
                        -1,
                        -1
                    )
                    g = gu[0].view(bsz, q_len, self.multi_gu[s].out_features)
                    u = gu[1].view(bsz, q_len, self.multi_gu[s].out_features)

                a = torch.empty_like(u, dtype = torch.half) if self.interm_dtype != torch.half else u
                self.activation_fn_call(g, u, a)
                d_ = self.downs[s].forward(a, params)

                if d is None: d = d_
                else: d += d_
                del d_
            if self.tp_reduce:
                params["backend"].all_reduce(d)

        return to2(d, out_dtype, self.out_dtype)


    def make_tp_allocation(self) -> list[TPAllocation]:
        storage = 0
        ims = min(self.intermediate_size, self.intermediate_split_size)
        for g in self.gates: storage += g.storage_size()
        for u in self.ups: storage += u.storage_size()
        for d in self.downs: storage += d.storage_size()
        overhead_d = self.hidden_size * (self.out_dtype or torch.half).itemsize
        overhead_s = 2 * ims * (self.interm_dtype or torch.half).itemsize
        if self.interm_dtype != torch.half:
            overhead_s += ims * torch.half.itemsize
        recons = max(
            self.gates[0].recons_size(),
            self.ups[0].recons_size(),
            self.downs[0].recons_size()
        )
        slice = self.num_slices > 1
        tpa = TPAllocation(
            key = self.key,
            channel_width = 1 if slice else 128,
            channel_unit = "slices" if slice else "channels",
            storage_per_device = 0,
            storage_to_split = storage,
            overhead_per_device = overhead_d,
            overhead_to_split = overhead_s,
            recons_temp = recons,
            channels_to_split = self.num_slices if slice else self.intermediate_size // 128,
            limit_key = "mlp"
        )
        return [tpa]


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."
        assert self.pad_to == 128, "Cannot export module for TP unless pad_to == 128."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": GatedMLP,
            "kwargs": {
                "key": self.key,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "activation_fn": self.activation_fn,
                "out_dtype": self.out_dtype,
                "interm_dtype": self.interm_dtype,
                "intermediate_split_size": self.intermediate_split_size,
            },
            "gates": [_export(self.gates[i]) for i in range(self.num_slices)],
            "ups": [_export(self.ups[i]) for i in range(self.num_slices)],
            "downs": [_export(self.downs[i]) for i in range(self.num_slices)],
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        key = exported["kwargs"]["key"]
        device = local_context["device"]
        first, last = plan[key]
        num_slices = len(exported["gates"])

        if first >= last:
            num_slices = 0

        def _import_i_split(name, i, split):
            nonlocal exported, plan
            return exported[name][i]["cls"].tp_import_split(local_context, exported[name][i], plan, split) \
                if exported.get(name) else None

        def _import_i(name, i):
            nonlocal exported, plan
            return exported[name][i]["cls"].tp_import(local_context, exported[name][i], plan) \
                if exported.get(name) else None

        if num_slices == 1:
            gu_split = (True, first, last)
            d_split = (False, first, last)
            exported["kwargs"]["intermediate_size"] = last - first
            module = GatedMLP(
                config = None,
                **exported["kwargs"],
                gates = [_import_i_split("gates", 0, gu_split)],
                ups = [_import_i_split("ups", 0, gu_split)],
                downs = [_import_i_split("downs", 0, d_split)],
            )

        elif num_slices > 1:
            module = GatedMLP(
                config = None,
                **exported["kwargs"],
                gates = [_import_i("gates", i) for i in range(first, last)],
                ups = [_import_i("ups", i) for i in range(first, last)],
                downs = [_import_i("downs", i) for i in range(first, last)],
            )

        else:
            module = GatedMLP(
                config = None,
                **exported["kwargs"],
                gates = [],
                ups = [],
                downs = [],
            )

        module.device = device
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True
        for i in range(module.num_slices):
            module.load_local(device, i)
        torch.cuda.synchronize()
        return module