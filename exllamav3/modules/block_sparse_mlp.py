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
from ..util import first_not_none


class MultiLinear:
    def __init__(
        self,
        device: torch.Device,
        linears: list[Linear]
    ):
        self.device = device
        self.linears = linears
        self.num_linears = len(linears)

        assert all(l.quant_type == "exl3" for l in linears)
        assert all(l.inner.bias is None for l in linears)
        assert all(not l.softcap for l in linears)
        assert all(l.post_scale == 1.0 for l in linears)

        self.in_features = linears[0].in_features
        self.out_features = linears[0].out_features
        self.K = linears[0].inner.K
        assert all(l.inner.K == self.K for l in linears)
        assert all(l.in_features == self.in_features for l in linears)
        assert all(l.out_features == self.out_features for l in linears)

        self.ptrs_suh = torch.tensor([l.inner.suh.data_ptr() for l in linears], dtype = torch.long, device = device)
        self.ptrs_svh = torch.tensor([l.inner.svh.data_ptr() for l in linears], dtype = torch.long, device = device)
        self.ptrs_trellis = torch.tensor([l.inner.trellis.data_ptr() for l in linears], dtype = torch.long, device = device)

    def unload(self):
        pass


class BlockSparseMLP(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        key_up: str | None = None,
        key_gate: str | None = None,
        key_down: str | None = None,
        key_routing_gate: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype = None,
        activation_fn: str = "silu",
        interm_dtype: torch.dtype = None,
    ):
        super().__init__(config, key, None)

        self.out_dtype = out_dtype
        self.interm_dtype = interm_dtype
        self.activation_fn = activation_fn
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size

        self.routing_gate = Linear(
            config = config,
            key = f"{key}.{key_routing_gate}",
            in_features = hidden_size,
            out_features = num_experts,
            qmap = None,
            out_dtype = torch.half,
        )
        self.register_submodule(self.routing_gate)

        self.gates = []
        self.ups = []
        self.downs = []

        for idx in range(num_experts):

            gate = Linear(
                config = config,
                key = f"{key}.{key_gate}".replace("{expert_idx}", str(idx)),
                in_features = hidden_size,
                out_features = intermediate_size,
                qmap = qmap + ".input",
                out_dtype = self.interm_dtype
            )
            up = Linear(
                config = config,
                key = f"{key}.{key_up}".replace("{expert_idx}", str(idx)),
                in_features = hidden_size,
                out_features = intermediate_size,
                qmap = qmap + ".input",
                out_dtype = self.interm_dtype
            )
            down = Linear(
                config = config,
                key = f"{key}.{key_down}".replace("{expert_idx}", str(idx)),
                in_features = intermediate_size,
                out_features = hidden_size,
                qmap = qmap + f".{idx}.down",
                out_dtype = torch.half,
                allow_input_padding = True,
            )

            self.ups.append(up)
            self.gates.append(gate)
            self.downs.append(down)

            self.register_submodule(up)
            self.register_submodule(gate)
            self.register_submodule(down)

        match activation_fn:
            case "silu": self.activation_fn_call = ext.silu_mul
            case "gelu": self.activation_fn_call = ext.gelu_mul

        self.is_quantized = False
        self.multi_gate = None
        self.multi_up = None
        self.multi_down = None


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)

        # Test if experts can be fused
        num_exl3_tensors = 0
        num_nonexl3_tensors = 0
        for l in self.gates + self.ups + self.downs:
            if l.quant_type == "exl3":
                num_exl3_tensors += 1
            else:
                num_nonexl3_tensors += 1
        if num_exl3_tensors and num_nonexl3_tensors:
            print(f" !! Warning, partially quantized block-sparse MLP layer: {self.key}")
        self.is_quantized = (num_exl3_tensors > 0 and num_nonexl3_tensors == 0)

        # Make fused modules
        if self.is_quantized:
            self.multi_gate = MultiLinear(self. device, self.gates)
            self.multi_up = MultiLinear(self. device, self.ups)
            self.multi_down = MultiLinear(self. device, self.downs)


    @override
    def unload(self):
        if self.multi_gate is not None:
            self.multi_gate.unload()
            self.multi_gate = None
        if self.multi_up is not None:
            self.multi_up.unload()
            self.multi_up = None
        if self.multi_down is not None:
            self.multi_down.unload()
            self.multi_down = None
        super().unload()


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        activate_all_experts = params.get("activate_all_experts", False)

        y = x.view(-1, self.hidden_size)
        bsz = y.shape[0]

        router_logits = self.routing_gate.forward(y, params)
        routing_weights = F.softmax(router_logits, dim = -1)
        routing_weights, selected_experts = torch.topk(
            routing_weights,
            self.num_experts if activate_all_experts else self.num_experts_per_tok,
            dim = -1
        )
        routing_weights /= routing_weights.sum(dim = -1, keepdim = True)

        # Torch path
        if bsz > 1 or not self.is_quantized:
            final_hidden_states = torch.zeros_like(y)

            expert_mask = torch.nn.functional.one_hot(
                selected_experts,
                num_classes = self.num_experts
            )
            expert_count = expert_mask.view(-1, self.num_experts).sum(dim = 0).cpu()
            expert_mask = expert_mask.permute(2, 1, 0)

            def mlp(exp_i, xc):
                g = self.gates[exp_i].forward(xc, params)
                u = self.ups[exp_i].forward(xc, params)
                self.activation_fn_call(g, u, u)
                return self.downs[exp_i].forward(u, params)

            for expert_idx in range(self.num_experts):
                if expert_count[expert_idx] == 0:
                    continue
                idx, top_x = torch.where(expert_mask[expert_idx])
                current_state = y[None, top_x].reshape(-1, self.hidden_size)
                current_state = mlp(expert_idx, current_state) * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_state)

            final_hidden_states = final_hidden_states.reshape(x.shape)
            return to2(final_hidden_states, out_dtype, self.out_dtype)

        # Fused path
        # TODO: Find good solution for 1 < bsz < 32
        else:
            y = y.unsqueeze(0)
            yh = torch.empty(
                (self.num_experts_per_tok, bsz, y.shape[-1]),
                dtype = y.dtype,
                device = y.device
            )
            interm_g = torch.empty(
                (self.num_experts_per_tok, bsz, self.intermediate_size),
                dtype = self.interm_dtype,
                device = y.device
            )
            interm_u = torch.empty_like(interm_g)
            interm_a = torch.empty_like(interm_u, dtype = torch.half) if self.interm_dtype != torch.half else interm_u
            out_d = torch.empty(
                (self.num_experts_per_tok, bsz, self.hidden_size),
                dtype = first_not_none(out_dtype, self.out_dtype, torch.half),
                device = y.device
            )

            # Gate
            ext.exl3_mgemm(
                y,
                self.multi_gate.ptrs_trellis,
                interm_g,
                self.multi_gate.ptrs_suh,
                yh,
                self.multi_gate.ptrs_svh,
                selected_experts,
                None,
                self.multi_gate.K,
                -1
            )

            # Up
            ext.exl3_mgemm(
                y,
                self.multi_up.ptrs_trellis,
                interm_u,
                self.multi_up.ptrs_suh,
                yh,
                self.multi_up.ptrs_svh,
                selected_experts,
                None,
                self.multi_up.K,
                -1
            )

            # Activation
            self.activation_fn_call(interm_g, interm_u, interm_a)

            # Down
            ext.exl3_mgemm(
                interm_a,
                self.multi_down.ptrs_trellis,
                out_d,
                self.multi_down.ptrs_suh,
                interm_a,
                self.multi_down.ptrs_svh,
                selected_experts,
                routing_weights,
                self.multi_down.K,
                -1
            )

            final_hidden_states = out_d.sum(dim = 0)
            return final_hidden_states.view(x.shape)