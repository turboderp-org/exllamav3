from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..models import Config
from ..util.tensor import to2
from . import Module, Linear
from .multilinear import MultiLinear
from ..ext import exllamav3_ext as ext
from ..constants import MAX_MLP_INTERMEDIATE
from ..util import first_not_none
from ..util import profile_opt
from dataclasses import dataclass
from .mlp import MLP, GatedMLP


@dataclass
class RoutingCFG:
    gate_tensor: torch.Tensor
    num_experts: int
    num_experts_per_tok: int
    router_logits_bsz1: torch.Tensor
    routing_weights_bsz1: torch.Tensor
    selected_experts_bsz1: torch.Tensor
    e_score_correction_bias: torch.Tensor | None
    routed_scaling_factor: float | None
    n_group: int | None
    topk_group: int | None


def routing_std(bsz, cfg, y, params):
    activate_all_experts = params.get("activate_all_experts")

    if bsz == 1 and not activate_all_experts:
        torch.matmul(y, cfg.gate_tensor, out = cfg.router_logits_bsz1)
        torch.topk(
            cfg.router_logits_bsz1,
            cfg.num_experts_per_tok,
            dim = -1,
            out = (cfg.routing_weights_bsz1, cfg.selected_experts_bsz1),
            sorted = False
        )
        torch.softmax(cfg.routing_weights_bsz1, dim = -1, out = cfg.routing_weights_bsz1)
        return cfg.selected_experts_bsz1, cfg.routing_weights_bsz1

    else:
        router_logits = torch.matmul(y, cfg.gate_tensor)
        routing_weights, selected_experts = torch.topk(
            router_logits,
            cfg.num_experts if activate_all_experts else cfg.num_experts_per_tok,
            dim = -1
        )
        routing_weights = torch.softmax(routing_weights, dim = -1)
        return selected_experts, routing_weights


# TODO: Optimize top_k groups (for DS3)
def routing_ds3(bsz, cfg, y, params):
    activate_all_experts = params.get("activate_all_experts")
    router_logits = torch.matmul(y, cfg.gate_tensor)

    scores = router_logits.sigmoid()
    scores_for_choice = scores.view(-1, cfg.num_experts) + cfg.e_score_correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(-1, cfg.n_group, cfg.num_experts // cfg.n_group)
        .topk(2, dim = -1)[0]
        .sum(dim = -1)
    )
    group_idx = torch.topk(group_scores, k = cfg.topk_group, dim = -1, sorted = False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, cfg.n_group, cfg.num_experts // cfg.n_group)
        .reshape(-1, cfg.num_experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

    topk_indices = torch.topk(
        scores_for_choice,
        k = cfg.num_experts if activate_all_experts else cfg.num_experts_per_tok,
        dim = -1,
        sorted = False
    )[1]
    topk_weights = scores.gather(1, topk_indices)
    denominator = topk_weights.sum(dim = -1, keepdim = True) + 1e-20
    topk_weights /= denominator
    topk_weights = topk_weights * cfg.routed_scaling_factor
    return topk_indices, topk_weights


def routing_dots(bsz, cfg, y, params):
    activate_all_experts = params.get("activate_all_experts")

    if bsz == 1 and not activate_all_experts:
        torch.matmul(y, cfg.gate_tensor, out = cfg.router_logits_bsz1)
        cfg.router_logits_bsz1 += cfg.e_score_correction_bias
        torch.topk(
            cfg.router_logits_bsz1,
            cfg.num_experts_per_tok,
            dim = -1,
            out = (cfg.routing_weights_bsz1, cfg.selected_experts_bsz1),
            sorted = False
        )
        # TODO: Custom kernel for sigmoid normalization
        cfg.routing_weights_bsz1.sigmoid_()
        factor = cfg.routed_scaling_factor / (cfg.routing_weights_bsz1.sum(dim = -1, keepdim = True) + 1e-20)
        cfg.routing_weights_bsz1 *= factor
        return cfg.selected_experts_bsz1, cfg.routing_weights_bsz1

    else:
        router_logits = torch.matmul(y, cfg.gate_tensor)
        router_logits += cfg.e_score_correction_bias
        routing_weights, selected_experts = torch.topk(
            router_logits,
            cfg.num_experts if activate_all_experts else cfg.num_experts_per_tok,
            dim = -1
        )
        # TODO: Custom kernel for sigmoid normalization
        routing_weights.sigmoid_()
        factor = cfg.routed_scaling_factor / (routing_weights.sum(dim = -1, keepdim = True) + 1e-20)
        routing_weights *= factor
        return selected_experts, routing_weights


@dataclass
class ExpertsCFG:
    yh: torch.Tensor
    interm_g: torch.Tensor
    interm_u: torch.Tensor
    interm_a: torch.Tensor
    out_d: torch.Tensor


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
        key_e_score_bias: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype = None,
        activation_fn: str = "silu",
        interm_dtype: torch.dtype = None,
        router_type: str = "std",
        routed_scaling_factor: float | None = None,
        n_group: int | None = None,
        topk_group: int | None = None,
        shared_experts: MLP | GatedMLP | None = None
    ):
        super().__init__(config, key, None)

        self.out_dtype = out_dtype
        self.interm_dtype = interm_dtype
        self.activation_fn = activation_fn
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size

        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group

        self.routing_gate = Linear(
            config = config,
            key = f"{key}.{key_routing_gate}",
            in_features = hidden_size,
            out_features = num_experts,
            qmap = None,
            out_dtype = torch.half,
            pad_to = 1,
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
                out_dtype = self.out_dtype,
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

        self.routing_cfg = None
        self.experts_cfg = None

        self.e_score_correction_bias = None
        self.e_score_correction_bias_key = key_e_score_bias or "gate.e_score_correction_bias"

        self.shared_experts = shared_experts
        if shared_experts is not None:
            self.register_submodule(shared_experts)

        match router_type:
            case "std": self.routing_fn = routing_std
            case "ds3": self.routing_fn = routing_ds3
            case "dots": self.routing_fn = routing_dots
            case _: raise ValueError(f"Unknown router type {router_type}")


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)

        self.e_score_correction_bias = \
            self.config.stc.get_tensor(f"{self.key}.{self.e_score_correction_bias_key}", self.device, optional = True)

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
            self.multi_gate = MultiLinear(self.device, self.gates)
            self.multi_up = MultiLinear(self.device, self.ups)
            self.multi_down = MultiLinear(self.device, self.downs)

        router_logits_bsz1 = torch.empty((1, self.num_experts), dtype = torch.half, device = self.device)
        routing_weights_bsz1 = torch.empty((1, self.num_experts_per_tok), dtype = torch.half, device = self.device)
        selected_experts_bsz1 = torch.empty((1, self.num_experts_per_tok), dtype = torch.long, device = self.device)

        self.routing_cfg = RoutingCFG(
            gate_tensor = self.routing_gate.inner.weight,
            num_experts = self.num_experts,
            num_experts_per_tok = self.num_experts_per_tok,
            router_logits_bsz1 = router_logits_bsz1,
            routing_weights_bsz1 = routing_weights_bsz1,
            selected_experts_bsz1 = selected_experts_bsz1,
            e_score_correction_bias = self.e_score_correction_bias,
            routed_scaling_factor = self.routed_scaling_factor,
            n_group = self.n_group,
            topk_group = self.topk_group,
        )

        yh = torch.empty(
            (self.num_experts_per_tok, 1, self.hidden_size),
            dtype = torch.half,
            device = self.device
        )
        interm_g = torch.empty(
            (self.num_experts_per_tok, 1, self.intermediate_size),
            dtype = self.interm_dtype,
            device = self.device
        )
        interm_u = torch.empty_like(interm_g)
        interm_a = torch.empty_like(interm_u, dtype = torch.half) if self.interm_dtype != torch.half else interm_u
        out_d = torch.empty(
            (self.num_experts_per_tok, 1, self.hidden_size),
            dtype = self.out_dtype or torch.half,
            device = self.device
        )

        self.experts_cfg = ExpertsCFG(
            yh = yh,
            interm_g = interm_g,
            interm_u = interm_u,
            interm_a = interm_a,
            out_d = out_d
        )

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
        self.routing_cfg = None
        self.experts_cfg = None
        self.e_score_correction_bias = None
        super().unload()


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        y = x.view(-1, self.hidden_size)
        bsz = y.shape[0]

        selected_experts, routing_weights = self.routing_fn(bsz, self.routing_cfg, y, params)

        # Torch path
        if bsz > 1 or not self.is_quantized:
            final_hidden_states = torch.zeros_like(y, dtype = self.out_dtype)

            expert_mask = torch.nn.functional.one_hot(
                selected_experts,
                num_classes = self.num_experts
            )
            expert_count = expert_mask.view(-1, self.num_experts).sum(dim = 0).cpu()
            expert_mask = expert_mask.permute(2, 1, 0)

            def mlp(exp_i, xc):
                g = self.gates[exp_i].forward(xc, params)
                u = self.ups[exp_i].forward(xc, params)
                a = u if self.interm_dtype == torch.half else torch.empty_like(u, dtype = torch.half)
                self.activation_fn_call(g, u, a)
                return self.downs[exp_i].forward(a, params)

            for expert_idx in range(self.num_experts):
                if expert_count[expert_idx] == 0:
                    continue
                idx, top_x = torch.where(expert_mask[expert_idx])
                current_state = y[None, top_x].reshape(-1, self.hidden_size)
                current_state = mlp(expert_idx, current_state) * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_state)

            final_hidden_states = final_hidden_states.reshape(x.shape)
            if self.shared_experts:
                final_hidden_states += self.shared_experts.forward(x, params)
            return to2(final_hidden_states, out_dtype, self.out_dtype)

        # Fused path
        # TODO: Find good solution for 1 < bsz < 32
        else:
            y = y.unsqueeze(0)

            cfg = self.experts_cfg

            # Gate
            ext.exl3_mgemm(
                y,
                self.multi_gate.ptrs_trellis,
                cfg.interm_g,
                self.multi_gate.ptrs_suh,
                cfg.yh,
                self.multi_gate.ptrs_svh,
                selected_experts,
                None,
                self.multi_gate.K,
                -1,
                self.multi_gate.mcg_mult,
                self.multi_gate.mul1_mult,
            )

            # Up
            ext.exl3_mgemm(
                y,
                self.multi_up.ptrs_trellis,
                cfg.interm_u,
                self.multi_up.ptrs_suh,
                cfg.yh,
                self.multi_up.ptrs_svh,
                selected_experts,
                None,
                self.multi_up.K,
                -1,
                self.multi_up.mcg_mult,
                self.multi_up.mul1_mult,
            )

            # Activation
            self.activation_fn_call(cfg.interm_g, cfg.interm_u, cfg.interm_a)

            # Down
            ext.exl3_mgemm(
                cfg.interm_a,
                self.multi_down.ptrs_trellis,
                cfg.out_d,
                self.multi_down.ptrs_suh,
                cfg.interm_a,
                self.multi_down.ptrs_svh,
                selected_experts,
                routing_weights,
                self.multi_down.K,
                -1,
                self.multi_down.mcg_mult,
                self.multi_down.mul1_mult,
            )

            final_hidden_states = cfg.out_d[:1, ...]
            final_hidden_states = final_hidden_states.view(x.shape)
            if self.shared_experts:
                final_hidden_states += self.shared_experts.forward(x, params)
            return final_hidden_states

    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.e_score_correction_bias is not None:
            t[f"{self.key}.gate.e_score_correction_bias"] = self.e_score_correction_bias.contiguous()
        return t