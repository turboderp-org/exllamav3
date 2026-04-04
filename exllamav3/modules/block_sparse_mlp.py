from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..util.tensor import to2
from . import Module, Linear
from .multilinear import MultiLinear
from ..ext import exllamav3_ext as ext
from dataclasses import dataclass
from .mlp import MLP, GatedMLP
from ..model.model_tp_alloc import TPAllocation
from ..util import profile_opt
from ..util.tensor import g_tensor_cache

TEMP_ROWS_FUSED = 128
TEMP_ROWS_GRAPH = 32

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

@dataclass
class FusedBuffers:
    temp_state_g: torch.Tensor
    temp_state_u: torch.Tensor
    temp_intermediate_g: torch.Tensor
    temp_intermediate_u: torch.Tensor


def routing_std(bsz, cfg, y, params):
    if bsz == 1:
        torch.matmul(y, cfg.gate_tensor, out = cfg.router_logits_bsz1)
        ext.routing_std(
            cfg.router_logits_bsz1,
            cfg.selected_experts_bsz1,
            cfg.routing_weights_bsz1,
        )
        return cfg.selected_experts_bsz1, cfg.routing_weights_bsz1
    else:
        router_logits = torch.matmul(y, cfg.gate_tensor)
        activate_all_experts = params.get("activate_all_experts")
        if activate_all_experts:
            routing_weights = torch.softmax(router_logits, dim = -1)
            selected_experts = (
                torch.arange(start = 0, end = cfg.num_experts, dtype = torch.long, device = y.device)
                .repeat((bsz, 1))
            )
            return selected_experts, routing_weights
        else:
            routing_weights = torch.empty((bsz, cfg.num_experts_per_tok), dtype = torch.half, device = y.device)
            selected_experts = torch.empty((bsz, cfg.num_experts_per_tok), dtype = torch.long, device = y.device)
            ext.routing_std(
                router_logits,
                selected_experts,
                routing_weights,
            )
        return selected_experts, routing_weights


# TODO: Optimize top_k groups (for DS3)
def routing_ds3(bsz, cfg, y, params):
    activate_all_experts = params.get("activate_all_experts")
    router_logits = torch.matmul(y, cfg.gate_tensor)

    scores = router_logits.sigmoid()
    scores_for_choice = scores.view(-1, cfg.num_experts)
    if cfg.e_score_correction_bias is not None:
        scores_for_choice = scores_for_choice + cfg.e_score_correction_bias.unsqueeze(0)

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

    if bsz == 1:
        torch.matmul(y, cfg.gate_tensor, out = cfg.router_logits_bsz1)
        ext.routing_ds3_nogroup(
            cfg.router_logits_bsz1,
            cfg.e_score_correction_bias,
            cfg.selected_experts_bsz1,
            cfg.routing_weights_bsz1,
            cfg.routed_scaling_factor
        )
        return cfg.selected_experts_bsz1, cfg.routing_weights_bsz1

    else:
        router_logits = torch.matmul(y, cfg.gate_tensor)
        activate_all_experts = params.get("activate_all_experts")
        if activate_all_experts:
            routing_weights = router_logits.sigmoid()
            if cfg.e_score_correction_bias is not None:
                routing_weights += cfg.e_score_correction_bias.unsqueeze(0)
            factor = cfg.routed_scaling_factor / (routing_weights.sum(dim = -1, keepdim = True) + 1e-20)
            routing_weights *= factor
            selected_experts = (
                torch.arange(start = 0, end = cfg.num_experts, dtype = torch.long, device = y.device)
                .repeat((bsz, 1))
            )
        else:
            routing_weights = torch.empty((bsz, cfg.num_experts_per_tok), dtype = torch.half, device = y.device)
            selected_experts = torch.empty((bsz, cfg.num_experts_per_tok), dtype = torch.long, device = y.device)
            ext.routing_ds3_nogroup(
                router_logits,
                cfg.e_score_correction_bias,
                selected_experts,
                routing_weights,
                cfg.routed_scaling_factor
            )
        return selected_experts, routing_weights


@dataclass
class ExpertsCFG:
    yh: torch.Tensor
    interm_g: torch.Tensor
    interm_u: torch.Tensor
    interm_a: torch.Tensor
    out_d: torch.Tensor
    out_d2: torch.Tensor
    min_expert: int
    max_expert: int


class BlockSparseMLP(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        num_local_experts: int | None = None,
        key_up: str | None = None,
        key_gate: str | None = None,
        key_down: str | None = None,
        key_gate_split: str | None = None,
        key_up_split: str | None = None,
        key_gate_up_split: str | None = None,
        key_down_split: str | None = None,
        key_routing_gate: str | None = None,
        key_shared_gate: str | None = None,
        key_e_score_bias: str | None = "gate.e_score_correction_bias",
        qmap: str | None = None,
        out_dtype: torch.dtype = None,
        activation_fn: str = "silu",
        act_limit: float = 0.0,
        interm_dtype: torch.dtype = None,
        router_type: str = "std",
        routing_gate: Linear | None = None,
        shared_gate: Linear | None = None,
        routed_scaling_factor: float | None = None,
        n_group: int | None = None,
        topk_group: int | None = None,
        shared_experts: MLP | GatedMLP | None = None,
        gates: list[Linear | Module] = None,
        ups: list[Linear | Module] = None,
        downs: list[Linear | Module] = None,
        routing_first: int | None = None,
        routing_last: int | None = None,
        routing_device: int | None = None,
        transposed_load: bool = True,
        transpose_fused_weights: bool = True,
    ):
        super().__init__(config, key, None)

        self.interm_dtype = interm_dtype
        self.activation_fn = activation_fn
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.f_threshold = min(self.num_experts // self.num_experts_per_tok, 32)
        self.num_local_experts = num_local_experts if num_local_experts is not None else num_experts
        self.hidden_size = hidden_size
        self.router_type = router_type
        self.act_limit = act_limit

        self.routing_first = routing_first
        self.routing_last = routing_last
        self.routing_device = routing_device

        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group

        assert out_dtype in (torch.float, None), \
            f"BlockSparseMLP output dtype must be float"

        assert shared_experts is None or shared_experts.out_dtype in (torch.float, None), \
            f"Shared experts output dtype must be float"

        assert num_experts_per_tok <= TEMP_ROWS_GRAPH, \
            f"Too many experts per token, max supported is {TEMP_ROWS_GRAPH}"

        if routing_gate is None and key_routing_gate is None:
            self.routing_gate = None
        elif routing_gate is None:
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
        else:
            self.routing_gate = routing_gate
            self.register_submodule(self.routing_gate)

        if shared_gate is None and key_shared_gate is None:
            self.shared_gate = None
        elif shared_gate is None:
            self.shared_gate = Linear(
                config = config,
                key = f"{key}.{key_shared_gate}",
                in_features = hidden_size,
                out_features = 1,
                qmap = None,
                out_dtype = torch.float,
                pad_to = 1,
            )
            self.register_submodule(self.shared_gate)
        else:
            self.shared_gate = shared_gate
            self.register_submodule(self.shared_gate)

        if gates is not None:
            assert ups is not None and len(ups) == len(gates)
            assert downs is not None and len(downs) == len(gates)
            self.num_slices = len(gates)
            self.gates = gates
            self.ups = ups
            self.downs = downs

        else:
            self.gates = []
            self.ups = []
            self.downs = []

            for idx in range(self.num_local_experts):

                fkey_gate, fkey_up, fkey_down = (
                    f"{key}.{key_gate_up_split}" if key_gate_up_split else
                    f"{key}.{key_gate_split}" if key_gate_split else
                    None,
                    f"{key}.{key_gate_up_split}" if key_gate_up_split else
                    f"{key}.{key_up_split}" if key_up_split else
                    None,
                    f"{key}.{key_down_split}" if key_down_split else
                    None
                )

                gate = Linear(
                    config = config,
                    key = f"{key}.{key_gate}".replace("{expert_idx}", str(idx)),
                    fkey = fkey_gate,
                    fidx = idx,
                    frange = (0, intermediate_size) if key_gate_up_split else None,
                    in_features = hidden_size,
                    out_features = intermediate_size,
                    qmap = qmap + ".input",
                    out_dtype = self.interm_dtype,
                    transposed_load = transposed_load,
                    transpose_fused_weights = transpose_fused_weights,
                    qgroup = key + ".block_gud",
                )
                up = Linear(
                    config = config,
                    key = f"{key}.{key_up}".replace("{expert_idx}", str(idx)),
                    fkey = fkey_up,
                    fidx = idx,
                    frange = (intermediate_size, intermediate_size * 2) if key_gate_up_split else None,
                    in_features = hidden_size,
                    out_features = intermediate_size,
                    qmap = qmap + ".input",
                    out_dtype = self.interm_dtype,
                    transposed_load = transposed_load,
                    transpose_fused_weights = transpose_fused_weights,
                    qgroup = key + ".block_gud",
                )
                down = Linear(
                    config = config,
                    key = f"{key}.{key_down}".replace("{expert_idx}", str(idx)),
                    fkey = fkey_down,
                    fidx = idx,
                    in_features = intermediate_size,
                    out_features = hidden_size,
                    qmap = qmap + f".{idx}.down",
                    out_dtype = torch.float,
                    allow_input_padding = True,
                    transposed_load = transposed_load,
                    transpose_fused_weights = transpose_fused_weights,
                    qgroup = key + ".block_gud",
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
        self.support_fused = False
        self.multi_gate = None
        self.multi_up = None
        self.multi_down = None

        self.routing_cfg = None
        self.experts_cfg = None

        self.e_score_correction_bias = None
        self.e_score_correction_bias_key = key_e_score_bias

        self.shared_experts = shared_experts
        if shared_experts is not None:
            self.register_submodule(shared_experts)

        match router_type:
            case "std": self.routing_fn = routing_std
            case "ds3": self.routing_fn = routing_ds3
            case "dots": self.routing_fn = routing_dots
            case _: raise ValueError(f"Unknown router type {router_type}")

        self.tp_reduce = False

        self.bc = None
        self.bc_sh_exp = False
        self.fused_mode_buffers = None


    @override
    def optimizer_targets(self):
        g, u, d = [], [], []
        for m in self.gates: g += m.optimizer_targets()
        for m in self.ups: u += m.optimizer_targets()
        for m in self.downs: d += m.optimizer_targets()
        if self.shared_experts:
            s = self.shared_experts.optimizer_targets()
            return [s, [g + u, d]]
        else:
            return [[g + u, d]]


    def load_local(self, **kwargs):

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

            # Enable fully fused kernel if possible
            self.support_fused = ((True, False) == self.multi_gate.q_cb() == self.multi_up.q_cb() == self.multi_down.q_cb())

        # Temp buffers for graph, dq and fused-bsz1 paths
        numex = self.num_experts_per_tok
        H = self.hidden_size
        I = self.intermediate_size
        device = self.device

        temp_hidden = g_tensor_cache.get(device, (TEMP_ROWS_GRAPH * 2, H), torch.half, "moe1_temp_hidden")
        temp_interm = g_tensor_cache.get(device, (TEMP_ROWS_GRAPH * 2, I), self.interm_dtype, "moe1_temp_interm")
        temp_activa = g_tensor_cache.get(device, (TEMP_ROWS_GRAPH, I), torch.half, "moe1_temp_activa")
        temp_output = g_tensor_cache.get(device, (TEMP_ROWS_GRAPH, H), torch.float, "moe1_temp_output")

        yh = temp_hidden[:numex].view(numex, 1, H)
        interm_g = temp_interm[:numex].view(numex, 1, I)
        interm_u = temp_interm[numex:numex*2].view(numex, 1, I)
        interm_a = temp_activa[:numex].view(numex, 1, I)
        yh2 = temp_hidden
        interm_gu = temp_interm
        interm_a2 = temp_activa
        out_d = temp_output[:numex].view(numex, 1, H)
        out_d2 = temp_output

        # Expert interval for split module (-1, -1) indicate no split
        mine, maxe = self.routing_first, self.routing_last
        if mine is None or maxe - mine == self.num_experts:
            mine, maxe = -1, -1

        cfg = ExpertsCFG(
            yh = yh,
            interm_g = interm_g,
            interm_u = interm_u,
            interm_a = interm_a,
            out_d = out_d,
            out_d2 = out_d2,
            min_expert = mine,
            max_expert = maxe,
        )
        self.experts_cfg = cfg

        if self.is_quantized:

            # Embed bound classes for shared experts and shared gate
            sh_exp_bc = None
            sh_exp_t = None
            sh_gate_bc = None
            sh_gate_t = None
            self.bc_sh_exp = False
            if self.shared_experts and isinstance(self.shared_experts, GatedMLP) and self.shared_experts.bc is not None:
                self.bc_sh_exp = True
                sh_exp_bc = self.shared_experts.bc
                sh_exp_t = torch.empty((1, 1, H), dtype = torch.float, device = self.device)
                if self.shared_gate:
                    assert self.shared_gate.quant_type == "fp16"
                    sh_gate_bc = self.shared_gate.inner.bc
                    sh_gate_t = torch.empty((1, 1, 1), dtype = self.shared_gate.out_dtype, device = self.device)

            # Pointer lists for fused modes
            g_trellis_ptr = torch.tensor([l.inner.trellis.data_ptr() for l in self.gates])
            u_trellis_ptr = torch.tensor([l.inner.trellis.data_ptr() for l in self.ups])
            g_suh_ptr = torch.tensor([l.inner.suh.data_ptr() for l in self.gates])
            u_suh_ptr = torch.tensor([l.inner.suh.data_ptr() for l in self.ups])
            g_svh_ptr = torch.tensor([l.inner.svh.data_ptr() for l in self.gates])
            u_svh_ptr = torch.tensor([l.inner.svh.data_ptr() for l in self.ups])
            gu_trellis_ptr = torch.stack((g_trellis_ptr, u_trellis_ptr), dim = 0).T.contiguous().to(self.device)
            gu_suh_ptr = torch.stack((g_suh_ptr, u_suh_ptr), dim = 0).T.contiguous().to(self.device)
            gu_svh_ptr = torch.stack((g_svh_ptr, u_svh_ptr), dim = 0).T.contiguous().to(self.device)

            dq_temp_up = g_tensor_cache.get(device, (H, I), torch.half, "dq_temp")
            dq_temp_down = dq_temp_up.view(I, H)

            # Bound class for graph, dq and fused-bsz1 paths
            self.bc = ext.BC_BlockSparseMLP(
                yh2,
                cfg.yh,
                interm_gu,
                cfg.interm_g,
                cfg.interm_u,
                cfg.interm_a,
                interm_a2,
                cfg.out_d,
                cfg.out_d2,
                sh_exp_t,
                sh_gate_t,
                dq_temp_up,
                dq_temp_down,
                cfg.min_expert,
                cfg.max_expert,
                self.multi_gate.ptrs_trellis,
                self.multi_gate.ptrs_suh,
                self.multi_gate.ptrs_svh,
                self.multi_gate.K,
                self.multi_gate.mcg,
                self.multi_gate.mul1,
                self.multi_up.ptrs_trellis,
                self.multi_up.ptrs_suh,
                self.multi_up.ptrs_svh,
                self.multi_up.K,
                self.multi_up.mcg,
                self.multi_up.mul1,
                self.multi_down.ptrs_trellis,
                self.multi_down.ptrs_suh,
                self.multi_down.ptrs_svh,
                self.multi_down.K,
                self.multi_down.mcg,
                self.multi_down.mul1,
                self.activation_fn == "silu",
                self.activation_fn == "gelu",
                sh_exp_bc,
                sh_gate_bc,
                self.act_limit,
                [x.inner.bc for x in self.gates],
                [x.inner.bc for x in self.ups],
                [x.inner.bc for x in self.downs],
                gu_trellis_ptr,
                gu_suh_ptr,
                gu_svh_ptr
            )

            # Larger buffers for fused path, if supported
            if self.support_fused:
                C = ext.exl3_moe_max_concurrency(torch.device(device).index)
                self.fused_mode_buffers = FusedBuffers(
                    temp_state_g = g_tensor_cache.get(device, (C, TEMP_ROWS_FUSED, H), torch.half, "moe2_temp_state_g"),
                    temp_state_u = g_tensor_cache.get(device, (C, TEMP_ROWS_FUSED, H), torch.half, "moe2_temp_state_u"),
                    temp_intermediate_g = g_tensor_cache.get(device, (C, TEMP_ROWS_FUSED, I), torch.half, "moe2_temp_intermediate_g"),
                    temp_intermediate_u = g_tensor_cache.get(device, (C, TEMP_ROWS_FUSED, I), torch.half, "moe2_temp_intermediate_u"),
                )
                self.f_threshold = min(self.num_experts // self.num_experts_per_tok, 4)



    def load_routing(self, **kwargs):

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


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)

        self.e_score_correction_bias = self.config.stc.get_tensor(
            f"{self.key}.{self.e_score_correction_bias_key}",
            self.device,
            optional = True,
            float2half = True,
        )
        if device is not None and torch.device(device).type == "cuda":
            self.load_local(**kwargs)
            self.load_routing(**kwargs)


    @override
    def unload(self):
        self.bc = None
        self.fused_mode_buffers = None
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
        bc_sh_exp = False

        # Routing
        if self.routing_gate is not None:
            selected_experts, routing_weights = self.routing_fn(bsz, self.routing_cfg, y, params)
        else:
            selected_experts = torch.empty((bsz, self.num_experts_per_tok), dtype = torch.long, device = self.device)
            routing_weights = torch.empty((bsz, self.num_experts_per_tok), dtype = torch.half, device = self.device)

        # Broadcast routing indices and weights
        if self.routing_device is not None:
            params["backend"].broadcast(selected_experts, src_device = self.routing_device)
            params["backend"].broadcast(routing_weights, src_device = self.routing_device)

        # Empty slice
        if self.intermediate_size == 0 or self.num_local_experts == 0:
            final_hidden_states = torch.zeros_like(x, dtype = torch.float)

        # Torch/C++/fused path
        elif bsz >= self.f_threshold or not self.is_quantized:
            final_hidden_states = torch.zeros_like(y, dtype = torch.float)

            # if self.routing_device is None or self.num_local_experts == self.num_experts:
            #     expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes = self.num_local_experts)
            # else:
            #     selected_experts -= self.routing_first
            #     invalid = (selected_experts < 0) | (selected_experts >= self.num_local_experts)
            #     shifted = torch.where(invalid, torch.zeros_like(selected_experts), selected_experts + 1)
            #     expert_mask = F.one_hot(shifted, num_classes = self.num_local_experts + 1)[..., 1:]

            if self.num_local_experts is None or self.num_local_experts > 0:

                num_ex = self.num_local_experts or self.num_experts

                num_tokens, top_k = selected_experts.shape
                E = self.num_local_experts

                # Flatten assignments
                flat_expert_global = selected_experts.reshape(-1)               # [num_tokens * top_k]
                flat_weight = routing_weights.reshape(-1)                       # [num_tokens * top_k]

                # Token indices corresponding to each flattened assignment
                flat_token = torch.arange(num_tokens, device = y.device)
                flat_token = flat_token.repeat_interleave(top_k)  # [num_tokens * top_k]

                if self.routing_device is None or self.num_local_experts == self.num_experts:
                    flat_expert_local = flat_expert_global
                else:
                    flat_expert_local = flat_expert_global - self.routing_first
                    valid = (flat_expert_local >= 0) & (flat_expert_local < E)
                    flat_expert_local = torch.where(valid, flat_expert_local, torch.full_like(flat_expert_local, E))

                # Group once by local expert id (including sentinel for expert-P mode)
                order = flat_expert_local.argsort()
                token_sorted = flat_token[order]
                weight_sorted = flat_weight[order]

                # Count how many assignments per expert
                expert_count = torch.bincount(flat_expert_local, minlength = E + 1)
                expert_ptr = torch.empty(E + 2, device = y.device, dtype = torch.long)
                expert_ptr[0] = 0
                expert_ptr[1:] = expert_count.cumsum(0)
                expert_ptr = expert_ptr.tolist()

                # Run fused path if possible, skips experts with more than TEMP_ROWS_FUSED tokens
                if self.fused_mode_buffers is not None:
                    ext.exl3_moe(
                        y,
                        final_hidden_states,
                        expert_count,
                        token_sorted,
                        weight_sorted,
                        self.fused_mode_buffers.temp_state_g,
                        self.fused_mode_buffers.temp_state_u,
                        self.fused_mode_buffers.temp_intermediate_g,
                        self.fused_mode_buffers.temp_intermediate_u,
                        0,  # SiLU
                        self.multi_gate.K,
                        self.multi_up.K,
                        self.multi_down.K,
                        self.multi_gate.ptrs_trellis,
                        self.multi_gate.ptrs_suh,
                        self.multi_gate.ptrs_svh,
                        self.multi_up.ptrs_trellis,
                        self.multi_up.ptrs_suh,
                        self.multi_up.ptrs_svh,
                        self.multi_down.ptrs_trellis,
                        self.multi_down.ptrs_suh,
                        self.multi_down.ptrs_svh,
                        self.multi_gate.mcg,
                        self.multi_gate.mul1,
                        self.multi_up.mcg,
                        self.multi_up.mul1,
                        self.multi_down.mcg,
                        self.multi_down.mul1,
                        self.act_limit
                    )
                    min_rows = TEMP_ROWS_FUSED
                else:
                    min_rows = 0

                out_state = None
                interm = None
                interm_a = None
                max_count = 0

                for expert_idx in range(num_ex):
                    start = expert_ptr[expert_idx]
                    end = expert_ptr[expert_idx + 1]
                    count = end - start
                    if count <= min_rows:
                        continue

                    top_x = token_sorted[start:end]
                    w = weight_sorted[start:end].unsqueeze(1)

                    current_state = y.index_select(0, top_x)

                    if self.bc is not None:
                        # Graph path
                        if count <= TEMP_ROWS_GRAPH:
                            self.bc.run_single_expert(current_state, expert_idx)
                            current_state = self.experts_cfg.out_d2[:count]

                        # DQ path
                        else:
                            if count > max_count:
                                out_state = torch.empty((count, self.hidden_size), dtype = torch.float, device = self.device)
                                interm = torch.empty((count * 2, self.intermediate_size), dtype = self.interm_dtype, device = self.device)
                                interm_a = interm[:count] if self.interm_dtype == torch.half else \
                                    torch.empty_like(interm[:count], dtype = torch.half)
                                out_state_ = out_state
                                interm_ = interm
                                interm_a_ = interm_a
                                max_count = count
                            elif count == max_count:
                                out_state_ = out_state
                                interm_ = interm
                                interm_a_ = interm_a
                            else:
                                out_state_ = out_state[:count]
                                interm_ = interm[:count * 2]
                                interm_a_ = interm_a[:count]

                            yh = torch.empty((count * 2, self.hidden_size), dtype = torch.half, device = self.device)
                            self.bc.run_single_expert_dq(current_state, expert_idx, yh, interm_, interm_a_, out_state)
                            current_state = out_state_
                    else:

                        # Torch path
                        def mlp(exp_i, xc):
                            g = self.gates[exp_i].forward(xc, params)
                            u = self.ups[exp_i].forward(xc, params)
                            a = u if self.interm_dtype == torch.half else torch.empty_like(u, dtype = torch.half)
                            self.activation_fn_call(g, u, a, self.act_limit)
                            return self.downs[exp_i].forward(a, params)

                        current_state = mlp(expert_idx, current_state)

                    current_state.mul_(w)
                    final_hidden_states.index_add_(0, top_x, current_state)

            final_hidden_states = final_hidden_states.reshape(x.shape)

        # Fused path, few tokens
        elif bsz > 1:

            final_hidden_states = torch.empty_like(y, dtype = torch.float)

            y = y.unsqueeze(1).unsqueeze(1)
            selected_experts = selected_experts.unsqueeze(1)
            routing_weights = routing_weights.unsqueeze(1)

            cfg = self.experts_cfg

            mine, maxe = self.routing_first, self.routing_last
            if mine is None or maxe - mine == self.num_experts:
                mine, maxe = -1, -1

            for i in range(bsz):

                # Gate
                ext.exl3_mgemm(
                    y[i],
                    self.multi_gate.ptrs_trellis,
                    cfg.interm_g,
                    self.multi_gate.ptrs_suh,
                    cfg.yh,
                    self.multi_gate.ptrs_svh,
                    selected_experts[i],
                    None,
                    self.multi_gate.K,
                    -1,
                    self.multi_gate.mcg,
                    self.multi_gate.mul1,
                    mine,
                    maxe,
                    0
                )

                # Up
                ext.exl3_mgemm(
                    y[i],
                    self.multi_up.ptrs_trellis,
                    cfg.interm_u,
                    self.multi_up.ptrs_suh,
                    cfg.yh,
                    self.multi_up.ptrs_svh,
                    selected_experts[i],
                    None,
                    self.multi_up.K,
                    -1,
                    self.multi_up.mcg,
                    self.multi_up.mul1,
                    mine,
                    maxe,
                    0
                )

                # Activation
                self.activation_fn_call(cfg.interm_g, cfg.interm_u, cfg.interm_a, self.act_limit)

                # Down
                ext.exl3_mgemm(
                    cfg.interm_a,
                    self.multi_down.ptrs_trellis,
                    cfg.out_d,
                    self.multi_down.ptrs_suh,
                    cfg.interm_a,
                    self.multi_down.ptrs_svh,
                    selected_experts[i],
                    routing_weights[i],
                    self.multi_down.K,
                    -1,
                    self.multi_down.mcg,
                    self.multi_down.mul1,
                    mine,
                    maxe,
                    0
                )

                t = cfg.out_d[0]
                final_hidden_states[i:i+1] = t

            final_hidden_states = final_hidden_states.view(x.shape)

        # Bsz 1
        elif self.bc is not None:
            self.bc.run_bsz1(y, selected_experts, routing_weights)
            final_hidden_states = self.experts_cfg.out_d[:1, ...].view(x.shape)
            bc_sh_exp = self.bc_sh_exp

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
                self.multi_gate.mcg,
                self.multi_gate.mul1,
                cfg.min_expert,
                cfg.max_expert,
                0
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
                self.multi_up.mcg,
                self.multi_up.mul1,
                cfg.min_expert,
                cfg.max_expert,
                0
            )

            # Activation
            self.activation_fn_call(cfg.interm_g, cfg.interm_u, cfg.interm_a, self.act_limit)

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
                self.multi_down.mcg,
                self.multi_down.mul1,
                cfg.min_expert,
                cfg.max_expert,
                0
            )

            final_hidden_states = cfg.out_d[:1, ...].view(x.shape)

        # Shared experts
        if self.shared_experts and not bc_sh_exp:
            y = self.shared_experts.forward(x, params)
            if self.shared_gate:
                if bsz > 32:
                    z = self.shared_gate.forward(x, params)
                    ext.add_sigmoid_gate(y, z, final_hidden_states)
                else:
                    ext.add_sigmoid_gate_proj(y, x, final_hidden_states, self.shared_gate.inner.weight)
            else:
                final_hidden_states += y

        # Output reduction
        if self.tp_reduce:
            params["backend"].all_reduce(
                final_hidden_states,
                (self.intermediate_size > 0 and self.num_local_experts > 0) or bool(self.shared_experts)
            )

        if out_dtype is not None:
            final_hidden_states = final_hidden_states.to(out_dtype)
        return final_hidden_states


    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.e_score_correction_bias is not None:
            t[f"{self.key}.gate.e_score_correction_bias"] = self.e_score_correction_bias.contiguous()
        return t


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        storage = 0
        storage += self.routing_gate.storage_size()
        for g in self.gates: storage += g.storage_size()
        for u in self.ups: storage += u.storage_size()
        for d in self.downs: storage += d.storage_size()
        # TODO: More precise overhead estimate accounting for gate etc.
        overhead_d = self.hidden_size * torch.float.itemsize
        overhead_s = 4 * self.intermediate_size * (self.interm_dtype or torch.half).itemsize
        if self.interm_dtype != torch.half:
            overhead_s += self.intermediate_size * torch.half.itemsize
        recons = max(
            self.gates[0].recons_size(),
            self.ups[0].recons_size(),
            self.downs[0].recons_size()
        )
        use_tp_split = options.get("moe_tensor_split", False)
        tpa = TPAllocation(
            key = self.key,
            channel_width = 128 if use_tp_split else 1,
            channel_unit = "channels" if use_tp_split else "experts",
            storage_per_device = 0,
            storage_to_split = storage,
            overhead_per_device = overhead_d,
            overhead_to_split = overhead_s,
            recons_temp = recons,
            channels_to_split = self.intermediate_size // 128 if use_tp_split else self.num_experts,
            limit_key = "moe"
        )
        tpa_list = [tpa]
        if self.shared_experts:
            tpa_list += self.shared_experts.make_tp_allocation(options)
        return tpa_list


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": BlockSparseMLP,
            "kwargs": {
                "key": self.key,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "activation_fn": self.activation_fn,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "interm_dtype": self.interm_dtype,
                "router_type": self.router_type,
                "routed_scaling_factor": self.routed_scaling_factor,
                "n_group": self.n_group,
                "topk_group": self.topk_group,
                "act_limit": self.act_limit,
            },
            "routing_gate": _export(self.routing_gate),
            "e_score_correction_bias": producer.send(self.e_score_correction_bias),
            "gates": [_export(self.gates[i]) for i in range(self.num_experts)],
            "ups": [_export(self.ups[i]) for i in range(self.num_experts)],
            "downs": [_export(self.downs[i]) for i in range(self.num_experts)],
            "shared_experts": self.shared_experts.tp_export(plan, producer) \
                if self.shared_experts is not None else None,
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        consumer = local_context["consumer"]
        key = exported["kwargs"]["key"]
        device = local_context["device"]
        output_device = local_context["output_device"]
        first, last, unit = plan[key]

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        def _import_no_reduce(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan, skip_reduction = True) \
                if exported.get(name) else None

        def _import_i(name, i):
            nonlocal exported, plan
            return exported[name][i]["cls"].tp_import(local_context, exported[name][i], plan) \
                if exported.get(name) else None

        def _import_i_split(name, i, split):
            nonlocal exported, plan
            return exported[name][i]["cls"].tp_import_split(local_context, exported[name][i], plan, split) \
                if exported.get(name) else None

        # Tensor parallel
        if unit == "channels":
            num_local_experts = exported["kwargs"]["num_experts"]
            gu_split = (True, first, last)
            d_split = (False, first, last)
            exported["kwargs"]["intermediate_size"] = last - first
            gates = [_import_i_split("gates", i, gu_split) for i in range(num_local_experts)]
            ups = [_import_i_split("ups", i, gu_split) for i in range(num_local_experts)]
            downs = [_import_i_split("downs", i, d_split) for i in range(num_local_experts)]
            routing_first = 0
            routing_last = num_local_experts

        # Expert parallel
        elif unit == "experts":
            num_local_experts = last - first
            gates = [_import_i("gates", i) for i in range(first, last)]
            ups = [_import_i("ups", i) for i in range(first, last)]
            downs = [_import_i("downs", i) for i in range(first, last)]
            routing_first = first
            routing_last = last

        else:
            assert False

        module = BlockSparseMLP(
            config = None,
            **exported["kwargs"],
            num_local_experts = num_local_experts,
            gates = gates,
            ups = ups,
            downs = downs,
            shared_experts = _import_no_reduce("shared_experts"),
            routing_gate = _import("routing_gate") if device == output_device else None,
            routing_first = routing_first,
            routing_last = routing_last,
            routing_device = output_device,
        )

        module.device = device
        module.e_score_correction_bias = consumer.recv(exported["e_score_correction_bias"], cuda = True)
        if unit == "channels" or num_local_experts > 0:
            module.load_local()
        if module.routing_gate is not None:
            module.load_routing()
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True
        return module