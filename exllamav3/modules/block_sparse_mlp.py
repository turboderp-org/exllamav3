from __future__ import annotations
from typing_extensions import override
import os
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..util.tensor import to2
from . import Module, Linear
from .multilinear import MultiLinear
from ..ext import exllamav3_ext as ext
from dataclasses import dataclass
from .mlp import MLP, GatedMLP
from .rmsnorm import RMSNorm
from .layernorm import LayerNorm
from .block_sparse_mlp_cpu import BlockSparseMLP_CPU
from .moe_batch_recon import PAD_MAX
from ..model.model_tp_alloc import TPAllocation
from ..util import profile_opt
from ..util.tensor import g_tensor_cache, buffered_interleaved_arange
from .block_sparse_mlp_routing import (
    RoutingCFG, ROUTING_ACT_SIGMOID, ROUTING_ACT_SQRTSP,
    routing_std, routing_std_bias, routing_ds3, routing_dots, routing_sqrtsp, routing_sqrtsp_hash,
)

# Row capacity of the fused MoE kernel's per-group temp buffers (experts with more assigned
# rows take the reconstruct paths); EXL3_MOE_FUSED_ROWS overrides for tuning sweeps
TEMP_ROWS_FUSED = int(os.environ.get("EXL3_MOE_FUSED_ROWS", 128))
TEMP_ROWS_GRAPH = 32
# Batched reconstruct tier for the experts above the fused kernel's row capacity at prefill
# (moe_batch_recon.py); EXL3_MOE_BATCH_RECON=0 restores the per-expert reconstruct loop
BATCH_RECON = os.environ.get("EXL3_MOE_BATCH_RECON", "1") != "0"
# Row tiles for the fused kernel: experts with more than MTILE_T1 rows run through a 32-row tile
# instance, more than MTILE_T2 through a 64-row one (32 for the N = 256 shape), each its own
# launch over its expert range (mul1 codebook only). EXL3_MOE_MTILE=0 keeps the single 16-row
# launch
MTILE = os.environ.get("EXL3_MOE_MTILE", "1") != "0"
MTILE_T1, MTILE_T2 = 16, 32
# Fused-kernel row capacity per expert when the wide tiles apply: with them the fused kernel
# beats the batched reconstruct tier up to 256 rows (Qwen3.8 4k chunk: 128 -> 256 rows +3%)
FUSED_ROWS_WIDE = int(os.environ.get("EXL3_MOE_FUSED_ROWS_WIDE", 256))
# Deterministic (slot + gather) accumulation for the fused kernel's outputs; EXL3_MOE_FUSED_DET=0
# restores the atomic adds
FUSED_DET = os.environ.get("EXL3_MOE_FUSED_DET", "1") != "0"
MAX_BSZN = 8  # must match MAX_BSZN in exllamav3_ext/libtorch/blocksparse_mlp.h

@dataclass
class FusedBuffers:
    temp_state_g: torch.Tensor
    temp_state_u: torch.Tensor
    temp_intermediate_g: torch.Tensor
    temp_intermediate_u: torch.Tensor



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
    out_trim: torch.Tensor | None = None


class BlockSparseMLP(BlockSparseMLP_CPU, Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        num_local_experts: int | None = None,
        latent_size: int | None = None,
        key_latent_in: str | None = None,
        key_latent_out: str | None = None,
        latent_in: Linear | None = None,
        latent_out: Linear | None = None,
        latent_hq_bits: int = 0,
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
        key_e_score_bias_vl: str | None = None,
        key_tid2eid: str | None = None,
        key_per_expert_scale: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype = None,
        activation_fn: str = "silu",
        act_limit: float = 0.0,
        interm_dtype: torch.dtype = None,
        interm_div: float = 1.0,
        router_type: str = "std",
        routing_gate: Linear | None = None,
        shared_gate: Linear | None = None,
        routed_scaling_factor: float | None = None,
        n_group: int | None = None,
        topk_group: int | None = None,
        shared_experts: MLP | GatedMLP | None = None,
        shared_experts_post_norm: RMSNorm | LayerNorm | None = None,
        router_pre_norm: RMSNorm | LayerNorm | None = None,
        routed_pre_norm: RMSNorm | LayerNorm | None = None,
        routed_post_norm: RMSNorm | LayerNorm | None = None,
        gates: list[Linear | Module] = None,
        ups: list[Linear | Module] = None,
        downs: list[Linear | Module] = None,
        routing_first: int | None = None,
        routing_last: int | None = None,
        routing_device: int | None = None,
        transposed_load: bool = True,
        transpose_fused_weights: bool = True,
        ftranspose_after_load: bool = True,
        frange_dim: int = 0,
        gate_up_interleaved: bool = False,
        alt_residual_channel: bool = False,
        qbits_key: str = "bits"
    ):
        super().__init__(config, key, None)

        self.interm_dtype = interm_dtype
        self.interm_div = interm_div
        self.router_type = router_type
        if interm_div != 1.0:
            assert router_type in ("dots", "ds3", "std"), \
                "interm_div requires a router type that can fold the compensation into the routing weights"
            if router_type != "std":
                routed_scaling_factor = (routed_scaling_factor if routed_scaling_factor is not None else 1.0) * interm_div
        self.activation_fn = activation_fn
        self.intermediate_size = intermediate_size
        self.intermediate_size_padded = (intermediate_size + 127) // 128 * 128
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.f_threshold = min(self.num_experts // self.num_experts_per_tok, 4)
        self.num_local_experts = num_local_experts if num_local_experts is not None else num_experts
        self.hidden_size = hidden_size
        # Latent MoE (Nemotron-3 Super): the routed experts run at latent_size, between a
        # projection of the block input down to that width and a projection of the routed sum
        # back up. The router and the shared experts see the full-width block input
        self.latent_size = latent_size
        self.expert_size = latent_size or hidden_size
        self.router_type = router_type
        self.act_limit = act_limit
        self.alt_residual_channel = alt_residual_channel

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

        if latent_in is None and key_latent_in is None:
            assert latent_size is None, "latent_size requires the latent projection keys"
            self.latent_in = None
            self.latent_out = None
        else:
            assert latent_size, "latent projections require latent_size"
            # The down projection shares the block input (and its Hessian) with the shared
            # experts; the up projection's input is the routed sum
            self.latent_in = latent_in if latent_in is not None else Linear(
                config = config,
                key = f"{key}.{key_latent_in}",
                in_features = hidden_size,
                out_features = latent_size,
                qmap = qmap + ".input" if qmap else None,
                out_dtype = torch.half,
                select_hq_bits = latent_hq_bits,
                qgroup = key + ".latent",
                qbits_key = qbits_key,
            )
            self.latent_out = latent_out if latent_out is not None else Linear(
                config = config,
                key = f"{key}.{key_latent_out}",
                in_features = latent_size,
                out_features = hidden_size,
                qmap = qmap + ".latent_out" if qmap else None,
                out_dtype = torch.float,
                select_hq_bits = latent_hq_bits,
                qgroup = key + ".latent",
                qbits_key = qbits_key,
            )
            self.register_submodule(self.latent_in)
            self.register_submodule(self.latent_out)

        # Non-gated experts (NemotronH: up/down with relu2). Gateless relu2 rides every quantized
        # fast path (relu(u) * u = relu2(u): the BC graphs and mgemm loops skip the gate GEMM, the
        # fused kernel synthesizes the gate lane, the batched reconstruct tier and the CPU worker
        # take up/down only); other gateless activations run the dense per-expert path
        self.gated = (
            key_gate is not None or key_gate_split is not None or key_gate_up_split is not None or
            (gates is not None and len(gates) > 0)
        )

        if gates is not None:
            assert ups is not None and (not self.gated or len(ups) == len(gates))
            assert downs is not None and len(downs) == len(ups)
            self.num_slices = len(ups)
            self.gates = gates
            self.ups = ups
            self.downs = downs

        else:
            self.gates = []
            self.ups = []
            self.downs = []

            # The experts' input is the block input, or the latent projection of it (a
            # different activation, so a different Hessian) in a latent MoE
            expert_qmap = (qmap + (".latent" if latent_size else ".input")) if qmap else None
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

                gate = None if not self.gated else Linear(
                    config = config,
                    key = f"{key}.{key_gate}".replace("{expert_idx}", str(idx)),
                    fkey = fkey_gate,
                    fidx = idx,
                    frange = (0, intermediate_size) if key_gate_up_split else None,
                    finterleaved = gate_up_interleaved,
                    in_features = self.expert_size,
                    out_features = intermediate_size,
                    qmap = expert_qmap,
                    out_dtype = self.interm_dtype,
                    transposed_load = transposed_load,
                    transpose_fused_weights = transpose_fused_weights,
                    ftranspose_after_load = ftranspose_after_load,
                    frange_dim = frange_dim,
                    qgroup = key + ".block_gud",
                    qbits_key = qbits_key,
                )
                up = Linear(
                    config = config,
                    key = f"{key}.{key_up}".replace("{expert_idx}", str(idx)),
                    fkey = fkey_up,
                    fidx = idx,
                    frange = (intermediate_size, intermediate_size * 2) if key_gate_up_split else None,
                    finterleaved = gate_up_interleaved,
                    in_features = self.expert_size,
                    out_features = intermediate_size,
                    qmap = expert_qmap,
                    out_dtype = self.interm_dtype,
                    transposed_load = transposed_load,
                    transpose_fused_weights = transpose_fused_weights,
                    ftranspose_after_load = ftranspose_after_load,
                    frange_dim = frange_dim,
                    qgroup = key + ".block_gud",
                    qbits_key = qbits_key,
                    weight_scale = 1.0 / interm_div,
                )
                down = Linear(
                    config = config,
                    key = f"{key}.{key_down}".replace("{expert_idx}", str(idx)),
                    fkey = fkey_down,
                    fidx = idx,
                    in_features = intermediate_size,
                    out_features = self.expert_size,
                    qmap = qmap + f".{idx}.down" if qmap else None,
                    out_dtype = torch.float,
                    allow_input_padding = True,
                    transposed_load = transposed_load,
                    transpose_fused_weights = transpose_fused_weights,
                    ftranspose_after_load = ftranspose_after_load,
                    # The input dim pads to match the gate/up padded output width; padded output
                    # columns (zeros, or quantization noise over zero weights) are trimmed
                    trim_padded_out = True,
                    qgroup = key + ".block_gud",
                    qbits_key = qbits_key,
                )

                self.ups.append(up)
                if gate is not None:
                    self.gates.append(gate)
                self.downs.append(down)

                self.register_submodule(up)
                self.register_submodule(gate)
                self.register_submodule(down)

        if self.gated:
            self.gateless_act = None
            match activation_fn:
                case "silu":
                    self.activation_fn_call = ext.silu_mul
                    self.activation_fn_idx = 0
                case "gelu":
                    self.activation_fn_call = ext.gelu_mul
                    self.activation_fn_idx = 1
                case "swiglu_oai":
                    self.activation_fn_call = ext.silu_oai_mul
                    self.activation_fn_idx = 3
                case "relu2":
                    self.activation_fn_call = ext.relu2_mul
                    self.activation_fn_idx = 2
                case _:
                    raise ValueError(f"Unknown activation function {activation_fn}")
        else:
            # Gateless relu2 rides the gated fast paths via relu(u) * u = relu2(u): the act call
            # sites pass (u, u, a) and the fused MoE kernel's MOE_ACT_RELU2_NOGATE synthesizes
            # the gate lane from u
            self.activation_fn_call = ext.relu_mul if activation_fn == "relu2" else None
            self.activation_fn_idx = 2 if activation_fn == "relu2" else -1
            match activation_fn:
                case "silu": self.gateless_act = F.silu
                case "gelu": self.gateless_act = lambda x: F.gelu(x, approximate = "tanh")
                case "relu2": self.gateless_act = lambda x: torch.square(F.relu(x))
                case _:
                    raise ValueError(f"Unknown gateless activation function {activation_fn}")

        self.is_quantized = False
        self.support_fused = False
        self.support_quant_paths = False
        self.multi_gate = None
        self.multi_up = None
        self.multi_down = None

        self.routing_cfg = None
        self.experts_cfg = None

        # Persistent broadcast targets for TP ranks without the router (see forward): the BC bsz-1
        # graph bakes the routing tensors' addresses into unpatched nodes, so they must be statics
        self.bcast_sel_bsz1 = None
        self.bcast_weights_bsz1 = None

        self.e_score_correction_bias = None
        self.e_score_correction_bias_key = key_e_score_bias
        self.e_score_bias_vl = None
        self.e_score_bias_vl_key = key_e_score_bias_vl
        self.tid2eid = None
        self.tid2eid_key = key_tid2eid
        self.per_expert_scale = None
        self.per_expert_scale_key = key_per_expert_scale

        self.shared_experts = shared_experts
        if shared_experts is not None:
            self.register_submodule(shared_experts)

        match router_type:
            case "std": self.routing_fn = routing_std
            case "std_bias": self.routing_fn = routing_std_bias
            case "ds3": self.routing_fn = routing_ds3
            case "dots": self.routing_fn = routing_dots
            case "sqrtsp": self.routing_fn = routing_sqrtsp
            case "sqrtsp_hash": self.routing_fn = routing_sqrtsp_hash
            case _: raise ValueError(f"Unknown router type {router_type}")

        self.tp_reduce = False

        self.shared_experts_post_norm = shared_experts_post_norm
        self.router_pre_norm = router_pre_norm
        self.routed_pre_norm = routed_pre_norm
        self.routed_post_norm = routed_post_norm
        self.register_submodule(self.shared_experts_post_norm)
        self.register_submodule(self.router_pre_norm)
        self.register_submodule(self.routed_pre_norm)
        self.register_submodule(self.routed_post_norm)

        self.bc = None
        self.bc_sh_exp = False
        self.fused_mode_buffers = None
        self.mtile_ok = False
        self.fused_rows = TEMP_ROWS_FUSED
        self.batch_recon = None
        self._cpu_init_state()

    @override
    def optimizer_targets(self):
        g, u, d = [], [], []
        for m in self.gates: g += m.optimizer_targets()
        for m in self.ups: u += m.optimizer_targets()
        for m in self.downs: d += m.optimizer_targets()
        if self.latent_in is not None:
            u = self.latent_in.optimizer_targets() + u
            d = d + self.latent_out.optimizer_targets()
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

        # The quantized fast paths (mgemm/BC/fused kernels) don't yet support per-expert biases,
        # activations other than silu/gelu (or gateless relu2), or trimmed (padded) down
        # projections; configurations with any of those run every batch size through the dense
        # per-expert path, which handles all of them (gpt-oss)
        self.support_quant_paths = (
            self.is_quantized and
            (self.activation_fn in ("silu", "gelu") if self.gated else self.activation_fn == "relu2") and
            all(l.inner.bias is None for l in self.gates + self.ups + self.downs) and
            all(not l.trim_padded_out or l.out_features == l.out_features_unpadded for l in self.downs)
        )

        # The BC bsz-1 graph additionally supports the gpt-oss activation, per-expert biases
        # (all-or-nothing per projection) and padded dims (zero-padded input staging + trimmed
        # output); the raw mgemm and dense BC paths do not, so those configurations run the
        # dense per-expert path for every other batch shape
        def _uniform_bias(ls):
            has = [l.inner.bias is not None for l in ls]
            return all(has) or not any(has)
        self.support_bc_bsz1 = (
            self.is_quantized and
            (self.activation_fn in ("silu", "gelu", "swiglu_oai") if self.gated else self.activation_fn == "relu2") and
            _uniform_bias(self.gates) and _uniform_bias(self.ups) and _uniform_bias(self.downs) and
            self.shared_experts is None and
            not self.config.infer_params.no_reconstruct
        )

        # Make fused modules (only used by the quantized fast paths). Gateless experts have no
        # gate MultiLinear; the up module doubles as a placeholder wherever the fast paths want
        # gate pointer tables (never dereferenced, the gate GEMMs are skipped)
        if (self.support_quant_paths or self.support_bc_bsz1) and not self.config.infer_params.no_reconstruct:
            self.multi_gate = MultiLinear(self.device, self.gates, allow_bias = True) if self.gated else None
            self.multi_up = MultiLinear(self.device, self.ups, allow_bias = True)
            self.multi_down = MultiLinear(self.device, self.downs, allow_bias = True)

            # Enable fully fused kernel if possible (uniform mcg or mul1 codebook across gate/up/down,
            # and an activation the fused kernel implements)
            cbs = (
                self.multi_gate.q_cb() if self.gated else self.multi_up.q_cb(),
                self.multi_up.q_cb(),
                self.multi_down.q_cb(),
            )
            self.support_fused = (
                cbs[0] == cbs[1] == cbs[2] and cbs[0] in ((True, False), (False, True)) and
                self.support_quant_paths
            )

        # Temp buffers for graph, dq and fused-bsz1 paths
        numex = self.num_experts_per_tok
        H = self.expert_size
        # The gate/up input width and the down output width are the (possibly 128-padded)
        # quantized dims; both equal H for aligned models
        Hi = self.ups[0].in_features
        Ho = self.downs[0].out_features
        I = self.intermediate_size_padded
        device = self.device

        # bszn_rows bounds the multi-row graph path (BC_BlockSparseMLP.run_bszN, bsz 1..MAX_BSZN,
        # bszm = num_tokens * numex slots); buffers grow to whichever of that or the single-expert
        # graph loop's TEMP_ROWS_GRAPH requirement is larger, sharing the one cache entry per name
        # (g_tensor_cache is exact-shape-keyed and never evicts -- growing a differently-shaped
        # second entry under the same name would silently double memory forever)
        bszn_rows = MAX_BSZN * numex

        temp_hidden = g_tensor_cache.get(device, (max(TEMP_ROWS_GRAPH * 2, bszn_rows), Hi), torch.half, "moe1_temp_hidden")
        temp_interm = g_tensor_cache.get(device, (max(TEMP_ROWS_GRAPH * 2, 2 * bszn_rows), I), self.interm_dtype, "moe1_temp_interm")
        temp_activa = g_tensor_cache.get(device, (max(TEMP_ROWS_GRAPH, bszn_rows), I), torch.half, "moe1_temp_activa")
        temp_output = g_tensor_cache.get(device, (max(TEMP_ROWS_GRAPH, bszn_rows), Ho), torch.float, "moe1_temp_output")

        yh = temp_hidden[:bszn_rows].view(bszn_rows, 1, Hi)
        interm_g = temp_interm[:bszn_rows].view(bszn_rows, 1, I)
        interm_u = temp_interm[bszn_rows:bszn_rows*2].view(bszn_rows, 1, I)
        interm_a = temp_activa[:bszn_rows].view(bszn_rows, 1, I)
        yh2 = temp_hidden
        interm_gu = temp_interm
        interm_a2 = temp_activa
        out_d = temp_output[:bszn_rows].view(bszn_rows, 1, Ho)
        out_d2 = temp_output

        # Static scratch for the num_tokens > 1 gathered input (BC_BlockSparseMLP.run_bszN); each
        # of num_tokens*numex slots holds a copy of its token's row, built via index_select
        a_gather = g_tensor_cache.get(device, (bszn_rows, Hi), torch.half, "moe1_a_gather")

        # Expert interval for split module (-1, -1) indicate no split
        mine, maxe = self.routing_first, self.routing_last
        if mine is None or maxe - mine == self.num_experts:
            mine, maxe = -1, -1

        # Exact-width output when the down projection is padded (the BC graph copies the trimmed
        # columns out of the padded reduction); sized for up to MAX_BSZN rows
        out_trim = None
        if Ho != H:
            out_trim = g_tensor_cache.get(device, (MAX_BSZN, H), torch.float, "moe1_out_trim")

        cfg = ExpertsCFG(
            yh = yh,
            interm_g = interm_g,
            interm_u = interm_u,
            interm_a = interm_a,
            out_d = out_d,
            out_d2 = out_d2,
            min_expert = mine,
            max_expert = maxe,
            out_trim = out_trim,
        )
        self.experts_cfg = cfg

        if (self.support_quant_paths or self.support_bc_bsz1) \
                and not self.config.infer_params.no_reconstruct:

            # Embed bound classes for shared experts and shared gate
            sh_exp_bc = None
            sh_exp_t = None
            sh_gate_bc = None
            sh_gate_t = None
            self.bc_sh_exp = False
            if (
                self.shared_experts
                and isinstance(self.shared_experts, GatedMLP)
                and self.shared_experts.bc is not None
                and self.shared_experts_post_norm is None   # TODO: embed post_norm in BC
                and not self.alt_residual_channel  # TODO: allow residual channel switching in BC (Gemma4)
                and self.latent_in is None  # the graph would merge the shared output at the latent width
            ):
                self.bc_sh_exp = True
                sh_exp_bc = self.shared_experts.bc
                sh_exp_t = torch.empty((1, MAX_BSZN, self.hidden_size), dtype = torch.float, device = self.device)
                if self.shared_gate:
                    assert self.shared_gate.quant_type == "fp16"
                    sh_gate_bc = self.shared_gate.inner.bc
                    sh_gate_t = torch.empty((1, 1, 1), dtype = self.shared_gate.out_dtype, device = self.device)

            # Pointer lists for fused modes. Gateless experts reuse the up tables as gate
            # placeholders: valid memory for the (uniform) table loads, never dereferenced
            u_trellis_ptr = torch.tensor([l.inner.trellis.data_ptr() for l in self.ups])
            u_suh_ptr = torch.tensor([l.inner.suh.data_ptr() for l in self.ups])
            u_svh_ptr = torch.tensor([l.inner.svh.data_ptr() for l in self.ups])
            if self.gated:
                g_trellis_ptr = torch.tensor([l.inner.trellis.data_ptr() for l in self.gates])
                g_suh_ptr = torch.tensor([l.inner.suh.data_ptr() for l in self.gates])
                g_svh_ptr = torch.tensor([l.inner.svh.data_ptr() for l in self.gates])
            else:
                g_trellis_ptr, g_suh_ptr, g_svh_ptr = u_trellis_ptr, u_suh_ptr, u_svh_ptr
            gu_trellis_ptr = torch.stack((g_trellis_ptr, u_trellis_ptr), dim = 0).T.contiguous().to(self.device)
            gu_suh_ptr = torch.stack((g_suh_ptr, u_suh_ptr), dim = 0).T.contiguous().to(self.device)
            gu_svh_ptr = torch.stack((g_svh_ptr, u_svh_ptr), dim = 0).T.contiguous().to(self.device)

            dq_temp_up = g_tensor_cache.get(device, (Hi, I), torch.half, "dq_temp")
            dq_temp_down = dq_temp_up.view(I, Ho)

            # Per-expert bias pointer tables and padded-input staging for the bsz-1 graph
            def _bias_ptrs(ls):
                if ls[0].inner.bias is None:
                    return None
                return torch.tensor([l.inner.bias.data_ptr() for l in ls],
                                    dtype = torch.long, device = device)
            gate_bias_ptrs = _bias_ptrs(self.gates) if self.gated else None
            up_bias_ptrs = _bias_ptrs(self.ups)
            down_bias_ptrs = _bias_ptrs(self.downs)
            y_pad = None
            if Hi != H:
                y_pad = g_tensor_cache.get(device, (MAX_BSZN, Hi), torch.half, "moe1_y_pad")
                y_pad.zero_()

            # Bound class for graph, dq and fused-bsz1 paths (gateless: the up module stands in
            # for the unused gate pointer args, and the gates list is empty)
            multi_gate = self.multi_gate if self.gated else self.multi_up
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
                multi_gate.ptrs_trellis,
                multi_gate.ptrs_suh,
                multi_gate.ptrs_svh,
                multi_gate.K,
                multi_gate.mcg,
                multi_gate.mul1,
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
                self.activation_fn == "swiglu_oai",
                sh_exp_bc,
                sh_gate_bc,
                self.act_limit,
                [x.inner.bc for x in self.gates],
                [x.inner.bc for x in self.ups],
                [x.inner.bc for x in self.downs],
                gu_trellis_ptr,
                gu_suh_ptr,
                gu_svh_ptr,
                a_gather,
                gate_bias_ptrs,
                up_bias_ptrs,
                down_bias_ptrs,
                y_pad,
                cfg.out_trim,
                act_relu2 = self.activation_fn == "relu2",
            )

            # Larger buffers for fused path, if supported. Wide row tiles (32 / 64 rows, separate
            # N = 128 kernel instances) exist for the mul1 codebook; with them the fused tier's
            # row capacity rises to FUSED_ROWS_WIDE. Dims that are multiples of 256 keep the
            # N = 256 instance for the <= 16-row launch only (it is the faster 16-row tiling)
            if self.support_fused:
                self.mtile_ok = MTILE and bool(self.multi_up.mul1)
                self.fused_rows = FUSED_ROWS_WIDE if self.mtile_ok else TEMP_ROWS_FUSED
                R = self.fused_rows
                C = ext.exl3_moe_max_concurrency(torch.device(device).index)
                self.fused_mode_buffers = FusedBuffers(
                    temp_state_g = g_tensor_cache.get(device, (C, R, H), torch.half, "moe2_temp_state_g"),
                    temp_state_u = g_tensor_cache.get(device, (C, R, H), torch.half, "moe2_temp_state_u"),
                    temp_intermediate_g = g_tensor_cache.get(device, (C, R, I), torch.half, "moe2_temp_intermediate_g"),
                    temp_intermediate_u = g_tensor_cache.get(device, (C, R, I), torch.half, "moe2_temp_intermediate_u"),
                )
                self.f_threshold = min(self.num_experts // self.num_experts_per_tok, 4)


    def load_routing(self, **kwargs):

        if self.interm_div != 1.0 and self.router_type == "std":
            # std routing has no scaling factor; fold the interm_div compensation into the
            # per-expert scale, which routing_std applies after top-k normalization. Both the
            # GPU and CPU-offload load paths come through here, and unload clears the tensor
            if self.per_expert_scale is None:
                self.per_expert_scale = torch.full(
                    (self.num_experts,), self.interm_div, dtype = torch.bfloat16, device = self.device)
            else:
                self.per_expert_scale = (self.per_expert_scale.float() * self.interm_div).to(torch.bfloat16)

        router_logits_bsz1 = torch.empty((1, self.num_experts), dtype = torch.half, device = self.device)
        routing_weights_bsz1 = torch.empty((1, self.num_experts_per_tok), dtype = torch.half, device = self.device)
        selected_experts_bsz1 = torch.empty((1, self.num_experts_per_tok), dtype = torch.long, device = self.device)

        self.routing_cfg = RoutingCFG(
            gate_tensor = self.routing_gate.inner.weight,
            router_bias = getattr(self.routing_gate.inner, "bias", None),
            gate_tensor_t = None,  # created lazily on first bsz-1 call (weights may be deferred here)
            num_experts = self.num_experts,
            num_experts_per_tok = self.num_experts_per_tok,
            router_logits_bsz1 = router_logits_bsz1,
            routing_weights_bsz1 = routing_weights_bsz1,
            selected_experts_bsz1 = selected_experts_bsz1,
            e_score_correction_bias = self.e_score_correction_bias,
            e_score_bias_h = None,
            tid2eid = self.tid2eid,
            e_score_bias_vl = self.e_score_bias_vl,
            routed_scaling_factor = self.routed_scaling_factor,
            n_group = self.n_group,
            topk_group = self.topk_group,
            per_expert_scale = self.per_expert_scale,
        )


    @override
    def load(self, device: torch.Device, **kwargs):
        # CPU expert offload (see block_sparse_mlp_cpu.py): a whole-layer claim replaces the
        # GPU load entirely; a split registration shrinks the module to its GPU slice first
        if self.cpu_maybe_offload_load(device, **kwargs):
            return
        self.cpu_maybe_split_load(device, **kwargs)
        super().load(device, **kwargs)

        if self.e_score_correction_bias_key:
            for k in [self.e_score_correction_bias_key, "gate.e_score_correction_bias"]:
                esb = self.config.stc.get_tensor(
                    f"{self.key}.{k}",
                    self.device,
                    optional = True,
                    allow_bf16 = True,
                    no_defer = True,
                )
                if esb is not None:
                    self.e_score_correction_bias = esb if esb.dtype == torch.half else esb.float()
                    break
        if self.e_score_bias_vl_key:
            esb_vl = self.config.stc.get_tensor(
                f"{self.key}.{self.e_score_bias_vl_key}",
                self.device,
                optional = True,
                allow_bf16 = True,
                no_defer = True,
            )
            self.e_score_bias_vl = esb_vl.float() if esb_vl is not None else None
        if self.tid2eid_key:
            self.tid2eid = self.config.stc.get_tensor(
                f"{self.key}.{self.tid2eid_key}",
                self.device,
                no_defer = True,
            )
        if self.per_expert_scale_key:
            self.per_expert_scale = self.config.stc.get_tensor(
                f"{self.key}.{self.per_expert_scale_key}",
                self.device,
                optional = True,
                allow_bf16 = True,
            )
        if device is not None and torch.device(device).type == "cuda":
            self.cpu_post_load()
            self.load_local(**kwargs)
            self.load_routing(**kwargs)


    def _batch_recon_layer(self, y):
        """Per-module batched reconstruct state (moe_batch_recon.BatchReconLayer), built on
        first use; None when the module's configuration isn't covered (same conditions as the
        DQ path plus unpadded hidden dims; fp32 intermediates as in the DQ path are supported)"""
        if self.batch_recon is False:
            return None
        if self.batch_recon is None:
            ok = (
                BATCH_RECON and self.bc is not None and self.support_quant_paths and
                self.interm_dtype in (torch.half, None, torch.float) and self.multi_up is not None and
                self.multi_up.in_features == y.shape[1] and
                self.multi_down.out_features == y.shape[1] and
                self.multi_down.in_features == self.multi_up.out_features
            )
            if not ok:
                self.batch_recon = False
                return None
            from .moe_batch_recon import BatchReconLayer
            mg, mu, md = self.multi_gate, self.multi_up, self.multi_down
            scales = {p: ([l.inner.suh for l in m.linears], [l.inner.svh for l in m.linears])
                      for p, m in (("g", mg), ("u", mu), ("d", md)) if m is not None}
            self.batch_recon = BatchReconLayer(
                (mg.in_features, mg.out_features, mg.K) if mg is not None else None,
                (mu.in_features, mu.out_features, mu.K),
                (md.in_features, md.out_features, md.K),
                mg.q_cb() if mg is not None else None, mu.q_cb(), md.q_cb(),
                self.activation_fn, self.act_limit, self.device, scales,
                interm_fp32 = self.interm_dtype != torch.half)
            self.batch_recon.set_static_pointers(
                mg.ptrs_trellis if mg is not None else None, mu.ptrs_trellis, md.ptrs_trellis)
        return self.batch_recon

    def prefill_worst_case_parts(self, rows: int, assignments: int) -> tuple[int, int]:
        """Upper bound on the GPU-expert prefill transients of a `rows`-token chunk with
        `assignments` routed rows landing on GPU-resident experts, as (fixed, variable) bytes:
        the fp32 accumulator, the output state and the deterministic slot scratch (every
        assignment slotted, padded) live for the whole call; the larger of one
        batched-reconstruct group and the per-expert dequant path with every row on one expert
        is the working set on top. The autosplit measuring forward routes a dummy state, so
        what it sees of these depends on that routing; the loader takes this bound instead
        (autosplit_extra_measure)"""
        if self.cpu_offload or self.multi_up is None:
            return 0, 0
        h = self.expert_size
        fixed = (rows + 1) * h * 4 + rows * h * 2
        if FUSED_DET:
            fixed += (int(assignments * PAD_MAX) + 1) * h * 4
        r = min(rows, assignments)
        isz = (self.interm_dtype or torch.float).itemsize
        per_expert = r * (2 * self.intermediate_size_padded * isz + 2 * h * 2 + h * 4)
        if self.interm_dtype != torch.half:
            per_expert += r * self.intermediate_size_padded * 2
        recon = self._batch_recon_layer(torch.empty((1, h), dtype = torch.half, device = self.device))
        batched = recon.worst_case_bytes(assignments, slot_mode = FUSED_DET) if recon is not None else 0
        return fixed, max(per_expert, batched)

    def autosplit_extra_measure(self, params):
        """Autosplit loader hook: allocate (and drop) the worst-case prefill transient so the
        device keeps headroom for it. The CPU-offload host's per-device stream state and this
        layer's tier statics are allocated for real here: the measuring forward skips the CPU
        path, and they would otherwise appear unaccounted on the first real prefill"""
        if os.environ.get("EXL3_AUTOSPLIT_WORSTCASE", "1") == "0":
            return
        rows = getattr(self, "_measure_rows", 0)
        if not rows or self.device is None or self.device.type != "cuda":
            return
        A = rows * self.num_experts_per_tok
        host = getattr(self, "cpu_host", None)
        if host is not None and getattr(self, "cpu_layer_idx", None) is not None:
            # Split layers: the assignments divide between GPU-resident and CPU-resident
            # experts. Both sides' whole-call buffers live at once; the CPU side's group
            # working set is freed before the GPU side allocates its own. Take the worst split
            total = 0
            for a in (A * i // 4 for i in range(5)):
                gf, gv = self.prefill_worst_case_parts(rows, a)
                cf, cv = host.prefill_worst_case_parts(self.cpu_layer_idx, rows, self.device, A - a)
                total = max(total, gf + cf + max(gv, cv))
        else:
            total = sum(self.prefill_worst_case_parts(rows, A))
        if total > 0:
            t = torch.empty((total,), dtype = torch.uint8, device = self.device)
            del t

    def _run_batch_recon(self, recon, y, fhs_ext, token_sorted, weight_sorted, expert_count_list, groups,
                         scratch = None, tables = None):
        """Run the planned groups through the batched reconstruct tier. With a slot scratch each
        group's down projection lands in its slots (summed later by exl3_moe_gather); otherwise
        each group accumulates into fhs_ext (rows + 1, the last row a padding sink). Returns the
        set of experts handled."""
        rows = y.shape[0]
        y_ext = torch.cat([y, torch.zeros((1, y.shape[1]), dtype = y.dtype, device = y.device)])
        tok_ext = torch.cat([token_sorted, torch.full((1,), rows, dtype = token_sorted.dtype, device = y.device)])
        w_ext = torch.cat([weight_sorted.half(), torch.zeros((1,), dtype = torch.half, device = y.device)])
        starts = [0]
        for c in expert_count_list:
            starts.append(starts[-1] + c)
        handled = set()
        slot = None
        if scratch is not None:
            # groups were laid out after the fused slots, in plan order, len(grp) * cmax each
            slot = sum(c for c in expert_count_list[:self.num_local_experts or self.num_experts]
                       if 0 < c <= self.fused_rows) if self.fused_mode_buffers is not None else 0
        for grp in groups:
            out_slab = None
            if scratch is not None:
                cmax = max(expert_count_list[e] for e in grp)
                out_slab = scratch[slot : slot + len(grp) * cmax]
                slot += len(grp) * cmax
            recon.run_group(
                y_ext, fhs_ext, tok_ext, w_ext,
                grp, [starts[e] for e in grp], [expert_count_list[e] for e in grp],
                out_slab = out_slab)
            handled.update(grp)
        return handled

    @override
    def unload(self):
        self.cpu_unload()
        self.bc = None
        self.fused_mode_buffers = None
        self.batch_recon = None
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
        self.tid2eid = None
        self.per_expert_scale = None
        self.bcast_sel_bsz1 = None
        self.bcast_weights_bsz1 = None
        super().unload()


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if self.alt_residual_channel:
            y = params["residual"].view(-1, self.hidden_size)
        else:
            y = x.view(-1, self.hidden_size)
        bsz = y.shape[0]
        bc_sh_exp = False
        if params.get("autosplit_measure"):
            self._measure_rows = bsz
        # Shape of the routed sum: the experts' width is the latent width in a latent MoE
        eshape = x.shape[:-1] + (self.expert_size,)

        # Eligibility for the multi-row CUDA-graph path (bsz 1..MAX_BSZN): computed up front so
        # it can override the f_threshold-based routing below (bsz>=f_threshold would otherwise
        # always fall through to the exl3_moe/dense path first, capping this tier's reach at
        # f_threshold-1 instead of MAX_BSZN). Expert-range shards (CPU split, TP) are handled
        # by the mgemm kernel's position-preserving mask at num_tokens > 1 (out-of-range picks
        # inactive in place, so the grouped reduction's slot runs stay intact). Shared experts
        # are supported at any bsz via BC_GatedMLP's own multi-row graph (see mlp.py)
        bszn_eligible = self.bc is not None and bsz <= MAX_BSZN

        # Routing
        if self.router_pre_norm:
            z = self.router_pre_norm.forward(y, params, out_dtype = torch.half)
        else:
            z = y

        if self.routing_gate is not None:
            selected_experts, routing_weights = self.routing_fn(bsz, self.routing_cfg, z, params)
        elif bsz == 1:
            # Stable buffers, not per-call allocations: the BC bsz-1 graph reads the routing tensors
            # through unpatched nodes (bias adds), whose addresses are baked at capture time
            if self.bcast_sel_bsz1 is None:
                self.bcast_sel_bsz1 = torch.empty((1, self.num_experts_per_tok), dtype = torch.long, device = self.device)
                self.bcast_weights_bsz1 = torch.empty((1, self.num_experts_per_tok), dtype = torch.half, device = self.device)
            selected_experts = self.bcast_sel_bsz1
            routing_weights = self.bcast_weights_bsz1
        else:
            selected_experts = torch.empty((bsz, self.num_experts_per_tok), dtype = torch.long, device = self.device)
            routing_weights = torch.empty((bsz, self.num_experts_per_tok), dtype = torch.half, device = self.device)

        # Extra norm (Gemma4)
        if self.routed_pre_norm:
            y = self.routed_pre_norm.forward(y, params, out_dtype = torch.half)

        # Latent MoE: the experts (GPU and CPU-resident alike) see the projected input
        if self.latent_in is not None:
            y = self.latent_in.forward(y, params)

        # Broadcast routing indices and weights
        if self.routing_device is not None:
            params["backend"].broadcast(selected_experts, src_device = self.routing_device)
            params["backend"].broadcast(routing_weights, src_device = self.routing_device)

        # CPU expert offload (block_sparse_mlp_cpu.py): split layers hand the tail experts'
        # share to the worker now so it computes concurrently with the GPU expert paths below
        # (folded back in by cpu_split_combine); whole-layer offload replaces the routed sum
        cpu_partial = None
        cpu_pending = None
        if self.cpu_split_first is not None and not params.get("autosplit_measure"):
            cpu_partial, cpu_pending = self.cpu_split_submit(y, bsz, selected_experts, routing_weights)

        if self.cpu_offload:
            final_hidden_states = self.cpu_offload_forward(eshape, y, selected_experts, routing_weights, params)

        # Empty slice
        elif self.intermediate_size == 0 or self.num_local_experts == 0:
            final_hidden_states = torch.zeros(eshape, dtype = torch.float, device = y.device)

        # Torch/C++/fused path
        elif (
            (bsz >= self.f_threshold and not bszn_eligible) or not self.is_quantized or
            self.config.infer_params.no_reconstruct or
            not (self.support_quant_paths or bszn_eligible)
        ):
            # One spare row: the batched reconstruct tier's padding sink (never read back)
            fhs_ext = torch.zeros((y.shape[0] + 1, y.shape[1]), dtype = torch.float, device = y.device)
            final_hidden_states = fhs_ext[:y.shape[0]]

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
                flat_token = buffered_interleaved_arange(num_tokens, top_k, device = y.device)

                # Map to local expert ids whenever this module holds a slice (TP shard or
                # CPU expert split), not only in the multi-device routing case
                if self.num_local_experts == self.num_experts:
                    flat_expert_local = flat_expert_global
                else:
                    flat_expert_local = flat_expert_global - self.routing_first
                    valid = (flat_expert_local >= 0) & (flat_expert_local < E)
                    flat_expert_local = torch.where(valid, flat_expert_local, torch.full_like(flat_expert_local, E))

                # Group once by local expert id (including sentinel for expert-P mode)
                order = flat_expert_local.argsort()
                token_sorted = flat_token[order]
                weight_sorted = flat_weight[order]

                # Count how many assignments per expert. With few enough total assignments no
                # expert can exceed the fused kernel's row capacity, so the readback (a CPU sync
                # per layer, ~33% idle at MTP verify shapes) is skipped and everything is fused
                expert_count = torch.bincount(flat_expert_local, minlength = E + 1)
                if self.fused_mode_buffers is not None and num_tokens * top_k <= self.fused_rows:
                    expert_count_list = None
                else:
                    expert_count_list = expert_count.tolist()

                # Tier plan: fused kernel for experts up to self.fused_rows rows, batched
                # reconstruct groups above that up to the tile cap, per-expert reconstruct beyond
                recon = None
                groups = []
                min_rows = 0
                if expert_count_list is None:
                    fused_total = num_tokens * top_k
                else:
                    fused_total = 0
                    if self.fused_mode_buffers is not None:
                        min_rows = self.fused_rows
                        fused_total = sum(c for c in expert_count_list[:num_ex] if 0 < c <= self.fused_rows)
                    recon = self._batch_recon_layer(y)
                    if recon is not None:
                        lim = max(min_rows, TEMP_ROWS_GRAPH)
                        heavy = [e for e in range(num_ex) if lim < expert_count_list[e] <= recon.max_rows]
                        if heavy:
                            from .moe_batch_recon import plan_groups
                            groups = plan_groups(heavy, lambda e: expert_count_list[e], recon.cap)

                # Deterministic accumulation (FUSED_DET): every assignment of the fused and
                # batched tiers gets a slot in one fp32 scratch (fused experts: count rows each,
                # weighted by the kernel; batched experts: their group's padded row count each,
                # unweighted) and one exl3_moe_gather per layer sums each token's slots in k
                # order, so identical inputs give identical outputs. The atomic alternative adds
                # contributions in arrival order. Slot tables come from the host-side counts, the
                # all-fused fast path builds them on the device. Scratch is prefill-shaped and per
                # call. Its traffic (written once by the GEMMs, read once by the gather) matches
                # what the atomic index_add_ path moved
                scratch = tables = inv_order = None
                if FUSED_DET and (fused_total or groups):
                    A = flat_expert_local.shape[0]
                    inv_order = torch.empty_like(order).scatter_(
                        0, order, torch.arange(A, device = order.device))
                    if expert_count_list is None:
                        expert_start = torch.cumsum(expert_count, 0) - expert_count
                        tables = torch.stack([expert_start, expert_start, (expert_count > 0).long()])
                        n_slots = fused_total
                    else:
                        import numpy as np
                        E1 = len(expert_count_list)
                        base = np.zeros(E1, dtype = np.int64)
                        kind = np.zeros(E1, dtype = np.int64)
                        starts_np = np.cumsum([0] + expert_count_list[:-1]).astype(np.int64)
                        n_slots = 0
                        if self.fused_mode_buffers is not None:
                            for e in range(num_ex):
                                c = expert_count_list[e]
                                if 0 < c <= self.fused_rows:
                                    base[e] = n_slots; kind[e] = 1; n_slots += c
                        for grp in groups:
                            cmax = max(expert_count_list[e] for e in grp)
                            for b, e in enumerate(grp):
                                base[e] = n_slots + b * cmax; kind[e] = 2
                            n_slots += len(grp) * cmax
                        tables = torch.from_numpy(np.stack([base, starts_np, kind])).to(y.device, non_blocking = True)
                    scratch = torch.empty((max(n_slots, 1), y.shape[1]), dtype = torch.float, device = y.device)

                def run_fused(num_active, count_lo = 1, count_hi = self.fused_rows, m_tile = 16):
                    # Gateless: the up module stands in for the gate pointer tables (the kernel
                    # skips the gate GEMM when activation_fn_idx is MOE_ACT_RELU2_NOGATE)
                    multi_gate = self.multi_gate if self.gated else self.multi_up
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
                        self.activation_fn_idx,
                        multi_gate.K,
                        self.multi_up.K,
                        self.multi_down.K,
                        multi_gate.ptrs_trellis,
                        multi_gate.ptrs_suh,
                        multi_gate.ptrs_svh,
                        self.multi_up.ptrs_trellis,
                        self.multi_up.ptrs_suh,
                        self.multi_up.ptrs_svh,
                        self.multi_down.ptrs_trellis,
                        self.multi_down.ptrs_suh,
                        self.multi_down.ptrs_svh,
                        multi_gate.mcg,
                        multi_gate.mul1,
                        self.multi_up.mcg,
                        self.multi_up.mul1,
                        self.multi_down.mcg,
                        self.multi_down.mul1,
                        self.act_limit,
                        num_active,
                        scratch, tables[0] if tables is not None else None,
                        count_lo, count_hi, m_tile
                    )

                # num_active -1 = unknown (all fused), kernel launches at max concurrency
                if self.fused_mode_buffers is not None:
                    if expert_count_list is None:
                        run_fused(-1)
                    else:
                        counts = [c for c in expert_count_list[:num_ex] if 0 < c <= self.fused_rows]
                        t1 = sum(1 for c in counts if MTILE_T1 < c <= MTILE_T2)
                        t2 = sum(1 for c in counts if c > MTILE_T2)
                        if self.mtile_ok and (t1 or t2):
                            # One launch per row tile over its expert range, largest first
                            t0 = len(counts) - t1 - t2
                            if t2:
                                run_fused(t2, MTILE_T2 + 1, self.fused_rows, 64)
                            if t1:
                                run_fused(t1, MTILE_T1 + 1, MTILE_T2, 32)
                            if t0:
                                run_fused(t0, 1, MTILE_T1, 16)
                        else:
                            run_fused(len(counts))

                # Batched reconstruct tier (into slots when deterministic, else accumulating)
                batched = ()
                if groups:
                    batched = self._run_batch_recon(
                        recon, y, fhs_ext, token_sorted, weight_sorted, expert_count_list, groups,
                        scratch, tables)

                # One fixed-order gather over every slot
                if scratch is not None:
                    ext.exl3_moe_gather(final_hidden_states, scratch, flat_expert_local, inv_order,
                                        tables[1], tables[0], tables[2], weight_sorted)

                out_state = None
                interm = None
                interm_a = None
                max_count = 0
                start = 0

                # expert_count_list None: everything already handled by the fused kernel above
                for expert_idx in range(num_ex if expert_count_list is not None else 0):
                    count = expert_count_list[expert_idx]
                    end = start + count
                    if count <= min_rows or expert_idx in batched:
                        start = end
                        continue

                    top_x = token_sorted[start:end]
                    w = weight_sorted[start:end].unsqueeze(1)

                    current_state = y.index_select(0, top_x)

                    if self.bc is not None and self.support_quant_paths:
                        # Graph path
                        if count <= TEMP_ROWS_GRAPH:
                            self.bc.run_single_expert(current_state, expert_idx)
                            current_state = self.experts_cfg.out_d2[:count]

                        # DQ path
                        else:
                            if count > max_count:
                                out_state = torch.empty((count, self.expert_size), dtype = torch.float, device = self.device)
                                interm = torch.empty((count * 2, self.intermediate_size_padded), dtype = self.interm_dtype, device = self.device)
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

                            yh = torch.empty((count * 2, self.expert_size), dtype = torch.half, device = self.device)
                            self.bc.run_single_expert_dq(current_state, expert_idx, yh, interm_, interm_a_, out_state)
                            current_state = out_state_
                    else:

                        # Torch path
                        def mlp(exp_i, xc):
                            u = self.ups[exp_i].forward(xc, params)
                            if self.gated:
                                g = self.gates[exp_i].forward(xc, params)
                                a = u if self.interm_dtype == torch.half else torch.empty_like(u, dtype = torch.half)
                                self.activation_fn_call(g, u, a, self.act_limit)
                            else:
                                a = self.gateless_act(u)
                                if a.dtype != torch.half:
                                    a = a.half()
                            return self.downs[exp_i].forward(a, params)

                        current_state = mlp(expert_idx, current_state)

                    current_state.mul_(w)
                    final_hidden_states.index_add_(0, top_x, current_state)
                    start = end

            final_hidden_states = final_hidden_states.reshape(eshape)

        # Multi-row CUDA-graph path (bsz 1..MAX_BSZN): a single cooperative mgemm call per
        # projection across all bsz*top_k assignment slots (no sort/dedup -- overlap between
        # tokens this small is rare and not worth the argsort/bincount host-sync cost that the
        # fused/exl3_moe path pays), captured as one CUDA graph per bsz and replayed with only a
        # few tensor pointers patched. Shared experts (if present) run through their own
        # multi-row BC_GatedMLP graph, fused into the same capture. Expert-range shards (CPU
        # split, TP) produce a partial sum here: out-of-range picks are masked inactive inside
        # the mgemm kernel and contribute exact zeros
        elif bszn_eligible:
            self.bc.run_bszN(y, selected_experts, routing_weights)
            if self.experts_cfg.out_trim is not None:
                final_hidden_states = self.experts_cfg.out_trim[:bsz].view(eshape)
            else:
                final_hidden_states = self.experts_cfg.out_d[:bsz, ...].view(eshape)
            bc_sh_exp = self.bc_sh_exp

        # Per-token mgemm loop: fallback for TP-sharded / shared-experts models at bsz 2..f_threshold-1
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
                if self.gated:
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
                        0,
                        1, None, None)

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
                    0,
                    1, None, None)

                # Activation (gateless: relu_mul(u, u, a) = relu2(u))
                act_g = cfg.interm_g if self.gated else cfg.interm_u
                self.activation_fn_call(act_g, cfg.interm_u, cfg.interm_a, self.act_limit)

                # Down
                # A_had must not alias A (the autotuner relaunches on the first call); the
                # gate buffer is free after the activation
                ext.exl3_mgemm(
                    cfg.interm_a,
                    self.multi_down.ptrs_trellis,
                    cfg.out_d,
                    self.multi_down.ptrs_suh,
                    cfg.interm_g,
                    self.multi_down.ptrs_svh,
                    selected_experts[i],
                    routing_weights[i],
                    self.multi_down.K,
                    -1,
                    self.multi_down.mcg,
                    self.multi_down.mul1,
                    mine,
                    maxe,
                    0,
                    1, None, None)

                t = cfg.out_d[0]
                final_hidden_states[i:i+1] = t

            final_hidden_states = final_hidden_states.view(eshape)

        else:
            y = y.unsqueeze(0)
            cfg = self.experts_cfg

            # Gate
            if self.gated:
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
                    0,
                    1, None, None)

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
                0,
                1, None, None)

            # Activation (gateless: relu_mul(u, u, a) = relu2(u))
            act_g = cfg.interm_g if self.gated else cfg.interm_u
            self.activation_fn_call(act_g, cfg.interm_u, cfg.interm_a, self.act_limit)

            # Down
            # A_had must not alias A (the autotuner relaunches on the first call)
            ext.exl3_mgemm(
                cfg.interm_a,
                self.multi_down.ptrs_trellis,
                cfg.out_d,
                self.multi_down.ptrs_suh,
                cfg.interm_g,
                self.multi_down.ptrs_svh,
                selected_experts,
                routing_weights,
                self.multi_down.K,
                -1,
                self.multi_down.mcg,
                self.multi_down.mul1,
                cfg.min_expert,
                cfg.max_expert,
                0,
                1, None, None)

            final_hidden_states = cfg.out_d[:1, ...].view(eshape)

        # CPU tail partial folds in before the post norms (nonlinear: they must see the
        # complete routed sum)
        final_hidden_states = self.cpu_split_combine(final_hidden_states, cpu_partial, cpu_pending, eshape)

        # Latent MoE: project the routed sum back to the residual width (linear, so under TP
        # each rank's partial sum projects on its own ahead of the reduction)
        if self.latent_out is not None:
            final_hidden_states = self.latent_out.forward(final_hidden_states.to(torch.half), params)

        # The post norms are nonlinear, so under TP their inputs must be complete sums, not
        # per-rank partials: reduce the routed and shared contributions separately before the
        # norms (Gemma4 MoE), after which every rank holds identical complete tensors and the
        # final reduction is skipped
        pre_norm_reduce = self.tp_reduce and (
            self.routed_post_norm is not None or
            (self.shared_experts is not None and self.shared_experts_post_norm is not None and not bc_sh_exp)
        )
        if pre_norm_reduce:
            params["backend"].all_reduce(
                final_hidden_states,
                self.intermediate_size > 0 and self.num_local_experts > 0
            )

        # Extra norm (Gemma4)
        if self.routed_post_norm:
            final_hidden_states = self.routed_post_norm.forward(final_hidden_states, params)

        # Shared experts
        if self.shared_experts and not bc_sh_exp:
            y = self.shared_experts.forward(x, params)
            if pre_norm_reduce:
                params["backend"].all_reduce(y, True)
            if self.shared_experts_post_norm:
                y = self.shared_experts_post_norm.forward(y, params)
            if self.shared_gate:
                if bsz > 32:
                    z = self.shared_gate.forward(x, params)
                    ext.add_sigmoid_gate(y, z, final_hidden_states)
                else:
                    ext.add_sigmoid_gate_proj(y, x, final_hidden_states, self.shared_gate.inner.weight)
            else:
                final_hidden_states += y

        # Output reduction
        if self.tp_reduce and not pre_norm_reduce:
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
            t[f"{self.key}.{self.e_score_correction_bias_key}"] = self.e_score_correction_bias.contiguous()
        if self.e_score_bias_vl is not None:
            t[f"{self.key}.{self.e_score_bias_vl_key}"] = self.e_score_bias_vl.contiguous()
        if self.tid2eid is not None:
            t[f"{self.key}.{self.tid2eid_key}"] = self.tid2eid.contiguous()
        if self.per_expert_scale is not None:
            t[f"{self.key}.{self.per_expert_scale_key}"] = self.per_expert_scale.contiguous()
        return t


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        storage = 0
        storage += self.routing_gate.storage_size()
        if self.shared_gate:
            storage += self.shared_gate.storage_size()
        for g in self.gates: storage += g.storage_size()
        for u in self.ups: storage += u.storage_size()
        for d in self.downs: storage += d.storage_size()
        # The latent projections are replicated on every rank
        storage_d = 0
        if self.latent_in is not None:
            storage_d += self.latent_in.storage_size() + self.latent_out.storage_size()
        # TODO: More precise overhead estimate accounting for gate etc.
        overhead_d = self.hidden_size * torch.float.itemsize
        overhead_s = 4 * self.intermediate_size * (self.interm_dtype or torch.half).itemsize
        if self.interm_dtype != torch.half:
            overhead_s += self.intermediate_size * torch.half.itemsize
        recons = max(
            self.gates[0].recons_size() if self.gated else 0,
            self.ups[0].recons_size(),
            self.downs[0].recons_size()
        )
        use_tp_split = options.get("moe_tensor_split", False)
        tpa = TPAllocation(
            key = self.key,
            channel_width = 128 if use_tp_split else 1,
            channel_unit = "channels" if use_tp_split else "experts",
            storage_per_device = storage_d,
            storage_to_split = storage,
            overhead_per_device = overhead_d,
            overhead_to_split = overhead_s,
            recons_temp = recons,
            channels_to_split = self.ups[0].out_features // 128 if use_tp_split else self.num_experts,
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
                "latent_size": self.latent_size,
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
                "alt_residual_channel": self.alt_residual_channel,
                "key_tid2eid": self.tid2eid_key,
                "key_e_score_bias_vl": self.e_score_bias_vl_key,
            },
            # Hash-MoE bootstrap layers (DeepSeek-V4): frozen token->experts table, needed
            # wherever routing runs (the output device, like the routing gate)
            "tid2eid": producer.send(self.tid2eid) if self.tid2eid is not None else None,
            "routing_gate": _export(self.routing_gate),
            "shared_gate": _export(self.shared_gate),
            "latent_in": _export(self.latent_in),
            "latent_out": _export(self.latent_out),
            "e_score_correction_bias": producer.send(self.e_score_correction_bias),
            "e_score_bias_vl": producer.send(self.e_score_bias_vl) if self.e_score_bias_vl is not None else None,
            "per_expert_scale": producer.send(self.per_expert_scale),
            "gates": [_export(self.gates[i]) for i in range(self.num_experts)] if self.gated else None,
            "ups": [_export(self.ups[i]) for i in range(self.num_experts)],
            "downs": [_export(self.downs[i]) for i in range(self.num_experts)],
            "shared_experts": self.shared_experts.tp_export(plan, producer) \
                if self.shared_experts is not None else None,
            "shared_experts_post_norm": _export(self.shared_experts_post_norm),
            "router_pre_norm": _export(self.router_pre_norm),
            "routed_pre_norm": _export(self.routed_pre_norm),
            "routed_post_norm": _export(self.routed_post_norm),
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

        # Gateless experts (NemotronH) export gates as None; the local module gets an empty
        # gates list so the ctor derives gated = False
        gated = exported.get("gates") is not None

        # Tensor parallel
        if unit == "channels":
            num_local_experts = exported["kwargs"]["num_experts"]
            gu_split = (True, first, last)
            d_split = (False, first, last)
            exported["kwargs"]["intermediate_size"] = last - first
            gates = [_import_i_split("gates", i, gu_split) for i in range(num_local_experts)] if gated else []
            ups = [_import_i_split("ups", i, gu_split) for i in range(num_local_experts)]
            downs = [_import_i_split("downs", i, d_split) for i in range(num_local_experts)]
            routing_first = 0
            routing_last = num_local_experts

        # Expert parallel
        elif unit == "experts":
            num_local_experts = last - first
            gates = [_import_i("gates", i) for i in range(first, last)] if gated else []
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
            shared_gate = _import("shared_gate"),
            latent_in = _import("latent_in"),
            latent_out = _import("latent_out"),
            routing_gate = _import("routing_gate") if device == output_device else None,
            routing_first = routing_first,
            routing_last = routing_last,
            routing_device = output_device,
            shared_experts_post_norm = _import("shared_experts_post_norm"),
            router_pre_norm = _import("router_pre_norm"),
            routed_pre_norm = _import("routed_pre_norm"),
            routed_post_norm = _import("routed_post_norm"),
        )

        module.device = device
        module.e_score_correction_bias = consumer.recv(exported["e_score_correction_bias"], cuda = True)
        if exported.get("e_score_bias_vl") is not None:
            module.e_score_bias_vl = consumer.recv(exported["e_score_bias_vl"], cuda = True)
        module.per_expert_scale = consumer.recv(exported["per_expert_scale"], cuda = True)
        if exported.get("tid2eid") is not None and device == output_device:
            module.tid2eid = consumer.recv(exported["tid2eid"], cuda = True)
        if unit == "channels" or num_local_experts > 0:
            module.load_local()
        if module.routing_gate is not None:
            module.load_routing()
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True
        return module
