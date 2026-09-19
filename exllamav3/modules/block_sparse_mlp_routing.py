"""
Expert routing for BlockSparseMLP: the RoutingCFG bundle each layer builds at load time and
the per-arch top-k functions the module dispatches to (router_type). Multi-row calls take
bucketed workspaces from the per-device static cache up to ROUTING_CACHE_ROWS rows; the bsz-1
buffers live in the RoutingCFG (CUDA-graph paths bake their addresses).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from ..util.device_copy import to_device
from ..ext import exllamav3_ext as ext
from ..util.tensor import g_tensor_cache
from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX

ROUTING_CACHE_ROWS = 128


def _routing_buffers(cfg, bsz, device):
    """Router outputs for a multi-row call: (router_logits (bsz, E) half, selected_experts
    (bsz, K) long, routing_weights (bsz, K) half). Decode/MTP-class row counts (verify windows,
    small batches: up to ROUTING_CACHE_ROWS) take bucketed workspaces from the per-device static
    cache, shared by every MoE layer on the device: a layer's routing outputs are consumed by
    its own expert compute (and offload/broadcast hand-offs) before the next layer routes on the
    same stream, so one set per device suffices and nearby row counts share a backing instead
    of one static per shape. Prefill-sized calls allocate per call (not CPU-bound, and the
    static cache is meant to hold only small buffers)."""
    E, K = cfg.num_experts, cfg.num_experts_per_tok
    if bsz <= ROUTING_CACHE_ROWS:
        ws = lambda numel, dtype, tag: g_tensor_cache.get_bucketed(device, numel, dtype, tag)
    else:
        ws = lambda numel, dtype, tag: torch.empty((numel,), dtype = dtype, device = device)
    return (
        ws(bsz * E, torch.half, "moe_route_logits").view(bsz, E),
        ws(bsz * K, torch.long, "moe_route_sel").view(bsz, K),
        ws(bsz * K, torch.half, "moe_route_w").view(bsz, K),
    )

# Score activations for the nogroup routing kernels (must match routing.cu)
ROUTING_ACT_SIGMOID = 0
ROUTING_ACT_SQRTSP = 1

def _esb_h(cfg):
    """fp16 selection-bias copy for the CUDA top-k kernels, built lazily (load may be
    deferred when the RoutingCFG is constructed). Mean-centered when the source is wider than
    fp16: selection is invariant to a constant shift, and centering keeps fp16 rounding well
    below the inter-expert score gaps even when the bias values are large (GLM-5.2 ~34.0)."""
    if cfg.e_score_bias_h is None and cfg.e_score_correction_bias is not None:
        esb = cfg.e_score_correction_bias
        cfg.e_score_bias_h = esb if esb.dtype == torch.half else (esb - esb.mean()).half()
    return cfg.e_score_bias_h


@dataclass
class RoutingCFG:
    gate_tensor: torch.Tensor
    gate_tensor_t: torch.Tensor | None
    num_experts: int
    num_experts_per_tok: int
    router_logits_bsz1: torch.Tensor
    routing_weights_bsz1: torch.Tensor
    selected_experts_bsz1: torch.Tensor
    e_score_correction_bias: torch.Tensor | None
    e_score_bias_h: torch.Tensor | None   # lazy, see _esb_h
    routed_scaling_factor: float | None
    n_group: int | None
    topk_group: int | None
    per_expert_scale: torch.Tensor | None
    router_bias: torch.Tensor | None = None
    tid2eid: torch.Tensor | None = None
    e_score_bias_vl: torch.Tensor | None = None   # DeepSeek-V4 vision: selection bias for image rows
    gate_i8: torch.Tensor | None = None     # (2, E, K) int8 hi/lo slices, lazy (see _gate_i8)
    gate_sb: torch.Tensor | None = None     # (E) fp32 row scales


def _gate_t(cfg):
    """Transposed (E, K) half gate for the single-row GEMV, plus the int8 hi/lo slices and row
    scales for the deterministic multi-row projection (ext.routing_gemm_det), both built lazily
    (weights may be deferred when the RoutingCFG is constructed)."""
    if cfg.gate_tensor_t is None:
        cfg.gate_tensor_t = cfg.gate_tensor.T.contiguous()
    if cfg.gate_i8 is None and cfg.gate_tensor_t.dtype == torch.half:
        E, K = cfg.gate_tensor_t.shape
        cfg.gate_i8 = torch.empty((2, E, K), dtype = torch.int8, device = cfg.gate_tensor_t.device)
        cfg.gate_sb = torch.empty((E,), dtype = torch.float, device = cfg.gate_tensor_t.device)
        ext.det_quant_weight(cfg.gate_tensor_t, cfg.gate_i8, cfg.gate_sb)
    return cfg.gate_tensor_t

def routing_std(bsz, cfg, y, params):
    if bsz == 1:
        _gate_t(cfg)
        ext.routing_std(
            y,
            cfg.gate_tensor,
            cfg.router_logits_bsz1,
            cfg.selected_experts_bsz1,
            cfg.routing_weights_bsz1,
            cfg.per_expert_scale,
            cfg.gate_tensor_t,
            None,
            cfg.gate_i8,
            cfg.gate_sb,
        )
        return cfg.selected_experts_bsz1, cfg.routing_weights_bsz1
    else:
        activate_all_experts = params.get("activate_all_experts")
        if activate_all_experts:
            router_logits = torch.matmul(y, cfg.gate_tensor)
            routing_weights = torch.softmax(router_logits, dim = -1)
            selected_experts = (
                torch.arange(start = 0, end = cfg.num_experts, dtype = torch.long, device = y.device)
                .repeat((bsz, 1))
            )
            if cfg.per_expert_scale is not None:
                routing_weights *= cfg.per_expert_scale.unsqueeze(0)
            return selected_experts, routing_weights
        else:
            router_logits, selected_experts, routing_weights = _routing_buffers(cfg, bsz, y.device)
            _gate_t(cfg)
            ext.routing_std(
                y,
                cfg.gate_tensor,
                router_logits,
                selected_experts,
                routing_weights,
                cfg.per_expert_scale,
                cfg.gate_tensor_t,
                None,
                cfg.gate_i8,
                cfg.gate_sb,
            )
        return selected_experts, routing_weights


def routing_std_bias(bsz, cfg, y, params):
    """Standard softmax routing with a bias on the router logits (gpt-oss): the bias enters
    before top-k selection, and the weights are the softmax over the selected biased logits
    (equivalent to renormalizing the full biased softmax over the top-k set)."""
    if params.get("activate_all_experts"):
        if cfg.router_bias is not None:
            router_logits = torch.addmm(cfg.router_bias, y, cfg.gate_tensor)
        else:
            router_logits = torch.matmul(y, cfg.gate_tensor)
        routing_weights = torch.softmax(router_logits.float(), dim = -1).half()
        selected_experts = (
            torch.arange(start = 0, end = cfg.num_experts, dtype = torch.long, device = y.device)
            .repeat((bsz, 1))
        )
        return selected_experts, routing_weights
    # Every batch size on the deterministic ext path (bias before top-k inside the kernel)
    _gate_t(cfg)
    if bsz == 1:
        router_logits, selected_experts, routing_weights = \
            cfg.router_logits_bsz1, cfg.selected_experts_bsz1, cfg.routing_weights_bsz1
    else:
        router_logits, selected_experts, routing_weights = _routing_buffers(cfg, bsz, y.device)
    ext.routing_std(
        y,
        cfg.gate_tensor,
        router_logits,
        selected_experts,
        routing_weights,
        cfg.per_expert_scale,
        cfg.gate_tensor_t,
        cfg.router_bias,
        cfg.gate_i8,
        cfg.gate_sb,
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
        _gate_t(cfg)
        ext.routing_ds3_nogroup(
            y,
            cfg.gate_tensor,
            cfg.router_logits_bsz1,
            _esb_h(cfg),
            cfg.selected_experts_bsz1,
            cfg.routing_weights_bsz1,
            cfg.routed_scaling_factor,
            cfg.gate_tensor_t,
            ROUTING_ACT_SIGMOID,
            cfg.gate_i8,
            cfg.gate_sb,
        )
        return cfg.selected_experts_bsz1, cfg.routing_weights_bsz1

    else:
        activate_all_experts = params.get("activate_all_experts")
        if activate_all_experts:
            router_logits = torch.matmul(y, cfg.gate_tensor)
            routing_weights = router_logits.sigmoid().float()
            if cfg.e_score_correction_bias is not None:
                routing_weights = routing_weights + cfg.e_score_correction_bias.unsqueeze(0).float()
            factor = cfg.routed_scaling_factor / (routing_weights.sum(dim = -1, keepdim = True) + 1e-20)
            routing_weights = (routing_weights * factor).half()
            selected_experts = (
                torch.arange(start = 0, end = cfg.num_experts, dtype = torch.long, device = y.device)
                .repeat((bsz, 1))
            )
        else:
            router_logits, selected_experts, routing_weights = _routing_buffers(cfg, bsz, y.device)
            _gate_t(cfg)
            ext.routing_ds3_nogroup(
                y,
                cfg.gate_tensor,
                router_logits,
                _esb_h(cfg),
                selected_experts,
                routing_weights,
                cfg.routed_scaling_factor,
                cfg.gate_tensor_t,
                ROUTING_ACT_SIGMOID,
                cfg.gate_i8,
                cfg.gate_sb,
            )
        return selected_experts, routing_weights


def _sqrtsp_scores(cfg, y):
    logits = torch.matmul(y.float(), cfg.gate_tensor.float())
    return F.softplus(logits).sqrt()


def _vl_rows(cfg, params, bsz, device):
    """DeepSeek-V4 vision: mask of image rows (multimodal embedding ids) in the current chunk,
    or None when there are none. Only chunks that carry indexed embeddings (prompt prefill)
    are inspected, computed once per forward per device, so text decode never syncs."""
    if cfg.e_score_bias_vl is None or not params.get("indexed_embeddings"):
        return None
    key = ("_vl_rows", str(device))
    if key in params:
        return params[key]
    from .attn import get_for_device
    ids = get_for_device(params, "input_ids", device).reshape(-1)
    assert ids.shape[0] == bsz, f"routing: {bsz} hidden rows but {ids.shape[0]} input ids"
    mask = ids >= FIRST_MM_EMBEDDING_INDEX
    mask = mask if bool(mask.any()) else None
    params[key] = mask
    return mask


def _esb_vl_h(cfg):
    """fp16 copy of the vision selection bias (mean-centered like _esb_h; selection is shift
    invariant), built lazily."""
    if getattr(cfg, "e_score_bias_vl_h", None) is None:
        esb = cfg.e_score_bias_vl
        cfg.e_score_bias_vl_h = esb if esb.dtype == torch.half else (esb - esb.mean()).half()
    return cfg.e_score_bias_vl_h


def _routing_sqrtsp_vl(cfg, y, vl, hash_sel):
    """sqrtsp routing for a chunk with image rows (reference Gate.forward): image rows select
    top-k on scores + bias_vl; text rows use the hash table selection when given (hash layers),
    else scores + bias. Weights: raw scores over the selected set, normalized, times
    routed_scaling_factor. Composed from the deterministic ext kernels (one top-k pass per bias
    over all rows, merged by the row mask), so tensor-parallel ranks agree bit for bit."""
    bsz = y.shape[0]
    _gate_t(cfg)
    vl_col = vl.unsqueeze(-1)

    def topk_with(bias_h):
        logits, sel, w = _routing_buffers(cfg, bsz, y.device)
        ext.routing_ds3_nogroup(
            y, cfg.gate_tensor, logits, bias_h, sel, w, cfg.routed_scaling_factor,
            cfg.gate_tensor_t, ROUTING_ACT_SQRTSP, cfg.gate_i8, cfg.gate_sb,
        )
        return sel.clone(), w.clone()

    sel_vl, w_vl = topk_with(_esb_vl_h(cfg))
    if hash_sel is None:
        sel_tx, w_tx = topk_with(_esb_h(cfg))
    else:
        logits, _, w_tx = _routing_buffers(cfg, bsz, y.device)
        sel_tx = hash_sel.long()
        ext.routing_sel_norm(
            y, cfg.gate_tensor, logits, sel_tx, w_tx, cfg.routed_scaling_factor,
            cfg.gate_tensor_t, ROUTING_ACT_SQRTSP, cfg.gate_i8, cfg.gate_sb,
        )
    sel = torch.where(vl_col, sel_vl, sel_tx)
    w = torch.where(vl_col, w_vl, w_tx)
    return sel, w


def routing_sqrtsp(bsz, cfg, y, params):
    """DeepSeek-V4 router: sqrt(softplus(logits)) affinity, noaux_tc bias for selection only,
    weights normalized over the selected set, times routed_scaling_factor. The nogroup top-k
    kernel serves every batch size (one block per row); bsz 1 reuses the cached output
    buffers, larger batches allocate per call. activate_all_experts (conversion) stays
    torch-composed."""
    if params.get("activate_all_experts"):
        scores = _sqrtsp_scores(cfg, y)
        routing_weights = scores / (scores.sum(dim = -1, keepdim = True) + 1e-20)
        routing_weights = (routing_weights * cfg.routed_scaling_factor).half()
        selected_experts = (
            torch.arange(start = 0, end = cfg.num_experts, dtype = torch.long, device = y.device)
            .repeat((bsz, 1))
        )
        return selected_experts, routing_weights
    vl = _vl_rows(cfg, params, bsz, y.device)
    if vl is not None:
        return _routing_sqrtsp_vl(cfg, y, vl, None)
    _gate_t(cfg)
    if bsz == 1:
        router_logits = cfg.router_logits_bsz1
        selected_experts = cfg.selected_experts_bsz1
        routing_weights = cfg.routing_weights_bsz1
    else:
        router_logits, selected_experts, routing_weights = _routing_buffers(cfg, bsz, y.device)
    ext.routing_ds3_nogroup(
        y,
        cfg.gate_tensor,
        router_logits,
        _esb_h(cfg),
        selected_experts,
        routing_weights,
        cfg.routed_scaling_factor,
        cfg.gate_tensor_t,
        ROUTING_ACT_SQRTSP,
        cfg.gate_i8,
        cfg.gate_sb,
    )
    return selected_experts, routing_weights


def routing_sqrtsp_hash(bsz, cfg, y, params):
    """DeepSeek-V4 hash-MoE bootstrap: expert indices come from the frozen tid2eid table
    indexed by the current tokens (params["input_ids"], flattened row-major); the learned
    gate still weights the selected experts."""
    if params.get("activate_all_experts"):
        return routing_sqrtsp(bsz, cfg, y, params)
    # One device copy of the ids per forward via the params cache, shared by every hash
    # layer on that device; batch dims flatten row-major, matching the hidden-state rows
    from .attn import get_for_device
    input_ids = get_for_device(params, "input_ids", cfg.tid2eid.device).reshape(-1)
    assert input_ids.shape[0] == bsz, \
        f"hash routing: {bsz} hidden rows but {input_ids.shape[0]} input ids"
    vl = _vl_rows(cfg, params, bsz, y.device)
    if vl is not None:
        # Image rows are outside the table: look up token 0 for them (discarded) and route
        # them by top-k with the vision bias, as the reference does
        safe_ids = torch.where(to_device(vl, input_ids.device), torch.zeros_like(input_ids), input_ids)
        hash_sel = to_device(cfg.tid2eid[safe_ids], y.device).long()
        return _routing_sqrtsp_vl(cfg, y, vl, hash_sel)
    selected_experts = to_device(cfg.tid2eid[input_ids], y.device).long()
    _gate_t(cfg)
    if bsz == 1:
        routing_weights = cfg.routing_weights_bsz1
        router_logits = cfg.router_logits_bsz1
    else:
        router_logits, _, routing_weights = _routing_buffers(cfg, bsz, y.device)
    ext.routing_sel_norm(
        y,
        cfg.gate_tensor,
        router_logits,
        selected_experts,
        routing_weights,
        cfg.routed_scaling_factor,
        cfg.gate_tensor_t,
        ROUTING_ACT_SQRTSP,
        cfg.gate_i8,
        cfg.gate_sb,
    )
    return selected_experts, routing_weights
