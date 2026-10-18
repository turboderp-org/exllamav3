"""
MoE routing test helpers: a RoutingCFG over a random gate, and the comparison predicates for top-k selections
against a reference (rows away from near-ties, order-insensitive selection equality).

    cfg = make_cfg(K, E, topk, device, bias = b)
    clear = clear_rows(ref_scores, topk, margin)          # rows whose k-th / (k+1)-th score gap exceeds margin
    assert same_sel(sel, ref_sel)[clear].all()
"""

import torch


def make_cfg(K: int, E: int, topk: int, device, bias = None, esb = None, esb_vl = None, scaling: float = 1.0,
             seed: int = 0):
    """RoutingCFG for a (K, E) fp16 gate of scale K^-0.5 (seeded), single group, with the optional router bias,
    e_score_correction_bias and its vision-row variant"""
    from exllamav3.modules.block_sparse_mlp_routing import RoutingCFG
    torch.manual_seed(seed)
    gate = (torch.randn(K, E, device = device) * K ** -0.5).half()
    return RoutingCFG(
        gate_tensor = gate, gate_tensor_t = None, num_experts = E, num_experts_per_tok = topk,
        router_logits_bsz1 = torch.empty(1, E, dtype = torch.half, device = device),
        routing_weights_bsz1 = torch.empty(1, topk, dtype = torch.half, device = device),
        selected_experts_bsz1 = torch.empty(1, topk, dtype = torch.long, device = device),
        e_score_correction_bias = esb, e_score_bias_h = None, routed_scaling_factor = scaling,
        n_group = 1, topk_group = 1, per_expert_scale = None, router_bias = bias, e_score_bias_vl = esb_vl,
    )


def clear_rows(scores: torch.Tensor, topk: int, margin: float) -> torch.Tensor:
    """(rows,) bool: rows whose topk-th and (topk+1)-th scores differ by more than margin, where any correct
    implementation must make the same selection"""
    tv = torch.topk(scores, topk + 1, dim = -1).values
    return (tv[:, topk - 1] - tv[:, topk]) > margin


def same_sel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(rows,) bool: the two (rows, topk) selections hold the same experts, in any order"""
    return (torch.sort(a, dim = 1).values == torch.sort(b, dim = 1).values).all(dim = 1)
