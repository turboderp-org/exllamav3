"""
Routing layouts, expert pointer tables and dense-path references for the fused MoE kernel tests.

Expert sets are the dicts of testlib.exl3.rand_experts: {"g" | "u" | "d": [(trellis, suh, svh)] * E}, gate/up
(H -> I), down (I -> H). The fused kernels take one int64 table of data pointers per tensor kind and projection.

Routing layout (as BlockSparseMLP.forward builds it): the (token, k) assignments flattened token-major, sorted by
expert (stable); token_sorted / weight_sorted are the sorted token ids and routing weights, expert_count the
per-expert histogram with one trailing sentinel bucket (index E) for picks outside the local expert range,
expert_start its exclusive prefix sum, and inv the inverse of the sort permutation (slot of each flat assignment).
"""
from types import SimpleNamespace

import torch


def ptr_table(tensors, device) -> torch.Tensor:
    """int64 tensor of the tensors' data pointers"""
    return torch.tensor([t.data_ptr() for t in tensors], dtype = torch.long, device = device)


def experts_to(ex: dict, device) -> dict:
    """Copy of an expert set with every tensor on device"""
    return {p: [tuple(t.to(device) for t in e) for e in ex[p]] for p in ex}


def expert_ptr_tables(ex: dict, device) -> dict:
    """{p: (trellis_ptrs, suh_ptrs, svh_ptrs)} for an expert set already on device"""
    return {p: tuple(ptr_table([e[i] for e in ex[p]], device) for i in range(3)) for p in ex}


def moe_buffers(H: int, I: int, cap: int, device) -> list[torch.Tensor]:
    """The four per-CTA staging buffers of ext.exl3_moe (hidden, hidden, intermediate, intermediate)"""
    from exllamav3.ext import exllamav3_ext as ext
    conc = ext.exl3_moe_max_concurrency(torch.device(device).index or 0)
    return [torch.empty((conc, cap, dim), dtype = torch.half, device = device) for dim in (H, H, I, I)]


def routing_layout(sel: torch.Tensor, w: torch.Tensor, E: int) -> SimpleNamespace:
    """Sorted routing layout of (T, topk) expert picks sel and weights w over E local experts; picks outside
    [0, E) go to the sentinel bucket E. Returns flat_e, order, inv, token_sorted, weight_sorted, expert_count
    (E + 1), expert_start (E + 1), all on sel's device"""
    T, topk = sel.shape
    dev = sel.device
    flat_e = sel.reshape(-1)
    flat_e = torch.where((flat_e >= 0) & (flat_e < E), flat_e, torch.full_like(flat_e, E))
    order = torch.argsort(flat_e, stable = True)
    flat_t = torch.arange(T, device = dev).repeat_interleave(topk)
    A = flat_e.shape[0]
    expert_count = torch.bincount(flat_e, minlength = E + 1)
    return SimpleNamespace(
        flat_e = flat_e,
        order = order,
        inv = torch.empty_like(order).scatter_(0, order, torch.arange(A, device = dev)),
        token_sorted = flat_t[order],
        weight_sorted = w.reshape(-1)[order],
        expert_count = expert_count,
        expert_start = torch.cumsum(expert_count, 0) - expert_count,
    )


def contiguous_layout(counts: list[int], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Layout with one assignment per token, expert e owning the next counts[e] tokens. Returns (token_sorted,
    expert_count (E + 1, empty sentinel), expert_start)"""
    A = sum(counts)
    expert_count = torch.tensor(list(counts) + [0], dtype = torch.long, device = device)
    return torch.arange(A, device = device), expert_count, torch.cumsum(expert_count, 0) - expert_count


def swizzle_trellis(t: torch.Tensor) -> torch.Tensor:
    """Band-swizzled copy of a (k/16, n/16, 16K) trellis as the CPU expert arena stores it: physical order
    (n/128 group, k-tile, member, tile)"""
    tk, tn, ps = t.shape
    return t.view(tk, tn // 8, 8, ps).permute(1, 0, 2, 3).contiguous().view(tk, tn, ps)


def exl3_linear_ref(x: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, K: float,
                    codebook: str = "mul1") -> torch.Tensor:
    """x (m, k) fp16 through one EXL3 linear as the dense reconstruct path evaluates it: input Hadamard with suh,
    unrotated reconstructed weight, fp16 GEMM, output Hadamard with svh. Returns (m, n) fp16"""
    from exllamav3.ext import exllamav3_ext as ext
    from testlib.exl3 import codebook_flags
    dev = x.device
    mcg, mul1 = codebook_flags(codebook)
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh.to(dev), None, 1.0)
    W = torch.empty((trellis.shape[0] * 16, trellis.shape[1] * 16), dtype = torch.half, device = dev)
    ext.reconstruct(W, trellis.to(dev), K, mcg, mul1)
    y = torch.empty((x.shape[0], W.shape[1]), dtype = torch.half, device = dev)
    ext.hgemm(xh, W, y)
    ext.had_r_128(y, y, None, svh.to(dev), 1.0)
    return y


def expert_mlp_ref(ex: dict, e: int, x: torch.Tensor, K: float, K_down: float | None = None,
                   codebook: str = "mul1") -> torch.Tensor:
    """SiLU-gated MLP of expert e of an expert set on x (m, H) fp16, through exl3_linear_ref. Returns (m, H) fp32"""
    from exllamav3.ext import exllamav3_ext as ext
    g = exl3_linear_ref(x, *ex["g"][e], K, codebook)
    u = exl3_linear_ref(x, *ex["u"][e], K, codebook)
    ext.silu_mul(g, u, u, 0.0)
    return exl3_linear_ref(u, *ex["d"][e], K if K_down is None else K_down, codebook).float()
