"""
Unit tests for the production DSA kernels (modules/attention_fn/dsa_triton.py) against fp64
torch references. Covers the constexpr toggle matrix: window on/off, sinks on/off, gathered
vs dense pool, D_c padding (448), plus the indexer scoring kernel with causal bounds.

    python tests/test_dsa_kernels.py --device cuda:1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3.modules.attention_fn.dsa_triton import dsa_attn, dsa_indexer_scores
from exllamav3.ext import exllamav3_ext as ext

PAGE_SIZE = 256


def ref_attn(q, pool_c, pool_r, bt, sinks, ring, kv_chunk, window, win_floor, ring_beg,
             indices, dense, pool_len, q_pos0, m):
    R, H, D = q.shape
    D_r = pool_r.shape[-1]
    D_c = D - D_r
    pc = pool_c.reshape(-1, D_c).double()
    pr = pool_r.reshape(-1, D_r).double()
    out = torch.zeros((R, H, D), dtype = torch.float64, device = q.device)
    scale = D ** -0.5
    for r in range(R):
        rows = []
        if window > 0:
            q_abs = q_pos0 + r
            for j in range(window):
                a = q_abs - j
                if a < win_floor:
                    continue
                src = kv_chunk[a - q_pos0] if a >= q_pos0 else ring[a - ring_beg]
                rows.append(src.double().unsqueeze(0))
            rows = [torch.cat(rows, dim = 0)] if rows else []
        if dense:
            bound = min((q_pos0 + r + 1) // m, pool_len)
            idx = torch.arange(bound, device = q.device)
        else:
            idx = indices[r]
            idx = idx[idx >= 0].long()
        if idx.numel():
            page = idx // PAGE_SIZE
            phys = bt[r, page].long()
            tok = phys * PAGE_SIZE + idx % PAGE_SIZE
            rows.append(torch.cat([pc[tok], pr[tok]], dim = -1))
        if not rows:
            if sinks is not None:
                continue  # sink-only softmax -> zero output
            continue
        kv = torch.cat(rows, dim = 0)                       # (n, D)
        s = (q[r].double() @ kv.T) * scale                  # (H, n)
        if sinks is not None:
            s = torch.cat([s, sinks.double().unsqueeze(1)], dim = 1)
            p = torch.softmax(s, dim = -1)[:, :-1]
        else:
            p = torch.softmax(s, dim = -1)
        out[r] = p @ kv
    return out.float()


def _derotate_ref(o_r, inv_freq, pos):
    """GPT-J pair rotation of (H, D_r) fp by angle pos * inv_freq (fp64 reference)."""
    theta = (inv_freq.double() * pos)
    cos, sin = theta.cos(), theta.sin()
    e, o = o_r[..., 0::2], o_r[..., 1::2]
    return torch.stack((e * cos - o * sin, o * cos + e * sin), dim = -1).flatten(-2)


def check_attn(device, R, H, D_c, D_r, n_keys, topk, window, sinks_on, dense, m, seed, tol = 6e-3,
               derot = False, groups = 1):
    torch.manual_seed(seed)
    D = D_c + D_r
    pages = (n_keys + PAGE_SIZE - 1) // PAGE_SIZE
    pool_c = torch.randn((pages, PAGE_SIZE, D_c), dtype = torch.half, device = device)
    pool_r = torch.randn((pages, PAGE_SIZE, D_r), dtype = torch.half, device = device)
    perm = torch.randperm(pages, device = device, dtype = torch.int32)
    bt = perm.unsqueeze(0).expand(R, pages).contiguous()
    q = torch.randn((R, H, D), dtype = torch.half, device = device)
    sinks = (torch.randn(H, device = device) * 2.0 + 4.0) if sinks_on else None

    indices = None

    k_len = 0
    q_pos0 = n_keys * m - R  # dense: queries at the tail
    # window sources: ring holds [ring_beg, q_pos0), chunk holds [q_pos0, q_pos0 + R)
    kv_chunk = torch.randn((R, D), dtype = torch.half, device = device)
    ring_beg = max(0, q_pos0 - window - 5)
    ring = torch.randn((max(q_pos0 - ring_beg, 1), D), dtype = torch.half, device = device)
    win_floor = max(ring_beg, q_pos0 - max(window - 1, 0), 0)
    if not dense:
        K_pad = ((topk + 31) // 32) * 32
        indices = torch.full((R, K_pad), -1, dtype = torch.int32, device = device)
        for r in range(R):
            n = min(topk, n_keys)
            indices[r, :n] = torch.randperm(n_keys, device = device)[:n].int()
        k_len = topk

    derot_inv_freq = None
    if derot:
        derot_inv_freq = -1.0 / (10000.0 ** (torch.arange(0, D_r, 2, dtype = torch.float,
                                                          device = device) / D_r))

    def run(ns):
        return dsa_attn(
            q, pool_c, pool_r, bt, sinks = sinks,
            ring = ring if window > 0 else None, kv_chunk = kv_chunk if window > 0 else None,
            win_len = window, win_floor = win_floor, ring_beg = ring_beg,
            indices = indices, k_len = k_len,
            pool_len = n_keys, q_pos0 = q_pos0, compress_rate = m,
            derot_inv_freq = derot_inv_freq, groups = groups, n_splits = ns,
        )
    got = run(1).clone()   # out comes from the tensor cache: un-alias the two runs
    got_split = run(8).clone()
    ref = ref_attn(q, pool_c, pool_r, bt, sinks, ring, kv_chunk, window, win_floor, ring_beg,
                   indices, dense, n_keys, q_pos0, m)
    if derot:
        for r in range(R):
            ref[r, :, D_c:] = _derotate_ref(ref[r, :, D_c:].double(), derot_inv_freq,
                                            q_pos0 + r).float()
    if groups > 1:
        ref = ref.view(R, groups, (H // groups) * D).transpose(0, 1).contiguous()
    err = (got.float() - ref).abs().max().item()
    rel = err / max(ref.abs().max().item(), 1e-6)
    rel_s = (got_split.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
    tag = (f"R{R} H{H} Dc{D_c} keys{n_keys} k{topk} win{window} sinks{int(sinks_on)} "
           f"dense{int(dense)} derot{int(derot)} g{groups}")
    ok = rel < tol and rel_s < tol
    print(f"  {'PASS' if ok else 'FAIL'} attn {tag}: rel {rel:.2e} split {rel_s:.2e}")
    return ok


def check_indexer(device, R, T, H_i, D_i, seed, tol = 6e-3):
    torch.manual_seed(seed)
    q_idx = torch.randn((R, H_i, D_i), dtype = torch.half, device = device) * 0.2
    w = torch.randn((R, H_i), dtype = torch.half, device = device) * 0.2
    k_idx = torch.randn((T, D_i), dtype = torch.half, device = device) * 0.2
    m = 4
    q_pos0 = torch.randint(0, T * m, (1,)).item()
    got = dsa_indexer_scores(q_idx, w, k_idx, q_pos0, m, T).float()
    bounds = ((q_pos0 + torch.arange(R, device = device) + 1) // m).clamp(max = T)
    logits = torch.einsum("rhd,td->rht", q_idx.double(), k_idx.double())
    ref = torch.einsum("rht,rh->rt", torch.relu(logits), w.double()) \
        * (D_i ** -0.5 * H_i ** -0.5)
    ref = ref.masked_fill(
        torch.arange(T, device = device)[None, :] >= bounds[:, None].long(), -float("inf")).float()
    fin = ref > -float("inf")
    err = (got[fin] - ref[fin]).abs().max().item() if fin.any() else 0.0
    rel = err / max(ref[fin].abs().max().item(), 1e-6)
    mask_ok = bool((got[~fin] == -float("inf")).all().item()) if (~fin).any() else True
    ok = rel < tol and mask_ok
    print(f"  {'PASS' if ok else 'FAIL'} indexer R{R} T{T} H{H_i} D{D_i}: rel {rel:.2e} mask {mask_ok}")
    return ok


def check_topk(device, R, T, k, mode, seed):
    """dsa_topk vs torch.topk: identical selected set apart from boundary TIES, where any
    tie member is valid; -inf never selected; -1 padded; bitwise deterministic."""
    torch.manual_seed(seed)
    if mode == "randn":
        scores = (torch.randn((R, T), device = device) * 2).half()
    elif mode == "ties":
        # heavy ties: quantized scores, many equal values at the boundary
        scores = (torch.randint(0, 8, (R, T), device = device).half() * 0.25)
    elif mode == "sparse":
        scores = torch.full((R, T), -float("inf"), device = device, dtype = torch.half)
        for r in range(R):
            n = torch.randint(1, max(k // 2, 2), (1,)).item()
            idx = torch.randperm(T)[:n]
            scores[r, idx] = torch.randn(n, device = device).half()
    K_pad = -(-k // 32) * 32
    out = torch.empty((R, K_pad), dtype = torch.int32, device = device)
    ext.dsa_topk(scores, out, k, None, 0)
    out2 = torch.empty_like(out)
    ext.dsa_topk(scores, out2, k, None, 0)
    det = torch.equal(out, out2)

    ok = det
    for r in range(R):
        sel = out[r][out[r] >= 0].long()
        fin = (scores[r] > -float("inf"))
        n_fin = int(fin.sum())
        expect_n = min(k, n_fin)
        if sel.numel() != expect_n or sel.unique().numel() != sel.numel():
            ok = False; break
        if (scores[r, sel] == -float("inf")).any():
            ok = False; break
        if expect_n > 0:
            ref = torch.topk(scores[r].float(), expect_n)
            vstar = ref.values.min()
            # every selected score must be >= vstar; every strictly-above-vstar entry selected
            if (scores[r, sel].float() < vstar).any():
                ok = False; break
            above = torch.nonzero(scores[r].float() > vstar).flatten()
            if not torch.isin(above, sel).all():
                ok = False; break
        if (out[r][expect_n:] != -1).any():
            ok = False; break
    print(f"  {'PASS' if ok else 'FAIL'} topk R{R} T{T} k{k} {mode}: det {det}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default = "cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    ok = True

    # R, H, D_c, D_r, n_keys, topk, window, sinks, dense, m
    cases = [
        (7, 64, 448, 64, 2048, 512, 128, True, False, 4),    # V4 CSA shape
        (5, 64, 448, 64, 300, 512, 128, True, False, 4),     # topk > keys
        (9, 64, 448, 64, 40, 0, 0, True, True, 128),         # V4 HCA (dense, no window idx... )
        (6, 64, 448, 64, 64, 0, 128, True, True, 128),       # HCA with window
        (4, 8, 448, 64, 500, 64, 16, True, False, 4),        # few heads
        (3, 128, 512, 64, 4096, 2048, 0, False, False, 1),   # V3.2 shape: no window, no sinks
        (2, 16, 256, 32, 700, 96, 8, False, False, 1),       # odd dims
        (1, 64, 448, 64, 1024, 512, 128, True, False, 4),    # decode shape
        (40, 64, 448, 64, 40, 8, 16, True, False, 1),        # q_pos0 = 0: window floor bites
        (24, 8, 448, 64, 24, 4, 128, True, False, 1),        # window > sequence, floor at 0
    ]
    for i, c in enumerate(cases):
        ok &= check_attn(device, *c, seed = 100 + i)

    # Fused epilogue: eq. 26 de-rotation and/or group-major output store
    epi_cases = [
        ((7, 64, 448, 64, 2048, 512, 128, True, False, 4), True, 8),    # V4 CSA, full fusion
        ((9, 64, 448, 64, 40, 0, 0, True, True, 128), True, 8),         # V4 HCA, full fusion
        ((5, 64, 448, 64, 300, 512, 128, True, False, 4), True, 1),     # derot only
        ((6, 64, 448, 64, 64, 0, 128, True, True, 128), False, 8),      # groups only
        ((1, 64, 448, 64, 1024, 512, 128, True, False, 4), True, 8),    # decode shape
        ((2, 16, 256, 32, 700, 96, 8, False, False, 1), True, 4),       # odd dims
    ]
    for i, (c, dr, g) in enumerate(epi_cases):
        ok &= check_attn(device, *c, seed = 300 + i, derot = dr, groups = g)

    for i, (R, T, H_i, D_i) in enumerate([
        (64, 512, 64, 128), (2048, 512, 64, 128), (7, 33, 4, 16), (128, 4096, 64, 128),
        (1, 512, 64, 128), (1, 13, 64, 128), (2, 4096, 64, 128), (3, 33, 4, 16),
    ]):
        ok &= check_indexer(device, R, T, H_i, D_i, seed = 200 + i)

    for i, (R, T, k, mode) in enumerate([
        (1, 2048, 512, "randn"), (4, 8192, 512, "randn"), (16, 700, 512, "randn"),
        (1, 4096, 512, "ties"), (3, 1000, 512, "ties"),
        (2, 2048, 512, "sparse"), (1, 40, 3, "randn"), (5, 33, 3, "ties"),
        (1, 32768, 512, "randn"),
    ]):
        ok &= check_topk(device, R, T, k, mode, seed = 500 + i)

    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
