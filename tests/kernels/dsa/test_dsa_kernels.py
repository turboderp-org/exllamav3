"""
Production DSA kernels (modules/attention_fn/dsa_triton.py) against fp64 torch references: dsa_attn over the
constexpr toggle matrix (window on/off, sinks on/off, gathered vs dense pool, D_c padding (448), fused de-rotation /
group-major epilogue, packed-quantized pools, non-causal image chunks, packed-pool prefill staging, split-kv),
dsa_indexer_scores with causal bounds, and ext.dsa_topk against torch.topk (tie-aware, deterministic). dsa_attn keeps
only decode-class workspaces in the global tensor cache: prefill-sized calls allocate their output and split
workspaces per call.
"""
import pytest
import torch

import exllamav3.modules.attention_fn.dsa_triton as dsa_triton
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.attention_fn.dsa_triton import dsa_attn, dsa_indexer_scores
from exllamav3.util.tensor import g_tensor_cache
from testlib.attention import quant_roundtrip

PAGE_SIZE = 256


def ref_attn(q, pool_c, pool_r, bt, sinks, ring, kv_chunk, window, win_floor, ring_beg,
             indices, dense, pool_len, q_pos0, m, nc_chunk = False):
    """Per-row fp64 reference: the row's window (ring / chunk rows above win_floor) plus its pool entries (top-k
    indices, or the causal dense prefix), softmax with optional per-head sink logits"""
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
            if nc_chunk:
                # Image chunk: every chunk row (bidirectional) + this row's own history window
                cand = list(range(q_pos0, q_pos0 + R)) + list(range(q_abs - window + 1, q_pos0))
            else:
                cand = [q_abs - j for j in range(window)]
            for a in cand:
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
            continue  # no keys (sink-only softmax included): zero output
        kv = torch.cat(rows, dim = 0)                       # (n, D)
        s = (q[r].double() @ kv.T) * scale                  # (H, n)
        if sinks is not None:
            s = torch.cat([s, sinks.double().unsqueeze(1)], dim = 1)
            p = torch.softmax(s, dim = -1)[:, :-1]
        else:
            p = torch.softmax(s, dim = -1)
        out[r] = p @ kv
    return out.float()


def derotate_ref(o_r, inv_freq, pos):
    """GPT-J pair rotation of (H, D_r) by angle pos * inv_freq, fp64"""
    theta = inv_freq.double() * pos
    cos, sin = theta.cos(), theta.sin()
    e, o = o_r[..., 0::2], o_r[..., 1::2]
    return torch.stack((e * cos - o * sin, o * cos + e * sin), dim = -1).flatten(-2)


def check_attn(device, monkeypatch, R, H, D_c, D_r, n_keys, topk, window, sinks_on, dense, m, seed, tol = 6e-3,
               derot = False, groups = 1, qc_bits = 0, nc_chunk = False, single_bt = False):
    torch.manual_seed(seed)
    D = D_c + D_r
    pages = (n_keys + PAGE_SIZE - 1) // PAGE_SIZE
    pool_c = torch.randn((pages, PAGE_SIZE, D_c), dtype = torch.half, device = device)
    pool_r = torch.randn((pages, PAGE_SIZE, D_r), dtype = torch.half, device = device)
    # Packed-quantized latent pool: the kernel dequantizes online; the reference reads the SAME values
    # dequantized by the CUDA kernel, so the tolerance tests the loader/rotation path, not the quantizer
    qc = None
    pc_arg = pool_c
    if qc_bits:
        pq, ps, deq = quant_roundtrip(pool_c.reshape(pages * PAGE_SIZE, D_c), qc_bits)
        pool_c = deq.view(pages, PAGE_SIZE, D_c)
        pc_arg = pq.view(pages, PAGE_SIZE, -1)
        qc = (ps, qc_bits)
    perm = torch.randperm(pages, device = device, dtype = torch.int32)
    bt = perm.unsqueeze(0).expand(R, pages).contiguous()
    q = torch.randn((R, H, D), dtype = torch.half, device = device)
    sinks = (torch.randn(H, device = device) * 2.0 + 4.0) if sinks_on else None

    indices = None
    k_len = 0
    q_pos0 = n_keys * m - R  # dense: queries at the tail
    # Window sources: ring holds [ring_beg, q_pos0), chunk holds [q_pos0, q_pos0 + R)
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
        derot_inv_freq = -1.0 / (10000.0 ** (torch.arange(0, D_r, 2, dtype = torch.float, device = device) / D_r))

    # Single-row (stride-0) block table, as the bsz-1 sparse prefill passes it: the packed-pool staging path keys
    # on it (and on pool_len = the referenced context)
    bt_k = bt[:1].contiguous() if single_bt else bt

    def run(ns):
        return dsa_attn(
            q, pc_arg, pool_r, bt_k, sinks = sinks,
            ring = ring if window > 0 else None, kv_chunk = kv_chunk if window > 0 else None,
            win_len = window, win_floor = win_floor, ring_beg = ring_beg,
            indices = indices, k_len = k_len,
            pool_len = n_keys, q_pos0 = q_pos0, compress_rate = m,
            derot_inv_freq = derot_inv_freq, groups = groups, n_splits = ns, qc = qc,
            nc_chunk = nc_chunk,
        )

    with monkeypatch.context() as mp:
        mp.setattr(dsa_triton, "_dsa_qc_stage", True)
        got = run(1).clone()   # out comes from the tensor cache: un-alias the two runs
        got_split = run(8).clone()
    if single_bt and qc_bits:
        # Also the online-dequant path over the same inputs (staging forced off)
        with monkeypatch.context() as mp:
            mp.setattr(dsa_triton, "_dsa_qc_stage", False)
            got_split = run(1).clone()

    ref = ref_attn(q, pool_c, pool_r, bt, sinks, ring, kv_chunk, window, win_floor, ring_beg,
                   indices, dense, n_keys, q_pos0, m, nc_chunk = nc_chunk)
    if derot:
        for r in range(R):
            ref[r, :, D_c:] = derotate_ref(ref[r, :, D_c:].double(), derot_inv_freq, q_pos0 + r).float()
    if groups > 1:
        ref = ref.view(R, groups, (H // groups) * D).transpose(0, 1).contiguous()
    scale = max(ref.abs().max().item(), 1e-6)
    rel = (got.float() - ref).abs().max().item() / scale
    rel_s = (got_split.float() - ref).abs().max().item() / scale
    assert rel < tol, f"single split: rel err {rel:.2e}"
    assert rel_s < tol, f"{'online dequant' if single_bt and qc_bits else '8 splits'}: rel err {rel_s:.2e}"


def _id(c, derot = False, groups = 1, qc_bits = 0):
    R, H, D_c, D_r, n_keys, topk, window, sinks, dense, m = c
    return (f"R{R}-H{H}-Dc{D_c}-keys{n_keys}-k{topk}-win{window}-sinks{int(sinks)}-dense{int(dense)}-m{m}"
            f"-derot{int(derot)}-g{groups}-qc{qc_bits}")


CSA = (7, 64, 448, 64, 2048, 512, 128, True, False, 4)       # V4 CSA shape
HCA = (9, 64, 448, 64, 40, 0, 0, True, True, 128)            # V4 HCA (dense pool, no window)
HCA_WIN = (6, 64, 448, 64, 64, 0, 128, True, True, 128)      # HCA with window
V32 = (3, 128, 512, 64, 4096, 2048, 0, False, False, 1)      # V3.2 shape: no window, no sinks
ODD = (2, 16, 256, 32, 700, 96, 8, False, False, 1)          # odd dims
DECODE = (1, 64, 448, 64, 1024, 512, 128, True, False, 4)    # decode shape

# R, H, D_c, D_r, n_keys, topk, window, sinks, dense, m
ATTN_CASES = [
    CSA,
    (5, 64, 448, 64, 300, 512, 128, True, False, 4),     # topk > keys
    HCA,
    HCA_WIN,
    (4, 8, 448, 64, 500, 64, 16, True, False, 4),        # few heads
    V32,
    ODD,
    DECODE,
    (40, 64, 448, 64, 40, 8, 16, True, False, 1),        # q_pos0 = 0: window floor bites
    (24, 8, 448, 64, 24, 4, 128, True, False, 1),        # window > sequence, floor at 0
]


@pytest.mark.parametrize("i,case", [pytest.param(i, c, id = _id(c)) for i, c in enumerate(ATTN_CASES)])
def test_attn(device, monkeypatch, i, case):
    check_attn(device, monkeypatch, *case, seed = 100 + i)


# Fused epilogue: eq. 26 de-rotation and/or group-major output store
EPI_CASES = [
    (CSA, True, 8),                                       # V4 CSA, full fusion
    (HCA, True, 8),                                       # V4 HCA, full fusion
    ((5, 64, 448, 64, 300, 512, 128, True, False, 4), True, 1),   # derot only
    (HCA_WIN, False, 8),                                  # groups only
    (DECODE, True, 8),
    (ODD, True, 4),
]


@pytest.mark.parametrize("i,case,derot,groups",
                         [pytest.param(i, c, d, g, id = _id(c, d, g)) for i, (c, d, g) in enumerate(EPI_CASES)])
def test_attn_epilogue(device, monkeypatch, i, case, derot, groups):
    check_attn(device, monkeypatch, *case, seed = 300 + i, derot = derot, groups = groups)


# Packed-quantized pools (QC): online dequant in the pool phase, q / window tiles rotated into the H32 domain,
# output rotated back in the epilogue or the combine. Covers the padded loader (D_c 448), the pow2 widths (512 /
# 256), window + sinks + derot/groups (V4), and the windowless gathered form (V3.2 / GLM)
QC_CASES = [
    (CSA, False, 1, 8),
    (CSA, True, 8, 4),
    ((5, 64, 448, 64, 300, 512, 128, True, False, 4), True, 1, 3),
    (HCA, True, 8, 8),
    (HCA_WIN, False, 8, 6),
    (V32, False, 1, 8),
    (V32, False, 1, 4),
    (V32, False, 1, 2),
    (ODD, True, 4, 5),
    (DECODE, True, 8, 7),
    ((24, 8, 448, 64, 24, 4, 128, True, False, 1), False, 1, 8),
]


@pytest.mark.parametrize("i,case,derot,groups,bits",
                         [pytest.param(i, c, d, g, b, id = _id(c, d, g, b)) for i, (c, d, g, b) in enumerate(QC_CASES)])
def test_attn_qc(device, monkeypatch, i, case, derot, groups, bits):
    check_attn(device, monkeypatch, *case, seed = 400 + i, derot = derot, groups = groups, qc_bits = bits,
               tol = 1.2e-2)


# Non-causal image chunk (V4 vision): every row sees the whole chunk plus its own window into the ring; chunks
# longer than the window, floor clipping, dense HCA pools, QC
NC_CASES = [
    ((200, 64, 448, 64, 2048, 512, 128, True, False, 4), True, 8, 0),    # CSA, chunk > window
    ((390, 64, 448, 64, 40, 0, 128, True, True, 128), True, 8, 0),       # HCA dense, long chunk
    ((64, 8, 448, 64, 24, 4, 128, True, False, 1), False, 1, 0),         # window floor at 0
    ((16, 16, 256, 32, 700, 96, 8, False, False, 1), True, 4, 0),        # odd dims, small window
    ((200, 64, 448, 64, 2048, 512, 128, True, False, 4), True, 8, 4),    # QC pools
    ((130, 64, 448, 64, 40, 0, 128, True, True, 128), False, 8, 6),
]


@pytest.mark.parametrize("i,case,derot,groups,bits",
                         [pytest.param(i, c, d, g, b, id = _id(c, d, g, b)) for i, (c, d, g, b) in enumerate(NC_CASES)])
def test_attn_nc_chunk(device, monkeypatch, i, case, derot, groups, bits):
    check_attn(device, monkeypatch, *case, seed = 600 + i, derot = derot, groups = groups, qc_bits = bits,
               tol = 1.2e-2 if bits else 6e-3, nc_chunk = True)


# Packed-pool prefill staging (single-row block table, R >= 64, pool_len = context): the staged fp16 path and the
# online path must both match the reference
STAGE_CASES = [
    ((200, 64, 448, 64, 2048, 512, 0, False, False, 1), False, 1, 4),    # V3.2/GLM gathered
    ((200, 64, 448, 64, 2048, 512, 0, False, False, 1), True, 8, 6),
    ((130, 64, 448, 64, 40, 0, 128, True, True, 128), True, 8, 6),      # V4 HCA dense + window
    ((64, 16, 256, 32, 700, 96, 8, False, False, 1), True, 4, 3),       # odd dims, min R
    ((4096, 64, 448, 64, 8192, 2048, 0, False, False, 1), True, 8, 6),  # full-size chunk
]


@pytest.mark.parametrize("i,case,derot,groups,bits",
                         [pytest.param(i, c, d, g, b, id = _id(c, d, g, b)) for i, (c, d, g, b) in enumerate(STAGE_CASES)])
def test_attn_qc_staging(device, monkeypatch, i, case, derot, groups, bits):
    check_attn(device, monkeypatch, *case, seed = 700 + i, derot = derot, groups = groups, qc_bits = bits,
               tol = 1.2e-2, single_bt = True)


INDEXER_CASES = [
    (64, 512, 64, 128), (2048, 512, 64, 128), (7, 33, 4, 16), (128, 4096, 64, 128),
    (1, 512, 64, 128), (1, 13, 64, 128), (2, 4096, 64, 128), (3, 33, 4, 16),
]


@pytest.mark.parametrize("i,R,T,H_i,D_i", [pytest.param(i, *c, id = "R{}-T{}-H{}-D{}".format(*c))
                                           for i, c in enumerate(INDEXER_CASES)])
def test_indexer_scores(device, i, R, T, H_i, D_i):
    torch.manual_seed(200 + i)
    q_idx = torch.randn((R, H_i, D_i), dtype = torch.half, device = device) * 0.2
    w = torch.randn((R, H_i), dtype = torch.half, device = device) * 0.2
    k_idx = torch.randn((T, D_i), dtype = torch.half, device = device) * 0.2
    m = 4
    q_pos0 = torch.randint(0, T * m, (1,)).item()
    got = dsa_indexer_scores(q_idx, w, k_idx, q_pos0, m, T).float()
    bounds = ((q_pos0 + torch.arange(R, device = device) + 1) // m).clamp(max = T)
    logits = torch.einsum("rhd,td->rht", q_idx.double(), k_idx.double())
    ref = torch.einsum("rht,rh->rt", torch.relu(logits), w.double()) * (D_i ** -0.5 * H_i ** -0.5)
    ref = ref.masked_fill(torch.arange(T, device = device)[None, :] >= bounds[:, None].long(), -float("inf")).float()
    fin = ref > -float("inf")
    if fin.any():
        rel = (got[fin] - ref[fin]).abs().max().item() / max(ref[fin].abs().max().item(), 1e-6)
        assert rel < 6e-3, f"rel err {rel:.2e}"
    assert (got[~fin] == -float("inf")).all(), "positions past the causal bound must score -inf"


TOPK_CASES = [
    (1, 2048, 512, "randn"), (4, 8192, 512, "randn"), (16, 700, 512, "randn"),
    (1, 4096, 512, "ties"), (3, 1000, 512, "ties"),
    (2, 2048, 512, "sparse"), (1, 40, 3, "randn"), (5, 33, 3, "ties"),
    (1, 32768, 512, "randn"),
]


@pytest.mark.parametrize("i,R,T,k,mode", [pytest.param(i, *c, id = "R{}-T{}-k{}-{}".format(*c))
                                          for i, c in enumerate(TOPK_CASES)])
def test_topk(device, i, R, T, k, mode):
    """dsa_topk vs torch.topk: identical selected set apart from boundary TIES, where any tie member is valid;
    -inf never selected; -1 padded; bitwise deterministic"""
    torch.manual_seed(500 + i)
    if mode == "randn":
        scores = (torch.randn((R, T), device = device) * 2).half()
    elif mode == "ties":
        # Heavy ties: quantized scores, many equal values at the boundary
        scores = torch.randint(0, 8, (R, T), device = device).half() * 0.25
    else:
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
    assert torch.equal(out, out2), "not deterministic"

    for r in range(R):
        sel = out[r][out[r] >= 0].long()
        n_fin = int((scores[r] > -float("inf")).sum())
        expect_n = min(k, n_fin)
        assert sel.numel() == expect_n, f"row {r}: {sel.numel()} selected, expected {expect_n}"
        assert sel.unique().numel() == sel.numel(), f"row {r}: duplicate indices"
        assert not (scores[r, sel] == -float("inf")).any(), f"row {r}: -inf selected"
        if expect_n > 0:
            vstar = torch.topk(scores[r].float(), expect_n).values.min()
            # Every selected score must be >= vstar; every strictly-above-vstar entry selected
            assert not (scores[r, sel].float() < vstar).any(), f"row {r}: selected below the k-th score"
            above = torch.nonzero(scores[r].float() > vstar).flatten()
            assert torch.isin(above, sel).all(), f"row {r}: missed an entry above the k-th score"
        assert (out[r][expect_n:] == -1).all(), f"row {r}: tail not -1 padded"


@pytest.mark.parametrize("R, n_splits, kept", [(1, 16, True), (8, 16, True), (2048, 8, False), (2048, 1, False)])
@torch.inference_mode()
def test_workspaces_kept_only_for_decode_rows(device, monkeypatch, R, n_splits, kept):
    """g_tensor_cache holds decode-class buffers only: a prefill-sized call (split or not) must not leave its
    output or split workspaces there, a decode-class call keeps them"""
    H, D_c, D_r, n_keys = 64, 448, 64, 1024
    pages = n_keys // PAGE_SIZE
    q = torch.randn((R, H, D_c + D_r), dtype = torch.half, device = device)
    pool_c = torch.randn((pages, PAGE_SIZE, D_c), dtype = torch.half, device = device)
    pool_r = torch.randn((pages, PAGE_SIZE, D_r), dtype = torch.half, device = device)
    bt = torch.arange(pages, dtype = torch.int32, device = device).unsqueeze(0).expand(R, pages).contiguous()
    monkeypatch.setattr(g_tensor_cache, "cache", {})
    dsa_attn(q, pool_c, pool_r, bt, pool_len = n_keys, q_pos0 = n_keys * 4 - R, compress_rate = 4, n_splits = n_splits)
    kept_tags = {k.rsplit("/", 1)[-1] for k in g_tensor_cache.cache}
    if kept:
        assert {"dsa_out", "dsa_ws_ml", "dsa_ws_acc"} <= kept_tags, f"decode-class call kept only {sorted(kept_tags)}"
    else:
        assert not kept_tags, f"prefill-sized call left kept workspaces: {sorted(kept_tags)}"
