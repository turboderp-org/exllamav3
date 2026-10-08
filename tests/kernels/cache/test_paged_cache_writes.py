"""
Paged-cache placement kernels against torch index-arithmetic references (exact: all of them copy or decode values
without accumulation across tokens):

- paged_kv_cache_update(k, v, k_cache, v_cache, block_table, cache_seqlens): new k / v (B, S, H, D) fp16 land at
  logical positions cache_seqlens[b] + s of their sequence, i.e. page block_table[b, pos // 256], row pos % 256,
  bit-exact; every other cache row is untouched; empty inputs are no-ops; D % 8, page size 256, dtypes and k / v
  shape mismatches are rejected. Includes a size past the kernel's 65535-block grid cap (grid-stride loop).
- dspark_write_rows(rows, kv, block_table, cache_seqlens): (bsz, s, w) fp16 rows into a (pages, 256, w) plane at
  the same paged positions (any width, scalar copies).
- dsv4_ring_append(kv, ring, pos, ring_beg, slot_ids): rows land at ring[pos - ring_beg + j]; rows that fall
  outside [0, ring_rows) are dropped; with slot_ids, job j's kv rows (seq = rows / jobs each) go to ring slot
  slot_ids[j] and other slots are untouched.
- dequant_cache_paged_window(k_in, k_scales, k_out, v_in, v_scales, v_out, cache_seqlens, block_table, 256,
  bonus_len, 0.0): the referenced window of a packed quantized cache (cache_seqlens[b] + bonus_len tokens of each
  sequence) decoded into a compact scratch where page p of row b lands at scratch page b * pages_per_seq + p;
  scratch rows outside the window are untouched. The reference decodes the packed format independently (power-of-
  two bit planes, midpoint grid, fp16 group scale / sqrt(32), inverse unnormalized Sylvester Hadamard over each
  32-group) in float64, over every K/V width pair 2..8 and head layouts with a partial last 4-group chunk.
"""
import math

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

PAGE = 256


def sentinel_like(shape, device, dtype = torch.half):
    # Distinct non-NaN pattern, so untouched rows compare exactly
    n = math.prod(shape)
    return (torch.arange(n, device = device, dtype = torch.float32) % 977 - 1000.0).to(dtype).view(shape)


def paged_positions(block_table, cache_seqlens, S):
    """(B, S) physical page and in-page row of logical positions cache_seqlens[b] + s"""
    bt = block_table.long().cpu()
    pos = cache_seqlens.long().cpu()[:, None] + torch.arange(S)[None, :]
    page = torch.gather(bt, 1, pos // PAGE)
    return page, pos % PAGE


def make_block_table(B, pages_per_seq, num_pages, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(num_pages, generator = g)[: B * pages_per_seq].view(B, pages_per_seq).int()


# ---------------------------------------------------------------------------------------------------------------
# paged_kv_cache_update

# (B, S, H, D, seqlens, pages_per_seq)
KV_CASES = [
    (1, 1, 8, 128, [0], 2),
    (1, 1, 8, 128, [255], 2),             # last row of a page
    (1, 1, 8, 128, [256], 2),             # first row of the next page
    (3, 7, 2, 64, [0, 250, 511], 3),       # crosses a page boundary in row 1
    (2, 300, 1, 256, [10, 200], 3),        # chunk spanning two boundaries
    (4, 5, 4, 8, [3, 0, 600, 255], 4),    # smallest head dim
    (2, 16, 8, 512, [100, 700], 4),
    (1, 513, 3, 96, [1], 3),              # non-pow2 head dim, odd head count
]


@pytest.mark.parametrize("B,S,H,D,seqlens,pps", KV_CASES,
                         ids = [f"B{c[0]}-S{c[1]}-H{c[2]}-D{c[3]}" for c in KV_CASES])
@torch.inference_mode()
def test_paged_kv_cache_update(device, B, S, H, D, seqlens, pps):
    torch.manual_seed(0)
    num_pages = B * pps + 3
    k = torch.randn((B, S, H, D), dtype = torch.half, device = device)
    v = torch.randn((B, S, H, D), dtype = torch.half, device = device)
    k_cache = sentinel_like((num_pages, PAGE, H, D), device)
    v_cache = -sentinel_like((num_pages, PAGE, H, D), device)
    block_table = make_block_table(B, pps, num_pages, 1).to(device)
    cache_seqlens = torch.tensor(seqlens, dtype = torch.int32, device = device)

    ref_k, ref_v = k_cache.clone(), v_cache.clone()
    page, row = paged_positions(block_table, cache_seqlens, S)
    ref_k[page.to(device), row.to(device)] = k
    ref_v[page.to(device), row.to(device)] = v

    ext.paged_kv_cache_update(k, v, k_cache, v_cache, block_table, cache_seqlens)
    assert torch.equal(k_cache.view(torch.int16), ref_k.view(torch.int16))
    assert torch.equal(v_cache.view(torch.int16), ref_v.view(torch.int16))
    # Inputs are read-only
    assert block_table.dtype == torch.int32 and cache_seqlens.tolist() == seqlens


@torch.inference_mode()
def test_paged_kv_cache_update_grid_stride(device):
    """More 16-byte vectors than 65535 blocks x 256 threads: the grid-stride loop must cover the remainder"""
    B, S, H, D = 1, 65536 + 4096, 8, 256
    assert B * S * H * D // 8 > 65535 * 256
    pps = -(-(S + 100) // PAGE)
    num_pages = pps + 1
    k = torch.randn((B, S, H, D), dtype = torch.half, device = device)
    v = torch.randn((B, S, H, D), dtype = torch.half, device = device)
    k_cache = torch.zeros((num_pages, PAGE, H, D), dtype = torch.half, device = device)
    v_cache = torch.zeros_like(k_cache)
    block_table = make_block_table(B, pps, num_pages, 2).to(device)
    cache_seqlens = torch.tensor([100], dtype = torch.int32, device = device)
    ext.paged_kv_cache_update(k, v, k_cache, v_cache, block_table, cache_seqlens)
    page, row = paged_positions(block_table, cache_seqlens, S)
    assert torch.equal(k_cache[page.to(device), row.to(device)], k)
    assert torch.equal(v_cache[page.to(device), row.to(device)], v)
    # Positions before cache_seqlens and the spare page stay zero
    assert k_cache[block_table[0, 0].item(), :100].abs().max().item() == 0
    used = set(block_table.flatten().tolist())
    for p in range(num_pages):
        if p not in used:
            assert k_cache[p].abs().max().item() == 0


@torch.inference_mode()
def test_paged_kv_cache_update_empty(device):
    k_cache = sentinel_like((2, PAGE, 2, 64), device)
    v_cache = sentinel_like((2, PAGE, 2, 64), device)
    ref = k_cache.clone()
    bt = torch.zeros((1, 2), dtype = torch.int32, device = device)
    sl = torch.zeros((1,), dtype = torch.int32, device = device)
    e = torch.empty((1, 0, 2, 64), dtype = torch.half, device = device)
    ext.paged_kv_cache_update(e, e, k_cache, v_cache, bt, sl)
    assert torch.equal(k_cache, ref)


@torch.inference_mode()
def test_paged_kv_cache_update_rejects(device):
    bt = torch.zeros((1, 2), dtype = torch.int32, device = device)
    sl = torch.zeros((1,), dtype = torch.int32, device = device)
    k = torch.zeros((1, 1, 2, 12), dtype = torch.half, device = device)
    cache = torch.zeros((2, PAGE, 2, 12), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "divisible by 8"):
        ext.paged_kv_cache_update(k, k, cache, cache, bt, sl)
    k = torch.zeros((1, 1, 2, 64), dtype = torch.half, device = device)
    small = torch.zeros((4, 128, 2, 64), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "page_size == 256"):
        ext.paged_kv_cache_update(k, k, small, small, bt, sl)
    cache = torch.zeros((2, PAGE, 2, 64), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError):
        ext.paged_kv_cache_update(k.float(), k.float(), cache, cache, bt, sl)
    with pytest.raises(RuntimeError):
        ext.paged_kv_cache_update(k, k, cache, cache, bt.long(), sl)
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.paged_kv_cache_update(k, torch.zeros((1, 2, 2, 64), dtype = torch.half, device = device),
                                  cache, cache, bt, sl)


# ---------------------------------------------------------------------------------------------------------------
# dspark_write_rows

@pytest.mark.parametrize("bsz,s,w,seqlens", [
    (1, 1, 512, [0]), (1, 4, 512, [254]), (4, 3, 512, [0, 255, 256, 700]),
    (2, 300, 64, [5, 200]), (3, 2, 7, [1, 300, 511]), (1, 600, 576, [0]),
])
@torch.inference_mode()
def test_dspark_write_rows(device, bsz, s, w, seqlens):
    torch.manual_seed(1)
    pps = 4
    num_pages = bsz * pps + 2
    rows = torch.randn((bsz, s, w), dtype = torch.half, device = device)
    kv = sentinel_like((num_pages, PAGE, w), device)
    block_table = make_block_table(bsz, pps, num_pages, 3).to(device)
    cache_seqlens = torch.tensor(seqlens, dtype = torch.int32, device = device)
    ref = kv.clone()
    page, row = paged_positions(block_table, cache_seqlens, s)
    ref[page.to(device), row.to(device)] = rows
    ext.dspark_write_rows(rows, kv, block_table, cache_seqlens)
    assert torch.equal(kv, ref)


@torch.inference_mode()
def test_dspark_write_rows_rejects(device):
    bt = torch.zeros((1, 1), dtype = torch.int32, device = device)
    sl = torch.zeros((1,), dtype = torch.int32, device = device)
    kv = torch.zeros((1, PAGE, 8), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError):
        ext.dspark_write_rows(torch.zeros((1, 1, 8), device = device), kv, bt, sl)
    with pytest.raises(RuntimeError):
        ext.dspark_write_rows(torch.zeros((1, 1, 8), dtype = torch.half, device = device), kv, bt.long(), sl)


# ---------------------------------------------------------------------------------------------------------------
# dsv4_ring_append

@pytest.mark.parametrize("seq,D,ring_rows,pos,beg", [
    (1, 512, 128, 10, 0),
    (1, 512, 128, 127, 0),          # last ring row
    (5, 512, 128, 300, 200),        # offset inside the ring
    (16, 64, 128, 120, 0),          # 8 rows past the end are dropped
    (4, 64, 128, 1000, 1002),       # first 2 rows fall before ring_beg and are dropped
    (3, 7, 16, 3, 0),               # odd width
])
@torch.inference_mode()
def test_dsv4_ring_append(device, seq, D, ring_rows, pos, beg):
    torch.manual_seed(2)
    kv = torch.randn((seq, D), dtype = torch.half, device = device)
    ring = sentinel_like((ring_rows, D), device)
    ref = ring.clone()
    for j in range(seq):
        r = pos - beg + j
        if 0 <= r < ring_rows:
            ref[r] = kv[j]
    ext.dsv4_ring_append(kv, ring, torch.tensor([pos], dtype = torch.int32, device = device),
                         torch.tensor([beg], dtype = torch.int32, device = device), None)
    assert torch.equal(ring, ref)


@pytest.mark.parametrize("jobs,seq,slots", [(1, 1, 3), (4, 1, 6), (3, 4, 5), (8, 2, 8)])
@torch.inference_mode()
def test_dsv4_ring_append_slots(device, jobs, seq, slots):
    torch.manual_seed(3)
    D, ring_rows = 512, 128
    g = torch.Generator().manual_seed(4)
    slot_ids = torch.randperm(slots, generator = g)[:jobs].int()
    beg = torch.randint(0, 1000, (jobs,), generator = g).int()
    off = torch.randint(-2, ring_rows - seq + 3, (jobs,), generator = g).int()   # incl. partially outside
    pos = beg + off
    kv = torch.randn((jobs * seq, D), dtype = torch.half, device = device)
    ring = sentinel_like((slots, ring_rows, D), device)
    ref = ring.clone()
    for j in range(jobs):
        for i in range(seq):
            r = int(pos[j] - beg[j]) + i
            if 0 <= r < ring_rows:
                ref[int(slot_ids[j]), r] = kv[j * seq + i]
    ext.dsv4_ring_append(kv, ring, pos.to(device), beg.to(device), slot_ids.to(device))
    assert torch.equal(ring, ref)


@torch.inference_mode()
def test_dsv4_ring_append_rejects(device):
    i1 = torch.zeros((1,), dtype = torch.int32, device = device)
    ring = torch.zeros((16, 64), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.dsv4_ring_append(torch.zeros((1, 32), dtype = torch.half, device = device), ring, i1, i1, None)
    with pytest.raises(RuntimeError):
        ext.dsv4_ring_append(torch.zeros((1, 64), device = device), ring, i1, i1, None)
    with pytest.raises(RuntimeError):
        ext.dsv4_ring_append(torch.zeros((1, 64), dtype = torch.half, device = device), ring, i1.long(), i1, None)


# ---------------------------------------------------------------------------------------------------------------
# dequant_cache_paged_window

def rand_packed(num_pages, n_kv, hd, bits, device):
    """Random packed words (any bit pattern is a valid payload) and group scales in 0.1..1.1"""
    dim = n_kv * hd
    words = torch.randint(-2 ** 31, 2 ** 31, (num_pages, PAGE, dim // 32 * bits), dtype = torch.int64)
    scales = torch.rand((num_pages, PAGE, dim // 32)) + 0.1
    return words.to(torch.int32).to(device), scales.half().to(device)


def sylvester(n):
    h = torch.ones((1, 1), dtype = torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def ref_dequant_rows(words: torch.Tensor, scales: torch.Tensor, bits: int) -> torch.Tensor:
    """words (N, G * bits) int32, scales (N, G) fp16 -> (N, G * 32) float64 decoded values"""
    N = words.shape[0]
    G = scales.shape[1]
    w = words.cpu().view(N, G, bits).to(torch.int64) & 0xffffffff
    e = torch.arange(32)
    q = torch.zeros((N, G, 32), dtype = torch.int64)
    base = 0
    for W in (8, 4, 2, 1):
        if not bits & W:
            continue
        bitpos = e * W                                   # element e's field starts at bit e * W of the plane
        word = w[:, :, base + bitpos // 32]              # (N, G, 32)
        plane = (word >> (bitpos % 32)) & ((1 << W) - 1)
        q = (q << W) | plane
        base += W
    m = 1 << (bits - 1)
    x = (q.double() - (m - 0.5)) * (scales.cpu().double()[:, :, None] / math.sqrt(32) / m)
    y = x @ sylvester(32).T
    return y.view(N, G * 32)


def check_decoded(out: torch.Tensor, ref: torch.Tensor):
    # Kernel: fp32 dequant and a 5-stage fp32 butterfly (relative error a few 2^-24 of sums of |x| <= ~6),
    # then one fp16 rounding: within one fp16 ulp of the float64 value (rtol 2^-10), plus an absolute term for
    # the fp32 error near zero
    torch.testing.assert_close(out.double().cpu(), ref, rtol = 2 ** -10, atol = 1e-5)


# (bsz, pages_per_seq, n_kv, head_dim, seqlens, bonus, k_bits, v_bits, out_4d)
WIN_CASES = [
    (1, 2, 8, 128, [100], 0, 4, 4, True),
    (1, 2, 8, 128, [256], 0, 8, 8, True),          # window ends on a page boundary
    (2, 3, 2, 64, [300, 0], 17, 2, 3, True),       # bonus rows past the past length; row 1 bonus only
    (3, 2, 1, 96, [5, 255, 400], 1, 5, 6, False),  # 3 groups per token: partial 4-group chunk
    (2, 4, 2, 96, [700, 1000], 24, 7, 8, True),    # 6 groups: second chunk half active
    (1, 1, 1, 32, [0], 1, 6, 2, False),            # single group, single token
    (2, 2, 4, 256, [511, 128], 1, 3, 5, True),     # window = whole span in row 0
]


def _win_id(c):
    return f"b{c[0]}-p{c[1]}-h{c[2]}x{c[3]}-k{c[6]}v{c[7]}-bonus{c[5]}-{'4d' if c[8] else '3d'}"


@pytest.mark.parametrize("bsz,pps,n_kv,hd,seqlens,bonus,kb,vb,out_4d", WIN_CASES, ids = [_win_id(c) for c in WIN_CASES])
@torch.inference_mode()
def test_dequant_cache_paged_window(device, bsz, pps, n_kv, hd, seqlens, bonus, kb, vb, out_4d):
    torch.manual_seed(5)
    dim = n_kv * hd
    G = dim // 32
    num_pages = bsz * pps + 3
    kc, ks = rand_packed(num_pages, n_kv, hd, kb, device)
    vc, vs = rand_packed(num_pages, n_kv, hd, vb, device)
    block_table = make_block_table(bsz, pps, num_pages, 6).to(device)
    cache_seqlens = torch.tensor(seqlens, dtype = torch.int32, device = device)

    scratch_pages = bsz * pps + 2          # spare scratch pages past the block-table span stay untouched
    shape = (scratch_pages, PAGE, n_kv, hd) if out_4d else (scratch_pages, PAGE, dim)
    k_out = sentinel_like(shape, device)
    v_out = -sentinel_like(shape, device)
    k_ref = k_out.clone().view(scratch_pages, PAGE, dim).double().cpu()
    v_ref = v_out.clone().view(scratch_pages, PAGE, dim).double().cpu()

    ext.dequant_cache_paged_window(kc, ks, k_out, vc, vs, v_out, cache_seqlens, block_table, PAGE, bonus, 0.0)

    bt = block_table.cpu().long()
    for b in range(bsz):
        n = seqlens[b] + bonus
        t = torch.arange(n)
        src_page = bt[b, t // PAGE]
        src_row = t % PAGE
        dst_page = b * pps + t // PAGE
        k_ref[dst_page, src_row] = ref_dequant_rows(kc[src_page.to(device), src_row.to(device)],
                                                    ks[src_page.to(device), src_row.to(device)], kb)
        v_ref[dst_page, src_row] = ref_dequant_rows(vc[src_page.to(device), src_row.to(device)],
                                                    vs[src_page.to(device), src_row.to(device)], vb)
    check_decoded(k_out.view(scratch_pages, PAGE, dim), k_ref)
    check_decoded(v_out.view(scratch_pages, PAGE, dim), v_ref)


@torch.inference_mode()
def test_dequant_cache_paged_window_matches_full_dequant(device):
    """Same decoded values as dequant_cache_paged (the in-place variant, same kernel), relocated: bitwise"""
    torch.manual_seed(7)
    bsz, pps, n_kv, hd, bits = 2, 3, 4, 128, 4
    num_pages = 8
    kc, ks = rand_packed(num_pages, n_kv, hd, bits, device)
    vc, vs = rand_packed(num_pages, n_kv, hd, bits, device)
    block_table = make_block_table(bsz, pps, num_pages, 8).to(device)
    seqlens = [400, 650]
    cache_seqlens = torch.tensor(seqlens, dtype = torch.int32, device = device)
    full_k = torch.zeros((num_pages, PAGE, n_kv, hd), dtype = torch.half, device = device)
    full_v = torch.zeros_like(full_k)
    ext.dequant_cache_paged(kc, ks, full_k, vc, vs, full_v, cache_seqlens, block_table, PAGE, -1, 0.0)
    win_k = torch.zeros((bsz * pps, PAGE, n_kv, hd), dtype = torch.half, device = device)
    win_v = torch.zeros_like(win_k)
    ext.dequant_cache_paged_window(kc, ks, win_k, vc, vs, win_v, cache_seqlens, block_table, PAGE, 0, 0.0)
    for b in range(bsz):
        for p in range(pps):
            n = min(max(seqlens[b] - p * PAGE, 0), PAGE)
            src = block_table[b, p].item()
            assert torch.equal(win_k[b * pps + p, :n], full_k[src, :n])
            assert torch.equal(win_v[b * pps + p, :n], full_v[src, :n])


@torch.inference_mode()
def test_dequant_cache_paged_window_rejects(device):
    kc, ks = rand_packed(4, 2, 64, 4, device)
    vc, vs = rand_packed(4, 2, 64, 4, device)
    bt = torch.arange(4, dtype = torch.int32, device = device).view(2, 2)
    sl = torch.zeros((2,), dtype = torch.int32, device = device)
    small = torch.zeros((3, PAGE, 2, 64), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "scratch too small"):
        ext.dequant_cache_paged_window(kc, ks, small, vc, vs, small, sl, bt, PAGE, 0, 0.0)
    out = torch.zeros((4, PAGE, 2, 64), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "Page size"):
        ext.dequant_cache_paged_window(kc, ks, out, vc, vs, out, sl, bt, 128, 0, 0.0)
    with pytest.raises(RuntimeError, match = "bitrate"):
        ext.dequant_cache_paged_window(kc[..., :2], ks, out, vc, vs, out, sl, bt, PAGE, 0, 0.0)
    with pytest.raises(RuntimeError, match = "multiple of 32"):
        o = torch.zeros((4, PAGE, 2, 40), dtype = torch.half, device = device)
        ext.dequant_cache_paged_window(kc, ks, o, vc, vs, o, sl, bt, PAGE, 0, 0.0)
