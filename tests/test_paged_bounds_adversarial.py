"""
Adversarial tests for the paged-cache bounds hardening (upstream #436 / PR #437).

Two claims under test, both empirically:

1. Containment: a torn/stale block_table or cache_seqlens entry must produce a
   skipped write or a bounded (in-pool) read -- never a store outside the cache
   pool and never a fault. Every write kernel is launched against a pool that is
   a *slice* of a larger allocation whose tail is a sentinel guard region, so a
   wild write past the pool lands in the guard and is detected deterministically
   (instead of corrupting unrelated tensors and maybe never being noticed).

2. No false truncation: the read-side clamps (total_k_len, phys) must not eat
   legitimate tokens. Sequences that exactly fill the block-table span
   (past + append == num_pages_per_seq * page_size) must attend over every
   token, and cacheless prefill (NEW_KV=2, dummy 1-page table) must attend over
   the full q_len for q_len > page_size -- the carve-out.

The pinned-memory microbenches at the bottom pin down the host-side hazard the
clamps defend against: (a) a host rewrite of pinned staging while its async H2D
copy is still queued is NOT ordered against the copy, and (b) whether freeing a
pinned tensor whose H2D is in flight can hand the block to a new pinned alloc
before the copy completes (the CachingHostAllocator event claim in job.py).
"""
import gc
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.modules.attention_fn.triton_paged import (
    paged_attn_triton_decode, paged_attn_triton_prefill, paged_attn_triton_longq,
)
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.constants import PAGE_SIZE

device = "cuda:0"

INT32_MAX = 2 ** 31 - 1

# Values a torn/stale staging buffer can hold. "pool+k" lands inside the guard
# region if a kernel fails to clamp (deterministic detection); the huge values
# exercise the clamp arithmetic itself (signedness, int64 widening).
PHYS_CORRUPT = [-1, "pool+1", "pool+guardhalf", 2 ** 20, INT32_MAX]
SEQ_CORRUPT = [-7, "width+512", INT32_MAX // 4, -INT32_MAX // 2]

WIDTH = 8          # block-table columns (logical pages per sequence)
GUARD = 8          # guard pages appended after the pool


def _phys(v, pool_pages, guard_pages):
    if v == "pool+1":
        return pool_pages + 1
    if v == "pool+guardhalf":
        return pool_pages + guard_pages // 2
    return v


def _seq(v, width):
    if v == "width+512":
        return width * PAGE_SIZE + 512
    return v


def ref_attn(q, k, v, causal, past, window=None):
    B, Q, H, D = q.shape
    T = k.shape[1]
    g = H // k.shape[2]
    kk = k.repeat_interleave(g, dim=2).float(); vv = v.repeat_interleave(g, dim=2).float()
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kk) * D ** -0.5
    qpos = (T - Q + torch.arange(Q, device=q.device)).view(Q, 1)
    kpos = torch.arange(T, device=q.device).view(1, T)
    mask = torch.ones((Q, T), dtype=torch.bool, device=q.device)
    if causal:
        mask &= kpos <= qpos
    if window is not None:
        mask &= kpos >= qpos - window
    s = s.masked_fill(~mask.view(1, 1, Q, T), -float("inf"))
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), vv)


def gather(kc, bt, T):
    B, pages = bt.shape
    flat = kc[bt.long().view(-1)].view(B, pages * PAGE_SIZE, kc.shape[2], kc.shape[3])
    return flat[:, :T]


def valid_table(B, width, pool_pages):
    """A plausible table: distinct in-pool pages per row."""
    assert pool_pages >= B * width
    torch.manual_seed(1234 + B * 10 + width)
    idx = torch.randperm(pool_pages, device=device, dtype=torch.int32)[: B * width]
    return idx.view(B, width).contiguous()


class GuardedPool:
    """fp16 K/V pool of `pages` physical pages carved out of a (pages + guard_pages)
    allocation; the tail guard is filled with a sentinel. Any store past the
    pool (by whole pages, the only granularity a page index gives) clobbers it."""

    def __init__(self, pages, guard_pages, kvh, hd):
        self.pages, self.guard_pages = pages, guard_pages
        self.big_k = torch.full((pages + guard_pages, PAGE_SIZE, kvh, hd), 1.5, dtype=torch.half, device=device)
        self.big_v = torch.full_like(self.big_k, -1.5)
        self.kc = self.big_k[:pages]
        self.vc = self.big_v[:pages]
        self.k_guard = self.big_k[pages:]
        self.v_guard = self.big_v[pages:]

    def assert_intact(self, ctx=""):
        torch.cuda.synchronize()
        assert (self.k_guard == 1.5).all(), f"K guard clobbered {ctx}"
        assert (self.v_guard == -1.5).all(), f"V guard clobbered {ctx}"


# ---------------------------------------------------------------------------
# 1. Write kernels: torn table/seqlen must not store outside the pool
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phys", PHYS_CORRUPT)
@pytest.mark.parametrize("seq", SEQ_CORRUPT)
def test_torn_write_paged_kv_update(phys, seq):
    B, H, D, S = 2, 4, 128, 300
    pool_pages = B * WIDTH
    pool = GuardedPool(pool_pages, GUARD, H, D)
    bt = valid_table(B, WIDTH, pool_pages)
    sl = torch.full((B,), 700, dtype=torch.int32, device=device)
    bt[1, 2] = _phys(phys, pool_pages, GUARD)
    sl[0] = _seq(seq, WIDTH)
    k = torch.randn((B, S, H, D), dtype=torch.half, device=device)
    v = torch.randn_like(k)
    ext.paged_kv_cache_update(k, v, pool.kc, pool.vc, bt, sl)
    pool.assert_intact(f"(paged_kv_update phys={phys} seq={sl[0].item()})")


@pytest.mark.parametrize("phys", PHYS_CORRUPT)
def test_torn_write_bighead_update(phys):
    """bighead_attn_paged runs kv_cache_update_kernel_paged internally before the
    chunked attention kernels, so a torn entry exercises both the write and the
    read guards in one call."""
    B, H, D, S = 2, 4, 128, 300
    pool_pages = B * WIDTH
    pool = GuardedPool(pool_pages, GUARD, H, D)
    bt = valid_table(B, WIDTH, pool_pages)
    sl = torch.full((B,), 700, dtype=torch.int32, device=device)
    bt[0, 2] = _phys(phys, pool_pages, GUARD)
    q = torch.randn((B, 16, H * 2, D), dtype=torch.half, device=device)
    k = torch.randn((B, S, H, D), dtype=torch.half, device=device)
    v = torch.randn_like(k)
    o = torch.empty_like(q)
    ext.bighead_attn_paged(q, k, v, pool.kc, pool.vc, bt, sl, o, 128, True, 0.0)
    pool.assert_intact(f"(bighead phys={phys})")
    assert torch.isfinite(o.float()).all(), "attention output not finite after torn index"


@pytest.mark.parametrize("phys", PHYS_CORRUPT)
@pytest.mark.parametrize("seq", [-7, "width+512", INT32_MAX // 4])
def test_torn_write_dspark(phys, seq):
    B, s, w = 2, 300, 64
    pool_pages = B * WIDTH
    big = torch.full((pool_pages + GUARD, PAGE_SIZE, w), 2.5, dtype=torch.half, device=device)
    kv = big[:pool_pages]
    guard_t = big[pool_pages:]
    bt = valid_table(B, WIDTH, pool_pages)
    sl = torch.full((B,), 700, dtype=torch.int32, device=device)
    bt[1, 2] = _phys(phys, pool_pages, GUARD)
    sl[0] = _seq(seq, WIDTH)
    rows = torch.randn((B, s, w), dtype=torch.half, device=device)
    ext.dspark_write_rows(rows, kv, bt, sl)
    torch.cuda.synchronize()
    assert (guard_t == 2.5).all(), f"guard clobbered (dspark phys={phys} seq={sl[0].item()})"


@pytest.mark.parametrize("phys", PHYS_CORRUPT)
def test_torn_write_quant_paged(phys):
    B, H, D = 2, 4, 128
    bits = 4
    pool_pages = B * WIDTH
    pool = GuardedPool(pool_pages, GUARD, H, D)
    qcols = H * D // 32 * bits
    scols = H * D // 32
    def guarded_int_pool():
        big = torch.full((pool_pages + GUARD, PAGE_SIZE, qcols), 0x5a5a5a5a, dtype=torch.int32, device=device)
        return big[:pool_pages], big[pool_pages:]

    def guarded_scale_pool():
        big = torch.full((pool_pages + GUARD, PAGE_SIZE, scols), 3.0, dtype=torch.half, device=device)
        return big[:pool_pages], big[pool_pages:]

    kq, kq_g = guarded_int_pool()
    ks, ks_g = guarded_scale_pool()
    vq, vq_g = guarded_int_pool()
    vs, vs_g = guarded_scale_pool()
    bt = valid_table(B, WIDTH, pool_pages)
    sl = torch.full((B,), 700, dtype=torch.int32, device=device)
    bt[0, 2] = _phys(phys, pool_pages, GUARD)
    ext.quant_cache_paged(
        pool.kc, kq, ks, pool.vc, vq, vs, sl, bt, PAGE_SIZE, 300, 0.0, False,
    )
    torch.cuda.synchronize()
    for name, g, sentinel in [("kq", kq_g, 0x5a5a5a5a), ("ks", ks_g, 3.0),
                              ("vq", vq_g, 0x5a5a5a5a), ("vs", vs_g, 3.0)]:
        assert (g == sentinel).all(), f"quant {name} guard clobbered phys={phys}"


@pytest.mark.parametrize("phys", ["pool+1", INT32_MAX, -1])
def test_torn_read_dequant(phys):
    B, H, D = 2, 4, 128
    bits = 4
    qcols = H * D // 32 * bits
    scols = H * D // 32
    pool_pages = B * WIDTH

    def qpool():
        big = torch.randint(-2 ** 31 + 1, 2 ** 31 - 1, (pool_pages + GUARD, PAGE_SIZE, qcols),
                            dtype=torch.int32, device=device)
        return big[:pool_pages]

    def spool():
        big = torch.rand((pool_pages + GUARD, PAGE_SIZE, scols), dtype=torch.half, device=device) + 0.5
        return big[:pool_pages]

    # K/V outputs each get their own allocation with a guard tail
    big_ko = torch.full((pool_pages + GUARD, PAGE_SIZE, H * D), 7.0, dtype=torch.half, device=device)
    ko, ko_g = big_ko[:pool_pages], big_ko[pool_pages:]
    big_vo = torch.full((pool_pages + GUARD, PAGE_SIZE, H * D), 7.0, dtype=torch.half, device=device)
    vo, vo_g = big_vo[:pool_pages], big_vo[pool_pages:]

    bt = valid_table(B, WIDTH, pool_pages)
    sl = torch.full((B,), 700, dtype=torch.int32, device=device)
    bt[1, 2] = _phys(phys, pool_pages, GUARD)
    ext.dequant_cache_paged(qpool(), spool(), ko, qpool(), spool(), vo, sl, bt, PAGE_SIZE, -1, 0.0)
    torch.cuda.synchronize()
    assert (ko_g == 7.0).all(), f"dequant K out-guard clobbered phys={phys}"
    assert (vo_g == 7.0).all(), f"dequant V out-guard clobbered phys={phys}"


def test_torn_write_triton_update():
    import triton
    from exllamav3.modules.attention_fn.triton_paged import _paged_kv_update_kernel
    B, H, D, S = 2, 4, 128, 300
    pool_pages = B * WIDTH
    pool = GuardedPool(pool_pages, GUARD, H, D)
    bt = valid_table(B, WIDTH, pool_pages)
    sl = torch.full((B,), 700, dtype=torch.int32, device=device)
    k = torch.randn((B, S, H, D), dtype=torch.half, device=device)
    v = torch.randn_like(k)
    for phys in [pool_pages + 1, INT32_MAX, -1]:
        bt[0, 2] = phys
        for seq in [WIDTH * PAGE_SIZE + 512, INT32_MAX // 4, -12345]:
            sl[0] = seq
            with torch.cuda.device(device):
                try:
                    _paged_kv_update_kernel[(B * S, H, 1)](
                        k, v, pool.kc, pool.vc, bt, sl, bt.shape[1], pool.kc.shape[0], S, H,
                        PAGE_SIZE, D, triton.next_power_of_2(D), num_warps=2, num_stages=3,
                    )
                except TypeError:
                    # pre-hardening signature: no num_cache_pages arg
                    _paged_kv_update_kernel[(B * S, H, 1)](
                        k, v, pool.kc, pool.vc, bt, sl, bt.shape[1], S, H,
                        PAGE_SIZE, D, triton.next_power_of_2(D), num_warps=2, num_stages=3,
                    )
            pool.assert_intact(f"(triton update phys={phys} seq={seq})")


# ---------------------------------------------------------------------------
# 2. Read kernels: torn table/seqlen must not fault or hang
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q_len", [1, 8])
def test_torn_read_decode(q_len):
    B, H, D = 2, 8, 128
    pool_pages = B * WIDTH
    pool = GuardedPool(pool_pages, 4, H // 2, D)
    bt = valid_table(B, WIDTH, pool_pages)
    sl = torch.full((B,), 700, dtype=torch.int32, device=device)
    q = torch.randn((B, q_len, H, D), dtype=torch.half, device=device)
    k = torch.randn((B, q_len, H // 2, D), dtype=torch.half, device=device)
    v = torch.randn_like(k)
    for phys in [pool_pages + 1, INT32_MAX, -1]:
        bt[0, 2] = phys
        out = paged_attn_triton_decode(q, k, v, pool.kc, pool.vc, bt, sl, causal=True)
        assert torch.isfinite(out.float()).all(), f"decode not finite phys={phys}"
    bt = valid_table(B, WIDTH, pool_pages)
    for seq in [WIDTH * PAGE_SIZE + 512, INT32_MAX // 4, -9999]:
        sl[0] = seq
        out = paged_attn_triton_decode(q, k, v, pool.kc, pool.vc, bt, sl, causal=True)
        assert torch.isfinite(out.float()).all(), f"decode not finite seq={seq}"


@pytest.mark.parametrize("q_len", [300, 1000])
def test_torn_read_prefill(q_len):
    B, H, D = 2, 8, 128
    pool_pages = B * WIDTH
    pool = GuardedPool(pool_pages, 4, H // 2, D)
    bt = valid_table(B, WIDTH, pool_pages)
    sl = torch.full((B,), 700, dtype=torch.int32, device=device)
    q = torch.randn((B, q_len, H, D), dtype=torch.half, device=device)
    k = torch.randn((B, q_len, H // 2, D), dtype=torch.half, device=device)
    v = torch.randn_like(k)
    bt[0, 2] = INT32_MAX
    bt[1, 1] = pool_pages + 1
    sl[0] = INT32_MAX // 4
    out = paged_attn_triton_prefill(q, k, v, pool.kc, pool.vc, bt, sl, causal=True)
    assert torch.isfinite(out.float()).all()


def test_torn_read_longq():
    B, H, D = 1, 8, 128
    pool_pages = B * WIDTH
    pool = GuardedPool(pool_pages, 4, H // 2, D)
    bt = valid_table(B, WIDTH, pool_pages)
    sl = torch.full((B,), 700, dtype=torch.int32, device=device)
    q = torch.randn((B, 300, H, D), dtype=torch.half, device=device)
    k = torch.randn((B, 1, H // 2, D), dtype=torch.half, device=device)
    v = torch.randn_like(k)
    bt[0, 2] = INT32_MAX
    out = paged_attn_triton_longq(q, k, v, pool.kc, pool.vc, bt, sl, causal=True)
    assert torch.isfinite(out.float()).all()
    bt = valid_table(B, WIDTH, pool_pages)
    sl[0] = INT32_MAX // 4
    out = paged_attn_triton_longq(q, k, v, pool.kc, pool.vc, bt, sl, causal=True)
    assert torch.isfinite(out.float()).all()


# ---------------------------------------------------------------------------
# 3. No false truncation: clamps must not eat legitimate tokens
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q_len,past", [(256, 768), (257, 767), (520, 504), (1, 1023), (8, 1016)])
def test_exact_table_span_prefill(q_len, past):
    """past + q_len == width * PAGE_SIZE exactly: every token is in the table span."""
    B, H, D = 2, 8, 128
    total = past + q_len
    assert total % PAGE_SIZE == 0
    width = total // PAGE_SIZE
    pool_pages = B * width
    pool = GuardedPool(pool_pages, 4, H // 2, D)
    bt = valid_table(B, width, pool_pages)
    sl = torch.full((B,), past, dtype=torch.int32, device=device)
    torch.manual_seed(99)
    q = torch.randn((B, q_len, H, D), dtype=torch.half, device=device)
    k = torch.randn((B, q_len, H // 2, D), dtype=torch.half, device=device)
    v = torch.randn_like(k)
    out = paged_attn_triton_prefill(q, k, v, pool.kc, pool.vc, bt, sl, causal=True)
    ref = ref_attn(q, gather(pool.kc, bt, total), gather(pool.vc, bt, total), True, past)
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 8e-3, f"exact-span prefill truncated? rel err {err:.3e}"


@pytest.mark.parametrize("q_len,past", [(1, 1023), (1, 2047), (8, 2040), (300, 1748)])
def test_exact_table_span_longq(q_len, past):
    B, H, D = 1, 8, 128
    total = past + q_len
    assert total % PAGE_SIZE == 0
    width = total // PAGE_SIZE
    pool_pages = B * width
    pool = GuardedPool(pool_pages, 4, H // 2, D)
    bt = valid_table(B, width, pool_pages)
    sl = torch.full((B,), past, dtype=torch.int32, device=device)
    torch.manual_seed(98)
    q = torch.randn((B, q_len, H, D), dtype=torch.half, device=device)
    k = torch.randn((B, q_len, H // 2, D), dtype=torch.half, device=device)
    v = torch.randn_like(k)
    if q_len <= 256:
        out = paged_attn_triton_decode(q, k, v, pool.kc, pool.vc, bt, sl, causal=True)
    else:
        out = paged_attn_triton_longq(q, k, v, pool.kc, pool.vc, bt, sl, causal=True)
    ref = ref_attn(q, gather(pool.kc, bt, total), gather(pool.vc, bt, total), True, past)
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 8e-3, f"exact-span longq/decode truncated? rel err {err:.3e}"


@pytest.mark.parametrize("q_len", [256, 257, 300, 512, 1024, 4096])
@pytest.mark.parametrize("causal", [True, False])
def test_nocache_carveout(q_len, causal):
    """NEW_KV=2 runs with a dummy 1-page table; the total_k_len clamp must not
    apply there or every kv index past 256 would be dropped."""
    B, H, D = 2, 8, 128
    torch.manual_seed(q_len + int(causal))
    q = torch.randn((B, q_len, H, D), dtype=torch.half, device=device)
    k = torch.randn((B, q_len, H // 2, D), dtype=torch.half, device=device)
    v = torch.randn_like(k)
    out = paged_attn_triton_prefill(q, None, None, None, None, None, None, causal=causal, k_new=k, v_new=v)
    ref = ref_attn(q, k, v, causal, 0)
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 8e-3, f"nocache q_len={q_len} causal={causal} rel err {err:.3e}"


@pytest.mark.parametrize("q_len,window", [(1000, 128), (600, 64)])
def test_nocache_window_carveout(q_len, window):
    """Windowed cacheless attention: window semantics need real positions, the
    dummy-table clamp must not distort them."""
    B, H, D = 2, 8, 128
    torch.manual_seed(q_len)
    q = torch.randn((B, q_len, H, D), dtype=torch.half, device=device)
    k = torch.randn((B, q_len, H // 2, D), dtype=torch.half, device=device)
    v = torch.randn_like(k)
    out = paged_attn_triton_prefill(
        q, None, None, None, None, None, None, causal=True, window_size=(window, 0), k_new=k, v_new=v,
    )
    ref = ref_attn(q, k, v, True, 0, window)
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 8e-3, f"nocache window rel err {err:.3e}"


# ---------------------------------------------------------------------------
# 4. Pinned-staging hazard microbenches (the race the clamps defend against)
# ---------------------------------------------------------------------------

def _queue_delay(ms):
    """Enqueue ~ms worth of GPU work and return while it is still pending."""
    a = torch.randn(4096, 4096, device=device, dtype=torch.float32)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float32)
    import time
    t0 = time.time()
    while (time.time() - t0) * 1000 < ms:
        a = a @ b * 1e-10
    return a


def test_pinned_h2d_tear():
    """A host rewrite of pinned staging while its async H2D copy sits behind a
    slow kernel is NOT ordered against the copy: the device receives the NEW
    bytes. This is the torn-staging primitive."""
    N = 1 << 20
    pin = torch.zeros(N, dtype=torch.int32, pin_memory=True)
    dev = torch.empty(N, dtype=torch.int32, device=device)
    _queue_delay(80)                 # keep the stream busy so the copy queues
    dev.copy_(pin, non_blocking=True)
    pin.fill_(0x7f7f7f7f)            # host rewrite while the copy is queued
    torch.cuda.synchronize()
    torn = bool((dev == 0x7f7f7f7f).all())
    print(f"\npinned H2D tear observed: {torn} (device saw the rewrite)")
    del pin
    gc.collect()


def test_pinned_free_reuse():
    """Free a pinned tensor while its async H2D is queued, then immediately
    allocate a same-size pinned tensor and scribble into it. If the caching
    host allocator reuses the block, the in-flight copy transmits the scribble
    (use-after-free torn staging); if it holds the block back, the copy lands
    with the original bytes."""
    N = 1 << 20
    pin1 = torch.ones(N, dtype=torch.int32, pin_memory=True)
    dev1 = torch.zeros(N, dtype=torch.int32, device=device)
    _queue_delay(120)
    dev1.copy_(pin1, non_blocking=True)
    handle = pin1.data_ptr()
    del pin1
    gc.collect()
    pin2 = torch.full((N,), -559038737, dtype=torch.int32, pin_memory=True)  # 0xdeadbeef signed
    reused = pin2.data_ptr() == handle
    torch.cuda.synchronize()
    spoiled = bool((dev1 == -559038737).any())
    print(f"\npinned block reused on realloc: {reused}; in-flight copy spoiled: {spoiled}")
    del pin2
    gc.collect()
