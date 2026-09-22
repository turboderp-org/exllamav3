"""
Quantized-cache prefill staging: one fp16 K/V window per device, reserved once at cache load and
sized to the cache's context, which every unbounded two-pass prefill call views instead of allocating.
Bounded calls (QSA's dense regime) stage through a small per-call transient and reserve nothing.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from types import SimpleNamespace
import pytest
import torch
from exllamav3.modules.attention_fn import triton_paged
from exllamav3.modules.attention_fn.triton_paged import (
    paged_attn_triton_prefill, reserve_qc_prefill_staging, release_qc_prefill_staging,
)
from exllamav3.cache.quant import CacheLayer_quant
from exllamav3.cache.qsa import CacheLayer_qsa_quant
from exllamav3.constants import PAGE_SIZE
from test_triton_paged_hdpad import make_cache, _quant_cache, ref_attn, gather

device = torch.device("cuda:0")
KVH, HD, BITS = 2, 128, 8
HUGE_TOKENS = 1 << 40   # fails to allocate on any device, without touching real memory

pytestmark = pytest.mark.skipif(triton_paged._qc_staging != 1, reason = "prefill staging disabled")


@pytest.fixture(autouse = True)
def no_reservation():
    triton_paged._qc_staging_windows.clear()
    yield
    triton_paged._qc_staging_windows.clear()


def staged_prefill_case(past, q_len, max_kv_len = None):
    kc, vc, bt, sl, k, v, q = make_cache(1, past + q_len + 3, KVH, HD, past, q_len, past + q_len)
    qk, sk, kdeq = _quant_cache(kc, BITS)
    qv, sv, vdeq = _quant_cache(vc, BITS)
    run = lambda: paged_attn_triton_prefill(
        q, None, None, qk, qv, bt, sl, causal = True, qc = (sk, sv, BITS, BITS),
        pre_appended_len = q_len, n_kv_heads_override = KVH, max_kv_len = max_kv_len,
    )
    ref = lambda: ref_attn(q, gather(kdeq, bt, past + q_len), gather(vdeq, bt, past + q_len), True, past)
    return run, ref, bt.shape[1], q


def window_numel():
    entry = triton_paged._qc_staging_windows.get(device)
    return 0 if entry is None else entry[0].numel() // 2


def test_reservation_is_sized_to_the_context_and_shared():
    reserve_qc_prefill_staging(device, 8 * PAGE_SIZE, KVH * HD)
    reserve_qc_prefill_staging(device, 8 * PAGE_SIZE, KVH * HD)   # second cache, same context
    assert window_numel() == 8 * PAGE_SIZE * KVH * HD
    release_qc_prefill_staging(device)
    assert window_numel() == 8 * PAGE_SIZE * KVH * HD
    release_qc_prefill_staging(device)
    assert window_numel() == 0


def test_prefill_reads_through_the_reserved_window():
    run, ref, pages, q = staged_prefill_case(past = 1500, q_len = 512)
    reserve_qc_prefill_staging(device, pages * PAGE_SIZE, KVH * HD)
    run()   # compile and pick num_stages outside the measurement
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    out = run()
    torch.cuda.synchronize()
    grown = torch.cuda.max_memory_allocated(device) - before
    window_bytes = pages * PAGE_SIZE * KVH * HD * 2
    assert grown < out.numel() * out.element_size() + window_bytes, f"prefill allocated {grown} bytes"
    expected = ref()
    err = (out.float() - expected).abs().max().item() / expected.abs().max().item()
    assert err < 1.2e-2, f"rel err {err:.3e}"


def test_prefill_without_reservation_raises():
    run, _, _, _ = staged_prefill_case(past = 1500, q_len = 512)
    with pytest.raises(RuntimeError, match = "reserve"):
        run()


def test_prefill_beyond_reservation_raises():
    run, _, pages, _ = staged_prefill_case(past = 1500, q_len = 512)
    reserve_qc_prefill_staging(device, (pages - 1) * PAGE_SIZE, KVH * HD)
    with pytest.raises(RuntimeError, match = "context"):
        run()


def test_cache_layers_reserve_on_alloc_and_release_on_free():
    attention = SimpleNamespace(num_kv_heads = KVH, head_dim = HD)
    layers = [CacheLayer_quant(None, attention, 0, 4 * PAGE_SIZE, BITS, BITS) for _ in range(2)]
    for layer in layers:
        layer.alloc(device)
    assert window_numel() == 4 * PAGE_SIZE * KVH * HD
    layers[0].free()
    assert window_numel() == 4 * PAGE_SIZE * KVH * HD
    layers[1].free()
    assert window_numel() == 0


@pytest.mark.parametrize("pages", [512, 513])
def test_reservation_is_exact_across_the_power_of_two_page_boundary(pages):
    # The per-call scratch this replaced rounded 513 pages up to 1024
    reserve_qc_prefill_staging(device, pages * PAGE_SIZE, 1)
    assert window_numel() == pages * PAGE_SIZE


def test_failed_first_reservation_leaves_no_window():
    with pytest.raises(torch.OutOfMemoryError):
        reserve_qc_prefill_staging(device, HUGE_TOKENS, 1)
    assert device not in triton_paged._qc_staging_windows


def test_failed_growth_keeps_the_existing_window_and_users():
    reserve_qc_prefill_staging(device, 8 * PAGE_SIZE, KVH * HD)
    with pytest.raises(torch.OutOfMemoryError):
        reserve_qc_prefill_staging(device, HUGE_TOKENS, KVH * HD)
    assert window_numel() == 8 * PAGE_SIZE * KVH * HD
    release_qc_prefill_staging(device)
    assert window_numel() == 0


def test_failed_cache_alloc_releases_nothing():
    attention = SimpleNamespace(num_kv_heads = KVH, head_dim = HD)
    held = CacheLayer_quant(None, attention, 0, 4 * PAGE_SIZE, BITS, BITS)
    held.alloc(device)
    failed = CacheLayer_quant(None, attention, 0, PAGE_SIZE << 24, BITS, BITS)
    with pytest.raises(torch.OutOfMemoryError):
        failed.alloc(device)
    failed.free()
    assert window_numel() == 4 * PAGE_SIZE * KVH * HD
    held.free()
    assert window_numel() == 0


def test_qsa_cache_layers_reserve_nothing():
    indexer = SimpleNamespace(head_dim = 128, compress_ratio = 4)
    attention = SimpleNamespace(num_kv_heads = KVH, head_dim = HD, qsa_indexer = indexer)
    layer = CacheLayer_qsa_quant(None, attention, 0, 4 * PAGE_SIZE, BITS, BITS)
    layer.alloc(device)
    assert window_numel() == 0
    layer.free()


def test_bounded_prefill_stages_without_a_reservation():
    past, q_len = 1500, 512
    run, ref, _, _ = staged_prefill_case(past, q_len, max_kv_len = past)
    out = run()
    expected = ref()
    err = (out.float() - expected).abs().max().item() / expected.abs().max().item()
    assert err < 1.2e-2, f"rel err {err:.3e}"
    assert device not in triton_paged._qc_staging_windows
