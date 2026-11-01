import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.attention_fn import sm120_tma
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill


device = os.environ.get("EXL3_TEST_DEVICE", "cuda:0")


def _device_index():
    index = torch.device(device).index
    return torch.cuda.current_device() if index is None else index


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("SM120 GPU required")
    if not sm120_tma.has_sm120_tma:
        pytest.skip("extension was not built with SM120 TMA attention support detection")
    if not ext.sm120_tma_attn_supported(_device_index()):
        pytest.skip("extension has no usable SM120 TMA kernel for this device")


def _make_case(q_len, q_heads, kv_heads, dim, past_lens, constant_values = False, uniform_scores = False):
    torch.manual_seed(1234 + dim + q_len)
    bsz = len(past_lens)
    pages_per_seq = math.ceil((max(past_lens) + q_len) / 256) + 1
    num_pages = bsz * pages_per_seq

    # Reverse each sequence's physical pages to ensure the kernel follows block_table.
    block_table = torch.arange(num_pages, dtype = torch.int32, device = device).reshape(bsz, pages_per_seq)
    block_table = block_table.flip(1).contiguous()
    cache_seqlens = torch.tensor(past_lens, dtype = torch.int32, device = device)
    k_cache = torch.full((num_pages, 256, kv_heads, dim), torch.nan, dtype = torch.float16, device = device)
    v_cache = torch.full_like(k_cache, torch.nan)

    for b, past_len in enumerate(past_lens):
        past_k = torch.randn((past_len, kv_heads, dim), device = device).half()
        past_v = torch.ones_like(past_k) if constant_values else (torch.randn_like(past_k) * 0.25).half()
        if uniform_scores:
            past_k.zero_()
        for token0 in range(0, past_len, 256):
            count = min(256, past_len - token0)
            physical_page = int(block_table[b, token0 // 256].item())
            k_cache[physical_page, :count].copy_(past_k[token0:token0 + count])
            v_cache[physical_page, :count].copy_(past_v[token0:token0 + count])

    q = torch.randn((bsz, q_len, q_heads, dim), device = device).half()
    k = torch.randn((bsz, q_len, kv_heads, dim), device = device).half()
    v = torch.ones_like(k) if constant_values else (torch.randn_like(k) * 0.25).half()
    if uniform_scores:
        q.zero_()
        k.zero_()
    return q, k, v, k_cache, v_cache, block_table, cache_seqlens


@pytest.fixture
def support_cache():
    sm120_tma._supported.cache_clear()
    yield
    sm120_tma._supported.cache_clear()


@pytest.mark.parametrize("available", [False, True])
def test_sm120_tma_support_is_cached_per_device(monkeypatch, support_cache, available):
    calls = []

    def supported(device_index):
        calls.append(device_index)
        return available

    monkeypatch.setattr(sm120_tma, "ext", SimpleNamespace(sm120_tma_attn_supported = supported))
    for device_index in (0, 0, 1, 1):
        assert sm120_tma._supported(device_index) is available
    assert calls == [0, 1]


def test_sm120_tma_unsupported_build_falls_through(monkeypatch, support_cache):
    monkeypatch.setattr(sm120_tma, "has_sm120_tma", True)
    monkeypatch.setattr(sm120_tma, "_enabled", True)
    monkeypatch.setattr(sm120_tma, "ext", SimpleNamespace(sm120_tma_attn_supported = lambda _: False))
    # No CUDA allocation is needed: an unavailable build must decline before reading cache tensors.
    args = SimpleNamespace(q = SimpleNamespace(device = torch.device("cuda:0")))
    assert sm120_tma.fn_sm120_tma_attn_prefill(args) is None


def test_sm120_tma_support_errors_propagate(monkeypatch, support_cache):
    def supported(_):
        raise RuntimeError("CUDA query failed")

    monkeypatch.setattr(sm120_tma, "ext", SimpleNamespace(sm120_tma_attn_supported = supported))
    with pytest.raises(RuntimeError, match = "CUDA query failed"):
        sm120_tma._supported(0)


@torch.inference_mode()
def test_sm120_tma_unsupported_build_preserves_cache():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not sm120_tma.has_sm120_tma:
        pytest.skip("extension has no SM120 TMA support detection")
    if ext.sm120_tma_attn_supported(_device_index()):
        pytest.skip("requires a device or build without SM120 TMA support")
    q, k, v, k_cache, v_cache, block_table, cache_seqlens = _make_case(64, 8, 1, 128, [17])
    saved_k, saved_v = k_cache.clone(), v_cache.clone()
    with pytest.raises(RuntimeError, match = "sm120_tma_attn requires"):
        ext.sm120_tma_attn_paged(
            q, k, v, k_cache, v_cache, block_table, cache_seqlens,
            True, 1.0 / math.sqrt(128), 0.0, 1, 8,
        )
    torch.testing.assert_close(k_cache, saved_k, atol = 0, rtol = 0, equal_nan = True)
    torch.testing.assert_close(v_cache, saved_v, atol = 0, rtol = 0, equal_nan = True)


@pytest.mark.parametrize(
    "q_len,q_heads,kv_heads,dim,past_lens,causal,softcap,split_k,q_group_mode",
    [
        (37, 8, 1, 128, [251], True, 0.0, 1, 8),
        (73, 12, 2, 256, [191, 319], True, 0.0, 1, 8),
        (35, 4, 2, 256, [257], False, 0.0, 1, 2),
        (67, 6, 1, 256, [1021], True, 0.0, 3, 8),
        (41, 8, 1, 512, [273], True, 30.0, 1, 8),
        (33, 24, 4, 256, [0], True, 0.0, 3, 2),
        (67, 24, 4, 256, [257], True, 30.0, 1, 2),
        (67, 24, 4, 256, [257], True, 30.0, 3, 2),
        (33, 32, 4, 512, [0], True, 0.0, 3, 8),
        (67, 32, 4, 512, [257], True, 30.0, 1, 8),
        (67, 32, 4, 512, [257], True, 30.0, 3, 8),
    ],
)
@torch.inference_mode()
def test_sm120_tma_matches_triton(
    q_len,
    q_heads,
    kv_heads,
    dim,
    past_lens,
    causal,
    softcap,
    split_k,
    q_group_mode,
):
    _require_sm120()
    q, k, v, k_cache, v_cache, block_table, cache_seqlens = _make_case(
        q_len, q_heads, kv_heads, dim, past_lens
    )
    scale = 1.0 / math.sqrt(dim)

    expected = paged_attn_triton_prefill(
        q,
        None,
        None,
        k_cache.clone(),
        v_cache.clone(),
        block_table,
        cache_seqlens,
        causal = causal,
        softmax_scale = scale,
        softcap = softcap,
        k_new = k,
        v_new = v,
    )
    actual = ext.sm120_tma_attn_paged(
        q,
        k,
        v,
        k_cache,
        v_cache,
        block_table,
        cache_seqlens,
        causal,
        scale,
        softcap,
        split_k,
        q_group_mode,
    )

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol = 5e-3, rtol = 5e-3)

    # The native path appends K/V directly to the paged cache before attention.
    for b, past_len in enumerate(past_lens):
        for i in (0, q_len - 1):
            logical_token = past_len + i
            physical_page = block_table[b, logical_token // 256]
            page_offset = logical_token % 256
            torch.testing.assert_close(k_cache[physical_page, page_offset], k[b, i], atol = 0, rtol = 0)
            torch.testing.assert_close(v_cache[physical_page, page_offset], v[b, i], atol = 0, rtol = 0)


@pytest.mark.parametrize("dim,split_k", [(128, 1), (256, 1), (256, 3), (512, 1), (512, 3)])
@pytest.mark.parametrize("past_len", [65536, 100000])
@pytest.mark.parametrize("uniform_scores", [False, True])
@torch.inference_mode()
def test_sm120_tma_long_context_constant_values(dim, split_k, past_len, uniform_scores):
    _require_sm120()
    q, k, v, k_cache, v_cache, block_table, cache_seqlens = _make_case(
        64, 8, 1, dim, [past_len], constant_values = True, uniform_scores = uniform_scores
    )
    actual = ext.sm120_tma_attn_paged(
        q, k, v, k_cache, v_cache, block_table, cache_seqlens,
        True, 1.0 / math.sqrt(dim), 0.0, split_k, 8,
    )

    # Any normalized attention distribution over constant values must return that constant.
    # This catches long-context numerator accumulation loss independently of another backend.
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.ones_like(actual), atol = 5e-3, rtol = 5e-3)
