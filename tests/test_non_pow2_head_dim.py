"""
Non-power-of-2 head-dim attention dispatch.

flash-attn 2's kernels read uninitialized memory for head dims that are not a power of
two (verified on flash-attn 2.8.3 / sm_86): outputs are nondeterministic across calls on
bit-identical inputs and can contain NaN, which poisons the KV cache. The failure is sharp
for dim % 32 != 0 (e.g. 72/80/104/112/144), and dim=160 (32*5) even crashes with an illegal
memory access, so power-of-2 is the safe conservative gate.

These tests verify the dispatch contract for non-pow2 head dims:
  * the FA2 backends decline non-pow2 (so the bug is not hit)
  * the Triton paged wrapper declines non-pow2 (instead of raising mid-dispatch)
  * the torch SDPA fallbacks accept non-pow2 and produce deterministic, correct output
  * end-to-end cacheless dispatch for non-pow2 is deterministic

Requires CUDA. Skips when CUDA or the relevant optional backend is unavailable.
"""
import os
import sys

import pytest
import torch
from torch.nn.attention.bias import causal_lower_right

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.modules.attention_fn import attn_dispatch
from exllamav3.modules.attention_fn.common import AttnArgs
from exllamav3.modules.attention_fn.flash_attn_2 import (
    fn_flash_attn_func,
    fn_flash_attn_varlen_func,
    fn_flash_attn_with_kvcache,
    has_fa2,
    _is_pow2,
)
from exllamav3.modules.attention_fn.torch import (
    fn_torch_sdpa_fallback_cache,
    fn_torch_sdpa_fallback_nocache,
)
from exllamav3.modules.attention_fn.triton_paged import (
    fn_triton_paged_attn,
    has_triton,
)

device = os.environ.get("EXL3_TEST_DEVICE", "cuda:0")


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


# -------------------------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------------------------

def _rand_qkv(bsz, q_len, n_q_heads, n_kv_heads, dim, *, g):
    q = torch.randn(bsz, q_len, n_q_heads, dim, dtype=torch.half, device=device, generator=g)
    k = torch.randn(bsz, q_len, n_kv_heads, dim, dtype=torch.half, device=device, generator=g)
    v = torch.randn(bsz, q_len, n_kv_heads, dim, dtype=torch.half, device=device, generator=g)
    return q, k, v


def _ref_sdpa_nocache(q, k, v, sm_scale, causal=False):
    """Reference attention via torch SDPA, head_dim-agnostic (works for any dim)."""
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=causal, enable_gqa=(q.shape[2] != k.shape[2]), scale=sm_scale,
    ).transpose(1, 2)


def _ref_sdpa_cached(q, k_new, v_new, k_cache, v_cache, block_table, cache_seqlens, sm_scale, causal=True):
    """Reference for paged-cache attention: gather pages, append new k/v, attend q over all.
    Uses causal_lower_right so q (len q_len) can attend over kv (len total = cache_len + q_len)
    with the correct offset causal mask (is_causal=True requires matching lengths)."""
    bsz, q_len, n_q_heads, dim = q.shape
    _, _, n_kv_heads, _ = k_new.shape
    block_size = k_cache.shape[1]
    outs = []
    for b in range(bsz):
        seq_len = cache_seqlens[b].item()
        total = seq_len + q_len
        n_blocks = (total + block_size - 1) // block_size
        phys = block_table[b, :n_blocks]
        kb = k_cache[phys].reshape(-1, n_kv_heads, dim)
        vb = v_cache[phys].reshape(-1, n_kv_heads, dim)
        kb[seq_len:total] = k_new[b]
        vb[seq_len:total] = v_new[b]
        kt = kb[:total].transpose(0, 1).unsqueeze(0)  # (1, n_kv_heads, total, dim)
        vt = vb[:total].transpose(0, 1).unsqueeze(0)
        qt = q[b].transpose(0, 1).unsqueeze(0)         # (1, n_q_heads, q_len, dim)
        if causal:
            mask = causal_lower_right(q_len, total)
            o = torch.nn.functional.scaled_dot_product_attention(
                qt, kt, vt, attn_mask=mask, enable_gqa=(n_q_heads != n_kv_heads), scale=sm_scale,
            )
        else:
            o = torch.nn.functional.scaled_dot_product_attention(
                qt, kt, vt, enable_gqa=(n_q_heads != n_kv_heads), scale=sm_scale,
            )
        outs.append(o.squeeze(0).transpose(0, 1))
    return torch.stack(outs)


# -------------------------------------------------------------------------------------------
# gate declines
# -------------------------------------------------------------------------------------------

def test_is_pow2_helper():
    # sanity: the gate predicate itself
    assert _is_pow2(64) and _is_pow2(128) and _is_pow2(256)
    assert not _is_pow2(72) and not _is_pow2(96) and not _is_pow2(112) and not _is_pow2(160)


def test_fa2_declines_non_pow2():
    _need_cuda()
    if not has_fa2:
        pytest.skip("flash-attn not available")
    g = torch.Generator(device=device).manual_seed(0)
    for dim in (72, 96, 160):
        q, k, v = _rand_qkv(1, 32, 16, 16, dim, g=g)
        sm = dim ** -0.5
        # cacheless
        a = AttnArgs(1, 32, 16, dim, 32, 16, q, k, v, None, None, False, sm,
                     None, None, None, 0.0, None, None, None, None, None)
        assert fn_flash_attn_func(a) is None, f"fn_flash_attn_func accepted non-pow2 dim={dim}"
        # cached
        kc = torch.zeros(1, 64, 16, dim, dtype=torch.half, device=device)
        vc = torch.zeros(1, 64, 16, dim, dtype=torch.half, device=device)
        bt = torch.arange(64, dtype=torch.int32, device=device).unsqueeze(0)
        cs = torch.tensor([0], dtype=torch.int32, device=device)
        a = AttnArgs(1, 32, 16, dim, 32, 16, q, k, v, kc, vc, True, sm,
                     None, None, None, 0.0, bt, cs, None, None, None)
        assert fn_flash_attn_with_kvcache(a) is None, f"fn_flash_attn_with_kvcache accepted non-pow2 dim={dim}"
        # varlen
        cu = torch.tensor([0, 32], dtype=torch.int32, device=device)
        a = AttnArgs(1, 32, 16, dim, 32, 16, q, k, v, None, None, False, sm,
                     cu, 32, None, 0.0, None, None, None, None, None)
        assert fn_flash_attn_varlen_func(a) is None, f"fn_flash_attn_varlen_func accepted non-pow2 dim={dim}"


def test_fa2_accepts_pow2():
    """Control: pow2 dims are NOT declined by the FA2 gates (when fa2 is present)."""
    _need_cuda()
    if not has_fa2:
        pytest.skip("flash-attn not available")
    g = torch.Generator(device=device).manual_seed(0)
    dim = 64
    q, k, v = _rand_qkv(1, 32, 16, 16, dim, g=g)
    sm = dim ** -0.5
    a = AttnArgs(1, 32, 16, dim, 32, 16, q, k, v, None, None, False, sm,
                 None, None, None, 0.0, None, None, None, None, None)
    out = fn_flash_attn_func(a)
    assert out is not None and out.shape == q.shape


def test_triton_paged_declines_non_pow2():
    """fn_triton_paged_attn must decline non-pow2 (return None) rather than raising."""
    _need_cuda()
    if not has_triton:
        pytest.skip("triton not available")
    g = torch.Generator(device=device).manual_seed(0)
    dim = 72
    q, k, v = _rand_qkv(1, 16, 16, 16, dim, g=g)
    kc = torch.zeros(1, 64, 16, dim, dtype=torch.half, device=device)
    vc = torch.zeros(1, 64, 16, dim, dtype=torch.half, device=device)
    bt = torch.arange(64, dtype=torch.int32, device=device).unsqueeze(0)
    cs = torch.tensor([0], dtype=torch.int32, device=device)
    sm = dim ** -0.5
    a = AttnArgs(1, 16, 16, dim, 16, 16, q, k, v, kc, vc, True, sm,
                 None, None, None, 0.0, bt, cs, None, None, None)
    # Must return None, not raise ValueError
    assert fn_triton_paged_attn(a) is None


# -------------------------------------------------------------------------------------------
# SDPA fallback accepts non-pow2 and is correct/deterministic
# -------------------------------------------------------------------------------------------

def test_sdpa_nocache_non_pow2_correct():
    _need_cuda()
    g = torch.Generator(device=device).manual_seed(1)
    dim = 72
    q, k, v = _rand_qkv(1, 64, 16, 16, dim, g=g)
    sm = dim ** -0.5
    a = AttnArgs(1, 64, 16, dim, 64, 16, q, k, v, None, None, False, sm,
                 None, None, None, 0.0, None, None, None, None, None)
    out = fn_torch_sdpa_fallback_nocache(a)
    assert out is not None and out.shape == q.shape
    ref = _ref_sdpa_nocache(q, k, v, sm, causal=False)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2), "non-pow2 nocache SDPA mismatch"


def test_sdpa_cache_non_pow2_correct():
    _need_cuda()
    g = torch.Generator(device=device).manual_seed(2)
    dim = 72
    block_size = 256
    cache_len = 48
    q_len = 16
    n_blocks = 4
    q, k_new, v_new = _rand_qkv(1, q_len, 16, 16, dim, g=g)
    # pre-fill the cache with some context
    _, k_ctx, v_ctx = _rand_qkv(1, cache_len, 16, 16, dim, g=g)
    k_cache = torch.zeros(n_blocks, block_size, 16, dim, dtype=torch.half, device=device)
    v_cache = torch.zeros(n_blocks, block_size, 16, dim, dtype=torch.half, device=device)
    k_cache.view(n_blocks * block_size, 16, dim)[:cache_len] = k_ctx[0]
    v_cache.view(n_blocks * block_size, 16, dim)[:cache_len] = v_ctx[0]
    block_table = torch.arange(n_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([cache_len], dtype=torch.int32, device=device)
    sm = dim ** -0.5
    a = AttnArgs(1, q_len, 16, dim, q_len, 16, q, k_new, v_new,
                 k_cache, v_cache, True, sm, None, None, None, 0.0,
                 block_table, cache_seqlens, None, None, None)
    out = fn_torch_sdpa_fallback_cache(a)
    assert out is not None and out.shape == q.shape
    ref = _ref_sdpa_cached(q, k_new, v_new, k_cache.clone(), v_cache.clone(),
                           block_table, cache_seqlens, sm, causal=True)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2), "non-pow2 cached SDPA mismatch"


def test_sdpa_cache_declines_pow2_small():
    """Pow2 dims < 512 must still be declined (left to the fast kernels)."""
    _need_cuda()
    g = torch.Generator(device=device).manual_seed(3)
    dim = 64
    q, k, v = _rand_qkv(1, 16, 16, 16, dim, g=g)
    kc = torch.zeros(1, 64, 16, dim, dtype=torch.half, device=device)
    vc = torch.zeros(1, 64, 16, dim, dtype=torch.half, device=device)
    bt = torch.arange(64, dtype=torch.int32, device=device).unsqueeze(0)
    cs = torch.tensor([0], dtype=torch.int32, device=device)
    sm = dim ** -0.5
    a = AttnArgs(1, 16, 16, dim, 16, 16, q, k, v, kc, vc, True, sm,
                 None, None, None, 0.0, bt, cs, None, None, None)
    assert fn_torch_sdpa_fallback_cache(a) is None, "SDPA cache fallback must decline pow2 < 512"


# -------------------------------------------------------------------------------------------
# end-to-end dispatch
# -------------------------------------------------------------------------------------------

def test_dispatch_cacheless_non_pow2_deterministic():
    """The full dispatcher must route non-pow2 cacheless attention to a deterministic backend
    (SDPA nocache) and produce identical output across repeated calls."""
    _need_cuda()
    g = torch.Generator(device=device).manual_seed(4)
    dim = 72
    q, k, v = _rand_qkv(1, 64, 16, 16, dim, g=g)
    sm = dim ** -0.5
    o1 = attn_dispatch(q, k, v, cache=None, causal=False, sm_scale=sm)
    o2 = attn_dispatch(q, k, v, cache=None, causal=False, sm_scale=sm)
    assert o1.shape == q.shape
    assert torch.equal(o1, o2), "non-pow2 cacheless dispatch nondeterministic"