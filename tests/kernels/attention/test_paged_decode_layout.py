"""
Dense (query, head) row layout of the paged decode split / combine kernels (triton_paged.decode_row_layout): every
GQA group size, multi-token query length, head_dim and cache width (fp16 and packed 4-bit) must give the same
result as the prefill kernel on the same cache, with the output landing in the right (query, head) slot, in the
single-split (FINAL) and the combine path alike. A wrong row mapping shows up as rows swapped between heads or
query positions, which the prefill kernel (its own row tiling) does not share.
"""
import pytest
import torch

from exllamav3.modules.attention_fn.triton_paged import (
    decode_row_layout, paged_attn_triton_decode, paged_attn_triton_prefill,
)
from testlib.attention import rand_packed_cache


def make_caches(n_kv, head_dim, bits, num_pages, page_size, device):
    if bits:
        return rand_packed_cache(num_pages, page_size, n_kv, head_dim, bits, device)
    kc = torch.randn((num_pages, page_size, n_kv, head_dim), dtype = torch.float16, device = device)
    vc = torch.randn((num_pages, page_size, n_kv, head_dim), dtype = torch.float16, device = device)
    return kc, vc, None


@pytest.mark.nogpu
def test_row_layout_shapes():
    # 5-token verify over a 6-head group: one 32-row program
    assert decode_row_layout(5, 6, 128) == (32, 1)
    assert decode_row_layout(5, 6, 256) == (32, 1)
    assert decode_row_layout(1, 6, 128) == (16, 1)
    assert decode_row_layout(1, 16, 128) == (16, 1)
    assert decode_row_layout(8, 8, 128) == (32, 2)
    assert decode_row_layout(1, 1, 256) == (16, 1)


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("bits", [0, 4])
@pytest.mark.parametrize("q_len", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("group", [1, 2, 6, 8, 16])
@pytest.mark.parametrize("num_splits", [1, 16])
@torch.inference_mode()
def test_decode_layout_vs_prefill(device, head_dim, bits, q_len, group, num_splits):
    torch.manual_seed(group * 1000 + q_len * 10 + bits)
    n_kv = 2
    n_q = n_kv * group
    bsz, ctx, page_size = 2, 1024, 256
    num_pages = ctx // page_size
    kc, vc, qc = make_caches(n_kv, head_dim, bits, bsz * num_pages, page_size, device)
    # Distinct per-head, per-position queries so a misplaced row cannot match by accident
    q = torch.randn((bsz, q_len, n_q, head_dim), dtype = torch.float16, device = device)
    bt = torch.arange(bsz * num_pages, dtype = torch.int32, device = device).view(bsz, num_pages)
    sl = torch.tensor([ctx - q_len, ctx - q_len - 37], dtype = torch.int32, device = device)
    common = dict(q = q, k = None, v = None, k_cache = kc, v_cache = vc, block_table = bt, cache_seqlens = sl,
                  causal = True, qc = qc, max_kv_len = ctx, pre_appended_len = q_len,
                  n_kv_heads_override = n_kv if bits else None)
    o_dec = paged_attn_triton_decode(**common, num_splits = num_splits)
    o_pre = paged_attn_triton_prefill(**common)
    torch.testing.assert_close(o_dec, o_pre, rtol = 0, atol = 2e-3)
