"""
ext.bighead_attn_workspace_size(bsz, q_len, n_q_heads, max_kv_len, kv_chunk_size, dim): the number of fp32 elements
of the split-kv partials the bighead attention kernels address, laid out [bsz, q_len, n_q_heads, n_chunks, dim + 2]
(dim output accumulators plus running max and sum per chunk), n_chunks = ceil(max_kv_len / kv_chunk_size). Closed
form, computed in 64 bits (no int32 wrap for large products).

Consistency with the kernels: bighead_attn / bighead_attn_paged no longer take a workspace; they use the fixed 16 MiB
device workspace (exl3_devctx.cuh WORKSPACE_SIZE) and double kv_chunk_size until this same element count fits. A
configuration whose requested chunk size needs more than that must still attend correctly (fp32 reference
testlib.attention.ref_attn, same tolerance as test_bighead_attn.py).
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.attention import ref_attn

DEV_WORKSPACE_FLOATS = 16 * 1024 * 1024 // 4


@pytest.mark.parametrize("bsz,q_len,heads,kv,chunk,dim", [
    (1, 1, 8, 512, 128, 64), (1, 1, 8, 500, 128, 64), (1, 1, 4, 129, 64, 64), (3, 4, 16, 4096, 512, 512),
    (2, 1, 1, 1, 1, 128), (1, 1, 1, 0, 128, 64), (7, 8, 16, 32768, 128, 256),
])
@pytest.mark.nogpu
def test_workspace_size_closed_form(bsz, q_len, heads, kv, chunk, dim):
    n_chunks = -(-kv // chunk)
    assert ext.bighead_attn_workspace_size(bsz, q_len, heads, kv, chunk, dim) == bsz * q_len * heads * n_chunks * (dim + 2)


@pytest.mark.nogpu
def test_workspace_size_large():
    # Product far past 2^32
    bsz, q_len, heads, kv, chunk, dim = 64, 512, 64, 1 << 20, 64, 512
    expect = bsz * q_len * heads * (kv // chunk) * (dim + 2)
    assert expect > 2 ** 40
    assert ext.bighead_attn_workspace_size(bsz, q_len, heads, kv, chunk, dim) == expect


@pytest.mark.parametrize("paged", [False, True])
@torch.inference_mode()
def test_chunk_growth_past_device_workspace(device, paged):
    bsz, q_len, hq, hkv, kv_len, dim, chunk = 1, 16, 16, 2, 8192, 512, 64
    assert ext.bighead_attn_workspace_size(bsz, q_len, hq, kv_len, chunk, dim) > DEV_WORKSPACE_FLOATS
    torch.manual_seed(0)
    q = torch.randn(bsz, q_len, hq, dim, dtype = torch.half, device = device)
    k = torch.randn(bsz, kv_len, hkv, dim, dtype = torch.half, device = device)
    v = torch.randn(bsz, kv_len, hkv, dim, dtype = torch.half, device = device)
    ref = ref_attn(q, k, v, causal = True).half()
    o = torch.full_like(q, float("nan"))
    if paged:
        n_pages = kv_len // 256
        kc = torch.zeros(n_pages, 256, hkv, dim, dtype = torch.half, device = device)
        vc = torch.zeros_like(kc)
        bt = torch.arange(n_pages, dtype = torch.int32, device = device).view(1, n_pages)
        sl = torch.zeros((1,), dtype = torch.int32, device = device)
        ext.bighead_attn_paged(q, k, v, kc, vc, bt, sl, o, chunk, True, 0.0)
    else:
        ext.bighead_attn(q, k, v, o, chunk, True, 0.0)
    torch.testing.assert_close(o, ref, atol = 1e-2, rtol = 1e-2)
