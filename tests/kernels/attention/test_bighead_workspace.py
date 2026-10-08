"""
ext.bighead_attn_workspace_size(bsz, q_len, n_q_heads, max_kv_len, kv_chunk_size, dim): the number of fp32 elements
of the split-kv partials the bighead attention kernels address, laid out [bsz, q_len, n_q_heads, n_chunks, dim + 2]
(dim output accumulators plus running max and sum per chunk), n_chunks = ceil(max_kv_len / kv_chunk_size). Closed
form, computed in 64 bits (no int32 wrap for large products). An empty batch, query, head or kv extent gives zero elements; a
non-positive kv_chunk_size has no chunk count and raises.

Consistency with the kernels: bighead_attn / bighead_attn_paged no longer take a workspace; they use the fixed 16 MiB
device workspace (exl3_devctx.cuh WORKSPACE_SIZE) and double kv_chunk_size until this same element count fits. A
configuration whose requested chunk size needs more than that must still attend correctly (fp32 reference
testlib.attention.ref_attn, same tolerance as test_bighead_attn.py). One whose query extent alone (a single chunk)
does not fit raises.

Empty inputs of the kernels (test_empty_*): no rows or no queries is a no-op (the paged variant still appends the
new k / v); attention over zero keys (kv_len 0, or a block table without pages) is undefined and raises, as do
zero kv heads and unsupported head dims / GQA ratios, also on empty inputs. Outputs are untouched and no CUDA
error is left pending.
"""
import re

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib import isolated
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


def _workspace_overflow_worker(device: str, paged: bool):
    # bsz * q_len * n_q_heads * (dim + 2) floats exceed the device workspace even with a single KV chunk
    torch.cuda.set_device(torch.device(device))
    bsz, q_len, hq, hkv, kv_len, dim = 1, 512, 16, 2, 256, 512
    q = torch.zeros(bsz, q_len, hq, dim, dtype = torch.half, device = device)
    k = torch.zeros(bsz, kv_len, hkv, dim, dtype = torch.half, device = device)
    o = torch.full_like(q, 7.0)
    try:
        if paged:
            kc = torch.zeros(1, 256, hkv, dim, dtype = torch.half, device = device)
            bt = torch.zeros((1, 1), dtype = torch.int32, device = device)
            sl = torch.zeros((1,), dtype = torch.int32, device = device)
            ext.bighead_attn_paged(q, k[:, :0], k[:, :0], kc, kc.clone(), bt, sl, o, 64, True, 0.0)
        else:
            ext.bighead_attn(q, k, k, o, 64, True, 0.0)
        msg = None
    except RuntimeError as e:
        msg = str(e)
    torch.cuda.synchronize()
    return msg, bool((o == 7.0).all())


@pytest.mark.parametrize("paged", [False, True])
def test_workspace_overflow_rejected(device, paged):
    """A query extent too large for the workspace at any chunk size raises (the chunk-doubling loop would
    otherwise never terminate, so the call runs in a child process under a timeout)"""
    assert ext.bighead_attn_workspace_size(1, 512, 16, 1, 1, 512) > DEV_WORKSPACE_FLOATS
    msg, untouched = isolated.run_isolated(_workspace_overflow_worker, str(device), paged, timeout = 300)
    fn = "bighead_attn_paged" if paged else "bighead_attn"
    assert msg is not None and re.search(f"{fn}: .*too large for the attention workspace", msg), msg
    assert untouched


@pytest.mark.parametrize("bsz,q_len,heads,kv,dim", [(0, 1, 8, 512, 64), (1, 0, 8, 512, 64), (1, 1, 0, 512, 64), (1, 1, 8, 0, 64)])
@pytest.mark.nogpu
def test_empty_workspace_size(bsz, q_len, heads, kv, dim):
    assert ext.bighead_attn_workspace_size(bsz, q_len, heads, kv, 128, dim) == 0


@pytest.mark.parametrize("chunk", [0, -128])
@pytest.mark.nogpu
def test_workspace_size_rejects_chunk(chunk):
    with pytest.raises(RuntimeError, match = "kv_chunk_size must be positive"):
        ext.bighead_attn_workspace_size(1, 1, 8, 512, chunk, 64)


def _device_still_works(device):
    # A launch error left pending by the call would surface in the next unrelated launch
    y = torch.ones(4, device = device) * 2
    torch.cuda.synchronize(device)
    assert y.sum().item() == 8


@pytest.mark.parametrize("bsz,q_len,kv_len,n_q_heads,n_kv_heads,dim,error", [
    (0, 1, 64, 8, 2, 128, None),
    (2, 0, 64, 8, 2, 128, None),
    (2, 1, 0, 8, 2, 128, "attention over zero keys"),
    (0, 0, 0, 8, 2, 128, "attention over zero keys"),
    (2, 1, 64, 0, 0, 128, "n_kv_heads must be positive"),
    (2, 1, 64, 0, 2, 128, "head_dim must be"),
    (0, 1, 64, 8, 2, 96, "head_dim must be"),
])
@torch.inference_mode()
def test_empty_bighead_attn(device, bsz, q_len, kv_len, n_q_heads, n_kv_heads, dim, error):
    q = torch.randn(bsz, q_len, n_q_heads, dim, dtype = torch.float16, device = device)
    k = torch.randn(bsz, kv_len, n_kv_heads, dim, dtype = torch.float16, device = device)
    o = torch.full_like(q, 7.0)
    if error:
        with pytest.raises(RuntimeError, match = error):
            ext.bighead_attn(q, k, k, o, 128, True, 0.0)
    else:
        ext.bighead_attn(q, k, k, o, 128, True, 0.0)
    assert (o == 7.0).all()
    _device_still_works(device)


@pytest.mark.parametrize("bsz,q_len,kv_append,n_q_heads,n_kv_heads,dim,pages,error", [
    (0, 1, 1, 8, 2, 128, 2, None),
    (2, 0, 3, 8, 2, 128, 2, None),          # no queries: the append still runs
    (2, 0, 0, 8, 2, 128, 2, None),
    (2, 1, 1, 8, 2, 128, 0, "attention over zero keys"),
    (0, 0, 0, 8, 2, 128, 0, "attention over zero keys"),
    (2, 1, 1, 0, 0, 128, 2, "n_kv_heads must be positive"),
    (0, 1, 1, 8, 2, 96, 2, "head_dim must be"),
])
@torch.inference_mode()
def test_empty_bighead_attn_paged(device, bsz, q_len, kv_append, n_q_heads, n_kv_heads, dim, pages, error):
    torch.manual_seed(0)
    q = torch.randn(bsz, q_len, n_q_heads, dim, dtype = torch.float16, device = device)
    k = torch.randn(bsz, kv_append, n_kv_heads, dim, dtype = torch.float16, device = device)
    v = torch.randn(bsz, kv_append, n_kv_heads, dim, dtype = torch.float16, device = device)
    num_pages = max(bsz * pages, 1)
    k_cache = torch.full((num_pages, 256, n_kv_heads, dim), 7.0, dtype = torch.half, device = device)
    v_cache = torch.full_like(k_cache, 7.0)
    block_table = torch.arange(bsz * pages, dtype = torch.int32, device = device).view(bsz, pages)
    cache_seqlens = torch.full((bsz,), 5, dtype = torch.int32, device = device)
    o = torch.full_like(q, 7.0)
    k_ref, v_ref = k_cache.clone(), v_cache.clone()
    for b in range(bsz if not error else 0):
        k_ref[b * pages, 5 : 5 + kv_append] = k[b]
        v_ref[b * pages, 5 : 5 + kv_append] = v[b]
    args = (q, k, v, k_cache, v_cache, block_table, cache_seqlens, o, 128, True, 0.0)
    if error:
        with pytest.raises(RuntimeError, match = error):
            ext.bighead_attn_paged(*args)
    else:
        ext.bighead_attn_paged(*args)
    assert (o == 7.0).all()
    assert torch.equal(k_cache, k_ref) and torch.equal(v_cache, v_ref)
    _device_still_works(device)
