"""
Paged KV cache quantization (ext.quant_cache_paged / ext.dequant_cache_paged): quantizing tokens through a block
table and dequantizing them again reproduces the cache contents to 8-bit tolerance, across batch/page layouts, head
dims and KV head counts, for incremental appends, a shuffled block table and full rewrites; with a sliding window
every in-window token is dequantized. The reference is the unquantized cache itself (8-bit only: this tests the
paging and window logic, not quantization accuracy).

Empty inputs of the contiguous and paged quant/dequant (test_empty_*): no rows, no tokens or a zero head dim is a
no-op with outputs untouched, the bitrate check still applies, and no CUDA error is left pending. Tokens to quantize
through a block table without pages raise.
"""

import random

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

page_size = 256
block_table_sizes = [(1,4), (1,8), (3, 4), (8,2)]
head_dims = [128, 64, 96, 32, 256]
num_kv_headss = [8, 2, 1]
cache_sizes = [32768]
bitss = [8]  # Not testing accuracy, so 8-bit only to test the paging logic


# Token contents are random: the quantizer Hadamard-rotates each 32-group and quantizes on a midpoint grid, so a
# constant group (31 exactly-zero coefficients) is a worst case whose rounding errors all land on element 0
def token_fill(gen, num_kv_heads, head_dim, device):
    return torch.randn((num_kv_heads, head_dim), generator = gen).half().to(device)


@pytest.mark.parametrize("block_table_size", block_table_sizes)
@pytest.mark.parametrize("head_dim", head_dims)
@pytest.mark.parametrize("num_kv_heads", num_kv_headss)
@pytest.mark.parametrize("cache_size", cache_sizes)
@pytest.mark.parametrize("bits", bitss)
@torch.inference_mode()
def test_kv_quant(device, block_table_size, head_dim, num_kv_heads, cache_size, bits):

    torch.manual_seed(0)

    bsz, pages = block_table_size

    block_table = torch.arange(bsz * pages, dtype = torch.int, device = device).view(bsz, pages)
    cache_seqlens = torch.zeros(size = (bsz,), dtype = torch.int, device = device)

    cache_shape = (cache_size // page_size, page_size, num_kv_heads, head_dim)
    cache_k_tensor = torch.zeros(cache_shape, dtype = torch.half, device = device)
    cache_v_tensor = torch.zeros(cache_shape, dtype = torch.half, device = device)
    cache_k_tensor_out = torch.zeros_like(cache_k_tensor)
    cache_v_tensor_out = torch.zeros_like(cache_v_tensor)

    qcache_shape = (cache_size // page_size, page_size, num_kv_heads * head_dim // 32 * bits)
    qscales_shape = (cache_size // page_size, page_size, num_kv_heads * head_dim // 32)
    cache_k_q = torch.zeros(qcache_shape, dtype = torch.int, device = device)
    cache_v_q = torch.zeros(qcache_shape, dtype = torch.int, device = device)
    cache_k_s = torch.zeros(qscales_shape, dtype = torch.half, device = device)
    cache_v_s = torch.zeros(qscales_shape, dtype = torch.half, device = device)


    def q(length):
        ext.quant_cache_paged(
            cache_k_tensor,
            cache_k_q,
            cache_k_s,
            cache_v_tensor,
            cache_v_q,
            cache_v_s,
            cache_seqlens,
            block_table,
            page_size,
            length,
            0.0,      # compand_a: no companding
            False,    # in_contiguous: the input is the pool-shaped cache tensor
        )

    def dq():
        ext.dequant_cache_paged(
            cache_k_q,
            cache_k_s,
            cache_k_tensor_out,
            cache_v_q,
            cache_v_s,
            cache_v_tensor_out,
            cache_seqlens,
            block_table,
            page_size,
            -1,
            0.0,      # compand_a
        )

    gen = torch.Generator().manual_seed(1)
    def tq():
        torch.testing.assert_close(cache_k_tensor, cache_k_tensor_out, atol = 0.08, rtol = 0.05)
        torch.testing.assert_close(cache_v_tensor, cache_v_tensor_out, atol = 0.08, rtol = 0.05)

    # Put some stuff in cache
    for i in range(bsz):
        cache_seqlens[i] = i
        cache_k_tensor[block_table[i, 0], i] = token_fill(gen, num_kv_heads, head_dim, device)
        cache_v_tensor[block_table[i, 0], i] = token_fill(gen, num_kv_heads, head_dim, device)
    q(1)
    for i in range(bsz):
        cache_seqlens[i] += 1
    dq()
    torch.cuda.synchronize(device)
    tq()

    # Put more stuff in the cache
    new_cache_seqlens = torch.zeros_like(cache_seqlens)
    random.seed(0)
    for i in range(bsz):
        l = random.randint(10, pages * page_size - 2)
        new_cache_seqlens[i] = l
        for j in range(l):
            cache_k_tensor[block_table[i, j // page_size], j % page_size] = token_fill(gen, num_kv_heads, head_dim, device)
            cache_v_tensor[block_table[i, j // page_size], j % page_size] = token_fill(gen, num_kv_heads, head_dim, device)
    cache_seqlens[:] = 0
    q(new_cache_seqlens.amax())
    cache_seqlens.copy_(new_cache_seqlens)
    dq()
    torch.cuda.synchronize(device)
    tq()

    # Mess up pages
    block_table = block_table.flatten()[torch.randperm(block_table.numel())].view(block_table.shape)
    cache_k_q[:, :, :] = 0
    cache_v_q[:, :, :] = 0
    cache_k_s[:, :, :] = 0
    cache_v_s[:, :, :] = 0
    for i in range(bsz):
        l = new_cache_seqlens[i]
        for j in range(l):
            cache_k_tensor[block_table[i, j // page_size], j % page_size, :, :] += 1
            cache_v_tensor[block_table[i, j // page_size], j % page_size, :, :] += 1
    cache_seqlens[:] = 0
    q(new_cache_seqlens.amax())
    cache_seqlens.copy_(new_cache_seqlens)
    dq()
    torch.cuda.synchronize(device)
    tq()

    # Update five tokens
    for i in range(bsz):
        l = cache_seqlens[i]
        for j in range(5):
            pos = l + j
            cache_k_tensor[block_table[i, pos // page_size], pos % page_size] = token_fill(gen, num_kv_heads, head_dim, device)
            cache_v_tensor[block_table[i, pos // page_size], pos % page_size] = token_fill(gen, num_kv_heads, head_dim, device)
    q(5)
    for i in range(bsz):
        cache_seqlens[i] += 5
    dq()
    torch.cuda.synchronize(device)
    tq()


# (num_kv_heads, head_dim). The dequant kernel walks the sequence in 4-group chunks, a fixed number per thread
# block, and with a sliding window skips the blocks below it. Widths whose chunks per token don't divide the block's
# chunk count put a token across two blocks, which is the case the skip has to get right; the pow2 widths are the
# aligned control
window_geometries = [(8, 128), (4, 128), (8, 96), (6, 128), (12, 64), (3, 128), (5, 64)]
window_sizes = [1, 64, 300]
window_chunks_per_block = 256


@pytest.mark.parametrize("geometry", window_geometries)
@pytest.mark.parametrize("window", window_sizes)
@pytest.mark.parametrize("bits", bitss)
@torch.inference_mode()
def test_kv_quant_sliding_window(device, geometry, window, bits):
    """
    With a sliding window, every token in [seqlen - window, seqlen) must be dequantized in full. The sequence
    lengths are chosen to put the oldest in-window token on and around each thread block boundary.
    """
    num_kv_heads, head_dim = geometry
    pages = 4
    max_len = pages * page_size
    groups = num_kv_heads * head_dim // 32
    chunks_per_token = -(-groups // 4)

    block_table = torch.arange(pages, dtype = torch.int, device = device).view(1, pages)
    cache_seqlens = torch.zeros(size = (1,), dtype = torch.int, device = device)

    gen = torch.Generator().manual_seed(1)
    cache_shape = (pages, page_size, num_kv_heads, head_dim)
    cache_k_tensor = torch.randn(cache_shape, generator = gen).half().to(device)
    cache_v_tensor = torch.randn(cache_shape, generator = gen).half().to(device)
    cache_k_q = torch.zeros((pages, page_size, groups * bits), dtype = torch.int, device = device)
    cache_v_q = torch.zeros_like(cache_k_q)
    cache_k_s = torch.zeros((pages, page_size, groups), dtype = torch.half, device = device)
    cache_v_s = torch.zeros_like(cache_k_s)

    ext.quant_cache_paged(
        cache_k_tensor, cache_k_q, cache_k_s,
        cache_v_tensor, cache_v_q, cache_v_s,
        cache_seqlens, block_table, page_size, max_len, 0.0, False,
    )

    # Oldest in-window token = first token of each thread block, and its neighbours
    oldest = set()
    for chunk in range(window_chunks_per_block, max_len * chunks_per_token, window_chunks_per_block):
        token = chunk // chunks_per_token
        oldest.update((token - 1, token, token + 1))
    seqlens = sorted(t + window for t in oldest if 0 <= t and t + window <= max_len)
    assert seqlens

    ref_k = cache_k_tensor.view(max_len, -1)
    ref_v = cache_v_tensor.view(max_len, -1)
    for seqlen in seqlens:
        cache_seqlens[0] = seqlen
        # Anything the kernel leaves unwritten inside the window stays NaN and fails the comparison
        out_k = torch.full(cache_shape, float("nan"), dtype = torch.half, device = device)
        out_v = torch.full(cache_shape, float("nan"), dtype = torch.half, device = device)
        ext.dequant_cache_paged(
            cache_k_q, cache_k_s, out_k,
            cache_v_q, cache_v_s, out_v,
            cache_seqlens, block_table, page_size, window, 0.0,
        )
        a, b = seqlen - window, seqlen
        torch.testing.assert_close(out_k.view(max_len, -1)[a:b], ref_k[a:b], atol = 0.08, rtol = 0.05)
        torch.testing.assert_close(out_v.view(max_len, -1)[a:b], ref_v[a:b], atol = 0.08, rtol = 0.05)


def _device_still_works(device):
    # A launch error left pending by the call would surface in the next unrelated launch
    y = torch.ones(4, device = device) * 2
    torch.cuda.synchronize(device)
    assert y.sum().item() == 8


@pytest.mark.parametrize("rows, dim", [(0, 128), (0, 0), (5, 0)])
@torch.inference_mode()
def test_empty_cache_cont(device, rows, dim):
    bits = 4
    x = torch.randn((rows, dim), device = device).half()
    q = torch.full((rows, dim // 32 * bits), 7, dtype = torch.int, device = device)
    sc = torch.full((rows, dim // 32), 7.0, dtype = torch.half, device = device)
    ext.quant_cache_cont(x, q, sc, 0.0)
    out = torch.full((rows, dim), 7.0, dtype = torch.half, device = device)
    ext.dequant_cache_cont(q, sc, out, 0.0)
    assert (out == 7.0).all()
    _device_still_works(device)
    if dim:
        with pytest.raises(RuntimeError, match = "bitrate"):
            ext.quant_cache_cont(x, q[:, :dim // 32], sc, 0.0)


@pytest.mark.parametrize("bsz, seq_len, dim", [(0, 3, 256), (2, 0, 256), (2, 3, 0), (0, 0, 0)])
@torch.inference_mode()
def test_empty_quant_cache_paged(device, bsz, seq_len, dim):
    bits = 4
    num_pages = 4
    x = torch.randn((num_pages, page_size, dim), device = device).half()
    q = torch.full((num_pages, page_size, dim // 32 * bits), 7, dtype = torch.int, device = device)
    sc = torch.full((num_pages, page_size, dim // 32), 7.0, dtype = torch.half, device = device)
    q_ref, sc_ref = q.clone(), sc.clone()
    bt = torch.arange(bsz * 2, dtype = torch.int, device = device).view(bsz, 2)
    sl = torch.zeros((bsz,), dtype = torch.int, device = device)
    ext.quant_cache_paged(x, q, sc, x, q, sc, sl, bt, page_size, seq_len, 0.0, False)
    assert torch.equal(q, q_ref) and torch.equal(sc, sc_ref)
    _device_still_works(device)
    if dim:
        with pytest.raises(RuntimeError, match = "bitrate"):
            ext.quant_cache_paged(x, q[..., :1], sc, x, q[..., :1], sc, sl, bt, page_size, seq_len, 0.0, False)
    with pytest.raises(RuntimeError, match = "negative seq_len"):
        ext.quant_cache_paged(x, q, sc, x, q, sc, sl, bt, page_size, -1, 0.0, False)


@torch.inference_mode()
def test_empty_block_table_rejected(device):
    x = torch.zeros((2, page_size, 64), dtype = torch.half, device = device)
    q = torch.zeros((2, page_size, 8), dtype = torch.int, device = device)
    sc = torch.zeros((2, page_size, 2), dtype = torch.half, device = device)
    bt = torch.zeros((1, 0), dtype = torch.int, device = device)
    sl = torch.zeros((1,), dtype = torch.int, device = device)
    with pytest.raises(RuntimeError, match = "quant_cache_paged: .*no pages"):
        ext.quant_cache_paged(x, q, sc, x, q, sc, sl, bt, page_size, 1, 0.0, False)
    _device_still_works(device)
