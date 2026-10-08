"""
Paged KV cache quantization (ext.quant_cache_paged / ext.dequant_cache_paged): quantizing tokens through a block
table and dequantizing them again reproduces the cache contents to 8-bit tolerance, across batch/page layouts, head
dims and KV head counts, for incremental appends, a shuffled block table and full rewrites; with a sliding window
every in-window token is dequantized. The reference is the unquantized cache itself (8-bit only: this tests the
paging and window logic, not quantization accuracy).
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
