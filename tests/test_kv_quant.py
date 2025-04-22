import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext
import random

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

devices = [
    "cuda:1"
]

page_size = 256
block_table_sizes = [(1,4), (1,8), (3, 4), (8,2)]
head_dims = [128, 64, 96, 32, 256]
num_kv_headss = [8, 2, 1]
cache_sizes = [32768]
bitss = [8]  # Not testing accuracy, so 8-bit only to test the paging logic

@pytest.mark.parametrize("device", devices)
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
            length
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
            page_size
        )

    def tq():
        torch.testing.assert_close(cache_k_tensor, cache_k_tensor_out, atol = 0.08, rtol = 0.01)
        torch.testing.assert_close(cache_v_tensor, cache_v_tensor_out, atol = 0.08, rtol = 0.01)

    # Put some stuff in cache
    for i in range(bsz):
        cache_seqlens[i] = i
        for h in range(num_kv_heads):
            cache_k_tensor[block_table[i, 0], i, h, :] = h
            cache_v_tensor[block_table[i, 0], i, h, :] = h + num_kv_heads
    q(1)
    for i in range(bsz):
        cache_seqlens[i] += 1
    dq()
    torch.cuda.synchronize()
    tq()

    # Put more stuff in the cache
    new_cache_seqlens = torch.zeros_like(cache_seqlens)
    random.seed(0)
    for i in range(bsz):
        l = random.randint(10, pages * page_size - 2)
        new_cache_seqlens[i] = l
        for j in range(l):
            m = j % 13
            for h in range(num_kv_heads):
                cache_k_tensor[block_table[i, j // page_size], j % page_size, h, :] = h + m
                cache_v_tensor[block_table[i, j // page_size], j % page_size, h, :] = h + m + num_kv_heads
    cache_seqlens[:] = 0
    q(new_cache_seqlens.amax())
    cache_seqlens.copy_(new_cache_seqlens)
    dq()
    torch.cuda.synchronize()
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
    torch.cuda.synchronize()
    tq()

    # Update five tokens
    for i in range(bsz):
        l = cache_seqlens[i]
        for j in range(5):
            pos = l + j
            cache_k_tensor[block_table[i, pos // page_size], + pos % page_size, :, :] = 32 + j
            cache_v_tensor[block_table[i, pos // page_size], + pos % page_size, :, :] = 32 + j
    q(5)
    for i in range(bsz):
        cache_seqlens[i] += 5
    dq()
    tq()

    xx = 0
