import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext
from itertools import pairwise

device = "cuda:2"
page_size = 256

cache_dims = [
    [2048, page_size, 16 * 128],
    [1024, page_size, 8 * 128],
    [512, page_size, 8 * 128],
    [256, page_size, 8 * 128],
    [100, page_size, 4 * 128],
    [32, page_size, 4 * 128],
    [32, page_size, 48],
    [2560, page_size, 16],
]

cache_dtypes = [torch.half, torch.float]

full_opt = [True, False]

@pytest.mark.parametrize("cache_dim", cache_dims)
@pytest.mark.parametrize("cache_dtype", cache_dtypes)
@pytest.mark.parametrize("full", full_opt)
@torch.inference_mode()
def test_rope(cache_dim, cache_dtype, full):

    torch.manual_seed(0)

    num_pages = cache_dim[0]
    cache = torch.randn(cache_dim, device = device, dtype = torch.half)
    order = torch.randperm(num_pages, device = device, dtype = torch.int)
    if not full:
        order = order[:num_pages // 4]

    ref_cache = cache.clone()
    ref_order = order.tolist()

    for _ in range(3):
        temp = torch.empty_like(ref_cache[0])
        temp.copy_(ref_cache[ref_order[0], ...])
        for a, b in pairwise(ref_order):
            ref_cache[a, ...].copy_(ref_cache[b, ...])
        ref_cache[ref_order[-1], ...].copy_(temp)

        temp = torch.empty_like(cache[0])
        ext.cache_rotate(cache, order, temp)

        torch.testing.assert_close(cache, ref_cache, rtol = 0, atol = 0)

