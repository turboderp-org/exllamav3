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

    order = order.repeat_interleave(2)
    m1 = torch.tensor([-1], device = device, dtype = torch.int)
    order = torch.cat([m1, order, m1], dim = -1)
    if not full:
        order = torch.cat([order, order], dim = -1)

    ref_cache = cache.clone()
    ref_order = order.tolist()

    for _ in range(3):
        temp = torch.empty_like(ref_cache[0])
        for i in range(0, len(ref_order), 2):
            a = ref_order[i]
            b = ref_order[i + 1]
            dst = ref_cache[a, ...] if a >= 0 else temp
            src = ref_cache[b, ...] if b >= 0 else temp
            dst.copy_(src)

        temp = torch.empty_like(cache[0])
        ext.cache_rotate(cache, order, temp)

        torch.testing.assert_close(cache, ref_cache, rtol = 0, atol = 0)

