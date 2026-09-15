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
    # Pages that are not a multiple of 16 bytes: DeepSeek-V4 HCA pools (128:1) hold 2 entries
    # per token page, so the quantized scale plane is (2, 14) fp16 = 56 bytes (issue #375);
    # 8/4/2-byte-aligned and odd sizes cover the narrower copy widths
    [300, 2, 14],          # 56 B
    [300, 2, 12],          # 48 B (16-aligned, small)
    [300, 3, 3],           # 18 B
    [300, 7, 11],          # 154 B
    [300, 1, 5, "u8"],     # 5 B (odd)
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
    if cache_dim[-1] == "u8":
        cache = torch.randint(0, 256, cache_dim[:-1], device = device, dtype = torch.uint8)
    else:
        cache = torch.randn(cache_dim, device = device, dtype = cache_dtype)
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



@pytest.mark.parametrize("k_bits", [0, 4, 8])
@torch.inference_mode()
def test_rotate_dsv4_hca_pool_layer(k_bits):
    """Issue #375: PageTable.defrag rotates every tensor of every cache layer. A DeepSeek-V4 HCA
    layer (128:1 compression, head_dim 512 of which 64 rope) with a quantized pool has a scale
    plane of (2, 14) fp16 = 56 bytes per page, which the kernel used to reject."""
    from types import SimpleNamespace
    from exllamav3.cache.dsa import CacheLayer_dsa
    attn = SimpleNamespace(compress_rate = 128, head_dim = 512, rope_head_dim = 64, index_head_dim = 128, layer_type = "hca")
    layer = CacheLayer_dsa(None, attn, 0, 64 * page_size, k_bits = k_bits)
    layer.alloc(torch.device(device))
    tensors = layer.get_tensors()
    sizes = {t.nbytes // t.shape[0] for t in tensors}
    if k_bits:
        assert 56 in sizes, f"expected the 56-byte scale page, got {sizes}"
    for t in tensors:
        t.copy_(torch.randint(-2**31, 2**31 - 1, t.shape, device = device, dtype = torch.int32).view(t.dtype) if t.dtype == torch.int32
                else torch.randn(t.shape, device = device, dtype = t.dtype))
    order = torch.randperm(64, device = device, dtype = torch.int).repeat_interleave(2)
    m1 = torch.tensor([-1], device = device, dtype = torch.int)
    order = torch.cat([m1, order, m1])
    ref_order = order.tolist()
    for t in tensors:
        ref = t.clone(); temp = torch.empty_like(ref[0])
        for i in range(0, len(ref_order), 2):
            a, b = ref_order[i], ref_order[i + 1]
            (ref[a] if a >= 0 else temp).copy_(ref[b] if b >= 0 else temp)
        ext.cache_rotate(t, order, torch.empty_like(t[0]))
        torch.testing.assert_close(t, ref, rtol = 0, atol = 0)
    layer.free()
