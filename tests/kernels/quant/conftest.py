import pytest
import torch

from exllamav3.modules.quant.exl3_lib.quantize import get_temp_buffers, get_temp_buffers_frac


@pytest.fixture(autouse = True)
def release_quant_scratch():
    """The quantizer's scratch buffers are memoized per (device, K, tile length, codebook), and the generic kernels'
    edge histories run to GBs at low K (4 GiB at K = 1). Conversion clears the cache per layer; the tests sweep K and
    codebooks, so they drop it after every test or the cached buffers fill a 24 GB card"""
    yield
    get_temp_buffers.cache_clear()
    get_temp_buffers_frac.cache_clear()
    torch.cuda.empty_cache()
