"""
Conversion releases the quantizer's memoized scratch before each module and before each module reloads. Both caches
must go: the half-integer bitrates (1.5 / 2.5 / 3.5 bpw) keep their own scratch, 2 GiB at 1.5 bpw, which otherwise
stays allocated for the rest of the run and leaves no room to reload the module.
"""

import inspect
from functools import lru_cache

import pytest
import torch

from exllamav3.conversion import convert_model
from exllamav3.modules.quant.exl3_lib import quantize


@pytest.mark.nogpu
def test_clear_temp_buffers_releases_both_caches(monkeypatch):
    # Stand-ins with the same lru_cache wrapper, so the check needs no device memory
    for name in ("get_temp_buffers", "get_temp_buffers_frac"):
        cached = lru_cache(lambda *args: object())
        cached(0, 1)
        monkeypatch.setattr(quantize, name, cached)
    quantize.clear_temp_buffers()
    assert quantize.get_temp_buffers.cache_info().currsize == 0
    assert quantize.get_temp_buffers_frac.cache_info().currsize == 0


def test_clear_temp_buffers_frees_the_frac_scratch_on_the_device(device):
    # The 1.5 bpw scratch has to go back to the allocator, not just out of the cache
    quantize.clear_temp_buffers()
    costs, edges = quantize.get_temp_buffers_frac(device, 1)
    size = costs.nbytes + edges.nbytes
    del costs, edges
    before = torch.cuda.memory_allocated(device)
    quantize.clear_temp_buffers()
    assert before - torch.cuda.memory_allocated(device) >= size


@pytest.mark.nogpu
def test_conversion_clears_through_the_helper():
    # The three release points (before each module, before each reload, side-model loop) must clear both caches
    source = inspect.getsource(convert_model)
    assert "get_temp_buffers.cache_clear" not in source
    assert "get_temp_buffers_frac.cache_clear" not in source
    assert source.count("clear_temp_buffers()") >= 3
