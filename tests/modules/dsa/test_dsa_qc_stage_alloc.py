"""
_stage_packed_pool (DSA prefill over a quantized MLA/DSA cache) dequantizes the referenced
window into an fp16 transient rounded up to a power of two in entries. The rounding must never
allocate past the pool itself: a window spanning a pool just above 2^k entries would otherwise
nearly double the transient, and MLAttention.autosplit_extra_measure (a window spanning the
whole pool) would reserve that doubled size on each device holding a DSA layer. Host-side
sizing only: the dequant kernel is stubbed, so this runs without a GPU.
"""

from types import SimpleNamespace

import pytest
import torch

import exllamav3.ext
from exllamav3.modules.attention_fn.dsa_triton import _stage_packed_pool

pytestmark = pytest.mark.nogpu

P, D_c, D_r, bits = 256, 64, 16, 4        # narrow latent: the sizing is per entry, D_c only scales it
G = D_c // 32


@pytest.fixture(autouse = True)
def _stub_dequant(monkeypatch):
    # Sizing is decided before the kernel runs; the stub only fills the destination rows
    stub = SimpleNamespace(dequant_cache_cont = lambda pc, ps, out, _: out.fill_(1.0))
    monkeypatch.setattr(exllamav3.ext, "exllamav3_ext", stub)


def _stage(pool_entries, pool_len):
    pages = pool_entries // P
    pool_c = torch.zeros((pages, P, G * bits), dtype = torch.int32)
    pool_s = torch.zeros((pages, P, G), dtype = torch.half)
    pool_r = torch.zeros((pages, P, D_r), dtype = torch.half)
    bt = torch.arange(pages, dtype = torch.int32).unsqueeze(0)
    return _stage_packed_pool(pool_c, (pool_s, bits), pool_r, bt, P, pool_len, D_c, D_r)


@pytest.mark.parametrize("pool_entries, pool_len, alloc_rows", [
    (524288 + P, 524288 + P, 524288 + P),      # window spans a pool just past 2^19: capped
    (600064, 600064, 600064),                  # window spans the pool: capped
    (600064, 524288 + 1, 600064),              # window rounds past the pool: capped at the pool
    (600064, 300000, 1 << 19),                 # window rounds inside the pool: pow2 as before
    (524288, 524288, 524288),                  # pool is 2^19: pow2 as before
    (1 << 20, 100, P),                         # short window: one page as before
])
def test_stage_transient_never_past_pool(pool_entries, pool_len, alloc_rows):
    pc, pr, bt = _stage(pool_entries, pool_len)
    npw = -(-pool_len // P)
    assert pc.shape == (npw, P, D_c) and pc.dtype == torch.half
    assert pr.shape == (npw, P, D_r)
    assert torch.equal(bt, torch.arange(npw, dtype = torch.int32).unsqueeze(0))
    assert bool((pc == 1.0).all())
    alloc = pc.untyped_storage().nbytes()   # (an int: a failing assert must not repr the storage)
    assert alloc == alloc_rows * D_c * 2

