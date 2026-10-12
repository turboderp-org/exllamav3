"""
DeepSeek-V4 packed-quantized DSA pools (CacheLayer_dsa with k_bits): the compressor stages this step's entries
and dsv4_pool_quant_scatter packs them into the paged pool, read online by the DSA attention kernels.

  1. Kernel: the scatter's packed rows equal quant_cache_cont on the same rows (bit-exact), the rope columns are
     copied exactly, entries land at position // m + w through the block table (single job and batched jobs with
     device positions), and nothing else in the pool is written.
  2. Model: the tiny V4 (rope 32 of head_dim 64 -> D_c 32) run through the cached path with fp16 pools vs Q8 / Q4
     pools, chunked prefill + decode steps, against the chunk-noise floor.
"""

import pytest
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.ext import exllamav3_ext as ext

from testlib.dsv4 import assert_logits_close, build_tiny_dsv4, fwd_cached, fwd_modules, noise_floor
from testlib.tiny_models import DSV4_TINY


@pytest.mark.parametrize("i, bits, D_c, D_r, m, seq, pos_list", [
    (i, *case) for i, case in enumerate([
        (8, 448, 64, 4, 1, [3]), (8, 448, 64, 4, 1, [4]), (4, 448, 64, 4, 16, [1023]),
        (3, 448, 64, 128, 300, [0]), (6, 96, 32, 4, 13, [255]), (2, 512, 64, 4, 9, [0]),
        (8, 448, 64, 4, 1, [3, 7, 11, 1022]), (5, 448, 64, 4, 16, [0, 5, 1000, 3000]),
        (7, 448, 64, 128, 16, [127, 128, 2000]),
    ])
])
def test_scatter(device, i, bits, D_c, D_r, m, seq, pos_list):
    torch.manual_seed(700 + i)
    hd = D_c + D_r
    G = D_c // 32
    B = len(pos_list)
    epp = PAGE_SIZE // m
    pages = 16
    nw_max = seq // m + 1
    stage = torch.randn((B, nw_max, hd), dtype = torch.half, device = device)
    pool_q = torch.zeros((pages * epp, G * bits), dtype = torch.int32, device = device)
    pool_s = torch.zeros((pages * epp, G), dtype = torch.half, device = device)
    pool_r = torch.zeros((pages * epp, D_r), dtype = torch.half, device = device)
    bt = torch.stack([torch.randperm(pages, device = device, dtype = torch.int32) for _ in range(B)])
    pos = torch.tensor(pos_list, dtype = torch.int32, device = device)
    if B == 1:
        ext.dsv4_pool_quant_scatter(stage[0], pool_q, pool_s, pool_r, bt, pos_list[0], None, m, seq, epp)
    else:
        ext.dsv4_pool_quant_scatter(stage, pool_q, pool_s, pool_r, bt, 0, pos, m, seq, epp)

    touched = set()
    for j, p0 in enumerate(pos_list):
        ec0 = p0 // m
        nw = (p0 + seq) // m - ec0
        for w in range(nw):
            e = ec0 + w
            row = int(bt[j, e // epp]) * epp + e % epp
            touched.add(row)
            ref_q = torch.empty((1, G * bits), dtype = torch.int32, device = device)
            ref_s = torch.empty((1, G), dtype = torch.half, device = device)
            ext.quant_cache_cont(stage[j, w, :D_c].reshape(1, D_c).contiguous(), ref_q, ref_s, 0.0)
            assert torch.equal(pool_q[row], ref_q[0]) and torch.equal(pool_s[row], ref_s[0]), \
                f"job {j} entry {e}: packed row differs from quant_cache_cont"
            assert torch.equal(pool_r[row], stage[j, w, D_c:]), f"job {j} entry {e}: rope columns differ"
    # nothing else written
    mask = torch.ones(pages * epp, dtype = torch.bool, device = device)
    mask[list(touched)] = False
    assert not pool_q[mask].any() and not pool_s[mask].any() and not pool_r[mask].any(), \
        "scatter wrote outside the addressed entries"


@pytest.fixture(scope = "module")
def model_caches(tmp_path_factory, device):
    from exllamav3.cache import Cache, CacheLayer_quant
    # rope 32 of head_dim 64 -> D_c = 32 (one group; the padded-loader widths are covered by the kernel tests above
    # and kernels/dsa)
    config, model = build_tiny_dsv4(tmp_path_factory.mktemp("dsv4_poolquant"), seed = 13,
                                    qk_rope_head_dim = 32, index_head_dim = 32)
    caches = {
        "fp16": Cache(model, max_num_tokens = 4096, max_batch_size = 2),
        "q8": Cache(model, max_num_tokens = 4096, max_batch_size = 2, layer_type = CacheLayer_quant,
                    k_bits = 8, v_bits = 8),
        "q4": Cache(model, max_num_tokens = 4096, max_batch_size = 2, layer_type = CacheLayer_quant,
                    k_bits = 4, v_bits = 4),
    }
    model.load(str(device))
    yield model, caches
    model.unload()


@pytest.fixture(scope = "module")
def ref_floor(model_caches):
    model, _ = model_caches
    torch.manual_seed(21)
    ids = torch.randint(0, DSV4_TINY["vocab_size"], (1, 315), dtype = torch.long)
    ref = fwd_modules(model, ids, {"attn_mode": "flash_attn_nc"})
    floor_kl, floor_am = noise_floor(model, ids, ref)
    return ids, floor_kl, floor_am


def test_pool_layers_quantized(model_caches):
    """The comparison below is only meaningful if the quantized caches really hold packed pools"""
    _, caches = model_caches
    for name, bits in (("fp16", 0), ("q8", 8), ("q4", 4)):
        got = [(type(layer).__name__, layer.k_bits) for layer in caches[name].layers.values()]
        assert got and all(k == bits for _, k in got), f"{name} cache layers: {got}"


@pytest.mark.parametrize("chunks", [
    pytest.param([315], id = "single"),
    pytest.param([100, 107, 108], id = "uneven"),
    pytest.param([300] + [1] * 15, id = "prefill_decode"),
])
def test_quant_pools_vs_fp16(model_caches, ref_floor, chunks):
    model, caches = model_caches
    ids, floor_kl, floor_am = ref_floor
    outs = {}
    for name, c in caches.items():
        with torch.inference_mode():
            state = c.get_new_state()
        try:
            outs[name] = fwd_cached(model, ids, state, chunks)
        finally:
            state.free()
    # Q8 must sit at the chunk-shape noise floor against the fp16 pools; Q4 within a generous multiple (the tiny
    # random model amplifies through routing near-ties)
    assert_logits_close(outs["q8"][-32:], outs["fp16"][-32:], max(5e-4, 2.0 * floor_kl),
                        min(0.99, floor_am - 0.05), f"q8 vs fp16 pools, chunks {chunks}")
    assert_logits_close(outs["q4"][-32:], outs["fp16"][-32:], max(5e-3, 12.0 * floor_kl),
                        min(0.95, floor_am - 0.15), f"q4 vs fp16 pools, chunks {chunks}")
