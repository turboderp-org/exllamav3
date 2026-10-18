"""
DeepSeek-V4 cached path (attn_mode flash_attn: SWA rings + compressed pools + DSA kernels) against the stateless
nc path on the full tiny random model, over chunkings up to and past the SWA ring size, plus rewind consistency.
Drives the module list directly (no generator); state advance is done manually like advance_recurrent_states
would. Tolerances derive from the measured chunk-shape noise floor (testlib.dsv4.noise_floor).
"""

import pytest
import torch

from exllamav3.cache.cache import Cache

from testlib.dsv4 import assert_logits_close, build_tiny_dsv4, fwd_cached, fwd_modules, noise_floor
from testlib.tiny_models import DSV4_TINY


@pytest.fixture(scope = "module")
def model_cache(tmp_path_factory, device):
    config, model = build_tiny_dsv4(tmp_path_factory.mktemp("dsv4_cached"), seed = 13)
    cache = Cache(model, max_num_tokens = 4096, max_batch_size = 2)
    model.load(str(device))
    yield model, cache
    model.unload()


@pytest.fixture(scope = "module")
def ref_short(model_cache):
    """315 random ids, their nc logits, and the KL / argmax tolerances from the noise floor"""
    model, _ = model_cache
    torch.manual_seed(21)
    ids = torch.randint(0, DSV4_TINY["vocab_size"], (1, 315), dtype = torch.long)
    ref = fwd_modules(model, ids, {"attn_mode": "flash_attn_nc"})
    floor_kl, floor_am = noise_floor(model, ids, ref)
    kl_tol = max(5e-4, 1.5 * floor_kl)
    arg_tol = min(0.99, floor_am - 0.05)
    return ids, ref, kl_tol, arg_tol


@pytest.fixture(scope = "module")
def ref_long(model_cache):
    """2048 random ids and their nc logits"""
    model, _ = model_cache
    torch.manual_seed(22)
    ids = torch.randint(0, DSV4_TINY["vocab_size"], (1, 2048), dtype = torch.long)
    return ids, fwd_modules(model, ids, {"attn_mode": "flash_attn_nc"})


def run_cached(cache, model, ids, chunks):
    with torch.inference_mode():
        state = cache.get_new_state()
    try:
        return fwd_cached(model, ids, state, chunks)
    finally:
        state.free()


@pytest.mark.parametrize("chunks", [
    pytest.param([315], id = "single"),
    pytest.param([100, 107, 108], id = "uneven"),
    pytest.param([256, 30, 29], id = "page_aligned_first"),
    pytest.param([300] + [1] * 15, id = "prefill_decode"),
])
def test_cached_vs_nc(model_cache, ref_short, chunks):
    model, cache = model_cache
    ids, ref, kl_tol, arg_tol = ref_short
    got = run_cached(cache, model, ids, chunks)
    # nc discards trailing sub-window compressor rows per chunk; only the FINAL positions of each run see
    # identical entry sets, so compare the last 32 positions
    assert_logits_close(got[-32:], ref[-32:], kl_tol, arg_tol, f"cached vs nc, chunks {chunks}")


@pytest.mark.parametrize("chunks", [
    pytest.param([1024, 1024], id = "2x1024"),
    pytest.param([768, 640, 640], id = "mixed"),
    pytest.param([2048], id = "single"),
    pytest.param([1024, 1] * 8, id = "big_decode_interleaved"),
])
def test_big_chunk_cached_vs_nc(model_cache, ref_short, ref_long, chunks):
    """Chunks larger than the SWA ring (768 rows at tiny window 8): the temp-window path and the ring rebase
    branch"""
    model, cache = model_cache
    _, _, kl_tol, arg_tol = ref_short
    ids, ref = ref_long
    got = run_cached(cache, model, ids, chunks)
    assert_logits_close(got[-32:], ref[-32:], kl_tol, arg_tol, f"big-chunk cached vs nc, chunks {chunks}")


def test_rewind_replay(model_cache, ref_short):
    """Logits for re-decoded tokens after a rewind must match the first pass exactly"""
    model, cache = model_cache
    ids = ref_short[0]
    with torch.inference_mode():
        state = cache.get_new_state()
    try:
        fwd_cached(model, ids[:, :300], state, [300])
        first = fwd_cached(model, ids[:, 300:312], state, [1] * 12)
        state.rewind(12)
        second = fwd_cached(model, ids[:, 300:312], state, [1] * 12)
    finally:
        state.free()
    err = (first - second).abs().max().item()
    assert err == 0.0, f"rewind-and-replay maxdiff {err:.2e}"
