"""
Recurrent checkpoint storage (cache/recurrent.py). HostPool: stashes copy into pooled host buffers keyed by
exact shape and dtype that eviction returns, so the allocator sees no per-checkpoint churn (issue #432), and
host_copy matches a plain .cpu() for strided device sources. RecurrentCache.close(): drops every checkpoint,
counts a stash shared by two keys once, returns the buffers and empties the pool so a replacement cache
allocates fresh memory. Checked against the pool's allocation/reuse counters and buffer identity.
GDNLayerState: fp32, fp16 and packed recurrent-state layout, storage, checkpoints and TP mappings.
"""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.cache import recurrent
from exllamav3.cache.cache import Cache
from exllamav3.cache.recurrent import HostPool, RecurrentCache, host_copy, host_pool
from exllamav3.modules.gated_delta_net import GDNLayerState, QuantizedStateUnsupported

NUM_K_SLICES = 4


@pytest.fixture(autouse = True)
def fresh_pool(monkeypatch):
    # host_pool is process-global: start every test from an empty free list and freed-bytes counter
    monkeypatch.setattr(recurrent, "_freed_bytes", 0)
    host_pool.free.clear()
    yield
    host_pool.free.clear()


def free_count(pool):
    return sum(len(v) for v in pool.free.values())


# HostPool

@pytest.mark.nogpu
def test_take_reuses_exact_shape_and_dtype():
    pool = HostPool()
    a = pool.take((4, 8), torch.float32)
    b = pool.take((4, 8), torch.float32)
    assert pool.allocated == 2
    pool.give(a)
    c = pool.take((4, 8), torch.float32)
    assert c is a
    assert (pool.allocated, pool.reused) == (2, 1)
    # neither a different shape nor a different dtype may be served from a free buffer
    pool.give(c)
    d = pool.take((8, 4), torch.float32)
    e = pool.take((4, 8), torch.float16)
    assert pool.allocated == 4
    assert d.shape == (8, 4) and e.dtype == torch.float16
    assert e is not c
    assert pool.take((4, 8), torch.float32) is c


@pytest.mark.nogpu
def test_give_walks_a_stashed_structure():
    pool = HostPool()
    t = [pool.take((3,), torch.float32) for _ in range(4)]
    stashed = {"position": 2048, "checkpoint_size": 12345, "tp_handle": 7,
               "layer_a": (t[0], t[1]), "layer_b": [t[2], (t[3],)]}
    pool.give(stashed)
    assert free_count(pool) == 4
    ids = {id(x) for x in t}
    for _ in range(4):
        assert id(pool.take((3,), torch.float32)) in ids
    assert pool.reused == 4


def test_give_does_not_pool_device_tensors(device):
    pool = HostPool()
    pool.give(torch.zeros(3, device = device))
    assert free_count(pool) == 0


@pytest.mark.nogpu
def test_release_drops_idle_buffers_only():
    pool = HostPool()
    held = pool.take((5,), torch.float32)
    idle = pool.take((5,), torch.float32)
    pool.give(idle)
    pool.release()
    assert free_count(pool) == 0
    assert pool.take((5,), torch.float32) is not idle
    held.fill_(1.0)                       # the buffer still in use is untouched


def test_host_copy_matches_cpu_for_strided_sources(device):
    before = host_pool.allocated
    state = torch.randn(4, 6, 16, 16, device = device)
    conv = torch.randn(4, 24, 8, device = device)
    for src in (state[2, :1], conv[1, :, :4], state[3]):
        out = host_copy(src)
        assert out.device.type == "cpu"
        assert torch.equal(out, src.cpu())
        assert out.shape == src.shape
    assert host_pool.allocated == before + 3
    # a second stash of the same shapes reuses the buffers once the first is returned
    host_pool.give([host_copy(state[0, :1])])
    host_copy(state[1, :1])
    assert host_pool.reused >= 1
    host_pool.release()


# RecurrentCache.close()

class FakeState:
    """Stand-in for a recurrent layer state: stash() copies two tensors into pooled host buffers."""

    def __init__(self, position, rows = 64):
        self.position = position
        self.rows = rows
        self.checkpoint_size = 2 * rows * 1024 * 2

    def stash(self):
        return {
            "position": self.position,
            "checkpoint_size": self.checkpoint_size,
            "a": recurrent.host_copy(torch.ones(self.rows, 1024, dtype = torch.float16)),
            "b": recurrent.host_copy(torch.ones(self.rows, 1024, dtype = torch.float16)),
        }


def make_cache(max_size = 10 * 1024**2):
    return RecurrentCache(SimpleNamespace(loaded_tp = False), max_size)


@pytest.mark.nogpu
def test_close_drops_checkpoints_and_releases_the_pool():
    cache = make_cache()
    for i in range(5):
        cache.put(i, FakeState(position = i * 256))
    assert len(cache) == 5 and cache.current_size == 5 * FakeState(0).checkpoint_size
    cache.pagetable = object()

    cache.close()

    assert len(cache) == 0
    assert cache.current_size == 0
    assert cache.pagetable is None
    # The buffers were handed back to the pool and the pool emptied, so the RAM is free
    assert host_pool.free == {}
    assert cache.get_stashed(0) is None


@pytest.mark.nogpu
def test_close_is_idempotent_and_the_cache_stays_usable_for_bookkeeping():
    cache = make_cache()
    cache.put(1, FakeState(position = 0))
    cache.close()
    cache.close()
    assert len(cache) == 0 and cache.current_size == 0


@pytest.mark.nogpu
def test_close_counts_a_shared_stash_once():
    cache = make_cache()
    cache.put(1, FakeState(position = 0))
    cache[2] = cache[1]  # two keys, one stash
    cache.update_total_size()
    assert cache.current_size == FakeState(0).checkpoint_size

    cache.close()
    assert len(cache) == 0 and host_pool.free == {}


@pytest.mark.nogpu
def test_closed_cache_frees_memory_for_a_replacement():
    """A new cache's stashes don't come out of the old cache's buffers once they are released."""
    allocated, reused = host_pool.allocated, host_pool.reused
    old = make_cache()
    old.put(1, FakeState(position = 0))
    old.close()
    assert host_pool.allocated == allocated + 2

    new = make_cache()
    new.put(1, FakeState(position = 0))
    # Allocated fresh, since the pool was emptied and the old buffers' memory returned
    assert host_pool.allocated == allocated + 4 and host_pool.reused == reused


# GDNLayerState


def _gdn_layer_state(state_dtype, state_bits = 0, slots = 2, max_history = 3):
    module = SimpleNamespace(
        fdim_qkv = 2 * 4 * 128 + 4 * 128,
        conv_kernel_size = 4,
        num_k_heads = 4,
        num_v_heads = 4,
        k_head_dim = 128,
        v_head_dim = 128,
    )
    return GDNLayerState(module, slots, max_history, cache_id = 0, state_dtype = state_dtype, state_bits = state_bits)


@pytest.mark.parametrize("state_dtype,state_bits",
                         [(torch.float32, 0), (torch.float16, 0), (torch.int8, 9)],
                         ids = ["fp32", "fp16", "int9"])
@pytest.mark.nogpu
def test_gdn_layer_state_pool_layout_and_accounting(state_dtype, state_bits):
    slots, max_history = 2, 3
    ls = _gdn_layer_state(state_dtype, state_bits, slots, max_history)
    m = ls.module
    rows = 2 * slots
    conv, state = ls.conv_state, ls.recurrent_state
    assert conv.shape == (slots, m.fdim_qkv, m.conv_kernel_size + max_history) and conv.dtype == torch.bfloat16
    if state_dtype == torch.int8:
        assert state.shape == (rows, 1, m.num_v_heads, m.v_head_dim, m.k_head_dim * state_bits // 8)
    else:
        assert state.shape == (rows, 1, m.num_v_heads, m.k_head_dim, m.v_head_dim)
    assert state.dtype == state_dtype
    scales = ls.recurrent_scales
    if state_dtype == torch.int8:
        assert scales.shape == (rows, 1, m.num_v_heads, m.v_head_dim, NUM_K_SLICES) and scales.dtype == torch.float16
    else:
        assert scales is None
    assert len(ls.get_state_tensors()) == 2
    conv_bytes = slots * m.fdim_qkv * (m.conv_kernel_size + max_history) * 2
    staging_rows = slots * (max_history + 1)
    staging_bytes = staging_rows * m.fdim_qkv * 2 + staging_rows * m.num_v_heads * 2 + staging_rows * m.num_v_heads * 4
    expected_state_bytes = rows * state[0].numel() * state.element_size()
    if state_dtype == torch.int8:
        expected_state_bytes += rows * scales[0].numel() * scales.element_size()
    assert ls.storage_size() == conv_bytes + expected_state_bytes + staging_bytes
    assert ls.get_checkpoint_size() == m.fdim_qkv * m.conv_kernel_size * 2 \
        + state[0].numel() * state.element_size() \
        + (scales[0].numel() * scales.element_size() if state_dtype == torch.int8 else 0)


@pytest.mark.nogpu
def test_gdn_layer_state_clear_zeroes_the_pool_and_the_scales():
    for state_dtype, state_bits in ((torch.float16, 0), (torch.int8, 8)):
        ls = _gdn_layer_state(state_dtype, state_bits)
        ls.alloc(torch.device("cpu"))
        ls.recurrent_state.fill_(1)
        if ls.recurrent_scales is not None:
            ls.recurrent_scales.fill_(1)
        ls.clear(0)
        assert not ls.recurrent_state[0:2].any().item()
        assert ls.recurrent_state[2:4].all().item()
        if ls.recurrent_scales is not None:
            assert not ls.recurrent_scales[0:2].any().item()
            assert ls.recurrent_scales[2:4].all().item()
        ls.free()
        assert ls.device is None


@pytest.mark.nogpu
def test_gdn_layer_state_packed_stash_round_trips_values_and_scales():
    bits = 9
    ls = _gdn_layer_state(torch.int8, bits)
    ls.alloc(torch.device("cpu"))
    m = ls.module
    values = torch.full(
        (1, 1, m.num_v_heads, m.v_head_dim, m.k_head_dim * bits // 8), 3, dtype = torch.int8
    )
    scales = torch.full((1, 1, m.num_v_heads, m.v_head_dim, NUM_K_SLICES), 0.5, dtype = torch.float16)
    ls.recurrent_state[0:1].copy_(values)
    ls.recurrent_scales[0:1].copy_(scales)
    before = host_pool.allocated

    stashed = ls.stash(0, 0, 0)
    stashed_values, stashed_scales, _ = stashed
    assert stashed_values.shape == values.shape and stashed_values.dtype == torch.int8
    assert stashed_scales.shape == scales.shape and stashed_scales.dtype == torch.float16
    assert torch.equal(stashed_values, values) and torch.equal(stashed_scales, scales)
    assert host_pool.allocated == before + 3

    ls.recurrent_state[0:1].zero_()
    ls.recurrent_scales[0:1].zero_()
    ls.unstash(0, stashed, 0, 0)
    assert torch.equal(ls.recurrent_state[0:1], values)
    assert torch.equal(ls.recurrent_scales[0:1], scales)

    host_pool.give(stashed)
    assert free_count(host_pool) == 3
    ls.free()


@pytest.mark.nogpu
def test_gdn_layer_state_fp16_stash_has_state_and_conv():
    ls = _gdn_layer_state(torch.float16)
    ls.alloc(torch.device("cpu"))
    stashed = ls.stash(0, 0, 0)
    assert len(stashed) == 2
    host_pool.give(stashed)
    ls.free()


@pytest.mark.nogpu
def test_gdn_layer_state_tp_export_preserves_packed_width():
    original = _gdn_layer_state(torch.int8, 9)
    exported = original.tp_export(None)
    reconstructed = GDNLayerState(original.module, **exported["args"])
    assert reconstructed.recurrent_bits == 9
    assert reconstructed.recurrent_state.dtype == torch.int8
    assert reconstructed.recurrent_scales.dtype == torch.float16
    assert reconstructed.quantized


@pytest.mark.nogpu
def test_gdn_layer_state_tp_stash_pieces_order_and_v_head_mapping():
    module = SimpleNamespace(
        fdim_qkv = 2 * 2 * 128 + 4 * 128,
        conv_kernel_size = 4,
        num_k_heads = 2,
        num_v_heads = 4,
        k_head_dim = 128,
        v_head_dim = 128,
        tp_kv_first = 1,
        tp_k_heads_total = 4,
    )
    state = GDNLayerState(module, 1, 0, cache_id = 0, state_dtype = torch.int8, state_bits = 9)
    v_head_mapping = {"dim": 2, "total": 8, "pieces": [(0, 4, 2)]}
    conv_mapping = {"dim": 0, "total": 2048, "pieces": [
        (0, 256, 128),
        (256, 512, 640),
        (512, 1024, 1280),
    ]}
    assert state.tp_stash_pieces() == [v_head_mapping, v_head_mapping, conv_mapping]


@pytest.mark.parametrize("attribute,value,error", [
    ("kda", True, "KDA"),
    ("k_head_dim", 64, "128x128"),
])
@pytest.mark.nogpu
def test_quantized_state_rejects_ineligible_layers(attribute, value, error):
    state = _gdn_layer_state(torch.float)
    setattr(state.module, attribute, value)
    with pytest.raises(QuantizedStateUnsupported, match = error):
        state.set_state_dtype(torch.int8, 9)


# Cache configuration


def _empty_model():
    return SimpleNamespace(config = None, cache_weakrefs = {}, get_cache_layers = lambda: [],
                           get_recurrent_layers = lambda: [], recurrent_state_cls = None,
                           get_layer_instances = lambda i: [])


@pytest.mark.parametrize("recurrent_dtype,recurrent_bits",
                         [(torch.int8, 0), (torch.int8, 1), (torch.int8, 13), (torch.float, 4), (torch.int16, 0)],
                         ids = ["missing_width", "width_1", "width_13", "fp_bits", "int16"])
@pytest.mark.nogpu
def test_cache_rejects_ineligible_recurrent_pairs(recurrent_dtype, recurrent_bits):
    with pytest.raises(ValueError):
        Cache(_empty_model(), max_num_tokens = 256, recurrent_dtype = recurrent_dtype,
              recurrent_bits = recurrent_bits)
