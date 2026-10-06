"""
A historyless short-conv rewind must be a position correction, not a crash.

ShortConvLayerState.rewind must gate on last_history == 0 BEFORE its
`num_tokens <= last_history` assert. Callers arrive with last_history == 0 whenever a
rewind follows a forward that recorded no per-token history: the prefill-overshoot
rewind in job.py reaches this class through ShortConvState.rewind or
_collect_rewind_jobs' non-GDN fallback with the overshoot as num_tokens. No in-tree
short-conv architecture triggers that combination today (the atomic-MM-prefill
overshoot that produces last_history == 0 lives in architectures without short-conv
layers), so this is a contract for future architectures, not a live crash: the moment
one adds a short-conv layer to such a model, an unguarded rewind either trips the
assert or -- if the bounds check were dropped along with the gate -- copies from a
negatively-indexed slice.

There is nothing to copy from when no history was recorded -- the history tail of
conv_state is only filled by forwards that record per-token windows -- so the guarded
rewind must leave the state untouched and return. This mirrors the gate the GDN job
builders (rewind_conv_job / rewind_state_job) apply for the same condition.

Where history does exist, the accepted range is the full 0 <= num_tokens <= last_history:
the upper bound is the recorded-history guard (rewinding deeper than the history would run
the source window off the front of the row), the lower bound stops a negative count from
walking the window off the back.

CPU-only: rewind is pure index arithmetic and copying over one tensor.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.modules import short_conv as sc

MAX_BATCH = 4
MAX_HIST = 8
CDIM = 4        # conv_kernel_size
HDIM = 6        # hidden_size
WIDTH = CDIM + MAX_HIST


class _Module:
    hidden_size = HDIM
    conv_kernel_size = CDIM


@pytest.fixture
def layer():
    state = sc.ShortConvLayerState(
        _Module(), max_batch_size = MAX_BATCH, max_history = MAX_HIST, cache_id = 0)
    state.alloc(torch.device("cpu"))
    return state


def _fill_columns(layer, slot):
    for col in range(WIDTH):
        layer.conv_state[slot, :, col] = float(slot * 100 + col)


# ---- the historyless gate (the contract for future architectures) ---------------------------------

def test_rewind_without_history_is_a_noop_not_a_crash(layer):
    """The historyless case: last_history == 0, arbitrary num_tokens. Without the gate this
    trips the unconditional assert; with it, it must return without moving a single byte."""
    for slot in range(MAX_BATCH):
        layer.conv_state[slot].fill_(float(slot))
    before = layer.conv_state.clone()

    for num_tokens in (1, 2, 5):
        layer.rewind(0, 0, num_tokens)

    assert torch.equal(layer.conv_state, before)


def test_historyless_rewind_does_not_pull_in_the_preceding_slot(layer):
    layer.conv_state[0].fill_(555.0)        # the neighbour that an unguarded read could reach
    layer.conv_state[1, :, :CDIM].fill_(-1.0)

    layer.rewind(1, 0, 2)

    assert torch.equal(layer.conv_state[1, :, :CDIM],
                       torch.full((HDIM, CDIM), -1.0, dtype = torch.half))
    assert 555.0 not in layer.conv_state[1].flatten()


# ---- the speculative path still works ---------------------------------------------------------------

@pytest.mark.parametrize("num_tokens", range(MAX_HIST + 1))
def test_rewind_restores_the_recorded_conv_state(layer, num_tokens):
    """Positive control: after rewinding j tokens the head must hold the window that ended
    j columns before the row's end, for every in-range j: j == 0 (the commit after a fully
    accepted forward), a plain rejection, and the clone-dependent overlap regime
    (j > max_history - conv_kernel_size, where the source window reaches under the
    destination head and at j == max_history coincides with it). The implementation clones
    the source before writing, so the expected window is captured before the rewind --
    comparing against the live source view afterwards would itself be torn in the overlap
    regime and could not tell a correct copy from a corrupted one."""
    _fill_columns(layer, 0)
    p = WIDTH - num_tokens
    expected = layer.conv_state[0, :, p - CDIM:p].clone()
    layer.rewind(0, last_history = MAX_HIST, num_tokens = num_tokens)
    assert torch.equal(layer.conv_state[0, :, :CDIM], expected)


def test_rewind_is_scoped_to_its_own_slot(layer):
    for slot in range(MAX_BATCH):
        _fill_columns(layer, slot)
    others = [layer.conv_state[s].clone() for s in range(1, MAX_BATCH)]

    layer.rewind(0, last_history = 4, num_tokens = 2)

    for s, expected in zip(range(1, MAX_BATCH), others):
        assert torch.equal(layer.conv_state[s], expected)


# ---- the bounds guard -------------------------------------------------------------------------------

def test_rewind_rejects_rewind_deeper_than_history(layer):
    """History exists but the rewind is deeper than it: a genuine accounting error, and one
    that would run the source window off the front of the row."""
    with pytest.raises(AssertionError):
        layer.rewind(0, 1, 2)


def test_rewind_rejects_negative_num_tokens(layer):
    """The guard is two-sided: a negative count would walk the source window off the back of
    the row."""
    with pytest.raises(AssertionError):
        layer.rewind(0, 4, -1)


# ---- the production entry-point wrapper -----------------------------------------------------------

class _MockModel:
    loaded_tp = False


class _MockCache:
    """Minimal Cache surface for ShortConvState: model flag and the recurrent-layer map."""
    def __init__(self, layers):
        self.model = _MockModel()
        self._layers = layers

    def get_all_recurrent_layers(self):
        return self._layers


def test_short_conv_state_rewind_wrapper_calls_layers_and_resets(layer):
    """ShortConvState.rewind is the production entry point: it must call each layer's
    rewind with its slot and last_history, decrement position by num_tokens, and reset
    last_history to 0. A regression that forgets to reset last_history, passes the wrong
    slot, or drops the layer loop entirely would not be caught by the layer.rewind tests
    alone, so the layer's conv_state is observed directly."""
    cache = _MockCache({0: layer})
    state = sc.ShortConvState(cache, slot = 2, position = 100, test_state = True)
    state.last_history = 4
    # The constructor clears slot 2, so fill after construction
    _fill_columns(layer, 2)
    src = layer.conv_state[2, :, WIDTH - 2 - CDIM:WIDTH - 2].clone()
    before = layer.conv_state.clone()
    state.rewind(2)
    assert state.position == 98
    assert state.last_history == 0
    # The layer's rewind ran for slot 2 with last_history = 4: the head now holds the
    # recorded window (a wrong slot or a dropped loop leaves it at the fill values)
    assert not torch.equal(layer.conv_state, before)
    assert torch.equal(layer.conv_state[2, :, :CDIM], src)
    for s in range(MAX_BATCH):
        if s == 2:
            continue
        assert torch.equal(layer.conv_state[s], before[s])


def test_short_conv_state_rewind_wrapper_with_historyless_state(layer):
    """last_history == 0: the layer's rewind is a no-op, but the wrapper still decrements
    position and resets last_history."""
    cache = _MockCache({0: layer})
    state = sc.ShortConvState(cache, slot = 1, position = 50, test_state = True)
    state.last_history = 0
    state.rewind(3)
    assert state.position == 47
    assert state.last_history == 0
