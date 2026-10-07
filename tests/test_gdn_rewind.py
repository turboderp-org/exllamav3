"""
A GDN rewind after a speculative pass must restore conv history without ever touching
another sequence's state, and must rebuild the recurrent state by scan replay.

Path A removed the per-token recurrent-state snapshot ring. The recurrent_state pool holds
two rows per batch slot (base = 2*slot + parity, scratch = 2*slot + 1 - parity); the verify
pass advances the scratch row from the base row, and a rejected suffix is undone by
replaying the scan kernel over the staged scan inputs for the accepted prefix
(GDNLayerState.replay_scan), not by copying a plane. Only the conv ring still rewinds by
copy: the batched kernel ext.batched_conv_rewind, fed by the pure pointer-arithmetic job
builder rewind_conv_job(). These tests pin the conv builder's offsets, the
_rewind_layers dispatch (conv jobs batched, replay calls routed per layer, non-GDN layers
handed their own .rewind()), the GDNState.rewind wrapper (position, parity flip, replay
arguments), the GDNState stash/unstash parity carry, the prepare_for_recurrence -> kernel
scan-row mapping (spec: write scratch, read base), the rule that spec passes never route
to the chunk kernels, the mixed-cache parity guard, and -- under a CUDA gate -- the kernels:
the conv rewind and the bit-exact equivalence of a base->scratch replay (gdn, m2 and KDA
channelwise-g branches) with the plane the old ring would have recorded.

The conv builder mints raw integer offsets (data_ptr + slot * stride(0) * element_size +
window * stride(2) * element_size) and the kernel validates only cdim, never the offset, so
an out-of-range window start does not fault; it reads a neighbouring slot's live conv state.
The bounds guards must therefore refuse before any descriptor is built.

Assertion order is part of the contract: the two-sided bound on num_tokens is an upstream
accounting error and is checked BEFORE the last_history == 0 gate, so a negative count is
refused identically whether or not history was recorded (GDNState.rewind moves the position
by -num_tokens either way).

CPU-only: the job builder is pure index arithmetic over tensor geometry and the dispatch
logic is Python. The tests that launch the real kernels are CUDA-gated.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.modules import gated_delta_net as gdn

MAX_BATCH = 4
MAX_HIST = 8
CDIM = 4        # conv_kernel_size
FDIM = 6
NV, HK, HV = 3, 2, 6
WIDTH = CDIM + MAX_HIST

# Wide enough to reach every out-of-range combination, not just the reachable ones, and
# dipping below zero so negative rewinds are pinned as refusals too.
NUM_TOKENS_RANGE = range(-2, MAX_HIST + 3)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


class _Module:
    conv_kernel_size = CDIM
    fdim_qkv = FDIM
    num_v_heads = NV
    k_head_dim = HK
    v_head_dim = HV
    name = "test.gdn"
    layer_idx = 3


def _make_layer(device = torch.device("cpu")):
    """A real GDNLayerState (the production class, meta-allocated then alloc()'d on the
    given device). _rewind_layers routes on isinstance(l, GDNLayerState), so the dispatch
    tests must drive the real type, not a duck-typed stand-in."""
    state = gdn.GDNLayerState(_Module(), max_batch_size = MAX_BATCH, max_history = MAX_HIST,
                              cache_id = 0)
    state.alloc(device)
    return state


@pytest.fixture
def layer():
    return _make_layer()


def _slot_base(t, slot):
    """Address of t[slot, ...], derived from the tensor's own slot stride and element size.
    Every expected address in this file is derived this way (never a bare data_ptr), so a
    stride swap or a dropped element_size() in the builder fails the checks."""
    return t.data_ptr() + slot * t.stride(0) * t.element_size()


def _try_build(build, slot, last_history, num_tokens):
    """Build a rewind job, treating an assertion as a rejection.

    An out-of-range rewind is refused two ways: by returning no job at all (there was no
    history to rewind from), or by asserting (an internal accounting error, which is a bug
    somewhere upstream rather than a normal condition). Both are refusals to touch state, so
    the invariant under test is that neither ever yields a job pointing outside the slot.
    """
    try:
        return build(slot, last_history, num_tokens)
    except AssertionError:
        return None


class _JobRecorder:
    """Wrap a pybind job class so tests can inspect the exact integers the builder passes to
    it. The bindings expose only the constructor -- src/dst are not readable attributes --
    so capturing the init args is the only way to pin the computed offsets without
    recompiling the extension."""

    def __init__(self, real, sink):
        self.real = real
        self.sink = sink

    def __call__(self, *args):
        self.sink.append(args)
        return self.real(*args)


@pytest.fixture
def captured(monkeypatch):
    """Record the constructor args of every rewind job built during the test."""
    calls = []
    monkeypatch.setattr(gdn.ext, "ConvRewindJob", _JobRecorder(gdn.ext.ConvRewindJob, calls))
    return calls


class _ReplayRecorder:
    """Replace GDNLayerState.replay_scan so tests can observe the replay routing (the real
    method launches scan kernels and needs staged CUDA inputs)."""

    def __init__(self, sink):
        self.sink = sink

    def __get__(self, obj, objtype = None):
        layer = obj
        def replay(row, prefix, shape, base_row, scratch_row):
            self.sink.append((id(layer), row, prefix, shape, base_row, scratch_row))
        return replay


def _patch_replay(monkeypatch, sink):
    monkeypatch.setattr(gdn.GDNLayerState, "replay_scan", _ReplayRecorder(sink))


# ---- the conv window invariant --------------------------------------------------------------------

def test_conv_job_offsets_pin_the_window_inside_the_slot(layer, captured):
    """Every built conv job must read conv_state[slot, :, p-cdim:p] with p = WIDTH -
    num_tokens, wholly inside its own slot's row: p-cdim must not run off the front and the
    window must not run off the back. The window start is derived from the job's own
    pointers, so a wrong stride/element_size, a dropped slot stride or an off-by-one cannot
    hide inside a re-derived range."""
    cs = layer.conv_state
    col_bytes = cs.stride(2) * cs.element_size()
    slot_bytes = cs.stride(0) * cs.element_size()
    for slot in range(MAX_BATCH):
        for last_history in range(MAX_HIST + 1):
            for num_tokens in NUM_TOKENS_RANGE:
                captured.clear()
                if _try_build(layer.rewind_conv_job, slot, last_history, num_tokens) is None:
                    continue
                (src, dst, dim, cdim, stride), = captured
                assert 0 <= num_tokens <= last_history, \
                    f"conv job built for {num_tokens} tokens with history {last_history}"
                base = _slot_base(cs, slot)
                assert dst == base, f"conv job dst is not slot {slot}'s row head"
                assert dim == FDIM and cdim == CDIM and stride == cs.stride(1)
                offset = src - dst
                assert offset % col_bytes == 0, "conv job src is not column-aligned"
                start = offset // col_bytes
                assert 0 <= start and start + CDIM <= WIDTH, \
                    f"conv job reads [{start}:{start + CDIM}], off the row"
                # The kernel touches cdim elements per channel row; the last byte read or
                # written is at offset + (dim-1)*stride + cdim, which must stay in the slot.
                assert offset + (dim - 1) * stride * cs.element_size() \
                    + cdim * cs.element_size() <= slot_bytes, \
                    f"conv job reads outside slot {slot}"
                assert start == WIDTH - num_tokens - CDIM, \
                    f"conv job reads window starting at {start}, expected {WIDTH - num_tokens - CDIM}"


def test_conv_jobs_are_scoped_to_their_own_slot(layer, captured):
    """A rewind of slot s may only ever address slot s's bytes. Building one job per slot
    with the same history and checking every pointer against its slot's own derived base and
    extent pins the slot term of the arithmetic: drop it and every slot's job targets slot 0
    (a cross-sequence write); get the element_size or stride wrong and the slot windows stop
    being disjoint."""
    cs = layer.conv_state
    c_slot_bytes = cs.stride(0) * cs.element_size()
    last_history, num_tokens = 4, 2
    seen = []
    for slot in range(MAX_BATCH):
        captured.clear()
        cj = layer.rewind_conv_job(slot, last_history, num_tokens)
        assert cj is not None
        (c_src, c_dst, c_dim, c_cdim, c_stride), = captured
        assert c_dst == _slot_base(cs, slot)
        assert 0 <= c_src - c_dst \
            and (c_src - c_dst) + (c_dim - 1) * c_stride * cs.element_size() \
                + c_cdim * cs.element_size() <= c_slot_bytes
        seen.append(c_dst)
    # One distinct destination window per slot: no two slots share a rewind target
    assert len(set(seen)) == MAX_BATCH


def test_conv_job_skips_rewind_without_history(layer):
    """The historyless case (last_history == 0, e.g. a prefill-overshoot position
    correction): nothing was recorded, the tokens are re-fed and recompute the contents, so
    no job may be built on any slot."""
    for slot in range(MAX_BATCH):
        for num_tokens in range(1, NUM_TOKENS_RANGE.stop):
            assert layer.rewind_conv_job(slot, 0, num_tokens) is None


def test_conv_job_rejects_rewind_deeper_than_history(layer):
    """History exists but the rewind is deeper than it: a genuine accounting error, and one
    the unguarded arithmetic would turn into an out-of-range pointer."""
    for slot in range(MAX_BATCH):
        for last_history, num_tokens in [(1, 2), (2, 3), (4, 5), (MAX_HIST, MAX_HIST + 1)]:
            with pytest.raises(AssertionError, match = "exceeds recorded history"):
                layer.rewind_conv_job(slot, last_history, num_tokens)


def test_conv_job_rejects_negative_num_tokens(layer):
    """The guard is two-sided: a negative count would walk the conv window off the back of
    the row. The assert ordering is part of the contract: the negativity check precedes the
    last_history == 0 gate, so a negative count raises identically with and without recorded
    history."""
    for slot in range(MAX_BATCH):
        for last_history in (0, 4):
            with pytest.raises(AssertionError, match = "negative"):
                layer.rewind_conv_job(slot, last_history, -1)


def test_conv_job_zero_tokens_builds_the_commit_job(layer):
    """num_tokens == 0 with history present is NOT a no-op on the conv side: the history-mode
    conv kernel fills only the right-aligned tail and never updates the head, so the
    tail->head copy is what commits conv state after a fully accepted verify round
    (generator.py's rewind(ids_width - accepted_length) reaches it when
    accepted_length == ids_width)."""
    assert layer.rewind_conv_job(0, 4, 0) is not None


# ---- the state pool geometry (Path A) --------------------------------------------------------------

def test_recurrent_state_pool_is_two_rows_per_slot(layer):
    """The scan state pool replaced the max_history+1 snapshot ring with a base/scratch row
    pair per slot; conv_state still reserves the max_history rewind window."""
    assert layer.recurrent_state.shape[0] == 2 * MAX_BATCH
    assert layer.recurrent_state.shape[1] == 1
    assert layer.conv_state.shape[-1] == WIDTH


def test_clear_zeroes_both_rows_of_the_slot(layer):
    layer.recurrent_state.fill_(7.0)
    layer.conv_state.fill_(7.0)
    layer.clear(2)
    assert torch.equal(layer.recurrent_state[4:6], torch.zeros(2, 1, NV, HK, HV))
    assert not torch.equal(layer.recurrent_state[0:4], torch.zeros(4, 1, NV, HK, HV))
    assert torch.equal(layer.conv_state[2], torch.zeros(FDIM, WIDTH, dtype = torch.bfloat16))


def test_stash_unstash_roundtrip_follows_parity(layer):
    """A checkpoint captures the base row (2*slot + parity) and restores it there; the
    scratch row and the other slots stay untouched."""
    layer.recurrent_state.zero_()
    base_row, scratch_row = 2 * 1 + 1, 2 * 1 + 0
    layer.recurrent_state[base_row].fill_(3.0)
    layer.recurrent_state[scratch_row].fill_(9.0)
    layer.conv_state.zero_()
    layer.conv_state[1, :, :CDIM].fill_(1.0)
    stashed = layer.stash(1, parity = 1)
    layer.recurrent_state[base_row].fill_(-2.0)
    layer.conv_state[1, :, :CDIM].zero_()
    layer.unstash(1, stashed, parity = 1)
    assert torch.equal(layer.recurrent_state[base_row], torch.full((1, NV, HK, HV), 3.0))
    assert torch.equal(layer.recurrent_state[scratch_row], torch.full((1, NV, HK, HV), 9.0))
    assert torch.equal(layer.conv_state[1, :, :CDIM], torch.full((FDIM, CDIM), 1.0, dtype = torch.bfloat16))


# ---- the batched dispatch (CPU: routing + filtering) ------------------------------------------------

class _LauncherRecorder:
    """Record ext.batched_conv_rewind launches so _rewind_layers' empty-list filtering is
    observable."""

    def __init__(self):
        self.launches = []

    def __call__(self, js, device_index):
        self.launches.append((len(js), device_index))


class _OtherLayerState:
    """A non-GDN recurrent layer state: _rewind_layers must hand it its own .rewind()."""

    def __init__(self):
        self.calls = []

    def rewind(self, slot, last_history, num_tokens):
        self.calls.append((slot, last_history, num_tokens))


def _patch_launcher(monkeypatch):
    rec = _LauncherRecorder()
    monkeypatch.setattr(gdn.ext, "batched_conv_rewind", rec)
    return rec


def test_rewind_layers_routes_replay_and_non_gdn_state(layer, monkeypatch):
    """With recorded history and a rejection, every GDN layer gets one conv job and one scan
    replay addressed at the accepted prefix; a non-GDN layer is handed its own rewind."""
    replays = []
    _patch_replay(monkeypatch, replays)
    _patch_launcher(monkeypatch)
    other = _OtherLayerState()
    gdn._rewind_layers([layer, other], slot = 2, last_history = 4, num_tokens = 2,
                       spec_row = 1, spec_shape = (3, 5), base_row = 4, scratch_row = 5)
    assert other.calls == [(2, 4, 2)]
    assert replays == [(id(layer), 1, 3, (3, 5), 4, 5)]


def test_rewind_layers_full_accept_skips_replay(layer, monkeypatch):
    """num_tokens == 0 (every draft accepted): the scratch row already holds the final
    state, so only the conv commit copy runs."""
    replays = []
    _patch_replay(monkeypatch, replays)
    rec = _patch_launcher(monkeypatch)
    gdn._rewind_layers([layer], slot = 0, last_history = 4, num_tokens = 0,
                       spec_row = 0, spec_shape = (1, 5), base_row = 0, scratch_row = 1)
    assert replays == []
    assert rec.launches == [(1, None)]


def test_rewind_layers_historyless_is_position_only(layer, monkeypatch):
    """last_history == 0 (prefill overshoot): no conv job, no replay, nothing launched; the
    non-GDN fallback still sees the correction."""
    replays = []
    _patch_replay(monkeypatch, replays)
    rec = _patch_launcher(monkeypatch)
    other = _OtherLayerState()
    gdn._rewind_layers([layer, other], slot = 1, last_history = 0, num_tokens = 3)
    assert other.calls == [(1, 0, 3)]
    assert replays == []
    assert rec.launches == []


def test_rewind_layers_refuses_replay_without_staging_key(layer, monkeypatch):
    """Recorded history with no staging key is an upstream accounting error: fail loudly
    rather than silently skipping the replay (which would commit a garbage scratch row)."""
    _patch_replay(monkeypatch, [])
    _patch_launcher(monkeypatch)
    with pytest.raises(AssertionError, match = "no staged scan inputs"):
        gdn._rewind_layers([layer], slot = 0, last_history = 4, num_tokens = 2)


# ---- the production entry-point wrapper -------------------------------------------------------------

class _MockModel:
    loaded_tp = False


class _MockCache:
    """Minimal Cache surface for GDNState: model flag and the recurrent-layer map."""
    def __init__(self, layers):
        self.model = _MockModel()
        self._layers = layers

    def get_all_recurrent_layers(self):
        return self._layers


def test_gdn_state_rewind_wrapper_replays_and_flips_parity(layer, monkeypatch):
    """GDNState.rewind is the production entry point: it must route the conv job and the
    scan replay for its slot, decrement position, reset last_history, and flip the base
    parity so the replayed scratch row becomes the new base. The replay must be addressed
    at (2*slot + parity, 2*slot + 1 - parity) before the flip."""
    jobs, replays = [], []
    monkeypatch.setattr(gdn.ext, "ConvRewindJob", _JobRecorder(gdn.ext.ConvRewindJob, jobs))
    _patch_replay(monkeypatch, replays)
    _patch_launcher(monkeypatch)
    cache = _MockCache({0: layer})
    state = gdn.GDNState(cache, slot = 2, position = 100, test_state = True)
    state.last_history = 4
    state.spec_row = 0
    state.spec_shape = (1, 5)
    state.rewind(2)
    assert state.position == 98
    assert state.last_history == 0
    assert state.parity == 1
    assert replays == [(id(layer), 0, 3, (1, 5), 4, 5)]
    assert len(jobs) == 1
    conv_base = _slot_base(layer.conv_state, 2)
    cs = layer.conv_state
    es = cs.element_size()
    p = cs.shape[-1] - 2
    assert jobs[0] == (conv_base + (p - CDIM) * cs.stride(2) * es, conv_base,
                       cs.shape[1], CDIM, cs.stride(1))


def test_gdn_state_rewind_full_accept_flips_without_replay(layer, monkeypatch):
    """Full accept (num_tokens == 0): conv commit launches, no replay, parity still flips
    (the scratch row holds the accepted final state)."""
    replays = []
    _patch_replay(monkeypatch, replays)
    rec = _patch_launcher(monkeypatch)
    cache = _MockCache({0: layer})
    state = gdn.GDNState(cache, slot = 1, position = 60, test_state = True)
    state.last_history = 4
    state.spec_row = 0
    state.spec_shape = (1, 5)
    state.rewind(0)
    assert state.parity == 1
    assert replays == []
    assert rec.launches == [(1, None)]


def test_gdn_state_rewind_wrapper_with_historyless_state(layer, monkeypatch):
    """last_history == 0: the builder refuses, dispatch launches nothing, but the wrapper
    still decrements position, resets last_history and leaves parity alone."""
    rec = _patch_launcher(monkeypatch)
    cache = _MockCache({0: layer})
    state = gdn.GDNState(cache, slot = 1, position = 50, test_state = True)
    state.last_history = 0
    state.rewind(3)
    assert state.position == 47
    assert state.last_history == 0
    assert state.parity == 0
    assert rec.launches == []


def test_gdn_state_rewind_refuses_history_without_staging(layer, monkeypatch):
    """last_history > 0 but no spec_shape: the verify pass recorded history yet staged
    nothing -- committing the scratch row would corrupt the sequence, so raise."""
    _patch_launcher(monkeypatch)
    cache = _MockCache({0: layer})
    state = gdn.GDNState(cache, slot = 0, position = 50, test_state = True)
    state.last_history = 4
    with pytest.raises(RuntimeError, match = "no replayable scan inputs"):
        state.rewind(2)


# ---- the kernels themselves (CUDA-gated) -------------------------------------------------------------

@needs_cuda
def test_batched_conv_rewind_restores_the_window_for_slot():
    """Same for the conv branch, including the overlap regime (num_tokens > WIDTH - 2*cdim,
    where the source window reaches under the destination head and even coincides with it
    at num_tokens == max_history) -- the kernel reads its whole window into registers before
    writing any of it back, which a plain in-place forward copy would not survive. Expected
    windows are captured from the live row before each launch, since earlier rounds on the
    same slot have already rewritten the head. Values stay under 256 so every fill is exact
    in bfloat16."""
    from exllamav3.ext import exllamav3_ext as ext
    layer = _make_layer(torch.device("cuda:0"))
    for slot in range(MAX_BATCH):
        for col in range(WIDTH):
            layer.conv_state[slot, :, col] = float(slot * 16 + col)
    before = layer.conv_state.clone()
    # 0: commit; 2: disjoint tail window; 6: source window overlapping the head;
    # MAX_HIST: source window exactly the head (self-copy)
    for slot in range(1, MAX_BATCH):
        for num_tokens in (0, 2, 6, MAX_HIST):
            p = WIDTH - num_tokens
            expected = layer.conv_state[slot, :, p - CDIM:p].clone()
            job = layer.rewind_conv_job(slot, MAX_HIST, num_tokens)
            assert job is not None
            ext.batched_conv_rewind([job], layer.device.index)
            torch.cuda.synchronize()
            assert torch.equal(layer.conv_state[slot, :, :CDIM], expected), \
                f"slot {slot} head does not hold its own window at num_tokens={num_tokens}"
    # Slot 0 was never rewound
    assert torch.equal(layer.conv_state[0], before[0])


# ---- scan replay equivalence (CUDA-gated) -------------------------------------------------------------

# Real kernel geometry: head_dim 64, 2 v-heads, 1 k-head group
RV, RK, RD = 2, 1, 64
SEQ = 8


class _ScanModule:
    conv_kernel_size = CDIM
    fdim_qkv = (2 * RK + RV) * RD
    num_k_heads = RK
    num_v_heads = RV
    k_head_dim = RD
    v_head_dim = RD
    k_dim = RK * RD
    v_dim = RV * RD
    kda = False
    name = "test.gdn.scan"
    layer_idx = 0


@needs_cuda
def test_scan_replay_matches_the_ring_plane_bit_exactly():
    """The core Path A invariant: rerunning the scan kernel over the first `prefix` tokens
    of a verify pass, from the base row into a scratch row, produces bit-identical state to
    the plane the old max_history+1 ring recorded at that prefix (same kernel, same inputs,
    same initial state). Driven through the production replay_scan() on a real
    GDNLayerState, with the ring plane computed by the still-present history=True path."""
    from exllamav3.ext import exllamav3_ext as ext
    dev = torch.device("cuda:0")
    torch.manual_seed(7)
    fdim = (2 * RK + RV) * RD
    qkv = (torch.randn(1, SEQ, fdim, device = dev) * 0.3).to(torch.bfloat16)
    beta = torch.rand(1, SEQ, RV, device = dev).to(torch.bfloat16)
    g = (-torch.rand(1, SEQ, RV, device = dev) * 0.5).float()

    # Reference: the history=True ring path. Token 0 reads plane 0, tokens save snapshots
    # into planes 1..SEQ-1, and the last token writes the final state back into plane 0 --
    # so the plane after `prefix` tokens is ring[0, prefix] for prefix < SEQ and ring[0, 0]
    # for the full pass.
    ring = torch.zeros(1, SEQ, RV, RD, RD, dtype = torch.float, device = dev)
    out = torch.empty(1, SEQ, RV, RD, dtype = torch.bfloat16, device = dev)
    slots = torch.tensor([0], dtype = torch.int32, device = dev)
    ext.cuda_recurrent_gated_delta_rule(
        qkv.contiguous(), g.contiguous(), beta.contiguous(), ring, out,
        RK, RV, RD, RD, slots, True, None)
    torch.cuda.synchronize()

    # Path A: base row zero, replay each prefix through the production wrapper
    layer = gdn.GDNLayerState(_ScanModule(), max_batch_size = 1, max_history = MAX_HIST,
                              cache_id = 0)
    layer.alloc(dev)
    layer.recurrent_state.zero_()
    layer.spec_inputs[(1, SEQ)] = ("gdn", qkv, beta, g)
    for prefix in range(1, SEQ + 1):
        base_row, scratch_row = 0, 1
        layer.recurrent_state[scratch_row].zero_()
        layer.replay_scan(0, prefix, (1, SEQ), base_row, scratch_row)
        torch.cuda.synchronize()
        expected = ring[0, 0] if prefix == SEQ else ring[0, prefix]
        assert torch.equal(layer.recurrent_state[scratch_row, 0], expected), \
            f"replay of {prefix} tokens differs from the ring plane"


@needs_cuda
def test_scan_replay_isolates_slots_and_rows():
    """A replay of one row must not touch the base row, another slot's pair, or a row of
    the batch that staged inputs occupy differently: staging is sliced to the replayed
    batch row only."""
    dev = torch.device("cuda:0")
    torch.manual_seed(11)
    fdim = (2 * RK + RV) * RD
    qkv = (torch.randn(2, SEQ, fdim, device = dev) * 0.3).to(torch.bfloat16)
    beta = torch.rand(2, SEQ, RV, device = dev).to(torch.bfloat16)
    g = (-torch.rand(2, SEQ, RV, device = dev) * 0.5).float()

    layer = gdn.GDNLayerState(_ScanModule(), max_batch_size = 2, max_history = MAX_HIST,
                              cache_id = 0)
    layer.alloc(dev)
    layer.recurrent_state.fill_(-1.0)
    layer.recurrent_state[0].zero_()      # base row for pool slot 0 (parity 0)
    layer.spec_inputs[(2, SEQ)] = ("gdn", qkv, beta, g)
    # Replay batch row 1 from the slot-1 base (flat row 2) into flat row 3
    layer.recurrent_state[2].zero_()
    layer.replay_scan(1, 4, (2, SEQ), 2, 3)
    torch.cuda.synchronize()
    assert not torch.any(layer.recurrent_state[3] == -1.0), "scratch row untouched"
    assert torch.equal(layer.recurrent_state[2], torch.zeros(1, RV, RD, RD, device = dev)), \
        "base row was written"
    assert torch.equal(layer.recurrent_state[0], torch.zeros(1, RV, RD, RD, device = dev)), \
        "unrelated base row was written"
    assert torch.equal(layer.recurrent_state[1], torch.full((1, RV, RD, RD), -1.0, device = dev)), \
        "unrelated scratch row was written"
    # Row 1's replay used only row 1's staged inputs: compare against a direct kernel call
    from exllamav3.ext import exllamav3_ext as ext
    ref = torch.zeros(1, 1, RV, RD, RD, dtype = torch.float, device = dev)
    out = torch.empty(1, 4, RV, RD, dtype = torch.bfloat16, device = dev)
    slots = torch.tensor([0], dtype = torch.int32, device = dev)
    ext.cuda_recurrent_gated_delta_rule(
        qkv[1:2, :4].contiguous(), g[1:2, :4].contiguous(), beta[1:2, :4].contiguous(),
        ref, out, RK, RV, RD, RD, slots, False, None)
    torch.cuda.synchronize()
    assert torch.equal(layer.recurrent_state[3, 0], ref[0, 0])


# ---- producer -> kernel slot mapping (CPU) -------------------------------------------------------------

from exllamav3.cache.recurrent_util import prepare_for_recurrence


class _RState:
    """A GDNState stand-in for prepare_for_recurrence: only slot/parity/position are read."""

    def __init__(self, slot, parity, position):
        self.slot = slot
        self.parity = parity
        self.position = position


def _prep_params(spec):
    params = {
        "batch_shape": (2, 5),
        "past_len": 7,
        "recurrent_states": [_RState(0, 0, 7), _RState(1, 1, 7)],
    }
    if spec:
        params["recurrent_history"] = True
    prepare_for_recurrence(None, params, None)
    return params


def test_prepare_for_recurrence_maps_scan_rows_by_pass_type():
    """The producer of the scan kernels' state rows: recurrent_slots_scan is the WRITE row
    and recurrent_slots_scan_in the READ row (the contract replay_scan and GDNState.rewind
    prove: a spec pass advances scratch 2s+1-p from base 2s+p). A non-spec pass runs in
    place on the live row 2s+p and stages no scan-in. The pass type is redundant in the
    slot-tensor cache key (the key already carries the slot values, and 2s+1-p never equals
    2s+p), so this pins the row values; the spec element of the key is defensive."""
    p = _prep_params(spec = True)
    assert p["recurrent_slots"].tolist() == [0, 1]
    assert p["recurrent_slots_scan"].tolist() == [1, 2], "spec scan must WRITE the scratch rows"
    assert p["recurrent_slots_scan_in"].tolist() == [0, 3], "spec scan must READ the base rows"
    p = _prep_params(spec = False)
    assert p["recurrent_slots_scan"].tolist() == [0, 3], "non-spec scan must run in place on the live rows"
    assert "recurrent_slots_scan_in" not in p


# ---- verify-pass kernel routing (CPU, ext + fla mocked) --------------------------------------------------

from exllamav3.modules.gated_delta_net_fn import gated_delta_rule as gdr


class _ExtRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)


class _FakeFla:
    """Stand-in for the lazily imported exllamav3.vendor.fla module: records which chunk
    kernel the routing picked and returns shape-correct empty results."""

    def __init__(self, sink):
        self.sink = sink

    def _chunk(self, name, q, k, v, initial_state, output_final_state):
        self.sink.append(name)
        out = torch.zeros(q.shape[0], q.shape[1], v.shape[2], v.shape[3], dtype = torch.bfloat16)
        state = initial_state.clone() if (output_final_state and initial_state is not None) else None
        return out, state

    def chunk_gated_delta_rule(self, q, k, v, g = None, beta = None, initial_state = None,
                               output_final_state = False, use_qk_l2norm_in_kernel = False):
        return self._chunk("chunk_gated_delta_rule", q, k, v, initial_state, output_final_state)

    def chunk_kda(self, q, k, v, g = None, beta = None, initial_state = None,
                  output_final_state = False, use_qk_l2norm_in_kernel = False):
        return self._chunk("chunk_kda", q, k, v, initial_state, output_final_state)


def _routing_call(spec, channelwise, monkeypatch):
    """Call gated_delta_rule_fn at seqlen >= num_v_heads (the chunk threshold) with the ext
    kernel and the fla module replaced; returns (ext_calls, chunk_names, slots, slots_in)."""
    ext_rec = _ExtRecorder()
    sink = []
    monkeypatch.setattr(gdr.ext, "cuda_recurrent_gated_delta_rule", ext_rec)
    monkeypatch.setitem(sys.modules, "exllamav3.vendor.fla", _FakeFla(sink))
    nkh, nvh, hd = 1, 2, 8
    mk = torch.zeros(1, 4, (2 * nkh + nvh) * hd, dtype = torch.bfloat16)
    beta = torch.zeros(1, 4, nvh, dtype = torch.bfloat16)
    g = (torch.zeros(1, 4, nvh, hd) if channelwise else torch.zeros(1, 4, nvh)).float()
    pool = torch.zeros(4, 1, nvh, hd, hd)
    slots = torch.tensor([3], dtype = torch.int32)      # scratch row, per the mapping above
    slots_in = torch.tensor([2], dtype = torch.int32)   # base row
    gdr.gated_delta_rule_fn(
        mixed_qkv = mk, beta = beta, g = g, recurrent_state = pool,
        recurrent_slots = slots, spec = spec, save_state = False,
        num_k_heads = nkh, num_v_heads = nvh, k_dim = nkh * hd, v_dim = nvh * hd,
        k_head_dim = hd, v_head_dim = hd, slots_in = slots_in, params = {},
        channelwise_g = channelwise)
    return ext_rec.calls, sink, slots, slots_in


@pytest.mark.parametrize("channelwise", [False, True])
def test_spec_passes_never_route_to_the_chunk_kernels(channelwise, monkeypatch):
    """The verify pass must run the CUDA recurrent kernel, even at seqlen >= num_v_heads
    where a plain pass chunks: it is the only kernel honoring slots_in and the one
    replay_scan reruns. The chunk kernels read/write the base row in place and would
    silently corrupt the parity pool (the exact regression this pins)."""
    calls, sink, slots, slots_in = _routing_call(spec = True, channelwise = channelwise,
                                                 monkeypatch = monkeypatch)
    assert sink == [], "spec pass routed to a chunk kernel"
    assert len(calls) == 1
    args = calls[0]
    assert args[9] is slots, "scan output row must be the scratch slots"
    assert args[10] is False, "the verify pass must not record a ring"
    assert args[11] is slots_in, "scan input row must be the base slots"
    calls, sink = _routing_call(spec = False, channelwise = channelwise, monkeypatch = monkeypatch)[:2]
    assert calls == []
    assert sink == ["chunk_kda" if channelwise else "chunk_gated_delta_rule"]


def test_gdn_state_stash_unstash_carries_parity(layer):
    """The state-level checkpoint path: stash() records the parity it snapshotted (the base
    row 2*slot+parity, pinned at layer level elsewhere) and unstash() carries it back onto
    the state; a legacy checkpoint without the key defaults to parity 0."""
    cache = _MockCache({0: layer})
    state = gdn.GDNState(cache, slot = 1, position = 50, test_state = True)
    state.parity = 1
    layer.recurrent_state.zero_()
    base_row, scratch_row = 2 * 1 + 1, 2 * 1 + 0
    layer.recurrent_state[base_row].fill_(5.0)
    layer.recurrent_state[scratch_row].fill_(-4.0)
    stashed = state.stash()
    assert stashed["parity"] == 1
    layer.recurrent_state[base_row].fill_(0.5)
    state.parity = 0
    state.unstash(stashed)
    assert state.parity == 1
    assert torch.equal(layer.recurrent_state[base_row], torch.full((1, NV, HK, HV), 5.0))
    assert torch.equal(layer.recurrent_state[scratch_row], torch.full((1, NV, HK, HV), -4.0))
    legacy = state.stash()   # parity is 1 here (payload 5.0); strip the key to simulate an old checkpoint
    del legacy["parity"]
    state.parity = 0
    layer.recurrent_state.zero_()
    layer.recurrent_state[2 * 1 + 0].fill_(6.0)
    state.unstash(legacy)
    assert state.parity == 0
    assert torch.equal(layer.recurrent_state[2 * 1 + 0], torch.full((1, NV, HK, HV), 5.0)), \
        "a parity-less checkpoint must restore onto row 2*slot + 0"


# ---- replay branches for mamba2 and KDA (CUDA-gated) -----------------------------------------------------

@needs_cuda
def test_scan_replay_m2_branch_matches_the_ring_plane_bit_exactly():
    """The kind == "m2" branch of replay_scan (ext.cuda_recurrent_mamba2 with the D skip
    weights, staged as ("m2", conv_out, dt, g)) was never exercised: pin the same
    bit-exactness against the history=True ring plane as the gdn branch."""
    from exllamav3.ext import exllamav3_ext as ext
    dev = torch.device("cuda:0")
    torch.manual_seed(13)
    fdim = (2 * RK + RV) * RD
    xbc = (torch.randn(1, SEQ, fdim, device = dev) * 0.3).to(torch.bfloat16)
    dt = torch.rand(1, SEQ, RV, device = dev).to(torch.bfloat16)
    g = (-torch.rand(1, SEQ, RV, device = dev) * 0.5).float()
    D = torch.rand(RV, device = dev)

    ring = torch.zeros(1, SEQ, RV, RD, RD, dtype = torch.float, device = dev)
    out = torch.empty(1, SEQ, RV, RD, dtype = torch.bfloat16, device = dev)
    slots = torch.tensor([0], dtype = torch.int32, device = dev)
    ext.cuda_recurrent_mamba2(
        xbc.contiguous(), g.contiguous(), dt.contiguous(), D, ring, out,
        RK, RV, RD, RD, slots, True, None)
    torch.cuda.synchronize()

    module = _ScanModule()
    module.d_skip_f = D
    layer = gdn.GDNLayerState(module, max_batch_size = 1, max_history = MAX_HIST, cache_id = 0)
    layer.alloc(dev)
    layer.recurrent_state.zero_()
    layer.spec_inputs[(1, SEQ)] = ("m2", xbc, dt, g)
    for prefix in range(1, SEQ + 1):
        layer.recurrent_state[1].zero_()
        layer.replay_scan(0, prefix, (1, SEQ), 0, 1)
        torch.cuda.synchronize()
        expected = ring[0, 0] if prefix == SEQ else ring[0, prefix]
        assert torch.equal(layer.recurrent_state[1, 0], expected), \
            f"m2 replay of {prefix} tokens differs from the ring plane"


class _KdaScanModule(_ScanModule):
    """Channelwise (KDA) decay requires 128x128 head dims."""
    k_head_dim = 128
    v_head_dim = 128
    k_dim = RK * 128
    v_dim = RV * 128
    fdim_qkv = (2 * RK + RV) * 128
    kda = True


@needs_cuda
def test_scan_replay_kda_channelwise_g_matches_the_ring_plane():
    """The commit claims KDA (per-k-channel log-decay, g shaped [b, s, num_v_heads, dk])
    replay support; the existing replay test only uses scalar per-head g. Pin prefix
    replays of channelwise g against the history=True ring plane (the kernel detects
    channelwise by g.dim() == 4)."""
    from exllamav3.ext import exllamav3_ext as ext
    dev = torch.device("cuda:0")
    torch.manual_seed(17)
    hd = 128
    fdim = (2 * RK + RV) * hd
    qkv = (torch.randn(1, SEQ, fdim, device = dev) * 0.3).to(torch.bfloat16)
    beta = torch.rand(1, SEQ, RV, device = dev).to(torch.bfloat16)
    g = (-torch.rand(1, SEQ, RV, hd, device = dev) * 0.5).float()

    ring = torch.zeros(1, SEQ, RV, hd, hd, dtype = torch.float, device = dev)
    out = torch.empty(1, SEQ, RV, hd, dtype = torch.bfloat16, device = dev)
    slots = torch.tensor([0], dtype = torch.int32, device = dev)
    ext.cuda_recurrent_gated_delta_rule(
        qkv.contiguous(), g.contiguous(), beta.contiguous(), ring, out,
        RK, RV, hd, hd, slots, True, None)
    torch.cuda.synchronize()

    layer = gdn.GDNLayerState(_KdaScanModule(), max_batch_size = 1, max_history = MAX_HIST,
                              cache_id = 0)
    layer.alloc(dev)
    layer.recurrent_state.zero_()
    layer.spec_inputs[(1, SEQ)] = ("gdn", qkv, beta, g)
    for prefix in range(1, SEQ + 1):
        layer.recurrent_state[1].zero_()
        layer.replay_scan(0, prefix, (1, SEQ), 0, 1)
        torch.cuda.synchronize()
        expected = ring[0, 0] if prefix == SEQ else ring[0, prefix]
        assert torch.equal(layer.recurrent_state[1, 0], expected), \
            f"KDA replay of {prefix} tokens differs from the ring plane"


# ---- the mixed-cache parity guard (CPU) ---------------------------------------------------------------

from exllamav3.cache.recurrent import resolve_recurrent_parity


def test_recurrent_parity_guard_refuses_parity_indexed_layers_without_parity(layer):
    """A state class that does not track parity (SWA, ShortConv, DSA) hands the recurrent
    dispatch None. That is legal only while the cache holds no parity-indexed layer: a
    GDN/Mamba2 layer checkpointed by such a caller would read and write row 2*slot + 0
    instead of its live base row, silently corrupting the checkpoint, so the dispatch
    refuses the combination instead of guessing a parity."""
    assert resolve_recurrent_parity([_OtherLayerState()], None) == 0
    assert resolve_recurrent_parity([_OtherLayerState()], 1) == 1
    assert layer.parity_indexed is True
    with pytest.raises(RuntimeError, match = "parity-indexed"):
        resolve_recurrent_parity([_OtherLayerState(), layer], None)
