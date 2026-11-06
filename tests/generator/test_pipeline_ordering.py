"""
PrefillPipeline's guarantees, on a stub model recording what runs where (no GPU; devices are only names here). Every item
walks one route and each device runs its steps first come, first served, so an item can't overtake one issued before it,
whatever the speed of the devices. A failed job's later items are walked through without running, its caller is never
left waiting, and a checkpoint that didn't complete is reported so that it can be dropped.
"""

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from exllamav3.model.prefill_pipeline import PrefillPipeline

pytestmark = pytest.mark.nogpu

D0, D1, D2 = torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")


class StubState:
    """Records the positions its layers were copied at, against what has run on each device"""

    def __init__(self, model, position = 0):
        self.model = model
        self.position = position
        self.last_history = 0
        self.checkpoint_size = 1
        self.cache = SimpleNamespace(get_all_recurrent_layers = lambda: {
            "a": SimpleNamespace(device = D0), "b": SimpleNamespace(device = D1)
        })

    def post_advance(self):
        pass

    def stash_header(self):
        return {"position": self.position}

    def stash_layers(self, layers):
        self.model.hook("stash", layers)
        return {k: (self.position, self.model.done[l.device]) for k, l in layers.items()}


class StubModel:

    def __init__(self, delays = None):
        self.fwd_modules = [None] * 4
        self.last_kv_end = 4
        self._get_prefetch_layers = []
        self.delays = delays or {}
        self.log = []
        self.done = {D0: 0, D1: 0}
        self.lock = threading.Lock()
        self.hook = lambda kind, what: None

    def prepare_inputs(self, input_ids, params):
        return input_ids

    def run_section(self, x, params, a, b, prefill):
        device = D0 if a == 0 else D1
        self.hook("section", (device, params))
        time.sleep(self.delays.get(device, 0))
        with self.lock:
            self.log.append((device, params["chunk"], threading.current_thread().name))
            self.done[device] = params["chunk"] + 1
        return x


STAGES = [(D0, 0, 2), (D1, 2, 4)]


def chunk_params(n, states = None):
    return {"chunk": n, "block_table": torch.zeros(1), "recurrent_states": states}


def make(model, draft = False):
    stages = STAGES + ([(D0, 0, 0)] if draft else [])
    return PrefillPipeline(model, stages, 2 if draft else None)


@pytest.mark.parametrize("delays", [{}, {D1: 0.02}, {D0: 0.02}])
def test_devices_run_chunks_in_order_on_one_thread_each(delays):
    model = StubModel(delays)
    pipe = make(model)
    for n in range(8):
        pipe.prefill(torch.zeros(1, 4), chunk_params(n), "job")
    pipe.flush()
    for device in (D0, D1):
        entries = [e for e in model.log if e[0] == device]
        assert [e[1] for e in entries] == list(range(8))
        assert len({e[2] for e in entries}) == 1
    assert len({e[2] for e in model.log}) == 2
    pipe.close()


@pytest.mark.parametrize("delays", [{}, {D1: 0.02}, {D0: 0.02}])
def test_checkpoint_reads_each_device_behind_the_chunks_before_it(delays):
    model = StubModel(delays)
    pipe = make(model)
    state = StubState(model)
    stashes = []
    for n in range(6):
        pipe.prefill(torch.zeros(1, 4), chunk_params(n, [state]), "job")
        paired = SimpleNamespace(target = state, draft = None, position = state.position, checkpoint_size = 1)
        stashes.append(pipe.stash(paired, "job"))
    pipe.flush()
    assert not pipe.pending_stashes
    for n, stashed in enumerate(stashes):
        # Taken at the position past chunk n, on each device exactly after that chunk
        side = stashed["target"]
        assert side["position"] == 4 * (n + 1)
        assert side["a"] == (4 * (n + 1), n + 1) and side["b"] == (4 * (n + 1), n + 1)
    pipe.close()


def test_later_devices_see_the_state_as_the_chunk_found_it():
    model = StubModel({D1: 0.01})
    pipe = make(model)
    seen = {}

    def hook(kind, what):
        if kind == "section" and what[0] == D1:
            seen[what[1]["chunk"]] = what[1]["recurrent_states"][0].position

    model.hook = hook
    state = StubState(model)
    for n in range(5):
        pipe.prefill(torch.zeros(1, 4), chunk_params(n, [state]), "job")
    pipe.flush()
    assert seen == {n: 4 * n for n in range(5)}
    assert state.position == 20
    pipe.close()


def test_draft_step_runs_on_its_device_after_the_target():
    model = StubModel({D1: 0.01})
    pipe = make(model, draft = True)
    order = []

    def then(n):
        def step(params):
            with model.lock:
                # The chunk has been through every device of the target by now
                order.append((n, model.done[D1], threading.current_thread().name))
        return step

    for n in range(5):
        pipe.prefill(torch.zeros(1, 4), chunk_params(n), "job", then = then(n))
    pipe.flush()
    assert [o[0] for o in order] == list(range(5))
    assert all(done >= n + 1 for n, done, _ in order)
    # It shares the thread of the device it's on, which is what keeps it apart from the target's work there
    d0_thread = next(e[2] for e in model.log if e[0] == D0)
    assert {o[2] for o in order} == {d0_thread}
    pipe.close()


def test_failure_on_a_later_device_is_reported_and_never_strands_the_caller():
    model = StubModel()
    pipe = make(model)

    def hook(kind, what):
        if kind == "section" and what[0] == D1 and what[1]["chunk"] == 1:
            raise RuntimeError("boom")

    model.hook = hook
    for n in range(6):
        pipe.prefill(torch.zeros(1, 4), chunk_params(n), "bad")
    pipe.prefill(torch.zeros(1, 4), chunk_params(100), "good")
    pipe.flush()
    failures = pipe.take_failures()
    assert [(o, str(e)) for o, e in failures] == [("bad", "boom")]
    assert (D1, 100) in [e[:2] for e in model.log]
    # Nothing of the failed job's runs past the failure on the device that reported it
    assert [e[1] for e in model.log if e[0] == D1] == [0, 100]
    pipe.close()


def test_failed_owner_stash_is_reported_as_incomplete():
    model = StubModel()
    pipe = make(model)
    state = StubState(model)
    armed = [True]

    def hook(kind, what):
        if kind == "stash" and armed[0]:
            armed[0] = False
            raise RuntimeError("boom")

    model.hook = hook
    pipe.prefill(torch.zeros(1, 4), chunk_params(0, [state]), "job")
    paired = SimpleNamespace(target = state, draft = None, position = state.position, checkpoint_size = 1)
    stashed = pipe.stash(paired, "job")
    pipe.flush()
    assert id(stashed) in pipe.failed_stashes and id(stashed) not in pipe.pending_stashes
    assert [o for o, _ in pipe.take_failures()] == ["job"]
    pipe.close()


def test_chunks_in_flight_are_bounded():
    release = threading.Event()
    model = StubModel()
    model.hook = lambda kind, what: release.wait(5) if kind == "section" and what[0] == D1 else None
    pipe = make(model)
    issued = []

    def issue():
        for n in range(12):
            pipe.prefill(torch.zeros(1, 4), chunk_params(n), "job")
            issued.append(n)

    t = threading.Thread(target = issue, daemon = True)
    t.start()
    time.sleep(0.1)
    assert len(issued) <= pipe.max_passes
    release.set()
    t.join(5)
    assert len(issued) == 12
    pipe.close()


def test_flush_is_a_no_op_on_a_worker():
    model = StubModel()
    pipe = make(model, draft = True)
    # A draft model's own prefill calls flush from the draft step: waiting there would wait for itself
    pipe.prefill(torch.zeros(1, 4), chunk_params(0), "job", then = lambda params: pipe.flush())
    pipe.flush()
    assert not pipe.busy
    pipe.close()


# Which splits can run this way

def module(device, *children):
    m = SimpleNamespace(device = device)

    class Mod:
        def __iter__(self):
            yield m
            yield from children

        device = m.device

    return Mod()


def plan_model(devices, **kw):
    mods = [(module(d), 0, i) for i, d in enumerate(devices)]
    return SimpleNamespace(fwd_modules = mods, loaded_tp = False, config = SimpleNamespace(), **kw)


def plan_cache(devices, state_cls = SimpleNamespace()):
    layers = {i: SimpleNamespace(device = d) for i, d in enumerate(devices)}
    return SimpleNamespace(recurrent_state_cls = state_cls, get_all_recurrent_layers = lambda: layers)


def test_plan_cuts_the_forward_at_device_changes():
    stages, draft = PrefillPipeline.plan(plan_model(["cpu", "cuda:0", "cuda:0", "cuda:1", "cuda:1"]), None)
    assert stages == [(D0, 0, 3), (D1, 3, 5)] and draft is None


def test_plan_appends_the_draft_device():
    draft = plan_model(["cpu", "cuda:1", "cuda:1"], caps = {})
    stages, stage = PrefillPipeline.plan(plan_model(["cuda:0", "cuda:1"]), None, draft)
    assert stage == 2 and stages[2] == (D1, 0, 0)


@pytest.mark.parametrize("devices", [["cuda:0", "cuda:0"], ["cuda:0", "cuda:1", "cuda:0"]])
def test_plan_declines_one_device_or_a_revisit(devices):
    assert PrefillPipeline.plan(plan_model(devices), None) is None


def test_plan_declines_tp_and_cpu_offload():
    m = plan_model(["cuda:0", "cuda:1"])
    m.loaded_tp = True
    assert PrefillPipeline.plan(m, None) is None
    m = plan_model(["cuda:0", "cuda:1"])
    m.config.moe_cpu_hosts = {"host": object()}
    assert PrefillPipeline.plan(m, None) is None


def test_plan_declines_a_layer_with_a_part_on_another_device():
    m = plan_model(["cuda:0", "cuda:1"])
    m.fwd_modules[1] = (module("cuda:1", SimpleNamespace(device = "cuda:0")), 0, 1)
    assert PrefillPipeline.plan(m, None) is None


def test_plan_declines_a_ring_state_without_a_layer_on_the_first_device():
    ring = SimpleNamespace(guaranteed_rollback = 256)
    m = plan_model(["cuda:0", "cuda:1"])
    assert PrefillPipeline.plan(m, plan_cache(["cuda:1"], ring)) is None
    assert PrefillPipeline.plan(m, plan_cache(["cuda:0", "cuda:1"], ring)) is not None
    assert PrefillPipeline.plan(m, plan_cache(["cuda:1"])) is not None


@pytest.mark.parametrize("draft", [
    ("cuda:0", "cuda:1"),
])
def test_plan_declines_a_draft_on_several_devices(draft):
    d = plan_model(list(draft), caps = {})
    assert PrefillPipeline.plan(plan_model(["cuda:0", "cuda:1"]), None, d) is None


@pytest.mark.parametrize("cap", ["load_with_target", "draft_uses_target_cache"])
def test_plan_declines_a_draft_tied_to_the_target(cap):
    d = plan_model(["cuda:0"], caps = {cap: True})
    assert PrefillPipeline.plan(plan_model(["cuda:0", "cuda:1"]), None, d) is None
