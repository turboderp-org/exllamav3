"""
Promote-on-miss is a per-step hook of the main model's forward. An MTP head shares the main model's config, and with
it the split-module registry whose first entry drives the hook, but it loads first and drafts several times per step.
The landed transfers take effect at the pass's first split layer; the decide (snapshot, promotions, demotions) waits
for the generator's promote_tick once the pass is queued, or for the next pass when nothing ticks. The step's hit-count
snapshot and the landed map updates are one host-device copy per device each, not one per module. Exercised with
stand-in modules carrying the state the step reads.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exllamav3.model.config import InferParams
from exllamav3.modules import block_sparse_mlp_cpu as bsc
from exllamav3.modules.block_sparse_mlp_cpu import BlockSparseMLP_CPU


def _split_module(ip, device = "cpu", num_experts = 8, first = 4):
    with torch.inference_mode():
        hist = torch.rand(num_experts, device = device)
    return SimpleNamespace(
        config = SimpleNamespace(infer_params = ip),
        cpu_component = getattr(ip, "moe_cpu_component", "text"),
        device = device,
        num_experts = num_experts,
        cpu_split_first = first,
        cpu_host = SimpleNamespace(stream_min_rows = 32),
        _split_spares = 2,
        _split_pending = [],
        _split_free_cpu = [],
        _split_mp = np.arange(num_experts),
        _split_hist = hist,
        _swap_tick_count = 0,
        _split_step_count = 0,
        _split_submit_rows = 1,
    )


def _record_steps(monkeypatch):
    """Replaces the step's two halves with recorders: land returns (and keeps) a minimal step state"""
    calls = []
    def land(ip, reg):
        calls.append("land")
        ps = getattr(ip, "moe_cpu_promote_state", None) or SimpleNamespace(decide_rows = None)
        ip.moe_cpu_promote_state = ps
        return ps
    monkeypatch.setattr(bsc, "_split_promote_land", land)
    monkeypatch.setattr(bsc, "_split_promote_decide", lambda ps, rows: calls.append(("decide", rows)))
    return calls


def _text_layers(ip, n = 3):
    layers = [_split_module(ip) for _ in range(n)]
    for m in layers:
        bsc.register_split_module(ip, m)
    return layers


@pytest.mark.nogpu
def test_main_model_drives_promote_step(monkeypatch):
    calls = _record_steps(monkeypatch)
    ip = InferParams()
    ip.moe_cpu_component = "mtp"
    mtp = _split_module(ip)
    bsc.register_split_module(ip, mtp)                      # the MTP head loads before the main model
    ip.moe_cpu_component = "text"
    layers = [_split_module(ip) for _ in range(3)]
    for m in layers:
        bsc.register_split_module(ip, m)
    assert ip.moe_cpu_swap_modules == layers + [mtp]

    for _ in range(3):                                      # draft forwards
        BlockSparseMLP_CPU._split_swap_tick(mtp)
    assert calls == []
    BlockSparseMLP_CPU._split_swap_tick(layers[0])          # verify forward
    assert calls == ["land"]


@pytest.mark.nogpu
def test_decide_follows_the_queued_pass(monkeypatch):
    calls = _record_steps(monkeypatch)
    ip = InferParams()
    bsc.promote_tick(ip)                                    # nothing ran yet: no state, nothing to decide
    layers = _text_layers(ip)
    BlockSparseMLP_CPU._split_swap_tick(layers[0])
    assert calls == ["land"]
    bsc.promote_tick(ip)
    assert calls == ["land", ("decide", 1)]
    bsc.promote_tick(ip)                                    # once per pass
    assert calls == ["land", ("decide", 1)]


@pytest.mark.nogpu
def test_unticked_pass_is_decided_at_the_next_pass(monkeypatch):
    """A pass the generator does not tick (prefill, or a driver without a generator) is decided before the next
    pass lands its transfers"""
    calls = _record_steps(monkeypatch)
    ip = InferParams()
    layers = _text_layers(ip)
    BlockSparseMLP_CPU._split_swap_tick(layers[0])
    layers[0]._split_submit_rows = 2
    BlockSparseMLP_CPU._split_swap_tick(layers[0])
    assert calls == ["land", ("decide", 1), "land"]
    bsc.promote_tick(ip)
    assert calls == ["land", ("decide", 1), "land", ("decide", 2)]


def test_promote_snapshot_is_one_copy_per_device(device):
    ip = SimpleNamespace()
    modules = [_split_module(ip, device) for _ in range(6)]
    with torch.inference_mode():
        ps = bsc._split_promote_land(ip, modules)
        bsc._split_promote_decide(ps, 1)
        before = [m._split_hist.clone() for m in modules]
        with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU]) as prof:
            bsc._split_promote_decide(ps, 1)
        torch.cuda.synchronize(device)

    assert sum(e.name == "aten::copy_" for e in prof.events()) == 1
    ps = ip.moe_cpu_promote_state
    for i, m in enumerate(modules):
        assert torch.equal(ps.snap[i, :m.num_experts], before[i].cpu())
        assert torch.allclose(m._split_hist, before[i] * bsc._PROMOTE_DECAY)


def _missing_module(ip, device, host, layer):
    """A split module with free GPU slots whose hit counts the test sets before a decide"""
    m = _split_module(ip, device)
    slot = SimpleNamespace(inner = None)
    m.cpu_host, m.cpu_layer_idx, m.gated = host, layer, False
    m.ups = m.downs = [slot] * m.num_experts
    m._split_free_gpu = [2, 3]
    return m


@pytest.mark.parametrize("budget, issued", [(0, 0), (2, 2), (24, 6)])
def test_budget_caps_promotions_per_pass(device, monkeypatch, budget, issued):
    """Every layer routes one expert its worker holds: a pass promotes at most the budget, one per layer"""
    monkeypatch.setattr(bsc, "_PROMOTE_BUDGET", budget)
    promoted = []
    def promote_expert(layer, c, linears):
        promoted.append((layer, c))
        ev = torch.cuda.Event()
        ev.record()
        return ev
    stream = torch.cuda.Stream(device)
    host = SimpleNamespace(
        stream_min_rows = 32, promote_ready = lambda layer, c: True, promo_stream = lambda dev: stream,
        promote_expert = promote_expert,
    )
    ip = SimpleNamespace()
    modules = [_missing_module(ip, device, host, i) for i in range(6)]
    with torch.inference_mode(), torch.cuda.device(device):
        ps = bsc._split_promote_land(ip, modules)
        bsc._split_promote_decide(ps, 1)                    # first snapshot goes out: nothing to rank yet
        assert promoted == []
        for m in modules:
            m._split_hist[5] = 10.0                         # expert 5 lives in worker slot 1
        bsc._split_promote_decide(ps, 1)                    # snapshot of the counts above goes out
        bsc._split_promote_decide(ps, 1)                    # and is ranked here
        torch.cuda.synchronize(device)

    assert len(promoted) == issued
    assert len({layer for layer, _ in promoted}) == issued
    assert all(c == 1 for _, c in promoted)
    for m in modules:
        took = m.cpu_layer_idx in {layer for layer, _ in promoted}
        assert len(m._split_pending) == took and len(m._split_free_gpu) == 2 - took


def _landed(kind, r, gpu, cpu):
    ev = torch.cuda.Event()
    ev.record()
    return dict(kind = kind, r = r, gpu = gpu, cpu = cpu, ev = ev)


def test_landed_transfers_upload_in_one_copy_per_device(device):
    ip = SimpleNamespace()
    modules = [_split_module(ip, device) for _ in range(3)]
    for m in modules:
        m._split_map = torch.tensor(m._split_mp, dtype = torch.long, device = device)
        m._split_free_gpu, m._split_promo_n, m._split_demo_n = [], 0, 0
    with torch.inference_mode(), torch.cuda.device(device):
        bsc._split_promote_land(ip, modules)
        for m in modules:
            # expert 5 (worker slot 1) lands in GPU slot 2; expert 1 (GPU slot 1) lands in worker slot 3
            m._split_pending = [_landed("p", 5, 2, 1), _landed("d", 1, 1, 3)]
        torch.cuda.synchronize(device)
        with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU]) as prof:
            bsc._split_promote_land(ip, modules)
        torch.cuda.synchronize(device)

    assert sum(e.name == "aten::copy_" for e in prof.events()) == 1
    for m in modules:
        assert m._split_pending == []
        assert m._split_mp[5] == 2 and m._split_mp[1] == m.cpu_split_first + 3
        assert m._split_map.cpu().tolist() == m._split_mp.tolist()
        assert m._split_free_cpu == [1] and m._split_free_gpu == [1]


def test_unload_drains_promotion_transfers_first():
    """The generator waits on the compute stream only, so a pass's transfers can still run at unload: they write the
    GPU slots and the worker's arena pages the unload releases"""
    calls = []
    host = SimpleNamespace(promo_drain = lambda: calls.append("drain"), unregister = lambda: calls.append("unregister"))
    m = SimpleNamespace(
        cpu_split_first = 4, cpu_host = host, cpu_offload = False, _split_map = None,
        _split_saved = (None,) * 7, config = SimpleNamespace(infer_params = InferParams()),
    )
    BlockSparseMLP_CPU.cpu_unload(m)
    assert calls == ["drain", "unregister"]


def test_promo_drain_waits_on_every_device_stream():
    import exllamav3.model.moe_cpu_host as mch
    waited = []
    host = object.__new__(mch.MoeCpuHost)
    host._promo = {i: dict(stream = SimpleNamespace(synchronize = lambda i = i: waited.append(i))) for i in (0, 1)}
    host.promo_drain()
    assert waited == [0, 1]
