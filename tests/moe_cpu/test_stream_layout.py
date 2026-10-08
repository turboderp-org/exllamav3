"""CPU-only host contracts: measured ring budget, native bypass and first-use stream ordering.

Exercise real host methods with simulated CUDA allocations and streams. Numerical layout
consumption and actual GPU synchronization are covered by the kernel tests, not this harness.
"""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.nogpu


@pytest.fixture
def harness(monkeypatch):
    from exllamav3.model import moe_cpu_host as mch

    log, restores = [], []

    class Stream:
        def __init__(self, device = None, name = "copy"):
            self.name = name

        def wait_stream(self, stream):
            log.append(("wait_stream", self.name, stream.name))

        def wait_event(self, event):
            pass

    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self, stream):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 1.0

    @contextmanager
    def stream_context(stream):
        yield

    compute = Stream(name = "compute")

    class CpuTorch:
        version = SimpleNamespace(hip = None)
        cuda = SimpleNamespace(Stream = Stream, Event = Event, current_stream = lambda: compute,
                               device = lambda device: nullcontext(), stream = stream_context)

        def __getattr__(self, name):
            return getattr(torch, name)

        def device(self, device):
            return SimpleNamespace(index = 0)

        def empty(self, *args, **kwargs):
            tensor = torch.empty(*args, **(kwargs | {"device": "cpu"}))
            log.append(("alloc", tensor.numel() * tensor.element_size()))
            return tensor

        def zeros(self, *args, **kwargs):
            return torch.zeros(*args, **(kwargs | {"device": "cpu"}))

    proxy = CpuTorch()
    layouts = {1: (0, 0), 2: (2, 1), 2.5: (8, 0)}

    def inverse(src, dst, *args):
        restores.append(args)
        log.append(("inverse",))
        dst.copy_(src)

    tuning = SimpleNamespace(swizzle = True, stream_fused_t = 0, stream_t_explicit = True,
                             stream_debug = False)
    monkeypatch.setattr(mch, "torch", proxy)
    monkeypatch.setattr(mch, "TUNING", tuning)
    monkeypatch.setattr(mch, "ext", SimpleNamespace(
        exl3_moe_cpu_swizzle_group = lambda K: layouts[K][0],
        exl3_moe_cpu_planar_layout = lambda K: layouts[K][1], moe_unswizzle_trellis = inverse))
    monkeypatch.setattr(mch, "host_to_device", lambda tensor, device: tensor)
    monkeypatch.setattr(mch, "probe_bandwidth", lambda copy: (copy(), 50.0)[1])

    def spec(K, gated = False):
        dims = {name: (16, 16, K) if name != "g" or gated else None for name in ("g", "u", "d")}
        size = int(32 * K)
        return dict(hi = 16, ho = 16, num_experts = 1, activation = 2, proj_dims = dims,
                    proj_bytes = (size if gated else 0, size, size),
                    expert_bytes = size * (3 if gated else 2))

    def host(specs):
        h = mch.MoeCpuHost.__new__(mch.MoeCpuHost)
        h.specs, h._dev_bufs, h.sstate = specs, {}, {}
        h.num_wslots, h.wslot_size = 2, 4096
        h.cap_rows, h.stream_min_rows, h.stream_t, h.batch_experts = 64, 1, 1, 24
        h.aux = {i: {key: [None] for key in ("suh_g", "svh_g", "suh_u", "svh_u", "suh_d", "svh_d")}
                 for i in range(len(specs))}
        h._stream_fused_t = lambda spec, aux, h: 0
        h._stream_recon_layer = lambda st, layer_idx, spec, aux, device: None
        h.pinned, h.next_wslot, h.wseq, h.gpu_base_ptr = True, 0, 0, 0
        h.layer_blocks = [[(0, 0)] for _ in specs]
        h.arena_views = [torch.ones(2048, dtype = torch.int16)]
        h.wviews = [torch.ones(2048, dtype = torch.int16)] * 2
        h._act = lambda spec, g, u: u

        def linear(x, trellis_view, dims, suh, svh, bias, w_scratch,
                   out_dtype = torch.half, group = 0, planar = 0):
            return torch.ones((x.shape[0], dims[1]), dtype = out_dtype)

        h._dq_linear = linear
        return h

    def run(h, index):
        y = torch.ones((2, 16), dtype = torch.half)
        sel, weights = torch.zeros((2, 1), dtype = torch.int64), torch.ones((2, 1))
        st = h._ensure_stream_state("cpu")
        ws = h.next_wslot
        h._submit_prefill_streamed(index, y, sel, weights, h.specs[index], [0], st,
                                  [2], sel.flatten(), sel.flatten() + 1, 0)
        return st, ws

    return SimpleNamespace(torch = proxy, tuning = tuning, spec = spec, host = host, run = run,
                           log = log, restores = restores)


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("hip, swizzle, K, restore", [
    (True, True, 1, False), (False, True, 2, False), (True, True, 2, True),
    (True, True, 2.5, True), (True, False, 2, False),
])
def test_measured_budget_and_runtime_bypass(harness, gated, hip, swizzle, K, restore):
    x = harness
    x.torch.version.hip = "stub-hip" if hip else None
    x.tuning.swizzle = swizzle
    h = x.host([x.spec(K, gated)])
    h.prefill_worst_case_parts(0, 2, "cpu", 2)
    # Two raw slots plus scratch; only HIP packed layouts need two additional native slots.
    expected_bytes = 2 * 4096 * (2 if restore else 1) + 16 * 16 * 2
    assert sum(item[1] for item in x.log if item[0] == "alloc") == expected_bytes
    allocations = len([item for item in x.log if item[0] == "alloc"])
    x.run(h, 0)
    assert len([item for item in x.log if item[0] == "alloc"]) == allocations
    assert bool(x.restores) == restore
    if restore:
        assert next(i for i, item in enumerate(x.log) if item[0] == "wait_stream") < \
               next(i for i, item in enumerate(x.log) if item[0] == "inverse")


@pytest.mark.parametrize("measured", [False, True])
def test_late_packed_layer_and_native_followup(harness, measured):
    x = harness
    x.torch.version.hip = "stub-hip"
    h = x.host([x.spec(1), x.spec(2), x.spec(1)])
    h.prefill_worst_case_parts(0, 2, "cpu", 2)
    x.run(h, 0)
    assert not x.restores
    before_bytes = sum(item[1] for item in x.log if item[0] == "alloc")
    if measured:
        h.prefill_worst_case_parts(1, 2, "cpu", 2)
        assert sum(item[1] for item in x.log if item[0] == "alloc") - before_bytes == 8192
    x.run(h, 1)
    assert sum(item[1] for item in x.log if item[0] == "alloc") - before_bytes == 8192
    assert next(i for i, item in enumerate(x.log) if item[0] == "wait_stream") < \
           next(i for i, item in enumerate(x.log) if item[0] == "inverse")
    before = len(x.log)
    h.prefill_worst_case_parts(2, 2, "cpu", 2)
    x.run(h, 2)
    assert not any(item[0] in ("alloc", "inverse") for item in x.log[before:])
