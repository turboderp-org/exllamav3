"""
CPU expert-offload worker loop (cpu/moe_handoff.cu exl3_moe_cpu_worker_run) driven through the handoff segment
without a model (testlib.moe_handoff): synthetic layers registered with exl3_moe_cpu_make_layer, the parent's role
played by host writes (protocol tests) or by GPU flag ops and copies on a stream, as MoeCpuHost issues and
collects jobs (end-to-end test). Each test module function runs in a fresh process (the layer registry, the
thread pool and its pinning are process-wide).

Contracts:
- The worker sets `ready` on entry and returns (stager joined) once `quit` is set: when idle, while a compute job
  waits for its data_ready flag and while a stage job waits for its pinned_free flag.
- Compute jobs are taken in ring order (across the 256-entry ring's wraparound); a job runs only after its slot's
  data_ready flag reaches the job's seq (cyclic >=), then done[slot] = seq. kind COMPUTE: out[:rows] =
  exl3_moe_cpu_forward(layer, x[:rows], sel[:rows], w[:rows]) of the slot's sections (an all-unselected job
  writes zeros); out rows past `rows` and the input sections are untouched. kind COMPUTE_GATED: the same, except a
  job with no selected expert (all sel < 0) skips the compute and leaves out untouched. `layer` is the
  registration index of exl3_moe_cpu_make_layer in the worker's process. Reference: exl3_moe_cpu_forward on the
  same inputs (bitwise: the forward is deterministic for any thread count; it is itself tested against an fp64
  dense reference in tests/moe_cpu/test_wide_rows.py).
- Stage jobs (own ring, own thread) wait until pinned_free[wslot] reaches prev_seq, then copy the listed experts'
  trellis bytes (gate, up, down per expert, gate absent for gateless layers) back to back into the weight slot
  and publish stage_done[wslot] = seq; nothing past the copied bytes is written. They complete while a compute
  job is blocked. Reference: the registered trellis tensors' bytes.
- exl3_moe_cpu_set_prof(True) reports the forward's phase times every 512 profiled forwards (one line on stdout)
  and changes no output; with it off nothing is printed.
- The segment layout of moe_cpu_host.py (offsets, job size) matches moe_handoff.h.
"""

import os
import tempfile
import time

import pytest
import torch

from testlib.exl3 import rand_experts
from testlib.isolated import device_env, run_isolated
from testlib.moe_handoff import (ABORT, CONSUMED, DATA_READY, DONE, KIND_COMPUTE, KIND_COMPUTE_GATED, PASS_WAKE,
                                 PINNED_FREE, QUIT, READY, STAGE_DONE, Segment)

pytestmark = pytest.mark.cpu_flags("avx2")

H, TOPK, CAP_ROWS, NUM_SLOTS = 512, 3, 8, 3
LAYERS = [dict(E = 6, I = 256, K = 4, gated = True), dict(E = 4, I = 384, K = 3, gated = False)]
SENTINEL = 0x7B


def make_layers(seed: int = 1):
    """Register the LAYERS in this process: [(handle, experts dict, gated)]"""
    from exllamav3.ext import exllamav3_ext as ext
    gen = torch.Generator().manual_seed(seed)
    out = []
    for spec in LAYERS:
        ex = rand_experts(spec["E"], H, spec["I"], spec["K"], gen)
        lists = []
        for p in ("g", "u", "d"):
            if p == "g" and not spec["gated"]:
                lists += [[], [], []]
                continue
            lists += [[v[i].contiguous() for v in ex[p]] for i in range(3)]
        h = ext.exl3_moe_cpu_make_layer(*lists, [], [], [], 0 if spec["gated"] else 2, 0.0, 0)
        out.append((h, ex, spec["gated"]))
    return out


def job_inputs(gen, layer: int, rows: int, mode: str = "random"):
    """x [rows, H] fp16, sel [rows, TOPK] int32 (mode: random | none (all -1) | partial (some -1)), w fp16"""
    E = LAYERS[layer]["E"]
    x = (torch.randn(rows, H, generator = gen) * 0.1).half()
    sel = torch.stack([torch.randperm(E, generator = gen)[:TOPK] for _ in range(rows)]).int()
    if mode == "none":
        sel[:] = -1
    elif mode == "partial":
        sel[torch.rand(rows, TOPK, generator = gen) < 0.5] = -1
        sel[0, 1:] = -1
        sel[0, 0] = 0
    w = torch.rand(rows, TOPK, generator = gen).half()
    return x, sel, w


def forward_ref(handle, x, sel, w, threads = 2):
    from exllamav3.ext import exllamav3_ext as ext
    out = torch.empty(x.shape[0], H, dtype = torch.float)
    ext.exl3_moe_cpu_forward(handle, x, sel, w, out, threads)
    return out


def sentinel_rows(n: int) -> torch.Tensor:
    return torch.full((n * H * 4,), SENTINEL, dtype = torch.uint8).view(torch.float).view(n, H)


def staged_bytes(ex, gated: bool, experts) -> torch.Tensor:
    parts = []
    for e in experts:
        for p in (("g", "u", "d") if gated else ("u", "d")):
            parts.append(ex[p][e][0].contiguous().view(torch.uint8).reshape(-1))
    return torch.cat(parts)


def per_expert_bytes(spec) -> int:
    k16, i16 = H // 16, spec["I"] // 16
    return (3 if spec["gated"] else 2) * k16 * i16 * 16 * spec["K"] * 2


WSLOT_SIZE = 3 * max(per_expert_bytes(s) for s in LAYERS) + 4096


# ---------------------------------------------------------------------------------------------------------------

def protocol_child() -> dict:
    """Host-driven protocol run; returns {check: value} for the tests"""
    torch.set_num_threads(1)
    layers = make_layers()
    R = {"handles": [h for h, _, _ in layers]}
    gen = torch.Generator().manual_seed(2)
    seg = Segment(NUM_SLOTS, CAP_ROWS, H, H, TOPK, num_wslots = 2, wslot_size = WSLOT_SIZE, sentinel = SENTINEL)
    state = dict(seq = 0, slot = 0)
    mismatches = []

    def stage_inputs(slot, layer, rows, mode):
        x, sel, w = job_inputs(gen, layer, rows, mode)
        seg.section(slot, "x")[:rows] = x
        seg.section(slot, "sel")[:rows] = sel
        seg.section(slot, "w")[:rows] = w
        seg.section(slot, "out").view(torch.uint8)[:] = SENTINEL
        return x, sel, w

    def check(name, slot, layer, rows, x, sel, w, mode, kind):
        out = seg.section(slot, "out")
        if mode == "none" and kind == KIND_COMPUTE_GATED:
            want = sentinel_rows(rows)
        elif mode == "none":
            want = torch.zeros(rows, H)
        else:
            want = forward_ref(layers[layer][0], x, sel, w)
        ok = torch.equal(out[:rows].view(torch.int32), want.view(torch.int32)) \
            and torch.equal(out[rows:].view(torch.int32), sentinel_rows(CAP_ROWS - rows).view(torch.int32)) \
            and torch.equal(seg.section(slot, "x")[:rows], x) and torch.equal(seg.section(slot, "sel")[:rows], sel) \
            and torch.equal(seg.section(slot, "w")[:rows], w)
        if not ok:
            mismatches.append(name)

    def job(name, layer, rows, mode = "random", kind = KIND_COMPUTE, release = True):
        state["seq"] += 1
        seq, slot = state["seq"], state["slot"]
        state["slot"] = (slot + 1) % NUM_SLOTS
        x, sel, w = stage_inputs(slot, layer, rows, mode)
        seg.push_job(seq, layer, rows, TOPK, slot, kind)
        if release:
            seg.set_flag(DATA_READY, slot, seq)
            seg.wait(DONE, slot, seq)
            check(name, slot, layer, rows, x, sel, w, mode, kind)
        return seq, slot, (x, sel, w, mode, kind, layer, rows)

    try:
        seg.start(threads = 2, stage_threads = 2)
        R["ready"] = seg.word(READY)

        # Plain compute jobs over both layers, all slots, 1..cap_rows rows
        for layer in (0, 1):
            for rows in (1, 3, CAP_ROWS):
                job(f"compute_l{layer}_r{rows}", layer, rows)
        job("compute_none", 0, 4, "none")
        job("gated_none", 1, 4, "none", KIND_COMPUTE_GATED)
        job("gated_partial", 0, 5, "partial", KIND_COMPUTE_GATED)
        job("compute_partial", 1, 5, "partial")
        seg.set_word(PASS_WAKE, seg.word(PASS_WAKE) + 1)
        job("after_pass_wake", 0, 2)

        # A job does not run before its data_ready flag reaches its seq; a later value releases it too
        seq, slot, args = job("gating", 1, 6, release = False)
        time.sleep(0.3)
        R["gating_done_early"] = seg.flag(DONE, slot) == seq
        R["gating_out_untouched"] = torch.equal(seg.section(slot, "out").view(torch.uint8),
                                                torch.full_like(seg.section(slot, "out").view(torch.uint8), SENTINEL))
        seg.set_flag(DATA_READY, slot, seq + 5)
        seg.wait(DONE, slot, seq)
        x, sel, w, mode, kind, layer, rows = args
        check("gating", slot, layer, rows, x, sel, w, mode, kind)

        # Pipelined: one job per slot, released out of order
        pend = [job(f"pipelined_{i}", i % 2, i + 2, release = False) for i in range(NUM_SLOTS)]
        for seq, slot, _ in reversed(pend):
            seg.set_flag(DATA_READY, slot, seq)
        for seq, slot, (x, sel, w, mode, kind, layer, rows) in pend:
            seg.wait(DONE, slot, seq)
            check(f"pipelined_{slot}", slot, layer, rows, x, sel, w, mode, kind)

        # Ring wraparound (256 entries)
        for i in range(300):
            job(f"ring_{i}", i % 2, 1 + i % 3)
        R["jobs_head"], R["jobs_tail"] = seg.word(320), seg.word(256)

        # Stage jobs: blocked on pinned_free, then expert trellis bytes back to back
        h0, ex0, g0 = layers[0]
        seg.push_stage(1, 0, [5, 0, 2], 0, prev_seq = 1)
        time.sleep(0.3)
        R["stage_done_early"] = seg.flag(STAGE_DONE, 0) != 0
        R["stage_slot_untouched_early"] = bool((seg.wslot_bytes(0) == SENTINEL).all())
        seg.set_flag(PINNED_FREE, 0, 1)
        seg.wait(STAGE_DONE, 0, 1)
        want = staged_bytes(ex0, g0, [5, 0, 2])
        got = torch.from_numpy(seg.wslot_bytes(0).copy())
        R["stage_gated_bytes"] = torch.equal(got[: want.numel()], want)
        R["stage_gated_tail_untouched"] = bool((got[want.numel():] == SENTINEL).all())

        # While a compute job waits for its data, a stage job (gateless layer) still completes
        seq, slot, args = job("blocked_compute", 0, 3, release = False)
        h1, ex1, g1 = layers[1]
        seg.push_stage(2, 1, [3, 1], 1, prev_seq = 0)
        seg.wait(STAGE_DONE, 1, 2)
        R["stage_while_compute_blocked"] = seg.flag(DONE, slot) != seq
        want = staged_bytes(ex1, g1, [3, 1])
        got = torch.from_numpy(seg.wslot_bytes(1).copy())
        R["stage_gateless_bytes"] = torch.equal(got[: want.numel()], want)
        R["stage_gateless_tail_untouched"] = bool((got[want.numel():] == SENTINEL).all())
        seg.set_flag(DATA_READY, slot, seq)
        seg.wait(DONE, slot, seq)
        x, sel, w, mode, kind, layer, rows = args
        check("blocked_compute", slot, layer, rows, x, sel, w, mode, kind)

        R["stopped_idle"] = seg.stop()
        R["error"] = [str(e) for e in seg.error]

        # Second run: quit while a compute job waits for data_ready and a stage job for pinned_free
        seg.set_word(QUIT, 0)
        seg.set_word(READY, 0)
        seg.start(threads = 2, stage_threads = 2)
        state["seq"] += 1
        seg.push_job(state["seq"], 0, 1, TOPK, 0)
        seg.push_stage(3, 0, [1], 1, prev_seq = 100)
        time.sleep(0.2)
        R["stopped_waiting"] = seg.stop()
        R["error2"] = [str(e) for e in seg.error]
        R["abort"] = seg.word(ABORT)
    finally:
        seg.set_word(QUIT, 1)
        seg.close()
    R["mismatches"] = mismatches
    return R


@pytest.fixture(scope = "module")
def protocol():
    return run_isolated(protocol_child, env = {"CUDA_VISIBLE_DEVICES": ""}, timeout = 600)


@pytest.mark.nogpu
def test_ready_and_layer_handles(protocol):
    assert protocol["ready"] == 1
    assert protocol["handles"] == [0, 1], "layer indices in jobs are registration indices"


@pytest.mark.nogpu
def test_compute_jobs_match_forward(protocol):
    assert not protocol["mismatches"], f"jobs whose output (or untouched regions) differ: {protocol['mismatches']}"


@pytest.mark.nogpu
def test_data_ready_gating(protocol):
    assert not protocol["gating_done_early"], "done published before data_ready reached the job's seq"
    assert protocol["gating_out_untouched"], "output written before data_ready reached the job's seq"


@pytest.mark.nogpu
def test_ring_wraparound(protocol):
    assert protocol["jobs_head"] == protocol["jobs_tail"] > 256


@pytest.mark.nogpu
def test_stage_jobs(protocol):
    assert not protocol["stage_done_early"] and protocol["stage_slot_untouched_early"], \
        "stage job ran before pinned_free reached prev_seq"
    assert protocol["stage_gated_bytes"] and protocol["stage_gated_tail_untouched"]
    assert protocol["stage_gateless_bytes"] and protocol["stage_gateless_tail_untouched"]
    assert protocol["stage_while_compute_blocked"], "stage job waited behind a blocked compute job"


@pytest.mark.nogpu
def test_quit(protocol):
    assert protocol["stopped_idle"], "worker did not return after quit (idle)"
    assert protocol["stopped_waiting"], "worker did not return after quit while jobs were waiting on their flags"
    assert protocol["error"] == [] and protocol["error2"] == []
    assert protocol["abort"] == 0


# ---------------------------------------------------------------------------------------------------------------

def gpu_child(memops: bool) -> dict:
    """GPU-driven run in one process: per job, H2D-staged inputs (D2H into the mapped slot), data_ready published
    and done awaited by stream flag ops, output read back on the stream, consumed published; slots reused behind
    the consumed flag, as MoeCpuHost._issue_compute / _collect_one. Everything enqueued before one synchronize"""
    from exllamav3.ext import exllamav3_ext as ext
    torch.set_num_threads(1)
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    ext.exl3_moe_cpu_set_memops(memops)
    layers = make_layers()
    seg = Segment(NUM_SLOTS, CAP_ROWS, H, H, TOPK)
    base = seg.register()
    gen = torch.Generator().manual_seed(3)
    jobs = [(i % 2, 1 + (i * 5) % CAP_ROWS) for i in range(40)]
    inputs = [job_inputs(gen, layer, rows) for layer, rows in jobs]
    outs = [torch.zeros(rows, H, device = dev) for _, rows in jobs]
    last = [0] * NUM_SLOTS
    try:
        seg.start(threads = 2, stage_threads = 1)
        dev_in = [(x.to(dev), sel.to(dev), w.to(dev)) for x, sel, w in inputs]
        torch.cuda.synchronize()
        for i, ((layer, rows), (x, sel, w)) in enumerate(zip(jobs, dev_in)):
            seq, slot = i + 1, i % NUM_SLOTS
            seg.push_job(seq, layer, rows, TOPK, slot)
            if last[slot]:
                ext.exl3_moe_flag_wait(base + CONSUMED + 64 * slot, last[slot], base + ABORT)
            seg.section(slot, "x")[:rows].copy_(x, non_blocking = True)
            seg.section(slot, "sel")[:rows].copy_(sel, non_blocking = True)
            seg.section(slot, "w")[:rows].copy_(w, non_blocking = True)
            ext.exl3_moe_flag_write(base + DATA_READY + 64 * slot, seq)
            ext.exl3_moe_flag_wait(base + DONE + 64 * slot, seq, base + ABORT)
            outs[i].copy_(seg.section(slot, "out")[:rows], non_blocking = True)
            ext.exl3_moe_flag_write(base + CONSUMED + 64 * slot, seq)
            last[slot] = seq
        torch.cuda.synchronize()
        stopped = seg.stop()
        err = [str(e) for e in seg.error]
        abort = seg.word(ABORT)
    finally:
        seg.set_word(QUIT, 1)
        seg.close()
    bad = [i for i, ((layer, rows), (x, sel, w)) in enumerate(zip(jobs, inputs))
           if not torch.equal(outs[i].cpu(), forward_ref(layers[layer][0], x, sel, w))]
    return dict(bad = bad, stopped = stopped, err = err, abort = abort)


@pytest.mark.parametrize("memops", [True, False], ids = ["memops", "kernels"])
def test_gpu_driven_jobs(device, memops):
    r = run_isolated(gpu_child, memops, env = device_env(device), timeout = 600)
    assert r["err"] == [] and r["abort"] == 0 and r["stopped"]
    assert not r["bad"], f"jobs with wrong output: {r['bad']}"


# ---------------------------------------------------------------------------------------------------------------

def prof_child(path: str) -> dict:
    """512 forwards with profiling on, 512 with it off; stdout (fd 1) to `path`"""
    import sys
    from exllamav3.ext import exllamav3_ext as ext
    torch.set_num_threads(1)
    (h, _, _), _ = make_layers()
    gen = torch.Generator().manual_seed(4)
    x, sel, w = job_inputs(gen, 0, 2)
    sys.stdout.flush()
    saved = os.dup(1)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1)
    try:
        ext.exl3_moe_cpu_set_prof(False)
        ref = forward_ref(h, x, sel, w, 1)
        ext.exl3_moe_cpu_set_prof(True)
        on = [forward_ref(h, x, sel, w, 1) for _ in range(512)]
        os.fsync(1)
        lines_on = open(path).read()
        ext.exl3_moe_cpu_set_prof(False)
        off = [forward_ref(h, x, sel, w, 1) for _ in range(512)]
        lines_all = open(path).read()
    finally:
        os.dup2(saved, 1)
        os.close(fd)
    return dict(lines_on = lines_on, lines_all = lines_all,
                same = all(torch.equal(o, ref) for o in on + off))


@pytest.mark.nogpu
def test_set_prof():
    with tempfile.TemporaryDirectory() as tmp:
        r = run_isolated(prof_child, os.path.join(tmp, "out.txt"), env = {"CUDA_VISIBLE_DEVICES": ""})
    reports = [l for l in r["lines_on"].splitlines() if "moe_cpu prof" in l]
    assert len(reports) == 1 and "512 jobs" in reports[0], r["lines_on"]
    assert r["lines_all"] == r["lines_on"], "reports printed with profiling off"
    assert r["same"], "profiling changed the output"


@pytest.mark.nogpu
def test_host_layout_matches_header():
    import testlib.moe_handoff as hdr
    from exllamav3.model import moe_cpu_host as host
    for name in ("MOE_JOB_RING", "MOE_MAX_SLOTS", "MOE_MAX_WSLOTS", "MOE_JOB_BYTES", "MOE_CTRL_JOBS_OFFSET",
                 "MOE_SLOT_FLAGS_OFFSET", "MOE_FLAGS_SIZE", "MOE_STAGE_RING", "MOE_STAGE_TAIL_OFFSET",
                 "MOE_STAGE_HEAD_OFFSET", "MOE_STAGE_JOBS_OFFSET", "MOE_CTRL_SIZE"):
        assert getattr(host, name) == getattr(hdr, name), name


def _empty_forward_worker():
    from exllamav3.ext import exllamav3_ext as ext
    (h, _, _), _ = make_layers()
    gen = torch.Generator().manual_seed(5)
    res = {}
    # No rows: nothing to do
    out = sentinel_rows(0)
    ext.exl3_moe_cpu_forward(h, torch.empty(0, H, dtype = torch.half), torch.empty(0, TOPK, dtype = torch.int),
                             torch.empty(0, TOPK, dtype = torch.half), out, 2)
    res["rows0"] = out.shape == (0, H)
    # No experts per token: every row is the empty sum
    x, _, _ = job_inputs(gen, 0, 3)
    out = sentinel_rows(3)
    ext.exl3_moe_cpu_forward(h, x, torch.empty(3, 0, dtype = torch.int), torch.empty(3, 0, dtype = torch.half), out, 2)
    res["topk0_zero"] = bool((out == 0).all())

    def raises(fn, match):
        try:
            fn()
        except RuntimeError as e:
            return match in str(e)
        return False

    x, sel, w = job_inputs(gen, 0, 2)
    res["bad_shape"] = raises(lambda: ext.exl3_moe_cpu_forward(h, x, sel[:1], w[:1], sentinel_rows(2), 2),
                              "selected and weights (rows, top_k)")
    res["bad_out"] = raises(lambda: ext.exl3_moe_cpu_forward(h, x, sel, w, sentinel_rows(1), 2),
                            "x and out must be (rows, hidden_size)")
    res["no_experts"] = raises(lambda: ext.exl3_moe_cpu_make_layer(*([[]] * 12), 0, 0.0, 0),
                               "exl3_moe_cpu_make_layer: no experts")
    z = torch.zeros(0, H // 16, 64, dtype = torch.int16)
    s0, sh = torch.zeros(0, dtype = torch.half), torch.ones(H, dtype = torch.half)
    res["empty_dim"] = raises(lambda: ext.exl3_moe_cpu_make_layer([], [], [], [z], [s0], [sh], [z], [s0], [sh],
                                                                  [], [], [], 2, 0.0, 0),
                              "CPU MoE: empty expert weight dimension")
    return res


def test_empty_forward():
    """exl3_moe_cpu_forward: no rows is a no-op, no experts per token writes the empty sum (zeros); mismatched shapes
    and layers with no experts or an empty weight dimension are rejected"""
    res = run_isolated(_empty_forward_worker)
    assert all(res.values()), res
