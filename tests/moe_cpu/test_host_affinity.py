"""
CPU expert-offload host affinity (exllamav3/model/moe_cpu_affinity.py): the native pool pins
worker threads to one LP per physical core; the host process must be kept off those LPs or a
preempted worker stalls every per-phase barrier. These tests cover the topology binding, the
pure placement planner (against hand-worked topologies), and the OS round-trip (against the affinity the OS
reports). No GPU or model needed.
"""

import json
import multiprocessing
import os
import subprocess
import sys

import pytest

import exllamav3.model.moe_cpu_host as mch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model.moe_cpu_affinity import (apply_process_affinity, default_worker_threads, plan_host_cpus,
                                              process_cpus)

pytestmark = pytest.mark.nogpu


def test_core_order_binding_shape():
    # The binding reports the pool's topology only while pinning is on (EXL3_MOE_CPU_PIN, default off on
    # Linux), so the shape is checked in a process that enables it; this process just has to agree
    # with its own setting
    order, n_phys = ext.exl3_moe_cpu_core_order()
    pin = os.environ.get("EXL3_MOE_CPU_PIN", "0" if sys.platform != "win32" else "1") == "1"
    assert bool(order) == pin and (n_phys > 0) == pin
    out = subprocess.run(
        [sys.executable, "-c", "import json, sys; from exllamav3.ext import exllamav3_ext as ext; "
                               "o, n = ext.exl3_moe_cpu_core_order(); print(json.dumps([list(o), n]))"],
        env = os.environ | {"EXL3_MOE_CPU_PIN": "1"},
        capture_output = True, text = True, check = True)
    order, n_phys = json.loads(out.stdout.strip().splitlines()[-1])
    ncpu = os.cpu_count() or 1
    assert 1 <= n_phys <= len(order) <= ncpu
    assert len(set(order)) == len(order)
    # Windows entries encode (group << 16) | bit-within-group
    if sys.platform == "win32":
        assert all(0 <= e & 0xFFFF < 64 for e in order)


# 4 physical cores, SMT: order = one LP per core, then the siblings (adjacent-pair layout)
ORDER_SMT = [0, 2, 4, 6, 1, 3, 5, 7]
ORDER_NOSMT = [0, 1, 2, 3]


def test_default_threads_reserves_host_cores():
    assert default_worker_threads(n_physical = 12, host_cores = 1) == 11
    assert default_worker_threads(n_physical = 12, host_cores = 0) == 12
    assert default_worker_threads(n_physical = 1, host_cores = 1) == 1   # never below one worker


def test_plan_reserves_last_core_with_both_siblings():
    # 3 workers on cores 0..2 -> host gets core 3 = LPs 6 and 7
    assert plan_host_cpus(ORDER_SMT, 4, threads = 3, host_cores = 1) == [6, 7]


def test_plan_reserves_two_cores():
    assert plan_host_cpus(ORDER_SMT, 4, threads = 2, host_cores = 2) == [4, 6, 5, 7]


def test_plan_more_host_cores_than_free_takes_all_free():
    assert plan_host_cpus(ORDER_SMT, 4, threads = 3, host_cores = 2) == [6, 7]


def test_plan_falls_back_to_free_siblings_when_workers_cover_every_core():
    assert plan_host_cpus(ORDER_SMT, 4, threads = 4, host_cores = 1) == [1, 3, 5, 7]
    # workers spill onto siblings 1 and 3: only the untouched siblings remain
    assert plan_host_cpus(ORDER_SMT, 4, threads = 6, host_cores = 1) == [5, 7]


def test_plan_none_when_nothing_is_free():
    assert plan_host_cpus(ORDER_SMT, 4, threads = 8, host_cores = 1) is None
    assert plan_host_cpus(ORDER_SMT, 4, threads = 9, host_cores = 1) is None
    assert plan_host_cpus(ORDER_NOSMT, 4, threads = 4, host_cores = 1) is None
    assert plan_host_cpus(ORDER_SMT, 4, threads = 3, host_cores = 0) is None
    assert plan_host_cpus([], 0, threads = 3, host_cores = 1) is None


def test_plan_no_smt_reserves_plain_core():
    assert plan_host_cpus(ORDER_NOSMT, 4, threads = 3, host_cores = 1) == [3]


def test_plan_rejects_multi_group_windows_encoding():
    # (group 1 << 16) | bit entries cannot be expressed in a single-group process mask
    assert plan_host_cpus([0, 2, (1 << 16) | 0, (1 << 16) | 2, 1, 3], 4, threads = 3, host_cores = 1) is None


def _current_process_cpus() -> set[int]:
    if sys.platform == "win32":
        import ctypes
        k = ctypes.WinDLL("kernel32", use_last_error = True)
        k.GetCurrentProcess.restype = ctypes.c_void_p
        proc_mask, sys_mask = ctypes.c_size_t(), ctypes.c_size_t()
        assert k.GetProcessAffinityMask(ctypes.c_void_p(k.GetCurrentProcess()), ctypes.byref(proc_mask), ctypes.byref(sys_mask))
        return {i for i in range(64) if proc_mask.value >> i & 1}
    return set(os.sched_getaffinity(0))


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason = "needs two LPs")
def test_apply_process_affinity_round_trip():
    before = _current_process_cpus()
    target = sorted(before)[-1:]          # last LP only
    try:
        assert apply_process_affinity(target) is None
        assert _current_process_cpus() == set(target)
    finally:
        assert apply_process_affinity(sorted(before)) is None
    assert _current_process_cpus() == before


def test_process_cpus_matches_os():
    assert set(process_cpus()) == _current_process_cpus()


def _widen_in_child(cpus, queue):
    # Spawned child of a confined parent: restore the parent's pre-pin mask, report the result
    from exllamav3.model.moe_cpu_affinity import apply_process_affinity, process_cpus
    queue.put((apply_process_affinity(cpus), sorted(process_cpus())))


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason = "needs two LPs")
def test_worker_spawned_after_host_pin_can_restore_full_mask():
    before = sorted(_current_process_cpus())
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    try:
        assert apply_process_affinity(before[-1:]) is None
        proc = ctx.Process(target = _widen_in_child, args = (before, queue))
        proc.start()
    finally:
        assert apply_process_affinity(before) is None
    err, child_cpus = queue.get(timeout = 120)
    proc.join(timeout = 60)
    assert err is None
    assert child_cpus == before


@pytest.fixture
def fresh_host_affinity(monkeypatch):
    calls = []
    monkeypatch.setattr(mch, "apply_process_affinity", lambda cpus: calls.append(list(cpus)) or None)
    monkeypatch.setattr(mch, "process_cpus", lambda: list(range(8)))
    monkeypatch.setattr(mch, "_HOST_AFFINITY_THREADS", None)
    monkeypatch.setattr(mch, "_HOST_ORIG_CPUS", None)
    monkeypatch.setattr(mch.ext, "exl3_moe_cpu_core_order", lambda: (ORDER_SMT, 4))
    return calls


def test_tuning_defaults_threads_to_physical_minus_host(monkeypatch):
    monkeypatch.delenv("EXL3_MOE_CPU_THREADS", raising = False)
    monkeypatch.setenv("EXL3_MOE_HOST_CORES", "1")
    tuning = mch.MoeCpuTuning()
    _, n_phys = ext.exl3_moe_cpu_core_order()
    if n_phys:
        assert tuning.threads == max(1, n_phys - 1)
    else:   # pinning disabled or topology unreadable: legacy default
        assert tuning.threads == max(1, (os.cpu_count() or 2) // 2)
    assert tuning.host_cores == 1


def test_tuning_host_cores_zero_keeps_legacy_default(monkeypatch):
    monkeypatch.delenv("EXL3_MOE_CPU_THREADS", raising = False)
    monkeypatch.setenv("EXL3_MOE_HOST_CORES", "0")
    assert mch.MoeCpuTuning().threads == max(1, (os.cpu_count() or 2) // 2)


def test_tuning_explicit_threads_wins(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_CPU_THREADS", "3")
    assert mch.MoeCpuTuning().threads == 3


def test_host_affinity_applied_once_per_process(fresh_host_affinity, capsys):
    mch._apply_host_affinity(threads = 3, host_cores = 1)
    mch._apply_host_affinity(threads = 3, host_cores = 1)      # MTP head / draft / reload
    mch._apply_host_affinity(threads = 4, host_cores = 1)      # wider later host: notice only
    assert fresh_host_affinity == [[6, 7]]
    assert mch._HOST_ORIG_CPUS == list(range(8))
    out = capsys.readouterr().out
    assert "host on LPs [6, 7]" in out
    assert "planned for 3 workers" in out


def test_host_affinity_disabled_does_not_pin(fresh_host_affinity):
    mch._apply_host_affinity(threads = 3, host_cores = 0)
    assert fresh_host_affinity == []
    assert mch._HOST_ORIG_CPUS is None


def test_host_affinity_os_failure_leaves_host_unpinned(fresh_host_affinity, monkeypatch, capsys):
    monkeypatch.setattr(mch, "apply_process_affinity", lambda cpus: "denied")
    mch._apply_host_affinity(threads = 3, host_cores = 1)
    assert mch._HOST_ORIG_CPUS is None
    assert "!! CPU MoE host affinity: denied; host threads left unpinned" in capsys.readouterr().out


def test_spawn_hands_pre_pin_mask_to_worker(monkeypatch):
    captured = {}

    class FakeConn:
        def close(self): pass

    class FakeProcess:
        def __init__(self, target, args, daemon):
            captured["target"], captured["args"] = target, args
        def start(self): pass

    class FakeCtx:
        Process = FakeProcess
        @staticmethod
        def Pipe(duplex): return FakeConn(), FakeConn()

    monkeypatch.setattr(mch.multiprocessing, "get_context", lambda method: FakeCtx())
    monkeypatch.setattr(mch.cleanupper, "register_atexit", lambda fn: None)
    monkeypatch.setattr(mch, "_HOST_ORIG_CPUS", [0, 1, 2, 3])
    host = object.__new__(mch.MoeCpuHost)
    host.proc, host.model_dir, host.threads, host.stage_threads, host.pinned = None, "m", 3, 1, False
    host._spawn()
    assert captured["target"] is mch._moe_cpu_child_main
    assert captured["args"][-1] == [0, 1, 2, 3]
