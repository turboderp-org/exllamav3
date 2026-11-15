"""
An MTP head's CPU MoE worker runs beside the main model's: its pool pins to the SMT siblings of the main pool's cores
instead of time-slicing the same LPs, and the host keeps each pool polling across the draft/verify handoff (the hot
word) so the next pass does not pay a nap's wake-up. No GPU or model needed.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import exllamav3.model.moe_cpu_host as mch

pytestmark = pytest.mark.nogpu


def _spawn_args(monkeypatch, component):
    captured = {}

    class FakeConn:
        def close(self): pass

    class FakeProcess:
        def __init__(self, target, args, daemon):
            captured["args"] = args
        def start(self): pass

    class FakeCtx:
        Process = FakeProcess
        @staticmethod
        def Pipe(duplex): return FakeConn(), FakeConn()

    monkeypatch.setattr(mch.multiprocessing, "get_context", lambda method: FakeCtx())
    monkeypatch.setattr(mch.cleanupper, "register_atexit", lambda fn: None)
    host = object.__new__(mch.MoeCpuHost)
    host.proc, host.model_dir, host.threads, host.stage_threads, host.pinned = None, "m", 3, 1, False
    host.component = component
    host._spawn()
    return captured["args"]


def test_draft_worker_pins_to_smt_siblings(monkeypatch):
    assert _spawn_args(monkeypatch, "text")[-1] is False
    assert _spawn_args(monkeypatch, "mtp")[-1] is True


def _started_host(threads = 11):
    host = object.__new__(mch.MoeCpuHost)
    host.started, host.threads = True, threads
    host.v_hot = np.zeros(1, dtype = np.uint32)
    return host


def _topology(monkeypatch, n_phys, n_lps):
    monkeypatch.setattr(mch.TUNING, "n_phys", n_phys)
    monkeypatch.setattr(mch.TUNING, "n_lps", n_lps)


def test_hot_flags_route_by_component(monkeypatch):
    _topology(monkeypatch, 12, 24)
    text, mtp = _started_host(), _started_host()
    config = SimpleNamespace(moe_cpu_hosts = {"text": text, "mtp": mtp})
    mch.set_moe_cpu_hot(config, text = True, draft = False)
    assert (text.v_hot[0], mtp.v_hot[0]) == (1, 0)
    mch.set_moe_cpu_hot(config, text = False, draft = True)
    assert (text.v_hot[0], mtp.v_hot[0]) == (0, 1)
    mch.set_moe_cpu_hot(config, text = True, draft = True)
    assert (text.v_hot[0], mtp.v_hot[0]) == (1, 1)


# (physical cores, LPs, main pool threads, draft pool threads, pools disjoint): full SMT, no SMT, hybrid with fewer
# siblings than draft threads, hybrid with a draft pool that fits the siblings, a main pool larger than the cores
@pytest.mark.parametrize("n_phys, n_lps, main, draft, disjoint", [
    (12, 24, 11, 11, True),
    (12, 12, 11, 11, False),
    (24, 32, 23, 23, False),
    (24, 32, 23, 8, True),
    (12, 24, 13, 4, False),
], ids = ["smt", "no_smt", "hybrid_short", "hybrid_fits", "main_past_cores"])
def test_shared_lps_keep_only_the_draft_pool_hot(monkeypatch, n_phys, n_lps, main, draft, disjoint):
    _topology(monkeypatch, n_phys, n_lps)
    text, mtp = _started_host(main), _started_host(draft)
    config = SimpleNamespace(moe_cpu_hosts = {"text": text, "mtp": mtp})
    mch.set_moe_cpu_hot(config, text = True, draft = True)
    assert (text.v_hot[0], mtp.v_hot[0]) == (1 if disjoint else 0, 1)
    mch.set_moe_cpu_hot(config, text = True, draft = False)        # a single hot pool contends with nothing
    assert (text.v_hot[0], mtp.v_hot[0]) == (1, 0)


def test_main_model_alone_stays_hot(monkeypatch):
    _topology(monkeypatch, 12, 12)
    text = _started_host()
    mch.set_moe_cpu_hot(SimpleNamespace(moe_cpu_hosts = {"text": text}), text = True, draft = True)
    assert text.v_hot[0] == 1


def test_hot_flags_skip_hosts_not_started():
    host = object.__new__(mch.MoeCpuHost)
    host.started, host.v_hot = False, None
    mch.set_moe_cpu_hot(SimpleNamespace(moe_cpu_hosts = {"text": host}), text = True, draft = True)
    mch.set_moe_cpu_hot(SimpleNamespace(), text = True, draft = True)      # no CPU MoE at all
