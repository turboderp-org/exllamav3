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


def _started_host():
    host = object.__new__(mch.MoeCpuHost)
    host.started = True
    host.v_hot = np.zeros(1, dtype = np.uint32)
    return host


def test_hot_flags_route_by_component():
    text, mtp = _started_host(), _started_host()
    config = SimpleNamespace(moe_cpu_hosts = {"text": text, "mtp": mtp})
    mch.set_moe_cpu_hot(config, text = True, draft = False)
    assert (text.v_hot[0], mtp.v_hot[0]) == (1, 0)
    mch.set_moe_cpu_hot(config, text = False, draft = True)
    assert (text.v_hot[0], mtp.v_hot[0]) == (0, 1)


def test_hot_flags_skip_hosts_not_started():
    host = object.__new__(mch.MoeCpuHost)
    host.started, host.v_hot = False, None
    mch.set_moe_cpu_hot(SimpleNamespace(moe_cpu_hosts = {"text": host}), text = True, draft = True)
    mch.set_moe_cpu_hot(SimpleNamespace(), text = True, draft = True)      # no CPU MoE at all
