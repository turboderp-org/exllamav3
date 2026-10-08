"""
util/device_copy.to_device: cross-device moves are probed once per device pair (both directions from one probe),
the verdict sticks, a failing pair bounces through host memory, host<->device moves never probe, and
EXLLAMA_NO_P2P_COPY overrides probing in both directions. Data is checked against the source tensor on the host;
P2P failure is simulated by patching the probe.
"""

import os
import subprocess
import sys

import pytest
import torch

from exllamav3.util import device_copy as dc
from exllamav3.util.device_copy import to_device


@pytest.fixture
def pair(devices, monkeypatch):
    """Two devices with fresh verdicts, counters and autodetect mode; the module state is restored afterwards"""
    saved_verdicts, saved_stats = dict(dc._verdicts), dict(dc.stats)
    monkeypatch.setattr(dc, "FORCED", None)
    dc.reset_verdicts()
    for k in dc.stats:
        dc.stats[k] = 0
    yield devices[0], devices[1]
    dc.reset_verdicts()
    dc._verdicts.update(saved_verdicts)
    dc.stats.update(saved_stats)


def keys(d0, d1):
    return (d0.index, d1.index), (d1.index, d0.index)


def reset(monkeypatch, forced = None):
    dc.reset_verdicts()
    monkeypatch.setattr(dc, "FORCED", forced)
    for k in dc.stats:
        dc.stats[k] = 0


@pytest.mark.multi_gpu(2)
def test_real_probe(pair, monkeypatch):
    # Data survives, exactly one probe for the pair, both directions cached
    d0, d1 = pair
    fwd, bwd = keys(d0, d1)
    a = torch.randn(1000, device = d0)
    b = to_device(a, d1)
    assert b.device == d1 and torch.equal(a.cpu(), b.cpu())
    c = to_device(b, d0)
    assert torch.equal(a, c)
    assert dc.stats["probes"] == 1, dc.stats
    assert set(dc._verdicts) == {fwd, bwd}
    real_verdict = dict(dc._verdicts)

    # Host<->device moves never touch the probe machinery
    h = to_device(a, "cpu")
    assert h.device.type == "cpu"
    p = torch.empty(16, pin_memory = True)
    g = to_device(p, d1, non_blocking = True)
    assert g.device == d1
    assert dc.stats["probes"] == 1

    # Same-device is a no-op returning the same object, and so is a None destination (TP-side modules have no
    # local device; seen with MTP + TP)
    assert to_device(a, d0) is a
    assert to_device(a, None) is a
    assert dc.stats["probes"] == 1

    # Probing again gives the verdict recorded by the first move
    reset(monkeypatch)
    fwd_ok, bwd_ok = dc._probe_direct(d0, d1)
    assert (not fwd_ok) == real_verdict[fwd]
    assert (not bwd_ok) == real_verdict[bwd]


@pytest.mark.multi_gpu(2)
def test_one_way_broken_fabric(pair, monkeypatch):
    # Only the failing direction bounces, the other stays direct, and the verdict sticks
    d0, d1 = pair
    fwd, bwd = keys(d0, d1)
    monkeypatch.setattr(dc, "_probe_direct", lambda src, dst: (False, True))      # src->dst corrupt, dst->src fine
    x = torch.randn(4096, device = d0)
    y = to_device(x, d1)
    assert torch.equal(x.cpu(), y.cpu())
    assert dc._verdicts == {fwd: True, bwd: False}, dc._verdicts
    assert dc.stats == {"probes": 1, "bounced": 1, "direct": 0}, dc.stats
    z = to_device(y, d0)
    assert torch.equal(x, z)
    assert dc.stats == {"probes": 1, "bounced": 1, "direct": 1}, dc.stats
    for _ in range(5):
        to_device(x, d1)
        to_device(y, d0)
    assert dc.stats == {"probes": 1, "bounced": 6, "direct": 6}, dc.stats


@pytest.mark.multi_gpu(2)
def test_probe_from_reverse_side_stores_both(pair, monkeypatch):
    d0, d1 = pair
    fwd, bwd = keys(d0, d1)
    monkeypatch.setattr(dc, "_probe_direct", lambda src, dst: (True, False))      # d1->d0 fine, d0->d1 corrupt
    y = torch.randn(4096, device = d1)
    z = to_device(y, d0)
    assert torch.equal(y.cpu(), z.cpu())
    assert dc._verdicts == {bwd: False, fwd: True}, dc._verdicts


@pytest.mark.multi_gpu(2)
@pytest.mark.parametrize("forced, want", ((True, {"probes": 0, "bounced": 2, "direct": 0}),
                                          (False, {"probes": 0, "bounced": 0, "direct": 2})))
def test_forced_modes_skip_probing(pair, monkeypatch, forced, want):
    d0, d1 = pair
    reset(monkeypatch, forced)
    x = torch.randn(4096, device = d0)
    y = torch.randn(4096, device = d1)
    assert torch.equal(to_device(x, d1).cpu(), x.cpu())
    assert torch.equal(to_device(y, d0).cpu(), y.cpu())
    assert dc.stats == want, dc.stats


@pytest.mark.nogpu
@pytest.mark.parametrize("value, want", ((None, "None"), ("1", "True"), ("yes", "True"), ("0", "False")))
def test_env_override_parsed_at_import(value, want):
    e = dict(os.environ)
    e.pop("EXLLAMA_NO_P2P_COPY", None)
    if value is not None:
        e["EXLLAMA_NO_P2P_COPY"] = value
    out = subprocess.run([sys.executable, "-c", "from exllamav3.util import device_copy as d; print(d.FORCED)"],
                         env = e, capture_output = True, text = True)
    assert out.stdout.strip() == want, (value, out.stdout, out.stderr[-500:])
