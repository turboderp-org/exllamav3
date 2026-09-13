"""
The pinned->device bandwidth probe behind the streamed-prefill threshold (EXL3_MOE_STREAM_T
unset). An idle PCIe link sits at Gen1 and retrains only after 0.2-0.3 s of sustained traffic,
reading a perfectly stable low rate until then, so the probe must outlast the retrain, wait for
steady state, and not let one copy that straddles the retrain step win.
"""
import os, sys, time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from exllamav3.model.moe_cpu_host import probe_bandwidth

FLOOR, CAP = 0.05, 0.3


def _link(schedule):
    """timed_copy stand-in: rate from `schedule(elapsed_s)`, each copy costing ~1 ms"""
    t0 = time.perf_counter()
    def copy():
        time.sleep(0.001)
        return schedule(time.perf_counter() - t0)
    return copy


def _run(schedule):
    t0 = time.perf_counter()
    bw = probe_bandwidth(_link(schedule), floor_s = FLOOR, cap_s = CAP)
    return bw, time.perf_counter() - t0


def test_awake_link_reads_its_rate_after_the_floor():
    bw, dt = _run(lambda t: 26.7)
    assert bw == pytest.approx(26.7)
    assert FLOOR <= dt < CAP


def test_link_that_wakes_after_the_floor_is_still_read_awake():
    bw, dt = _run(lambda t: 3.4 if t < FLOOR * 2 else 26.7)
    assert bw == pytest.approx(26.7)
    assert dt < CAP


def test_one_copy_straddling_the_retrain_step_does_not_win():
    # a single intermediate reading between two stretches of the sleeping rate
    bw, dt = _run(lambda t: 6.8 if FLOOR * 0.5 < t < FLOOR * 0.5 + 0.002 else 3.4)
    assert bw == pytest.approx(3.4)
    assert dt >= CAP   # sleeping rate: keep pushing traffic up to the cap


def test_genuinely_slow_link_above_the_sleeping_rate_stops_at_the_floor():
    bw, dt = _run(lambda t: 6.7)   # gen4 x4 chipset link
    assert bw == pytest.approx(6.7)
    assert FLOOR <= dt < CAP
