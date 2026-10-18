"""Spreading the pool must preserve distinct cores and the host's SMT correspondence."""
from collections import defaultdict
from pathlib import Path

import pytest

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model.moe_cpu_affinity import plan_host_cpus

pytestmark = [pytest.mark.nogpu, pytest.mark.platform("linux")]


def core(cpu):
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    return tuple(int((root / name).read_text()) for name in ("physical_package_id", "core_id"))


def llc(cpu):
    for index in range(8):
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/cache/index{index}")
        if not (root / "level").exists():
            break
        if int((root / "level").read_text()) == 3:
            return core(cpu)[0], int((root / "id").read_text())
    return None


def test_spread_order_and_smt(monkeypatch):
    monkeypatch.delenv("EXL3_MOE_CPU_SPREAD", raising=False)
    compact, n = ext.exl3_moe_cpu_core_order()
    monkeypatch.setenv("EXL3_MOE_CPU_SPREAD", "0")
    assert ext.exl3_moe_cpu_core_order() == (compact, n)
    monkeypatch.setenv("EXL3_MOE_CPU_SPREAD", "1")
    spread, spread_n = ext.exl3_moe_cpu_core_order()
    assert spread_n == n
    assert sorted(spread) == sorted(compact)
    assert len(set(spread)) == len(spread)
    assert set(spread[:n]) == set(compact[:n])
    domains = defaultdict(list)
    for cpu in compact[:n]:
        domains[llc(cpu)].append(cpu)
    if None in domains or len(domains) <= 1:
        assert spread == compact
        return
    groups = [domains[k] for k in sorted(domains)]
    expected = [group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)]
    assert spread[:n] == expected
    expected_smt = [s for c in expected for s in compact[n:] if core(s) == core(c)]
    assert spread[n:] == expected_smt


def test_spread_host_reservation_keeps_siblings_together(monkeypatch):
    monkeypatch.setenv("EXL3_MOE_CPU_SPREAD", "1")
    order, n = ext.exl3_moe_cpu_core_order()
    if n < 4 or len(order) != 2 * n:
        pytest.skip("requires at least four cores with two-way SMT")
    reserved = plan_host_cpus(order, n, n - 2, 2)
    assert len(reserved) == 4
    assert {core(cpu) for cpu in reserved} == {core(cpu) for cpu in order[n - 2:n]}
    assert not ({core(cpu) for cpu in reserved} & {core(cpu) for cpu in order[:n - 2]})
