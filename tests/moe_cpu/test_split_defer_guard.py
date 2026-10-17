"""
BlockSparseMLP_CPU.can_defer_load must refuse deferred loading whenever an expert placement permutation is in
effect (an EXL3_MOE_CPU_SPLIT_STATS file, or a saved expert frequency profile for the model), regardless of
whether the split came from the -mcs / --moe_cpu_split CLI flag (infer_params.moe_cpu_split) or the
EXL3_MOE_CPU_SPLIT env (which only seeds that field). Regression: the guard read the env directly, so with -mcs
the deferred router fill landed in the pre-permutation tensor and the model emitted garbage; the saved profile
later reintroduced the same failure through the second source. Checked on the real class, built without weights.
"""

import os
from types import SimpleNamespace

import pytest

from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
import exllamav3.modules.block_sparse_mlp_cpu as bsm_cpu
from exllamav3.modules.block_sparse_mlp_cpu import _expert_stats_path

pytestmark = pytest.mark.nogpu


def _stub(split):
    # BlockSparseMLP_CPU is a mixin ahead of Module in BlockSparseMLP's MRO; build the real class without weights
    # so super().can_defer_load() resolves to Module's
    m = BlockSparseMLP.__new__(BlockSparseMLP)
    m.modules = []  # Module.can_defer_load() -> True for an empty module list
    m.config = SimpleNamespace(infer_params = SimpleNamespace(moe_cpu_split = split), directory = "/nonexistent/model")
    m.num_experts = 32
    return m


@pytest.mark.parametrize("split, stats, expect", [
    (250, "/nonexistent/stats.json", False),   # -mcs + stats file: must not defer
    (0, "/nonexistent/stats.json", True),      # stats file alone: no split, defer is fine
    (250, None, True),                         # split without stats: dynamic placement, defer is fine
])
def test_can_defer_load_reads_infer_params(monkeypatch, tmp_path, split, stats, expect):
    monkeypatch.setattr(bsm_cpu, "EXPERT_STATS_DIR", str(tmp_path))   # no saved profile for the stub model
    monkeypatch.delenv("EXL3_MOE_CPU_SPLIT", raising = False)
    if stats is None:
        monkeypatch.delenv("EXL3_MOE_CPU_SPLIT_STATS", raising = False)
    else:
        monkeypatch.setenv("EXL3_MOE_CPU_SPLIT_STATS", stats)
    assert _stub(split).can_defer_load() is expect


def test_can_defer_load_sees_saved_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(bsm_cpu, "EXPERT_STATS_DIR", str(tmp_path))
    monkeypatch.delenv("EXL3_MOE_CPU_SPLIT", raising = False)
    monkeypatch.delenv("EXL3_MOE_CPU_SPLIT_STATS", raising = False)
    m = _stub(250)
    assert m.can_defer_load() is True
    path = _expert_stats_path(m.config, m.num_experts)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    open(path, "w").write("{}")
    assert m.can_defer_load() is False
    assert _stub(0).can_defer_load() is True
