"""
BlockSparseMLP_CPU.can_defer_load must refuse deferred loading whenever a static expert
placement (EXL3_MOE_CPU_SPLIT_STATS) is in effect, regardless of whether the split came from
the -mcs / --moe_cpu_split CLI flag (infer_params.moe_cpu_split) or the EXL3_MOE_CPU_SPLIT env
(which only seeds that field). Regression: the guard read the env directly, so with -mcs the
deferred router fill landed in the pre-permutation tensor and the model emitted garbage.
"""
import os, sys
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP


def _stub(split):
    # BlockSparseMLP_CPU is a mixin ahead of Module in BlockSparseMLP's MRO; build the real
    # class without weights so super().can_defer_load() resolves to Module's
    m = BlockSparseMLP.__new__(BlockSparseMLP)
    m.modules = []  # Module.can_defer_load() -> True for an empty module list
    m.config = SimpleNamespace(infer_params = SimpleNamespace(moe_cpu_split = split))
    return m


@pytest.mark.parametrize("split, stats, expect", [
    (250, "/nonexistent/stats.json", False),   # -mcs + stats file: must not defer
    (0, "/nonexistent/stats.json", True),      # stats file alone: no split, defer is fine
    (250, None, True),                         # split without stats: dynamic placement, defer is fine
])
def test_can_defer_load_reads_infer_params(monkeypatch, split, stats, expect):
    monkeypatch.delenv("EXL3_MOE_CPU_SPLIT", raising = False)
    if stats is None:
        monkeypatch.delenv("EXL3_MOE_CPU_SPLIT_STATS", raising = False)
    else:
        monkeypatch.setenv("EXL3_MOE_CPU_SPLIT_STATS", stats)
    assert _stub(split).can_defer_load() is expect


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
