"""
A CPU-MoE split whose experts are permuted by a placement profile (an EXL3_MOE_CPU_SPLIT_STATS file, or a saved
expert frequency profile for the model) permutes its router tensors right after they load. Regression: a deferred
router fill landed in the pre-permutation tensor and the model emitted garbage. The fix used to be refusing deferred
loading for the whole module, which loaded every GPU-resident expert tensor one direct read at a time (roughly
doubling load time for a large split model); now only the router gate loads eagerly, inside the module's deferred
bracket, and everything else keeps the batched fill. Checked on the real class, built without weights.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
import exllamav3.modules.block_sparse_mlp_cpu as bsm_cpu
from exllamav3.modules.block_sparse_mlp_cpu import _expert_stats_path
from exllamav3.loader.safetensors import SafetensorsCollection, VariantSafetensorsCollection

pytestmark = pytest.mark.nogpu


class _StubSTC:
    """Just the deferred-mode state and the eager_load bracket of SafetensorsCollection"""
    deferred_mode = False
    eager_load = SafetensorsCollection.eager_load


class _Sub:
    """Submodule stand-in recording whether the stc was deferring when it loaded"""
    def __init__(self, stc):
        self.stc = stc
        self.loaded_deferred = None
    def can_defer_load(self):
        return True
    def load(self, device, **kwargs):
        self.loaded_deferred = self.stc.deferred_mode


def _stub(split, stc = None):
    # BlockSparseMLP_CPU is a mixin ahead of Module in BlockSparseMLP's MRO; build the real class without weights
    m = BlockSparseMLP.__new__(BlockSparseMLP)
    m.modules = []  # Module.can_defer_load() -> True for an empty module list
    m.config = SimpleNamespace(infer_params = SimpleNamespace(moe_cpu_split = split), directory = "/nonexistent/model",
                               stc = stc)
    m.num_experts = 32
    m.key = "model.layers.0.mlp"
    m.cpu_split_first = None
    return m


def _stub_load(stc, perm):
    """A module whose load() reaches the submodule loop: split registration and the post-load steps stubbed out"""
    m = _stub(250, stc)
    gate, expert = _Sub(stc), _Sub(stc)
    m.routing_gate = gate
    m.modules = [gate, expert]
    m.e_score_correction_bias_key = m.e_score_bias_vl_key = m.tid2eid_key = m.per_expert_scale_key = None
    m.cpu_maybe_offload_load = lambda device, **kwargs: False
    def split_load(device, **kwargs):
        m.cpu_split_first = 28
        m._split_perm = perm
    m.cpu_maybe_split_load = split_load
    return m, gate, expert


@pytest.mark.parametrize("split, stats", [(250, "/nonexistent/stats.json"), (0, "/nonexistent/stats.json"),
                                          (250, None)])
def test_can_defer_load_with_placement_stats(monkeypatch, tmp_path, split, stats):
    monkeypatch.setattr(bsm_cpu, "EXPERT_STATS_DIR", str(tmp_path))   # no saved profile for the stub model
    monkeypatch.delenv("EXL3_MOE_CPU_SPLIT", raising = False)
    if stats is None:
        monkeypatch.delenv("EXL3_MOE_CPU_SPLIT_STATS", raising = False)
    else:
        monkeypatch.setenv("EXL3_MOE_CPU_SPLIT_STATS", stats)
    assert _stub(split).can_defer_load() is True


def test_can_defer_load_with_saved_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(bsm_cpu, "EXPERT_STATS_DIR", str(tmp_path))
    monkeypatch.delenv("EXL3_MOE_CPU_SPLIT", raising = False)
    monkeypatch.delenv("EXL3_MOE_CPU_SPLIT_STATS", raising = False)
    m = _stub(250)
    path = _expert_stats_path(m.config, m.num_experts)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    open(path, "w").write("{}")
    assert m.can_defer_load() is True


@pytest.mark.parametrize("perm, gate_deferred", [(list(range(32))[::-1], False), (None, True)])
def test_permuted_split_loads_router_gate_eagerly(perm, gate_deferred):
    stc = _StubSTC()
    stc.deferred_mode = True        # inside the loader's deferred bracket
    m, gate, expert = _stub_load(stc, perm)
    m.load(torch.device("cpu"))
    assert gate.loaded_deferred is gate_deferred
    assert expert.loaded_deferred is True
    assert stc.deferred_mode is True


def test_eager_load_restores_mode_on_error():
    stc = _StubSTC()
    stc.deferred_mode = True
    with pytest.raises(RuntimeError):
        with stc.eager_load():
            assert stc.deferred_mode is False
            raise RuntimeError
    assert stc.deferred_mode is True
    stc.deferred_mode = False
    with stc.eager_load():
        assert stc.deferred_mode is False
    assert stc.deferred_mode is False


def test_variant_eager_load_covers_every_collection():
    v = VariantSafetensorsCollection.__new__(VariantSafetensorsCollection)
    v.main = _StubSTC()
    v.stcs = [(None, None, _StubSTC()), (None, None, _StubSTC())]
    every = [s for _, _, s in v.stcs] + [v.main]
    for s in every:
        s.deferred_mode = True
    with v.eager_load():
        assert all(s.deferred_mode is False for s in every)
    assert all(s.deferred_mode is True for s in every)
