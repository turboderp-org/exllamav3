"""
Tests for DFlash2 tap ordering.

export_states arrive from the target forward in ascending target-layer order
(order contract pinned at TransformerBlock's export site). The checkpoint's
fc projection was trained against the listed config.target_layer_ids order,
so DFlash2Model permutes ascending taps into listed order before the v1
projection, via the pure _tap_permutation_for mapping. No model weights or
GPU needed.
"""
from types import SimpleNamespace

import pytest
import torch

from exllamav3.architecture.dflash import DFlashModel
from exllamav3.architecture.dflash2 import DFlash2Model, _tap_permutation_for


def test_permutation_unsorted_ids():
    # listed [20, 6, 34]; ascending taps are [tap6, tap20, tap34], so position
    # 0 (layer 20) takes taps[1], position 1 (layer 6) takes taps[0]
    assert _tap_permutation_for([20, 6, 34]) == [1, 0, 2]


def test_permutation_non_monotonic_ids():
    # reversed listing must invert the tap order, not just shift it
    assert _tap_permutation_for([34, 20, 6]) == [2, 1, 0]
    assert _tap_permutation_for([34, 6, 20]) == [2, 0, 1]
    # four taps, scrambled: position i takes the tap of layer ids[i]
    assert _tap_permutation_for([48, 6, 34, 20]) == [3, 0, 2, 1]


def test_permutation_ascending_is_none():
    assert _tap_permutation_for([6, 20, 34, 48, 62]) is None


def test_permutation_single_id_is_none():
    assert _tap_permutation_for([3]) is None


def _capture_projection(monkeypatch):
    seen = {}

    def _stub(self, target_hidden, cache, params, lengths = None):
        seen["taps"] = target_hidden
        return None

    monkeypatch.setattr(DFlashModel, "update_kv_from_target", _stub)
    return seen


def _model_with(ids, permutation):
    m = DFlash2Model.__new__(DFlash2Model)
    m.config = SimpleNamespace(target_layer_ids = ids)
    m._tap_permutation = permutation
    return m


def test_update_applies_stored_permutation(monkeypatch):
    seen = _capture_projection(monkeypatch)
    m = _model_with([20, 6, 34], _tap_permutation_for([20, 6, 34]))
    taps = [torch.tensor(float(i)) for i in range(3)]
    m.update_kv_from_target(taps, None, {}, None)
    assert [t.item() for t in seen["taps"]] == [1.0, 0.0, 2.0]


def test_update_passes_through_when_none(monkeypatch):
    seen = _capture_projection(monkeypatch)
    m = _model_with([6, 20, 34, 48, 62], None)
    taps = [torch.tensor(float(i)) for i in range(5)]
    m.update_kv_from_target(taps, None, {}, None)
    assert seen["taps"] is taps


def test_tap_count_mismatch_fails(monkeypatch):
    seen = _capture_projection(monkeypatch)
    m = _model_with([6, 20, 34], None)
    taps = [torch.zeros(1, 1, 4) for _ in range(2)]
    with pytest.raises(AssertionError, match = "2.*3|taps for 3"):
        m.update_kv_from_target(taps, None, {}, None)
    assert "taps" not in seen


def test_attach_rejects_tp_target():
    m = DFlash2Model.__new__(DFlash2Model)
    m.attached_model = None
    target = SimpleNamespace(loaded_tp = True)
    with pytest.raises(NotImplementedError, match = "tensor-parallel targets"):
        m.attach_to(target)
