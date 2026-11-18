"""Both Loop visits contribute to calibration and the quantized replay."""
from types import SimpleNamespace

import pytest
import torch

from exllamav3.conversion.convert_model import capture_module_serial
from exllamav3.conversion.loop_replay import replay_region

pytestmark = pytest.mark.nogpu


class Block:
    def __init__(self, key, factor):
        self.key, self.factor = key, factor

    def prepare_for_device(self, x, params):
        return x

    def forward(self, x, params):
        if "capture" in params:
            h = params["capture"].setdefault("qmap", {"sum": 0., "count": 0})
            h["sum"] += x.square().sum().item()
            h["count"] += x.numel()
        return x * self.factor


def test_both_visits_and_quantized_replay():
    blocks = [Block("layer4", 2), Block("layer5", 3)]
    rows = [torch.tensor([[1.]]), torch.tensor([[2.]])]
    ids = [torch.tensor([[1]]), torch.tensor([[2]])]
    loaded, unloaded = [], []
    load = lambda m: loaded.append(m.key)
    unload = lambda m: unloaded.append(m.key)
    second = replay_region([x.clone() for x in rows], blocks, ids, set(), load, unload, "reference")
    assert [x.item() for x in second] == [6, 12]
    model = SimpleNamespace(calibration_all_experts = False)
    hess, _, _ = capture_module_serial(model, blocks[0], rows + second, ids + ids,
                                        lambda i, p: None, lambda i, p: None, False, 0, None, set())
    assert hess["qmap"] == {"sum": 185., "count": 4}
    quant = [Block("layer4", 1.5), Block("layer5", 2.5)]
    for _ in range(2):
        replay_region(rows, quant, ids, set(), load, unload, "quantized")
    assert [x.item() for x in rows] == [14.0625, 28.125]
    assert loaded == unloaded
