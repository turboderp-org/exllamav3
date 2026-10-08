"""
Issue #329: run_pending_swap_sweeps() mutates inference tensors (placement map, hit histogram, arena slots) in
place. The generator reaches it from cancel()/clear_queue() outside any forward, i.e. outside inference mode,
where such updates raise. The sweep and the generator's queue-drained hook must therefore establish inference
mode themselves. Exercised with a stand-in module whose sweep does the same kind of update the real one does, and
with a stub generator whose defrag records the mode it runs in.
"""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.generator.generator import Generator
from exllamav3.modules.block_sparse_mlp_cpu import run_pending_swap_sweeps

pytestmark = pytest.mark.nogpu


class FakeSplitModule:
    def __init__(self):
        with torch.inference_mode():            # as at load time
            self._split_map = torch.arange(16, dtype = torch.int32)
            self._split_hist = torch.ones(16)
        self.device = None
        self.sweeps = 0

    def _split_sweep_layer_reset(self):
        pass

    def _split_sweep_layer(self, budget):
        # the real sweep's kind of updates: map rewrite, histogram decay
        self._split_map.copy_(self._split_map.flip(0))
        self._split_hist.mul_(0.5)
        self.sweeps += 1
        return 1


def test_sweep_outside_inference_mode():
    assert not torch.is_inference_mode_enabled()
    mod = FakeSplitModule()
    ip = SimpleNamespace(moe_cpu_swap_pending = True, moe_cpu_swap_modules = [mod])
    run_pending_swap_sweeps(ip)                     # outside inference mode, like cancel()
    assert mod.sweeps == 1 and not ip.moe_cpu_swap_pending
    assert torch.equal(mod._split_map, torch.arange(15, -1, -1, dtype = torch.int32))
    assert float(mod._split_hist[0]) == 0.5


def test_queue_drained_hook_enters_inference_mode():
    seen = []

    class Stub:
        recurrent_cache = None
        pagetable = SimpleNamespace(defrag = lambda: seen.append(torch.is_inference_mode_enabled()))
        model = SimpleNamespace(config = SimpleNamespace(infer_params = SimpleNamespace(moe_cpu_swap_pending = False)))

    assert not torch.is_inference_mode_enabled()
    Generator.on_queue_drained(Stub())
    assert seen == [True]
