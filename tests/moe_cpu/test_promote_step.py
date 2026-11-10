"""
Promote-on-miss (_split_promote_step) is a per-step hook of the main model's forward. An MTP head shares the main
model's config, and with it the split-module registry whose first entry drives the hook, but it loads first and
drafts several times per step. The step's hit-count snapshot is one device-to-host copy per device, not one per
module. Exercised with stand-in modules carrying the state the step reads.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exllamav3.model.config import InferParams
from exllamav3.modules import block_sparse_mlp_cpu as bsc
from exllamav3.modules.block_sparse_mlp_cpu import BlockSparseMLP_CPU


def _split_module(ip, device = "cpu", num_experts = 8, first = 4):
    with torch.inference_mode():
        hist = torch.rand(num_experts, device = device)
    return SimpleNamespace(
        config = SimpleNamespace(infer_params = ip),
        cpu_component = getattr(ip, "moe_cpu_component", "text"),
        device = device,
        num_experts = num_experts,
        cpu_split_first = first,
        cpu_host = SimpleNamespace(stream_min_rows = 32),
        _split_spares = 2,
        _split_pending = [],
        _split_free_cpu = [],
        _split_mp = np.arange(num_experts),
        _split_hist = hist,
        _swap_tick_count = 0,
        _split_step_count = 0,
        _split_submit_rows = 1,
    )


@pytest.mark.nogpu
def test_main_model_drives_promote_step(monkeypatch):
    calls = []
    monkeypatch.setattr(bsc, "_split_promote_step", lambda ip, reg, rows: calls.append(rows))
    ip = InferParams()
    ip.moe_cpu_component = "mtp"
    mtp = _split_module(ip)
    bsc.register_split_module(ip, mtp)                      # the MTP head loads before the main model
    ip.moe_cpu_component = "text"
    layers = [_split_module(ip) for _ in range(3)]
    for m in layers:
        bsc.register_split_module(ip, m)
    assert ip.moe_cpu_swap_modules == layers + [mtp]

    for _ in range(3):                                      # draft forwards
        BlockSparseMLP_CPU._split_swap_tick(mtp)
    assert calls == []
    BlockSparseMLP_CPU._split_swap_tick(layers[0])          # verify forward
    assert calls == [1]


def test_promote_snapshot_is_one_copy_per_device(device):
    ip = SimpleNamespace()
    modules = [_split_module(ip, device) for _ in range(6)]
    with torch.inference_mode():
        bsc._split_promote_step(ip, modules, 1)
        before = [m._split_hist.clone() for m in modules]
        with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU]) as prof:
            bsc._split_promote_step(ip, modules, 1)
        torch.cuda.synchronize(device)

    assert sum(e.name == "aten::copy_" for e in prof.events()) == 1
    ps = ip.moe_cpu_promote_state
    for i, m in enumerate(modules):
        assert torch.equal(ps.snap[i, :m.num_experts], before[i].cpu())
        assert torch.allclose(m._split_hist, before[i] * bsc._PROMOTE_DECAY)
