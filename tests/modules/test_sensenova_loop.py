"""CPU contracts for SenseNova Loop's execution plan and mHC router."""
import copy

import pytest
import torch
import torch.nn.functional as F

from exllamav3.architecture.sensenova_loop import loop_plan
from exllamav3.modules.loop_router import LoopRouter, merge
from testlib.checkpoint import module_config

pytestmark = pytest.mark.nogpu


def config():
    return {
        "text_config": {"num_hidden_layers": 32, "layer_types": ["linear_attention"] * 31 + ["full_attention"]},
        "loop_settings": {"start": 4, "end": 28, "eval_count": 2, "counts": [2]},
        "mhc": {"start": 4, "end": 28, "streams": 2, "iterations": 20, "fixed_identity": False},
    }


def test_plan_and_visits():
    plan = loop_plan(config())
    assert plan == list(range(4)) + list(range(4, 28)) * 2 + list(range(28, 32))
    counts = {}
    slots = []
    for layer in plan:
        visit = counts.get(layer, 0)
        slots.append((layer, visit))
        counts[layer] = visit + 1
    assert len(set(slots)) == 56
    assert all(counts[i] == (2 if 4 <= i < 28 else 1) for i in range(32))


@pytest.mark.parametrize("section,key,value", [
    (None, "loop_settings", None), ("loop_settings", "start", -1),
    ("loop_settings", "end", 33), ("loop_settings", "eval_count", 0),
    ("loop_settings", "counts", [1, 2]), ("loop_settings", "start", True),
    ("text_config", "layer_types", []), ("text_config", "layer_types", ["unknown"] * 32),
    ("text_config", "layer_types", ["linear_attention"] * 32),
    (None, "mhc", None), ("mhc", "streams", 3), ("mhc", "iterations", 0),
    ("mhc", "start", 3), ("mhc", "fixed_identity", 1),
    ("mhc", "unknown", 1), ("mhc", "pre_scale", float("nan")),
    ("mhc", "pre_scale", 0),
])
def test_invalid_plan(section, key, value):
    raw = copy.deepcopy(config())
    (raw[section] if section else raw)[key] = value
    with pytest.raises(ValueError):
        loop_plan(raw)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("streams", [2, 4])
@pytest.mark.parametrize("fixed", [False, True])
def test_router_reference(tmp_path, dtype, streams, fixed):
    torch.manual_seed(981)
    size = 8
    outputs = 2 * streams + (0 if fixed else streams ** 2)
    weight = torch.randn(outputs, streams * size)
    bias = torch.randn(outputs)
    cfg = module_config({"router.weight": weight, "router.bias": bias}, tmp_path)
    cfg.stc.load_method = "python"
    settings = dict(streams = streams, iterations = 20, fixed_identity = fixed, pre_scale = 1.3)
    router = LoopRouter(cfg, "router", settings, size)
    router.load(torch.device("cpu"))
    x = torch.randn(2, 3, streams, size).to(dtype)
    branch, post, mix = router.forward(x, {})
    flat = x.flatten(-2).float()
    logits = F.linear(flat * torch.rsqrt(flat.square().mean(-1, keepdim = True) + 1e-6), weight, bias)
    pre = logits[..., :streams].sigmoid()
    expected_post = 2 * logits[..., streams:2 * streams].sigmoid()
    if fixed:
        expected_mix = torch.eye(streams).expand(*x.shape[:-2], streams, streams)
    else:
        expected_mix = logits[..., 2 * streams:].unflatten(-1, (streams, streams))
        for _ in range(20):
            expected_mix = expected_mix - torch.logsumexp(expected_mix, -1, keepdim = True)
            expected_mix = expected_mix - torch.logsumexp(expected_mix, -2, keepdim = True)
        expected_mix = expected_mix.exp()
    expected_branch = torch.matmul(pre.to(dtype).unsqueeze(-2), x).squeeze(-2) * 1.3
    torch.testing.assert_close(branch, expected_branch, rtol = 0, atol = 0)
    torch.testing.assert_close(post, expected_post, rtol = 0, atol = 0)
    torch.testing.assert_close(mix, expected_mix, rtol = 0, atol = 0)
    expected = torch.matmul(expected_mix.to(dtype), x) + expected_post.to(dtype).unsqueeze(-1) * branch.unsqueeze(-2)
    torch.testing.assert_close(merge(x, branch, post, mix), expected, rtol = 0, atol = 0)
    router.unload()


def test_block_boundaries_and_export():
    from exllamav3.architecture.sensenova_loop import LoopBlock
    from exllamav3.modules.loop_router import route

    class Norm:
        def forward(self, x, params, out_dtype = None):
            return x.to(out_dtype)

    class Branch:
        def forward(self, x, params):
            return x.float() * .25

    class Router:
        def __init__(self):
            self.weight = torch.randn(8, 16)
            self.bias = torch.randn(8)

        def forward(self, x, params):
            return route(x, self.weight, self.bias, 2, 20, False, 1.)

    torch.manual_seed(42)
    block = LoopBlock.__new__(LoopBlock)
    block.loop_settings = dict(start = 4, end = 28, streams = 2)
    block.attn_norm = block.mlp_norm = Norm()
    block.attn = block.mlp = Branch()
    block.loop_attn, block.loop_ffn = Router(), Router()
    x = torch.randn(1, 3, 8)
    for index in (4, 5, 27, 4, 27):
        block.layer_idx = index
        streams = x.unsqueeze(-2).expand(*x.shape[:-1], 2, x.shape[-1]).contiguous() if index == 4 else x.unflatten(-1, (2, -1))
        for router in (block.loop_attn, block.loop_ffn):
            branch, post, mix = router.forward(streams, {})
            streams = merge(streams, branch.half().float() * .25, post, mix)
        expected = streams.mean(-2) if index == 27 else streams.flatten(-2)
        params = {"prefill": True, "export_state_layers": [index]}
        actual = block.forward(x, params)
        torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)
        torch.testing.assert_close(params["export_states"][0], streams.mean(-2).half(), rtol = 0, atol = 0)
        x = actual
