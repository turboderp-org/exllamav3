"""
ext.count_inf_nan (quant/util.cu), the Hessian-capture activation check (LinearEXL3.capture_H): adds the number of
+-inf values of x to y[0] and the number of NaNs to y[1] (y int64 (2,), accumulated across calls), for fp16 and fp32
x of any size; other dtypes are rejected. Exact, against torch.isinf / torch.isnan.

Edge cases: sizes around the 32768-element block, an element count past 2^31 (64-bit indexing), and an empty x,
which must be a no-op that leaves no pending CUDA error behind for later, unrelated launches to report.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.isolated import run_isolated


def _plant(x: torch.Tensor, n_inf: int, n_nan: int, gen: torch.Generator):
    pos = torch.randperm(x.numel(), generator = gen)[: n_inf + n_nan].to(x.device)
    flat = x.view(-1)
    signs = torch.where(torch.rand(n_inf, generator = gen) < 0.5, -1.0, 1.0).to(x.device, x.dtype)
    flat[pos[:n_inf]] = signs * float("inf")
    flat[pos[n_inf:]] = float("nan")


@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("numel", [1, 1000, 32767, 32768, 32769, (1 << 20) + 7])
@torch.inference_mode()
def test_counts(device, dtype, numel):
    gen = torch.Generator().manual_seed(numel)
    x = torch.randn(numel, generator = gen).to(device, dtype)
    n_inf, n_nan = min(numel // 3, 50), min(numel // 3, 70)
    _plant(x, n_inf, n_nan, gen)
    y = torch.tensor([5, 7], dtype = torch.long, device = device)
    ext.count_inf_nan(x, y)
    assert y.tolist() == [5 + torch.isinf(x).sum().item(), 7 + torch.isnan(x).sum().item()]
    assert y.tolist() == [5 + n_inf, 7 + n_nan]
    # Accumulates
    ext.count_inf_nan(x, y)
    assert y.tolist() == [5 + 2 * n_inf, 7 + 2 * n_nan]


@torch.inference_mode()
def test_fp16_overflow_values(device):
    """Saturated but finite fp16 (+-65504) is not inf"""
    x = torch.tensor([65504.0, -65504.0, float("inf"), -float("inf"), float("nan"), 0.0], dtype = torch.half,
                     device = device)
    y = torch.zeros(2, dtype = torch.long, device = device)
    ext.count_inf_nan(x, y)
    assert y.tolist() == [2, 1]


@pytest.mark.vram(10)
@torch.inference_mode()
def test_past_int32_elements(device):
    numel = (1 << 31) + 4099
    x = torch.zeros(numel, dtype = torch.half, device = device)
    x[-1] = float("inf")
    x[-2] = float("nan")
    x[(1 << 31) + 1] = -float("inf")
    x[5] = float("nan")
    y = torch.zeros(2, dtype = torch.long, device = device)
    ext.count_inf_nan(x, y)
    assert y.tolist() == [2, 2]


@torch.inference_mode()
def test_rejects_dtypes(device):
    y = torch.zeros(2, dtype = torch.long, device = device)
    with pytest.raises(RuntimeError, match = "Unsupported dtype"):
        ext.count_inf_nan(torch.zeros(4, dtype = torch.bfloat16, device = device), y)
    with pytest.raises(RuntimeError):
        ext.count_inf_nan(torch.zeros(4, dtype = torch.half, device = device), y.int())


def _empty_worker():
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from testlib.env import get_test_device
    device = get_test_device()
    torch.cuda.set_device(device)
    y = torch.tensor([3, 4], dtype = torch.long, device = device)
    ext.count_inf_nan(torch.empty(0, dtype = torch.half, device = device), y)
    out = {"y": y.tolist()}
    # The next, unrelated launch must not inherit an error from the empty call
    try:
        z = torch.ones(4, device = device) + 1
        torch.cuda.synchronize(device)
        out["followup"] = None
    except Exception as e:
        out["followup"] = str(e).splitlines()[0]
    return out


def test_empty_is_noop(device):
    out = run_isolated(_empty_worker)
    assert out["y"] == [3, 4]
    assert out["followup"] is None, f"a later torch op failed after count_inf_nan on an empty tensor: {out['followup']}"
