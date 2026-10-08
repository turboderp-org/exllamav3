"""
Exact contracts of two small counting helpers:

- histogram(input half/float, any shape, contiguous; output (num_bins <= 1024,) int64; min, max, exclusive):
  output[b] = number of elements with bin b, bin = clamp(trunc((x - min) / (max - min) * num_bins), 0,
  num_bins - 1); exclusive skips x < min and x > max (x == max lands in the last bin), inclusive clamps them into
  the end bins. Exactly num_bins outputs are written. Reference: NumPy bincount over float64 bin positions, on
  inputs kept at least 1e-3 bins away from every bin edge so the fp32 (fast-math) division cannot move an element
  across one. Used by eval/prequant_test.py only.
- count_match_tensor(a, b, max_a) (host, int64): the length of the common prefix of a.flatten() and b's row,
  stopping at min(max_a, b.size(1)). Reference: a Python loop. Used by the generator's partial-page reuse
  (job.py) with a page's token row and a narrowed view of the sequence ids.
"""

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


def _edge_safe(x: np.ndarray, lo: float, hi: float, bins: int) -> np.ndarray:
    pos = (x.astype(np.float64) - lo) / (hi - lo) * bins
    frac = pos - np.floor(pos)
    return x[(frac > 1e-3) & (frac < 1 - 1e-3)]


def _hist_ref(x: np.ndarray, lo: float, hi: float, bins: int, exclusive: bool) -> np.ndarray:
    x = x.astype(np.float64)
    if exclusive:
        x = x[(x >= lo) & (x <= hi)]
    b = np.clip(np.trunc((x - lo) / (hi - lo) * bins), 0, bins - 1).astype(np.int64)
    return np.bincount(b, minlength = bins)


@pytest.mark.parametrize("exclusive", [True, False])
@pytest.mark.parametrize("dtype", [torch.float, torch.half])
@pytest.mark.parametrize("bins, lo, hi", [(1, -1.0, 1.0), (7, -3.0, 2.5), (256, 0.0, 4.0), (1000, -0.5, 0.5), (1024, -6.0, 6.0)])
@pytest.mark.parametrize("shape", [(1,), (999,), (64, 129), (3, 1024, 1365)])
@torch.inference_mode()
def test_histogram(shape, bins, lo, hi, dtype, exclusive, device):
    g_ = torch.Generator().manual_seed(bins + len(shape))
    x = (torch.randn(int(np.prod(shape)), generator = g_) * (hi - lo) * 0.4 + (lo + hi) / 2).to(dtype)
    xs = _edge_safe(x.float().numpy(), lo, hi, bins)
    # Exact range ends: min lands in bin 0, max in the last bin, in both modes
    xs = np.concatenate([xs, np.array([lo, hi], dtype = np.float32)])
    xt = torch.from_numpy(xs.astype(np.float32)).to(dtype)
    if len(shape) > 1 and xt.numel() % shape[-1] == 0:
        xt = xt.view(-1, shape[-1])
    ref = _hist_ref(xt.float().numpy(), lo, hi, bins, exclusive)
    out_full = torch.full((bins + 16,), -7, dtype = torch.long, device = device)
    ext.histogram(xt.to(device), out_full[:bins], lo, hi, exclusive)
    out = out_full.cpu().numpy()
    assert np.array_equal(out[:bins], ref), f"max bin diff {np.abs(out[:bins] - ref).max()}"
    assert (out[bins:] == -7).all(), "wrote past num_bins"


@torch.inference_mode()
def test_histogram_multidim_input_and_outliers(device):
    x = torch.tensor([[-100.0, -1.0, 0.25], [0.75, 1.0, 100.0]], device = device)
    out = torch.empty(4, dtype = torch.long, device = device)
    ext.histogram(x, out, -1.0, 1.0, True)
    assert out.tolist() == [1, 0, 1, 2]       # -1 -> 0, 0.25 -> 2, 0.75 -> 3, 1.0 -> 3; outliers skipped
    ext.histogram(x, out, -1.0, 1.0, False)
    assert out.tolist() == [2, 0, 1, 3]       # outliers clamped into the end bins


@torch.inference_mode()
def test_histogram_rejects(device):
    x = torch.randn(10, device = device)
    with pytest.raises(RuntimeError):
        ext.histogram(x, torch.empty(1025, dtype = torch.long, device = device), 0.0, 1.0, True)
    with pytest.raises(RuntimeError):
        ext.histogram(x, torch.empty(8, dtype = torch.int, device = device), 0.0, 1.0, True)
    with pytest.raises(RuntimeError):
        ext.histogram(x.bfloat16(), torch.empty(8, dtype = torch.long, device = device), 0.0, 1.0, True)


# count_match_tensor

def _count_ref(a: torch.Tensor, b: torch.Tensor, max_a: int) -> int:
    fa, fb = a.flatten().tolist(), b[0].tolist()
    n = 0
    while n < min(max_a, len(fb)) and fa[n] == fb[n]:
        n += 1
    return n


@pytest.mark.nogpu
def test_count_match_tensor():
    rng = np.random.default_rng(0)
    page_size = 256
    for case in range(400):
        page = torch.from_numpy(rng.integers(0, 6, (1, page_size))).long()
        seq = torch.from_numpy(rng.integers(0, 6, (1, 3 * page_size))).long()
        start = int(rng.integers(0, 2 * page_size))
        common = int(rng.integers(0, page_size + 1))
        seq[0, start: start + common] = page[0, :common]
        end = int(rng.integers(start, 3 * page_size + 1))
        prefill_ids = seq.narrow(1, start, end - start)          # like seq.sequence_ids.torch_slice(...)
        max_a = int(rng.integers(-1, page_size + 1))
        assert ext.count_match_tensor(page, prefill_ids, max_a) == _count_ref(page, prefill_ids, max_a), case


@pytest.mark.nogpu
def test_count_match_tensor_edges():
    a = torch.arange(10, dtype = torch.long).view(1, 10)
    assert ext.count_match_tensor(a, a.clone(), 10) == 10
    assert ext.count_match_tensor(a, a.clone(), 4) == 4
    assert ext.count_match_tensor(a, a[:, :3].clone(), 10) == 3
    assert ext.count_match_tensor(a, torch.empty(1, 0, dtype = torch.long), 10) == 0
    assert ext.count_match_tensor(a, a.clone(), 0) == 0
    b = a.clone()
    b[0, 0] = 99
    assert ext.count_match_tensor(a, b, 10) == 0
    b = a.clone()
    b[0, 9] = 99
    assert ext.count_match_tensor(a, b, 10) == 9
    # Full 64-bit comparison
    a2 = torch.tensor([[1 << 40, 2]], dtype = torch.long)
    b2 = torch.tensor([[(1 << 40) + (1 << 33), 2]], dtype = torch.long)
    assert ext.count_match_tensor(a2, b2, 2) == 0
