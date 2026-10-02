import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.ext import exllamav3_ext as ext

# fa75 (flash-attention prefill kernel for head_dim 256 on sm_75 HMMA) against an fp32 reference, for prefill
# chunks appended to a cache (bottom-right causal), with and without GQA

device = "cuda:0"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (7, 5),
    reason = "needs an sm_75+ CUDA device",
)


def _reference(q, k, v, scale, causal):
    Tq, Hq, D = q.shape
    Tkv, Hkv, _ = k.shape
    grp = Hq // Hkv
    qf = q.float().transpose(0, 1)                                   # [Hq, Tq, D]
    kf = k.float().repeat_interleave(grp, dim = 1).transpose(0, 1)   # [Hq, Tkv, D]
    vf = v.float().repeat_interleave(grp, dim = 1).transpose(0, 1)
    s = (qf @ kf.transpose(1, 2)) * scale
    if causal:
        rows = torch.arange(Tq, device = q.device)[:, None] + (Tkv - Tq)
        cols = torch.arange(Tkv, device = q.device)[None, :]
        s = s.masked_fill(cols > rows, float("-inf"))
    return (torch.softmax(s, dim = -1) @ vf).transpose(0, 1)         # [Tq, Hq, D]


@pytest.mark.parametrize("Tq, Tkv", [(256, 256), (300, 1000), (37, 37), (64, 5000), (1000, 1000), (1, 700)])
@pytest.mark.parametrize("Hq, Hkv", [(24, 4), (8, 8)])
@pytest.mark.parametrize("causal", [True, False])
def test_fa75_matches_reference(Tq, Tkv, Hq, Hkv, causal):
    torch.cuda.set_device(device)
    gen = torch.Generator(device = device).manual_seed(Tq * 7 + Tkv)
    q = torch.randn(Tq, Hq, 256, device = device, dtype = torch.half, generator = gen)
    k = torch.randn(Tkv, Hkv, 256, device = device, dtype = torch.half, generator = gen)
    v = torch.randn(Tkv, Hkv, 256, device = device, dtype = torch.half, generator = gen)
    scale = 256 ** -0.5
    o = torch.empty_like(q)
    ext.fa75_fwd(q, k, v, o, scale, causal)
    ref = _reference(q, k, v, scale, causal)
    assert torch.isfinite(o).all()
    err = ((o.float() - ref).abs().max() / ref.abs().max()).item()
    assert err < 2e-3, err


def test_fa75_strided_q():
    # q as a head-slice view of a wider projection output (row stride > Hq * 256)
    torch.cuda.set_device(device)
    gen = torch.Generator(device = device).manual_seed(5)
    qkv = torch.randn(333, 32, 256, device = device, dtype = torch.half, generator = gen)
    q = qkv[:, :24]
    k = torch.randn(1333, 4, 256, device = device, dtype = torch.half, generator = gen)
    v = torch.randn(1333, 4, 256, device = device, dtype = torch.half, generator = gen)
    o = torch.empty(333, 24, 256, device = device, dtype = torch.half)
    ext.fa75_fwd(q, k, v, o, 0.0625, True)
    ref = _reference(q, k, v, 0.0625, True)
    assert ((o.float() - ref).abs().max() / ref.abs().max()).item() < 2e-3
