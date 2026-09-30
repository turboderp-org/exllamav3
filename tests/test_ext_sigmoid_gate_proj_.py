"""
Reference test for add_sigmoid_gate_proj (z += x * sigmoid(y @ w)), the fused shared-expert gate used for
batches of up to 32 rows. The gate logit is a block-wide sum broadcast to all 1024 threads, so a wrong
broadcast shows up as a wrong gate. The one-hot case puts the whole dot product in thread 0.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext

device = "cuda:0"


def reference(x, y, z, w):
    return z + x * torch.sigmoid(y.float() @ w.float())


@pytest.mark.parametrize("bsz", [1, 4, 32])
@pytest.mark.parametrize("dim", [256, 2048, 4096])
@torch.inference_mode()
def test_add_sigmoid_gate_proj(bsz, dim):
    x = torch.randn(bsz, dim, dtype = torch.float, device = device)
    y = torch.randn(bsz, dim, dtype = torch.half, device = device)
    z = torch.randn(bsz, dim, dtype = torch.float, device = device)
    w = (torch.randn(dim, 1, device = device) / dim ** 0.5).half()
    ref_z = reference(x, y, z, w)
    ext.add_sigmoid_gate_proj(x, y, z, w)
    torch.testing.assert_close(z, ref_z, rtol = 1e-4, atol = 1e-4)


@torch.inference_mode()
def test_add_sigmoid_gate_proj_one_hot():
    bsz, dim = 32, 2048
    x = torch.ones(bsz, dim, dtype = torch.float, device = device)
    y = torch.zeros(bsz, dim, dtype = torch.half, device = device)
    y[:, 0] = 1.0
    z = torch.zeros(bsz, dim, dtype = torch.float, device = device)
    w = torch.zeros(dim, 1, dtype = torch.half, device = device)
    w[0, 0] = 0.5
    for _ in range(20):
        z.zero_()
        ext.add_sigmoid_gate_proj(x, y, z, w)
        torch.testing.assert_close(z, torch.full_like(z, 0.62245935), rtol = 1e-5, atol = 1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
