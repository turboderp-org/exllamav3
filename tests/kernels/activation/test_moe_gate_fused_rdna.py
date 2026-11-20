"""
Reference test for moe_gate_fused_rdna (dst = routed + sigmoid(y @ w) * sh), the fused
shared-expert gate tail of the RDNA block-sparse MLP decode route. The fused kernel replaces
the eager to(Float)/matmul/sigmoid/copy_/addcmul_ chain (and, with the optional stages
skipped, the copy_/add_ pair and the plain copy_): with sh absent the kernel is a copy, with
y/w absent the shared half is added unscaled. Checked against an fp32 torch reference, the
same rounding the kernel applies: fp32 math, one round to the accumulator dtype on store.

The gate logit is a block-wide sum broadcast over the row (one block per row), so a wrong
broadcast shows up as a wrong gate; the one-hot case puts the whole dot product in lane 0.
H sweeps past blockDim (256) in strides, so shapes are covered for H above, below and off
multiples of the block size. bsz runs over several grid.y rows. Only runs on ROCm builds
(the kernel is not compiled for CUDA); reject (TORCH_CHECK): mixed accumulator dtypes,
non-contiguous operands, mismatched shapes, and y without w.

Reference: torch elementwise ops + matmul (fp32), rounded once to the accumulator dtype.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(
    not getattr(ext, "HAS_RDNA_GATE_FUSED", False),
    reason = "moe_gate_fused_rdna is compiled for ROCm (RDNA) builds only",
)

def reference(routed, sh, gate):
    # fp32 math, one round to the accumulator dtype on store (same as the kernel)
    out = routed.float()
    if sh is not None:
        add = sh.float()
        if gate is not None:
            add = add * (1.0 / (1.0 + torch.exp(-gate))).unsqueeze(-1)
        out = out + add
    return out.to(routed.dtype)


@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("bsz, H, W", [(1, 256, 2048), (1, 1000, 4096), (3, 4096, 2048), (8, 512, 2048)])
@torch.inference_mode()
def test_moe_gate_fused_rdna_gated(device, dtype, bsz, H, W):
    y = (torch.randn(bsz, W, device = device) / 8).half()
    w = (torch.randn(W, device = device) / W ** 0.5).half()
    routed = torch.randn(bsz, H, dtype = dtype, device = device)
    sh = torch.randn(bsz, H, dtype = dtype, device = device)
    dst = torch.empty_like(routed)
    gate = y.float() @ w.float()
    ref = reference(routed, sh, gate)
    ext.moe_gate_fused_rdna(y, w, routed, sh, dst)
    # the fused dot accumulates in fp32 with its own (deterministic) deposition order, so the
    # gate scalar differs from the torch reduction within fp32 rounding: compare with tolerance
    torch.testing.assert_close(dst, ref, rtol = 1e-3 if dtype == torch.half else 1e-4,
                               atol = 1e-3 if dtype == torch.half else 1e-4)


@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("bsz, H", [(1, 256), (2, 3000), (4, 4096)])
@torch.inference_mode()
def test_moe_gate_fused_rdna_ungated_add(device, dtype, bsz, H):
    # No gate (the shared-expert-without-gate tail): dst = routed + sh, unscaled
    routed = torch.randn(bsz, H, dtype = dtype, device = device)
    sh = torch.randn(bsz, H, dtype = dtype, device = device)
    dst = torch.empty_like(routed)
    ext.moe_gate_fused_rdna(None, None, routed, sh, dst)
    torch.testing.assert_close(dst, reference(routed, sh, None), rtol = 0, atol = 0)


@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("bsz, H", [(1, 256), (2, 3000), (4, 4096)])
@torch.inference_mode()
def test_moe_gate_fused_rdna_copy(device, dtype, bsz, H):
    # No shared expert either (the plain-copy tail): dst = routed, bit-exact in both dtypes
    routed = torch.randn(bsz, H, dtype = dtype, device = device)
    dst = torch.empty_like(routed)
    ext.moe_gate_fused_rdna(None, None, routed, None, dst)
    torch.testing.assert_close(dst, reference(routed, None, None), rtol = 0, atol = 0)


@torch.inference_mode()
def test_moe_gate_fused_rdna_one_hot(device):
    # The whole dot in one lane: any wrong broadcast of the block-wide sum shows up here
    bsz, H, W = 4, 512, 2048
    y = torch.zeros(bsz, W, dtype = torch.half, device = device)
    y[:, 0] = 1.0
    w = torch.zeros(W, dtype = torch.half, device = device)
    w[0] = 0.5
    routed = torch.zeros(bsz, H, dtype = torch.half, device = device)
    sh = torch.ones(bsz, H, dtype = torch.half, device = device)
    dst = torch.empty_like(routed)
    ext.moe_gate_fused_rdna(y, w, routed, sh, dst)
    torch.testing.assert_close(dst, torch.full_like(dst, torch.sigmoid(torch.tensor(0.5)).item()),
                               rtol = 1e-4, atol = 1e-4)


@pytest.mark.parametrize("bsz, H", [(0, 256), (1, 0)])
@torch.inference_mode()
def test_moe_gate_fused_rdna_empty(device, bsz, H):
    # No rows or zero-width output: the kernel returns without launching, dst is untouched
    dst = torch.full((bsz, H), 7.0, device = device)
    ext.moe_gate_fused_rdna(None, None, torch.empty(bsz, H, device = device), None, dst)
    if bsz * H:
        assert (dst.view(-1) == 7.0).all()


def assert_device_ok(device):
    # A failed launch would leave an error for the next op on the device
    torch.cuda.synchronize(device)
    assert torch.ones(8, device = device).sum().item() == 8


@pytest.mark.parametrize("bad", ["sh_dtype", "routed_stride", "y_only", "shape"])
@torch.inference_mode()
def test_moe_gate_fused_rdna_rejects(device, bad):
    # TORCH_CHECK contract of the wrapper: one accumulator dtype, contiguous, matching shapes,
    # y and w as a pair
    routed = torch.randn(2, 256, dtype = torch.half, device = device)
    sh = torch.randn(2, 256, dtype = torch.half, device = device)
    y = torch.randn(2, 64, dtype = torch.half, device = device)
    w = torch.randn(64, dtype = torch.half, device = device)
    dst = torch.empty_like(routed)
    if bad == "sh_dtype":
        with pytest.raises(RuntimeError):
            ext.moe_gate_fused_rdna(None, None, routed, sh.float(), dst)
    elif bad == "routed_stride":
        with pytest.raises(RuntimeError):
            ext.moe_gate_fused_rdna(None, None, routed.t(), None, dst)
    elif bad == "y_only":
        with pytest.raises(RuntimeError):
            ext.moe_gate_fused_rdna(y, None, routed, sh, dst)
    else:
        with pytest.raises(RuntimeError):
            ext.moe_gate_fused_rdna(None, None, routed, torch.randn(2, 257, dtype = torch.half, device = device), dst)
    assert_device_ok(device)