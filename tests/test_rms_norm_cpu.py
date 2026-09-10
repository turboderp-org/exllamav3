"""CPU RMSNorm (native fused kernel) vs torch fallback reference. CPU-only, no CUDA.

Run with EXL3_RMSNORM_CPU_ISA=avx512|avx2|scalar to pin each kernel tier.
"""
import pytest
import torch
from torch import nn
from exllamav3.modules.rmsnorm import RMSNorm


def make_mod(dim_w, dtype_out=None, **kw):
    kw.setdefault("constant_bias", 0.0)
    kw.setdefault("constant_scale", 1.0)
    m = RMSNorm(None, "test", 1e-6, out_dtype=dtype_out, **kw)
    if not kw.get("unweighted", False):
        m.weight = nn.Parameter(torch.randn(dim_w, dtype=torch.half), requires_grad=False)
    else:
        m.weight = None
    return m


def check(cpu_y, ref_y, tol):
    torch.testing.assert_close(cpu_y.float(), ref_y.float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("dim", [100, 132, 128])
@pytest.mark.parametrize("dt", [torch.half, torch.float, torch.bfloat16])
@torch.inference_mode()
def test_cpu_plain_odd_dims(dim, dt):
    # 100/132 exercise the AVX2/AVX-512 scalar tails; 128 is the aligned control
    m = make_mod(dim)
    x = torch.randn(7, dim, dtype=dt)
    check(m.forward(x.clone(), {}), m.forward_torch(x.clone(), {}), 2e-3)


@pytest.mark.parametrize("dt", [torch.half, torch.float, torch.bfloat16])
@torch.inference_mode()
def test_cpu_bias_scale_unweighted(dt):
    m = make_mod(128, constant_bias=1.0, constant_scale=0.5)
    x = torch.randn(8, 128, dtype=dt)
    check(m.forward(x.clone(), {}), m.forward_torch(x.clone(), {}), 2e-3)
    m = make_mod(0, unweighted=True, dtype_out=torch.float)
    check(m.forward(x.clone(), {}), m.forward_torch(x.clone(), {}), 2e-3)


@pytest.mark.parametrize("kwn", ["residual", "residual_in"])
@pytest.mark.parametrize("dt", [torch.half, torch.float, torch.bfloat16])
@torch.inference_mode()
def test_cpu_fused_residual(kwn, dt):
    m = make_mod(132)
    x = torch.randn(6, 132, dtype=dt)
    r1, r2 = torch.randn_like(x), None
    r1 = r1.clone()
    r2 = r1.clone()
    y1 = m.forward(x.clone(), {}, **{kwn: r1})
    y2 = m.forward_torch(x.clone(), {}, **{kwn: r2})
    check(y1, y2, 1e-2)
    check(r1, r2, 1e-2)


@torch.inference_mode()
def test_cpu_groups_span_fallback():
    m = make_mod(100, groups=4)
    x = torch.randn(2, 3, 4, 25, dtype=torch.half)
    check(m.forward(x.clone(), {}), m.forward_torch(x.clone(), {}), 2e-3)
    m = make_mod(128, span_heads=True)
    x = torch.randn(2, 4, 32, dtype=torch.half)
    check(m.forward(x.clone(), {}), m.forward_torch(x.clone(), {}), 2e-3)
    # non-contiguous input takes the torch fallback through the same entry
    m = make_mod(128)
    xn = torch.randn(8, 256, dtype=torch.half)[:, ::2]
    check(m.forward(xn.clone(), {}), m.forward_torch(xn.clone(), {}), 2e-3)
