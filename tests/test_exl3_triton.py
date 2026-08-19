"""Tests for the Triton EXL3 linear implementation (modules/quant/exl3_triton.py).

Mirrors the coverage of the C++ implementation's tests:

- shapes, bit widths (K), mcg/mul1 combinations and tolerances follow
  tests/test_reconstruct_had.py (the reconstruct + Hadamard kernel test),
  with the full K = 1..8 sweep from tests/test_quant_fn.py
- row counts follow tests/test_quant_fn.py's batch sizes (1, 16, 17, 128)

The C++ reconstruct + hgemm + hadamard pipeline is treated as the reference
and assumed correct (same assumption test_reconstruct_had.py makes about the
plain reconstruct kernel).
"""
import os

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3_triton import (
    had_r_128_triton as had_r_128_triton_fn,
    has_triton,
    linear_exl3_triton,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA (or ROCm) device required"),
    pytest.mark.skipif(not has_triton, reason="Triton not available"),
]


def device():
    return torch.device(os.environ.get("EXL_TEST_DEVICE", "cuda:0"))


# ---------------------------------------------------------------------------
# Reference (the C++ pipeline; assumed correct)
# ---------------------------------------------------------------------------

def reference_reconstruct_hgemm(x, trellis, suh, svh, K, mcg, mul1,
                                in_features, out_features, dev):
    original_shape = x.shape
    x = x.view(-1, in_features)
    rows = x.shape[0]
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    w = torch.empty((in_features, out_features), dtype=torch.half, device=dev)
    ext.reconstruct(w, trellis, K, mcg, mul1)
    y = torch.empty((rows, out_features), dtype=torch.half, device=dev)
    ext.hgemm(xh, w, y)
    ext.had_r_128(y, y, None, svh, 1.0)
    return y.view(original_shape[:-1] + (out_features,))


def make_trellis(in_features, out_features, K, dev):
    # Same construction as test_reconstruct_had.py
    return torch.randint(
        0, 65536, (in_features // 16, out_features // 16, 256 * K // 16),
        dtype=torch.int32, device=dev,
    ).to(torch.short)


def make_suh_svh(in_features, out_features, dev):
    suh = torch.sign(torch.randn(in_features, device=dev)).half()
    svh = torch.sign(torch.randn(out_features, device=dev)).half()
    return suh, svh


# ---------------------------------------------------------------------------
# Full linear vs reference
# ---------------------------------------------------------------------------

# (in_features, out_features, K) — shapes from test_reconstruct_had.py
SHAPES = [
    (256, 128, 3),
    (512, 384, 2),
    (1024, 512, 5),
    (384, 256, 4),
    (256, 512, 3),
    (4096, 1024, 3),
]

# Full bit-width sweep (K = 1..8, as in test_quant_fn.py) on two small shapes
K_SWEEP_SHAPES = [
    (256, 128),
    (512, 384),
]

CB_VARIANTS = [(False, False), (True, False), (False, True)]
ROWS = [1, 16, 17, 128]


def _rel_err(y, y_ref):
    err = (y.float() - y_ref.float()).abs().max().item()
    scale = y_ref.float().abs().max().item()
    return err / scale


@pytest.mark.parametrize("mcg,mul1", CB_VARIANTS)
@pytest.mark.parametrize("in_features,out_features,K", SHAPES)
def test_linear_vs_reference_shapes(in_features, out_features, K, mcg, mul1):
    dev = device()
    torch.manual_seed(in_features * 7 + out_features + K)
    trellis = make_trellis(in_features, out_features, K, dev)
    suh, svh = make_suh_svh(in_features, out_features, dev)
    x = torch.randn(16, in_features, dtype=torch.half, device=dev) * 0.1

    y_ref = reference_reconstruct_hgemm(
        x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev
    )
    y = linear_exl3_triton(
        x, trellis, suh, svh, K, mcg, mul1,
        in_features, out_features, dev, torch.half,
    )
    assert _rel_err(y, y_ref) < 2e-2


@pytest.mark.parametrize("mcg,mul1", CB_VARIANTS)
@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("in_features,out_features", K_SWEEP_SHAPES)
def test_linear_vs_reference_all_k(in_features, out_features, K, mcg, mul1):
    dev = device()
    torch.manual_seed(in_features * 7 + out_features + K)
    trellis = make_trellis(in_features, out_features, K, dev)
    suh, svh = make_suh_svh(in_features, out_features, dev)
    x = torch.randn(16, in_features, dtype=torch.half, device=dev) * 0.1

    y_ref = reference_reconstruct_hgemm(
        x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev
    )
    y = linear_exl3_triton(
        x, trellis, suh, svh, K, mcg, mul1,
        in_features, out_features, dev, torch.half,
    )
    assert _rel_err(y, y_ref) < 2e-2


# Shapes whose K/N are NOT multiples of 128: the fused kernel's fast decode
# requires full tiles, so these exercise the generic staged-row gather path
# (in particular the K=7 window table and the K=1 decode there). The Hadamard
# wrappers (C++ and Triton) require 128-divisible dimensions, so this tests
# the fused GEMM kernel directly on pre-transformed input; the reference
# weight matrix comes from the C++ reconstruct on zero-padded trellis tiles.
NONDIV_SHAPES = [
    (4112, 384),   # K % 128 == 16
    (4096, 4112),  # N % 128 == 16
]


@pytest.mark.parametrize("mcg,mul1", [(False, False), (True, True)])
@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("in_features,out_features", NONDIV_SHAPES)
@pytest.mark.parametrize("rows", [1, 17])
def test_gemm_generic_path_all_k(in_features, out_features, K, mcg, mul1, rows):
    from exllamav3.modules.quant.exl3_triton import (
        _decode_lut, _get_perm, exl3_gemm_triton,
    )

    dev = device()
    torch.manual_seed(in_features + out_features * 3 + K * 11 + rows)
    trellis = make_trellis(in_features, out_features, K, dev)
    xh = torch.randn(rows, in_features, dtype=torch.half, device=dev) * 0.1

    # reference: reconstruct on zero-padded (128-divisible) trellis, then slice
    in_pad = (in_features + 127) // 128 * 128
    out_pad = (out_features + 127) // 128 * 128
    trellis_pad = torch.cat([
        trellis,
        torch.zeros(in_pad // 16 - trellis.shape[0], trellis.shape[1],
                    trellis.shape[2], dtype=trellis.dtype, device=dev),
    ], dim=0)
    trellis_pad = torch.cat([
        trellis_pad,
        torch.zeros(trellis_pad.shape[0], out_pad // 16 - trellis_pad.shape[1],
                    trellis_pad.shape[2], dtype=trellis.dtype, device=dev),
    ], dim=1)
    w = torch.empty((in_pad, out_pad), dtype=torch.half, device=dev)
    ext.reconstruct(w, trellis_pad, K, mcg, mul1)
    w = w[:in_features, :out_features].contiguous()

    y_ref = torch.empty((rows, out_features), dtype=torch.half, device=dev)
    ext.hgemm(xh, w, y_ref)

    y = torch.empty((rows, out_features), dtype=torch.half, device=dev)
    exl3_gemm_triton(
        xh, trellis, y, _decode_lut(1 if mcg else (2 if mul1 else 0), dev),
        _get_perm(dev), K, trellis.shape[1], 1 if mcg else (2 if mul1 else 0),
    )
    assert _rel_err(y, y_ref) < 2e-2


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("mcg,mul1", [(False, False), (True, True)])
@pytest.mark.parametrize("in_features,out_features,K", SHAPES[:4])
def test_linear_vs_reference_rows(in_features, out_features, K, mcg, mul1, rows):
    dev = device()
    torch.manual_seed(1234 + rows)
    trellis = make_trellis(in_features, out_features, K, dev)
    suh, svh = make_suh_svh(in_features, out_features, dev)
    x = torch.randn(rows, in_features, dtype=torch.half, device=dev) * 0.1

    y_ref = reference_reconstruct_hgemm(
        x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev
    )
    y = linear_exl3_triton(
        x, trellis, suh, svh, K, mcg, mul1,
        in_features, out_features, dev, torch.half,
    )
    assert _rel_err(y, y_ref) < 2e-2


@pytest.mark.parametrize("mcg,mul1", CB_VARIANTS)
@pytest.mark.parametrize("in_features,out_features,K", SHAPES[:3])
def test_linear_with_bias(in_features, out_features, K, mcg, mul1):
    dev = device()
    torch.manual_seed(7)
    trellis = make_trellis(in_features, out_features, K, dev)
    suh, svh = make_suh_svh(in_features, out_features, dev)
    x = torch.randn(2, in_features, dtype=torch.half, device=dev) * 0.1
    bias = (torch.randn(out_features, device=dev) * 0.1).to(torch.half)

    y_ref = reference_reconstruct_hgemm(
        x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev
    ) + bias

    y = linear_exl3_triton(
        x, trellis, suh, svh, K, mcg, mul1,
        in_features, out_features, dev, torch.half, bias,
    )
    assert _rel_err(y, y_ref) < 2e-2


@pytest.mark.parametrize("mcg,mul1", CB_VARIANTS)
@pytest.mark.parametrize("in_features,out_features,K", SHAPES[:3])
def test_linear_fp32_output(in_features, out_features, K, mcg, mul1):
    dev = device()
    torch.manual_seed(11)
    trellis = make_trellis(in_features, out_features, K, dev)
    suh, svh = make_suh_svh(in_features, out_features, dev)
    x = torch.randn(1, in_features, dtype=torch.half, device=dev) * 0.1

    y = linear_exl3_triton(
        x, trellis, suh, svh, K, mcg, mul1,
        in_features, out_features, dev, torch.float,
    )
    assert y.dtype == torch.float

    y_ref = reference_reconstruct_hgemm(
        x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev
    )
    assert _rel_err(y, y_ref.float()) < 2e-2


@pytest.mark.parametrize("rows", ROWS)
def test_linear_noncontiguous_batch(rows):
    """3D input (as the model feeds it) must produce the same result as 2D."""
    dev = device()
    torch.manual_seed(rows)
    in_features, out_features, K = 256, 512, 3
    trellis = make_trellis(in_features, out_features, K, dev)
    suh, svh = make_suh_svh(in_features, out_features, dev)
    x3 = torch.randn(1, rows, in_features, dtype=torch.half, device=dev) * 0.1
    x2 = x3.view(rows, in_features)

    y_ref = reference_reconstruct_hgemm(
        x2, trellis, suh, svh, K, False, False, in_features, out_features, dev
    )
    y = linear_exl3_triton(
        x3, trellis, suh, svh, K, False, False,
        in_features, out_features, dev, torch.half,
    )
    assert y.shape == (1, rows, out_features)
    assert _rel_err(y.view(rows, -1), y_ref) < 2e-2


# ---------------------------------------------------------------------------
# Triton Hadamard vs the C++ kernel (bit-identical)
# ---------------------------------------------------------------------------

HAD_DIMS = [128, 256, 384, 512, 1024, 2048]


@pytest.mark.parametrize("dim", HAD_DIMS)
@pytest.mark.parametrize("dtype", [torch.half, torch.float])
def test_had_r_128_triton_vs_cpp(dim, dtype):
    dev = device()
    torch.manual_seed(42)
    x = torch.randn(2, dim, dtype=dtype, device=dev)
    s = torch.sign(torch.randn(dim, device=dev)).half()

    for which in ("pre", "post"):
        y_tri = torch.empty_like(x)
        y_cpp = torch.empty_like(x)
        if which == "pre":
            had_r_128_triton_fn(x, y_tri, s, None, 1.0)
            ext.had_r_128(x, y_cpp, s, None, 1.0)
        else:
            had_r_128_triton_fn(x, y_tri, None, s, 1.0)
            ext.had_r_128(x, y_cpp, None, s, 1.0)
        torch.testing.assert_close(y_tri, y_cpp, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# LinearEXL3.forward dispatch: EXL3_PREFER_TRITON_LINEAR=1 vs the reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mcg,mul1", CB_VARIANTS)
@pytest.mark.parametrize("in_features,out_features,K", SHAPES)
def test_forward_dispatch_matches_reference(in_features, out_features, K, mcg, mul1, monkeypatch):
    """With EXL3_PREFER_TRITON_LINEAR=1 the module forward must match the C++ path."""
    from exllamav3.modules.quant import exl3 as exl3_mod

    dev = device()
    torch.manual_seed(123)
    trellis = make_trellis(in_features, out_features, K, dev)
    suh, svh = make_suh_svh(in_features, out_features, dev)

    mcg_tensor = torch.ones(1, dtype=torch.half, device=dev) if mcg else None
    mul1_tensor = torch.ones(1, dtype=torch.half, device=dev) if mul1 else None

    lin = exl3_mod.LinearEXL3(
        None, in_features, out_features,
        suh=suh, svh=svh, trellis=trellis,
        mcg=mcg_tensor, mul1=mul1_tensor,
    )

    x = torch.randn(17, in_features, dtype=torch.half, device=dev) * 0.1

    # Reference: reconstruct_hgemm through the normal dispatch
    y_ref = lin.reconstruct_hgemm(x, None)

    # Triton: force the env flag through the module-level gate
    monkeypatch.setattr(exl3_mod, "use_triton", True)
    y = lin.forward(x, {})

    assert _rel_err(y, y_ref) < 2e-2
