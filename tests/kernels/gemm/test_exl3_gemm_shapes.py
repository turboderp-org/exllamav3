"""
The EXL3 GEMM shape-table queries (quant/exl3_kernel_map.cu), against what exl3_gemm does with a forced shape:

- exl3_gemm_num_kernel_shapes() = N: shape indices 1..N are the kernel shapes exl3_gemm instantiates (each runs when
  forced and returns its index), N + 1 is rejected as an invalid forced shape
- exl3_gemm_shape_compat(shape, m, k, n, K, half_k) is the product of a divisibility condition (k a multiple of the
  shape's k tile, n of its n tile), a row-count condition that only excludes small m (a wide row tile is skipped
  when it would not replace two 16-row passes: upward closed in m) and the shared-memory fit of the (shape, K)
  instance on this device. A compatible call runs correctly under that forced shape, for every m (including m that
  the row tile does not divide); a call failing divisibility is rejected by the forced exl3_gemm instead of
  silently dropping the remainder

Reference for the forced runs: testlib.trellis.linear (float64, independently decoded weight). Tolerance: the GEMM
rounds the rotated input to fp16 and accumulates in fp32 (fp16 MMA partials on some paths); a few fp16 unit
roundoffs (u ~ 4.9e-4) relative to the output scale, bounded at relative RMS < 2.5e-3 and max < 4e-3.
"""

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib import trellis as tref
from testlib.exl3 import generator, rand_scale


def _n():
    return ext.exl3_gemm_num_kernel_shapes()


def _tiles(s):
    """(tk, tn, m_min) of shape s as the predicate reports them: smallest k / n tile it accepts, smallest m"""
    tk = next(k for k in range(16, 1024, 16) if ext.exl3_gemm_shape_compat(s, 128, k, 3072, 4))
    tn = next(n for n in range(128, 4096, 128) if ext.exl3_gemm_shape_compat(s, 128, 3072, n, 4))
    m_min = next(m for m in range(1, 1025) if ext.exl3_gemm_shape_compat(s, m, 3072, 3072, 4))
    return tk, tn, m_min


def _gemm(A, tr, suh, svh, n, s, mcg, mul1):
    C = torch.empty((A.shape[0], n), dtype = torch.half, device = A.device)
    r = ext.exl3_gemm(A, tr, C, suh, torch.empty_like(A), svh, s, mcg, mul1, 0)
    return r, C


def _assert_close(C, ref):
    e = C.cpu().double() - ref
    rel_rms = (e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    rel_max = (e.abs().max() / ref.abs().max()).item()
    assert rel_rms < 2.5e-3 and rel_max < 4e-3, f"relative RMS {rel_rms:.2e}, max {rel_max:.2e}"


@torch.inference_mode()
def test_num_shapes(device):
    N = _n()
    assert N >= 1
    A = torch.zeros((16, 1536), dtype = torch.half, device = device)
    tr = torch.zeros((96, 96, 32), dtype = torch.int16, device = device)
    ones_k = torch.ones(1536, dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "invalid forced shape index"):
        _gemm(A, tr, ones_k, ones_k, 1536, N + 1, True, False)


@torch.inference_mode()
def test_compat_structure(device):
    """Divisibility x upward-closed row condition, independent of K wherever the instance fits"""
    for s in range(1, _n() + 1):
        tk, tn, m_min = _tiles(s)
        assert tk % 16 == 0 and tn % 128 == 0
        for m in (1, 2, 8, 16, 17, 24, 25, 32, 33, 64, 200):
            for k in (tk, 2 * tk, 3 * tk, tk + 16, 3 * tk - 16):
                for n in (tn, 2 * tn, tn + 128, 3 * tn - 128):
                    expect = k % tk == 0 and n % tn == 0 and m >= m_min
                    for K in (2, 4, 8):
                        assert ext.exl3_gemm_shape_compat(s, m, k, n, K) == expect, (s, m, k, n, K)


def _cases():
    out = []
    for s in range(1, ext.exl3_gemm_num_kernel_shapes() + 1):
        for K, cb in ((2, "mcg"), (4, "3inst"), (6, "mcg"), (8, "mcg"), (2.5, "mul1")):
            out.append((s, K, cb))
    return out


@pytest.mark.parametrize("s, K, cb", _cases())
@torch.inference_mode()
def test_compatible_shapes_run(device, s, K, cb):
    k = n = 1536    # divisible by every k and n tile
    half = not float(K).is_integer()
    seed = s * 100 + int(K * 2)
    g = generator(seed)
    p = tref.random_packed((k // 16) * (n // 16), K, np.random.default_rng(seed)).reshape(k // 16, n // 16, -1)
    suh, svh = rand_scale(k, g), rand_scale(n, g)
    tr = torch.from_numpy(p.copy()).to(device)
    W = tref.dequant(p, suh, svh, K, cb)
    ran = 0
    # m >= 3 keeps mul1 calls off the int8 GEMV, which takes small-m mul1 calls ahead of any shape selection
    for m in (3, 17, 50, 100):
        if not ext.exl3_gemm_shape_compat(s, m, k, n, int(K), half):
            continue
        A = (torch.randn(m, k, generator = g) * 0.5).half()
        r, C = _gemm(A.to(device), tr, suh.to(device), svh.to(device), n, s, cb == "mcg", cb == "mul1")
        assert r == s, f"forced shape {s} reported {r}"
        _assert_close(C, A.double() @ W)
        ran += 1
    assert ran > 0, "no compatible row count"


@torch.inference_mode()
def test_incompatible_divisibility_rejected(device):
    """A forced shape whose k or n tile does not divide the problem is refused, matching shape_compat"""
    K = 3
    checked = 0
    for s in range(1, _n() + 1):
        tk, tn, _ = _tiles(s)
        for k, n in ((tk * 4 + 16, tn * 2), (tk * 4, tn * 2 + 128)):
            if k % tk == 0 and n % tn == 0:
                continue
            assert not ext.exl3_gemm_shape_compat(s, 64, k, n, K)
            A = torch.zeros((64, k), dtype = torch.half, device = device)
            tr = torch.zeros((k // 16, n // 16, 16 * K), dtype = torch.int16, device = device)
            with pytest.raises(RuntimeError, match = "does not divide"):
                _gemm(A, tr, torch.ones(k, dtype = torch.half, device = device),
                      torch.ones(n, dtype = torch.half, device = device), n, s, True, False)
            checked += 1
    assert checked > 0
