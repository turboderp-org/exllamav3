"""
ext.exl3_gemv (quant/exl3_gemv.cu), the forced entry point of the cooperative small-m GEMV that exl3_gemm dispatches
to when its heuristic applies, and ext.exl3_gemv_int8_max_k, the per-device bitrate cap of the int8-activation GEMV
that Config.use_mgemm keys its fuse/unfuse decision on.

exl3_gemv contract: C (..., n) = A (..., k) @ W for W = diag(suh) H W_hat H diag(svh) (the EXL3 linear), every
leading dim of A flattened into m <= 8 rows, C fp16 or fp32; A is not modified, nothing outside C is written, and
repeat launches are bit-identical. Eligible: k, n multiples of 128; K = 2..4 with mcg / mul1, K = 4 with 3INST,
K = 1.5 / 2.5 / 3.5 with mul1; suh, A_had (scratch) and svh all given. Everything else is rejected with an error.
Both kernel configurations are exercised (n <= 8192 narrow, n > 8192 wide), and m = 1 (MMODE 0) and 2..8.

Reference: testlib.trellis.linear, the float64 product with the independently decoded weight. Tolerance: the kernel
rounds the rotated input to fp16 (A_had) and accumulates 64-term runs (4 tiles) of the MMA in fp16 before folding to
fp32, so its error is a few fp16 unit roundoffs (u = 2^-11 ~ 4.9e-4) relative to the output scale: relative RMS
< 2.5e-3 (~5u) and max error < 4e-3 of the output's max. A tile mapped or decoded wrongly is an O(1) error.

exl3_gemv_int8_max_k contract: 6 on Hopper and Blackwell (compute capability >= 9), 5 elsewhere; EXL3_INT8_GEMV_MAX_K
overrides it (capped at 8). It is the gate of the int8 GEMV: an m = 1 mul1 exl3_gemm at K = max_k takes the int8
path (recognizable by its int8-activation error level), at K = max_k + 1 it does not.
"""

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib import trellis as tref
from testlib.env import compute_capability
from testlib.exl3 import generator, rand_linear, rand_scale
from testlib.isolated import run_isolated

CASES = [
    # K, codebook, m, k, n, fp32 out
    (4, "3inst", 1, 1024, 1024, False),
    (4, "mcg", 2, 512, 384, True),
    (4, "mul1", 8, 256, 2048, False),
    (2, "mcg", 1, 512, 8320, False),        # wide config
    (2, "mul1", 3, 1024, 640, True),
    (3, "mcg", 5, 384, 512, False),
    (3, "mul1", 7, 512, 8448, True),        # wide config
    (1.5, "mul1", 1, 1024, 1024, False),
    (2.5, "mul1", 4, 512, 768, True),
    (3.5, "mul1", 8, 256, 8320, False),     # wide config
]


def _weights(k, n, K, seed):
    g = generator(seed)
    p = tref.random_packed((k // 16) * (n // 16), K, np.random.default_rng(seed)).reshape(k // 16, n // 16, -1)
    return p, rand_scale(k, g), rand_scale(n, g), g


def _gemv(A, tr, suh, svh, n, cb, fp32, device):
    lead = A.shape[:-1]
    m = int(np.prod(lead))
    rows = 2 * n
    big = torch.full((m * n + 2 * rows,), 7.0, dtype = torch.float if fp32 else torch.half, device = device)
    C = big[rows : rows + m * n].view(*lead, n)
    A_had = torch.empty(A.shape, dtype = torch.half, device = device)
    ext.exl3_gemv(A, tr, C, suh, A_had, svh, cb == "mcg", cb == "mul1")
    assert (big[:rows] == 7.0).all() and (big[-rows:] == 7.0).all(), "wrote outside C"
    return C


def _assert_close(C, ref):
    e = C.cpu().double() - ref
    rel_rms = (e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    rel_max = (e.abs().max() / ref.abs().max()).item()
    assert rel_rms < 2.5e-3 and rel_max < 4e-3, f"relative RMS {rel_rms:.2e}, max {rel_max:.2e}"


@pytest.mark.parametrize("K, cb, m, k, n, fp32", CASES)
@torch.inference_mode()
def test_gemv_matches_reference(device, K, cb, m, k, n, fp32):
    p, suh, svh, g = _weights(k, n, K, m * 1000 + k + n)
    A = (torch.randn(m, k, generator = g) * 0.5).half()
    ref = tref.linear(A, p, suh, svh, K, cb)
    tr = torch.from_numpy(p.copy()).to(device)
    Ad = A.to(device)
    A0 = Ad.clone()
    C = _gemv(Ad, tr, suh.to(device), svh.to(device), n, cb, fp32, device)
    _assert_close(C, ref)
    assert torch.equal(Ad, A0), "A was modified"
    C2 = _gemv(Ad, tr, suh.to(device), svh.to(device), n, cb, fp32, device)
    assert torch.equal(C, C2), "repeat launch differs"


@torch.inference_mode()
def test_gemv_flattens_leading_dims(device):
    K, cb, k, n = 3, "mul1", 512, 384
    p, suh, svh, g = _weights(k, n, K, 3)
    A = (torch.randn(2, 3, k, generator = g) * 0.5).half()
    ref = tref.linear(A.view(6, k), p, suh, svh, K, cb).view(2, 3, n)
    C = _gemv(A.to(device), torch.from_numpy(p.copy()).to(device), suh.to(device), svh.to(device), n, cb, False, device)
    assert C.shape == (2, 3, n)
    _assert_close(C, ref)


@torch.inference_mode()
def test_gemv_rejects_ineligible(device):
    def call(m, k, n, K, mcg, mul1, with_scales = True, c_dtype = torch.half, c_rows = None):
        tr = torch.zeros((k // 16, n // 16, int(16 * K)), dtype = torch.int16, device = device)
        A = torch.zeros((m, k), dtype = torch.half, device = device)
        C = torch.empty((m if c_rows is None else c_rows, n), dtype = c_dtype, device = device)
        suh = torch.ones(k, dtype = torch.half, device = device) if with_scales else None
        svh = torch.ones(n, dtype = torch.half, device = device) if with_scales else None
        A_had = torch.empty_like(A) if with_scales else None
        ext.exl3_gemv(A, tr, C, suh, A_had, svh, mcg, mul1)

    call(1, 256, 256, 3, False, True)   # eligible baseline
    torch.cuda.synchronize(device)
    for args, match in [
        ((9, 256, 256, 3, False, True), "not eligible"),            # m > 8
        ((1, 256, 256, 5, False, True), "not eligible"),            # K = 5
        ((1, 256, 256, 1, False, True), "not eligible"),            # K = 1
        ((1, 256, 256, 3, False, False), "not eligible"),           # 3INST below K = 4
        ((1, 272, 256, 3, False, True), "not eligible"),            # k % 128
        ((1, 256, 272, 3, False, True), "not eligible"),            # n % 128
        ((1, 256, 256, 2.5, True, False), "require the mul1 codebook"),
        ((1, 256, 256, 3, True, True), "both mcg and mul1"),
    ]:
        with pytest.raises(RuntimeError, match = match):
            call(*args)
    with pytest.raises(RuntimeError, match = "requires suh, A_had and svh"):
        call(1, 256, 256, 3, False, True, with_scales = False)
    with pytest.raises(RuntimeError):
        call(1, 256, 256, 3, False, True, c_dtype = torch.bfloat16)
    with pytest.raises(RuntimeError, match = "exl3_gemv: C must hold one output row per row of A"):
        call(2, 256, 256, 3, False, True, c_rows = 1)


@torch.inference_mode()
def test_int8_max_k_default(device):
    idx = torch.device(device).index
    expected = 6 if compute_capability(device)[0] >= 9 else 5
    assert ext.exl3_gemv_int8_max_k(idx) == expected


def _int8_gate_worker(A, w, K):
    """(max_k, C) of an m = 1 mul1 exl3_gemm at bitrate K, in a process with the int8 GEMV enabled"""
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from testlib.env import get_test_device
    device = get_test_device()
    torch.cuda.set_device(device)
    w = {key: t.to(device) for key, t in w.items()}
    A = A.to(device)
    C = torch.empty((1, w["svh"].numel()), dtype = torch.half, device = device)
    ext.exl3_gemm(A, w["trellis"], C, w["suh"], torch.empty_like(A), w["svh"], 0, False, True, 0)
    torch.cuda.synchronize(device)
    return ext.exl3_gemv_int8_max_k(device.index), C.cpu()


def test_int8_max_k_gates_int8_path(device):
    """The int8 path has a distinctive error signature: it quantizes the activations to int8, ~1e-2 relative RMS
    against the exact product, while every fp16 path stays at a few fp16 roundoffs (< 2.5e-3, see above)"""
    max_k = ext.exl3_gemv_int8_max_k(torch.device(device).index)
    k, n = 1024, 1024
    for K, expect_int8 in ((max_k, True), (max_k + 1, False)):
        if K > 8:
            continue
        g = generator(K)
        w = rand_linear(k, n, K, g)
        A = (torch.randn(1, k, generator = g) * 0.5).half()
        mk, C = run_isolated(_int8_gate_worker, A, w, K, env = {"EXL3_INT8_GEMV": "2"})
        assert mk == max_k
        ref = tref.linear(A, w["trellis"].numpy(), w["suh"], w["svh"], K, "mul1")
        e = C.double() - ref
        rel_rms = (e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        assert (rel_rms > 2.5e-3) == expect_int8, \
            f"K = {K}, max_k = {max_k}: relative RMS {rel_rms:.2e}, int8 path expected: {expect_int8}"


def _max_k_worker():
    from exllamav3.ext import exllamav3_ext as ext
    from testlib.env import get_test_device
    return ext.exl3_gemv_int8_max_k(get_test_device().index)


@pytest.mark.parametrize("env, expected", [("3", 3), ("7", 7), ("12", 8)])
def test_int8_max_k_env_override(device, env, expected):
    assert run_isolated(_max_k_worker, env = {"EXL3_INT8_GEMV_MAX_K": env}) == expected


def _device_still_works(device):
    # A launch error left pending by the call would surface in the next unrelated launch
    y = torch.ones(4, device = device) * 2
    torch.cuda.synchronize(device)
    assert y.sum().item() == 8


@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("m, k, n", [(0, 256, 256), (4, 256, 0), (4, 0, 256), (0, 0, 0)])
@torch.inference_mode()
def test_empty_gemv(device, fp32, m, k, n):
    """No rows or no output columns: a no-op. An empty reduction (k = 0): the product is zero. suh / svh / A_had
    are still required (an empty tensor counts as given)"""
    K = 3
    tr = torch.zeros((k // 16, n // 16, 16 * K), dtype = torch.int16, device = device)
    A = torch.randn((m, k), device = device).half()
    C = torch.full((m, n), 5.0, dtype = torch.float if fp32 else torch.half, device = device)
    suh = torch.ones(k, dtype = torch.half, device = device)
    svh = torch.ones(n, dtype = torch.half, device = device)
    ext.exl3_gemv(A, tr, C, suh, torch.empty_like(A), svh, False, True)
    torch.cuda.synchronize(device)
    assert (C == (0.0 if k == 0 and m and n else 5.0)).all()
    with pytest.raises(RuntimeError, match = "requires suh, A_had and svh"):
        ext.exl3_gemv(A, tr, C, None, torch.empty_like(A), svh, False, True)
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        ext.exl3_gemv(A, tr, C.bfloat16(), suh, torch.empty_like(A), svh, False, True)
    _device_still_works(device)
