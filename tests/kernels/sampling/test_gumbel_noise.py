"""
Contracts of the Gumbel-noise kernels, against an independent NumPy reimplementation of cuRAND's Philox4x32-10
stream (testlib.sampling) and the exact float64 Gumbel transform:

- gumbel_noise_f32(logits_in, logits, seed): logits[i] = logits_in[i] + G(u_i) for every flat index i of logits
  (fp32, in place allowed), u_i the first curand_uniform of curand_init(seed, i, 0), G(u) = -ln(-ln(min(u, 1 -
  2^-24))). Nothing past numel is written. Deterministic per seed.
- gumbel_noise_log(probs, logits, seed): logits[i] = ln(probs[i]) + G(u_i), same stream; probs == 0 gives -inf;
  shapes must match (TORCH_CHECK).
- adaptivep_gumbel_noise_f32(probs, logits, seed, target, inv_width, peak, sharpness): probs < 1e-8 gives -inf,
  otherwise peak - sharpness * a^2 / (a + 1) + G(u_i) with a = |p - target| * inv_width, same stream.
- gumbel_sample(logits (bsz, V) half, ids (bsz, 1) long, max_logit, seed): per row the argmax over i < max_logit
  (0 = all) of logits[i] + G(u), the uniforms drawn per thread (curand_init(seed, thread, 0), two draws per
  element pair and 2048-element round). No Python caller (see the report); its batch rows share one noise
  stream and its paired half2 reads assume an even row length.

Empty inputs are no-ops for all of them (elementwise), with no CUDA error left pending; gumbel_sample's empty
cases are in test_sampling_edges.py.

The kernels use __logf, so the per-element comparison carries a tolerance derived from __logf's error
(testlib.sampling.gumbel_noise_tolerance); elements with u within 1e-6 of 1, where -ln(u) is too small for
__logf's absolute error, are only checked to be finite and bounded. The distribution is additionally checked by
a Kolmogorov-Smirnov statistic against the exact Gumbel CDF and by its first two moments.
"""

import math

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib import isolated
from testlib.sampling import (
    certain_argmax,
    element_noise,
    gumbel_noise_tolerance,
    gumbel_sample_noise,
)

# Uniforms this close to 1 make -ln(u) smaller than __logf's absolute error: the kernel value is then only
# bounded, not compared (the clamp to 1 - 2^-24 caps the exact noise at 16.64; __logf at the clamp gives ~18.6)
U_TIGHT = 1.0 - 1e-6
NOISE_MAX = 19.5
SEEDS = [0, 1, 12345, 0xFFFFFFFF]


def _check_noise(out: np.ndarray, base: np.ndarray, u: np.ndarray, g: np.ndarray, msg: str):
    """out == base + g within the __logf tolerance plus the fp32 rounding of the addition"""
    exp = base.astype(np.float64) + g
    tight = u < U_TIGHT
    finite = np.isfinite(base)
    sel = tight & finite
    tol = gumbel_noise_tolerance(u) + 2.0 ** -23 * np.abs(exp)
    with np.errstate(invalid = "ignore"):
        err = np.abs(out.astype(np.float64) - exp)
    bad = sel & ~(err <= tol)
    assert not bad.any(), \
        f"{msg}: {bad.sum()} elements off, first at {np.argmax(bad)}: {out[np.argmax(bad)]} vs {exp[np.argmax(bad)]}"
    loose = ~tight & finite
    if loose.any():
        d = out[loose].astype(np.float64) - base[loose]
        assert np.isfinite(d).all() and (d > 10.0).all() and (d < NOISE_MAX).all(), f"{msg}: tail noise {d}"
    # Masked (-inf) inputs stay -inf
    assert np.all(out[~finite] == base[~finite]), f"{msg}: -inf inputs changed"


# gumbel_noise_f32

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape", [(1,), (3,), (1, 1023), (1, 1024), (1, 1025), (1, 4097), (3, 32001), (1, 151937)])
@torch.inference_mode()
def test_gumbel_noise_f32_matches_philox(shape, seed, device):
    g_ = torch.Generator().manual_seed(seed & 0xFFFF)
    x = (torch.randn(shape, generator = g_) * 4).to(device)
    flat = x.view(-1)
    if flat.numel() > 4:
        flat[1] = float("-inf")
    n = x.numel()
    # Output with a NaN guard region past numel
    out_full = torch.full((n + 256,), float("nan"), device = device)
    out = out_full[:n].view(shape)
    ext.gumbel_noise_f32(x, out, seed)
    u, g = element_noise(seed, n)
    _check_noise(out.view(-1).cpu().numpy(), x.view(-1).cpu().numpy(), u, g, f"shape {shape} seed {seed}")
    assert torch.isnan(out_full[n:]).all(), "gumbel_noise_f32 wrote past numel"


@torch.inference_mode()
def test_gumbel_noise_f32_in_place_and_deterministic(device):
    x = torch.randn(2, 50257, device = device)
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    ext.gumbel_noise_f32(x, a, 99)
    ext.gumbel_noise_f32(x, b, 99)
    assert torch.equal(a, b), "same seed must give bit-identical noise"
    c = x.clone()
    ext.gumbel_noise_f32(c, c, 99)
    assert torch.equal(a, c), "in-place result differs from out-of-place"
    ext.gumbel_noise_f32(x, b, 100)
    assert (a != b).float().mean().item() > 0.999, "different seeds must give different noise"


@torch.inference_mode()
def test_gumbel_noise_f32_distribution(device):
    n = 1 << 21
    x = torch.zeros(n, device = device)
    y = torch.empty_like(x)
    ext.gumbel_noise_f32(x, y, 2024)
    s = np.sort(y.cpu().numpy().astype(np.float64))
    cdf = np.exp(-np.exp(-s))
    i = np.arange(1, n + 1)
    ks = max((i / n - cdf).max(), (cdf - (i - 1) / n).max())
    # KS critical value at significance 1e-4 is ~2.2 / sqrt(n); the stream is fixed, so this is deterministic
    assert ks < 2.2 / math.sqrt(n), f"KS statistic {ks:.2e} vs exact Gumbel CDF"
    mean, var = s.mean(), s.var()
    euler_gamma = 0.5772156649015329
    # Standard errors: sigma / sqrt(n) for the mean, sqrt((kurtosis_excess + 2) / n) * var for the variance
    # (Gumbel excess kurtosis 12/5)
    var_ref = math.pi ** 2 / 6
    assert abs(mean - euler_gamma) < 5 * math.sqrt(var_ref / n), f"mean {mean} vs {euler_gamma}"
    assert abs(var - var_ref) < 5 * math.sqrt((2.4 + 2) / n) * var_ref, f"variance {var} vs {var_ref}"


@torch.inference_mode()
def test_gumbel_noise_f32_rejects_wrong_dtype(device):
    x = torch.zeros(8, device = device, dtype = torch.half)
    with pytest.raises(RuntimeError):
        ext.gumbel_noise_f32(x, x, 0)


# gumbel_noise_log

@pytest.mark.parametrize("seed", [0, 7, 0xFFFFFFFF])
@pytest.mark.parametrize("shape", [(1,), (1, 1025), (2, 32003), (1, 151936)])
@torch.inference_mode()
def test_gumbel_noise_log_matches_philox(shape, seed, device):
    g_ = torch.Generator().manual_seed(seed & 0xFFFF)
    p = torch.softmax(torch.randn(shape, generator = g_, dtype = torch.float64) * 3, dim = -1).float()
    flat = p.view(-1)
    flat[::7] = 0.0            # truncated entries, as left by top-K/top-P on the probs path
    p = p.to(device)
    n = p.numel()
    out = torch.full_like(p, float("nan"))
    ext.gumbel_noise_log(p, out, seed)
    u, g = element_noise(seed, n)
    pn = p.view(-1).cpu().numpy().astype(np.float64)
    with np.errstate(divide = "ignore"):
        logp = np.log(pn)
    o = out.view(-1).cpu().numpy()
    zero = pn == 0
    assert np.all(o[zero] == -np.inf), "probs == 0 must give -inf"
    # __logf(p): absolute error ~2^-21 near 1, relative ~2^-21 for |ln p| > 1
    tol_log = 2.0 ** -21 * np.maximum(1.0, np.abs(np.where(zero, 0.0, logp)))
    exp = np.where(zero, -np.inf, logp + g)
    tight = (u < U_TIGHT) & ~zero
    tol = gumbel_noise_tolerance(u) + tol_log + 2.0 ** -23 * np.abs(np.where(zero, 0.0, exp))
    with np.errstate(invalid = "ignore"):
        err = np.abs(o.astype(np.float64) - exp)
    bad = tight & ~(err <= tol)
    assert not bad.any(), f"{bad.sum()} elements off, first {np.argmax(bad)}: {o[np.argmax(bad)]} vs {exp[np.argmax(bad)]}"
    loose = ~(u < U_TIGHT) & ~zero
    if loose.any():
        assert np.isfinite(o[loose]).all()


@torch.inference_mode()
def test_gumbel_noise_log_in_place(device):
    p = torch.softmax(torch.randn(1, 4099, device = device), dim = -1)
    a = torch.empty_like(p)
    ext.gumbel_noise_log(p, a, 5)
    ext.gumbel_noise_log(p, p, 5)
    assert torch.equal(a, p)


@pytest.mark.parametrize("fn", ["gumbel_noise_f16", "gumbel_noise_f32", "gumbel_noise_log", "adaptivep_gumbel_noise_f32"])
@torch.inference_mode()
def test_gumbel_noise_rejects(device, fn):
    # Input and output must match in shape (a smaller input would be read out of bounds) and dtype
    dtype, other = (torch.half, torch.float) if fn == "gumbel_noise_f16" else (torch.float, torch.half)
    extra = (0.5, 2.0, 5.0, 1.0) if fn == "adaptivep_gumbel_noise_f32" else ()
    p = torch.rand(1, 64, device = device).to(dtype)
    for x in (torch.rand(1, 63, device = device).to(dtype), torch.rand(2, 32, device = device).to(dtype)):
        with pytest.raises(RuntimeError, match = "incompatible shapes"):
            getattr(ext, fn)(x, torch.empty(1, 64, dtype = dtype, device = device), 0, *extra)
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        getattr(ext, fn)(p.to(other), torch.empty(1, 64, dtype = dtype, device = device), 0, *extra)


# adaptivep_gumbel_noise_f32

ADAPTIVEP = dict(inv_width = 1.0 / 0.3, peak = 5.0, sharpness = 10.0)


@pytest.mark.parametrize("target", [0.0, 0.3, 0.75, 1.0])
@pytest.mark.parametrize("n", [1, 1025, 32000])
@torch.inference_mode()
def test_adaptivep_matches_closed_form(n, target, device):
    seed = 31337 + n
    g_ = torch.Generator().manual_seed(n)
    # Sorted, normalized, truncated probabilities as the caller passes them (PROBS_N_S); include values on both
    # sides of the 1e-8 cutoff
    p = torch.sort(torch.softmax(torch.randn(1, n, generator = g_, dtype = torch.float64) * 4, dim = -1),
                   descending = True).values.float()
    flat = p.view(-1)
    if n > 8:
        flat[n // 2:] = 0.0
        flat[n // 2 - 1] = 5e-9
        flat[n // 2 - 2] = 2e-8
    p = p.to(device)
    out = torch.full_like(p, float("nan"))
    ext.adaptivep_gumbel_noise_f32(p, out, seed, target, ADAPTIVEP["inv_width"], ADAPTIVEP["peak"], ADAPTIVEP["sharpness"])
    u, g = element_noise(seed, n)
    pn = p.view(-1).cpu().numpy()
    o = out.view(-1).cpu().numpy()
    cut = pn.astype(np.float64) < 1e-8
    assert np.all(o[cut] == -np.inf), "probs below 1e-8 must give -inf"
    a = np.abs(pn.astype(np.float64) - target) * ADAPTIVEP["inv_width"]
    f = ADAPTIVEP["peak"] - ADAPTIVEP["sharpness"] * a * a / (a + 1.0)
    exp = f + g
    # fp32 evaluation of f (|f| <= ~40, a handful of roundings incl. an approximate division) plus the noise bound
    tol = gumbel_noise_tolerance(u) + 1e-6 * (1.0 + np.abs(f)) * 8 + 2.0 ** -23 * np.abs(exp)
    tight = ~cut & (u < U_TIGHT)
    err = np.abs(o.astype(np.float64) - exp)
    bad = tight & ~(err <= tol)
    assert not bad.any(), f"{bad.sum()} off, first {np.argmax(bad)}: {o[np.argmax(bad)]} vs {exp[np.argmax(bad)]}"


# gumbel_sample

def _gumbel_sample_ref(logits: torch.Tensor, max_logit: int, seed: int) -> list[int | None]:
    bsz, v = logits.shape
    ml = v if max_logit == 0 else min(max_logit, v)
    out = []
    for r in range(bsz):
        u, g = gumbel_sample_noise(seed, v, row = r)
        tol = gumbel_noise_tolerance(u) + 1e-5
        tol[u >= U_TIGHT] = 4.0          # bounded tail (see U_TIGHT)
        val = logits[r].float().cpu().numpy().astype(np.float64) + g
        val[ml:] = -np.inf
        out.append(certain_argmax(val, tol))
    return out


@pytest.mark.parametrize("max_logit", [0, 1, 1000, 2047, 2049])
@pytest.mark.parametrize("bsz, v", [(1, 64), (1, 2048), (3, 4096), (2, 32000), (1, 128256)])
@torch.inference_mode()
def test_gumbel_sample_matches_reference(bsz, v, max_logit, device):
    g_ = torch.Generator().manual_seed(v + max_logit)
    checked = 0
    for seed in range(24):
        logits = (torch.randn(bsz, v, generator = g_) * 2).half().to(device)
        ids = torch.full((bsz, 1), -1, dtype = torch.long, device = device)
        ext.gumbel_sample(logits, ids, max_logit, seed)
        ref = _gumbel_sample_ref(logits, max_logit, seed)
        ml = v if max_logit == 0 else min(max_logit, v)
        for r in range(bsz):
            got = ids[r, 0].item()
            assert 0 <= got < ml, f"row {r}: sampled {got} outside [0, {ml})"
            if ref[r] is not None:
                assert got == ref[r], f"seed {seed} row {r}: sampled {got}, reference {ref[r]}"
                checked += 1
    # Rows whose winner is within the noise tolerance of a runner-up are undecidable; large vocabularies put the
    # winners deep in the Gumbel tail where the __logf tolerance is widest
    assert checked >= 0.6 * 24 * bsz


@torch.inference_mode()
def test_gumbel_sample_rows_independent(device):
    """Rows of one batch are separate samples: identical rows must not always draw the same token (each row draws
    from its own Philox subsequences)"""
    v = 4096
    logits = torch.zeros(2, v, dtype = torch.half, device = device)
    ids = torch.empty((2, 1), dtype = torch.long, device = device)
    same = 0
    for seed in range(16):
        ext.gumbel_sample(logits, ids, 0, seed)
        same += int(ids[0, 0].item() == ids[1, 0].item())
    # Independent uniform draws over 4096 tokens collide with probability 1/4096 per seed
    assert same <= 1, f"identical rows drew the same token for {same}/16 seeds"


def _gumbel_sample_odd_worker(device: str, v: int):
    torch.cuda.set_device(torch.device(device))
    logits = (torch.randn(2, v, generator = torch.Generator().manual_seed(v)) * 2).half().to(device)
    ids = torch.empty((2, 1), dtype = torch.long, device = device)
    ext.gumbel_sample(logits, ids, 0, 77)
    torch.cuda.synchronize()
    return ids.cpu(), logits.cpu()


@pytest.mark.parametrize("v", [4097, 32001])
def test_gumbel_sample_odd_vocab_batched(v, device):
    """Odd row length with bsz > 1: rows after the first start 2-byte aligned, and read2f loads half2 pairs
    (runs in a child process since a misaligned access poisons the CUDA context)"""
    ids, logits = isolated.run_isolated(_gumbel_sample_odd_worker, str(device), v)
    ref = _gumbel_sample_ref(logits, 0, 77)
    for r in range(2):
        assert 0 <= ids[r, 0].item() < v
        if ref[r] is not None:
            assert ids[r, 0].item() == ref[r]


def _device_still_works(device):
    # A launch error left pending by the call would surface in the next unrelated launch
    y = torch.ones(4, device = device) * 2
    torch.cuda.synchronize(device)
    assert y.sum().item() == 8


@pytest.mark.parametrize("fn", ["gumbel_noise_f16", "gumbel_noise_f32", "gumbel_noise_log", "adaptivep_gumbel_noise_f32"])
@pytest.mark.parametrize("shape", [(0,), (0, 128), (3, 0)])
@torch.inference_mode()
def test_empty_noise(device, fn, shape):
    dtype = torch.half if fn == "gumbel_noise_f16" else torch.float
    x = torch.empty(shape, dtype = dtype, device = device)
    out = torch.empty(shape, dtype = dtype, device = device)
    extra = (0.5, 2.0, 5.0, 1.0) if fn == "adaptivep_gumbel_noise_f32" else ()
    getattr(ext, fn)(x, out, 7, *extra)
    _device_still_works(device)
