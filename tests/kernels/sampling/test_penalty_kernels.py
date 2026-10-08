"""
Kernel-level contracts of the repetition penalties (generator/rep_pen.cu, generator/dry.cu) in isolation, against
brute-force Python references over the token history. tests/generator/sampler/test_sampler.py checks the same
penalties through the sampler steps; these tests pin down the kernels' own argument contracts.

apply_rep_pens(in (1, V) half/float, out (1, V) float, past_ids (1, L) long, rep_p, sustain, decay):
    a token at distance d = L - i from the end (d = 1 is the most recent) has factor 1 for d <= sustain,
    1 - (d - sustain) / decay for sustain < d < sustain + decay, 0 beyond; f = max over its occurrences;
    out = (1 - f) * x + f * (x > 0 ? x / rep_p : x * rep_p). Ids outside [0, V) are ignored, bsz must be 1,
    in place allowed. Exact for f in {0, 1} up to the (fast-math) division.
apply_pres_freq_pens(in, out, past_ids, pres_p, freq_p, sustain, decay): same factors;
    out = x - freq_p * sum(factors) - pres_p * max(factors).
dry_penalty(in (bsz, V) half/float, out float, past_ids (bsz, L) long, breakers (V,) bool | None,
    workspace (>= bsz * V) int32 = -1, counters (>= bsz) int32 = -1, multiplier, base, allowed_length, range,
    max_exponent, match_cap): llama.cpp DRY over the last min(L, range) tokens (range <= 0: all): rep_limit is
    the distance to the most recent breaker; nothing applies if the window has <= allowed_length tokens or
    rep_limit < allowed_length; for every earlier position the suffix match length (capped at rep_limit, and at
    match_cap when max_exponent == 0) charges the token that followed it; a charged non-breaker token gets
    x - float(multiplier * base ** min(len - allowed_length, max_exponent or inf)) with the power in double.
    Bit-exact (one fp32 subtraction of a correctly rounded double).
"""

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


# Repetition / presence / frequency penalties

def pen_factors(past: list[int], vocab: int, sustain: int, decay: int) -> dict[int, list[float]]:
    """Per token: the factors of all its occurrences in the window"""
    out = {}
    n = len(past)
    for i, t in enumerate(past):
        if not 0 <= t < vocab:
            continue
        d = n - i
        if d <= sustain:
            f = 1.0
        elif decay > 0 and d < sustain + decay:
            f = 1.0 - (d - sustain) / decay
        else:
            continue
        out.setdefault(t, []).append(f)
    return out


def rep_pen_ref(x: np.ndarray, past, rep_p, sustain, decay) -> np.ndarray:
    x = x.astype(np.float64)
    out = x.copy()
    for t, fs in pen_factors(past, len(x), sustain, decay).items():
        f = max(fs)
        v = x[t]
        if not np.isfinite(v):
            continue
        w = v / rep_p if v > 0 else v * rep_p
        out[t] = (1 - f) * v + f * w
    return out


def pres_freq_ref(x: np.ndarray, past, pres_p, freq_p, sustain, decay) -> tuple[np.ndarray, np.ndarray]:
    """(reference, per-element tolerance)"""
    x = x.astype(np.float64)
    out = x.copy()
    tol = np.abs(x) * 2.0 ** -24
    for t, fs in pen_factors(past, len(x), sustain, decay).items():
        out[t] = x[t] - freq_p * sum(fs) - pres_p * max(fs)
        # fp32 atomic accumulation of len(fs) terms in any order, plus the factor's fp32 evaluation
        tol[t] += (len(fs) + 2) * 2.0 ** -22 * (abs(freq_p) * sum(fs) + abs(pres_p) + abs(out[t]))
    return out, tol


def _history(rng, n, vocab, alphabet = None):
    alphabet = alphabet or vocab
    return [int(rng.integers(0, alphabet)) for _ in range(n)]


PEN_RANGES = [(10 ** 8, 0), (1, 0), (5, 0), (5, 1), (5, 4), (0, 6), (3, 10), (300, 700)]


@pytest.mark.parametrize("sustain, decay", PEN_RANGES)
@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("vocab, n", [(1, 5), (17, 40), (4096, 300), (4097, 2000), (32000, 1), (151936, 5000)])
@torch.inference_mode()
def test_apply_rep_pens(vocab, n, dtype, sustain, decay, device):
    rng = np.random.default_rng(vocab * 31 + n)
    # A small alphabet makes repeats (and so max-over-occurrences) common; place tokens on both sides of every
    # 4096-token block boundary of the kernel
    past = _history(rng, n, vocab, alphabet = min(vocab, 64))
    for j, t in enumerate([4095, 4096, 8191, 8192, vocab - 1]):
        if t < vocab and j < n:
            past[-1 - j * 3 % n] = t
    rep_p = 1.3
    x = (torch.randn(1, vocab, generator = torch.Generator().manual_seed(n)) * 4).to(dtype)
    x[0, :: 13] = float("-inf")
    past_t = torch.tensor([past], dtype = torch.long, device = device)
    out = torch.full((1, vocab), float("nan"), device = device)
    ext.apply_rep_pens(x.to(device), out, past_t, rep_p, sustain, decay)
    ref = rep_pen_ref(x.float().numpy()[0], past, rep_p, sustain, decay)
    o = out.cpu().numpy()[0].astype(np.float64)
    fin = np.isfinite(ref)
    assert np.all(o[~fin] == ref[~fin]), "-inf logits must stay -inf"
    # Approximate fp32 division (fast math, <= 2 ulp) and the fp32 decay factor
    tol = np.abs(ref[fin]) * 2.0 ** -21 + 1e-30
    err = np.abs(o[fin] - ref[fin])
    bad = np.nonzero(err > tol)[0]
    assert len(bad) == 0, f"{len(bad)} tokens off, first {np.nonzero(fin)[0][bad[0]]}: {o[fin][bad[0]]} vs {ref[fin][bad[0]]}"


@torch.inference_mode()
def test_rep_pens_sustain_boundary(device):
    """The sustain_range most recent tokens get the full penalty, including the sustain_range-th one (d ==
    sustain), with or without a decay range. sustain_range = 1 penalizes the last token"""
    vocab = 16
    x = torch.full((1, vocab), 2.0, device = device)
    for sustain, decay in [(1, 0), (3, 0), (3, 2)]:
        past = list(range(8))        # token t sits at distance 8 - t
        out = torch.empty_like(x)
        ext.apply_rep_pens(x, out, torch.tensor([past], dtype = torch.long, device = device), 2.0, sustain, decay)
        o = out.cpu()[0]
        for d in range(1, sustain + 1):
            t = 8 - d
            assert o[t].item() == 1.0, f"sustain {sustain} decay {decay}: token at distance {d} got {o[t].item()}"
        out2 = torch.empty_like(x)
        ext.apply_pres_freq_pens(x, out2, torch.tensor([past], dtype = torch.long, device = device), 0.5, 0.0,
                                 sustain, decay)
        o2 = out2.cpu()[0]
        for d in range(1, sustain + 1):
            assert o2[8 - d].item() == 1.5, f"pres: sustain {sustain} decay {decay}: distance {d} got {o2[8 - d].item()}"


@pytest.mark.parametrize("sustain, decay", PEN_RANGES)
@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("vocab, n", [(1, 5), (17, 40), (4097, 2000), (151936, 5000)])
@torch.inference_mode()
def test_apply_pres_freq_pens(vocab, n, dtype, sustain, decay, device):
    rng = np.random.default_rng(vocab * 7 + n)
    past = _history(rng, n, vocab, alphabet = min(vocab, 50))
    pres_p, freq_p = 0.7, 0.15
    x = (torch.randn(1, vocab, generator = torch.Generator().manual_seed(n + 1)) * 4).to(dtype)
    x[0, :: 11] = float("-inf")
    out = torch.full((1, vocab), float("nan"), device = device)
    ext.apply_pres_freq_pens(x.to(device), out, torch.tensor([past], dtype = torch.long, device = device),
                             pres_p, freq_p, sustain, decay)
    ref, tol = pres_freq_ref(x.float().numpy()[0], past, pres_p, freq_p, sustain, decay)
    o = out.cpu().numpy()[0].astype(np.float64)
    fin = np.isfinite(ref)
    assert np.all(o[~fin] == ref[~fin])
    err = np.abs(o[fin] - ref[fin])
    bad = np.nonzero(err > tol[fin])[0]
    assert len(bad) == 0, f"{len(bad)} tokens off, first {np.nonzero(fin)[0][bad[0]]}: {o[fin][bad[0]]} vs {ref[fin][bad[0]]}"


@torch.inference_mode()
def test_penalties_in_place_empty_and_foreign_ids(device):
    vocab = 9000
    x = torch.randn(1, vocab, device = device) * 3
    # Empty history: plain copy (half input upcast exactly)
    for fn, args in [(ext.apply_rep_pens, (1.5, 100, 10)), (ext.apply_pres_freq_pens, (0.5, 0.2, 100, 10))]:
        out = torch.full_like(x, float("nan"))
        fn(x.half(), out, torch.empty((1, 0), dtype = torch.long, device = device), *args)
        assert torch.equal(out, x.half().float())
    # Ids outside the vocabulary (negative padding, >= vocab) are ignored; in place equals out of place
    past = torch.tensor([[5, -1, vocab, vocab + 100, 5, 8191, 8192, -7]], dtype = torch.long, device = device)
    for fn, args in [(ext.apply_rep_pens, (1.5, 100, 10)), (ext.apply_pres_freq_pens, (0.5, 0.2, 100, 10))]:
        a = torch.empty_like(x)
        fn(x, a, past, *args)
        b = x.clone()
        fn(b, b, past, *args)
        assert torch.equal(a, b)
        changed = set(torch.nonzero(a[0] != x[0]).view(-1).tolist())
        assert changed <= {5, 8191, 8192} and 5 in changed


@torch.inference_mode()
def test_penalties_reject(device):
    x = torch.randn(1, 100, device = device)
    past = torch.tensor([[1, 2]], dtype = torch.long, device = device)
    for fn, args in [(ext.apply_rep_pens, (1.5, 100, 0)), (ext.apply_pres_freq_pens, (0.5, 0.2, 100, 0))]:
        for bad in [
            (torch.randn(2, 100, device = device), torch.empty(2, 100, device = device), past.repeat(2, 1)),
            (x, torch.empty(1, 100, dtype = torch.half, device = device), past),
            (x, torch.empty(1, 99, device = device), past),
            (x, torch.empty_like(x), past.int()),
            (x, torch.empty_like(x), torch.tensor([[1], [2]], dtype = torch.long, device = device)),
        ]:
            with pytest.raises(RuntimeError):
                fn(*bad, *args)


# DRY

def dry_ref(
    x: np.ndarray, seq: list[int], breakers: set[int], multiplier, base, allowed, rng_, max_exp, match_cap,
) -> np.ndarray:
    """Brute-force llama.cpp DRY for one row, float32 output"""
    vocab = len(x)
    out = x.astype(np.float32).copy()
    m = len(seq) if rng_ <= 0 else min(len(seq), rng_)
    w = seq[len(seq) - m:]
    if m <= allowed:
        return out
    rep_limit = m
    for j in range(m):
        t = w[m - 1 - j]
        if 0 <= t < vocab and t in breakers:
            rep_limit = j
            break
    if rep_limit < allowed:
        return out
    best = {}
    for end in range(m - 1):                 # earlier sequence ends at w[end], followed by w[end + 1]
        length = 0
        while length <= end and w[end - length] == w[m - 1 - length]:
            length += 1
        length = min(length, rep_limit)
        if max_exp == 0:
            length = min(length, match_cap)
        tok = w[end + 1]
        if length >= allowed and 0 <= tok < vocab:
            best[tok] = max(best.get(tok, -1), length)
    for tok, length in best.items():
        if tok in breakers:
            continue
        e = length - allowed
        if max_exp > 0:
            e = min(e, max_exp)
        with np.errstate(over = "ignore"):
            pen = np.float32(np.float64(multiplier) * np.float64(base) ** np.float64(e))
            out[tok] = np.float32(out[tok] - pen)
    return out


def run_dry(x, past, breakers_mask, multiplier, base, allowed, rng_, max_exp, match_cap, out = None, ws_fill = -1):
    bsz, vocab = x.shape
    dev = x.device
    if out is None:
        out = torch.full((bsz, vocab), float("nan"), device = dev)
    scratch = torch.full((bsz * vocab + bsz,), ws_fill, dtype = torch.int32, device = dev)
    ext.dry_penalty(x, out, past, breakers_mask, scratch[:bsz * vocab], scratch[bsz * vocab:],
                    float(np.float32(multiplier)), float(np.float32(base)), allowed, rng_, max_exp, match_cap)
    return out


def _np32(v):
    return float(np.float32(v))


@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@torch.inference_mode()
def test_dry_penalty_random(dtype, device):
    """Randomized histories (small alphabets, so repeats are frequent) over the parameter space, including
    windows long enough to spread the offset scan over several kernel blocks, batches with distinct histories,
    and ids outside the vocabulary"""
    rng = np.random.default_rng(5)
    for case in range(160):
        bsz = int(rng.choice([1, 1, 3]))
        vocab = int(rng.choice([8, 33, 4099]))
        n = int(rng.integers(0, 70)) if case < 130 else int(rng.integers(1500, 5000))
        alphabet = int(rng.integers(2, 6))
        seqs = [[int(t) for t in rng.integers(0, alphabet, n)] for _ in range(bsz)]
        if case % 4 == 0 and n > 0:
            for s in seqs:
                s[int(rng.integers(0, n))] = int(rng.choice([-1, vocab, vocab + 3]))
        multiplier = _np32(rng.choice([0.5, 1.0, 2.7]))
        base = _np32(rng.choice([1.0, 1.05, 1.75, 3.0]))
        allowed = int(rng.choice([0, 1, 2, 5]))
        rng_ = int(rng.choice([0, -1, 3, 9, 100]))
        max_exp = int(rng.choice([0, 0, 3, 60]))
        match_cap = int(rng.choice([4, 7, 2048]))
        breakers = set(int(t) for t in rng.choice(alphabet + 2, int(rng.integers(0, 3)), replace = False))
        bmask = None
        if breakers:
            bmask = torch.zeros(vocab, dtype = torch.bool)
            bmask[[b for b in breakers if b < vocab]] = True
            bmask = bmask.to(device)
        x = (torch.randn(bsz, vocab, generator = torch.Generator().manual_seed(case)) * 3).to(dtype)
        past = torch.tensor(seqs, dtype = torch.long).view(bsz, n).to(device)
        out = run_dry(x.to(device), past, bmask, multiplier, base, allowed, rng_, max_exp, match_cap)
        o = out.cpu().numpy()
        for r in range(bsz):
            ref = dry_ref(x[r].float().numpy(), seqs[r], breakers if bmask is not None else set(), multiplier,
                          base, allowed, rng_, max_exp, match_cap)
            diff = np.nonzero(~((o[r] == ref) | (np.isnan(o[r]) & np.isnan(ref))))[0]
            assert len(diff) == 0, (
                f"case {case} row {r} (n={n} allowed={allowed} range={rng_} max_exp={max_exp} cap={match_cap} "
                f"breakers={sorted(breakers)}): token {diff[0]} {o[r][diff[0]]} vs {ref[diff[0]]}"
            )


@pytest.mark.parametrize("max_exp, match_cap, expect_len", [
    (0, 2048, 2048),            # no exponent clamp: the scan stops at match_cap
    (0, 50, 50),
    (3, 50, None),              # with the clamp the cap is allowed_length + max_exponent (no change to the penalty)
])
@torch.inference_mode()
def test_dry_penalty_long_repeat_caps(max_exp, match_cap, expect_len, device):
    """A context that is one long verbatim repeat (period 3): the follower token is charged with the full match
    length, capped as documented"""
    vocab = 10
    period = [1, 2, 3]
    seq = period * 1400
    base = _np32(1.001)
    allowed = 2
    x = torch.zeros(1, vocab, device = device)
    out = run_dry(x, torch.tensor([seq], dtype = torch.long, device = device), None, 1.0, base, allowed, 0,
                  max_exp, match_cap)
    ref = dry_ref(np.zeros(vocab, dtype = np.float32), seq, set(), 1.0, base, allowed, 0, max_exp, match_cap)
    assert np.array_equal(out.cpu().numpy()[0], ref)
    if expect_len is not None:
        assert ref[1] == np.float32(-np.float32(base ** (expect_len - allowed)))


@torch.inference_mode()
def test_dry_penalty_overflow_to_neg_inf(device):
    """A penalty beyond float range lands the logit at -inf"""
    seq = [4, 5] * 200
    x = torch.zeros(1, 8, device = device)
    out = run_dry(x, torch.tensor([seq], dtype = torch.long, device = device), None, 1.0, 10.0, 2, 0, 0, 2048)
    assert out[0, 4].item() == float("-inf")
    assert torch.equal(out[0, [0, 1, 2, 3, 6, 7]], x[0, [0, 1, 2, 3, 6, 7]])


@torch.inference_mode()
def test_dry_penalty_in_place_and_empty(device):
    rng = np.random.default_rng(1)
    seq = [int(t) for t in rng.integers(0, 4, 500)]
    x = torch.randn(1, 50, device = device)
    past = torch.tensor([seq], dtype = torch.long, device = device)
    a = run_dry(x, past, None, 0.8, 1.75, 2, 0, 20, 2048)
    b = x.clone()
    run_dry(b, past, None, 0.8, 1.75, 2, 0, 20, 2048, out = b)
    assert torch.equal(a, b)
    # Empty history and a history of exactly allowed_length tokens: plain copy
    for n in (0, 2):
        out = run_dry(x.half(), torch.tensor([seq[:n]], dtype = torch.long, device = device).view(1, n), None,
                      0.8, 1.75, 2, 0, 20, 2048)
        assert torch.equal(out, x.half().float())


@torch.inference_mode()
def test_dry_penalty_rejects(device):
    x = torch.zeros(2, 32, device = device)
    past = torch.zeros(2, 10, dtype = torch.long, device = device)
    ws = torch.full((64,), -1, dtype = torch.int32, device = device)
    cnt = torch.full((2,), -1, dtype = torch.int32, device = device)
    ok = dict(x = x, out = torch.empty_like(x), past = past, brk = None, ws = ws, cnt = cnt)

    def call(**kw):
        a = dict(ok, **kw)
        ext.dry_penalty(a["x"], a["out"], a["past"], a["brk"], a["ws"], a["cnt"], 1.0, 1.75, 2, 0, 0, 2048)

    call()
    for kw in [
        dict(out = torch.empty(2, 32, dtype = torch.half, device = device)),
        dict(out = torch.empty(2, 31, device = device)),
        dict(past = past.int()),
        dict(past = torch.zeros(3, 10, dtype = torch.long, device = device)),
        dict(past = torch.zeros(10, 2, dtype = torch.long, device = device).t()),
        dict(past = past.cpu()),
        dict(ws = ws[:63]),
        dict(cnt = cnt[:1]),
        dict(ws = ws.float()),
        dict(brk = torch.zeros(31, dtype = torch.bool, device = device)),
        dict(brk = torch.zeros(32, dtype = torch.uint8, device = device)),
        dict(x = x.bfloat16()),
    ]:
        with pytest.raises(RuntimeError):
            call(**kw)
