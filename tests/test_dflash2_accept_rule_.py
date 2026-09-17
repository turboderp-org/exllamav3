"""
Losslessness of the q-aware DFlash2 rejection rule, on synthetic p/q with no
model. Runs _dflash2_accept_step (the exact shipped core) thousands of times
per config and asserts the output token distribution matches the target p
(chi-square/df within 3 sigma of 1). CPU only, seeded, CI-safe.

Claim scope: lossless w.r.t. the quantized target, for eligible (plain
temp/top-k/top-p) samplers only.
"""
import math

import pytest
import torch

from exllamav3.generator.generator import _dflash2_accept_step, _dflash2_sampling_dist

V = 257  # small, odd, non-multiple-of-anything vocab
K = 7    # candidate list length (matches serve window)
N = 30000
TEMP, TOP_K, TOP_P = 0.8, 32, 0.9


def _trial_inputs(rng, mode):
    logits = torch.randn(V, generator = rng) * 2.0
    p = _dflash2_sampling_dist(
        logits.view(1, 1, V), TEMP, TOP_K, TOP_P).flatten()
    perm = torch.randperm(V, generator = rng)
    if mode == "matched":
        # candidates carry the top-p mass: high accept rate
        cands = torch.topk(p, K).indices
        q = p[cands] / p[cands].sum()
    elif mode == "flat":
        cands = perm[:K]
        q = torch.full((K,), 1.0 / K)
    elif mode == "peaked_low":
        # q peaked on a low-p token: mass rejection + residual path
        cands = perm[:K]
        q = torch.full((K,), 0.02)
        q[0] = 1.0 - 0.02 * (K - 1)
        # force candidates onto low-p tokens
        cands = torch.topk(-p, K).indices
    elif mode == "near_disjoint":
        # candidates avoid p's mass: worst case for the residual sampler
        cands = torch.topk(-p, K).indices
        q = torch.softmax(torch.randn(K, generator = rng), dim = 0)
    else:
        raise AssertionError(mode)
    d = cands[torch.multinomial(q, 1, generator = rng)[0]].item()
    return logits.view(1, 1, V), cands, q, d, p


def _chi2_over_df(counts, expected):
    structural = expected <= 0  # p == 0 outside top-k: unsampleable, skip
    assert (counts[structural] == 0).all()
    counts, expected = counts[~structural], expected[~structural]
    keep = expected >= 5
    o = [counts[keep], counts[~keep].sum().reshape(1)]
    e = [expected[keep], expected[~keep].sum().reshape(1)]
    if e[1].item() == 0:  # nothing pooled: drop the empty tail bin
        o, e = o[:1], e[:1]
    o, e = torch.cat(o), torch.cat(e)
    chi2 = ((o - e) ** 2 / e).sum().item()
    return chi2 / (o.numel() - 1)


@pytest.mark.parametrize("mode", ["matched", "flat", "peaked_low", "near_disjoint"])
def test_accept_step_output_matches_p(mode):
    rng = torch.Generator().manual_seed(1234 + len(mode))
    torch.manual_seed(999)
    counts = torch.zeros(V, dtype = torch.float64)
    p_sum = torch.zeros(V, dtype = torch.float64)
    for _ in range(N):
        logits, cands, q, d, p = _trial_inputs(rng, mode)
        acc, bonus = _dflash2_accept_step(
            logits, cands, q, d, TEMP, TOP_K, TOP_P)
        counts[bonus.item() if not acc else d] += 1
        p_sum += p.double()
    expected = p_sum / p_sum.sum() * N
    ratio = _chi2_over_df(counts, expected)
    # E[ratio] == 1, Var == 2/df; df ~ 200 here, so +-0.3 is ~3 sigma
    assert 0.7 < ratio < 1.3, f"{mode}: chi2/df = {ratio:.3f}"


def test_method_delegates_to_step():
    # The Generator method must be a thin wrapper: identical outputs for
    # identical RNG state on synthetic inputs.
    import types
    from exllamav3.generator.generator import Generator
    rng = torch.Generator().manual_seed(7)
    logits, cands, q, d, _ = _trial_inputs(rng, "flat")
    logits_b = logits.clone()
    fake = types.SimpleNamespace(
        _dflash2_propose = {"candidates": cands.view(1, 1, K),
                            "q": q.view(1, 1, K)},
        _dflash2_sampling = (False, TEMP, TOP_K, TOP_P),
    )
    torch.manual_seed(31337)
    r1 = _dflash2_accept_step(logits, cands, q, d, TEMP, TOP_K, TOP_P)
    torch.manual_seed(31337)
    r2 = Generator._dflash2_accept(
        fake, None, 0, 0,
        logits_b, torch.tensor([[[d]]]))
    assert r1[0] == r2[0]
    if not r1[0]:
        assert r1[1].item() == r2[1].item()
