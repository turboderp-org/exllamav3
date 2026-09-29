"""
Speculative (rejection) sampling for DFlash2 drafts (Generator._spec_verify): the emitted tokens must follow the
target sampler's distribution exactly, whatever the draft distribution q. Monte Carlo on a small vocabulary with the
real sampler stack (temperature / top-k / top-p): the first emitted token matches p_0, and after an accepted first
draft token the second matches p_1 (chi-square).

    python -m pytest tests/test_dflash_spec_sampling.py -v
"""
import random
import pytest
import torch

from exllamav3.generator.generator import Generator
from exllamav3.generator.sampler.presets import ComboSampler

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEV = "cuda:0"


class _Job:
    def __init__(self, sampler, seed):
        self.sampler = sampler
        self.rng = random.Random(seed)


class _Gen:
    tokenizer = None
    _spec_verify = Generator._spec_verify


def _chi2_ok(counts, probs, n):
    exp = probs * n
    keep = exp > 5
    chi2 = (((counts - exp) ** 2) / exp.clamp_min(1e-9))[keep].sum().item()
    dof = max(int(keep.sum().item()) - 1, 1)
    # generous bound: mean + 6 sd of chi-square(dof)
    return chi2 < dof + 6 * (2 * dof) ** 0.5, chi2, dof


@pytest.mark.parametrize("sampler_args", [dict(temperature = 1.0, top_k = 6, top_p = 0.95), dict(temperature = 0.7), dict(temperature = 1.3, top_p = 0.8)])
def test_spec_verify_preserves_distribution(sampler_args):
    torch.manual_seed(0)
    V, L, K, N = 12, 3, 4, 40000
    sampler = ComboSampler(**sampler_args)
    assert sampler.spec_ok
    logits = torch.randn((1, L + 1, V), device = DEV) * 1.5
    P = sampler.probs(logits[0], None)
    # draft: fixed candidate sets and proposal distributions per position (deliberately unlike P)
    cands = torch.stack([torch.randperm(V, device = DEV)[:K] for _ in range(L)])[None]
    q = torch.softmax(torch.randn((1, L, K), device = DEV) * 2.0, dim = -1)
    job = _Job(sampler, 1)
    gen = _Gen()
    first = torch.zeros(V, device = DEV)
    second = torch.zeros(V, device = DEV)
    n_second = 0
    x_all = torch.multinomial(q[0], N, replacement = True).T      # (N, L) candidate indices
    for t in range(N):
        ids = cands[0, torch.arange(L, device = DEV), x_all[t]][None]
        seq = gen._spec_verify(job, logits, {"ids": ids, "q": q, "cands": cands})
        first[seq[0]] += 1
        if seq[0] == ids[0, 0].item():      # first draft token accepted: next emitted token ~ p_1
            second[seq[1]] += 1
            n_second += 1
    ok, chi2, dof = _chi2_ok(first, P[0], N)
    assert ok, f"first token: chi2 {chi2:.1f} dof {dof}"
    ok, chi2, dof = _chi2_ok(second, P[1], n_second)
    assert ok, f"second token: chi2 {chi2:.1f} dof {dof} (n {n_second})"
