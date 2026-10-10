"""
Speculative (rejection) sampling for DFlash2 drafts (Generator._spec_verify): the emitted tokens must follow the
target sampler's distribution exactly, whatever the draft distribution q. Monte Carlo on a small vocabulary with the
real sampler stack (temperature / top-k / top-p): the first emitted token follows ordinary sampling at position 0, and
after an accepted first draft token the second follows it at position 1 (chi-square against the empirical distribution
of ordinary draws from the same sampler, not against CustomSampler.probs). Also: probs() follows the fused sampler's
tie handling at a top-k / top-p cutoff, and jobs under a logit mask (min_new_tokens, a banned-string checkpoint) stay
on match-the-sample verification.

    python -m pytest tests/test_dflash_spec_sampling.py -v
"""
import random
from types import SimpleNamespace
import pytest
import torch

from exllamav3.generator.generator import Generator, _spec_sampling_ok
from exllamav3.generator.sampler.custom import SS_Fused
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


def _ordinary(sampler, logits_row, n = 400000, chunk = 20000):
    """Empirical distribution of n ordinary draws (the job's sampling path) from one row of logits, in batches."""
    rows = logits_row.reshape(1, -1).expand(chunk, -1).contiguous()
    counts = torch.zeros(rows.shape[-1], device = rows.device)
    for _ in range(n // chunk):
        counts += torch.bincount(sampler.forward(rows.clone()).reshape(-1).long(), minlength = rows.shape[-1])
    return counts / (n // chunk * chunk)


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
    ref0 = _ordinary(sampler, logits[0, 0])
    ref1 = _ordinary(sampler, logits[0, 1])
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
    ok, chi2, dof = _chi2_ok(first, ref0, N)
    assert ok, f"first token: chi2 {chi2:.1f} dof {dof}"
    ok, chi2, dof = _chi2_ok(second, ref1, n_second)
    assert ok, f"second token: chi2 {chi2:.1f} dof {dof} (n {n_second})"


# Rows with exact ties at a cutoff: the fused sampler keeps a tie group whole at a top-k cutoff and drops it whole at a
# top-p crossing (unless it is the top group), where a sort-position truncation would split it
_TIES = [
    (dict(temperature = 1.0, top_k = 2), [1.0, 1.0, 1.0, 1.0, -0.5, -1.0, -2.0, -3.0]),
    (dict(temperature = 0.8, top_k = 3), [2.0, 0.5, 0.5, 0.5, 0.5, -1.0, -1.0, -4.0]),
    (dict(temperature = 1.0, top_p = 0.3), [0.0, 0.0, 0.0, 0.0, -1.0, -2.0, -2.0, -3.0]),
    (dict(temperature = 1.0, top_p = 0.73), [1.5, 0.0, 0.0, 0.0, -1.0, -2.0, -2.0, -3.0]),
    (dict(temperature = 1.2, top_k = 4, top_p = 0.9), [1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -3.0, -3.0]),
]


@pytest.mark.parametrize("sampler_args, row", _TIES)
def test_probs_follows_fused_ties(sampler_args, row):
    torch.manual_seed(0)
    sampler = ComboSampler(**sampler_args)
    assert sampler.spec_ok
    assert isinstance(sampler.steps[-1], SS_Fused), "the stack must collapse to the fused step for this test"
    logits = torch.tensor(row, device = DEV)
    P = sampler.probs(logits[None], None)[0]
    n = 400000
    ref = _ordinary(sampler, logits, n)
    # every token the ordinary path draws has p > 0, and the frequencies match p (4.5 sd per token)
    assert torch.all(P[ref > 0] > 0), f"probs() drops a token the sampler draws: {P.tolist()} vs {ref.tolist()}"
    sd = (P * (1 - P) / n).sqrt().clamp_min(1e-6)
    assert torch.all((ref - P).abs() <= 4.5 * sd), f"probs {P.tolist()} vs ordinary {ref.tolist()}"


def _job(**kw):
    j = SimpleNamespace(
        sampler = SimpleNamespace(spec_ok = True), new_tokens = 5, forced_ids = None, return_probs = False,
        return_top_tokens = False, min_new_tokens = 0, checkpoint = None, filters = [],
    )
    j.__dict__.update(kw)
    return j


def test_spec_eligibility_masks():
    assert _spec_sampling_ok(_job())
    # stop tokens are masked until min_new_tokens: speculative verification would let EOS through
    assert not _spec_sampling_ok(_job(min_new_tokens = 6))
    assert _spec_sampling_ok(_job(min_new_tokens = 5))
    # a banned-string checkpoint blocks its explored tokens after a rewind
    assert not _spec_sampling_ok(_job(checkpoint = {"offset": 0, "explored_tokens": [3]}))
    assert not _spec_sampling_ok(_job(filters = [SimpleNamespace(is_active = True)]))
    assert _spec_sampling_ok(_job(filters = [SimpleNamespace(is_active = False)]))


_STACKS = [
    dict(temperature = 0.7),
    dict(temperature = 1.3, min_p = 0.05),
    dict(temperature = 1.3, min_p = 0.05, temp_last = True),
    dict(temperature = 1.0, top_k = 20, top_p = 0.95),
    dict(temperature = 0.6, top_k = 40),
    dict(temperature = 1.1, top_p = 0.8, temp_last = True),
    dict(temperature = 0.9, min_p = 0.02, top_k = 50, top_p = 0.9),
]


@pytest.mark.parametrize("sampler_args", _STACKS)
def test_fused_probs_match_unfused_without_ties(sampler_args, monkeypatch):
    # Without ties every truncation is the same set either way: probs() through the fused step's thresholds must match
    # the unfused step stack (the reference before the fused path existed)
    import exllamav3.generator.sampler.custom as custom
    torch.manual_seed(1)
    logits = torch.randn((8, 4096), device = DEV) * 3.0
    fused = ComboSampler(**sampler_args)
    assert isinstance(fused.steps[-1], SS_Fused)
    monkeypatch.setattr(custom, "fused_sampler_enable", False)
    eager = ComboSampler(**sampler_args)
    assert not isinstance(eager.steps[-1], SS_Fused)
    torch.testing.assert_close(fused.probs(logits), eager.probs(logits), atol = 2e-6, rtol = 1e-4)
