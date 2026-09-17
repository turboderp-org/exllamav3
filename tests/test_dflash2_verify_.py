"""
Tests for the DFlash2 q-aware verify path in the generator: sampler
introspection, target-distribution construction, and the rejection sampler.
Only the reference-comparison test needs the optional dflash package; the
rest run in the default suite.
"""
from types import SimpleNamespace

import pytest
import torch

try:
    from dflash.model import _sampling_probs as ref_sampling_probs
    has_ref = True
except ImportError:
    has_ref = False

requires_ref = pytest.mark.skipif(
    not has_ref,
    reason = "reference dflash package not installed",
)

from exllamav3.generator.generator import (  # noqa: E402
    _dflash2_sampler_params,
    _dflash2_sampling_dist,
    Generator,
)


# ---------------------------------------------------------------------------
# _dflash2_sampler_params
# ---------------------------------------------------------------------------

def test_sampler_params_default_sampler_ineligible_min_p():
    # DefaultSampler is [MinP(0.08), Temperature(0.8), Sample] — min-p excludes it
    from exllamav3.generator.sampler.presets import DefaultSampler
    assert _dflash2_sampler_params(DefaultSampler()) is None


def test_sampler_params_plain_samplers_eligible():
    from exllamav3.generator.sampler.presets import CategoricalSampler, TopPSampler
    ps = _dflash2_sampler_params(CategoricalSampler(temperature = 1.0))
    assert ps == (False, 1.0, 0, 1.0)
    ps = _dflash2_sampler_params(TopPSampler(top_p = 0.9, temperature = 0.7))
    assert ps == (False, 0.7, 0, 0.9)


def test_sampler_params_plain_combo_eligible():
    from exllamav3.generator.sampler.presets import ComboSampler
    sampler = ComboSampler(temperature = 1.0, top_p = 0.95, top_k = 20)
    ps = _dflash2_sampler_params(sampler)
    assert ps == (False, 1.0, 20, 0.95)


def test_sampler_params_greedy_combo():
    from exllamav3.generator.sampler.presets import ComboSampler
    sampler = ComboSampler(temperature = 0.0)
    ps = _dflash2_sampler_params(sampler)
    assert ps is not None and ps[0] is True


def test_sampler_params_penalties_ineligible():
    from exllamav3.generator.sampler.presets import ComboSampler
    sampler = ComboSampler(temperature = 1.0, rep_p = 1.5)
    assert _dflash2_sampler_params(sampler) is None


def test_sampler_params_min_p_ineligible():
    from exllamav3.generator.sampler.presets import ComboSampler
    sampler = ComboSampler(temperature = 1.0, min_p = 0.08)
    assert _dflash2_sampler_params(sampler) is None


# ---------------------------------------------------------------------------
# _dflash2_sampling_dist vs the reference transform
# ---------------------------------------------------------------------------

SAMPLING_GRID = [
    (1.0, 0, 1.0),
    (0.7, 0, 1.0),
    (1.0, 20, 1.0),
    (1.0, 0, 0.95),
    (1.0, 20, 0.95),
    (0.85, 50, 0.9),
]


@pytest.mark.parametrize("temperature,top_k,top_p", SAMPLING_GRID)
@requires_ref
def test_sampling_dist_matches_reference(temperature, top_k, top_p):
    torch.manual_seed(11)
    logits = torch.randn(1, 1, 500)
    ours = _dflash2_sampling_dist(logits, temperature, top_k, top_p)
    ref = ref_sampling_probs(logits, temperature, top_p, top_k)
    assert torch.allclose(ours, ref, atol = 1e-6, rtol = 1e-5)


# ---------------------------------------------------------------------------
# _dflash2_accept
# ---------------------------------------------------------------------------

def _stub_gen(candidates, q, sampling):
    g = SimpleNamespace()
    g._dflash2_propose = {"candidates": candidates, "q": q}
    g._dflash2_sampling = sampling  # (greedy, temperature, top_k, top_p)
    return g


def _fixed_case(vocab=64, K=4, W=3):
    torch.manual_seed(21)
    # draft tokens row 0: positions 0..W-1
    draft_tokens = torch.randint(0, vocab, (1, W))
    # candidate lists: ensure position 0 includes the draft token at index 1
    candidates = torch.stack([
        torch.randperm(vocab)[:K] for _ in range(W)
    ]).unsqueeze(0)
    candidates[0, 0, 1] = draft_tokens[0, 0]
    q = torch.rand(1, W, K)
    q = q / q.sum(-1, keepdim = True)
    logits = torch.randn(1, 1, vocab)
    return draft_tokens, candidates, q, logits


def test_accept_always_when_p_covers_draft_token():
    # p puts all mass on the draft token at position 0 -> u*q_d < 1 for every u -> accept
    draft_tokens, candidates, q, _ = _fixed_case()
    logits = torch.full((1, 1, 64), -30.0)
    logits[0, 0, draft_tokens[0, 0]] = 30.0
    g = _stub_gen(candidates, q, (False, 1.0, 0, 1.0))
    for _ in range(20):
        accepted, bonus = Generator._dflash2_accept(g, None, 0, 0, logits, draft_tokens)
        assert accepted and bonus is None


def test_impossible_event_q_zero_p_positive_always_accepts():
    # q_d = 0 (draft token not among the selector's candidates — impossible under a
    # correct walk) and p_d > 0: the acceptance test u*q_d < p_d degenerates to
    # accept, matching the reference _rejection_sample for this impossible event.
    draft_tokens, candidates, q, _ = _fixed_case()
    candidates[0, 0, :] = torch.arange(40, 44)
    draft_tokens[0, 0] = 7
    logits = torch.randn(1, 1, 64)
    g = _stub_gen(candidates, q, (False, 1.0, 0, 1.0))
    for _ in range(20):
        accepted, bonus = Generator._dflash2_accept(g, None, 0, 0, logits, draft_tokens)
        assert accepted


def test_reject_when_target_assigns_zero_to_draft_token():
    # p_d = 0 via top-k truncation (draft token outside the kept set) -> always
    # reject; the bonus comes from p itself
    draft_tokens, candidates, q, _ = _fixed_case()
    candidates[0, 0, :] = torch.arange(0, 4)
    draft_tokens[0, 0] = 7
    q[0, 0, :] = 0.25
    logits = torch.full((1, 1, 64), 20.0)
    logits[0, 0, 7] = -20.0
    g = _stub_gen(candidates, q, (False, 1.0, 4, 1.0))  # top-4 keeps tokens 60..63-ish, excludes 7
    torch.manual_seed(9)
    for _ in range(20):
        accepted, bonus = Generator._dflash2_accept(g, None, 0, 0, logits, draft_tokens)
        assert not accepted
        assert int(bonus) != 7


def test_acceptance_rate_matches_p_over_q():
    # q puts 0.8 on the draft token, p puts 0.4 -> acceptance rate ~ 0.5
    draft_tokens, candidates, q, _ = _fixed_case()
    K = candidates.shape[-1]
    q[0, 0, :] = 0.05
    q[0, 0, 1] = 0.85  # index 1 is the draft token at position 0
    logits = torch.full((1, 1, 64), -12.0)
    logits[0, 0, draft_tokens[0, 0]] = 0.0  # softmax mass ~ 0.42 on the draft token
    p = torch.softmax(logits[0, 0], dim = -1)
    ratio = float(p[draft_tokens[0, 0]] / 0.85)
    g = _stub_gen(candidates, q, (False, 1.0, 0, 1.0))
    torch.manual_seed(5)
    accepts = sum(
        Generator._dflash2_accept(g, None, 0, 0, logits, draft_tokens)[0]
        for _ in range(400)
    )
    assert 0.65 * ratio * 400 < accepts < 1.35 * ratio * 400


def test_bonus_comes_from_residual_not_from_q_mass():
    # p uniform-ish, q concentrated on candidates that p discounts -> bonus must
    # not be any candidate with q mass above its p mass
    draft_tokens, candidates, q, _ = _fixed_case()
    q[0, 0, :] = 0.25  # uniform over 4 candidates
    # p heavily favors a token OUTSIDE the candidate list
    logits = torch.full((1, 1, 64), -20.0)
    outside = int(candidates[0, 0, 0])
    while outside in candidates[0, 0].tolist():
        outside = (outside + 1) % 64
    logits[0, 0, outside] = 20.0
    g = _stub_gen(candidates, q, (False, 1.0, 0, 1.0))
    torch.manual_seed(6)
    seen = set()
    for _ in range(50):
        accepted, bonus = Generator._dflash2_accept(g, None, 0, 0, logits, draft_tokens)
        assert not accepted
        seen.add(int(bonus))
    assert seen == {outside}


# ---------------------------------------------------------------------------
# scalar token= fast path agrees with indexing the full distribution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("temperature,top_k,top_p", SAMPLING_GRID)
def test_scalar_token_path_matches_full_distribution(temperature, top_k, top_p):
    torch.manual_seed(31)
    logits = torch.randn(1, 1, 500)
    full = _dflash2_sampling_dist(logits, temperature, top_k, top_p)
    for token in (0, 7, 123, 499):
        fast = _dflash2_sampling_dist(logits, temperature, top_k, top_p, token = token)
        assert fast.shape == ()
        assert torch.allclose(fast, full[0, 0, token], atol = 1e-6, rtol = 1e-5)


# ---------------------------------------------------------------------------
# _dflash2_job_is_cfg and _dflash2_check_page_budget (setup guards)
# ---------------------------------------------------------------------------

def _stub_job(n_sequences, cfg_scale = "absent"):
    job = SimpleNamespace(sequences = [object()] * n_sequences)
    if cfg_scale != "absent":
        job.gen_settings = SimpleNamespace(cfg_scale = cfg_scale)
    return job


def test_job_is_cfg_two_sequences():
    from exllamav3.generator.generator import _dflash2_job_is_cfg
    assert _dflash2_job_is_cfg(_stub_job(2)) is True


def test_job_is_cfg_live_scale_on_one_sequence():
    # cfg_scale exists only as a commented TODO today; when someone turns it
    # on, CFG-on-one-sequence must still be caught.
    from exllamav3.generator.generator import _dflash2_job_is_cfg
    assert _dflash2_job_is_cfg(_stub_job(1, cfg_scale = 1.2)) is True


def test_job_is_cfg_plain_job():
    from exllamav3.generator.generator import _dflash2_job_is_cfg
    assert _dflash2_job_is_cfg(_stub_job(1)) is False
    assert _dflash2_job_is_cfg(_stub_job(1, cfg_scale = None)) is False
    assert _dflash2_job_is_cfg(_stub_job(1, cfg_scale = 1.0)) is False


def test_page_budget_rejects_short_cache():
    from exllamav3.generator.generator import _dflash2_check_page_budget
    with pytest.raises(ValueError, match = "holds 7 tokens.*needs 8 rows"):
        _dflash2_check_page_budget(7, 4, 8)


def test_page_budget_rejects_zero_pages():
    from exllamav3.generator.generator import _dflash2_check_page_budget
    with pytest.raises(ValueError, match = "pages=0 cannot hold block_size=8"):
        _dflash2_check_page_budget(64, 0, 8)


def test_page_budget_passes():
    from exllamav3.generator.generator import _dflash2_check_page_budget
    _dflash2_check_page_budget(8192, 32, 8)
