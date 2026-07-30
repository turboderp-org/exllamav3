import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3 import (
    TopKSampler,
    TopPSampler,
)
import torch.testing
import random
from exllamav3.generator.sampler.custom import *
import exllamav3.generator.sampler.custom as sampler_custom
from exllamav3.generator.sampler.presets import DefaultSampler, CategoricalSampler, ComboSampler

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 150)

device = "cuda:2"
dims = [
    (1, 16),
    (9, 16),
    (1, 32768),
    (2, 128256),
    (1, 256000),
]

ni = -float("inf")

custom_test_cases = [
    {
        "name": "presfreq_p 1",
        "sampler": CustomSampler([
            SS_PresFreqP(0.5, 0.5),
            SS_Sample_mn()
        ]),
        "input": [[2] * 256000],
        "input_seq": [[0, 1000, 20000, 200000, 1000]],
        "expect_logits": [[1] + [2] * 999 + [0.5] + [2] * 18999 + [1] + [2] * 179999 + [1] + [2] * 55999],
    },
    {
        "name": "presfreq_p 2",
        "sampler": CustomSampler([
            SS_PresFreqP(1, 1),
            SS_Sample_mn()
        ]),
        "input": [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]],
        "input_seq": [[0, 0, 0, 1, 1, 1, 1, 1, 1, 9]],
        "expect_logits": [[6, 3, 10, 10, 10, 10, 10, 10, 10, 8]],
    },
    {
        "name": "presfreq_p 3",
        "sampler": CustomSampler([
            SS_PresFreqP(1, 0, 4, 4),
            SS_Sample_mn()
        ]),
        "input": [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
        "input_seq": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        "expect_logits": [[2, 2, 2, 1.75, 1.5, 1.25, 1, 1, 1, 1]],
    },
    {
        "name": "rep_p 1",
        "sampler": CustomSampler([
            SS_RepP(2),
            SS_Sample_mn()
        ]),
        "input": [[2] * 256000],
        "input_seq": [[0, 1000, 20000, 200000]],
        "expect_logits": [[1] + [2] * 999 + [1] + [2] * 18999 + [1] + [2] * 179999 + [1] + [2] * 55999],
    },
    {
        "name": "rep_p 2",
        "sampler": CustomSampler([
            SS_RepP(2, 4, 4),
            SS_Sample_mn()
        ]),
        "input": [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
        "input_seq": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        "expect_logits": [[2, 2, 2, 1.75, 1.5, 1.25, 1, 1, 1, 1]],
    },
    {
        "name": "rep_p 3",
        "sampler": CustomSampler([
            SS_RepP(2),
            SS_Sample_mn()
        ]),
        "input": [[2, 2, -2, 2, 2, 2]],
        "input_seq": [[1, 2, 3]],
        "expect_logits": [[2, 1, -4, 1, 2, 2]],
    },
    {
        "name": "temp, top_p, sample",
        "sampler": CustomSampler([
            SS_Temperature(0.75),
            SS_TopP(0.95),
            SS_Sample_mn()
        ]),
        "input": [[5, 3, 2.5, 1, 4, 2, 1.5]],
        "expect_indices": [[0, 4, 1, 2, 5, 6, 3]],
        "expect_probs": [[0.79139, 0.20861, 0, 0, 0, 0, 0]],
    },
    {
        "name": "min_p, sample",
        "sampler": CustomSampler([
            SS_MinP(0.16),
            SS_Sample_mn()
        ]),
        "input": [[3, 3.5, 4, 4.5, 5, 5.5]] * 2,
        "expect_probs": [[0, 0, 0.10154, 0.16741, 0.27600, 0.45505]] * 2,
    },
    {
        "name": "sort, min_p, sample",
        "sampler": CustomSampler([
            SS_Sort(),
            SS_MinP(0.16),
            SS_Sample_mn()
        ]),
        "input": [[3, 3.5, 4, 4.5, 5, 5.5]] * 2,
        "expect_indices": [[5, 4, 3, 2, 1, 0]] * 2,
        "expect_probs": [[0.45505, 0.27600, 0.16741, 0.10154, 0, 0]] * 2,
    },
    {
        "name": "top_k",
        "sampler": CustomSampler([
            SS_TopK(5),
        ]),
        "input": [[3.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9]] * 3,
        "expect_logits": [[3.0, 2.9, 2.8, 2.7, 2.6]] * 3,
        "expect_indices": [[0, 9, 8, 7, 6]] * 3,
    },
]


@pytest.mark.parametrize("case", custom_test_cases)
@torch.inference_mode()
def test_cases(case: dict):
    sampler = case["sampler"]
    inputs = torch.tensor(case["input"], dtype = torch.float, device = device)
    sequence_ids = torch.tensor(case["input_seq"], dtype = torch.long, device = "cpu", pin_memory = True) \
        if "input_seq" in case else None
    state = sampler.forward(
        inputs,
        rand_u32 = 0,
        return_state = True,
        sequence_ids = sequence_ids
    )

    if "expect_probs" in case:
        expect_probs = torch.tensor(case["expect_probs"], dtype = torch.float, device = device)
        test_probs = state.probs[:, :expect_probs.shape[-1]]
        torch.testing.assert_close(test_probs, expect_probs)

    if "expect_indices" in case:
        expect_indices = torch.tensor(case["expect_indices"], dtype = torch.long, device = device)
        test_indices = state.indices[:, :expect_indices.shape[-1]]
        torch.testing.assert_close(test_indices, expect_indices)

    if "expect_logits" in case:
        expect_logits = torch.tensor(case["expect_logits"], dtype = torch.float, device = device)
        test_logits = state.logits[:, :expect_logits.shape[-1]]
        torch.testing.assert_close(test_logits, expect_logits)

    if "expect_sample" in case:
        expect_sample = torch.tensor(case["expect_sample"], dtype = torch.float, device = device)
        torch.testing.assert_close(state.sample, expect_sample)


def compare(histogram, true_dist, min_p = 0.00001):
    observed_counts = histogram.clamp(min = min_p)
    expected_counts = true_dist.clamp(min = min_p)
    chisq = ((observed_counts - expected_counts).square() / expected_counts).sum(dim = -1, keepdim = True)
    # print(f"chi_squared: {chisq}")
    return chisq.max().item()


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("k", [1, 24, 8, 32, 50])
# @pytest.mark.parametrize("k", [1])
@torch.inference_mode()
def test_topk(dim: tuple, k):
    torch.manual_seed(0)
    random.seed(0)
    temperature = 0.8
    if k > dim[-1]:
        return

    logits = torch.randn(dim, dtype = torch.half, device = device) * 2

    # Reference. Tokens tied exactly at the k-th value are excluded from the comparison: the
    # fused sampler keeps all ties, torch.topk (and the eager sort path) truncates them in
    # arbitrary sort order, and both are valid for indistinguishable tokens
    logits_ref = logits.float() / temperature
    probs_ref = torch.softmax(logits_ref, dim = -1)
    topk_values, topk_indices = torch.topk(probs_ref, k, dim = -1)
    tie_mask = probs_ref == topk_values[..., -1:]
    mask = probs_ref >= topk_values[..., -1:]
    probs_ref = probs_ref.masked_fill(~mask, 0)
    probs_ref /= probs_ref.sum(dim = -1, keepdim = True)

    sampler = TopKSampler(top_k = k, temperature = temperature)

    num_samples = min(dim[-1] * 200, 10000)
    samples = torch.empty((dim[0], 0), dtype = torch.long, device = device)
    for _ in range(num_samples):
        sample = sampler.forward(logits).unsqueeze(-1)
        samples = torch.cat((samples, sample), dim = -1)

    hb = [torch.bincount(samples[b], minlength = dim[1]) for b in range(dim[0])]
    histogram = torch.stack(hb).float()
    histogram /= num_samples

    probs_ref = torch.where(tie_mask, histogram, probs_ref)
    chisq = compare(histogram, probs_ref)
    assert chisq < 0.01


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("p", [0.1, 0.45, 0.50])
@torch.inference_mode()
def test_topp(dim: tuple, p):
    torch.manual_seed(0)
    random.seed(0)
    temperature = 0.6

    logits = torch.randn(dim, dtype = torch.half, device = device) * 2

    # Reference
    logits_ref = logits.float() / temperature
    probs_ref = torch.softmax(logits_ref, dim = -1)
    sorted_values, sorted_indices = torch.sort(probs_ref, descending = True, dim = 1)
    cumsum = sorted_values.cumsum(dim = -1)
    mask = cumsum <= p
    mask[:, 0] = True
    sorted_values *= mask
    probs_ref.scatter_(1, sorted_indices, sorted_values)
    probs_ref /= probs_ref.sum(dim = -1, keepdim = True)

    sampler = TopPSampler(top_p = p, temperature = temperature)

    num_samples = min(dim[-1] * 200, 20000)
    samples = torch.empty((dim[0], 0), dtype = torch.long, device = device)
    for _ in range(num_samples):
        sample = sampler.forward(logits).unsqueeze(-1)
        samples = torch.cat((samples, sample), dim = -1)

    hb = [torch.bincount(samples[b], minlength = dim[1]) for b in range(dim[0])]
    histogram = torch.stack(hb).float()
    histogram /= num_samples

    chisq = compare(histogram, probs_ref)
    assert chisq < 0.02



# The statistical tests above and below are only meaningful if the chains they build actually
# collapse to the fused kernel path; pin the collapse here so a matcher regression can't
# silently divert everything to the eager path and make them vacuous. Reference nodes
# (SS_Sample_mn) and non-canonical stacks must keep using the step-by-step path.

def fused_mode(sampler):
    fs = [s for s in sampler.steps if isinstance(s, SS_Fused)]
    return fs[0].mode if fs else None


@torch.inference_mode()
def test_fused_collapse():
    if not sampler_custom.fused_sampler_enable:
        pytest.skip("fused sampler disabled by env")
    assert fused_mode(TopKSampler(1, 0.8)) == SS_Fused.MODE_GREEDY
    assert fused_mode(CustomSampler([SS_Argmax()])) == SS_Fused.MODE_GREEDY
    assert fused_mode(CategoricalSampler(0.7)) == SS_Fused.MODE_SAMPLE
    assert fused_mode(DefaultSampler()) == SS_Fused.MODE_SAMPLE_MINP
    assert fused_mode(CustomSampler([SS_Temperature(0.8), SS_MinP(0.1), SS_Sample()])) == SS_Fused.MODE_SAMPLE_MINP
    assert fused_mode(TopKSampler(50, 0.8)) == SS_Fused.MODE_SAMPLE_FILTERS
    assert fused_mode(TopPSampler(0.9, 0.8)) == SS_Fused.MODE_SAMPLE_FILTERS
    assert fused_mode(TopPSampler(0.9, 0.8, temperature_last = True)) == SS_Fused.MODE_SAMPLE_FILTERS
    assert fused_mode(ComboSampler(temperature = 0.8, min_p = 0.05, top_k = 50, top_p = 0.9)) == SS_Fused.MODE_SAMPLE_FILTERS
    # Leading penalties keep a fused tail
    assert fused_mode(ComboSampler(rep_p = 1.2, temperature = 0.8, min_p = 0.05)) == SS_Fused.MODE_SAMPLE_MINP
    # Reference/multinomial nodes and non-canonical filter orders stay on the eager path
    assert fused_mode(CustomSampler([SS_MinP(0.1), SS_Sample_mn()])) is None
    assert fused_mode(CustomSampler([SS_TopP(0.9), SS_MinP(0.1), SS_Temperature(0.8), SS_Sample()])) is None


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("min_p", [0.05, 0.25])
@pytest.mark.parametrize("temp_first", [False, True])
@torch.inference_mode()
def test_minp(dim: tuple, min_p, temp_first):
    torch.manual_seed(0)
    random.seed(0)
    temperature = 0.8

    logits = torch.randn(dim, dtype = torch.half, device = device) * 2

    # Reference: min-P thresholds the tempered distribution when temperature comes first,
    # the untempered one when it comes last
    logits_f = logits.float()
    if temp_first:
        probs_ref = torch.softmax(logits_f / temperature, dim = -1)
        mask = probs_ref >= probs_ref.amax(dim = -1, keepdim = True) * min_p
        probs_ref = probs_ref.masked_fill(~mask, 0)
    else:
        probs_pre = torch.softmax(logits_f, dim = -1)
        mask = probs_pre >= probs_pre.amax(dim = -1, keepdim = True) * min_p
        probs_ref = torch.softmax(logits_f / temperature, dim = -1).masked_fill(~mask, 0)
    probs_ref /= probs_ref.sum(dim = -1, keepdim = True)

    if temp_first:
        sampler = CustomSampler([SS_Temperature(temperature), SS_MinP(min_p), SS_Sample()])
    else:
        sampler = CustomSampler([SS_MinP(min_p), SS_Temperature(temperature), SS_Sample()])
    if sampler_custom.fused_sampler_enable:
        assert fused_mode(sampler) == SS_Fused.MODE_SAMPLE_MINP

    num_samples = min(dim[-1] * 200, 10000)
    samples = torch.empty((dim[0], 0), dtype = torch.long, device = device)
    for _ in range(num_samples):
        sample = sampler.forward(logits).unsqueeze(-1)
        samples = torch.cat((samples, sample), dim = -1)

    hb = [torch.bincount(samples[b], minlength = dim[1]) for b in range(dim[0])]
    histogram = torch.stack(hb).float()
    histogram /= num_samples

    # The chi-square statistic grows with the number of unclamped cells (~k_eff / n even for a
    # perfect sampler), so the bound scales with the effective support
    k_eff = (probs_ref > 1e-5).sum(dim = -1).max().item()
    chisq = compare(histogram, probs_ref)
    assert chisq < max(0.01, 3.0 * k_eff / num_samples)


@pytest.mark.parametrize("dim", dims)
@torch.inference_mode()
def test_gumbel(dim: tuple):
    torch.manual_seed(0)
    random.seed(0)
    temperature = 0.7

    logits = torch.randn(dim, dtype = torch.half, device = device) * 2
    probs_ref = torch.softmax(logits.float() / temperature, dim = -1)

    sampler = CategoricalSampler(temperature)
    if sampler_custom.fused_sampler_enable:
        assert fused_mode(sampler) == SS_Fused.MODE_SAMPLE

    num_samples = min(dim[-1] * 200, 10000)
    samples = torch.empty((dim[0], 0), dtype = torch.long, device = device)
    for _ in range(num_samples):
        sample = sampler.forward(logits).unsqueeze(-1)
        samples = torch.cat((samples, sample), dim = -1)

    hb = [torch.bincount(samples[b], minlength = dim[1]) for b in range(dim[0])]
    histogram = torch.stack(hb).float()
    histogram /= num_samples

    # Untruncated sampling has a large effective support on big vocabularies; see test_minp
    k_eff = (probs_ref > 1e-5).sum(dim = -1).max().item()
    chisq = compare(histogram, probs_ref)
    assert chisq < max(0.01, 3.0 * k_eff / num_samples)


@pytest.mark.parametrize("dim", dims)
@torch.inference_mode()
def test_fused_eager_parity(dim: tuple):
    """
    For temperature/min-P chains the collapsed path draws the same Gumbel noise per token as
    the step-by-step path (Philox keyed on (rand_u32, flat index)), so both must pick the same
    token for the same seed, modulo float rounding at exact ties.
    """
    torch.manual_seed(0)
    random.seed(0)
    logits = torch.randn(dim, dtype = torch.half, device = device) * 2

    def build():
        return CustomSampler([SS_MinP(0.08), SS_Temperature(0.8), SS_Sample()])

    enabled = sampler_custom.fused_sampler_enable
    try:
        sampler_custom.fused_sampler_enable = True
        fused = build()
        sampler_custom.fused_sampler_enable = False
        eager = build()
    finally:
        sampler_custom.fused_sampler_enable = enabled
    assert fused_mode(fused) == SS_Fused.MODE_SAMPLE_MINP
    assert fused_mode(eager) is None

    mismatches = 0
    for seed in range(100):
        a = fused.forward(logits.clone(), rand_u32 = seed)
        b = eager.forward(logits.clone(), rand_u32 = seed)
        if not torch.equal(a, b):
            mismatches += 1
    assert mismatches == 0
