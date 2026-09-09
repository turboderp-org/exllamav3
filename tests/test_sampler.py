import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np
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
        # OAI/vLLM semantics: negative penalties reward reuse (presence used to be silently dropped)
        "name": "presfreq_p negative presence",
        "sampler": CustomSampler([
            SS_PresFreqP(-1, 0),
            SS_Sample_mn()
        ]),
        "input": [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]],
        "input_seq": [[0, 0, 0, 1, 1, 1, 1, 1, 1, 9]],
        "expect_logits": [[11, 11, 10, 10, 10, 10, 10, 10, 10, 11]],
    },
    {
        "name": "presfreq_p negative frequency",
        "sampler": CustomSampler([
            SS_PresFreqP(0, -1),
            SS_Sample_mn()
        ]),
        "input": [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]],
        "input_seq": [[0, 0, 0, 1, 1, 1, 1, 1, 1, 9]],
        "expect_logits": [[13, 16, 10, 10, 10, 10, 10, 10, 10, 11]],
    },
    {
        "name": "presfreq_p negative presence with decay",
        "sampler": CustomSampler([
            SS_PresFreqP(-1, 0, 4, 4),
            SS_Sample_mn()
        ]),
        "input": [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
        "input_seq": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        "expect_logits": [[2, 2, 2, 2.25, 2.5, 2.75, 3, 3, 3, 3]],
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
    {
        "name": "ban_tokens",
        "sampler": CustomSampler([
            SS_BanTokens([1, 3])
        ]),
        "input": [[1.0, 2.0, 3.0, 4.0, 5.0]] * 2,
        "expect_logits": [[1.0, ni, 3.0, ni, 5.0]] * 2,
    },
    {
        "name": "ban_tokens, sorted",
        "sampler": CustomSampler([
            SS_Sort(),
            SS_BanTokens([0, 2])
        ]),
        "input": [[5.0, 3.0, 4.0, 1.0]],
        "expect_indices": [[0, 2, 1, 3]],
        "expect_logits": [[ni, ni, 3.0, 1.0]],
    },
    {
        "name": "ban_tokens, after temperature",
        "sampler": CustomSampler([
            SS_Temperature(2.0),
            SS_BanTokens([1])
        ]),
        "input": [[2.0, 4.0, 6.0, 8.0]],
        "expect_logits": [[1.0, ni, 3.0, 4.0]],
    },
    {
        "name": "ban_tokens, normalized",
        "sampler": CustomSampler([
            SS_Normalize(),
            SS_BanTokens([0])
        ]),
        "input": [[2.0, 1.0, 0.0, -1.0]],
        "expect_probs": [[0.0, 0.236883, 0.087144, 0.032059]],
    },
    {
        # Token IDs past the end of the vocabulary are ignored
        "name": "ban_tokens, normalized and sorted",
        "sampler": CustomSampler([
            SS_Normalize(),
            SS_Sort(),
            SS_BanTokens([1, 999])
        ]),
        "input": [[2.0, 1.0, 0.0, -1.0]],
        "expect_indices": [[0, 1, 2, 3]],
        "expect_probs": [[0.643914, 0.0, 0.087144, 0.032059]],
    },
    {
        # Two tokens over the threshold, so only the more likely one is excluded
        "name": "xtc",
        "sampler": CustomSampler([
            SS_XTC(1.0, 0.1)
        ]),
        "input": [[2.0, 1.0, 0.0, -1.0]] * 2,
        "expect_probs": [[0.0, 0.236883, 0.087144, 0.032059]] * 2,
    },
    {
        # Only one token over the threshold, leaving nothing to choose between
        "name": "xtc, single candidate",
        "sampler": CustomSampler([
            SS_XTC(1.0, 0.3)
        ]),
        "input": [[2.0, 1.0, 0.0, -1.0]],
        "expect_probs": [[0.643914, 0.236883, 0.087144, 0.032059]],
    },
    {
        # Three tokens over the threshold, of which the two more likely are excluded
        "name": "xtc, unprotected",
        "sampler": CustomSampler([
            SS_XTC(1.0, 0.1)
        ]),
        "input": [[2.0, 1.0, 0.5, -1.0]],
        "expect_probs": [[0.0, 0.0, 0.135989, 0.030343]],
    },
    {
        # Same distribution with the least likely of the three protected, so it neither is excluded
        # nor counts as the one to keep, and the token above it survives in its place
        "name": "xtc, protected token",
        "sampler": CustomSampler([
            SS_XTC(1.0, 0.1, protected_token_ids = [2])
        ]),
        "input": [[2.0, 1.0, 0.5, -1.0]],
        "expect_probs": [[0.0, 0.224208, 0.135989, 0.030343]],
    },
    {
        # Reweighting for a probability below 1, leaving the result unnormalized
        "name": "xtc, partial probability",
        "sampler": CustomSampler([
            SS_XTC(0.5, 0.1)
        ]),
        "input": [[2.0, 1.0, 0.0, -1.0]],
        "expect_probs": [[0.169081, 0.236883, 0.087144, 0.032059]],
    },
    {
        # logit_bias adds a per-token value to the logits; out-of-range IDs are ignored and the
        # two batch rows share the same bias vector
        "name": "logit_bias",
        "sampler": CustomSampler([
            SS_LogitBias({1: 5.0, 3: -2.0, 100: 9.0}),
            SS_Sample_mn()
        ]),
        "input": [[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
        "expect_logits": [[1.0, 6.0, 1.0, -1.0, 1.0], [0.0, 5.0, 0.0, -2.0, 0.0]],
    },
    {
        # logit_bias with -inf bans a token, matching a hard mask
        "name": "logit_bias ban",
        "sampler": CustomSampler([
            SS_LogitBias({2: -float("inf")})
        ]),
        "input": [[1.0, 2.0, 3.0, 4.0]],
        "expect_logits": [[1.0, 2.0, ni, 4.0]],
    },
    {
        # logit_bias after another logits step exercises the SS.LOGITS branch; zero and
        # non-finite bias values are dropped, so token 0 (nan) and token 3 (0.0) are unchanged
        "name": "logit_bias, after ban_tokens",
        "sampler": CustomSampler([
            SS_BanTokens([2]),
            SS_LogitBias({0: float("nan"), 1: 4.0, 3: float("inf")})
        ]),
        "input": [[1.0, 1.0, 1.0, 1.0]],
        # nan and +inf biases are dropped (tokens 0 and 3 unchanged); 4.0 adds to token 1
        "expect_logits": [[1.0, 5.0, ni, 1.0]],
    },
    {
        # DRY: the context suffix [1, 2, 3] repeats earlier, so token 1, which would extend that
        # repeat to length 4, is penalized by multiplier * base ** (3 - allowed_length) = 1.75
        "name": "dry",
        "sampler": CustomSampler([
            SS_DRY(1.0, 1.75, 2, 0, []),
            SS_Sample_mn()
        ]),
        "input": [[2] * 8],
        "input_seq": [[0, 1, 2, 3, 1, 2, 3]],
        "expect_logits": [[2, 0.25, 2, 2, 2, 2, 2, 2]],
    },
    {
        # A repeat no longer than allowed_length is not penalized
        "name": "dry, allowed length",
        "sampler": CustomSampler([
            SS_DRY(1.0, 1.75, 4, 0, []),
            SS_Sample_mn()
        ]),
        "input": [[2] * 8],
        "input_seq": [[0, 1, 2, 3, 1, 2, 3]],
        "expect_logits": [[2] * 8],
    },
    {
        # dry_range bounds the window: the earlier occurrence of [1, 2, 3] falls outside the
        # last 4 tokens
        "name": "dry, range",
        "sampler": CustomSampler([
            SS_DRY(1.0, 1.75, 2, 4, []),
            SS_Sample_mn()
        ]),
        "input": [[2] * 8],
        "input_seq": [[1, 2, 3, 1, 2, 3]],
        "expect_logits": [[2] * 8],
    },
    {
        # Same context without the range cap
        "name": "dry, full range",
        "sampler": CustomSampler([
            SS_DRY(1.0, 1.75, 2, 0, []),
            SS_Sample_mn()
        ]),
        "input": [[2] * 8],
        "input_seq": [[1, 2, 3, 1, 2, 3]],
        "expect_logits": [[2, 0.25, 2, 2, 2, 2, 2, 2]],
    },
    {
        # A sequence breaker (9) caps the match at its distance from the end of the context:
        # the full repeat of [1, 2, 9, 3] has length 4, but rep_limit is 1, so the exponent is
        # 1 - allowed_length = 0 and token 1 is penalized by the bare multiplier, not by
        # multiplier * base ** 3
        "name": "dry, breaker caps match",
        "sampler": CustomSampler([
            SS_DRY(0.5, 2.0, 1, 0, [9]),
            SS_Sample_mn()
        ]),
        "input": [[2] * 10],
        "input_seq": [[1, 2, 9, 3, 1, 2, 9, 3]],
        "expect_logits": [[2, 1.5, 2, 2, 2, 2, 2, 2, 2, 2]],
    },
    {
        # A breaker token (9) is never penalized, even where it would extend a repeat
        "name": "dry, breaker not penalized",
        "sampler": CustomSampler([
            SS_DRY(1.0, 1.75, 2, 0, [9]),
            SS_Sample_mn()
        ]),
        "input": [[2] * 10],
        "input_seq": [[1, 2, 9, 1, 2]],
        "expect_logits": [[2] * 10],
    },
    {
        # Control for the case above: without breakers the same context penalizes token 9
        "name": "dry, no breakers control",
        "sampler": CustomSampler([
            SS_DRY(1.0, 1.75, 2, 0, []),
            SS_Sample_mn()
        ]),
        "input": [[2] * 10],
        "input_seq": [[1, 2, 9, 1, 2]],
        "expect_logits": [[2, 2, 2, 2, 2, 2, 2, 2, 2, 1]],
    },
    {
        # Two rows with independent histories; the second row's run of token 4 gives the
        # longest match 6 and penalty base ** (6 - 2)
        "name": "dry, batch",
        "sampler": CustomSampler([
            SS_DRY(1.0, 1.75, 2, 0, []),
            SS_Sample_mn()
        ]),
        "input": [[2] * 8] * 2,
        "input_seq": [[0, 1, 2, 3, 1, 2, 3], [4, 4, 4, 4, 4, 4, 4]],
        "expect_logits": [[2, 0.25, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, -7.37890625, 2, 2, 2]],
    },
    {
        # The exponent clamp: llama.cpp computes FLOAT_MAX_LOG / log(base) in float32, which
        # for base = 2.0 is exactly 128, so the penalty is 2 ** 128 and saturates the float32
        # logit to -inf. A float64 clamp would land on 127 and a finite logit
        "name": "dry, exponent clamp",
        "sampler": CustomSampler([
            SS_DRY(1.0, 2.0, 2, 0, []),
            SS_Sample_mn()
        ]),
        "input": [[2] * 8],
        "input_seq": [[3] * 200],
        "expect_logits": [[2, 2, 2, ni, 2, 2, 2, 2]],
    },
    {
        # DRY after another logits step exercises the SS.LOGITS branch
        "name": "dry, after ban_tokens",
        "sampler": CustomSampler([
            SS_BanTokens([0]),
            SS_DRY(1.0, 1.75, 2, 0, []),
            SS_Sample_mn()
        ]),
        "input": [[2] * 8],
        "input_seq": [[1, 2, 3, 1, 2, 3]],
        "expect_logits": [[ni, 0.25, 2, 2, 2, 2, 2, 2]],
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
    # A leading ban step also keeps a fused tail
    assert fused_mode(CustomSampler([
        SS_BanTokens([1, 2]), SS_Temperature(0.8), SS_MinP(0.05), SS_Sample()
    ])) == SS_Fused.MODE_SAMPLE_MINP
    # A leading logit_bias step also keeps a fused tail
    assert fused_mode(ComboSampler(
        temperature = 0.8, min_p = 0.05, logit_bias = {1: 5.0}
    )) == SS_Fused.MODE_SAMPLE_MINP
    # As does a leading DRY step, standalone or through ComboSampler
    assert fused_mode(CustomSampler([
        SS_DRY(0.8, 1.75, 2, 0, [1]), SS_Temperature(0.8), SS_MinP(0.05), SS_Sample()
    ])) == SS_Fused.MODE_SAMPLE_MINP
    assert fused_mode(ComboSampler(dry_multiplier = 0.8, temperature = 0.8, min_p = 0.05)) \
        == SS_Fused.MODE_SAMPLE_MINP
    # Reference/multinomial nodes and non-canonical filter orders stay on the eager path
    assert fused_mode(CustomSampler([SS_MinP(0.1), SS_Sample_mn()])) is None
    assert fused_mode(CustomSampler([SS_TopP(0.9), SS_MinP(0.1), SS_Temperature(0.8), SS_Sample()])) is None
    # XTC needs the sorted distribution, so it has no fused form
    assert fused_mode(CustomSampler([SS_Temperature(0.8), SS_XTC(0.5, 0.1), SS_Sample()])) is None
    # An empty ban list, a zero XTC probability, an XTC threshold above half, a zero DRY
    # multiplier and a DRY base below 1 are no-ops, and a no-op does not keep the tail off the
    # fused path
    for neutral in [
        SS_BanTokens([]),
        SS_XTC(0.0, 0.1),
        SS_XTC(0.5, 0.6),
        SS_DRY(0.0),
        SS_DRY(0.5, 0.5),
    ]:
        assert fused_mode(CustomSampler([
            neutral, SS_Temperature(0.8), SS_MinP(0.05), SS_Sample()
        ])) == SS_Fused.MODE_SAMPLE_MINP


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


@torch.inference_mode()
def test_argmax_sorted():
    """
    Stacks ending in Argmax normally collapse to the fused step. An explicit sort, XTC, or a
    disabled fused sampler reaches the eager path, where Argmax has to map a sorted position back
    to a token ID per row.
    """
    logits = torch.tensor(
        [[2.0, 1.0, 0.0, -1.0],
         [0.0, 3.0, 1.0, 2.0]],
        dtype = torch.float,
        device = device
    )

    enabled = sampler_custom.fused_sampler_enable
    try:
        sampler_custom.fused_sampler_enable = False
        cases = [
            ([SS_Argmax()], [0, 1]),
            ([SS_Sort(), SS_Argmax()], [0, 1]),
            ([SS_TopK(3), SS_Argmax()], [0, 1]),
            ([SS_MinP(0.1), SS_Argmax()], [0, 1]),
            # Two tokens per row clear the threshold, and scaling the more likely one drops it
            # below the one XTC keeps
            ([SS_XTC(1.0, 0.1), SS_Argmax()], [1, 3]),
            # The same scaling at a low probability is too small to reorder them
            ([SS_XTC(0.05, 0.1), SS_Argmax()], [0, 1]),
        ]
        for steps, expected in cases:
            sample = CustomSampler(steps).forward(logits, rand_u32 = 0)
            assert sample.tolist() == expected, (steps, sample.tolist(), expected)
    finally:
        sampler_custom.fused_sampler_enable = enabled


def dry_reference_penalties(window, breakers, multiplier, base, allowed_length):
    """
    Brute-force DRY oracle: computes each match length by direct comparison of the window with
    the sequence ending at every earlier position, rather than SS_DRY's chunked run-length scan,
    so the two cannot share an indexing bug. The penalty arithmetic follows
    llama_sampler_dry_apply, with the exponent clamp computed in float32 via numpy.
    """
    m = len(window)
    mult = float(np.float32(multiplier))
    b = float(np.float32(base))
    if np.float32(b) > np.float32(1.000001):
        max_exp = int(np.float32(88.7228391) / np.log(np.float32(b)))
    else:
        max_exp = 0
    rep_limit = m
    for i in range(m):
        if window[m - 1 - i] in breakers:
            rep_limit = i
            break
    if rep_limit < allowed_length:
        return {}
    best = {}
    for i in range(m - 1):
        length = 0
        while length <= i and window[i - length] == window[m - 1 - length]:
            length += 1
        length = min(length, rep_limit)
        tok = window[i + 1]
        if length >= allowed_length and tok not in breakers:
            best[tok] = max(best.get(tok, 0), length)
    penalties = {}
    for tok, length in best.items():
        exponent = length - allowed_length
        if max_exp and exponent > max_exp:
            exponent = max_exp
        penalties[tok] = mult * float(np.float64(b) ** exponent)
    return penalties


@pytest.mark.parametrize("past_device", ["cpu", device])
@torch.inference_mode()
def test_dry_oracle(past_device):
    """
    Randomized differential test of SS_DRY against the brute-force oracle, exact to the bit
    since both sides compute the penalty in float64 and round once to float32. past_ids are
    given on the host and on the device (the kernel uploads host ids); a slice of the cases
    injects out-of-vocabulary and negative past IDs, which match as ordinary tokens but are
    never penalized or treated as breakers, and the last cases use windows long enough to
    spread the offset scan over several kernel blocks.
    """
    rng = random.Random(11)
    try:
        for case in range(220):
            dim = rng.choice([16, 33, 64])
            # the last cases span several scan blocks of the kernel (1024 offsets each)
            n = rng.randint(3, 60) if case < 200 else rng.randint(1500, 3000)
            alphabet = rng.randint(2, 8)
            seq = [rng.randrange(alphabet) for _ in range(n)]
            if case % 5 == 0:
                seq[rng.randrange(n)] = rng.choice([-1, dim, dim + 7])
            multiplier = rng.choice([0.5, 1.0, 2.7])
            base = rng.choice([1.0, 1.05, 1.75, 2.0, 3.0])
            allowed = rng.choice([0, 1, 2, 5])
            dry_range = rng.choice([0, 3, 7, 100])
            breakers = frozenset(rng.sample(range(alphabet + 2), rng.randint(0, 3)))

            sampler = CustomSampler([SS_DRY(multiplier, base, allowed, dry_range, breakers or None)])
            logits = torch.zeros((1, dim), dtype = torch.float, device = device)
            seq_t = torch.tensor([seq], dtype = torch.long, device = past_device)
            if past_device == "cpu":
                seq_t = seq_t.pin_memory()
            state = sampler.forward(logits, sequence_ids = seq_t, rand_u32 = 0, return_state = True)

            m = n if dry_range <= 0 else min(n, dry_range)
            window = seq[-m:]
            if m <= allowed:
                penalties = {}
            else:
                penalties = dry_reference_penalties(window, breakers, multiplier, base, allowed)
            expect = torch.zeros(dim)
            for tok, val in penalties.items():
                if 0 <= tok < dim:
                    expect[tok] = torch.tensor(val, dtype = torch.float64).to(torch.float32)
            assert torch.equal(state.logits[0].cpu(), -expect), \
                (seq, multiplier, base, allowed, dry_range, sorted(breakers))
    finally:
        pass


def test_dry_validation():
    with pytest.raises(AssertionError):
        SS_DRY(-0.5)
    with pytest.raises(AssertionError):
        SS_DRY(1.0, 1.75, 2.5)
    with pytest.raises(AssertionError):
        SS_DRY(1.0, 1.75, 2, 0.5)


class _DryStubTokenizer:
    """
    Minimal stand-in for the breaker-derivation interface of Tokenizer, matching what
    dry_default_sequence_breaker_tokens reads.
    """
    def __init__(self, breaker_pieces = True):
        self.extended_id_to_piece = {7: "<x>"} if breaker_pieces else {}
        self.actual_vocab_size = 12
        self.breaker_pieces = breaker_pieces

    def get_id_to_piece_list(self, include_special_tokens = False):
        # IDs 1 and 3 contain default breaker characters
        if self.breaker_pieces:
            return ["a", "b.", "c", "d\n", "e", "f", "g"]
        return ["a", "b", "c", "d", "e", "f", "g"]


@torch.inference_mode()
def test_dry_default_breakers():
    """
    With dry_sequence_breakers = None the default set derives per tokenizer from the one passed
    to forward(), as the generator does: token 3 is a breaker for the stub, so it caps the match
    at rep_limit 2 and is itself not penalized, while the tokenizer-less call penalizes it, and
    a later tokenizer without breaker pieces penalizes it again through the same sampler
    """
    logits = torch.zeros((1, 12), dtype = torch.float, device = device)
    seq_t = torch.tensor([[0, 5, 2, 3, 5, 2]], dtype = torch.long, device = "cpu", pin_memory = True)
    sampler = CustomSampler([SS_DRY(1.0, 1.75, 2, 0)])

    state = sampler.forward(logits.clone(), sequence_ids = seq_t, rand_u32 = 0, return_state = True)
    assert state.logits[0, 3].item() == -1.0, state.logits

    state = sampler.forward(
        logits.clone(), sequence_ids = seq_t, rand_u32 = 0, return_state = True,
        tokenizer = _DryStubTokenizer(),
    )
    assert torch.equal(state.logits[0], logits[0]), state.logits

    # Not latched to the first tokenizer seen: a vocabulary without breaker pieces penalizes
    state = sampler.forward(
        logits.clone(), sequence_ids = seq_t, rand_u32 = 0, return_state = True,
        tokenizer = _DryStubTokenizer(breaker_pieces = False),
    )
    assert state.logits[0, 3].item() == -1.0, state.logits


@pytest.mark.parametrize("dim", dims)
@torch.inference_mode()
def test_ban_tokens(dim: tuple):
    """
    The ban runs before the collapsed tail, so the fused kernel has to honor the -inf logits it
    writes.
    """
    torch.manual_seed(0)
    random.seed(0)
    logits = torch.randn(dim, dtype = torch.half, device = device) * 2

    banned = sorted({0, 1, 2, dim[-1] // 2, dim[-1] - 1})
    sampler = CustomSampler([
        SS_BanTokens(banned),
        SS_Temperature(0.8),
        SS_MinP(0.02),
        SS_Sample()
    ])
    if sampler_custom.fused_sampler_enable:
        assert fused_mode(sampler) == SS_Fused.MODE_SAMPLE_MINP

    banned_t = torch.tensor(banned, dtype = torch.long, device = device)
    for seed in range(200):
        sample = sampler.forward(logits, rand_u32 = seed)
        assert not torch.isin(sample, banned_t).any()


@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("probability", [0.5, 1.0])
@torch.inference_mode()
def test_xtc(dim: tuple, probability):
    """
    SS_XTC reweights instead of drawing an outcome per token. The reference below is the
    distribution that draw produces on average, so the sampled tokens have to follow it.
    """
    torch.manual_seed(0)
    random.seed(0)
    threshold = 0.1

    logits = torch.randn(dim, dtype = torch.half, device = device) * 2
    probs_ref = torch.softmax(logits.float(), dim = -1)

    # Reference for the excluded set, which is every token above the threshold except the least
    # likely of them
    sorted_probs, sorted_indices = torch.sort(probs_ref, dim = -1, descending = True)
    qualifies = sorted_probs >= threshold
    counts = qualifies.sum(dim = -1, keepdim = True)
    excluded = torch.zeros_like(qualifies).scatter_(
        -1, sorted_indices, qualifies & (qualifies.cumsum(dim = -1) < counts)
    )

    truncated = probs_ref.masked_fill(excluded, 0.0)
    truncated /= truncated.sum(dim = -1, keepdim = True)
    probs_ref = (1.0 - probability) * probs_ref + probability * truncated

    sampler = CustomSampler([SS_XTC(probability, threshold), SS_Sample()])
    assert fused_mode(sampler) is None

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
@pytest.mark.parametrize("threshold", [0.02, 0.1, 0.4])
@pytest.mark.parametrize("probability", [0.35, 1.0])
@torch.inference_mode()
def test_xtc_bound(dim: tuple, threshold, probability):
    """
    SS_XTC reads only the first 1/threshold positions. The reference below scans the whole
    vocabulary, so a wrong bound shows up as a wrong distribution. Both are normalized before
    comparing, the step leaving its result unnormalized.
    """
    torch.manual_seed(0)
    random.seed(0)

    logits = torch.randn(dim, dtype = torch.half, device = device) * 2
    probs = torch.softmax(logits.float(), dim = -1)

    sorted_probs, sorted_indices = torch.sort(probs, dim = -1, descending = True)
    qualifies = sorted_probs >= threshold
    counts = qualifies.sum(dim = -1, keepdim = True)
    excluded = qualifies & (qualifies.cumsum(dim = -1) < counts)
    x_mass = (sorted_probs * excluded).sum(dim = -1, keepdim = True)
    scale = (1.0 - probability) / (1.0 + probability * x_mass / (1.0 - x_mass))
    reference = torch.zeros_like(probs).scatter_(
        -1,
        sorted_indices,
        sorted_probs * torch.where(excluded, scale, torch.ones_like(scale))
    )
    reference /= reference.sum(dim = -1, keepdim = True)

    state = CustomSampler([SS_XTC(probability, threshold)]).forward(logits, return_state = True)
    result = torch.zeros_like(probs).scatter_(-1, state.indices, state.probs)
    result /= result.sum(dim = -1, keepdim = True)

    torch.testing.assert_close(result, reference)
