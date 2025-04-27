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

    # Reference
    logits_ref = logits.float() / temperature
    probs_ref = torch.softmax(logits_ref, dim = -1)
    topk_values, topk_indices = torch.topk(probs_ref, k, dim = -1)
    mask = torch.zeros_like(probs_ref, dtype = torch.bool)
    mask.scatter_(1, topk_indices, True)
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

