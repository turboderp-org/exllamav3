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

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 150)

device = "cuda:2"
dims = [
    (1, 16),
    (9, 16),
    (1, 32768),
    (2, 128256),
    (1, 256000),
]

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
        sample = sampler.forward(logits)
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
        sample = sampler.forward(logits)
        samples = torch.cat((samples, sample), dim = -1)

    hb = [torch.bincount(samples[b], minlength = dim[1]) for b in range(dim[0])]
    histogram = torch.stack(hb).float()
    histogram /= num_samples

    chisq = compare(histogram, probs_ref)
    assert chisq < 0.02