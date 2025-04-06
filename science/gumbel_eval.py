import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3 import GumbelSampler, ArgmaxSampler
import random
import matplotlib.pyplot as plt

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

random.seed(4)
torch.manual_seed(4)

device = "cuda:2"
dim = 128256
shape = (1, 1, dim)
num_samples = 250000

# Create logits tensor
logits = torch.randn(shape, dtype = torch.half, device = device) * 3
logits = logits.contiguous()

# Reference distribution
true_dist = torch.softmax(logits.float(), dim = -1).flatten()

# Ignore probabilities below threshold
min_p = 0.00001

def test_dist(mode: str):

    # Do the stuff
    sampler = GumbelSampler()
    histogram = torch.zeros((dim,), dtype = torch.float, device = device)
    graph = []
    for i in range(num_samples):
        match mode:
            case "mn":
                sample = torch.multinomial(true_dist, num_samples = 1)
            case "g_krn":
                sample = sampler.forward(logits).flatten()
        histogram[sample] += 1
        if (i + 1) % 10000 == 0:
            observed_counts = (histogram / (i + 1)).clamp(min = min_p)
            expected_counts = true_dist.clamp(min = min_p)
            chisq = ((observed_counts - expected_counts).square() / expected_counts).sum().item()
            graph.append((i + 1, chisq))
            print(f"{i + 1} / {num_samples}    chi_squared: {chisq}")

    print("------")
    print(expected_counts)
    print(observed_counts)
    print("------")
    return graph

print("Gumbel")
gumbel = test_dist("g_krn")
print("Softmax + multinomial")
multinomial = test_dist("mn")

gx, gy = zip(*gumbel)
mx, my = zip(*multinomial)
plt.plot(gx, gy, label = "Gumbel", marker = None)
plt.plot(mx, my, label='Softmax + multinomial', marker = None)
plt.xlabel("Samples")
plt.ylabel(f"chi_squared, p > {min_p}")
plt.title(f"Vocab size: {dim}, {num_samples} samples")
plt.legend()
plt.show()