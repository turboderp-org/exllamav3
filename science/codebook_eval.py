import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3.ext import exllamav3_ext as ext
import matplotlib.pyplot as plt

torch.set_printoptions(precision = 8, sci_mode = False, linewidth = 200)

r = torch.arange(65536, device = "cuda:0", dtype = torch.short).unsqueeze(0)
codebook_lut = torch.zeros_like(r, dtype = torch.float)
ext.decode(r, codebook_lut)
codebook_lut = codebook_lut[0]

# RMS of codebook
rms = codebook_lut.square().mean().sqrt()
print(f"Codebook RMS: {rms:.6f}")

# Figure
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(3, 4)

# Correlation
for i, K in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):

    num_samples = 20000
    a = torch.randint(low = 0, high = 65536, size = (num_samples, 1), dtype = torch.int)
    b = torch.randint(low = 0, high = 1 << K, size = (num_samples, 1), dtype = torch.int)
    c = (a << K) | b
    v = torch.cat((a, c), dim = 1) & 0xFFFF
    x = codebook_lut[v]

    x_np = x.cpu().numpy()
    ax = fig.add_subplot(gs[i // 4, i % 4])
    ax.scatter(x_np[:, 0], x_np[:, 1], s = 1, alpha = 0.5)
    ax.set_title(f"K={K}")
    ax.axis("equal")

# Distribution
codebook_lut_np = codebook_lut.cpu().numpy()
ax = fig.add_subplot(gs[2, :])
ax.hist(codebook_lut_np, bins = 256)
ax.set_title(f"Distribution, RMS: {rms:.6f}")

plt.tight_layout()
plt.show()