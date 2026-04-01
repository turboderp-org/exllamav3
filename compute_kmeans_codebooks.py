#!/usr/bin/env python3
"""
Compute k-means optimal codebooks for the actual post-WHT, post-max-normalized
distribution used by ExLlamaV3.

The target distribution is:
  X_i / max_j(|X_j|)   where X_j = WHT(N(0,1)) / sqrt(block_size)

This is bounded in [-1, 1], platykurtic (lighter tails than Gaussian),
std ≈ 0.43.
"""

import sys
import numpy as np

try:
    import torch
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch available, using device: {DEVICE}")
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available, using NumPy")

BLOCK_SIZE = 32
N_SAMPLES   = 10_000_000     # 10M Gaussian values → 312 500 blocks
# k-means settings: uniform init + up to 5000 iters
# Convergence: centroid_delta < 1e-6  OR  rel_mse_change < 1e-9
N_KMEANS_ITER       = 5000
TOL_CENTROID        = 1e-6   # well below float32 precision in [-1,1]
TOL_REL_MSE         = 1e-9   # relative MSE change below numerical noise
SEED                = 42

HEADER_PATH = "/tmp/exllamav3-tq/exllamav3/exllamav3_ext/cache/lloyd_max_codebooks.cuh"

# ---------------------------------------------------------------------------
# Step 1: Generate the actual distribution
# ---------------------------------------------------------------------------

def wht_butterfly_numpy(x):
    """In-place Walsh-Hadamard Transform (butterfly) on last axis."""
    n = x.shape[-1]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            a = x[..., i:i+h].copy()
            b = x[..., i+h:i+2*h].copy()
            x[..., i:i+h]     = a + b
            x[..., i+h:i+2*h] = a - b
        h *= 2
    return x


def generate_distribution():
    """Return flat array of all post-WHT, post-max-normalised values."""
    if HAS_TORCH and DEVICE == "cuda":
        print("Generating samples on GPU...")
        chunk_blocks = 100_000
        total_blocks  = N_SAMPLES // BLOCK_SIZE
        all_vals = []
        torch.manual_seed(SEED)
        for start in range(0, total_blocks, chunk_blocks):
            end = min(start + chunk_blocks, total_blocks)
            n_blocks = end - start
            x = torch.randn(n_blocks, BLOCK_SIZE, device=DEVICE)
            h = 1
            while h < BLOCK_SIZE:
                for i in range(0, BLOCK_SIZE, h * 2):
                    a = x[:, i:i+h].clone()
                    b = x[:, i+h:i+2*h].clone()
                    x[:, i:i+h]     = a + b
                    x[:, i+h:i+2*h] = a - b
                h *= 2
            x = x / (BLOCK_SIZE ** 0.5)
            max_abs = x.abs().max(dim=1, keepdim=True).values
            x = x / max_abs
            all_vals.append(x.cpu().numpy().ravel())
        samples = np.concatenate(all_vals)
    else:
        print("Generating samples on CPU...")
        rng = np.random.default_rng(SEED)
        total_blocks = N_SAMPLES // BLOCK_SIZE
        raw = rng.standard_normal((total_blocks, BLOCK_SIZE)).astype(np.float32)
        wht_butterfly_numpy(raw)
        raw /= np.sqrt(BLOCK_SIZE)
        max_abs = np.abs(raw).max(axis=1, keepdims=True)
        raw /= max_abs
        samples = raw.ravel()

    print(f"Generated {len(samples):,} samples")
    print(f"  mean  = {samples.mean():.6f}")
    print(f"  std   = {samples.std():.6f}")
    print(f"  min   = {samples.min():.6f}")
    print(f"  max   = {samples.max():.6f}")
    kurt = float(np.mean((samples - samples.mean())**4) / samples.std()**4) - 3
    print(f"  kurtosis excess = {kurt:.4f}")
    return samples


# ---------------------------------------------------------------------------
# Step 2: Empirical PDF from histogram (for fast Lloyd-Max iteration)
# ---------------------------------------------------------------------------

def build_histogram(samples, n_bins=100_000):
    """Return (bin_centers, pdf) from empirical samples."""
    counts, edges = np.histogram(samples, bins=n_bins, range=(-1.0, 1.0), density=False)
    centers = (edges[:-1] + edges[1:]) / 2.0
    pdf = counts / counts.sum()          # normalised probability
    return centers.astype(np.float64), pdf.astype(np.float64)


# ---------------------------------------------------------------------------
# Step 3: Lloyd-Max on the HISTOGRAM (fast, deterministic, no sampling noise)
# ---------------------------------------------------------------------------

def lloyd_max_histogram(bin_centers, pdf, bits,
                         n_iter=N_KMEANS_ITER,
                         tol_centroid=TOL_CENTROID,
                         tol_rel_mse=TOL_REL_MSE):
    """
    Lloyd-Max on a histogram (bin_centers, pdf).
    Initialises uniformly for platykurtic near-uniform distributions.
    """
    n_levels = 1 << bits
    print(f"\n--- {bits}-bit ({n_levels} levels) ---", flush=True)

    # Uniform initialisation: optimal starting point for near-uniform PDFs
    step = 2.0 / n_levels
    centroids = np.linspace(-1.0 + step/2, 1.0 - step/2, n_levels, dtype=np.float64)

    mse = np.inf

    for it in range(n_iter):
        # Decision boundaries: midpoints between adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Assign histogram bins to nearest centroid
        indices = np.searchsorted(boundaries, bin_centers)
        indices = np.clip(indices, 0, n_levels - 1)

        # Update centroids: weighted mean of assigned bins
        new_centroids = np.empty_like(centroids)
        for k in range(n_levels):
            mask = indices == k
            w = pdf[mask]
            if w.sum() > 0:
                new_centroids[k] = np.sum(bin_centers[mask] * w) / w.sum()
            else:
                new_centroids[k] = centroids[k]

        # Compute MSE
        new_mse = float(np.sum(pdf * (bin_centers - new_centroids[indices])**2))
        centroid_delta = float(np.max(np.abs(new_centroids - centroids)))
        rel_mse_change = abs(mse - new_mse) / max(abs(mse), 1e-30)

        centroids = new_centroids
        mse = new_mse

        if (it + 1) % 100 == 0:
            print(f"  iter {it+1:4d}: MSE={mse:.8f} cd={centroid_delta:.2e} rel_mse={rel_mse_change:.2e}", flush=True)

        if (centroid_delta < tol_centroid or rel_mse_change < tol_rel_mse) and it > 5:
            print(f"  Converged at iter {it+1}: MSE={mse:.8f} cd={centroid_delta:.2e} rel={rel_mse_change:.2e}", flush=True)
            break
    else:
        print(f"  Max iters reached: MSE={mse:.8f}", flush=True)

    # Final boundaries from converged centroids
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0

    print(f"  Centroids: {np.array2string(centroids, precision=6, separator=', ')}", flush=True)
    return centroids, boundaries, mse


# ---------------------------------------------------------------------------
# Compute uniform quantiser MSE on the histogram
# ---------------------------------------------------------------------------

def compute_uniform_mse_hist(bin_centers, pdf, n_levels):
    step = 2.0 / n_levels
    centroids  = np.linspace(-1 + step/2, 1 - step/2, n_levels)
    boundaries = np.linspace(-1 + step,   1 - step,   n_levels - 1)
    indices = np.searchsorted(boundaries, bin_centers)
    indices = np.clip(indices, 0, n_levels - 1)
    mse = float(np.sum(pdf * (bin_centers - centroids[indices])**2))
    return mse


# ---------------------------------------------------------------------------
# Also measure on the raw samples for reporting
# ---------------------------------------------------------------------------

def mse_on_samples(samples, centroids):
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    n_levels = len(centroids)
    idx = np.searchsorted(boundaries, samples)
    idx = np.clip(idx, 0, n_levels - 1)
    return float(np.mean((samples.astype(np.float64) - centroids[idx])**2))


def uniform_mse_on_samples(samples, n_levels):
    step = 2.0 / n_levels
    centroids  = np.linspace(-1 + step/2, 1 - step/2, n_levels)
    boundaries = np.linspace(-1 + step,   1 - step,   n_levels - 1)
    idx = np.searchsorted(boundaries, samples)
    idx = np.clip(idx, 0, n_levels - 1)
    return float(np.mean((samples.astype(np.float64) - centroids[idx])**2))


# ---------------------------------------------------------------------------
# Format results as C arrays
# ---------------------------------------------------------------------------

def fmt_array(values, name, dtype="float"):
    n = len(values)
    lines = [f"__constant__ {dtype} {name}[{n}] = {{"]
    for i, v in enumerate(values):
        comma = "," if i < n - 1 else ""
        lines.append(f"    {v:.10f}f{comma}")
    lines.append("};")
    return "\n".join(lines)


def build_header(results_by_bits, bin_centers, pdf):
    bit_labels = {
        2: "2-bit:  4 levels,   3 boundaries",
        3: "3-bit:  8 levels,   7 boundaries",
        4: "4-bit: 16 levels,  15 boundaries",
        5: "5-bit: 32 levels,  31 boundaries",
        6: "6-bit: 64 levels,  63 boundaries",
        7: "7-bit: 128 levels, 127 boundaries",
        8: "8-bit: 256 levels, 255 boundaries",
    }

    lines = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("// Lloyd-Max optimal codebooks for the actual ExLlamaV3 post-WHT/max-norm distribution")
    lines.append("// Computed via Lloyd-Max algorithm on 100k-bin histogram of 10M samples")
    lines.append("// of: X_i / max_j(|X_j|)  where X_j = WHT(N(0,1)) / sqrt(32)")
    lines.append("//")
    lines.append("// Distribution: bounded [-1,1], std≈0.43, platykurtic (kurtosis excess≈-0.27)")
    lines.append("//")
    lines.append("// For each bit-width N, there are 2^N centroids and 2^N-1 boundaries.")
    lines.append("// Centroids are NOT pinned to ±1; they sit at the optimal Lloyd-Max positions")
    lines.append("//")

    for bits, (centroids, boundaries, _km_mse) in sorted(results_by_bits.items()):
        n_levels = 1 << bits
        u_mse = compute_uniform_mse_hist(bin_centers, pdf, n_levels)
        km_mse = _km_mse
        improvement = (u_mse - km_mse) / u_mse * 100.0 if u_mse > 0 else 0
        sqnr = 10 * np.log10(u_mse / km_mse) if km_mse > 0 else 0
        lines.append(f"//   {bits}-bit: {improvement:.1f}% MSE reduction vs uniform  (+{sqnr:.2f} dB SQNR)")

    lines.append("//")
    lines.append("// Use __constant__ memory for fast cached access from all threads.")
    lines.append("// Max 256 values per array (8-bit = 256 levels).")
    lines.append("")

    bit_name_map = {2:"2bit", 3:"3bit", 4:"4bit", 5:"5bit", 6:"6bit", 7:"7bit", 8:"8bit"}
    for bits in range(2, 9):
        centroids, boundaries, _ = results_by_bits[bits]
        bit_name = bit_name_map[bits]

        lines.append("// " + "-" * 75)
        lines.append(f"// {bit_labels[bits]}")
        lines.append("// " + "-" * 75)
        lines.append("")
        lines.append(fmt_array(boundaries, f"lm_boundaries_{bit_name}"))
        lines.append("")
        lines.append(fmt_array(centroids,  f"lm_centroids_{bit_name}"))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Step 1: generate samples
    print("=" * 60)
    print("Step 1: Generating distribution samples")
    print("=" * 60, flush=True)
    samples = generate_distribution()

    # Step 2: build histogram (avoids sampling noise in k-means)
    print("\nBuilding histogram (100k bins)...", flush=True)
    bin_centers, pdf = build_histogram(samples, n_bins=100_000)
    print(f"Histogram: {len(bin_centers)} bins, total weight = {pdf.sum():.8f}", flush=True)

    # Step 3: Lloyd-Max on histogram
    print("\n" + "=" * 60)
    print("Step 2: Running Lloyd-Max for each bit-width")
    print("=" * 60, flush=True)

    results = {}
    for bits in range(2, 9):
        centroids, boundaries, km_mse = lloyd_max_histogram(bin_centers, pdf, bits)
        results[bits] = (centroids, boundaries, km_mse)

    # Step 4: Print improvement table
    print("\n" + "=" * 60)
    print("Step 4: Improvement table (measured on raw samples)")
    print("=" * 60)
    print(f"{'Bits':>4}  {'Uniform MSE':>12}  {'k-means MSE':>12}  {'Improvement':>12}")
    print("-" * 50)
    for bits in range(2, 9):
        n_levels = 1 << bits
        u_mse   = uniform_mse_on_samples(samples, n_levels)
        centroids, _, _ = results[bits]
        km_mse  = mse_on_samples(samples, centroids)
        improvement = (u_mse - km_mse) / u_mse * 100.0
        print(f"{bits:>4}  {u_mse:>12.6f}  {km_mse:>12.6f}  {improvement:>11.1f}%")
        # Update stored MSE with the sample-based one
        results[bits] = (results[bits][0], results[bits][1], km_mse)

    # Step 5: Write header
    print("\n" + "=" * 60)
    print("Step 3: Writing updated header file")
    print("=" * 60)
    header_content = build_header(results, bin_centers, pdf)

    with open(HEADER_PATH, "w") as f:
        f.write(header_content)
    print(f"Written: {HEADER_PATH}")

    # Basic syntax verification
    with open(HEADER_PATH) as f:
        content = f.read()

    open_braces  = content.count("{")
    close_braces = content.count("}")
    # Count actual array declarations (lines that start with __constant__)
    decl_lines = [l for l in content.splitlines() if l.strip().startswith("__constant__")]
    const_count = len(decl_lines)
    ok = "OK" if open_braces == close_braces else "BRACE MISMATCH!"
    print(f"\nSyntax check: {open_braces} open / {close_braces} close braces ({ok})")
    print(f"  __constant__ array declarations: {const_count} (expected 14)")
    assert open_braces == close_braces, "Brace mismatch in output!"
    assert const_count == 14, f"Expected 14 constant declarations, got {const_count}"
    print("Header file looks valid.")


if __name__ == "__main__":
    main()
