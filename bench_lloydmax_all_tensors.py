"""
Benchmark: Lloyd-Max vs Uniform quantization across ALL LLM tensor types.

For each tensor type, we:
1. Generate data matching real-world distributions.
2. Apply Hadamard rotation (as ExLlamaV3 does for KV cache and activations).
3. Compute the optimal Lloyd-Max codebook via k-means on the ACTUAL normalized
   post-Hadamard distribution of that tensor type (not on raw N(0,1)).
4. Quantize with both Lloyd-Max and Uniform at 2-8 bits.
5. Measure MSE, SQNR, and the relative improvement.

Key design decision
-------------------
ExLlamaV3 quantizes blocks of 32 values with per-block max-scaling:
    normalized_i = x_i / max_j(|x_j|)   (so |normalized_i| <= 1)

After this normalization the effective distribution is NOT Gaussian — it is the
distribution of X_i / max_j(|X_j|) for a block of 32 values drawn from whatever
distribution the tensor actually follows.  That distribution is:
  - Symmetric around 0 (for zero-mean tensors)
  - Bounded in [-1, 1]
  - Platykurtic (kurtosis < 3) — lighter tails than Gaussian because the max
    normalization suppresses extreme values relative to their block

The Lloyd-Max codebook is therefore computed on samples from THIS normalized
distribution, not from raw N(0,1).  This makes the codebook exactly optimal
for the quantization scheme as implemented in ExLlamaV3.

Self-contained — requires only PyTorch (no scipy required).
"""

import torch
import math
import sys
import time


# ---------------------------------------------------------------------------
# Hadamard transform
# ---------------------------------------------------------------------------

BLOCK_SIZE = 32   # Must be a power of 2


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Walsh-Hadamard Transform in non-overlapping blocks of BLOCK_SIZE,
    matching ExLlamaV3's pre-quantization rotation approach.
    """
    orig_shape = x.shape
    x = x.float().reshape(-1, BLOCK_SIZE)
    n, h = BLOCK_SIZE, 1
    while h < n:
        for i in range(0, n, 2 * h):
            j = torch.arange(i, i + h, device=x.device)
            k = j + h
            a = x[:, j].clone()
            b = x[:, k].clone()
            x[:, j] = a + b
            x[:, k] = a - b
        h *= 2
    return (x / math.sqrt(n)).reshape(orig_shape)


# ---------------------------------------------------------------------------
# Lloyd-Max codebook computation
# ---------------------------------------------------------------------------

def compute_lloyd_max_codebook(
    num_levels: int,
    sample_generator,
    n_blocks: int = 100_000,
    max_iter: int = 300,
) -> tuple:
    """
    Compute the optimal Lloyd-Max codebook for the quantization scheme used by
    ExLlamaV3: Hadamard rotation + per-block max-scaling.

    Algorithm
    ---------
    1. Generate n_blocks * BLOCK_SIZE samples from the target distribution
       using `sample_generator()`.
    2. Apply Hadamard rotation and per-block max-scaling to obtain the
       normalized distribution in [-1, 1].
    3. Run k-means (Lloyd algorithm) on these normalized samples to find
       the optimal quantizer for THIS distribution.

    Returns
    -------
    (boundaries, centroids) as Python lists, both sorted ascending.
    boundaries : num_levels - 1 decision thresholds
    centroids  : num_levels  reconstruction values
    """
    torch.manual_seed(42)
    raw = sample_generator(n_blocks * BLOCK_SIZE)

    # Apply Hadamard + per-block max-normalization
    data_h = hadamard_transform(raw)
    blocks = data_h.reshape(-1, BLOCK_SIZE)
    scales = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
    normalized = (blocks / scales).reshape(-1)      # shape (n_blocks * BLOCK_SIZE,)

    n = len(normalized)

    # Initialization: equal-count bins from sorted data (equivalent to quantile init)
    data_sorted, _ = normalized.sort()
    step = n // num_levels
    cents = torch.zeros(num_levels, dtype=torch.float32)
    for k in range(num_levels):
        lo = k * step
        hi = min((k + 1) * step, n)
        cents[k] = data_sorted[lo:hi].mean()

    # Iterate: update boundaries → assignments → centroids
    for _ in range(max_iter):
        bounds = (cents[:-1] + cents[1:]) / 2          # num_levels - 1 thresholds
        assignments = torch.bucketize(normalized, bounds)
        counts = torch.zeros(num_levels)
        sums   = torch.zeros(num_levels)
        counts.scatter_add_(0, assignments, torch.ones(n))
        sums.scatter_add_(0, assignments, normalized)
        new_cents = torch.where(counts > 0, sums / counts, cents)
        delta = (new_cents - cents).abs().max().item()
        cents = new_cents
        if delta < 1e-8:
            break

    boundaries = [(cents[i] + cents[i + 1]).item() / 2 for i in range(num_levels - 1)]
    centroids  = cents.tolist()
    return boundaries, centroids


# ---------------------------------------------------------------------------
# Quantization routines
# ---------------------------------------------------------------------------

def quantize_uniform(data: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    Symmetric uniform quantization with per-block max-scaling (ExLlamaV3 style).
    """
    blocks = data.float().reshape(-1, BLOCK_SIZE)
    scales = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
    normalized = blocks / scales

    num_levels = 1 << num_bits
    m = num_levels // 2     # levels span [-m, m-1] as signed integers
    q = (normalized * m).round().clamp(-m, m - 1).long() + m   # unsigned [0, 2m-1]
    reconstructed = (q.float() - m) / m * scales
    return reconstructed.reshape(data.shape)


def quantize_lloyd_max(
    data: torch.Tensor,
    num_bits: int,
    boundaries: list,
    centroids: list,
) -> torch.Tensor:
    """
    Lloyd-Max quantization with per-block max-scaling.

    The codebook (boundaries / centroids) was computed for the normalized
    distribution in [-1, 1], so we apply it directly to normalized blocks
    without any additional rescaling.
    """
    blocks = data.float().reshape(-1, BLOCK_SIZE)
    scales = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
    normalized = blocks / scales

    nb_t = torch.tensor(boundaries, device=data.device, dtype=torch.float32)
    nc_t = torch.tensor(centroids,  device=data.device, dtype=torch.float32)

    flat = normalized.reshape(-1)
    q    = torch.searchsorted(nb_t, flat).reshape(normalized.shape)
    q    = q.clamp(0, len(centroids) - 1)
    reconstructed = nc_t[q] * scales
    return reconstructed.reshape(data.shape)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    diff = original.float() - reconstructed.float()
    return (diff ** 2).mean().item()


def sqnr_db(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    sig_power   = (original.float() ** 2).mean().item()
    noise_power = mse(original, reconstructed)
    if noise_power < 1e-20:
        return float('inf')
    return 10.0 * math.log10(sig_power / noise_power)


# ---------------------------------------------------------------------------
# Tensor type sample generators
# ---------------------------------------------------------------------------

def gen_kv_cache(size: int) -> torch.Tensor:
    """
    KV cache vectors — output of the K/V linear projections applied to
    RMSNorm-normalised inputs.

    Post-projection outputs are approximately Gaussian.  After the Hadamard
    rotation ExLlamaV3 applies, individual elements are approximately iid N(0,1).
    """
    return torch.randn(size, dtype=torch.float32)


def gen_activations(size: int) -> torch.Tensor:
    """
    Hidden states / inter-layer activations (post-LayerNorm).

    After RMSNorm each token vector has unit RMS, but different tokens have
    different per-channel statistics.  We model this as a scaled Gaussian mixture
    with block-level scale jitter (each group of 128 values shares a scale drawn
    from a distribution centred near 1) plus a small outlier component.
    """
    x = torch.randn(size, dtype=torch.float32)
    block = 128
    block_scales = torch.randn(size // block).abs() * 0.5 + 0.75
    x = x.reshape(-1, block) * block_scales.unsqueeze(1)
    outlier_mask = torch.rand(size) < 0.005   # ~0.5% strong outliers
    x = x.reshape(-1)
    x[outlier_mask] = x[outlier_mask] * 5.0
    return x


def gen_embeddings(size: int) -> torch.Tensor:
    """
    Embedding table rows.

    Trained embeddings settle at lower magnitudes than Kaiming-initialised
    weights and tend toward a more uniform (platykurtic) distribution.
    """
    gaussian_part = torch.randn(size, dtype=torch.float32) * 0.1
    uniform_part  = (torch.rand(size) * 2 - 1) * 0.05
    return gaussian_part + uniform_part


def gen_moe_weights(size: int) -> torch.Tensor:
    """
    MoE expert weight matrices.

    Expert weights follow approximately N(0, 2/fan_in) from Kaiming
    initialisation, often with slightly heavier tails after fine-tuning.
    The effective fan_in for a typical intermediate MLP is ~4096; we use
    a representative scale.
    """
    fan_in = 4096
    scale = math.sqrt(2.0 / fan_in)
    base = torch.randn(size, dtype=torch.float32) * scale
    # Mild heavy-tail component simulating post-training drift
    laplace = -(torch.rand(size).log()) * torch.sign(torch.randn(size)) * (scale * 0.3)
    return base + laplace


def gen_allreduce(size: int) -> torch.Tensor:
    """
    All-reduce communication buffers — partial sums of activations/gradients
    across TP (tensor-parallel) ranks.

    Sum of N independent Gaussian tensors is still Gaussian with std = sqrt(N)
    times the single-rank std.  We normalise by sqrt(N) so the effective std
    matches a single-rank activation, but with a slightly wider spread.
    """
    num_ranks = 4
    accumulated = sum(
        torch.randn(size, dtype=torch.float32) for _ in range(num_ranks)
    )
    return accumulated / math.sqrt(num_ranks)


# ---------------------------------------------------------------------------
# Per-tensor-type data distribution diagnostics
# ---------------------------------------------------------------------------

def distribution_stats(data: torch.Tensor) -> dict:
    x = data.float()
    mean = x.mean().item()
    std  = x.std().item()
    if std < 1e-10:
        return {"mean": mean, "std": std, "skew": 0.0, "ex_kurt": 0.0}
    z = (x - mean) / std
    skew    = (z ** 3).mean().item()
    ex_kurt = (z ** 4).mean().item() - 3.0
    return {"mean": mean, "std": std, "skew": skew, "ex_kurt": ex_kurt}


def print_distribution_summary(name: str, generator, size: int = 65_536) -> None:
    """Print one-line stats about the raw (pre-Hadamard) data distribution."""
    samples = [generator(size) for _ in range(3)]
    combined = torch.cat([s.float().reshape(-1) for s in samples])
    st = distribution_stats(combined)
    print(
        f"  Raw distribution: mean={st['mean']:+.4f}  std={st['std']:.4f}  "
        f"skew={st['skew']:+.3f}  excess_kurt={st['ex_kurt']:+.3f}"
    )
    print()


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

TENSOR_CONFIGS = [
    # (display_name, generator_fn, size_for_eval)
    ("KV Cache",          gen_kv_cache,   65_536),
    ("Activations",       gen_activations, 65_536),
    ("Embeddings",        gen_embeddings,  65_536),
    ("MoE Weights",       gen_moe_weights, 65_536),
    ("AllReduce Buffers", gen_allreduce,   65_536),
]

BIT_RANGE = range(2, 9)   # 2-bit through 8-bit inclusive

# Codebook cache: (num_levels, tensor_type_name) → (boundaries, centroids)
_CODEBOOK_CACHE: dict = {}


def get_codebook(num_bits: int, name: str, generator, n_blocks: int = 100_000):
    key = (num_bits, name)
    if key not in _CODEBOOK_CACHE:
        num_levels = 1 << num_bits
        _CODEBOOK_CACHE[key] = compute_lloyd_max_codebook(
            num_levels,
            sample_generator=lambda sz: generator(sz),
            n_blocks=n_blocks,
        )
    return _CODEBOOK_CACHE[key]


def run_tensor_benchmark(
    name: str,
    generator,
    eval_size: int,
    num_trials: int = 5,
    n_codebook_blocks: int = 100_000,
) -> list:
    """
    Benchmark Lloyd-Max vs Uniform for one tensor type across all bit widths.

    Returns a list of dicts with keys:
      bits, unif_sqnr, lm_sqnr, improvement_db, mse_reduction_pct
    """
    results = []

    for num_bits in BIT_RANGE:
        boundaries, centroids = get_codebook(
            num_bits, name, generator, n_blocks=n_codebook_blocks
        )

        sqnr_unif_list = []
        sqnr_lm_list   = []
        mse_unif_list  = []
        mse_lm_list    = []

        for trial in range(num_trials):
            torch.manual_seed(1000 + trial * 97)   # reproducible trials
            raw  = generator(eval_size)

            # Align to BLOCK_SIZE
            aligned = (raw.numel() // BLOCK_SIZE) * BLOCK_SIZE
            data    = raw.reshape(-1)[:aligned]

            # Apply Hadamard rotation (ExLlamaV3 pre-quantization step)
            data_h = hadamard_transform(data)

            # Quantize
            recon_unif = quantize_uniform(data_h, num_bits)
            recon_lm   = quantize_lloyd_max(data_h, num_bits, boundaries, centroids)

            sqnr_unif_list.append(sqnr_db(data_h, recon_unif))
            sqnr_lm_list.append(sqnr_db(data_h, recon_lm))
            mse_unif_list.append(mse(data_h, recon_unif))
            mse_lm_list.append(mse(data_h, recon_lm))

        def avg(lst):
            return sum(lst) / len(lst)

        u_sqnr = avg(sqnr_unif_list)
        l_sqnr = avg(sqnr_lm_list)
        u_mse  = avg(mse_unif_list)
        l_mse  = avg(mse_lm_list)

        imp_db     = l_sqnr - u_sqnr
        mse_red_pct = 100.0 * (u_mse - l_mse) / max(u_mse, 1e-20)

        results.append({
            "bits":             num_bits,
            "unif_sqnr":        u_sqnr,
            "lm_sqnr":          l_sqnr,
            "improvement_db":   imp_db,
            "mse_reduction_pct": mse_red_pct,
        })

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_tensor_table(name: str, results: list) -> None:
    print(f"=== {name} ===")
    print(
        f"{'Bits':<6} {'Uniform SQNR':>14} {'LloydMax SQNR':>15} "
        f"{'Improvement':>13} {'MSE reduction':>14}"
    )
    print("-" * 64)
    for r in results:
        bits    = r["bits"]
        u_sqnr  = r["unif_sqnr"]
        l_sqnr  = r["lm_sqnr"]
        imp_db  = r["improvement_db"]
        mse_pct = r["mse_reduction_pct"]

        u_str   = f"{u_sqnr:>7.2f} dB"
        l_str   = f"{l_sqnr:>7.2f} dB"
        imp_str = f"{imp_db:>+7.2f} dB"
        mse_str = f"{mse_pct:>7.1f}%"

        print(f"{bits:<6} {u_str:>14} {l_str:>15} {imp_str:>13} {mse_str:>14}")
    print()


def verdict(results: list) -> tuple:
    """
    Derive a human-readable verdict from per-bit results.

    Sweet-spot bits: 3–4 (most common low-bit quantization targets).
    """
    sweet = [r for r in results if r["bits"] in (3, 4)]
    if not sweet:
        return "3-4", "N/A", "UNKNOWN"

    best = max(sweet, key=lambda r: r["mse_reduction_pct"])
    bits_range = f"{min(r['bits'] for r in sweet)}-{max(r['bits'] for r in sweet)}"
    imp_str = f"{best['mse_reduction_pct']:.0f}% MSE / {best['improvement_db']:+.2f} dB"

    if best["mse_reduction_pct"] >= 15.0:
        verd = "RECOMMENDED"
    elif best["mse_reduction_pct"] >= 7.0:
        verd = "MARGINAL BENEFIT"
    else:
        verd = "MINIMAL BENEFIT"

    return bits_range, imp_str, verd


def print_summary(all_results: list) -> None:
    sep = "=" * 72
    print(sep)
    print("=== SUMMARY: Where does Lloyd-Max help? ===")
    print(sep)
    print(
        f"{'Tensor Type':<22} {'Best Bits':>10} {'Best Improvement':>24} {'Verdict':<18}"
    )
    print("-" * 72)
    for name, results in all_results:
        b_range, imp_str, verd = verdict(results)
        print(f"{name:<22} {b_range:>10} {imp_str:>24} {verd:<18}")
    print()
    print("Notes:")
    print("  - 'Best Bits' reports the 3-4 bit sweet spot common in LLM inference.")
    print("  - MSE and dB values are averages over multiple evaluation trials.")
    print("  - Codebooks are computed on the ACTUAL normalized post-Hadamard")
    print("    distribution of each tensor type (not on raw N(0,1)).")
    print("  - RECOMMENDED:      > 15% MSE reduction — clear win, worth implementing.")
    print("  - MARGINAL BENEFIT:  7-15% MSE reduction — situational.")
    print("  - MINIMAL BENEFIT:   < 7%  MSE reduction — stick with uniform.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 72)
    print(" Lloyd-Max vs Uniform Quantization — LLM Tensor Type Benchmark")
    print("=" * 72)
    print(f" PyTorch {torch.__version__}  |  Python {sys.version.split()[0]}")
    print(f" Block size : {BLOCK_SIZE}")
    print(f" Bit widths : {list(BIT_RANGE)}")
    print(f" Eval trials: 5 per cell")
    print(f" Codebook   : k-means on actual normalized post-Hadamard distribution")
    print(f" Hadamard   : enabled (matches ExLlamaV3 approach)")
    print("=" * 72)
    print()

    print("Phase 1: Pre-computing per-tensor Lloyd-Max codebooks ...")
    print("  (One codebook per tensor type per bit width = "
          f"{len(TENSOR_CONFIGS) * len(list(BIT_RANGE))} total)")
    print()

    for name, generator, eval_size in TENSOR_CONFIGS:
        print(f"  Precomputing codebooks for: {name}")
        for num_bits in BIT_RANGE:
            t0 = time.time()
            b, c = get_codebook(num_bits, name, generator)
            dt = time.time() - t0
            print(
                f"    {num_bits}-bit: {1 << num_bits:3d} levels  "
                f"centroid range [{c[0]:.4f}, {c[-1]:.4f}]  ({dt:.1f}s)"
            )
        print()

    print("Phase 2: Running evaluations ...")
    print()

    all_results = []

    for name, generator, eval_size in TENSOR_CONFIGS:
        print(f"Benchmarking: {name}")
        print_distribution_summary(name, generator, size=65_536)

        results = run_tensor_benchmark(
            name, generator, eval_size, num_trials=5
        )
        print_tensor_table(name, results)
        all_results.append((name, results))

    print_summary(all_results)


if __name__ == "__main__":
    main()
