"""
Independent references for the sampling kernels: a NumPy reimplementation of cuRAND's Philox4x32-10 generator
(curand_init(seed, subsequence, 0) followed by curand_uniform draws) and the exact Gumbel transform, so tests can
reproduce the noise the kernels add element by element instead of only checking its distribution.

cuRAND's state after curand_init(seed, subsequence, offset = 0): key = (seed & 0xffffffff, seed >> 32), counter =
(0, 0, subsequence & 0xffffffff, subsequence >> 32); draw n of the stream is word n % 4 of
Philox4x32_10(counter + n // 4, key). curand_uniform maps a word x to float(x) * 2^-32 + 2^-33 in float32 (the
product is an exact power-of-two scale, so the single rounding happens in float(x) and in the addition).
"""

import numpy as np

_M0 = np.uint64(0xD2511F53)
_M1 = np.uint64(0xCD9E8D57)
_W0 = np.uint32(0x9E3779B9)
_W1 = np.uint32(0xBB67AE85)
_MASK = np.uint64(0xFFFFFFFF)


def philox4x32_10(ctr: np.ndarray, key: tuple[int, int]) -> np.ndarray:
    """ctr: (n, 4) uint32 counters, key: (k0, k1). Returns the (n, 4) uint32 outputs"""
    c0, c1, c2, c3 = (ctr[:, i].astype(np.uint64) for i in range(4))
    k0 = np.uint32(key[0] & 0xFFFFFFFF)
    k1 = np.uint32(key[1] & 0xFFFFFFFF)
    with np.errstate(over = "ignore"):
        for r in range(10):
            p0 = _M0 * c0
            p1 = _M1 * c2
            hi0, lo0 = p0 >> np.uint64(32), p0 & _MASK
            hi1, lo1 = p1 >> np.uint64(32), p1 & _MASK
            c0, c1, c2, c3 = (
                hi1 ^ c1 ^ np.uint64(k0),
                lo1,
                hi0 ^ c3 ^ np.uint64(k1),
                lo0,
            )
            if r < 9:
                k0 = np.uint32(k0 + _W0)
                k1 = np.uint32(k1 + _W1)
    return np.stack([c0, c1, c2, c3], axis = 1).astype(np.uint32)


def curand_words(seed: int, subsequences: np.ndarray, num_draws: int = 1) -> np.ndarray:
    """First num_draws curand() words of curand_init(seed, subsequence, 0) for each subsequence, (n, num_draws)
    uint32"""
    sub = np.asarray(subsequences, dtype = np.uint64).reshape(-1)
    n = sub.shape[0]
    out = np.empty((n, num_draws), dtype = np.uint32)
    key = (seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF)
    for blk in range((num_draws + 3) // 4):
        ctr = np.zeros((n, 4), dtype = np.uint32)
        # Counter increment by blk on ctr.x (no carry for the small block counts used here)
        ctr[:, 0] = blk
        ctr[:, 2] = (sub & _MASK).astype(np.uint32)
        ctr[:, 3] = (sub >> np.uint64(32)).astype(np.uint32)
        words = philox4x32_10(ctr, key)
        w = min(4, num_draws - 4 * blk)
        out[:, 4 * blk: 4 * blk + w] = words[:, :w]
    return out


def curand_uniform(words: np.ndarray) -> np.ndarray:
    """curand_uniform's float32 mapping of 32-bit words: (0, 1]"""
    return (words.astype(np.float32) * np.float32(2.0 ** -32)) + np.float32(2.0 ** -33)


def gumbel_exact(u: np.ndarray) -> np.ndarray:
    """The kernels' Gumbel transform in float64: u clamped to the largest float32 below 1, then -log(-log(u))
    with the same 1e-20 floors"""
    u = np.minimum(u.astype(np.float64), float(np.float32(0.99999994)))
    return -np.log(np.maximum(-np.log(np.maximum(u, 1e-20)), 1e-20))


def element_noise(seed: int, num_elements: int, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(u, g) of the per-element noise stream curand_init(seed, flat_index, 0) + one curand_uniform, as used by
    gumbel_noise_f32/_log, adaptivep_gumbel_noise_f32 and fused_sampler (flat index row * dim + i)"""
    idx = np.arange(offset, offset + num_elements, dtype = np.uint64)
    u = curand_uniform(curand_words(seed, idx, 1)[:, 0])
    return u, gumbel_exact(u)


def gumbel_noise_tolerance(u: np.ndarray) -> np.ndarray:
    """Per-element bound on |g_kernel - g_exact| for the kernels' __logf-based transform. __logf is
    lg2.approx * ln 2 with absolute error <= ~2^-21.4 on [0.5, 2] and relative error ~2^-22 elsewhere, so the
    inner log carries an absolute error e1 <= 2^-21 * max(1, |ln u|); the outer log of v = -ln u then errs by
    about e1 / v plus its own 2^-21 * max(1, |ln v|). Where e1 / v is not small (u within ~1e-6 of 1) the
    bound degenerates and the comparison is skipped by the caller."""
    u64 = np.minimum(u.astype(np.float64), float(np.float32(0.99999994)))
    v = -np.log(u64)
    e1 = 2.0 ** -21 * np.maximum(1.0, np.abs(np.log(u64)))
    e2 = 2.0 ** -21 * np.maximum(1.0, np.abs(np.log(v)))
    return 2.0 * (e1 / v + e2) + 1e-6


def gumbel_sample_noise(seed: int, num_logits: int, threads: int = 1024, row: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(u, g) of gumbel_sample's noise for row `row` of num_logits: thread t owns the element pairs
    (2t + 2 * threads * j, +1) and draws two uniforms per pair, in order, from curand_init(seed, row * threads + t, 0)"""
    span = 2 * threads
    rounds = (num_logits + span - 1) // span
    subseq = np.arange(threads, dtype = np.uint64) + np.uint64(row * threads)
    words = curand_words(seed, subseq, 2 * rounds)                                    # (threads, 2 * rounds)
    idx = np.arange(num_logits)
    t = (idx % span) // 2
    draw = 2 * (idx // span) + (idx % 2)
    u = curand_uniform(words[t, draw])
    return u, gumbel_exact(u)


def certain_argmax(value: np.ndarray, tol: np.ndarray) -> int | None:
    """Index of the maximum of value if it beats every other element by more than their combined tolerances,
    else None (the reference cannot decide)"""
    w = int(np.argmax(value))
    lo = value[w] - tol[w]
    hi = value + tol
    hi[w] = -np.inf
    return w if lo > hi.max() else None
