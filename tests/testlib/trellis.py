"""
Independent NumPy / torch references for the EXL3 trellis format: codebook decode, the packed bitstream layout,
tile element order and dense dequantization. Nothing here calls the extension, so the pack/unpack, reconstruct,
quantizer and GEMV tests can use these as references rather than another kernel.

Format (as written by the converter: quantize_tiles -> pack_trellis, read by every reconstruct/GEMM kernel):

- A 16x16 weight tile is a ring of 256 positions. Position i advances the bitstream by D(i) bits: D(i) = K for
  integer K, D(i) = KA + bit (i mod 16) of MASK for the half-integer rates (MASK 0xAAAA, K = KA + 0.5). S(i) is the
  prefix sum D(0) + .. + D(i); the ring holds R = sum D = 256 * K bits.
- Position i's 16-bit codeword ("window") is ring bits [S(i) - 16, S(i)) read MSB first, wrapping mod R. So its low
  D(i) bits are the position's own new bits and its top 16 - D(i) bits are the previous window's low bits
  (tail-biting).
- Packed tile: the ring as 32-bit words, ring bit 0 at the MSB of word 0, stored little-endian as 2 * (R / 32) uint16
  (so uint16 2j is the low half of word j). 16 * K uint16 per tile.
- Ring position i holds row-major tile element tile_perm()[i] (the tensor-core fragment order).
- Codebooks: idx -> fp16 value; 3INST (cb 0), MCG (cb 1) and mul1 (cb 2), see decode().
- Dense weight: W_hat[16 tk + r, 16 tn + c] = decode(window) of tile (tk, tn); the EXL3 linear is
  W = diag(suh) H W_hat H diag(svh) with H the 128x128 Sylvester Hadamard / sqrt(128) applied per 128-block.
"""

import numpy as np
import torch

CODEBOOKS = ("3inst", "mcg", "mul1")
FRAC_MASK = 0xAAAA


def frac(K) -> tuple[int, int] | None:
    """(KA, MASK) of a half-integer rate, None for integer K"""
    if float(K).is_integer():
        return None
    ka = int(K)
    assert ka + 0.5 == K
    return ka, FRAC_MASK


def step_widths(K) -> np.ndarray:
    """(256,) D(i), bits each ring position adds"""
    f = frac(K)
    if f is None:
        return np.full(256, int(K), dtype = np.int64)
    ka, mask = f
    return np.array([ka + ((mask >> (i & 15)) & 1) for i in range(256)], dtype = np.int64)


def tile_perm() -> np.ndarray:
    """(256,) row-major tile element held by ring position i"""
    perm = np.zeros(256, dtype = np.int64)
    for t in range(32):
        for j in range(8):
            r = (t % 4) * 2 + (j & 1) + (8 if j & 2 else 0)
            c = t // 4 + (8 if j & 4 else 0)
            perm[t * 8 + j] = r * 16 + c
    return perm


def _f16_bits(u: np.ndarray) -> np.ndarray:
    return u.astype(np.uint16).view(np.float16)


def decode(idx, codebook: str) -> np.ndarray:
    """fp16 codebook value of every 16-bit index (any integer array; only the low 16 bits count)"""
    x = np.asarray(idx).astype(np.int64) & 0xFFFF
    x = x.astype(np.uint64)
    M = np.uint64(0xFFFFFFFF)
    if codebook in ("3inst", "mcg"):
        if codebook == "3inst":
            x = (x * np.uint64(89226354) + np.uint64(64248484)) & M
        else:
            x = (x * np.uint64(0xCBAC1FED)) & M
        x = (x & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)
        lo = _f16_bits(x & np.uint64(0xFFFF)).astype(np.float64)
        hi = _f16_bits(x >> np.uint64(16)).astype(np.float64)
        return (lo + hi).astype(np.float16)    # exact in float64, one fp16 rounding (= __hadd)
    assert codebook == "mul1", codebook
    x = (x * np.uint64(0x83DCD12D)) & M
    bsum = sum((x >> np.uint64(s)) & np.uint64(0xFF) for s in (0, 8, 16, 24)).astype(np.int64)
    h = (1024 + bsum).astype(np.float64)       # fp16 bits 0x6400 + bytesum = 1024 + bytesum exactly
    k_inv = float(_f16_bits(np.array(0x1EEE)))
    k_bias = float(_f16_bits(np.array(0xC931)))
    return (h * k_inv + k_bias).astype(np.float16)   # exact in float64, one fp16 rounding (= __hfma)


def ring_ends(K) -> np.ndarray:
    """(256,) S(i)"""
    return np.cumsum(step_widths(K))


def _ring_bits(packed: np.ndarray) -> np.ndarray:
    """(T, words) uint16/int16 packed tiles -> (T, 16 * words) ring bits, MSB first"""
    p = np.ascontiguousarray(packed).view(np.uint16).astype(np.uint32)
    T, W = p.shape
    u32 = p[:, 0::2] | (p[:, 1::2] << 16)
    be = u32.astype(">u4").view(np.uint8).reshape(T, W * 2)
    return np.unpackbits(be, axis = 1)


def unpack(packed, K, chunk: int = 2048) -> np.ndarray:
    """(..., 16 K) packed tiles -> (..., 256) uint16 windows in ring order"""
    packed = np.asarray(packed)
    flat = packed.reshape(-1, packed.shape[-1])
    S = ring_ends(K)
    R = int(S[-1])
    assert flat.shape[1] * 16 == R
    pos = (S[:, None] - 16 + np.arange(16)[None, :]) % R           # (256, 16), MSB first
    weights = (1 << np.arange(15, -1, -1)).astype(np.uint32)
    out = np.empty((flat.shape[0], 256), dtype = np.uint16)
    for i in range(0, flat.shape[0], chunk):
        bits = _ring_bits(flat[i : i + chunk])
        out[i : i + chunk] = (bits[:, pos].astype(np.uint32) * weights).sum(axis = -1)
    return out.reshape(packed.shape[:-1] + (256,))


def pack(words, K) -> np.ndarray:
    """(T, 256) windows -> (T, 16 K) packed int16 tiles. Uses only the low D(i) bits of each window"""
    words = np.asarray(words).astype(np.int64) & 0xFFFF
    lead = words.shape[:-1]
    words = words.reshape(-1, 256)
    T = words.shape[0]
    D = step_widths(K)
    S = np.cumsum(D)
    R = int(S[-1])
    bits = np.zeros((T, R), dtype = np.uint8)
    for i in range(256):
        for b in range(D[i]):
            bits[:, S[i] - D[i] + b] = (words[:, i] >> (D[i] - 1 - b)) & 1
    be = np.packbits(bits, axis = 1).reshape(T, R // 32, 4)
    u32 = be.copy().view(">u4").reshape(T, R // 32).astype(np.uint32)
    out = np.empty((T, R // 16), dtype = np.uint16)
    out[:, 0::2] = u32 & 0xFFFF
    out[:, 1::2] = u32 >> 16
    return out.view(np.int16).reshape(lead + (R // 16,))


def is_tail_biting(words, K) -> bool:
    """Every window's top 16 - D(i) bits equal the previous window's low 16 - D(i) bits, circularly"""
    w = np.asarray(words).astype(np.int64).reshape(-1, 256) & 0xFFFF
    D = step_widths(K)
    prev = np.roll(w, 1, axis = 1)
    return bool(np.all((w >> D) == (prev & ((1 << (16 - D)) - 1))))


def random_packed(tiles: int, K, gen: np.random.Generator) -> np.ndarray:
    """(tiles, 16 K) uniformly random packed tiles (every bit pattern is a valid tile)"""
    return gen.integers(0, 1 << 16, size = (tiles, int(16 * K)), dtype = np.uint16).view(np.int16)


def dequant_rotated(trellis, K, codebook: str) -> np.ndarray:
    """(k / 16, n / 16, 16 K) packed trellis -> (k, n) fp16 W_hat (no Hadamards, no scales)"""
    t = trellis.cpu().numpy() if isinstance(trellis, torch.Tensor) else np.asarray(trellis)
    tk, tn, _ = t.shape
    vals = decode(unpack(t.reshape(tk * tn, -1), K), codebook)       # (T, 256) ring order
    tiles = np.empty_like(vals)
    tiles[:, tile_perm()] = vals
    return tiles.reshape(tk, tn, 16, 16).transpose(0, 2, 1, 3).reshape(tk * 16, tn * 16)


def sylvester(n: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype = torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def dequant(trellis, suh, svh, K, codebook: str) -> torch.Tensor:
    """(k, n) float64 dense weight of the EXL3 linear (CPU)"""
    w = torch.from_numpy(dequant_rotated(trellis, K, codebook).astype(np.float64))
    k, n = w.shape
    H = sylvester(128) / 128 ** 0.5
    w = torch.einsum("ij,bjn->bin", H, w.view(k // 128, 128, n)).reshape(k, n)
    w = torch.einsum("kbi,ij->kbj", w.view(k, n // 128, 128), H).reshape(k, n)
    return w * suh.cpu().double()[:, None] * svh.cpu().double()[None, :]


def linear(x: torch.Tensor, trellis, suh, svh, K, codebook: str) -> torch.Tensor:
    """x @ W in float64 (CPU)"""
    return x.cpu().double() @ dequant(trellis, suh, svh, K, codebook)
