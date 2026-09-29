"""
Codec for trellis-quantized embedding rows: hashed n-gram tables (the exl3_ngram_trellis format
produced by util/convert_ngram.py) and token embedding tables.

A ring is a tail-biting trellis over the mul1 codebook plus an fp16 scale, packed as
(1 + dim * K / 16) little-endian uint16 words (stored as int16): word 0 holds the scale's bit
pattern, the remaining words hold the dim * K bit ring bitstream where stream bits
[i*K, (i+1)*K) are the low K bits of position i's 16-bit trellis state. The state's higher bits
are the symbols of the preceding positions, K bits each (mod dim).

N-gram rows are a single 160-wide ring with a per-hash-head bias vector:

    row[i] = decode_mul1(state_i) * scale + head_bias[head]

Token embedding rows are zero-padded to a multiple of 256 and split into 256-wide groups, one
ring each, stored back to back. Groups are rotated before quantization, so the elements the
trellis sees are near Gaussian whatever the table's own statistics:

    group = signs[g] * hadamard(decode_mul1(states) * scale)
"""

from __future__ import annotations
import torch

ROW_DIM = 160       # n-gram row
GROUP_DIM = 256     # token embedding group
MUL1 = 0x83DCD12D


def words_per_row(K: int, dim: int = ROW_DIM) -> int:
    return 1 + dim * K // 16


def ring_dim(words: int, K: int) -> int:
    return (words - 1) * 16 // K


def mul1_codebook(device) -> torch.Tensor:
    """All 65536 decoded mul1 values, bit-exact with decode_3inst<2> (fp16)."""
    s = torch.arange(65536, dtype = torch.int64, device = device)
    prod = (s * MUL1) & 0xFFFFFFFF
    bsum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + ((prod >> 24) & 255)
    h = (1024 + bsum).float()
    k_inv = torch.tensor([0x1eee], dtype = torch.uint16).view(torch.float16).float().item()
    k_bias = torch.tensor([0xc931], dtype = torch.uint16).view(torch.float16).float().item()
    return (h * k_inv + k_bias).to(torch.float16)


def pack_rows(states: torch.Tensor, scales_f16: torch.Tensor, K: int) -> torch.Tensor:
    """
    states: (N, dim) int16/int32/int64 trellis states from quantize_tiles
    scales_f16: (N,) float16 ring scales
    Returns (N, 1 + dim * K / 16) int16 packed rings.
    """
    N, dim = states.shape
    dev = states.device
    new_bits = states.to(torch.int64) & ((1 << K) - 1)                             # (N, dim)
    bits = (new_bits.unsqueeze(-1) >> torch.arange(K, device = dev)) & 1           # (N, dim, K)
    bits = bits.reshape(N, dim * K // 16, 16)
    words = (bits << torch.arange(16, device = dev)).sum(dim = -1)
    words = (words & 0xFFFF).to(torch.uint16).view(torch.int16)
    scale_words = scales_f16.to(torch.float16).view(torch.int16).unsqueeze(1)
    return torch.cat((scale_words, words), dim = 1).contiguous()


def unpack_rows(packed: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of pack_rows: returns (states (N, dim) int64, scales (N,) float16)."""
    dev = packed.device
    words = packed.shape[1] - 1
    dim = ring_dim(packed.shape[1], K)
    scales = packed[:, 0].contiguous().view(torch.float16)
    stream = packed[:, 1:].contiguous().view(torch.uint16).to(torch.int64)
    # Symbol of every position, read through a two-word window
    bit = torch.arange(dim, device = dev) * K
    window = stream[:, bit >> 4] | (stream[:, ((bit >> 4) + 1) % words] << 16)
    symbols = (window >> (bit & 15)) & ((1 << K) - 1)
    # State i stacks the symbols of positions i, i - 1, ... from bit 0 up
    states = torch.zeros_like(symbols)
    for j in range((15 + K) // K):
        states |= torch.roll(symbols, j, dims = 1) << (j * K)
    return states & 0xFFFF, scales


def hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform over the last dimension (a power of two), in the
    butterfly order of the dequant kernel."""
    shape = x.shape
    n = shape[-1]
    x = x.reshape(-1, n)
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        x = torch.stack((x[:, :, 0] + x[:, :, 1], x[:, :, 0] - x[:, :, 1]), dim = 2)
        h *= 2
    return (x.reshape(shape) * (n ** -0.5))


def dequant_rows(
    packed: torch.Tensor,
    K: int,
    codebook: torch.Tensor,
    bias: torch.Tensor | None = None,
    signs: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    packed: (N, words) int16 rings
    bias: (N, dim) per-ring bias (already gathered per head), n-gram rows
    signs: (N, dim) or (dim,) sign vector of rotated rings, token embedding groups
    """
    states, scales = unpack_rows(packed, K)
    out = codebook[states].float() * scales.float().unsqueeze(1)
    if signs is not None:
        out = hadamard_rotate(out) * signs.float()
    if bias is not None:
        out = out + bias.float()
    return out
