"""
Codec for trellis-quantized n-gram embedding tables (the exl3_ngram_trellis format produced by
util/convert_ngram.py).

Each 160-D embedding row is a tail-biting trellis ring over the mul1 codebook plus a per-row fp16
scale and a per-hash-head bias vector. A packed row is (1 + 10*K) little-endian uint16 words
(stored as int16): word 0 holds the fp16 row scale's bit pattern, the remaining words hold the
160*K-bit ring bitstream where stream bits [i*K, (i+1)*K) are the low K bits of position i's
16-bit trellis state. Reconstruction:

    row[i] = decode_mul1(state_i) * scale + head_bias[head]

where state_i is read as the 16 ring bits ending at stream bit (i+1)*K - 1 (mod 160*K).

TODO: CUDA kernels for this
"""

from __future__ import annotations
import torch

ROW_DIM = 160
MUL1 = 0x83DCD12D


def words_per_row(K: int) -> int:
    return 1 + ROW_DIM * K // 16


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
    states: (N, 160) int16/int32/int64 trellis states from quantize_tiles
    scales_f16: (N,) float16 row scales
    Returns (N, 1 + 10*K) int16 packed rows.
    """
    N = states.shape[0]
    dev = states.device
    new_bits = states.to(torch.int64) & ((1 << K) - 1)                             # (N, 160)
    bits = (new_bits.unsqueeze(-1) >> torch.arange(K, device = dev)) & 1           # (N, 160, K)
    bits = bits.reshape(N, ROW_DIM * K // 16, 16)
    words = (bits << torch.arange(16, device = dev)).sum(dim = -1)                 # (N, 10*K)
    words = (words & 0xFFFF).to(torch.uint16).view(torch.int16)
    scale_words = scales_f16.to(torch.float16).view(torch.int16).unsqueeze(1)
    return torch.cat((scale_words, words), dim = 1).contiguous()


def unpack_rows(packed: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of pack_rows: returns (states (N, 160) int64, scales (N,) float16)."""
    dev = packed.device
    scales = packed[:, 0].contiguous().view(torch.float16)
    words = packed[:, 1:].view(torch.uint16).to(torch.int64)                       # (N, 10*K)
    stream = ((words.unsqueeze(-1) >> torch.arange(16, device = dev)) & 1).reshape(packed.shape[0], ROW_DIM * K)
    # state_i bit m lives at stream bit ((i - m // K) mod 160) * K + m % K
    i = torch.arange(ROW_DIM, device = dev).unsqueeze(1)
    m = torch.arange(16, device = dev).unsqueeze(0)
    src = ((i - m // K) % ROW_DIM) * K + m % K                                     # (160, 16)
    states = (stream[:, src] << m).sum(dim = -1)                                   # (N, 160)
    return states, scales


def dequant_rows(
    packed: torch.Tensor,
    K: int,
    codebook: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """packed: (N, 1 + 10*K) int16; bias: (N, 160) per-row bias (already gathered per head) or None."""
    states, scales = unpack_rows(packed, K)
    out = codebook[states].float() * scales.float().unsqueeze(1)
    if bias is not None:
        out = out + bias.float()
    return out
