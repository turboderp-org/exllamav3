from __future__ import annotations
import torch
from ..ext import exllamav3_ext as ext


class TQ3ActivationCompressor:
    """
    Compresses inter-layer hidden states using TQ3 (WHT + Lloyd-Max ternary).

    Usage:
        compressor = TQ3ActivationCompressor()

        # In model forward loop:
        compressed = compressor.compress(hidden_states)  # (batch, seq, hidden) -> TQ3
        hidden_states = compressor.decompress(compressed) # TQ3 -> (batch, seq, hidden)

    Memory savings: ~6.4x compression (fp16 -> 2.5 bpv effective)
    """

    @staticmethod
    def compress(x: torch.Tensor) -> dict:
        """Compress fp16 activation tensor to TQ3 format.

        Args:
            x: Tensor of any shape, dtype float16, on CUDA device.

        Returns:
            dict with keys: packed (int32), scales (fp16), shape, pad.
        """
        orig_shape = x.shape
        x_flat = x.contiguous().view(-1)

        # Ensure multiple of 32
        numel = x_flat.numel()
        pad = (32 - numel % 32) % 32
        if pad > 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, pad))

        num_blocks = x_flat.numel() // 32
        packed = torch.empty(num_blocks * 2, dtype=torch.int32, device=x.device)
        scales = torch.empty(num_blocks, dtype=torch.float16, device=x.device)

        ext.quant_tq3_cache_cont(x_flat, packed, scales)

        return {
            "packed": packed,
            "scales": scales,
            "shape": orig_shape,
            "pad": pad,
        }

    @staticmethod
    def decompress(compressed: dict) -> torch.Tensor:
        """Decompress TQ3 format back to fp16 tensor.

        Args:
            compressed: dict produced by compress().

        Returns:
            fp16 Tensor with original shape restored.
        """
        packed = compressed["packed"]
        scales = compressed["scales"]
        orig_shape = compressed["shape"]
        pad = compressed["pad"]

        num_blocks = packed.numel() // 2
        x_flat = torch.empty(num_blocks * 32, dtype=torch.float16, device=packed.device)

        ext.dequant_tq3_cache_cont(packed, scales, x_flat)

        if pad > 0:
            x_flat = x_flat[:x_flat.numel() - pad]

        return x_flat.view(orig_shape)

    @staticmethod
    def memory_savings(shape: tuple) -> dict:
        """Calculate memory savings for a given tensor shape.

        Args:
            shape: Tuple representing the tensor dimensions.

        Returns:
            dict with fp16_bytes, tq3_bytes, ratio, savings_pct.
        """
        numel = 1
        for s in shape:
            numel *= s
        fp16_bytes = numel * 2
        num_blocks = (numel + 31) // 32
        tq3_bytes = num_blocks * 2 * 4 + num_blocks * 2  # packed (int32) + scales (fp16)
        return {
            "fp16_bytes": fp16_bytes,
            "tq3_bytes": tq3_bytes,
            "ratio": fp16_bytes / tq3_bytes,
            "savings_pct": (1 - tq3_bytes / fp16_bytes) * 100,
        }
