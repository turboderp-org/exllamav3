"""
TQ3 weight quantization: FP16 weight -> TQ3 packed format.

Steps:
1. Apply Hadamard rotation (same su/sv sign vectors as EXL3)
2. For each 32-element block along input dimension:
   a. Compute scale = max(|block|)
   b. Normalize to [-1, 1]
   c. Apply Lloyd-Max ternary quantization (boundary at +/- 0.5)
   d. Pack into 2 bitplanes
3. Store: tq3_packed (uint32), tq3_scale (fp16), suh, svh
"""

import torch
import numpy as np

TQ3_BOUNDARY = 0.5  # Lloyd-Max decision boundary (normalized)
HAD_BLOCK = 128     # Hadamard block size (must match CUDA kernel)


def _apply_had_block(w: torch.Tensor, dim: int, block_size: int) -> torch.Tensor:
    """Apply block-diagonal WHT along specified dimension."""
    assert w.shape[dim] % block_size == 0
    shape = list(w.shape)
    num_blocks = shape[dim] // block_size

    # Reshape to isolate blocks
    new_shape = shape[:dim] + [num_blocks, block_size] + shape[dim+1:]
    w = w.reshape(new_shape).float()

    # In-place butterfly WHT
    n = block_size
    h = 1
    while h < n:
        idx1 = torch.arange(0, n, 2 * h, device=w.device)
        for offset in range(h):
            i = idx1 + offset
            j = i + h
            # Vectorized butterfly
            a = torch.index_select(w, dim + 1, i)
            b = torch.index_select(w, dim + 1, j)
            w.index_copy_(dim + 1, i, a + b)
            w.index_copy_(dim + 1, j, a - b)
        h *= 2

    w = w / (block_size ** 0.5)
    return w.reshape(shape).half()


def generate_random_signs(size: int, device: torch.device, seed: int = 42) -> torch.Tensor:
    """Generate random +/- 1 sign vector for Hadamard rotation."""
    rng = torch.Generator(device='cpu')
    rng.manual_seed(seed)
    signs = torch.randint(0, 2, (size,), generator=rng, device='cpu').float() * 2 - 1
    return signs.half().to(device)


def quantize_tq3(
    weight: torch.Tensor,          # (in_features, out_features), float or half
    suh: torch.Tensor | None = None,
    svh: torch.Tensor | None = None,
    progress_str: str | None = None,
) -> dict:
    """
    Quantize a weight matrix to TQ3 format.

    Args:
        weight: (in_features, out_features) weight matrix
        suh: optional pre-rotation signs, shape (in_features,)
        svh: optional post-rotation signs, shape (out_features,)

    Returns:
        dict with keys: tq3_packed, tq3_scale, suh, svh
    """
    device = weight.device
    in_features, out_features = weight.shape
    assert in_features % 32 == 0, "in_features must be divisible by 32"

    w = weight.float()

    # Apply Hadamard pre-rotation
    if suh is not None:
        w = w * suh.float().unsqueeze(1)
        w = _apply_had_block(w.half(), dim=0, block_size=HAD_BLOCK).float()

    if svh is not None:
        w = w * svh.float().unsqueeze(0)
        w = _apply_had_block(w.half(), dim=1, block_size=HAD_BLOCK).float()

    num_blocks = in_features // 32

    # Reshape to blocks: (num_blocks, 32, out_features)
    w_blocks = w.reshape(num_blocks, 32, out_features)

    # Per-block scales
    scales = w_blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
    scales_flat = scales.squeeze(1)  # (num_blocks, out_features)

    # Normalize
    w_norm = w_blocks / scales

    # Lloyd-Max ternary quantization
    # trit = 0 if |w_norm| < boundary, else sign(w_norm)
    nonzero = (w_norm.abs() >= TQ3_BOUNDARY).int()
    positive = ((w_norm > 0) & (nonzero == 1)).int()

    # Pack into bitplanes: one uint32 per 32 values
    # nonzero bits
    bit_indices = torch.arange(32, device=device).view(1, 32, 1)
    bp0 = (nonzero << bit_indices).sum(dim=1).to(torch.int32)   # (num_blocks, out_features)
    bp1 = (positive << bit_indices).sum(dim=1).to(torch.int32)  # (num_blocks, out_features)

    # Interleave bitplanes: row 2*i = bp0[i], row 2*i+1 = bp1[i]
    tq3_packed = torch.zeros(num_blocks * 2, out_features, dtype=torch.int32, device=device)
    tq3_packed[0::2, :] = bp0
    tq3_packed[1::2, :] = bp1

    tq3_scale = scales_flat.half()

    return {
        "tq3_packed": tq3_packed,
        "tq3_scale": tq3_scale,
        "suh": suh,
        "svh": svh,
    }
