import math

import torch
import triton
import triton.language as tl


def geodesic_update_torch(residual, update, scale, bias, depth):
    norm_squared = residual.square().sum(dim = -1, keepdim = True).clamp_min(1e-12)
    update_scale = update.abs().amax(dim = -1, keepdim = True).clamp_min(1)
    update = update / update_scale
    tangent = update - residual * (residual * update).sum(dim = -1, keepdim = True) / norm_squared
    tangent_norm = torch.maximum(tangent.norm(dim = -1, keepdim = True), 1e-8 / update_scale)
    radius = residual.norm(dim = -1, keepdim = True).clamp_min(1e-6)
    angle = ((tangent_norm / radius) * update_scale).clamp_max(math.pi / 4)
    angle = ((angle * scale + bias) / depth).clamp_max(math.pi / 4)
    return residual * angle.cos() + (tangent / tangent_norm) * radius * angle.sin()


@triton.jit
def _geodesic(Residual, Update, Scale, Bias, Output,
              Width: tl.constexpr, Tile: tl.constexpr, Depth: tl.constexpr):
    i = tl.arange(0, Tile)
    p = tl.program_id(0) * Width + i
    x = tl.load(Residual + p, i < Width, 0)
    u = tl.load(Update + p, i < Width, 0)
    scale = tl.maximum(tl.max(tl.abs(u), 0), 1.0)
    u = u / scale
    norm2 = tl.sum(x * x, 0)
    tangent = u - x * (tl.sum(x * u, 0) / tl.maximum(norm2, 1.e-12))
    tangent_norm = tl.maximum(tl.sqrt(tl.sum(tangent * tangent, 0)), 1.e-8 / scale)
    radius = tl.maximum(tl.sqrt(norm2), 1.e-6)
    angle = tl.minimum((tangent_norm / radius) * scale, math.pi / 4)
    angle = tl.minimum((angle * tl.load(Scale).to(tl.float32) + tl.load(Bias).to(tl.float32)) / Depth,
                       math.pi / 4)
    y = x * tl.cos(angle) + ((tangent / tangent_norm) * radius) * tl.sin(angle)
    tl.store(Output + p, y, i < Width)


def geodesic_update(residual, update, scale, bias, depth, *, use_triton = True):
    """Apply a tangent update while preserving the FP32 residual norm."""
    if not (
        use_triton and residual.is_cuda and residual.is_contiguous() and update.is_contiguous()
        and residual.dtype == update.dtype == torch.float32
        and isinstance(scale, torch.Tensor) and isinstance(bias, torch.Tensor)
        and scale.device == bias.device == residual.device
        and scale.numel() == bias.numel() == 1
    ):
        return geodesic_update_torch(residual, update, scale, bias, depth)
    width = residual.shape[-1]
    output = torch.empty_like(residual)
    _geodesic[(residual.numel() // width,)](
        residual, update, scale, bias, output, width, triton.next_power_of_2(width), depth,
        enable_fp_fusion=False,
    )
    return output
