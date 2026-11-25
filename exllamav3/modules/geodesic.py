import math

import torch
import triton
import triton.language as tl


def geodesic_update_torch(residual, update, scale, bias, depth, *, fuse_pointwise = False):
    norm_squared = residual.square().sum(dim = -1, keepdim = True).clamp_min(1e-12)
    update_scale = update.abs().amax(dim = -1, keepdim = True).clamp_min(1)
    update = update / update_scale
    dot = (residual * update).sum(dim = -1, keepdim = True)
    # Preserve the reduction order used during prefill; fuse only the elementwise arithmetic.
    if fuse_pointwise:
        width = residual.shape[-1]
        grid = (residual.numel() // width,)
        tile = triton.next_power_of_2(width)
        tangent = torch.empty_like(update)
        _tangent[grid](residual, update, dot, norm_squared, tangent, width, tile, enable_fp_fusion = False)
    else:
        tangent = update - residual * dot / norm_squared
    tangent_norm = torch.maximum(tangent.norm(dim = -1, keepdim = True), 1e-8 / update_scale)
    radius = residual.norm(dim = -1, keepdim = True).clamp_min(1e-6)
    angle = ((tangent_norm / radius) * update_scale).clamp_max(math.pi / 4)
    angle = ((angle * scale + bias) / depth).clamp_max(math.pi / 4)
    if fuse_pointwise:
        output = torch.empty_like(residual)
        _apply_angle[grid](residual, tangent, tangent_norm, radius, angle.cos(), angle.sin(), output, width, tile,
                           enable_fp_fusion = False)
        return output
    return residual * angle.cos() + (tangent / tangent_norm) * radius * angle.sin()


@triton.jit
def _tangent(X, U, Dot, Norm2, T, WIDTH: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, TILE)
    index = row * WIDTH + i
    x = tl.load(X + index, i < WIDTH, 0)
    u = tl.load(U + index, i < WIDTH, 0)
    tl.store(T + index, u - tl.div_rn(x * tl.load(Dot + row), tl.load(Norm2 + row)), i < WIDTH)


@triton.jit
def _apply_angle(X, T, Norm, Radius, Cos, Sin, Y, WIDTH: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, TILE)
    index = row * WIDTH + i
    x = tl.load(X + index, i < WIDTH, 0)
    t = tl.load(T + index, i < WIDTH, 0)
    direction = tl.div_rn(t, tl.load(Norm + row)) * tl.load(Radius + row)
    tl.store(Y + index, x * tl.load(Cos + row) + direction * tl.load(Sin + row), i < WIDTH)


@triton.jit
def _geodesic(Residual, Update, Scale, Bias, Output, Compensation,
              Width: tl.constexpr, Tile: tl.constexpr, Depth: tl.constexpr):
    i = tl.arange(0, Tile)
    p = tl.program_id(0) * Width + i
    x = tl.load(Residual + p, i < Width, 0)
    u = tl.load(Update + p, i < Width, 0)
    if Compensation is not None:
        u *= tl.load(Compensation + tl.program_id(0))
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


def geodesic_update(residual, update, scale, bias, depth, *, use_triton = True,
                    compensation = None, preserve_reductions = False):
    """Apply a tangent update while preserving the FP32 residual norm."""
    eligible = (
        use_triton and residual.is_cuda and residual.is_contiguous() and update.is_contiguous()
        and residual.shape == update.shape and residual.dtype == update.dtype == torch.float32
        and isinstance(scale, torch.Tensor) and isinstance(bias, torch.Tensor)
        and scale.device == bias.device == update.device == residual.device
        and scale.numel() == bias.numel() == 1
    )
    per_row_scale = compensation is None or (
        compensation.is_contiguous() and compensation.device == update.device
        and compensation.dtype == torch.float32 and compensation.shape == (*update.shape[:-1], 1)
    )
    if preserve_reductions or not (eligible and per_row_scale):
        if compensation is not None:
            update = update * compensation
        return geodesic_update_torch(residual, update, scale, bias, depth,
                                     fuse_pointwise = preserve_reductions and eligible and per_row_scale)
    width = residual.shape[-1]
    output = torch.empty_like(residual)
    _geodesic[(residual.numel() // width,)](
        residual, update, scale, bias, output, compensation, width, triton.next_power_of_2(width), depth,
        enable_fp_fusion=False,
    )
    return output
