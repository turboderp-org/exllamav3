import math

import torch


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


def geodesic_update(residual, update, scale, bias, depth):
    """Apply a tangent update while preserving the FP32 residual norm."""
    return geodesic_update_torch(residual, update, scale, bias, depth)
