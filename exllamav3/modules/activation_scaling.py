import torch


def scale_activation(x, multiplier, *, squared = False, out_dtype = torch.half):
    """Scale rows into FP16 range and return the compensation for the rescaled result."""
    scale = (x.abs().amax(dim = -1, keepdim = True) * multiplier).clamp_min(1)
    return (x / scale).to(out_dtype), scale.square() if squared else scale
