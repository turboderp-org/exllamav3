import torch
import triton
import triton.language as tl


@triton.jit
def _scale_activation(Input, Output, Scale,
                      Width: tl.constexpr, Tile: tl.constexpr, Multiplier: tl.constexpr, Squared: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, Tile)
    x = tl.load(Input + row * Width + i, i < Width, 0)
    scale = tl.maximum(tl.max(tl.abs(x), 0) * Multiplier, 1.)
    tl.store(Scale + row, scale * scale if Squared else scale)
    tl.store(Output + row * Width + i, x / scale, i < Width)


def scale_activation(x, multiplier, *, squared = False, out_dtype = torch.half, use_triton = True):
    """Scale rows into FP16 range and return the compensation for the rescaled result."""
    if not (use_triton and x.is_cuda and x.is_contiguous() and x.dtype == torch.float32):
        scale = (x.abs().amax(dim = -1, keepdim = True) * multiplier).clamp_min(1)
        return (x / scale).to(out_dtype), scale.square() if squared else scale

    width = x.shape[-1]
    scaled = torch.empty_like(x, dtype = out_dtype)
    scale = x.new_empty((*x.shape[:-1], 1))
    _scale_activation[(x.numel() // width,)](
        x, scaled, scale, width, triton.next_power_of_2(width), multiplier, squared,
    )
    return scaled, scale
