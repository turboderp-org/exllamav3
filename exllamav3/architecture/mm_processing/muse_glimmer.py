import math
import itertools
import numpy as np
import torch

def muse_smart_resize(
    size: tuple,
    patch_size: int,
    max_tokens: int,
):
    """
    Pick the integer patch grid closest to the input aspect ratio under the token cap. `size` is
    (width, height) in pixels (PIL convention), `patch_size` is the merged patch size in pixels
    (model patch size * spatial merge size), `max_tokens` caps the number of merged patches.
    Returns the resize target (width, height) in pixels.
    """
    width, height = size
    ideal_patches_height = height / patch_size
    ideal_patches_width = width / patch_size
    ratio = ideal_patches_width / ideal_patches_height if ideal_patches_height > 0 else 1.0
    if ideal_patches_height * ideal_patches_width > max_tokens:
        ideal_patches_height = (max_tokens / ratio) ** 0.5
        ideal_patches_width = ideal_patches_height * ratio
    candidates = list(set(itertools.product(
        [math.floor(ideal_patches_height), math.ceil(ideal_patches_height)],
        [math.floor(ideal_patches_width), math.ceil(ideal_patches_width)],
    )))
    candidates = [
        (ph, pw) for ph, pw in candidates
        if ph >= 1 and pw >= 1 and ph * pw <= max_tokens
    ]
    if not candidates:
        candidates = [(max(1, round(ideal_patches_height)), max(1, round(ideal_patches_width)))]
    ph, pw = min(candidates, key = lambda grid: abs(grid[0] / grid[1] - height / width))
    return pw * patch_size, ph * patch_size


def muse_patchify(
    image_np: np.ndarray,
    patch_size: int,
    temporal_patch_size: int,
):
    """
    Flatten a (C, H, W) image into (grid_h * grid_w, temporal_patch_size * C * patch_size**2)
    patches in raster order. Each flattened patch is laid out (temporal, channel, py, px), with the
    single frame repeated along the temporal axis.
    """
    c, height, width = image_np.shape
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = image_np.reshape(c, grid_h, patch_size, grid_w, patch_size)
    patches = patches.transpose(1, 3, 0, 2, 4)  # (grid_h, grid_w, c, py, px)
    flat = patches.reshape(grid_h * grid_w, c * patch_size ** 2)
    flat = np.tile(flat, (1, temporal_patch_size))
    return np.ascontiguousarray(flat), grid_h, grid_w


def muse_bilinear_pos_emb(
    grid_h: int,
    grid_w: int,
    side: int,
):
    """
    Bilinear interpolation of a (side x side) learned position embedding table onto a
    (grid_h x grid_w) patch grid, equivalent to F.grid_sample(align_corners = False,
    padding = "zeros"). Returns (indices, weights): (4, grid_h * grid_w) long/float32 tensors such
    that pos_emb = (table[indices] * weights[..., None]).sum(0), rows in raster order.
    """
    h_grid = (torch.arange(grid_h).float() + 0.5) * (side / grid_h) - 0.5
    w_grid = (torch.arange(grid_w).float() + 0.5) * (side / grid_w) - 0.5

    h_floor = torch.floor(h_grid).long()
    w_floor = torch.floor(w_grid).long()
    h_ceil = h_floor + 1
    w_ceil = w_floor + 1
    h_frac = h_grid - h_floor.float()
    w_frac = w_grid - w_floor.float()

    h_floor_valid = (h_floor >= 0) & (h_floor <= side - 1)
    h_ceil_valid = (h_ceil >= 0) & (h_ceil <= side - 1)
    w_floor_valid = (w_floor >= 0) & (w_floor <= side - 1)
    w_ceil_valid = (w_ceil >= 0) & (w_ceil <= side - 1)
    h_floor = h_floor.clamp(0, side - 1)
    h_ceil = h_ceil.clamp(0, side - 1)
    w_floor = w_floor.clamp(0, side - 1)
    w_ceil = w_ceil.clamp(0, side - 1)

    h_floor_offset = h_floor * side
    h_ceil_offset = h_ceil * side

    indices = torch.stack([
        (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
        (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
        (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
        (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
    ])
    weights = torch.stack([
        ((1 - h_frac)[:, None] * (1 - w_frac)[None, :] * (h_floor_valid[:, None] & w_floor_valid[None, :])).flatten(),
        ((1 - h_frac)[:, None] * w_frac[None, :] * (h_floor_valid[:, None] & w_ceil_valid[None, :])).flatten(),
        (h_frac[:, None] * (1 - w_frac)[None, :] * (h_ceil_valid[:, None] & w_floor_valid[None, :])).flatten(),
        (h_frac[:, None] * w_frac[None, :] * (h_ceil_valid[:, None] & w_ceil_valid[None, :])).flatten(),
    ])
    return indices, weights


def muse_position_embedding_grid_2d(
    grid_thw: tuple,
    head_dim: int,
    theta: float,
):
    """
    Per-token rotary frequency grid for the MuseGlimmer vision tower: rows in raster order, each
    row [w_freqs, h_freqs] (head_dim // 2 total), which NEOX duplication turns into the reference
    [freq_w, freq_h, freq_w, freq_h] layout. Positions are 1-indexed per the reference.
    """
    t, h, w = grid_thw
    hpos = torch.arange(h).unsqueeze(1).expand(h, w).flatten()
    wpos = torch.arange(w).unsqueeze(0).expand(h, w).flatten()
    ids = torch.stack([wpos, hpos], dim = -1) + 1
    ids = ids.repeat(t, 1)

    dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype = torch.float) / dim))
    seq = torch.arange(max(h, w) + 1, dtype = torch.float)
    freqs = torch.outer(seq, inv_freq)
    emb = freqs[ids].flatten(1)
    return emb
