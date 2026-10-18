import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def nemotronh_omni_round_to_divisor(rows: int, cols: int, divisor: int, budget: int) -> tuple[int, int]:
    """
    Round each grid dimension to a multiple of the pixel-shuffle factor, rounding up when the
    result stays within the patch budget and down otherwise.
    """
    rem = rows % divisor
    if rem:
        inc = divisor - rem
        rows = rows + inc if (rows + inc) * cols <= budget else max(divisor, rows - rem)
    rem = cols % divisor
    if rem:
        inc = divisor - rem
        cols = cols + inc if rows * (cols + inc) <= budget else max(divisor, cols - rem)
    return rows, cols


def nemotronh_omni_image_grid(
    size: tuple,
    patch_size: int,
    budget: int,
    min_num_patches: int,
    divisor: int,
) -> tuple[int, int]:
    """
    Dynamic-resolution patch grid for a single image. `size` is (width, height) in pixels (PIL
    convention). The image is scaled down to fit `budget` patches, scaled up to `min_num_patches` if
    below it, then each dimension is rounded to a multiple of `divisor`. Returns (rows, cols).
    """
    width, height = size
    closest_rows = round(height / patch_size + 0.5)
    closest_cols = round(width / patch_size + 0.5)
    factor = min(math.sqrt(budget / (closest_rows * closest_cols)), 1.0)
    rows = math.floor(factor * closest_rows)
    cols = math.floor(factor * closest_cols)
    if budget > min_num_patches and rows * cols < min_num_patches:
        up = math.sqrt(min_num_patches / (rows * cols))
        rows, cols = math.ceil(up * rows), math.ceil(up * cols)
    return nemotronh_omni_round_to_divisor(rows, cols, divisor, budget)


def nemotronh_omni_video_grid(
    size: tuple,
    target_num_patches: int,
    divisor: int,
    maintain_aspect_ratio: bool = True,
) -> tuple[int, int]:
    """
    Fixed-budget patch grid for video frames: the aspect-preserving grid nearest to
    `target_num_patches`, each dimension rounded to a multiple of `divisor` (up when both rounded
    dimensions still fit the target, else down). Returns (rows, cols).
    """
    if not maintain_aspect_ratio:
        side = int(math.sqrt(target_num_patches))
        side = max(divisor, (side // divisor) * divisor)
        return side, side
    width, height = size
    aspect = width / max(height, 1)
    rows = max(round(math.sqrt(target_num_patches / aspect)), 1)
    cols = max(round(math.sqrt(target_num_patches * aspect)), 1)
    if divisor > 1:
        rows_up = rows + (-rows % divisor)
        cols_up = cols + (-cols % divisor)
        if rows_up * cols_up <= target_num_patches:
            rows, cols = rows_up, cols_up
        else:
            rows = max(divisor, rows - rows % divisor)
            cols = max(divisor, cols - cols % divisor)
    return rows, cols


def nemotronh_omni_image_to_tensor(
    image: Image.Image,
    size: tuple,
    mean: list,
    std: list,
) -> torch.Tensor:
    """
    Resize to `size` = (height, width) pixels with antialiased bicubic interpolation (the reference
    processor's torch resample, which PIL's bicubic does not reproduce) and normalize. Returns a
    float (3, height, width) tensor.
    """
    arr = np.array(image.convert("RGB"), dtype = np.uint8)
    pixels = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    if pixels.shape[-2:] != tuple(size):
        pixels = F.interpolate(pixels, size = tuple(size), mode = "bicubic", align_corners = False, antialias = True)
    mean = torch.tensor(mean, dtype = torch.float).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype = torch.float).view(1, 3, 1, 1)
    return ((pixels / 255.0 - mean) / std)[0]


def nemotronh_omni_patchify(pixels: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    (n, C, H, W) -> (n, rows * cols, C * p * p) patches in raster order, each flattened as
    (channel, patch row, patch column). For a video tubelet the n temporal frames are stacked
    along C beforehand, so the flattened layout is (frame, channel, py, px) as the video embedder
    expects.
    """
    n, c, height, width = pixels.shape
    rows, cols = height // patch_size, width // patch_size
    patches = pixels.reshape(n, c, rows, patch_size, cols, patch_size)
    return patches.permute(0, 2, 4, 1, 3, 5).reshape(n, rows * cols, c * patch_size ** 2)


def nemotronh_omni_frame_labels(
    num_frames: int,
    temporal_patch_size: int,
    fps: float | None = None,
    frame_indices: list[int] | None = None,
) -> list[str]:
    """
    One text label per temporal patch (tubelet) of a video, as the reference processor writes
    them ahead of each tubelet's image block: "Frame 1 sampled at 0.00 seconds and frame 2 sampled
    at 0.50 seconds: ". With `frame_indices` (source frame numbers of the sampled frames) and
    `fps` (source frame rate) the timestamps are those of the source frames; with `fps` alone the
    frames are assumed consecutive at that rate; without `fps` no timestamps are written.
    """
    frame_duration = int(1000.0 / fps) / 1000.0 if fps else None
    labels = []
    for g in range(-(-num_frames // temporal_patch_size)):
        parts = []
        for j in range(temporal_patch_size):
            fi = g * temporal_patch_size + j
            if fi >= num_frames:
                break
            prefix = "Frame" if j == 0 else "frame"
            if fps and frame_indices is not None and fi < len(frame_indices):
                parts.append(f"{prefix} {fi + 1} sampled at {int(frame_indices[fi]) * frame_duration:.2f} seconds")
            elif fps:
                parts.append(f"{prefix} {fi + 1} sampled at {fi / fps:.2f} seconds")
            else:
                parts.append(f"{prefix} {fi + 1}")
        labels.append(" and ".join(parts) + ": ")
    return labels


def nemotronh_omni_video_retention_mask(
    embeddings: torch.Tensor,
    pruning_rate: float,
) -> torch.Tensor:
    """
    Efficient video sampling: keep the (1 - pruning_rate) share of tokens over all tubelets whose
    cosine dissimilarity to the same position in the previous tubelet is largest. The first
    tubelet always ranks first, and at least one tubelet's worth of tokens is kept.
    `embeddings` is (tubelets, rows, cols, dim); returns a (tubelets, rows * cols) bool mask.
    """
    t, h, w, _ = embeddings.shape
    if pruning_rate <= 0 or t <= 1:
        return torch.ones((t, h * w), dtype = torch.bool)
    similarity = F.cosine_similarity(embeddings[1:].float(), embeddings[:-1].float(), dim = -1)
    dissimilarity = torch.cat((255 * torch.ones((1, h, w)), 1 - similarity), dim = 0).reshape(-1)
    keep = max(h * w, int(t * h * w * (1 - pruning_rate)))
    indices = torch.argsort(dissimilarity, descending = True, stable = True)[:keep]
    mask = torch.zeros_like(dissimilarity, dtype = torch.bool)
    mask[indices] = True
    return mask.reshape(t, h * w)
