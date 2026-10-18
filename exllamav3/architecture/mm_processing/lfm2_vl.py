import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .common import find_closest_aspect_ratio


def lfm2_vl_smart_resize(
    size: tuple,
    patch_size: int,
    downsample_factor: int,
    min_image_tokens: int,
    max_image_tokens: int,
) -> tuple:
    """
    Single-tile target size: both dimensions a multiple of patch_size * downsample_factor, pixel count
    within [min_image_tokens, max_image_tokens] worth of merged tokens, aspect ratio preserved as closely
    as possible. `size` is (width, height); returns (width, height).
    """
    width, height = size
    factor = patch_size * downsample_factor
    min_pixels = min_image_tokens * factor ** 2
    max_pixels = max_image_tokens * factor ** 2
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return w_bar, h_bar


def lfm2_vl_is_image_too_large(
    size: tuple,
    patch_size: int,
    downsample_factor: int,
    max_image_tokens: int,
    max_pixels_tolerance: float,
) -> bool:
    """Whether an image is split into tiles rather than resized to a single tile"""
    width, height = size
    factor = patch_size * downsample_factor
    h_bar = max(patch_size, round(height / factor) * factor)
    w_bar = max(patch_size, round(width / factor) * factor)
    return h_bar * w_bar > max_image_tokens * factor ** 2 * max_pixels_tolerance


def lfm2_vl_grid_layout(
    size: tuple,
    min_tiles: int,
    max_tiles: int,
    tile_size: int,
) -> tuple:
    """(cols, rows) tile grid for a large image, over every grid with min_tiles..max_tiles tiles"""
    ratios = sorted({
        (w, h)
        for n in range(min_tiles, max_tiles + 1)
        for w in range(1, n + 1)
        for h in range(1, n + 1)
        if min_tiles <= w * h <= max_tiles
    }, key = lambda r: r[0] * r[1])
    return find_closest_aspect_ratio(size, ratios, tile_size)


def lfm2_vl_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """(3, H, W) uint8 tensor of an image"""
    arr = np.array(image.convert("RGB"), dtype = np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def lfm2_vl_resize(pixels: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Resize a (3, H, W) uint8 tensor to `size` = (height, width) with antialiased bicubic filtering in the
    uint8 domain, which is what torchvision does for the reference processor's uint8 inputs (resample 3)
    """
    if pixels.shape[-2:] == tuple(size):
        return pixels
    return F.interpolate(pixels.unsqueeze(0), size = tuple(size), mode = "bicubic", align_corners = False, antialias = True)[0]


def lfm2_vl_split_tiles(pixels: torch.Tensor, rows: int, cols: int, tile_size: int) -> torch.Tensor:
    """(3, rows * tile, cols * tile) -> (rows * cols, 3, tile, tile) in raster order"""
    c = pixels.shape[0]
    tiles = pixels.view(c, rows, tile_size, cols, tile_size).permute(1, 3, 0, 2, 4)
    return tiles.reshape(rows * cols, c, tile_size, tile_size)


def lfm2_vl_patchify(
    pixels: torch.Tensor,
    patch_size: int,
    rescale_factor: float,
    mean: list,
    std: list,
) -> torch.Tensor:
    """
    (n, 3, H, W) uint8 -> rescaled and normalized (n, rows * cols, p * p * 3) float patches, each flattened
    as (patch row, patch column, channel) like the SigLIP2 processor
    """
    n, c, height, width = pixels.shape
    x = pixels.float() * rescale_factor
    x = (x - torch.tensor(mean, dtype = torch.float).view(1, c, 1, 1)) / torch.tensor(std, dtype = torch.float).view(1, c, 1, 1)
    rows, cols = height // patch_size, width // patch_size
    x = x.reshape(n, c, rows, patch_size, cols, patch_size).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(n, rows * cols, patch_size * patch_size * c)
