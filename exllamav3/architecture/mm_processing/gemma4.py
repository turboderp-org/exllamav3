import math
import numpy as np
import torch

def get_aspect_ratio_preserving_size(
    size: tuple[int, int],
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
    """
    Image is resized to preserve aspect ratio so it fits within the patch budget.
    Target dimensions are the largest that:
    1) Produce at most `max_patches` patches when patchified with `patch_size`
    2) Have height and width divisible by `pooling_kernel_size * patch_size`
    """
    width, height = size

    total_px = height * width
    target_px = max_patches * (patch_size ** 2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to a 0 x 0 image. "
            f"Resized height should be divisible by pooling_kernel_size * patch_size = {side_mult}."
        )

    max_side_length = (max_patches // (pooling_kernel_size ** 2)) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)

    if target_height * target_width > target_px:
        raise ValueError(
            f"Resizing [{width}x{height}] to [{target_width}x{target_height}] exceeds patch budget"
        )

    return target_width, target_height


def pad_patches_and_positions(
    patches: np.ndarray,
    positions: np.ndarray,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    current_length = patches.shape[0]
    if current_length > target_length:
        raise ValueError(
            f"Cannot pad Gemma4 patches from {current_length} down to target length {target_length}"
        )

    padding_length = target_length - current_length
    if padding_length == 0:
        return patches, positions

    patch_paddings = [(0, padding_length)] + [(0, 0)] * (patches.ndim - 1)
    position_paddings = [(0, padding_length), (0, 0)]
    patches = np.pad(patches, patch_paddings, mode = "constant", constant_values = 0)
    positions = np.pad(positions, position_paddings, mode = "constant", constant_values = -1)
    return patches, positions


def convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Convert 3D array image of shape (num_channels, image_height, image_width) into 2D array of patches of shape
    (num_patches_height * num_patches_width, patch_size * patch_size * num_channels).
    """
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patched_image = patched_image.transpose(1, 3, 2, 4, 0)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image


def precompute_rope_invfreq(image_position_ids, rope_theta, head_dim):
    spatial_dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, spatial_dim, 2, dtype = torch.int64) / spatial_dim))
    inv_freq_expanded = inv_freq[None, :, None].float().expand(image_position_ids.shape[0], -1, 1)
    exp = image_position_ids[:, :, 0][:, None, :].float()
    freqs_x = (inv_freq_expanded.float() @ exp.float()).transpose(1, 2)
    exp = image_position_ids[:, :, 1][:, None, :].float()
    freqs_y = (inv_freq_expanded.float() @ exp.float()).transpose(1, 2)
    freqs = torch.cat((freqs_x, freqs_x, freqs_y, freqs_y), dim = -1)
    return freqs

