from PIL import Image
from itertools import product
from math import ceil
import numpy as np
import torch

MAX_IMAGE_SIZE = 3024

def determine_window_size(long: int, short: int) -> int:
    if long <= 728:
        return short if long / short > 1.5 else 0
    return min(short, 504) if long / short > 4 else 504


def slide_window(
    width: int,
    height: int,
    sizes: list[tuple[int, int]],
    steps: list[tuple[int, int]],
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    windows = []
    for size, step in zip(sizes, steps):
        size_w, size_h = size
        step_w, step_h = step

        x_num = 1 if width <= size_w else ceil((width - size_w) / step_w + 1)
        x_start = [step_w * i for i in range(x_num)]
        if len(x_start) > 1 and x_start[-1] + size_w > width:
            x_start[-1] = width - size_w

        y_num = 1 if height <= size_h else ceil((height - size_h) / step_h + 1)
        y_start = [step_h * i for i in range(y_num)]
        if len(y_start) > 1 and y_start[-1] + size_h > height:
            y_start[-1] = height - size_h

        start = np.array(list(product(y_start, x_start)), dtype = int)
        start[:, [0, 1]] = start[:, [1, 0]]
        windows.append(np.concatenate([start, start + size], axis = 1))

    windows = np.concatenate(windows, axis = 0)
    return [(int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])) for b in windows], (x_num, y_num)


def square_pad(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    padded = Image.new(img.mode, (size, size), 0)
    padded.paste(img, (0, 0))
    return padded


def get_image_size_for_padding(img_width: int, img_height: int) -> tuple[int, int]:
    ratio = img_width / img_height
    if min(img_height, img_width) < 32 and (ratio > 4 or ratio < 1 / 4):
        new_size = max(img_height, img_width)
        return new_size, new_size
    return img_width, img_height


def get_image_size_for_preprocess(img_width: int, img_height: int) -> tuple[int, int]:
    if max(img_height, img_width) > MAX_IMAGE_SIZE:
        scale_factor = MAX_IMAGE_SIZE / max(img_height, img_width)
        img_width = int(img_width * scale_factor)
        img_height = int(img_height * scale_factor)
    return img_width, img_height


def get_image_size_for_crop(img_width: int, img_height: int, window_size: int):
    w_ratio = img_width / window_size
    h_ratio = img_height / window_size

    if w_ratio < 1:
        width_new = img_width
    else:
        decimal_w = w_ratio - img_width // window_size
        w_ratio = int(w_ratio) + 1 if decimal_w > 0.2 else int(w_ratio)
        width_new = window_size * w_ratio

    if h_ratio < 1:
        height_new = img_height
    else:
        decimal_h = h_ratio - img_height // window_size
        h_ratio = int(h_ratio) + 1 if decimal_h > 0.2 else int(h_ratio)
        height_new = window_size * h_ratio

    return int(width_new), int(height_new)


def get_patches(img: Image.Image) -> tuple | None:
    img_width, img_height = img.size
    new_img_width, new_img_height = get_image_size_for_padding(img_width, img_height)
    if new_img_width != img_width or new_img_height != img_height:
        img = square_pad(img)
        img_width, img_height = img.size

    new_img_width, new_img_height = get_image_size_for_preprocess(img_width, img_height)
    img = img.resize((new_img_width, new_img_height), Image.Resampling.BILINEAR)
    window_size = determine_window_size(max(new_img_height, new_img_width), min(new_img_height, new_img_width))
    if window_size == 0:
        return img, [], None

    crop_width, crop_height = get_image_size_for_crop(new_img_width, new_img_height, window_size)
    if (crop_width, crop_height) != (img_width, img_height):
        img_for_crop = img.resize((crop_width, crop_height), Image.Resampling.BILINEAR)
    else:
        img_for_crop = img

    patches = []
    newlines = []
    center_list, (x_num, _) = slide_window(crop_width, crop_height, [(window_size, window_size)], [(window_size, window_size)])
    for patch_id, (x, y, patch_w, patch_h) in enumerate(center_list):
        patches.append(img_for_crop.crop((x, y, x + patch_w, y + patch_h)))
        if (patch_id + 1) % x_num == 0:
            newlines.append(patch_id)

    if newlines and newlines[-1] == len(patches) - 1:
        newlines.pop()

    return img, patches, [i in newlines for i in range(len(patches))] if patches else None


def get_vision_position_ids(grid_h: int, grid_w: int):
    rows = torch.arange(grid_h).view(-1, 1)
    cols = torch.arange(grid_w).view(1, -1)
    pos_w = cols.expand(grid_h, grid_w).reshape(-1)
    pos_h = rows.expand(grid_h, grid_w).reshape(-1)
    return torch.stack([pos_w, pos_h], dim = -1).unsqueeze(0).int()


def image_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    image_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype = np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype = np.float32)
    image_np = (image_np - mean) / std
    image_np = image_np.transpose(2, 0, 1)
    return torch.from_numpy(image_np).unsqueeze(0)


def step3_7_position_embedding_grid_2d(
    grid_hw: tuple,
    head_dim: int,
    theta: float,
):
    h, w = grid_hw
    ids = torch.cartesian_prod(torch.arange(h), torch.arange(w))

    # Frequencies
    dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype = torch.float) / dim))
    seq = torch.arange(max(h, w), dtype = torch.float)
    freqs = torch.outer(seq, inv_freq)
    emb = freqs[ids]
    emb = torch.cat((emb[:, 1:2, :], emb[:, 0:1, :]), dim = 1)
    emb = emb.flatten(1)
    return emb
