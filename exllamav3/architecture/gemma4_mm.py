from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from typing_extensions import override

from ..model.model import Model
from ..modules import (
    GatedMLP,
    RMSNorm,
    TransformerBlock,
    Gemma4VisionAttention,
    Gemma4VisionPatchEmbedder,
    Gemma4VisionPooler,
    Gemma4VisionProjector,
    Gemma4VisionStandardize,
)
from ..tokenizer import MMEmbedding, Tokenizer
from ..util.vision import convert_to_rgb

if TYPE_CHECKING:
    from .gemma4 import Gemma4Config


def set_gemma4_vision_groups(
    input_ids: torch.Tensor,
    params: dict,
    boi_token_id: int | None,
    eoi_token_id: int | None,
) -> None:
    indexed_embeddings = params.get("indexed_embeddings") or []
    if not indexed_embeddings:
        params.pop("vision_group_ids", None)
        return

    vision_group_ids = torch.full(input_ids.shape, -1, dtype = torch.int32)
    group_index = 0
    for embedding in indexed_embeddings:
        mask = (input_ids >= embedding.first_index) & (input_ids < embedding.last_index)
        if not mask.any():
            continue

        for row in range(input_ids.shape[0]):
            row_mask = mask[row]
            if not row_mask.any():
                continue

            positions = torch.nonzero(row_mask, as_tuple = False).flatten()
            start = int(positions[0].item())
            end = int(positions[-1].item())
            # Gemma4's multimodal bidirectional mask groups only the soft image
            # tokens. BOI/EOI remain text tokens in HF/vLLM and should not be
            # merged into the vision group span.
            vision_group_ids[row, start : end + 1] = group_index

        group_index += 1

    if group_index > 0:
        params["vision_group_ids"] = vision_group_ids
    else:
        params.pop("vision_group_ids", None)


def _get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
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
            f"Resizing [{height}x{width}] to [{target_height}x{target_width}] exceeds patch budget"
        )

    return target_height, target_width


def _convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patched_image = patched_image.transpose(1, 3, 2, 4, 0)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image


def _pad_patches_and_positions(
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


class Gemma4VisionModel(Model):

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: "Gemma4Config") -> dict:
        return (
            config.stc.list_tensors(prefix = "model.vision_tower") |
            config.stc.list_tensors(prefix = "model.embed_vision")
        )


    def __init__(
        self,
        config: "Gemma4Config",
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.caps.update({
            "image_input": True,
            "supports_tp": False,
        })
        v = self.config.vision

        self.modules += [
            Gemma4VisionPatchEmbedder(
                config = config,
                key = "model.vision_tower.patch_embedder",
                hidden_size = v.hidden_size,
                patch_dim = v.patch_dim,
            )
        ]

        for idx in range(v.num_hidden_layers):
            key = f"model.vision_tower.encoder.layers.{idx}"
            self.modules.append(
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key}.input_layernorm",
                        rms_norm_eps = v.rms_norm_eps,
                    ),
                    attn = Gemma4VisionAttention(
                        config = config,
                        key = f"{key}.self_attn",
                        layer_idx = idx,
                        hidden_size = v.hidden_size,
                        head_dim = v.head_dim,
                        num_q_heads = v.num_q_heads,
                        num_kv_heads = v.num_kv_heads,
                        rope_theta = v.rope_theta,
                        rms_norm_eps = v.rms_norm_eps,
                    ),
                    attn_post_norm = RMSNorm(
                        config = config,
                        key = f"{key}.post_attention_layernorm",
                        rms_norm_eps = v.rms_norm_eps,
                        out_dtype = torch.float,
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key}.pre_feedforward_layernorm",
                        rms_norm_eps = v.rms_norm_eps,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"{key}.mlp",
                        hidden_size = v.hidden_size,
                        intermediate_size = v.intermediate_size,
                        key_up = "up_proj.linear",
                        key_gate = "gate_proj.linear",
                        key_down = "down_proj.linear",
                        activation_fn = "gelu",
                        qmap = "block.mlp",
                    ),
                    mlp_post_norm = RMSNorm(
                        config = config,
                        key = f"{key}.post_feedforward_layernorm",
                        rms_norm_eps = v.rms_norm_eps,
                        out_dtype = torch.float,
                    ),
                )
            )

        self.modules += [
            Gemma4VisionPooler(
                config = config,
                key = "model.vision_tower.pooler",
                hidden_size = v.hidden_size,
            )
        ]

        if v.standardize:
            self.modules += [
                Gemma4VisionStandardize(
                    config = config,
                    key = "model.vision_tower",
                )
            ]

        self.modules += [
            Gemma4VisionProjector(
                config = config,
                key = "model.embed_vision.embedding_projection",
                in_features = v.hidden_size,
                out_features = config.hidden_size,
                rms_norm_eps = v.rms_norm_eps,
            ),
        ]


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return input_ids


    def default_load_shape_dtype(self, chunk_size):
        return (
            1,
            self.config.vision.max_patches,
            self.config.vision.patch_dim,
        ), torch.half


    def default_load_params(self, chunk_size):
        h_patches = 45
        w_patches = self.config.vision.max_patches // h_patches
        grid_x, grid_y = np.meshgrid(np.arange(w_patches), np.arange(h_patches), indexing = "xy")
        position_ids = np.stack([grid_x, grid_y], axis = -1).reshape(self.config.vision.max_patches, 2)
        return {
            "input_ids": torch.zeros((1, self.config.vision.max_patches, self.config.vision.patch_dim), dtype = torch.half),
            "image_position_ids": torch.from_numpy(position_ids).unsqueeze(0).to(torch.long),
            "image_output_length": self.config.vision_pp.max_soft_tokens,
            "causal": False,
        }


    def preprocess(
        self,
        image: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor, int, tuple[int, int]]:
        vpp = self.config.vision_pp
        image = convert_to_rgb(image)
        old_size = image.size
        width, height = old_size
        target_height, target_width = _get_aspect_ratio_preserving_size(
            height = height,
            width = width,
            patch_size = vpp.patch_size,
            max_patches = self.config.vision.max_patches,
            pooling_kernel_size = vpp.pooling_kernel_size,
        )

        if (target_width, target_height) != old_size:
            image = image.resize((target_width, target_height), resample = Image.Resampling(vpp.resample))

        image_np = np.array(image).astype(np.float32)
        image_np = image_np.transpose(2, 0, 1)

        if vpp.do_rescale:
            image_np *= vpp.rescale_factor

        if vpp.do_normalize:
            image_mean = np.asarray(vpp.image_mean, dtype = np.float32).reshape(-1, 1, 1)
            image_std = np.asarray(vpp.image_std, dtype = np.float32).reshape(-1, 1, 1)
            image_np = (image_np - image_mean) / image_std

        patches = _convert_image_to_patches(image_np, vpp.patch_size)
        num_soft_tokens = patches.shape[0] // (vpp.pooling_kernel_size ** 2)

        patch_height = image_np.shape[-2] // vpp.patch_size
        patch_width = image_np.shape[-1] // vpp.patch_size
        grid_x, grid_y = np.meshgrid(np.arange(patch_width), np.arange(patch_height), indexing = "xy")
        positions = np.stack([grid_x, grid_y], axis = -1).reshape(patches.shape[0], 2)

        patches, positions = _pad_patches_and_positions(
            patches,
            positions,
            self.config.vision.max_patches,
        )

        pixel_values = torch.from_numpy(patches).half().unsqueeze(0)
        image_position_ids = torch.from_numpy(positions).to(torch.long).unsqueeze(0)
        return pixel_values, image_position_ids, num_soft_tokens, (target_width, target_height)


    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image.Image | list[Image.Image],
        text_alias: str | None = None,
    ):
        if isinstance(image, list):
            assert text_alias is None, "Cannot apply a single alias to a list of images"
            return [self.get_image_embeddings(tokenizer, i) for i in image]

        pixel_values, image_position_ids, _, prep_size = self.preprocess(image)
        params = {
            "causal": False,
            "image_position_ids": image_position_ids,
            "image_output_length": self.config.vision_pp.max_soft_tokens,
        }
        embedding_tensor = self.forward(
            pixel_values,
            params = params,
        ).cpu()
        pooler_mask = params.get("image_pooler_mask")
        if pooler_mask is not None:
            pooler_mask = pooler_mask.cpu()
            embedding_tensor = embedding_tensor[0][pooler_mask[0]].unsqueeze(0)
            num_soft_tokens = int(pooler_mask[0].sum().item())
        else:
            num_soft_tokens = embedding_tensor.shape[1]

        boi_token_id = tokenizer.single_id("<|image>")
        image_token_id = tokenizer.single_id("<|image|>")
        eoi_token_id = tokenizer.single_id("<image|>")
        token_string = torch.tensor(
            [[boi_token_id] + [image_token_id] * num_soft_tokens + [eoi_token_id]],
            dtype = torch.long,
        )
        token_string[:, 1:-1] = -1

        mme = MMEmbedding(
            embeddings = embedding_tensor[0],
            text_alias = text_alias,
            token_string = token_string,
        )
        mme.metadata.update({
            "original_size": image.size,
            "preprocessed_size": prep_size,
            "model_architecture": self.config.architecture,
        })
        return mme
