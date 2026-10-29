from __future__ import annotations
from typing_extensions import override
import json
import math
import os
import numpy as np
import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import (
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    Linear,
    GatedMLP,
)
from ..modules.arch_specific.lfm3_vl import (
    Lfm3VlAxialRope,
    Lfm3VlProjector,
    Lfm3VlVisionConv,
)
from ..tokenizer.mm_embedding import MMEmbedding
from .lfm3 import Lfm3Config, Lfm3Model, read_lfm3_layer_types


def lfm3_vl_smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
):
    """Round the image size to a multiple of the patch grid while keeping the pixel count in range."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def lfm3_vl_position_ids(grid_thw: tuple, spatial_merge_size: int) -> torch.Tensor:
    """
    Row and column index of every patch, laid out block-major over the merge blocks (the order the
    patches arrive in).
    """
    position_ids = []
    for t, h, w in grid_thw:
        hpos = torch.arange(h).unsqueeze(1).expand(-1, w)
        wpos = torch.arange(w).unsqueeze(0).expand(h, -1)
        hpos = hpos.reshape(h // spatial_merge_size, spatial_merge_size, w // spatial_merge_size, spatial_merge_size)
        wpos = wpos.reshape(h // spatial_merge_size, spatial_merge_size, w // spatial_merge_size, spatial_merge_size)
        hpos = hpos.permute(0, 2, 1, 3).reshape(h * w)
        wpos = wpos.permute(0, 2, 1, 3).reshape(h * w)
        position_ids.append(torch.stack([hpos, wpos], dim = -1))
    return torch.cat(position_ids, dim = 0)


class Lfm3VlVisionConfig:
    def __init__(self, config_dict: dict):
        self.hidden_size = config_dict["hidden_size"]
        self.num_hidden_layers = config_dict["num_hidden_layers"]
        self.num_attention_heads = config_dict["num_attention_heads"]
        self.num_key_value_heads = config_dict.get("num_key_value_heads", self.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.intermediate_size = config_dict["intermediate_size"]
        self.norm_eps = config_dict.get("norm_eps", 1e-6)
        self.patch_size = config_dict.get("patch_size", 16)
        self.temporal_patch_size = config_dict.get("temporal_patch_size", 2)
        self.spatial_merge_size = config_dict.get("spatial_merge_size", 2)
        self.conv_kernel_size = config_dict.get("conv_kernel_size", 3)
        self.layer_types = config_dict.get(
            "layer_types",
            ["conv", "full_attention"]
        )
        assert len(self.layer_types) == self.num_hidden_layers, \
            "Length of vision_config->layer_types doesn't match the number of vision layers"
        for t in self.layer_types:
            if t not in ["full_attention", "conv"]:
                raise ValueError(f"Unknown layer type in vision_config->layer_types: {t}")
        self.rope_theta = config_dict.get("rope_theta", 1000000.0)
        self.patch_dim = 3 * self.temporal_patch_size * self.patch_size * self.patch_size


class Lfm3VlConfig(Lfm3Config):
    arch_string = "Lfm3VlForConditionalGeneration"
    config_key = None
    prefix = "text_config->"

    def get_model_classes(self) -> dict:
        return {"text": Lfm3VlTextModel, "vision": Lfm3VlVisionModel}

    def get_text_config_dict(self) -> dict:
        return self.read_cfg(dict, "text_config", no_default)

    def __init__(self, directory: str, **kwargs):
        super().__init__(directory, **kwargs)

        # Vision tower
        self.vision = Lfm3VlVisionConfig(self.read_cfg(dict, "vision_config", no_default))

        # Projector
        self.projector_hidden_size = self.read_cfg(int, "projector_hidden_size", no_default)
        self.projector_hidden_act = self.read_cfg(str, "projector_hidden_act", "gelu")
        self.projector_bias = self.read_cfg(bool, "projector_bias", True)
        self.projector_layer_norm_eps = self.read_cfg(float, "projector_layer_norm_eps", 1e-6)
        self.downsample_factor = self.read_cfg(int, "downsample_factor", 2)

        # Token ids
        self.image_token_id = self.read_cfg(int, "image_token_id", no_default)
        self.video_token_id = self.read_cfg(int, "video_token_id", no_default)
        self.vision_start_token_id = self.read_cfg(int, "vision_start_token_id", no_default)
        self.vision_end_token_id = self.read_cfg(int, "vision_end_token_id", no_default)

        # Image processor
        self.image_mean = self.read_cfg(list, "image_mean", [0.5, 0.5, 0.5])
        self.image_std = self.read_cfg(list, "image_std", [0.5, 0.5, 0.5])
        self.rescale_factor = self.read_cfg(float, "rescale_factor", 1.0 / 255.0)
        self.min_pixels = self.read_cfg(int, "min_pixels", 65536)
        self.max_pixels = self.read_cfg(int, "max_pixels", 16777216)
        self.resample = self.read_cfg(str, "resample", "bicubic")

        prep_path = os.path.join(self.directory, "processor_config.json")
        if os.path.exists(prep_path):
            with open(prep_path, encoding = "utf8") as f:
                prep_config = json.load(f)
            image_processor = prep_config.get("image_processor", {})
            self.image_mean = image_processor.get("image_mean", self.image_mean)
            self.image_std = image_processor.get("image_std", self.image_std)
            self.rescale_factor = image_processor.get("rescale_factor", self.rescale_factor)
            size = image_processor.get("size", {})
            self.min_pixels = size.get("shortest_edge", self.min_pixels)
            self.max_pixels = size.get("longest_edge", self.max_pixels)


class Lfm3VlTextModel(Lfm3Model):
    config_class = Lfm3VlConfig

    def __init__(
        self,
        config: Lfm3VlConfig,
        key_prefix: str = "model.language_model",
        **kwargs,
    ):
        super().__init__(config, key_prefix = key_prefix, **kwargs)


class Lfm3VlVisionAttention(Attention):
    """
    Attention over the patches of one frame: bidirectional, grouped-query, with an axial 2D rope that
    uses the same frequencies for the row and the column axis.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        hidden_size: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        rope_theta: float,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
        qmap: str | None = "block.attn",
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(
            config = config,
            key = key,
            layer_idx = layer_idx,
            hidden_size = hidden_size,
            head_dim = head_dim,
            num_q_heads = num_q_heads,
            num_kv_heads = num_kv_heads,
            rope_settings = None,
            sm_scale = None,
            key_q = "q_proj",
            key_k = "k_proj",
            key_v = "v_proj",
            key_o = "out_proj",
            qmap = qmap,
            q_norm = q_norm,
            k_norm = k_norm,
            out_dtype = out_dtype,
            use_cu_seqlens = True,
        )
        self.rope_theta = rope_theta
        self.norm_eps = q_norm.rms_norm_eps

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.rope = Lfm3VlAxialRope(device, self.head_dim, self.rope_theta)

    @override
    def unload(self):
        super().unload()
        self.rope = None


class Lfm3VlVisionModel(Model):
    config_class = Lfm3VlConfig

    def __init__(
        self,
        config: Lfm3VlConfig,
        key_prefix: str = "model.vision_tower",
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        v = config.vision

        self.modules += [
            Linear(
                config = config,
                key = f"{key_prefix}.patchifier",
                in_features = v.patch_dim,
                out_features = v.hidden_size,
                qmap = "block.attn",
            ),
            RMSNorm(
                config = config,
                key = f"{key_prefix}.embedding_norm",
                rms_norm_eps = v.norm_eps,
            ),
        ]

        for idx in range(v.num_hidden_layers):
            layer_type = v.layer_types[idx]
            if layer_type == "conv":
                attn = Lfm3VlVisionConv(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.conv",
                    hidden_size = v.hidden_size,
                    conv_kernel_size = v.conv_kernel_size,
                    spatial_merge_size = v.spatial_merge_size,
                    qmap = "block.attn",
                    out_dtype = torch.float,
                )
            else:
                attn = Lfm3VlVisionAttention(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.self_attn",
                    layer_idx = idx,
                    hidden_size = v.hidden_size,
                    head_dim = v.head_dim,
                    num_q_heads = v.num_attention_heads,
                    num_kv_heads = v.num_key_value_heads,
                    rope_theta = v.rope_theta,
                    q_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.self_attn.q_layernorm",
                        rms_norm_eps = v.norm_eps,
                    ),
                    k_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.self_attn.k_layernorm",
                        rms_norm_eps = v.norm_eps,
                    ),
                    qmap = "block.attn",
                    out_dtype = torch.float,
                )

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.operator_norm",
                        rms_norm_eps = v.norm_eps,
                    ),
                    attn = attn,
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.ffn_norm",
                        rms_norm_eps = v.norm_eps,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.feed_forward",
                        hidden_size = v.hidden_size,
                        intermediate_size = v.intermediate_size,
                        key_up = "w3",
                        key_gate = "w1",
                        key_down = "w2",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                    ),
                )
            ]

        merge = config.downsample_factor
        self.modules += [
            Lfm3VlProjector(
                config = config,
                key = "model.multi_modal_projector",
                in_features = v.hidden_size * merge * merge,
                hidden_size = config.projector_hidden_size,
                out_features = config.hidden_size,
                merge_size = merge,
                layernorm_eps = config.projector_layer_norm_eps,
                qmap = "block",
                out_dtype = torch.half,
            )
        ]

        self.caps.update({
            "supports_tp": False,
        })

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return input_ids

    def preprocess(
        self,
        images,
    ) -> (torch.Tensor, tuple, tuple):
        v = self.config.vision
        from PIL import Image

        resample = getattr(Image.Resampling, self.config.resample.upper())

        if not isinstance(images, list):
            mode = "image"
            images = [images]
        else:
            mode = "video"
            frames = len(images)
            images = images[:frames]

        images = [image.convert("RGB") for image in images]

        old_size = images[0].size
        assert all(old_size == image.size for image in images), \
            "All frames in a video must have the same dimensions"

        new_size = lfm3_vl_smart_resize(
            old_size[1],
            old_size[0],
            v.patch_size * v.spatial_merge_size,
            self.config.min_pixels,
            self.config.max_pixels,
        )
        # new_size is (height, width), PIL wants (width, height)
        pil_size = (new_size[1], new_size[0])
        if old_size != pil_size:
            images = [image.resize(pil_size, resample = resample) for image in images]

        arrays = [np.array(image).astype(np.float32) for image in images]
        arrays = [array * self.config.rescale_factor for array in arrays]
        mean = np.array(self.config.image_mean, dtype = np.float32)
        std = np.array(self.config.image_std, dtype = np.float32)
        arrays = [(array - mean) / std for array in arrays]

        patches = np.array(arrays)
        patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] == 1:
            patches = np.tile(patches, (v.temporal_patch_size, 1, 1, 1))
        channels = patches.shape[1]
        grid_t = patches.shape[0] // v.temporal_patch_size
        grid_h = new_size[0] // v.patch_size
        grid_w = new_size[1] // v.patch_size
        patches = patches.reshape(
            grid_t,
            v.temporal_patch_size,
            channels,
            grid_h // v.spatial_merge_size,
            v.spatial_merge_size,
            v.patch_size,
            grid_w // v.spatial_merge_size,
            v.spatial_merge_size,
            v.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channels * v.temporal_patch_size * v.patch_size ** 2,
        )

        image = torch.from_numpy(flatten_patches).half()
        return image, new_size, (grid_t, grid_h, grid_w)

    def get_image_embeddings(
        self,
        tokenizer,
        image,
        text_alias: str | None = None,
    ):
        v = self.config.vision

        if isinstance(image, list):
            assert text_alias is None, "Cannot apply single alias to list of images"
            image_tensor = []
            grids = []
            for i in image:
                t, prep_image_size, grid_thw = self.preprocess(i)
                image_tensor.append(t.unsqueeze(0))
                grids.append(grid_thw)
            image_tensor = torch.cat(image_tensor, dim = 0)
            return_batch = True
        else:
            image_tensor, prep_image_size, grid_thw = self.preprocess(image)
            image = [image]
            image_tensor = image_tensor.unsqueeze(0)
            grids = [grid_thw]
            return_batch = False

        position_ids = lfm3_vl_position_ids(grids, v.spatial_merge_size)
        params = {
            "causal": False,
            "position_ids": position_ids.unsqueeze(0),
            "frame_shapes": [(h, w) for t, h, w in grids],
            "use_cu_seqlens": True,
        }
        cu_seqlens = torch.tensor(
            [0] + [h * w for t, h, w in grids],
            dtype = torch.int32,
            device = image_tensor.device,
        )
        params["cu_seqlens"] = cu_seqlens
        params["max_seqlen"] = max(h * w for t, h, w in grids)

        embedding_tensor = self.forward(image_tensor, params = params).cpu()

        merge = v.spatial_merge_size
        split_sizes = [(t * h * w) // (merge * merge) for t, h, w in grids]
        if embedding_tensor.dim() == 3 and embedding_tensor.shape[0] == len(split_sizes):
            chunks = [embedding_tensor[i] for i in range(len(split_sizes))]
        else:
            if embedding_tensor.dim() == 3:
                embedding_tensor = embedding_tensor.reshape(-1, embedding_tensor.shape[-1])
            chunks = list(torch.split(embedding_tensor, split_sizes, dim = 0))

        mmes = []
        for i in range(len(chunks)):
            num_emb_tokens = chunks[i].shape[0]
            # The vision start/end tokens are part of the prompt text (the chat template renders
            # "<|vision_start|><|image|><|vision_end|>"), so only the image tokens are replaced here
            token_string = torch.tensor(
                [[-1] * num_emb_tokens],
                dtype = torch.long,
            )
            mme = MMEmbedding(
                embeddings = chunks[i],
                text_alias = text_alias,
                token_string = token_string,
                grid_thw = grids[i],
            )
            mme.metadata.update({
                "original_size": image[i].size,
                "preprocessed_size": prep_image_size,
                "model_architecture": self.config.architecture,
            })
            mmes.append(mme)

        return mmes if return_batch else mmes[0]
