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
    Lfm3VlProjector,
    Lfm3VlVisionConv,
)
from ..tokenizer.mm_embedding import MMEmbedding
from ..util.rope import RopeSettings, RopeStyle
from .lfm3 import Lfm3Config, Lfm3Model, read_lfm3_layer_types
from .mm_processing.qwen2 import qwen2_position_embedding_grid_2d


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


class Lfm3VlVisionModel(Model):
    config_class = Lfm3VlConfig

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: Lfm3VlConfig) -> dict:
        # The tower and the projector ride along unquantized unless --vision_bits quantizes them
        return (
            config.stc.list_tensors(prefix = "model.vision_tower") |
            config.stc.list_tensors(prefix = "model.multi_modal_projector")
        )

    def __init__(
        self,
        config: Lfm3VlConfig,
        key_prefix: str = "model.vision_tower",
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.caps.update({
            "image_input": True,
            # The tower quantizes cleanly at 6 bpw (validated on LFM3-VL-2B)
            "default_vision_bits": 6,
        })

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
                # Bidirectional GQA over one frame. The axial 2D rope (the same frequencies over half the
                # head dim for the row and the column axis, laid out [h, w] and NEOX-paired) is the Qwen2
                # grid table, handed to the rope kernel per patch through params["inv_freq"]
                attn = Attention(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.self_attn",
                    layer_idx = idx,
                    hidden_size = v.hidden_size,
                    head_dim = v.head_dim,
                    num_q_heads = v.num_attention_heads,
                    num_kv_heads = v.num_key_value_heads,
                    rope_settings = RopeSettings(
                        head_dim = v.head_dim,
                        rope_style = RopeStyle.NEOX,
                    ),
                    sm_scale = None,
                    key_q = "q_proj",
                    key_k = "k_proj",
                    key_v = "v_proj",
                    key_o = "out_proj",
                    use_cu_seqlens = True,
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

    def make_vision_params(self, grid_thw: tuple) -> dict:
        """Forward params for one frame grid: bidirectional attention over the frame, the axial rope
        angles per patch and the grid shape for the convolution's block reorder"""
        v = self.config.vision
        t, h, w = grid_thw
        return {
            "causal": False,
            "inv_freq": qwen2_position_embedding_grid_2d(grid_thw, v.head_dim, v.spatial_merge_size, v.rope_theta),
            "frame_shapes": [(h, w)] * t,
            "use_cu_seqlens": True,
            "cu_seqlens": torch.arange(0, t * h * w + 1, h * w, dtype = torch.int32),
            "max_seqlen": h * w,
        }

    def _measure_grid(self) -> tuple:
        # The autosplit loader's measuring forward: a square frame of 32 x 32 patches
        m = self.config.vision.spatial_merge_size
        side = 32 // m * m
        return (1, side, side)

    def default_load_shape_dtype(self, chunk_size):
        t, h, w = self._measure_grid()
        return (1, t * h * w, self.config.vision.patch_dim), torch.half

    def default_load_params(self, max_chunk_size):
        return self.make_vision_params(self._measure_grid())

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
            return [self.get_image_embeddings(tokenizer, i) for i in image]

        image_tensor, prep_image_size, grid_thw = self.preprocess(image)
        params = self.make_vision_params(grid_thw)
        embedding_tensor = self.forward(image_tensor.unsqueeze(0), params = params).cpu()
        embedding_tensor = embedding_tensor.reshape(-1, embedding_tensor.shape[-1])
        num_emb_tokens = embedding_tensor.shape[0]

        # The span the chat template renders around an image: <|vision_start|> image tokens <|vision_end|>
        id_start = self.config.vision_start_token_id
        id_end = self.config.vision_end_token_id
        token_string = torch.tensor([[id_start] + [-1] * num_emb_tokens + [id_end]], dtype = torch.long)
        mme = MMEmbedding(
            embeddings = embedding_tensor,
            text_alias = text_alias,
            token_string = token_string,
            grid_thw = grid_thw,
        )
        mme.metadata.update({
            "original_size": image.size,
            "preprocessed_size": prep_image_size,
            "model_architecture": self.config.architecture,
        })
        return mme
