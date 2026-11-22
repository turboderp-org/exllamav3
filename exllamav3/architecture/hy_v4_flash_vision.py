from __future__ import annotations
from typing_extensions import override
import json, math, os
import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace
from ..model.model import Model
from ..modules import TransformerBlock, Attention, Conv, MLP, LayerNorm
from ..modules.arch_specific.hy_v4_vl import HyV4VisionPosEmbedding, HyV4VisionMerger
from ..tokenizer import Tokenizer, MMEmbedding
from .mm_processing.common import convert_to_rgb, normalize_image

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .hy_v4_flash import HyV4FlashConfig

# Hy4 flash vision component (the Hy4 VL tower, stored under vit.*): a SigLIP2-shaped ViT at native
# resolution over 16 px patches, with a learned position grid resampled to the image, and a merger
# that joins 2 x 2 patches into one text-width token and closes every row of them with a newline token.
# An image is its start token, rows * (columns + 1) feature tokens and its end token; in the text model
# those rows take three-axis positions (see HyV4FlashModel).
# Reference: transformers models/hy_v4_vl (vision tower, image processor)


def read_hy_v4_vision_config(config) -> SimpleNamespace | None:
    p = "vision_config->"
    n_layers = config.read_cfg(int, p + "num_hidden_layers", 0)
    if n_layers <= 0 or not config.stc.has_tensor("vit.embeddings.patch_embedding.weight"):
        return None
    v = SimpleNamespace(
        n_layers = n_layers,
        hidden_size = config.read_cfg(int, p + "hidden_size", 1152),
        n_heads = config.read_cfg(int, p + "num_attention_heads", 16),
        intermediate_size = config.read_cfg(int, p + "intermediate_size", 4304),
        out_hidden_size = config.read_cfg(int, p + "out_hidden_size", config.hidden_size),
        patch_size = config.read_cfg(int, p + "patch_size", 16),
        merge_size = config.read_cfg(int, p + "spatial_merge_size", 2),
        max_image_size = config.read_cfg(int, p + "max_image_size", 2048),
        norm_eps = config.read_cfg(float, p + "rms_norm_eps", 1e-5),
        interpolate_mode = config.read_cfg(str, p + "interpolate_mode", "bilinear"),
    )
    config.assert_cfg(str, p + "hidden_act", "gelu", True)
    config.assert_cfg(str, p + "projector_type", "rmsnorm_mlp_adapter", True)
    config.assert_cfg(bool, p + "use_2d_rope", False, True)
    config.assert_cfg(int, p + "cat_extra_token", 0, True)
    config.assert_cfg(int, p + "temporal_patch_size", 1, True)
    v.head_dim = v.hidden_size // v.n_heads
    v.num_channels = 3

    # Image processor settings
    pp = {}
    path = os.path.join(config.directory, "processor_config.json")
    if os.path.exists(path):
        with open(path, encoding = "utf8") as f:
            pp = json.load(f).get("image_processor", {})
    v.min_pixels = pp.get("min_pixels", 64 * 64)
    v.max_pixels = pp.get("max_pixels", v.max_image_size ** 2)
    v.image_mean = tuple(pp.get("image_mean", (0.5, 0.5, 0.5)))
    v.image_std = tuple(pp.get("image_std", (0.5, 0.5, 0.5)))
    v.rescale_factor = pp.get("rescale_factor", 1 / 255)
    return v


def hy_v4_smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple:
    """Target (height, width): both multiples of factor, the pixel count within [min_pixels, max_pixels],
    the aspect ratio kept as closely as that allows"""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Image aspect ratio must be below 200, got {max(height, width) / min(height, width):.1f}")
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


class HyV4FlashVisionModel(Model):

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: HyV4FlashConfig) -> dict:
        return config.stc.list_tensors(prefix = "vit")

    def __init__(
        self,
        config: HyV4FlashConfig,
        key_prefix: str = "vit",
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.caps.update({
            "image_input": True,
            "default_vision_bits": 6,
        })
        v = config.vision
        assert v is not None, "HyV4Flash: the model directory carries no vision tower"

        self.modules += [
            Conv(
                config = config,
                key = f"{key_prefix}.embeddings.patch_embedding",
                in_channels = v.num_channels,
                out_channels = v.hidden_size,
                kernel_size = (v.patch_size, v.patch_size),
                out_dtype = torch.float,
            ),
            HyV4VisionPosEmbedding(
                config = config,
                key = f"{key_prefix}.embeddings.position_embedding",
                hidden_size = v.hidden_size,
                position_edge = v.max_image_size // v.patch_size,
                interpolate_mode = v.interpolate_mode,
                out_dtype = torch.float,
            ),
        ]

        for idx in range(v.n_layers):
            key = f"{key_prefix}.layers.{idx}"
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = LayerNorm(config, f"{key}.input_layernorm", v.norm_eps),
                    attn = Attention(
                        config = config,
                        key = f"{key}.self_attn",
                        layer_idx = idx,
                        hidden_size = v.hidden_size,
                        head_dim = v.head_dim,
                        num_q_heads = v.n_heads,
                        num_kv_heads = v.n_heads,
                        rope_settings = None,
                        key_q = "q_proj",
                        key_k = "k_proj",
                        key_v = "v_proj",
                        key_o = "o_proj",
                        qmap = "block.attn",
                    ),
                    mlp_norm = LayerNorm(config, f"{key}.post_attention_layernorm", v.norm_eps),
                    mlp = MLP(
                        config = config,
                        key = f"{key}.mlp",
                        hidden_size = v.hidden_size,
                        intermediate_size = v.intermediate_size,
                        key_up = "dense_h_to_4h",
                        key_down = "dense_4h_to_h",
                        activation_fn = "gelu_exact",
                        qmap = "block.mlp",
                    ),
                )
            ]

        self.modules += [
            HyV4VisionMerger(
                config = config,
                key = f"{key_prefix}.perceive",
                hidden_size = v.hidden_size,
                out_hidden_size = v.out_hidden_size,
                merge_size = v.merge_size,
                rms_norm_eps = v.norm_eps,
                out_dtype = torch.float,
            )
        ]


    def preprocess(self, image: Image) -> tuple:
        """Returns the normalized image as (1, 3, height, width) fp16, its size as (width, height) and its
        patch grid as (rows, columns)"""
        v = self.config.vision
        image = convert_to_rgb(image)
        width, height = image.size
        new_h, new_w = hy_v4_smart_resize(height, width, v.patch_size * v.merge_size, v.min_pixels, v.max_pixels)
        if (new_w, new_h) != (width, height):
            image = image.resize((new_w, new_h), resample = Image.Resampling.BICUBIC)
        pixels = np.array(image).astype(np.float32) * v.rescale_factor
        pixels = normalize_image(pixels, v.image_mean, v.image_std)
        pixels = torch.from_numpy(pixels.transpose(2, 0, 1)).half().unsqueeze(0).contiguous()
        return pixels, (new_w, new_h), (new_h // v.patch_size, new_w // v.patch_size)


    def default_load_shape_dtype(self, chunk_size):
        return (1, 3, 1024, 1024), torch.half


    def default_load_params(self, max_chunk_size):
        return {"grid_hw": (64, 64), "causal": False}


    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image | list[Image],
        text_alias: str | None = None,
    ):
        config = self.config
        v = config.vision
        if isinstance(image, list):
            assert text_alias is None, "Cannot apply single alias to list of images"
            return [self.get_image_embeddings(tokenizer, i) for i in image]

        pixels, prep_size, (grid_h, grid_w) = self.preprocess(image)
        embedding_tensor = self.forward(
            pixels,
            params = {"causal": False, "grid_hw": (grid_h, grid_w)},
        ).cpu()

        # Feature rows end in a newline token: rows x (columns + 1) positions, which is also the grid
        # their three-axis positions follow in the text model
        rows, cols = grid_h // v.merge_size, grid_w // v.merge_size + 1
        num_emb_tokens = embedding_tensor.shape[1]
        assert num_emb_tokens == rows * cols
        token_string = torch.tensor(
            [[config.image_start_token_id] + [-1] * num_emb_tokens + [config.image_end_token_id]],
            dtype = torch.long
        )
        mme = MMEmbedding(
            embeddings = embedding_tensor[0],
            text_alias = text_alias,
            token_string = token_string,
            grid_thw = (1, rows, cols),
            mrope_merge_size = 1,
        )
        mme.metadata.update({
            "original_size": image.size,
            "preprocessed_size": prep_size,
            "model_architecture": config.architecture,
        })
        return mme


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return input_ids
