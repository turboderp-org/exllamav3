"""
LFM2-VL (LiquidAI LFM2.5-VL): the LFM2 text decoder under model.language_model, a SigLIP2 NaFlex vision
tower under model.vision_tower.vision_model and the pixel-unshuffle MLP projector under
model.multi_modal_projector
"""

from __future__ import annotations
from typing_extensions import override
import json
import os
from types import SimpleNamespace
import torch
from PIL import Image
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.file import read_dict
from ..modules import (
    TransformerBlock,
    Attention,
    MLP,
    LayerNorm,
    Linear,
    PosEmbedding,
    PixelUnshuffle,
)
from ..tokenizer import Tokenizer, MMEmbedding
from .lfm2 import Lfm2Config, Lfm2Model
from .mm_processing.lfm2_vl import (
    lfm2_vl_smart_resize,
    lfm2_vl_is_image_too_large,
    lfm2_vl_grid_layout,
    lfm2_vl_image_to_tensor,
    lfm2_vl_resize,
    lfm2_vl_split_tiles,
    lfm2_vl_patchify,
)

# HF activation names to the MLP module's (HF "gelu" is the exact erf form)
_activations = {
    "gelu": "gelu_exact",
    "gelu_pytorch_tanh": "gelu",
    "silu": "silu",
}


class Lfm2VlConfig(Lfm2Config):
    arch_string = "Lfm2VlForConditionalGeneration"
    config_key = "text_config"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(directory, **kwargs)
        self.model_classes["text"] = Lfm2VlTextModel
        self.model_classes["vision"] = Lfm2VlVisionModel

        self.vision = read_lfm2_vl_vision_config(self.read_cfg(dict, "vision_config", no_default))

        # Projector
        self.downsample_factor = self.read_cfg(int, "downsample_factor", 2)
        self.projector_hidden_size = self.read_cfg(int, "projector_hidden_size", no_default)
        self.projector_hidden_act = _activations[self.read_cfg(str, "projector_hidden_act", "gelu")]
        self.projector_use_layernorm = self.read_cfg(bool, "projector_use_layernorm", False)
        self.use_image_special_tokens = self.read_cfg(bool, "use_image_special_tokens", True)
        self.image_token_id = self.read_cfg(int, "image_token_id", None)

        prep_path = os.path.join(directory, "processor_config.json")
        if not os.path.exists(prep_path):
            prep_path = os.path.join(directory, "preprocessor_config.json")
        with open(prep_path, encoding = "utf8") as f:
            self.vision_pp = read_lfm2_vl_pp_config(json.load(f))


def read_lfm2_vl_vision_config(config_dict: dict):
    v = SimpleNamespace(**{
        k: read_dict(config_dict, t, k, no_default)
        for k, t in [
            ("hidden_size", int),
            ("intermediate_size", int),
            ("num_hidden_layers", int),
            ("num_attention_heads", int),
            ("patch_size", int),
            ("num_patches", int),
            ("layer_norm_eps", float),
        ]
    })
    v.hidden_act = _activations[read_dict(config_dict, str, "hidden_act", "gelu_pytorch_tanh")]
    v.num_channels = read_dict(config_dict, int, "num_channels", 3)
    v.head_dim = v.hidden_size // v.num_attention_heads
    v.patch_dim = v.num_channels * v.patch_size ** 2
    v.pos_grid = int(round(v.num_patches ** 0.5))
    assert v.pos_grid ** 2 == v.num_patches, "Expected a square vision_config->num_patches position table"
    return v


def read_lfm2_vl_pp_config(config_dict: dict):
    ip = read_dict(config_dict, dict, "image_processor", config_dict)
    pp = SimpleNamespace(**{
        k: read_dict(ip, t, k, no_default)
        for k, t in [
            ("image_mean", list),
            ("image_std", list),
            ("rescale_factor", float),
            ("encoder_patch_size", int),
            ("downsample_factor", int),
            ("min_image_tokens", int),
            ("max_image_tokens", int),
            ("min_tiles", int),
            ("max_tiles", int),
            ("tile_size", int),
        ]
    })
    pp.use_thumbnail = read_dict(ip, bool, "use_thumbnail", True)
    pp.do_image_splitting = read_dict(ip, bool, "do_image_splitting", True)
    pp.max_pixels_tolerance = read_dict(ip, float, "max_pixels_tolerance", 2.0)
    assert read_dict(ip, int, "resample", 3) == 3, \
        "Expected bicubic resampling (resample 3) in the image processor config"
    return pp


class Lfm2VlTextModel(Lfm2Model):
    config_class = Lfm2VlConfig

    def __init__(
        self,
        config: Lfm2VlConfig,
        **kwargs,
    ):
        super().__init__(config, key_prefix = "model.language_model", **kwargs)


class Lfm2VlVisionModel(Model):
    """SigLIP2 NaFlex tower and the pixel-unshuffle MLP projector"""

    config_class = Lfm2VlConfig

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: Lfm2VlConfig) -> dict:
        return (
            config.stc.list_tensors(prefix = "model.vision_tower") |
            config.stc.list_tensors(prefix = "model.multi_modal_projector")
        )

    def __init__(
        self,
        config: Lfm2VlConfig,
        key_prefix: str = "model.vision_tower.vision_model",
        projector_prefix: str = "model.multi_modal_projector",
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.caps.update({
            "image_input": True,
            "default_vision_bits": 6,
            "supports_tp": False,
        })
        v = config.vision

        # Blocks keep an fp32 residual stream. Inputs are flattened patches; the learned square position
        # table is resampled to each tile's patch grid
        self.modules += [
            Linear(
                config = config,
                key = f"{key_prefix}.embeddings.patch_embedding",
                in_features = v.patch_dim,
                out_features = v.hidden_size,
                qmap = None,
                out_dtype = torch.float,
                pad_to = 1,
            ),
            PosEmbedding(
                config = config,
                key = f"{key_prefix}.embeddings.position_embedding",
                hidden_size = v.hidden_size,
                base_grid = v.pos_grid,
                antialias = True,
                out_dtype = torch.float,
            ),
        ]

        for idx in range(v.num_hidden_layers):
            key = f"{key_prefix}.encoder.layers.{idx}"
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = LayerNorm(
                        config = config,
                        key = f"{key}.layer_norm1",
                        layernorm_eps = v.layer_norm_eps,
                    ),
                    attn = Attention(
                        config = config,
                        key = f"{key}.self_attn",
                        layer_idx = idx,
                        hidden_size = v.hidden_size,
                        head_dim = v.head_dim,
                        num_q_heads = v.num_attention_heads,
                        num_kv_heads = v.num_attention_heads,
                        rope_settings = None,
                        key_q = "q_proj",
                        key_k = "k_proj",
                        key_v = "v_proj",
                        key_o = "out_proj",
                        qmap = "block.attn",
                        out_dtype = torch.float,
                    ),
                    mlp_norm = LayerNorm(
                        config = config,
                        key = f"{key}.layer_norm2",
                        layernorm_eps = v.layer_norm_eps,
                    ),
                    mlp = MLP(
                        config = config,
                        key = f"{key}.mlp",
                        hidden_size = v.hidden_size,
                        intermediate_size = v.intermediate_size,
                        key_up = "fc1",
                        key_down = "fc2",
                        activation_fn = v.hidden_act,
                        qmap = "block.mlp",
                        out_dtype = torch.float,
                        pad_to = 1,
                    ),
                )
            ]

        f = config.downsample_factor
        self.modules += [
            LayerNorm(
                config = config,
                key = f"{key_prefix}.post_layernorm",
                layernorm_eps = v.layer_norm_eps,
                out_dtype = torch.float,
            ),
            PixelUnshuffle(
                config = config,
                key = f"{projector_prefix}.pixel_unshuffle",
                factor = f,
                out_dtype = torch.half,
            ),
        ]
        if config.projector_use_layernorm:
            self.modules += [
                LayerNorm(
                    config = config,
                    key = f"{projector_prefix}.layer_norm",
                    layernorm_eps = 1e-5,
                    out_dtype = torch.half,
                )
            ]
        self.modules += [
            MLP(
                config = config,
                key = projector_prefix,
                hidden_size = v.hidden_size * f ** 2,
                intermediate_size = config.projector_hidden_size,
                key_up = "linear_1",
                key_down = "linear_2",
                activation_fn = config.projector_hidden_act,
                out_size = config.hidden_size,
                qmap = "block.mlp",
                out_dtype = torch.half,
                pad_to = 1,
            )
        ]


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return input_ids


    def default_load_shape_dtype(self, chunk_size):
        v = self.config.vision
        side = self.config.vision_pp.tile_size // v.patch_size
        return (1, side * side, v.patch_dim), torch.half


    def default_load_params(self, max_chunk_size):
        side = self.config.vision_pp.tile_size // self.config.vision.patch_size
        return self.make_vision_params((side, side))


    def make_vision_params(self, grid_hw: tuple) -> dict:
        return {
            "causal": False,
            "grid_hw": grid_hw,
        }


    def preprocess(
        self,
        image: Image.Image,
    ) -> tuple:
        """
        The reference processor's resize-and-split: a large image becomes a (rows x cols) grid of
        tile_size tiles plus a smart-resized thumbnail of the whole image, any other image a single
        smart-resized tile. Returns (tiles, rows, cols, single) with tiles a (rows * cols, 3, T, T) uint8
        tensor or None, and single the (1, 3, H, W) uint8 thumbnail or lone tile.
        """
        pp = self.config.vision_pp
        pixels = lfm2_vl_image_to_tensor(image)
        new_w, new_h = lfm2_vl_smart_resize(
            image.size,
            pp.encoder_patch_size,
            pp.downsample_factor,
            pp.min_image_tokens,
            pp.max_image_tokens,
        )
        split = pp.do_image_splitting and not (pp.min_tiles == pp.max_tiles == 1) and lfm2_vl_is_image_too_large(
            image.size,
            pp.encoder_patch_size,
            pp.downsample_factor,
            pp.max_image_tokens,
            pp.max_pixels_tolerance,
        )
        if split:
            cols, rows = lfm2_vl_grid_layout(image.size, pp.min_tiles, pp.max_tiles, pp.tile_size)
            split = rows * cols > 1
        if split:
            grid = lfm2_vl_resize(pixels, (rows * pp.tile_size, cols * pp.tile_size))
            tiles = lfm2_vl_split_tiles(grid, rows, cols, pp.tile_size)
            single = lfm2_vl_resize(pixels, (new_h, new_w)).unsqueeze(0) if pp.use_thumbnail else None
            return tiles, rows, cols, single
        return None, 1, 1, lfm2_vl_resize(pixels, (new_h, new_w)).unsqueeze(0)


    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """(n, 3, H, W) uint8 tiles of one size -> (n, tokens, hidden) projected embeddings"""
        pp = self.config.vision_pp
        patches = lfm2_vl_patchify(pixels, pp.encoder_patch_size, pp.rescale_factor, pp.image_mean, pp.image_std)
        grid_hw = (pixels.shape[-2] // pp.encoder_patch_size, pixels.shape[-1] // pp.encoder_patch_size)
        return self.forward(patches.half(), params = self.make_vision_params(grid_hw)).cpu()


    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image.Image | list[Image.Image],
        text_alias: str | None = None,
    ):
        """
        One MMEmbedding per image, with the reference layout: <|image_start|>, then for a split image
        each tile introduced by its <|img_row_R_col_C|> token and the thumbnail by <|img_thumbnail|>,
        then <|image_end|>. The special tokens are omitted when the config disables them.
        """
        if isinstance(image, list):
            assert text_alias is None, "Cannot apply a single alias to a list of images"
            return [self.get_image_embeddings(tokenizer, i) for i in image]

        special = self.config.use_image_special_tokens
        tiles, rows, cols, single = self.preprocess(image)
        parts = []
        token_ids = []
        if special:
            token_ids.append(tokenizer.single_id("<|image_start|>"))
        if tiles is not None:
            emb = self.encode(tiles)
            for row in range(rows):
                for col in range(cols):
                    if special:
                        token_ids.append(tokenizer.single_id(f"<|img_row_{row + 1}_col_{col + 1}|>"))
                    token_ids += [-1] * emb.shape[1]
            parts.append(emb.reshape(-1, emb.shape[-1]))
            if single is not None:
                if special:
                    token_ids.append(tokenizer.single_id("<|img_thumbnail|>"))
                emb = self.encode(single)[0]
                token_ids += [-1] * emb.shape[0]
                parts.append(emb)
        else:
            emb = self.encode(single)[0]
            token_ids += [-1] * emb.shape[0]
            parts.append(emb)
        if special:
            token_ids.append(tokenizer.single_id("<|image_end|>"))

        mme = MMEmbedding(
            embeddings = torch.cat(parts, dim = 0),
            text_alias = text_alias,
            token_string = torch.tensor([token_ids], dtype = torch.long),
        )
        mme.metadata.update({
            "original_size": image.size,
            "tile_grid": (rows, cols),
            "model_architecture": self.config.architecture,
        })
        return mme
