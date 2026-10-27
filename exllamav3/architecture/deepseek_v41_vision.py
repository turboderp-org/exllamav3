from __future__ import annotations
from typing_extensions import override
import math
import torch
from PIL import Image
from types import SimpleNamespace
from ..tokenizer import Tokenizer, MMEmbedding
from .deepseek_v4_vision import DeepseekV4VisionModel

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .deepseek_v41 import DeepseekV41Config

# DeepSeek-V4.1 vision component: the DeepSeek-V4 tower and aligner with the settings nested under
# vision_config, no image_pad marker, a one-pass resize to the token budget, and the token block in
# reading order with no position alignment (the trunk stays causal over an image span).


def read_deepseek_v41_vision_config(config) -> SimpleNamespace | None:
    n_layers = config.read_cfg(int, "vision_config->num_hidden_layers", 0)
    if n_layers <= 0:
        return None
    v = SimpleNamespace(
        n_layers = n_layers,
        dim = config.read_cfg(int, "vision_config->hidden_size", 1024),
        n_heads = config.read_cfg(int, "vision_config->num_attention_heads", 16),
        inter_dim = config.read_cfg(int, "vision_config->intermediate_size", 2816),
        patch_size = config.read_cfg(int, "vision_config->patch_size", 14),
        rope_theta = config.read_cfg(float, "vision_config->rope_theta", 10000.0),
        downsample_ratio = config.read_cfg(int, "vision_config->downsample_ratio", 3),
        max_n_token = config.read_cfg(int, "vision_config->max_image_tokens", 1024),
        min_pixels = config.read_cfg(int, "vision_config->min_pixels", 295936),
        max_wh_ratio = config.read_cfg(int, "vision_config->max_wh_ratio", None),
    )
    v.head_dim = v.dim // v.n_heads
    v.num_channels = 3
    v.rms_norm_eps = 1e-6
    return v


class DeepseekV41VisionModel(DeepseekV4VisionModel):

    def __init__(
        self,
        config: DeepseekV41Config,
        **kwargs
    ):
        v = config.vision
        # The shortest image block (1 x 1 grid) is 4 rows and must cover the Engram look-back, so that
        # padding image rows there equals the reference's look-back stopping at an image row
        assert v is not None and v.patch_size > 0 and v.downsample_ratio > 0 and \
            v.max_n_token >= 4 >= config.engram_ngram_size - 1, \
            "DeepseekV41 vision: unsupported vision_config or engram_max_ngram_size"
        super().__init__(
            config,
            marker_keys = ("image_start", "image_newline", "image_end"),
            load_grid = (v.downsample_ratio, v.downsample_ratio * (v.max_n_token - 3)),
            **kwargs
        )

    @override
    def plan_resize(self, height, width, best_height, best_width):
        v = self.config.vision
        p, cell, budget = v.patch_size, v.patch_size * v.downsample_ratio, v.max_n_token - 2
        def grid():
            return math.ceil(best_height // p / v.downsample_ratio), math.ceil(best_width // p / v.downsample_ratio)
        n_llm_h, n_llm_w = grid()
        if n_llm_h * (n_llm_w + 1) > budget:
            r = height / width
            max_w = math.sqrt(budget / r + 0.25) - 0.5
            max_h = max_w * r
            if max_w < 1.0:
                best_height, best_width = budget // 2 * cell, cell
            elif max_h < 1.0:
                best_height, best_width = cell, (budget - 1) * cell
            else:
                beta = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
                best_height, best_width = math.floor(height * beta / p) * p, math.floor(width * beta / p) * p
            n_llm_h, n_llm_w = grid()
        assert n_llm_h > 0 and n_llm_w > 0 and n_llm_h * (n_llm_w + 1) <= budget, \
            "DeepseekV41 vision: image does not fit max_image_tokens"
        return n_llm_h, n_llm_w, best_height, best_width

    @override
    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image | list[Image],
        text_alias: str | None = None,
    ):
        if isinstance(image, list):
            assert text_alias is None, "Cannot apply single alias to list of images"
            return [self.get_image_embeddings(tokenizer, i, None) for i in image]

        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = self.preprocess(image)
        out = self.encode_patches(patches, n_vit_h, n_vit_w)
        assert out.shape[0] == n_llm_h * n_llm_w
        start, newline, end = self.aligner.markers.to(out.device)
        rows = torch.cat((out.view(n_llm_h, n_llm_w, -1), newline.expand(n_llm_h, 1, -1)), dim = 1)
        block = torch.cat((start[None], rows.flatten(0, 1), end[None])).cpu()

        mme = MMEmbedding(
            embeddings = block,
            text_alias = text_alias,
            token_string = torch.full((1, block.shape[0]), -1, dtype = torch.long),
        )
        mme.metadata.update({
            "original_size": image.size,
            "preprocessed_size": (n_vit_w * self.config.vision.patch_size, n_vit_h * self.config.vision.patch_size),
            "grid_vit": (n_vit_h, n_vit_w),
            "grid_llm": (n_llm_h, n_llm_w),
            "model_architecture": self.config.architecture,
        })
        return mme
