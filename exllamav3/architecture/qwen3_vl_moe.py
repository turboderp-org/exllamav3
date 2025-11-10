from __future__ import annotations
from typing_extensions import override
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle, position_embedding_grid_2d, RopeSettings, RoPE
from ..util.vision import convert_to_rgb, normalize_image, smart_resize
from ..modules.attn import prepare_for_attn
from ..util.file import read_dict, no_value, no_default
from ..modules import (
    Module,
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    GatedMLP,
    Linear,
    Conv,
    MLP,
    LayerNorm,
    Qwen3VLPosEmbedding
)
from .qwen3_moe import Qwen3MoeModel
from ..modules.attn import prepare_for_attn
from ..tokenizer import Tokenizer, MMEmbedding
from types import SimpleNamespace
from PIL import Image
import os, json
from .qwen3_vl import read_qwen3_vl_vision_config, read_qwen3_vl_pp_config, Qwen3VLVisionModel

class Qwen3VLMoeConfig(Config):
    arch_string = "Qwen3VLMoeForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Qwen3VLMoeModel, "vision": Qwen3VLVisionModel},
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "text_config->head_dim", None)
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "text_config->num_key_value_heads", self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.assert_cfg(str, "text_config->hidden_act", "silu", True)
        self.assert_cfg(bool, "text_config->norm_topk_prob", True, True)
        self.moe_intermediate_size = self.read_cfg(int, "text_config->moe_intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, "text_config->num_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "text_config->num_experts_per_tok", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 5000000,
            config_dict = self.read_cfg(dict, "text_config", no_default)
        )

        # Vision model settings
        read_vision_config = self.read_cfg(dict, "vision_config", no_default)
        self.vision = read_qwen3_vl_vision_config(read_vision_config)

        prep_path = os.path.join(self.directory, "preprocessor_config.json")
        with open(prep_path, encoding = "utf8") as f:
            read_prep_config = json.load(f)
        self.vision_pp = read_qwen3_vl_pp_config(read_prep_config)

        self.vision_start_token_id = self.read_cfg(int, "vision_start_token_id", 151652)
        self.vision_end_token_id = self.read_cfg(int, "vision_end_token_id", 151653)


class Qwen3VLMoeModel(Qwen3MoeModel):
    config_class = Qwen3VLMoeConfig

    def __init__(
        self,
        config: Qwen3VLMoeConfig,
        **kwargs
    ):
        super().__init__(config, key_prefix = "model.language_model", **kwargs)
