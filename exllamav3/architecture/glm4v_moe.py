from __future__ import annotations
from typing_extensions import override
import numpy as np
import torch
import torch.nn.functional as F
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle, position_embedding_grid_2d, RopeSettings
from ..util.file import read_dict, no_value, no_default
from ..util.vision import convert_to_rgb, normalize_image, smart_resize
from ..modules import (
    Module,
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    GatedMLP,
    Linear,
    Conv,
    LayerNorm,
    Glm4VPosEmbedding,
    MLP
)
from ..modules.attn import prepare_for_attn
from .glm4_moe import Glm4MoeModel
from types import SimpleNamespace
from ..tokenizer import Tokenizer, MMEmbedding
from PIL import Image
import os, json
from .glm4v import read_glm4v_vision_config, read_glm4v_pp_config, Glm4VVisionModel

class Glm4VMoeConfig(Config):
    arch_string = "Glm4vMoeForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Glm4VMoeModel, "vision": Glm4VVisionModel},
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "text_config->head_dim", None)
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "text_config->num_key_value_heads", self.num_q_heads)
        self.use_qk_norm = self.read_cfg(bool, "text_config->use_qk_norm", False)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.assert_cfg(str, "text_config->hidden_act", "silu", True)
        self.assert_cfg(bool, "text_config->norm_topk_prob", True, True)
        self.intermediate_size = self.read_cfg(int, "text_config->intermediate_size", no_default)
        self.moe_intermediate_size = self.read_cfg(int, "text_config->moe_intermediate_size", no_default)
        self.num_shared_experts = self.read_cfg(int, "text_config->n_shared_experts", 1)
        self.num_experts = self.read_cfg(int, "text_config->n_routed_experts", 128)
        self.num_experts_per_tok = self.read_cfg(int, "text_config->num_experts_per_tok", 8)
        self.first_k_dense_replace = self.read_cfg(int, "text_config->first_k_dense_replace", 3)
        self.routed_scaling_factor = self.read_cfg(float, "text_config->routed_scaling_factor", 2.5)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 10000,
            config_dict = self.read_cfg(dict, "text_config", no_default)
        )

        # Vision model settings
        read_vision_config = self.read_cfg(dict, "vision_config", no_default)
        self.vision = read_glm4v_vision_config(read_vision_config)

        prep_path = os.path.join(self.directory, "preprocessor_config.json")
        with open(prep_path, encoding = "utf8") as f:
            read_prep_config = json.load(f)
        self.vision_pp = read_glm4v_pp_config(read_prep_config)

        self.vision_start_token_id = self.read_cfg(int, "image_start_token_id", 151339)
        self.vision_end_token_id = self.read_cfg(int, "image_end_token_id", 151340)


class Glm4VMoeModel(Glm4MoeModel):
    config_class = Glm4VMoeConfig

    def __init__(
        self,
        config: Glm4VMoeConfig,
        **kwargs
    ):
        super().__init__(config, key_prefix = "model.language_model", **kwargs)