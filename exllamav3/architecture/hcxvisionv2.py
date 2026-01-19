from __future__ import annotations
from typing_extensions import override
import numpy as np
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..model.model import Model
from ..util.rope import RopeStyle, position_embedding_grid_2d, RopeSettings, RoPE
from ..util.vision import convert_to_rgb, normalize_image, smart_resize, get_qwen2_window_index
from ..util.file import read_dict,  no_default
from ..util.tensor import get_for_device
from ..modules import (
    Module,
    TransformerBlock,
    Attention,
    Linear,
    Conv,
    GatedMLP,
    RMSNorm,
    Qwen3VLPosEmbedding
)
from .hyperclovax import HyperClovaxModel
from ..tokenizer import Tokenizer, MMEmbedding
from types import SimpleNamespace
from PIL import Image
import os, json
from .qwen2_5_vl import read_qwen2_5_vl_vision_config, read_qwen2_5_vl_pp_config, Qwen2_5VLVisionModel

class HCXVisionV2Config(Config):
    arch_string = "HCXVisionV2ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": HCXVisionV2Model, "vision": HCXVisionV2VisionModel},
            **kwargs
        )

        self.embedding_multiplier = self.read_cfg(float, "text_config->embedding_multiplier", 1.0)
        self.logits_scaling = self.read_cfg(float, "text_config->logits_scaling", no_default)

        self.assert_cfg(float, "text_config->residual_multiplier", 1.0, True)

        # Attention params
        self.head_dim = self.read_cfg(int, "text_config->head_dim", None)
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "text_config->num_key_value_heads", self.num_q_heads)
        self.attention_multiplier = self.read_cfg(float, "text_config->attention_multiplier", None)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.assert_cfg(str, "text_config->hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "text_config->intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "text_config->tie_word_embeddings", False)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            config_dict = self.read_cfg(dict, "text_config", no_default)
        )

        # Vision model settings
        read_vision_config = self.read_cfg(dict, "vision_config", no_default)
        self.vision = read_qwen2_5_vl_vision_config(read_vision_config)

        prep_path = os.path.join(self.directory, "preprocessor_config.json")
        with open(prep_path, encoding = "utf8") as f:
            read_prep_config = json.load(f)
        self.vision_pp = read_qwen2_5_vl_pp_config(read_prep_config)

        self.vision_start_token_id = self.read_cfg(int, "vision_start_token_id", None)
        self.vision_end_token_id = self.read_cfg(int, "vision_end_token_id", None)


class HCXVisionV2Model(HyperClovaxModel):
    config_class = HCXVisionV2Config

    def __init__(
        self,
        config: HCXVisionV2Config,
        **kwargs
    ):
        super().__init__(
            config,
            key_prefix = "model.language_model.model",
            post_norms = False,
            head_key = "model.language_model.lm_head",
            **kwargs
        )


class HCXVisionV2VisionModel(Qwen2_5VLVisionModel):

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: HCXVisionV2Config) -> dict:
        vlm_tensors = config.stc.list_tensors(prefix = "model.vision_model")
        mmp_tensors = config.stc.list_tensors(prefix = "model.mm_projector")
        return vlm_tensors | mmp_tensors

    def __init__(
        self,
        config: HCXVisionV2Config,
        key_prefix = "model.vision_model",
        mm_prefix = "model.mm_projector",
        **kwargs
    ):
        super().__init__(config, key_prefix, mm_prefix, **kwargs)