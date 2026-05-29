from __future__ import annotations
from typing_extensions import override
import torch

from ..cache.recurrent_util import prepare_for_recurrence
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle, RoPE
from ..modules import (
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    BlockSparseMLP,
    Linear,
    GatedMLP,
    SlidingAttention
)
from ..modules.attn import prepare_for_attn
from .step3_5 import Step3_5Model

class Step3_7Config(Config):
    arch_string = "Step3p7ForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Step3_5Model},
            **kwargs
        )

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "text_config->tie_word_embeddings", False)

        # Attention params
        self.head_dim = self.read_cfg(int, "text_config->head_dim", None)
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "text_config->num_attention_groups", no_default)  # !!!

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Sliding attn layers use alternative settings
        self.sliding_window = self.read_cfg(int, "text_config->sliding_window", -1)
        self.alt_head_dim = self.read_cfg(int, "text_config->attention_other_setting->head_dim", None)
        self.alt_num_q_heads = self.read_cfg(int, "text_config->attention_other_setting->num_attention_heads", no_default)
        self.alt_num_kv_heads = self.read_cfg(int, "text_config->attention_other_setting->num_attention_groups", no_default)  # !!!

        # Layer types
        self.layer_types = self.read_cfg(list, "text_config->layer_types", no_default)

        # RoPE
        text_config = self.read_cfg(dict, "text_config", no_default)
        yarn_only_types = self.read_cfg(list, "text_config->yarn_only_types", None)
        rope_theta = self.read_cfg(list, "text_config->rope_theta", no_default)
        partial_rotary_factors = self.read_cfg(list, "text_config->partial_rotary_factors", no_default)
        self.rope_settings_list = [
            self.read_rope_settings_default(
                RopeStyle.NEOX,
                default_rope_theta = rt,
                default_partial_rotary_factor = prf,
                config_dict = text_config,
                override_type = "default" if yarn_only_types and lt not in yarn_only_types else None,
            )
            for (rt, prf, lt) in zip(rope_theta, partial_rotary_factors, self.layer_types)
        ]

        # MLP params
        self.intermediate_size = self.read_cfg(int, "text_config->intermediate_size", no_default)

        self.assert_cfg(bool, "text_config->norm_expert_weight", True, True)
        self.moe_intermediate_size = self.read_cfg(int, "text_config->moe_intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, "text_config->moe_num_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "text_config->moe_top_k", no_default)
        self.shared_expert_intermediate_size = self.read_cfg(int, "text_config->share_expert_dim", no_default)
        self.assert_cfg(bool, "text_config->use_moe_router_bias", True, True)
        self.routed_scaling_factor = self.read_cfg(float, "text_config->moe_router_scaling_factor", 3.0)

        moe_layers = self.read_cfg(str, "text_config->moe_layers_enum", no_default)
        self.moe_layers = set(int(l) for l in moe_layers.split(","))

        self.swiglu_limits = self.read_cfg(list, "text_config->swiglu_limits", no_default)
        self.swiglu_limits_shared = self.read_cfg(list, "text_config->swiglu_limits_shared", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", 1e-5)
        # Step 3.7 config currently says use_qk_norm=false, but the reference
        # modeling_step3p7.py constructs and applies q_norm/k_norm unconditionally.
        self.use_qk_norm = True


class Step3_7Model(Step3_5Model):
    config_class = Step3_7Config

    def __init__(
        self,
        config: Step3_7Config,
        key_prefix: str = "model",
        swa_full: bool = False,
        **kwargs
    ):
        super().__init__(config, key_prefix, swa_full, **kwargs)
