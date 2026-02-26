from __future__ import annotations
from typing_extensions import override
import torch
import os
import json

from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import (
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    BlockSparseMLP,
    Linear,
    GatedDeltaNet,
    GatedMLP,
)
from ..modules.attn import prepare_for_attn
from ..modules.gated_delta_net import prepare_for_recurrence
from .qwen3_vl import read_qwen3_vl_vision_config, read_qwen3_vl_pp_config, Qwen3VLVisionModel


def conditional(condition, a, b):
    return a if condition else b


def read_qwen3_5_layer_types(config: Config, text_config_path: str, num_layers: int, full_attention_interval: int) -> list[str]:
    layer_types = config.read_cfg(list, f"{text_config_path}->layer_types", None)
    if layer_types is not None:
        assert len(layer_types) == num_layers, \
            "Length of text_config->layer_types key doesn't match number of hidden layers"
        for t in layer_types:
            if t not in ["linear_attention", "full_attention"]:
                raise ValueError(f"Unknown layer type in text_config->layer_types: {t}")
        return layer_types

    return [
        "full_attention" if (idx + 1) % full_attention_interval == 0 else "linear_attention"
        for idx in range(num_layers)
    ]


class Qwen3_5Config(Config):
    arch_string = "Qwen3_5ForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Qwen3_5Model, "vision": Qwen3VLVisionModel},
            **kwargs
        )

        text_cfg = "text_config"

        # Attention params
        self.head_dim = self.read_cfg(int, f"{text_cfg}->head_dim", None)
        self.hidden_size = self.read_cfg(int, f"{text_cfg}->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, f"{text_cfg}->num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, f"{text_cfg}->num_key_value_heads", self.num_q_heads)
        self.full_attention_interval = self.read_cfg(int, f"{text_cfg}->full_attention_interval", 4)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Linear attn params
        self.linear_conv_kernel_dim = self.read_cfg(int, f"{text_cfg}->linear_conv_kernel_dim", 4)
        self.linear_num_key_heads = self.read_cfg(int, f"{text_cfg}->linear_num_key_heads", 16)
        self.linear_num_value_heads = self.read_cfg(int, f"{text_cfg}->linear_num_value_heads", 32)
        self.linear_key_head_dim = self.read_cfg(int, f"{text_cfg}->linear_key_head_dim", 128)
        self.linear_value_head_dim = self.read_cfg(int, f"{text_cfg}->linear_value_head_dim", 128)

        # MLP params
        self.assert_cfg(str, f"{text_cfg}->hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, f"{text_cfg}->intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, f"{text_cfg}->rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, f"{text_cfg}->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        self.layer_types = read_qwen3_5_layer_types(
            self,
            text_cfg,
            self.num_hidden_layers,
            self.full_attention_interval,
        )

        # RoPE
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 10000000,
            config_dict = self.read_cfg(dict, text_cfg, no_default)
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


class Qwen3_5MoeConfig(Config):
    arch_string = "Qwen3_5MoeForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Qwen3_5MoeModel, "vision": Qwen3VLVisionModel},
            **kwargs
        )

        text_cfg = "text_config"

        # Attention params
        self.head_dim = self.read_cfg(int, f"{text_cfg}->head_dim", None)
        self.hidden_size = self.read_cfg(int, f"{text_cfg}->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, f"{text_cfg}->num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, f"{text_cfg}->num_key_value_heads", self.num_q_heads)
        self.full_attention_interval = self.read_cfg(int, f"{text_cfg}->full_attention_interval", 4)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Linear attn params
        self.linear_conv_kernel_dim = self.read_cfg(int, f"{text_cfg}->linear_conv_kernel_dim", 4)
        self.linear_num_key_heads = self.read_cfg(int, f"{text_cfg}->linear_num_key_heads", 16)
        self.linear_num_value_heads = self.read_cfg(int, f"{text_cfg}->linear_num_value_heads", 32)
        self.linear_key_head_dim = self.read_cfg(int, f"{text_cfg}->linear_key_head_dim", 128)
        self.linear_value_head_dim = self.read_cfg(int, f"{text_cfg}->linear_value_head_dim", 128)

        # MLP params
        self.assert_cfg(str, f"{text_cfg}->hidden_act", "silu", True)
        self.assert_cfg(bool, f"{text_cfg}->norm_topk_prob", True, True)
        self.moe_intermediate_size = self.read_cfg(int, f"{text_cfg}->moe_intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, f"{text_cfg}->num_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, f"{text_cfg}->num_experts_per_tok", no_default)
        self.shared_expert_intermediate_size = self.read_cfg(int, f"{text_cfg}->shared_expert_intermediate_size", 512)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, f"{text_cfg}->rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, f"{text_cfg}->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        self.layer_types = read_qwen3_5_layer_types(
            self,
            text_cfg,
            self.num_hidden_layers,
            self.full_attention_interval,
        )

        # RoPE
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 10000000,
            config_dict = self.read_cfg(dict, text_cfg, no_default)
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


class Qwen3_5BaseModel(Model):

    def __init__(
        self,
        config: Qwen3_5Config | Qwen3_5MoeConfig,
        key_prefix: str,
        use_moe: bool,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.use_moe = use_moe

        self.modules += [
            Embedding(
                config = config,
                key = f"{key_prefix}.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        self.modules += [
            TransformerBlock(
                config = config,
                key = f"{key_prefix}.layers.{idx}",
                attn_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.input_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                attn = conditional(
                    config.layer_types[idx] == "linear_attention",
                    GatedDeltaNet(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.linear_attn",
                        layer_idx = idx,
                        hidden_size = config.hidden_size,
                        k_head_dim = config.linear_key_head_dim,
                        v_head_dim = config.linear_value_head_dim,
                        num_k_heads = config.linear_num_key_heads,
                        num_v_heads = config.linear_num_value_heads,
                        rms_norm_eps = config.rms_norm_eps,
                        conv_kernel_size = config.linear_conv_kernel_dim,
                        key_a_log = "A_log",
                        key_dt_bias = "dt_bias",
                        key_conv1d = "conv1d",
                        key_qkv = "in_proj_qkv",
                        key_z = "in_proj_z",
                        key_b = "in_proj_b",
                        key_a = "in_proj_a",
                        key_norm = "norm",
                        key_o = "out_proj",
                        qmap = "block.attn",
                        out_dtype = torch.float,
                    ),
                    Attention(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.self_attn",
                        layer_idx = idx,
                        hidden_size = config.hidden_size,
                        head_dim = config.head_dim,
                        num_q_heads = config.num_q_heads,
                        num_kv_heads = config.num_kv_heads,
                        rope_settings = config.rope_settings,
                        sm_scale = None,
                        key_q = "q_proj",
                        key_k = "k_proj",
                        key_v = "v_proj",
                        key_o = "o_proj",
                        qmap = "block.attn",
                        q_norm = RMSNorm(
                            config = config,
                            key = f"{key_prefix}.layers.{idx}.self_attn.q_norm",
                            rms_norm_eps = config.rms_norm_eps,
                            constant_bias = 1.0,
                        ),
                        k_norm = RMSNorm(
                            config = config,
                            key = f"{key_prefix}.layers.{idx}.self_attn.k_norm",
                            rms_norm_eps = config.rms_norm_eps,
                            constant_bias = 1.0,
                        ),
                        out_dtype = torch.float,
                        interleaved_gate = True,
                    ),
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                mlp = (
                    BlockSparseMLP(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.moe_intermediate_size,
                        num_experts = config.num_experts,
                        num_experts_per_tok = config.num_experts_per_tok,
                        key_up = "experts.{expert_idx}.up_proj",
                        key_gate = "experts.{expert_idx}.gate_proj",
                        key_down = "experts.{expert_idx}.down_proj",
                        key_gate_up_split = "experts.gate_up_proj",
                        key_down_split = "experts.down_proj",
                        key_routing_gate = "gate",
                        key_shared_gate = "shared_expert_gate",
                        experts_transposed_load = False,
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        shared_experts = GatedMLP(
                            config = config,
                            key = f"{key_prefix}.layers.{idx}.mlp.shared_expert",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.shared_expert_intermediate_size,
                            key_up = "up_proj",
                            key_gate = "gate_proj",
                            key_down = "down_proj",
                            qmap = "block.mlp",
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                        )
                    ) if use_moe else
                    GatedMLP(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                    )
                ),
            )
            for idx in range(config.num_hidden_layers)
        ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = f"{key_prefix}.embed_tokens"

        self.modules += [
            RMSNorm(
                config = config,
                key = f"{key_prefix}.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
                constant_bias = 1.0,
            ),
            Linear(
                config = config,
                key = "lm_head",
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True}
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        if use_moe:
            self.calibration_all_experts = True

        # Mark that we need recurrent cache for generation
        self.caps.update({"recurrent_states": True})

        # TP for this architecture is not implemented yet
        self.caps.update({"supports_tp": False})

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        prepare_for_recurrence(input_ids, params, self)
        return input_ids

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"<|im_start|>system\n"
            p += f"{system_prompt}<|im_end|>\n"
        p += f"<|im_start|>user\n"
        p += f"{prompt}<|im_end|>\n"
        p += f"<|im_start|>assistant\n"
        return p


class Qwen3_5Model(Qwen3_5BaseModel):
    config_class = Qwen3_5Config

    def __init__(
        self,
        config: Qwen3_5Config,
        **kwargs
    ):
        super().__init__(
            config = config,
            key_prefix = "model.language_model",
            use_moe = False,
            **kwargs,
        )


class Qwen3_5MoeModel(Qwen3_5BaseModel):
    config_class = Qwen3_5MoeConfig

    def __init__(
        self,
        config: Qwen3_5MoeConfig,
        **kwargs
    ):
        super().__init__(
            config = config,
            key_prefix = "model.language_model",
            use_moe = True,
            **kwargs,
        )
