from __future__ import annotations
from typing_extensions import override

import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import (
    RMSNorm,
    Embedding,
    TransformerBlock,
    GatedMLP,
    BlockSparseMLP,
    Linear,
)
from ..modules.attn import prepare_for_attn
from ..modules.deepseek_v2_mla_attn import DeepseekV2MLAAttention
from ..util.rope import RopeStyle


class DeepseekV2Config(Config):
    arch_string = "DeepseekV2ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": DeepseekV2Model},
            **kwargs
        )

        # Attention params
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        self.q_lora_rank = self.read_cfg(int, "q_lora_rank", None)
        self.kv_lora_rank = self.read_cfg(int, "kv_lora_rank", no_default)
        self.qk_nope_head_dim = self.read_cfg(int, "qk_nope_head_dim", no_default)
        self.qk_rope_head_dim = self.read_cfg(int, "qk_rope_head_dim", no_default)
        self.v_head_dim = self.read_cfg(int, "v_head_dim", no_default)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # MLP params
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.first_k_dense_replace = self.read_cfg(int, "first_k_dense_replace", 0)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, "n_routed_experts", no_default)
        self.num_shared_experts = self.read_cfg(int, "n_shared_experts", 0)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", no_default)
        self.routed_scaling_factor = self.read_cfg(float, "routed_scaling_factor", 1.0)
        self.n_group = self.read_cfg(int, "n_group", 1)
        self.topk_group = self.read_cfg(int, "topk_group", 1)
        self.topk_method = self.read_cfg(str, "topk_method", "greedy")
        self.norm_topk_prob = self.read_cfg(bool, "norm_topk_prob", False)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # RoPE
        self.head_dim = self.qk_rope_head_dim
        self.rope_settings = self.read_rope_settings_default(RopeStyle.GPTJ)


class DeepseekV2Model(Model):
    config_class = DeepseekV2Config

    def __init__(
        self,
        config: DeepseekV2Config,
        key_prefix: str = "model",
        **kwargs
    ):
        super().__init__(config, **kwargs)

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
                ),
                attn = DeepseekV2MLAAttention(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.self_attn",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    num_q_heads = config.num_q_heads,
                    kv_lora_rank = config.kv_lora_rank,
                    qk_nope_head_dim = config.qk_nope_head_dim,
                    qk_rope_head_dim = config.qk_rope_head_dim,
                    v_head_dim = config.v_head_dim,
                    q_lora_rank = config.q_lora_rank,
                    rope_settings = config.rope_settings,
                    qmap = "block.attn",
                    out_dtype = torch.float,
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                mlp = (
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
                    if idx < config.first_k_dense_replace else
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
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        router_type = "deepseek_v2",
                        routed_scaling_factor = config.routed_scaling_factor,
                        n_group = config.n_group,
                        topk_group = config.topk_group,
                        topk_method = config.topk_method,
                        norm_topk_prob = config.norm_topk_prob,
                        shared_experts = (
                            GatedMLP(
                                config = config,
                                key = f"{key_prefix}.layers.{idx}.mlp.shared_experts",
                                hidden_size = config.hidden_size,
                                intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                                key_up = "up_proj",
                                key_gate = "gate_proj",
                                key_down = "down_proj",
                                qmap = "block.mlp",
                                interm_dtype = torch.half,
                                out_dtype = torch.float,
                            ) if config.num_shared_experts > 0 else None
                        ),
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

        # Activate all experts during H capture pass in quantization
        self.calibration_all_experts = True

        # TP for this architecture is not implemented yet
        self.caps.update({"supports_tp": False})


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"<｜User｜>{system_prompt}\n"
        p += f"<｜User｜>{prompt}\n"
        p += "<｜Assistant｜>"
        return p


class DeepseekConfig(DeepseekV2Config):
    arch_string = "DeepseekForCausalLM"


class DeepseekModel(DeepseekV2Model):
    config_class = DeepseekConfig


class DeepseekV3Config(DeepseekV2Config):
    arch_string = "DeepseekV3ForCausalLM"


class DeepseekV3Model(DeepseekV2Model):
    config_class = DeepseekV3Config


class GlmMoeDsaConfig(DeepseekV2Config):
    arch_string = "GlmMoeDsaForCausalLM"


class GlmMoeDsaModel(DeepseekV2Model):
    config_class = GlmMoeDsaConfig
