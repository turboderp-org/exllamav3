from __future__ import annotations

import math

import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..cache.recurrent_util import prepare_for_recurrence
from ..modules import (
    Attention,
    BlockSparseMLP,
    Embedding,
    GatedMLP,
    Linear,
    RMSNorm,
    SlidingAttention,
    TransformerBlock,
)
from ..modules.attn import prepare_for_attn
from ..util.rope import RoPE, RopeStyle


class AfmoeConfig(Config):
    arch_string = "AfmoeForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": AfmoeModel},
            **kwargs,
        )

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.num_dense_layers = self.read_cfg(int, "num_dense_layers", 0)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        self.layer_types = self.read_cfg(list, "layer_types", no_default)
        self.sliding_window = self.read_cfg(int, "sliding_window", -1)

        # MLP/MoE params
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.assert_cfg(str, "score_func", "sigmoid", True)
        self.assert_cfg(int, "topk_group", 1, True)
        self.assert_cfg(int, "n_group", 1, True)

        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, "num_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", no_default)
        self.num_shared_experts = self.read_cfg(int, "num_shared_experts", 0)
        self.route_norm = self.read_cfg(bool, "route_norm", True)
        self.route_scale = self.read_cfg(float, "route_scale", 1.0)
        self.mup_enabled = self.read_cfg(bool, "mup_enabled", False)

        # Norms/RoPE
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", 1e-5)
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)


class AfmoeModel(Model):
    config_class = AfmoeConfig

    def __init__(
        self,
        config: AfmoeConfig,
        key_prefix: str = "model",
        swa_full: bool = False,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.swa_full = swa_full

        embedding_multiplier = math.sqrt(config.hidden_size) if config.mup_enabled else 1.0
        self.modules += [
            Embedding(
                config=config,
                key=f"{key_prefix}.embed_tokens",
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                multiplier=embedding_multiplier,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            is_moe = idx >= config.num_dense_layers
            is_swa = config.layer_types[idx] == "sliding_attention"
            rope_settings = config.rope_settings if is_swa else None
            attn_cls = Attention if swa_full or not is_swa else SlidingAttention

            self.modules += [
                TransformerBlock(
                    config=config,
                    key=f"{key_prefix}.layers.{idx}",
                    layer_idx=idx,
                    attn_norm=RMSNorm(
                        config=config,
                        key=f"{key_prefix}.layers.{idx}.input_layernorm",
                        rms_norm_eps=config.rms_norm_eps,
                    ),
                    attn=attn_cls(
                        config=config,
                        key=f"{key_prefix}.layers.{idx}.self_attn",
                        layer_idx=idx,
                        hidden_size=config.hidden_size,
                        head_dim=config.head_dim,
                        num_q_heads=config.num_q_heads,
                        num_kv_heads=config.num_kv_heads,
                        rope_settings=rope_settings,
                        sm_scale=None,
                        sliding_window=config.sliding_window if is_swa else -1,
                        key_q="q_proj",
                        key_k="k_proj",
                        key_v="v_proj",
                        key_o="o_proj",
                        key_g="gate_proj",
                        full_gate=True,
                        qmap="block.attn",
                        q_norm=RMSNorm(
                            config=config,
                            key=f"{key_prefix}.layers.{idx}.self_attn.q_norm",
                            rms_norm_eps=config.rms_norm_eps,
                        ),
                        k_norm=RMSNorm(
                            config=config,
                            key=f"{key_prefix}.layers.{idx}.self_attn.k_norm",
                            rms_norm_eps=config.rms_norm_eps,
                        ),
                        out_dtype=torch.float,
                        select_hq_bits=2,
                    ),
                    attn_post_norm=RMSNorm(
                        config=config,
                        key=f"{key_prefix}.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps=config.rms_norm_eps,
                    ),
                    mlp_norm=RMSNorm(
                        config=config,
                        key=f"{key_prefix}.layers.{idx}.pre_mlp_layernorm",
                        rms_norm_eps=config.rms_norm_eps,
                    ),
                    mlp=(
                        BlockSparseMLP(
                            config=config,
                            key=f"{key_prefix}.layers.{idx}.mlp",
                            hidden_size=config.hidden_size,
                            intermediate_size=config.moe_intermediate_size,
                            num_experts=config.num_experts,
                            num_experts_per_tok=config.num_experts_per_tok,
                            key_up="experts.{expert_idx}.up_proj",
                            key_gate="experts.{expert_idx}.gate_proj",
                            key_down="experts.{expert_idx}.down_proj",
                            key_routing_gate="router.gate",
                            key_e_score_bias="expert_bias",
                            qmap="block.mlp",
                            router_type="dots",
                            routed_scaling_factor=config.route_scale,
                            interm_dtype=torch.half,
                            out_dtype=torch.float,
                            shared_experts=(
                                GatedMLP(
                                    config=config,
                                    key=f"{key_prefix}.layers.{idx}.mlp.shared_experts",
                                    hidden_size=config.hidden_size,
                                    intermediate_size=(
                                        config.moe_intermediate_size
                                        * config.num_shared_experts
                                    ),
                                    key_up="up_proj",
                                    key_gate="gate_proj",
                                    key_down="down_proj",
                                    qmap="block.mlp",
                                    interm_dtype=torch.half,
                                    out_dtype=torch.float,
                                    select_hq_bits=2,
                                )
                                if config.num_shared_experts > 0
                                else None
                            ),
                        )
                        if is_moe
                        else GatedMLP(
                            config=config,
                            key=f"{key_prefix}.layers.{idx}.mlp",
                            hidden_size=config.hidden_size,
                            intermediate_size=config.intermediate_size,
                            key_up="up_proj",
                            key_gate="gate_proj",
                            key_down="down_proj",
                            qmap="block.mlp",
                            interm_dtype=torch.half,
                            out_dtype=torch.float,
                        )
                    ),
                    mlp_post_norm=RMSNorm(
                        config=config,
                        key=f"{key_prefix}.layers.{idx}.post_mlp_layernorm",
                        rms_norm_eps=config.rms_norm_eps,
                    ),
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = f"{key_prefix}.embed_tokens"

        self.modules += [
            RMSNorm(
                config=config,
                key=f"{key_prefix}.norm",
                rms_norm_eps=config.rms_norm_eps,
                out_dtype=torch.half,
            ),
            Linear(
                config=config,
                key="lm_head",
                qbits_key="head_bits",
                alt_key=head_alt_key,
                in_features=config.hidden_size,
                out_features=config.vocab_size,
                qmap="block",
                caps={"logits_output": True},
            ),
        ]

        self.logit_layer_idx = len(self.modules) - 1
        self.g_rope = RoPE("cpu", config.rope_settings)

        # Activate all experts during H capture pass in quantization
        self.calibration_all_experts = True

        if not self.swa_full:
            self.caps.update({
                "supports_tp": False,
                "recurrent_states": True,
                "default_recurrent_checkpoint_interval": 6144,
            })

    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        if not self.swa_full:
            prepare_for_recurrence(input_ids, params, self)
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids
