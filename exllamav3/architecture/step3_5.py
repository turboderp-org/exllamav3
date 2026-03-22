from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle, RoPE
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, BlockSparseMLP, Linear, DeepstackEmbed, GatedMLP
from ..modules.attn import prepare_for_attn

class Step3_5Config(Config):
    arch_string = "Step3p5ForCausalLM"

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
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_attention_groups", no_default)  # !!!

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Sliding attn layers use alternative settings
        self.sliding_window = self.read_cfg(int, "sliding_window", -1)
        self.alt_head_dim = self.read_cfg(int, "attention_other_setting->head_dim", None)
        self.alt_num_q_heads = self.read_cfg(int, "attention_other_setting->num_attention_heads", no_default)
        self.alt_num_kv_heads = self.read_cfg(int, "attention_other_setting->num_attention_groups", no_default)  # !!!

        # Layer types
        self.layer_types = self.read_cfg(list, "layer_types", no_default)

        # RoPE
        rope_theta = self.read_cfg(list, "rope_theta", no_default)
        partial_rotary_factors = self.read_cfg(list, "partial_rotary_factors", no_default)
        self.rope_settings_list = [
            self.read_rope_settings_default(
                RopeStyle.NEOX,
                default_rope_theta = rt,
                default_partial_rotary_factor = prf,
                override_type = "default" if lt == "sliding_attention" else None,
            )
            for (rt, prf, lt) in zip(rope_theta, partial_rotary_factors, self.layer_types)
        ]

        # MLP params
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)

        self.assert_cfg(bool, "norm_expert_weight", True, True)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, "moe_num_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "moe_top_k", no_default)
        self.shared_expert_intermediate_size = self.read_cfg(int, "share_expert_dim", no_default)
        self.assert_cfg(bool, "use_moe_router_bias", True, True)
        self.routed_scaling_factor = self.read_cfg(float, "moe_router_scaling_factor", 3.0)

        moe_layers = self.read_cfg(str, "moe_layers_enum", no_default)
        self.moe_layers = set(int(l) for l in moe_layers.split(","))

        self.swiglu_limits = self.read_cfg(list, "swiglu_limits", no_default)
        self.swiglu_limits_shared = self.read_cfg(list, "swiglu_limits_shared", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", 1e-5)
        self.assert_cfg(bool, "use_qk_norm", True, True)


class Step3_5Model(Model):
    config_class = Step3_5Config

    def __init__(
        self,
        config: Step3_5Config,
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

        for idx in range(config.num_hidden_layers):
            is_moe = idx in config.moe_layers
            is_swa = config.layer_types[idx] == "sliding_attention"
            act_limit = float(config.swiglu_limits[idx])
            act_limit_shared = float(config.swiglu_limits_shared[idx])

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
                    attn = Attention(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.self_attn",
                        layer_idx = idx,
                        hidden_size = config.hidden_size,
                        head_dim = config.head_dim,
                        num_q_heads = config.num_q_heads if not is_swa else config.alt_num_q_heads,
                        num_kv_heads = config.num_kv_heads if not is_swa else config.alt_num_kv_heads,
                        rope_settings = config.rope_settings_list[idx],
                        sm_scale = None,
                        sliding_window = config.sliding_window if is_swa else -1,
                        key_q = "q_proj",
                        key_k = "k_proj",
                        key_v = "v_proj",
                        key_o = "o_proj",
                        key_g = "g_proj",
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
                        tp_split_norm = False,
                        select_hq_bits = 2,
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    mlp = (
                        GatedMLP(
                            config = config,
                            key = f"model.layers.{idx}.mlp",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.intermediate_size,
                            key_up = "up_proj",
                            key_gate = "gate_proj",
                            key_down = "down_proj",
                            qmap = "block.mlp",
                            act_limit = act_limit,
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                            select_hq_bits = 1,
                        )
                        if not is_moe else
                        BlockSparseMLP(
                            config = config,
                            key = f"{key_prefix}.layers.{idx}.moe",
                            hidden_size = config.hidden_size,
                            intermediate_size = config.moe_intermediate_size,
                            num_experts = self.config.num_experts,
                            num_experts_per_tok = self.config.num_experts_per_tok,
                            router_type = "dots",
                            routed_scaling_factor = config.routed_scaling_factor,
                            key_e_score_bias = "router_bias",
                            key_up = "experts.{expert_idx}.up_proj",
                            key_gate = "experts.{expert_idx}.gate_proj",
                            key_down = "experts.{expert_idx}.down_proj",
                            key_gate_split = "gate_proj.weight",
                            key_up_split = "up_proj.weight",
                            key_down_split = "down_proj.weight",
                            key_routing_gate = "gate",
                            qmap = "block.mlp",
                            act_limit = act_limit,
                            interm_dtype = torch.half,
                            out_dtype = torch.float,
                            shared_experts = GatedMLP(
                                config = config,
                                key = f"{key_prefix}.layers.{idx}.share_expert",
                                hidden_size = config.hidden_size,
                                intermediate_size = config.shared_expert_intermediate_size,
                                key_up = "up_proj",
                                key_gate = "gate_proj",
                                key_down = "down_proj",
                                qmap = "block.mlp",
                                act_limit = act_limit_shared,
                                interm_dtype = torch.half,
                                out_dtype = torch.float,
                                select_hq_bits = 2,
                            ),
                            transposed_load = False,
                        )
                    ),
                )
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

        # Activate all experts during H capture pass in quantization
        self.calibration_all_experts = True


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
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