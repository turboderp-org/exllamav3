from __future__ import annotations

from typing_extensions import override
import math
import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import Attention, Embedding, Linear, MLP, RMSNorm, TransformerBlock
from ..modules.attn import prepare_for_attn
from ..modules.gated_delta_net import prepare_for_recurrence
from ..modules.nemotron_h_mamba import NemotronHMamba2
from ..modules.nemotron_h_moe import NemotronHMoE
from ..util.rope import RopeStyle


class NemotronHConfig(Config):
    arch_string = "NemotronHForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": NemotronHModel},
            **kwargs,
        )

        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        self.layer_norm_epsilon = self.read_cfg(float, "layer_norm_epsilon", 1e-5)

        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        self.head_dim = self.read_cfg(int, "head_dim", self.hidden_size // self.num_q_heads)
        self.attention_bias = self.read_cfg(bool, "attention_bias", False)

        self.hybrid_override_pattern = self.read_cfg(str, "hybrid_override_pattern", no_default)
        if len(self.hybrid_override_pattern) != self.num_hidden_layers:
            raise ValueError("hybrid_override_pattern must match num_hidden_layers")
        invalid = set(self.hybrid_override_pattern) - {"M", "*", "E", "-"}
        if invalid:
            raise ValueError(f"Unsupported NemotronH layer types: {sorted(invalid)}")
        self.layer_types = [
            "mamba" if c == "M" else
            "attention" if c == "*" else
            "moe" if c == "E" else
            "mlp"
            for c in self.hybrid_override_pattern
        ]

        self.mlp_hidden_act = self.read_cfg(str, "mlp_hidden_act", "relu2")
        self.ssm_state_size = self.read_cfg(int, "ssm_state_size", no_default)
        self.mamba_num_heads = self.read_cfg(int, "mamba_num_heads", no_default)
        self.mamba_head_dim = self.read_cfg(int, "mamba_head_dim", no_default)
        self.mamba_n_groups = self.read_cfg(int, ["mamba_n_groups", "n_groups"], 8)
        self.mamba_d_conv = self.read_cfg(int, ["mamba_d_conv", "conv_kernel"], 4)
        self.mamba_hidden_act = self.read_cfg(str, "mamba_hidden_act", "silu")
        self.time_step_limit = tuple(
            self.read_cfg(list, ["mamba_dt_limit", "time_step_limit"], [0.0, math.inf])
        )
        self.use_conv_bias = self.read_cfg(bool, ["mamba_conv_bias", "use_conv_bias"], True)
        self.use_bias = self.read_cfg(bool, "use_bias", False)

        self.n_routed_experts = self.read_cfg(int, "n_routed_experts", 0)
        self.n_shared_experts = self.read_cfg(int, "n_shared_experts", 1)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", 0)
        self.moe_shared_expert_intermediate_size = self.read_cfg(int, "moe_shared_expert_intermediate_size", 0)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", 1)
        self.routed_scaling_factor = self.read_cfg(float, "routed_scaling_factor", 1.0)
        self.n_group = self.read_cfg(int, "n_group", 1)
        self.topk_group = self.read_cfg(int, "topk_group", 1)
        self.norm_topk_prob = self.read_cfg(bool, "norm_topk_prob", True)

        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 10000,
        )


class NemotronHModel(Model):
    config_class = NemotronHConfig

    def __init__(
        self,
        config: NemotronHConfig,
        key_prefix: str = "backbone",
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.key_prefix = key_prefix

        self.modules += [
            Embedding(
                config = config,
                key = f"{key_prefix}.embeddings",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx, layer_type in enumerate(config.layer_types):
            norm = RMSNorm(
                config = config,
                key = f"{key_prefix}.layers.{idx}.norm",
                rms_norm_eps = config.layer_norm_epsilon,
            )

            if layer_type == "mamba":
                block = TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = norm,
                    attn = NemotronHMamba2(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.mixer",
                        layer_idx = idx,
                        hidden_size = config.hidden_size,
                        num_heads = config.mamba_num_heads,
                        head_dim = config.mamba_head_dim,
                        ssm_state_size = config.ssm_state_size,
                        n_groups = config.mamba_n_groups,
                        conv_kernel_size = config.mamba_d_conv,
                        rms_norm_eps = config.layer_norm_epsilon,
                        time_step_limit = config.time_step_limit,
                        qmap = "block.mamba",
                        out_dtype = torch.float,
                    ),
                )
            elif layer_type == "attention":
                block = TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = norm,
                    attn = Attention(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.mixer",
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
                        out_dtype = torch.float,
                    ),
                )
            elif layer_type == "moe":
                block = TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}",
                    layer_idx = idx,
                    mlp_norm = norm,
                    mlp = NemotronHMoE(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.mixer",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.moe_intermediate_size,
                        shared_intermediate_size = config.moe_shared_expert_intermediate_size * config.n_shared_experts,
                        num_experts = config.n_routed_experts,
                        num_experts_per_tok = config.num_experts_per_tok,
                        routed_scaling_factor = config.routed_scaling_factor,
                        n_group = config.n_group,
                        topk_group = config.topk_group,
                        norm_topk_prob = config.norm_topk_prob,
                        qmap = "block.moe",
                        out_dtype = torch.float,
                    ),
                )
            else:
                block = TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}",
                    layer_idx = idx,
                    mlp_norm = norm,
                    mlp = MLP(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.mixer",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj",
                        key_down = "down_proj",
                        activation_fn = config.mlp_hidden_act,
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                    ),
                )

            self.modules.append(block)

        cache_layers = [
            i for i, layer_type in enumerate(config.layer_types)
            if layer_type in ("mamba", "attention")
        ]
        self.last_kv_module_idx = self.first_block_idx + cache_layers[-1]

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = f"{key_prefix}.embeddings"

        self.modules += [
            RMSNorm(
                config = config,
                key = f"{key_prefix}.norm_f",
                rms_norm_eps = config.layer_norm_epsilon,
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
                caps = {"logits_output": True},
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1
        self.caps.update({"recurrent_states": True})
        self.caps.update({"supports_tp": False})
        self.calibration_all_experts = True

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
