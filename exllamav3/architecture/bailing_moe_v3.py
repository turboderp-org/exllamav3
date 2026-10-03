from __future__ import annotations
import math
import torch
from typing_extensions import override

from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import (
    RMSNorm, Embedding, TransformerBlock, MLAttention, GatedMLP, Linear,
    BlockSparseMLP, GatedDeltaNet,
)
from ..modules.gated_delta_net import GDNState
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence
from ..util.rope import RopeSettings, RopeStyle


class BailingMoeV3Config(Config):
    arch_string = "BailingMoeV3ForCausalLM"

    def __init__(self, directory: str, **kwargs):
        from .bailing_moe_v3_mtp import BailingMoeV3MTPModel
        super().__init__(directory, {"text": BailingMoeV3Model}, **kwargs)

        for name in (
            "hidden_size", "num_hidden_layers", "num_attention_heads", "head_dim",
            "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
            "intermediate_size", "moe_intermediate_size", "num_experts",
            "num_experts_per_tok", "n_group", "topk_group", "layer_group_size",
            "short_conv_kernel_size", "moe_shared_expert_intermediate_size",
            "vocab_size", "max_position_embeddings",
        ):
            value = self.read_cfg(int, name, no_default)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            setattr(self, name, value)
        self.num_q_heads = self.num_attention_heads
        self.linear_head_dim = self.head_dim
        self.linear_num_heads = self.num_q_heads
        self.linear_conv_kernel_size = self.short_conv_kernel_size
        if self.short_conv_kernel_size > 16:
            raise ValueError("short_conv_kernel_size must be in [1, 16] for slotted convolution")
        self.q_lora_rank = self.read_cfg(int, "q_lora_rank", None)
        if self.q_lora_rank is not None:
            raise ValueError("BailingMoeV3 currently requires direct MLA queries")
        self.num_shared_experts = self.read_cfg(int, "num_shared_experts", 0)
        self.first_k_dense_replace = self.read_cfg(int, "first_k_dense_replace", 0)
        self.num_mtp_layers = self.read_cfg(int, "num_nextn_predict_layers", 0)
        if self.num_mtp_layers not in (0, 1):
            raise ValueError("BailingMoeV3 supports zero or one MTP layer")
        if self.num_shared_experts < 0 or not 0 <= self.first_k_dense_replace <= self.num_hidden_layers:
            raise ValueError("Invalid shared-expert count or dense-layer prefix")
        if self.num_experts % self.n_group or self.num_experts // self.n_group < 2:
            raise ValueError("Expert groups must contain at least two experts and divide num_experts")
        if not 1 <= self.topk_group <= self.n_group:
            raise ValueError("topk_group must be within the expert group count")
        if not 1 <= self.num_experts_per_tok <= self.topk_group * (self.num_experts // self.n_group):
            raise ValueError("num_experts_per_tok exceeds the selected groups")

        # Reject unsupported numerical variants instead of quietly loading a different model.
        for name, expected in (
            ("use_bias", False), ("use_qkv_bias", False), ("use_qk_norm", True),
            ("linear_silu", True), ("no_kda_lora", True), ("use_kda_lora", False),
            ("kda_safe_gate", True), ("mtp_use_kda", False), ("rope_interleave", True),
            ("use_mla_nope", False), ("scale_router_input", False),
            ("norm_topk_prob", True), ("moe_router_enable_expert_bias", True),
            ("value_norm", False), ("up_proj_norm", False), ("use_nGPT", False),
        ):
            self.assert_cfg(bool, name, expected, True)
        for name, expected in (
            ("hidden_act", "silu"), ("score_function", "sigmoid"),
            ("scoring_func", "sigmoid"), ("topk_method", "noaux_tc"),
            ("router_dtype", "fp32"), ("gated_attention_proj_granularity_type", "head_wise"),
        ):
            self.assert_cfg(str, name, expected, True)
        self.assert_cfg(int, "group_norm_size", 1, True)
        self.assert_cfg(int, "num_kv_heads_for_linear_attn", 0, True)
        self.assert_cfg(int, "num_key_value_heads", self.num_q_heads, True)

        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)
        self.linear_lower_bound = self.read_cfg(float, "kda_lower_bound", -5.0)
        self.routed_scaling_factor = self.read_cfg(float, "routed_scaling_factor", 1.0)
        if not math.isfinite(self.linear_lower_bound) or self.linear_lower_bound >= 0:
            raise ValueError("kda_lower_bound must be finite and negative")
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be finite and positive")
        if not math.isfinite(self.routed_scaling_factor) or self.routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be finite and positive")
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.assert_cfg(int, "qk_head_dim", self.qk_head_dim, True)
        self.assert_cfg(int, "rotary_dim", self.qk_rope_head_dim, True)
        if self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be even")
        self.sm_scale = self.qk_head_dim ** -0.5
        rope_scaling = self.read_cfg(dict, "rope_scaling", None)
        if rope_scaling is not None:
            raise ValueError("BailingMoeV3 scaled RoPE has not been qualified")
        rope_theta = self.read_cfg(float, "rope_theta", no_default)
        if not math.isfinite(rope_theta) or rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        # qk_rope_head_dim is already the rotary slice. Applying partial_rotary_factor
        # to it again would rotate only half the trained dimensions.
        self.rope_settings = RopeSettings(
            head_dim = self.qk_rope_head_dim,
            rotary_dim = self.qk_rope_head_dim,
            rope_theta = rope_theta,
            max_position_embeddings = self.max_position_embeddings,
            partial_rotary_factor = 1.0,
            rope_style = RopeStyle.GPTJ,
        )
        full_groups_end = self.num_hidden_layers // self.layer_group_size * self.layer_group_size
        self.layer_types = [
            "full_attention" if (i + 1) % self.layer_group_size == 0 or i >= full_groups_end
            else "linear_attention"
            for i in range(self.num_hidden_layers)
        ]
        self.expert_swiglu_limits = self._read_limits("expert_swiglu_limit_list")
        self.shared_swiglu_limits = self._read_limits("share_expert_swiglu_limit_list")
        if self.num_mtp_layers:
            self.model_classes["mtp"] = BailingMoeV3MTPModel

    def read_cfg(self, expected_type, keys, default = no_default):
        # The generic reader coerces integral floats and accepts bool as int.
        # Ling dimensions/counts must be genuine integers, including base config
        # vocab/context reads. Check the raw value before any numeric coercion.
        if expected_type in (int, float):
            value = super().read_cfg(None, keys, default)
            if value is not default:
                if expected_type is int and type(value) is not int:
                    raise ValueError(f"{keys} must be an integer, not bool or float")
                if expected_type is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
                    raise ValueError(f"{keys} must be numeric, not bool")
        return super().read_cfg(expected_type, keys, default)

    def _read_limits(self, name):
        values = self.read_cfg(list, name, no_default)
        if len(values) != self.num_hidden_layers:
            raise ValueError(f"{name} must have exactly num_hidden_layers entries")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
               for v in values):
            raise ValueError(f"{name} must contain finite nonnegative limits")
        return [float(v) for v in values]


def bailing_mla(config, key, layer_idx, qbits_key = "bits"):
    return MLAttention(
        config = config, key = key, layer_idx = layer_idx,
        hidden_size = config.hidden_size, num_q_heads = config.num_q_heads,
        kv_lora_rank = config.kv_lora_rank, q_lora_rank = config.q_lora_rank,
        qk_nope_head_dim = config.qk_nope_head_dim,
        qk_rope_head_dim = config.qk_rope_head_dim, v_head_dim = config.v_head_dim,
        rope_settings = config.rope_settings, sm_scale = config.sm_scale,
        rms_norm_eps = config.rms_norm_eps, key_o = "dense", key_gate = "g_proj",
        qmap = "block.attn", out_dtype = torch.float, select_hq_bits = 2,
        qbits_key = qbits_key,
    )


def bailing_mlp(config, key, idx, qbits_key = "bits"):
    # The schedules cover trunk layers only. MTP has no inherited tail clamp.
    expert_limit = config.expert_swiglu_limits[idx] if idx < config.num_hidden_layers else 0.0
    shared_limit = config.shared_swiglu_limits[idx] if idx < config.num_hidden_layers else 0.0
    if idx < config.first_k_dense_replace:
        return GatedMLP(
            config = config, key = key, hidden_size = config.hidden_size,
            intermediate_size = config.intermediate_size,
            key_up = "up_proj", key_gate = "gate_proj", key_down = "down_proj",
            activation_fn = "silu", act_limit = expert_limit, qmap = "block.mlp",
            interm_dtype = torch.half, out_dtype = torch.float, qbits_key = qbits_key,
        )
    shared = GatedMLP(
        config = config, key = f"{key}.shared_experts", hidden_size = config.hidden_size,
        intermediate_size = config.moe_shared_expert_intermediate_size * config.num_shared_experts,
        key_up = "up_proj", key_gate = "gate_proj", key_down = "down_proj",
        activation_fn = "silu", act_limit = shared_limit, qmap = "block.mlp",
        interm_dtype = torch.half, out_dtype = torch.float, select_hq_bits = 2,
        qbits_key = qbits_key,
    ) if config.num_shared_experts else None
    return BlockSparseMLP(
        config = config, key = key, hidden_size = config.hidden_size,
        intermediate_size = config.moe_intermediate_size, num_experts = config.num_experts,
        num_experts_per_tok = config.num_experts_per_tok,
        key_up = "experts.{expert_idx}.up_proj", key_gate = "experts.{expert_idx}.gate_proj",
        key_down = "experts.{expert_idx}.down_proj", key_routing_gate = "gate",
        key_e_score_bias = "gate.expert_bias", require_e_score_bias = True,
        activation_fn = "silu", act_limit = expert_limit,
        qmap = "block.mlp", interm_dtype = torch.half, out_dtype = torch.float,
        router_type = "ds3_fp32", routed_scaling_factor = config.routed_scaling_factor,
        n_group = config.n_group, topk_group = config.topk_group,
        shared_experts = shared, qbits_key = qbits_key,
    )


class BailingMoeV3Model(Model):
    config_class = BailingMoeV3Config

    def __init__(self, config: BailingMoeV3Config, key_prefix: str = "model", **kwargs):
        super().__init__(config, **kwargs)
        self.modules = [Embedding(
            config = config, key = f"{key_prefix}.word_embeddings",
            vocab_size = config.vocab_size, hidden_size = config.hidden_size,
        )]
        self.first_block_idx = len(self.modules)
        for idx, layer_type in enumerate(config.layer_types):
            key = f"{key_prefix}.layers.{idx}"
            if layer_type == "linear_attention":
                attn = GatedDeltaNet(
                    config = config, key = f"{key}.attention", layer_idx = idx,
                    hidden_size = config.hidden_size,
                    k_head_dim = config.linear_head_dim, v_head_dim = config.linear_head_dim,
                    num_k_heads = config.linear_num_heads, num_v_heads = config.linear_num_heads,
                    rms_norm_eps = config.rms_norm_eps, conv_kernel_size = config.linear_conv_kernel_size,
                    key_qkv = "qkv_proj", key_qkv_alt = ["q_proj", "k_proj", "v_proj"],
                    key_conv1d = "conv1d", key_conv1d_q = "q_conv1d",
                    key_conv1d_k = "k_conv1d", key_conv1d_v = "v_conv1d",
                    key_b = "b_proj", key_norm = "o_norm", key_o = "o_proj",
                    key_f = "f_proj", key_g = "g_proj",
                    key_a_log = "A_log", key_dt_bias = "dt_bias",
                    gate_lower_bound = config.linear_lower_bound,
                    qmap = "block.attn", out_dtype = torch.float, select_hq_bits = 2,
                )
            else:
                attn = bailing_mla(config, f"{key}.attention", idx)
            self.modules.append(TransformerBlock(
                config = config, key = key, layer_idx = idx,
                attn_norm = RMSNorm(config, f"{key}.input_layernorm", config.rms_norm_eps),
                attn = attn,
                mlp_norm = RMSNorm(config, f"{key}.post_attention_layernorm", config.rms_norm_eps),
                mlp = bailing_mlp(config, f"{key}.mlp", idx),
            ))
        self.last_kv_module_idx = len(self.modules) - 1
        head_alt_key = f"{key_prefix}.word_embeddings" if config.tie_word_embeddings else None
        self.modules.extend([
            RMSNorm(config, f"{key_prefix}.norm", config.rms_norm_eps, out_dtype = torch.half),
            Linear(
                config = config, key = "lm_head", alt_key = head_alt_key, qbits_key = "head_bits",
                in_features = config.hidden_size, out_features = config.vocab_size,
                qmap = "block", caps = {"logits_output": True},
            ),
        ])
        self.logit_layer_idx = len(self.modules) - 1
        self.calibration_all_experts = True
        self.recurrent_state_cls = GDNState
        self.caps.update({
            "supports_tp": False, "recurrent_states": True, "linear_attn": True,
            "default_recurrent_checkpoint_interval": 2048,
        })

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        prepare_for_recurrence(input_ids, params, self)
        return input_ids

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        # Single-user, optional-system subset of Ling's tokenizer template, with
        # its default thinking option on. History/tools/options use the template.
        p = "<role>SYSTEM</role>"
        if system_prompt is not None:
            p += system_prompt
            if "detailed thinking on" not in system_prompt and "detailed thinking off" not in system_prompt:
                p += "\ndetailed thinking on"
        else:
            p += "detailed thinking on"
        return p + f"<|role_end|><role>HUMAN</role>{prompt}<|role_end|><role>ASSISTANT</role>\n<think>"
