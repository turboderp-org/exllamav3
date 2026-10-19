from __future__ import annotations
from typing_extensions import override
import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import (
    RMSNorm, Embedding, TransformerBlock, MLAttention, GatedMLP, Linear, BlockSparseMLP,
    GatedDeltaNet,
)
from ..modules.gated_delta_net import GDNState
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence

# Ling 3.x Flash (BailingMoeV3ForCausalLM, model_type bailing_hybrid): KDA linear attention with
# full-rank forget and output gates (no_kda_lora), MLA on every layer_group_size-th layer, the
# first first_k_dense_replace layers dense, then a DeepSeek-V3-style sigmoid MoE with group-
# limited routing and one shared expert. Reference: bundled modeling_bailing_moe_v3.py
# (inclusionAI).
#
# Late layers clamp the SwiGLU: act = min(silu(gate), L) * clamp(up, -L, L) with a per-layer L
# (expert_swiglu_limit_list for the routed experts, share_expert_swiglu_limit_list for the shared
# expert, 0 = off). The bundled HF modeling ignores the lists; inclusionAI's vLLM
# (SwigluStepAndMul) and SGLang (BailingMoeV3MLP) apply them. This is exllamav3's act_limit with
# activation_fn "silu" (as in Step-3.5), which every MoE/MLP tier implements.


class BailingHybridConfig(Config):
    arch_string = "BailingMoeV3ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        from .bailing_hybrid_mtp import BailingHybridMTPModel
        super().__init__(
            directory,
            {"text": BailingHybridModel, "mtp": BailingHybridMTPModel},
            **kwargs
        )

        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # Layer schedule: MLA on every group's last layer and on any tail past the last full group
        group = self.read_cfg(int, "layer_group_size", no_default)
        full = self.num_hidden_layers // group * group
        self.layer_types = [
            "full_attention" if (idx + 1) % group == 0 or idx >= full else "linear_attention"
            for idx in range(self.num_hidden_layers)
        ]

        # KDA: one head geometry for q, k and v, full-rank gates, "safe gate" lower bound
        self.assert_cfg(bool, "no_kda_lora", True, True)
        self.linear_num_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.linear_head_dim = self.read_cfg(int, "head_dim", no_default)
        self.linear_conv_kernel_size = self.read_cfg(int, "short_conv_kernel_size", no_default)
        safe_gate = self.read_cfg(bool, "kda_safe_gate", False)
        self.linear_lower_bound = self.read_cfg(float, "kda_lower_bound", None) if safe_gate else None

        # MLA
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.q_lora_rank = self.read_cfg(int, "q_lora_rank", None)
        self.kv_lora_rank = self.read_cfg(int, "kv_lora_rank", no_default)
        self.qk_nope_head_dim = self.read_cfg(int, "qk_nope_head_dim", no_default)
        self.qk_rope_head_dim = self.read_cfg(int, "qk_rope_head_dim", no_default)
        self.v_head_dim = self.read_cfg(int, "v_head_dim", no_default)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_head_dim
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.GPTJ,
            override_head_dim = self.qk_rope_head_dim,
        )
        self.sm_scale = self.qk_head_dim ** -0.5
        gate = self.read_cfg(str, "gated_attention_proj_granularity_type", None)
        assert gate in (None, "head_wise"), f"MLA output gate {gate} is not supported"
        self.head_gate = gate == "head_wise"

        # MLP
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", no_default)
        self.num_shared_experts = self.read_cfg(int, "num_shared_experts", 0)
        self.num_experts = self.read_cfg(int, "num_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", no_default)
        self.routed_scaling_factor = self.read_cfg(float, "routed_scaling_factor", 1.0)
        self.n_group = self.read_cfg(int, "n_group", 1)
        self.topk_group = self.read_cfg(int, "topk_group", 1)
        self.assert_cfg(str, "score_function", "sigmoid", True)
        self.assert_cfg(bool, "norm_topk_prob", True, True)
        first_k_dense = self.read_cfg(int, "first_k_dense_replace", 0)
        self.mlp_layer_types = ["dense" if idx < first_k_dense else "sparse" for idx in range(self.num_hidden_layers)]
        self.act_limits = self.read_cfg(list, "expert_swiglu_limit_list", [0] * self.num_hidden_layers)
        self.shared_act_limits = self.read_cfg(list, "share_expert_swiglu_limit_list", [0] * self.num_hidden_layers)

        # MTP head: one DeepSeek-V3-style layer stored past the trunk (model.layers.N), see
        # bailing_hybrid_mtp.py. The clamp lists cover the trunk only
        self.num_mtp_layers = self.read_cfg(int, "num_nextn_predict_layers", 0)
        if not self.num_mtp_layers:
            self.model_classes.pop("mtp", None)


def bailing_mla(config, key, layer_idx, qbits_key = "bits"):
    """MLA with the head-wise sigmoid output gate (g_proj) and `dense` as the output projection"""
    return MLAttention(
        config = config,
        key = key,
        layer_idx = layer_idx,
        hidden_size = config.hidden_size,
        num_q_heads = config.num_q_heads,
        kv_lora_rank = config.kv_lora_rank,
        qk_nope_head_dim = config.qk_nope_head_dim,
        qk_rope_head_dim = config.qk_rope_head_dim,
        v_head_dim = config.v_head_dim,
        rope_settings = config.rope_settings,
        q_lora_rank = config.q_lora_rank,
        sm_scale = config.sm_scale,
        rms_norm_eps = config.rms_norm_eps,
        key_o = "dense",
        key_head_gate = "g_proj" if config.head_gate else None,
        qmap = "block.attn",
        out_dtype = torch.float,
        select_hq_bits = 2,
        qbits_key = qbits_key,
    )


def bailing_moe(config, key, act_limit, shared_act_limit, qbits_key = "bits"):
    """Sigmoid MoE with group-limited routing (ds3) and one shared expert"""
    return BlockSparseMLP(
        config = config,
        key = key,
        hidden_size = config.hidden_size,
        intermediate_size = config.moe_intermediate_size,
        num_experts = config.num_experts,
        num_experts_per_tok = config.num_experts_per_tok,
        key_up = "experts.{expert_idx}.up_proj",
        key_gate = "experts.{expert_idx}.gate_proj",
        key_down = "experts.{expert_idx}.down_proj",
        key_routing_gate = "gate",
        key_e_score_bias = "gate.expert_bias",
        activation_fn = "silu",
        act_limit = float(act_limit or 0.0),
        qmap = "block.mlp",
        interm_dtype = torch.half,
        out_dtype = torch.float,
        router_type = "ds3",
        routed_scaling_factor = config.routed_scaling_factor,
        n_group = config.n_group,
        topk_group = config.topk_group,
        qbits_key = qbits_key,
        shared_experts = GatedMLP(
            config = config,
            key = f"{key}.shared_experts",
            hidden_size = config.hidden_size,
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
            key_up = "up_proj",
            key_gate = "gate_proj",
            key_down = "down_proj",
            activation_fn = "silu",
            act_limit = float(shared_act_limit or 0.0),
            qmap = "block.mlp",
            interm_dtype = torch.half,
            out_dtype = torch.float,
            select_hq_bits = 2,
            qbits_key = qbits_key,
        ) if config.num_shared_experts else None,
    )


class BailingHybridModel(Model):
    config_class = BailingHybridConfig

    def __init__(
        self,
        config: BailingHybridConfig,
        key_prefix: str = "model",
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = f"{key_prefix}.word_embeddings",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            key = f"{key_prefix}.layers.{idx}"

            if config.layer_types[idx] == "linear_attention":
                attn = GatedDeltaNet(
                    config = config,
                    key = f"{key}.attention",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    k_head_dim = config.linear_head_dim,
                    v_head_dim = config.linear_head_dim,
                    num_k_heads = config.linear_num_heads,
                    num_v_heads = config.linear_num_heads,
                    rms_norm_eps = config.rms_norm_eps,
                    conv_kernel_size = config.linear_conv_kernel_size,
                    key_qkv = "qkv_proj",
                    key_qkv_alt = ["q_proj", "k_proj", "v_proj"],
                    key_conv1d = "conv1d",
                    key_conv1d_q = "q_conv1d",
                    key_conv1d_k = "k_conv1d",
                    key_conv1d_v = "v_conv1d",
                    key_b = "b_proj",
                    key_f = "f_proj",
                    key_g = "g_proj",
                    gate_lower_bound = config.linear_lower_bound,
                    key_a_log = "A_log",
                    key_dt_bias = "dt_bias",
                    key_norm = "o_norm",
                    key_o = "o_proj",
                    qmap = "block.attn",
                    out_dtype = torch.float,
                    select_hq_bits = 2,
                )
            else:
                attn = bailing_mla(config, f"{key}.attention", idx)

            if config.mlp_layer_types[idx] == "dense":
                mlp = GatedMLP(
                    config = config,
                    key = f"{key}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = "up_proj",
                    key_gate = "gate_proj",
                    key_down = "down_proj",
                    activation_fn = "silu",
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    select_hq_bits = 1,
                )
            else:
                mlp = bailing_moe(config, f"{key}.mlp", config.act_limits[idx], config.shared_act_limits[idx])

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = attn,
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = mlp,
                )
            ]

        self.last_kv_module_idx = len(self.modules) - 1

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
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True}
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        self.calibration_all_experts = True
        self.caps.update({
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
            "linear_attn": True,
        })
        self.recurrent_state_cls = GDNState


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        prepare_for_recurrence(input_ids, params, self)
        return input_ids
