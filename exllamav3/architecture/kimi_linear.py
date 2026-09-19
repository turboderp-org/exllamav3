from __future__ import annotations
from typing_extensions import override
import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import (
    RMSNorm, Embedding, TransformerBlock, MLAttention, GatedMLP, Linear, BlockSparseMLP,
    GatedDeltaNet,
)
from ..modules.gated_delta_net import GDNState
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence

# Kimi Linear (KimiLinearForCausalLM): hybrid of KDA linear attention (per-channel-decay delta
# rule, 3:1) and MLA, DeepSeek-V3-style sigmoid MoE with a shared expert. The MLA layers use no
# positional encoding: the projections still carry the DeepSeek "rope" slices (q: nope + pe per
# head, kv_a: latent + one shared pe key) but nothing is ever rotated, so the module is built
# without rope settings and the pe slices go into the scores as projected. Reference: bundled
# modeling_kimi.py (Moonshot AI).


class KimiLinearConfig(Config):
    arch_string = "KimiLinearForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": KimiLinearModel},
            **kwargs
        )

        # Attention (MLA, no positional encoding)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.q_lora_rank = self.read_cfg(int, "q_lora_rank", None)
        self.kv_lora_rank = self.read_cfg(int, "kv_lora_rank", no_default)
        self.qk_nope_head_dim = self.read_cfg(int, "qk_nope_head_dim", no_default)
        self.qk_rope_head_dim = self.read_cfg(int, "qk_rope_head_dim", 0)
        self.v_head_dim = self.read_cfg(int, "v_head_dim", no_default)
        self.assert_cfg(bool, "mla_use_nope", True, True)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_head_dim
        self.sm_scale = self.qk_head_dim ** -0.5
        self.rope_settings = None

        # Layer schedule: 1-indexed layer lists
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        la = self.read_cfg(dict, "linear_attn_config", no_default)
        kda_layers = set(la["kda_layers"])
        full_layers = set(la["full_attn_layers"])
        self.layer_types = [
            "linear_attention" if (idx + 1) in kda_layers else "full_attention"
            for idx in range(self.num_hidden_layers)
        ]
        assert all((idx + 1) in kda_layers or (idx + 1) in full_layers for idx in range(self.num_hidden_layers)), \
            "linear_attn_config must assign every layer to kda_layers or full_attn_layers"

        # KDA linear attention
        self.linear_num_heads = la["num_heads"]
        self.linear_head_dim = la["head_dim"]
        self.linear_conv_kernel_size = la["short_conv_kernel_size"]
        self.linear_lower_bound = la.get("gate_lower_bound")

        # MLP params
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", no_default)
        self.num_shared_experts = self.read_cfg(int, "num_shared_experts", 0)
        self.num_experts = self.read_cfg(int, "num_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_token", 8)
        self.routed_scaling_factor = self.read_cfg(float, "routed_scaling_factor", 1.0)
        first_k_dense = self.read_cfg(int, "first_k_dense_replace", 0)
        moe_layer_freq = self.read_cfg(int, "moe_layer_freq", 1)
        self.mlp_layer_types = [
            "sparse" if idx >= first_k_dense and idx % moe_layer_freq == 0 else "dense"
            for idx in range(self.num_hidden_layers)
        ]
        self.n_group = self.read_cfg(int, "num_expert_group", 1)
        self.topk_group = self.read_cfg(int, "topk_group", 1)
        assert self.n_group in (None, 1) and self.topk_group in (None, 1), \
            "Group-limited expert routing is not supported"
        self.assert_cfg(str, "moe_router_activation_func", "sigmoid", True)
        self.assert_cfg(bool, "moe_renormalize", True, True)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        self.assert_cfg(int, "num_nextn_predict_layers", 0, True)


class KimiLinearModel(Model):
    config_class = KimiLinearConfig

    def __init__(
        self,
        config: KimiLinearConfig,
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
            key = f"{key_prefix}.layers.{idx}"

            if config.layer_types[idx] == "linear_attention":
                attn = GatedDeltaNet(
                    config = config,
                    key = f"{key}.self_attn",
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
                    key_f_a = "f_a_proj",
                    key_f_b = "f_b_proj",
                    key_g_a = "g_a_proj",
                    key_g_b = "g_b_proj",
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
                attn = MLAttention(
                    config = config,
                    key = f"{key}.self_attn",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    num_q_heads = config.num_q_heads,
                    kv_lora_rank = config.kv_lora_rank,
                    qk_nope_head_dim = config.qk_nope_head_dim,
                    qk_rope_head_dim = config.qk_rope_head_dim,
                    v_head_dim = config.v_head_dim,
                    rope_settings = None,
                    q_lora_rank = config.q_lora_rank,
                    sm_scale = config.sm_scale,
                    rms_norm_eps = config.rms_norm_eps,
                    qmap = "block.attn",
                    out_dtype = torch.float,
                    select_hq_bits = 2,
                )

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
                mlp = BlockSparseMLP(
                    config = config,
                    key = f"{key}.block_sparse_moe",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.moe_intermediate_size,
                    num_experts = config.num_experts,
                    num_experts_per_tok = config.num_experts_per_tok,
                    key_up = "experts.{expert_idx}.w3",
                    key_gate = "experts.{expert_idx}.w1",
                    key_down = "experts.{expert_idx}.w2",
                    key_routing_gate = "gate",
                    key_e_score_bias = "gate.e_score_correction_bias",
                    activation_fn = "silu",
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    router_type = "dots",
                    routed_scaling_factor = config.routed_scaling_factor,
                    n_group = config.n_group,
                    topk_group = config.topk_group,
                    shared_experts = GatedMLP(
                        config = config,
                        key = f"{key}.block_sparse_moe.shared_experts",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        activation_fn = "silu",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        select_hq_bits = 2,
                    ) if config.num_shared_experts else None,
                )

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

        self.calibration_all_experts = True
        self.caps.update({
            "supports_tp": True,
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


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"<|im_system|>system<|im_middle|>{system_prompt}<|im_end|>"
        p += f"<|im_user|>user<|im_middle|>{prompt}<|im_end|>"
        p += f"<|im_assistant|>assistant<|im_middle|>"
        return p
