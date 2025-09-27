from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, BlockSparseMLP, Linear, GatedDeltaNet, GatedMLP
from ..modules.attn import prepare_for_attn
from ..modules.gated_delta_net import prepare_for_recurrence

class Qwen3NextConfig(Config):
    arch_string = "Qwen3NextForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Qwen3NextModel},
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        self.assert_cfg(bool, "use_sliding_window", False, True)
        self.full_attention_interval = self.read_cfg(int, "full_attention_interval", 4)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Linear attn params
        self.full_attention_interval = self.read_cfg(int, "full_attention_interval", 4)
        self.linear_conv_kernel_dim = self.read_cfg(int, "linear_conv_kernel_dim", 4)
        self.linear_num_key_heads = self.read_cfg(int, "linear_num_key_heads", 16)
        self.linear_num_value_heads = self.read_cfg(int, "linear_num_value_heads", 32)
        self.linear_key_head_dim = self.read_cfg(int, "linear_key_head_dim", 128)
        self.linear_value_head_dim = self.read_cfg(int, "linear_value_head_dim", 128)

        # MLP params
        self.decoder_sparse_step = self.read_cfg(int, "decoder_sparse_step", 0)
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.assert_cfg(bool, "norm_topk_prob", True, True)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, "num_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", no_default)

        self.mlp_only_layers = self.read_cfg(list, "mlp_only_layers", [])
        self.shared_expert_intermediate_size = self.read_cfg(int, "shared_expert_intermediate_size", 512)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)


def conditional(condition, a, b):
    return a if condition else b


class Qwen3NextModel(Model):
    config_class = Qwen3NextConfig

    def __init__(
        self,
        config: Qwen3NextConfig,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = "model.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

        self.modules += [
            TransformerBlock(
                config = config,
                key = f"model.layers.{idx}",
                attn_norm = RMSNorm(
                    config = config,
                    key = f"model.layers.{idx}.input_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                attn = conditional(
                    (idx + 1) % config.full_attention_interval != 0,
                    GatedDeltaNet(
                        config = config,
                        key = f"model.layers.{idx}.linear_attn",
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
                        key_fused_ba = "in_proj_ba",
                        key_fused_qkvz = "in_proj_qkvz",
                        key_norm = "norm",
                        key_o = "out_proj",
                        qmap = "block.attn",
                        out_dtype = torch.float,
                    ),
                    Attention(
                        config = config,
                        key = f"model.layers.{idx}.self_attn",
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
                            key = f"model.layers.{idx}.self_attn.q_norm",
                            rms_norm_eps = config.rms_norm_eps,
                            constant_bias = 1.0,
                        ),
                        k_norm = RMSNorm(
                            config = config,
                            key = f"model.layers.{idx}.self_attn.k_norm",
                            rms_norm_eps = config.rms_norm_eps,
                            constant_bias = 1.0,
                        ),
                        out_dtype = torch.float,
                        interleaved_gate = True,
                    )
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"model.layers.{idx}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                mlp = BlockSparseMLP(
                    config = config,
                    key = f"model.layers.{idx}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.moe_intermediate_size,
                    num_experts = self.config.num_experts,
                    num_experts_per_tok = self.config.num_experts_per_tok,
                    key_up = "experts.{expert_idx}.up_proj",
                    key_gate = "experts.{expert_idx}.gate_proj",
                    key_down = "experts.{expert_idx}.down_proj",
                    key_routing_gate = "gate",
                    key_shared_gate = "shared_expert_gate",
                    qmap = "block.mlp",
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
                    shared_experts = GatedMLP(
                        config = config,
                        key = f"model.layers.{idx}.mlp.shared_expert",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.shared_expert_intermediate_size,
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
            head_alt_key = "model.embed_tokens"

        self.modules += [
            RMSNorm(
                config = config,
                key = "model.norm",
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

        # Mark that we need recurrent cache for generation
        self.caps.update({"recurrent_states": True})

        # TODO: Enable TP for linear attn
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


    @override
    def check_compat(self):
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
            from fla.modules.fused_norm_gate import rms_norm_gated
        except ModuleNotFoundError as e:
            print(" ## Qwen3-Next requires flash-linear-attention (https://github.com/fla-org/flash-linear-attention)")
            raise e