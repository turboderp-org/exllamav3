from __future__ import annotations
from typing_extensions import override
import torch
from .config import Config, no_default
from .model import Model
from ..util.rope import RopeSettings, RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear, BlockSparseMLP
from ..modules.attn import prepare_for_attn

class Ernie4_5MoEConfig(Config):
    arch_string = "Ernie4_5_MoeForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Ernie4_5MoEModel},
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.assert_cfg(str, "moe_gate_act", "softmax", True)
        self.num_shared_experts = self.read_cfg(int, "moe_num_shared_experts", 0)
        self.moe_intermediate_size = self.read_cfg(int, "moe_intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, "moe_num_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "moe_k", no_default)
        self.first_k_dense_replace = self.read_cfg(int, "moe_layer_start_index", 0)
        self.assert_cfg(int, "moe_layer_interval", 1, True)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.GPTJ)


class Ernie4_5MoEModel(Model):
    config_class = Ernie4_5MoEConfig

    def __init__(
        self,
        config: Ernie4_5MoEConfig,
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
                ),
                attn = Attention(
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
                    qmap = "block.attn"
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"model.layers.{idx}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
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
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                    )
                    if idx < config.first_k_dense_replace else
                    BlockSparseMLP(
                        config = config,
                        key = f"model.layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.moe_intermediate_size,
                        num_experts = config.num_experts,
                        num_experts_per_tok = config.num_experts_per_tok,
                        key_up = "experts.{expert_idx}.up_proj",
                        key_gate = "experts.{expert_idx}.gate_proj",
                        key_down = "experts.{expert_idx}.down_proj",
                        key_routing_gate = "gate",
                        key_e_score_bias = "moe_statics.e_score_correction_bias",
                        qmap = "block.mlp",
                        interm_dtype = torch.float,
                        out_dtype = torch.float,
                        routed_scaling_factor = 1.0,
                        n_group = 1,
                        topk_group = 1,
                        shared_experts = (
                            GatedMLP(
                                config = config,
                                key = f"model.layers.{idx}.mlp.shared_experts",
                                hidden_size = config.hidden_size,
                                intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                                key_up = "up_proj",
                                key_gate = "gate_proj",
                                key_down = "down_proj",
                                qmap = "block.mlp",
                                interm_dtype = torch.half,
                                out_dtype = torch.float,
                            )
                            if config.num_shared_experts > 0 else
                            None
                        ),
                    )
                )
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
        params["input_ids"] = input_ids
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = "<|begin_of_sentence|>"
        if system_prompt:
            p += f"{system_prompt}\n"
        p += f"User: {prompt}\n"
        p += f"Assistant: "
        return p