from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, BlockSparseMLP, Linear
from ..modules.attn import prepare_for_attn

class GptOssConfig(Config):
    arch_string = "GptOssForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": GptOssModel},
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Alternating sliding-window/full attention. The HF sliding mask keeps keys with
        # q_pos - k_pos < sliding_window, i.e. the window includes the query itself
        self.sliding_window = self.read_cfg(int, "sliding_window", no_default)
        self.layer_types = self.read_cfg(list, "layer_types", no_default)

        # MLP params. hidden_act in the config reads "silu" but the reference implementation
        # hardcodes the clamped variant with alpha = 1.702 and the +1 on the up projection
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, "num_local_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "num_experts_per_tok", no_default)
        self.swiglu_limit = self.read_cfg(float, "swiglu_limit", 7.0)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # RoPE (YaRN with truncate = false)
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)


class GptOssModel(Model):
    config_class = GptOssConfig

    def __init__(
        self,
        config: GptOssConfig,
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
            swa = config.layer_types[idx] == "sliding_attention"
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    # TODO: Use SlidingAttention module for SWA layer, mark model as recurrent, verify checkpointing etc.
                    attn = Attention(
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
                        key_sinks = "sinks",
                        qmap = "block.attn",
                        # The kernels' left window keeps window + 1 keys including the query, the
                        # HF mask keeps sliding_window including the query
                        sliding_window = config.sliding_window - 1 if swa else -1,
                        out_dtype = torch.float,
                        select_hq_bits = 2,
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = BlockSparseMLP(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        num_experts = config.num_experts,
                        num_experts_per_tok = config.num_experts_per_tok,
                        key_up = "experts.{expert_idx}.up_proj",
                        key_gate = "experts.{expert_idx}.gate_proj",
                        key_down = "experts.{expert_idx}.down_proj",
                        key_gate_up_split = "experts.gate_up_proj",
                        key_down_split = "experts.down_proj",
                        gate_up_interleaved = True,
                        key_routing_gate = "router",
                        key_e_score_bias = None,
                        router_type = "std_bias",
                        activation_fn = "swiglu_oai",
                        act_limit = config.swiglu_limit,
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
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

        self.caps.update({
            "supports_tp": True,
        })


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"<|start|>system<|message|>{system_prompt}<|end|>"
        p += f"<|start|>user<|message|>{prompt}<|end|>"
        p += f"<|start|>assistant"
        return p
