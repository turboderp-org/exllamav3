from __future__ import annotations
from typing_extensions import override
import torch
from .config import Config, no_default
from .model import Model
from ..util.rope import RopeSettings, RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear
from ..modules.attn import prepare_for_attn

class Gemma2Config(Config):
    arch_string = "Gemma2ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            Gemma2Model,
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)

        self.query_pre_attn_scalar = self.read_cfg(float, "query_pre_attn_scalar", 224)
        self.attn_logit_softcapping = self.read_cfg(float, "attn_logit_softcapping", 50.0)
        self.sliding_window = self.read_cfg(int, "sliding_window_size", -1)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.assert_cfg(str, "hidden_act", "gelu_pytorch_tanh", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)

        # Output softcap
        self.final_logit_softcapping = self.read_cfg(float, "final_logit_softcapping", 30.0)


class Gemma2Model(Model):
    config_class = Gemma2Config

    def __init__(
        self,
        config: Gemma2Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = "model.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
                normalize = True,
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
                attn = Attention(
                    config = config,
                    key = f"model.layers.{idx}.self_attn",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    head_dim = config.head_dim,
                    num_q_heads = config.num_q_heads,
                    num_kv_heads = config.num_kv_heads,
                    rope_settings = config.rope_settings,
                    sm_scale = config.query_pre_attn_scalar ** (-0.5),
                    logit_softcapping = config.attn_logit_softcapping,
                    sliding_window = config.sliding_window if not bool(idx % 2) else -1,
                    key_q = "q_proj",
                    key_k = "k_proj",
                    key_v = "v_proj",
                    key_o = "o_proj",
                    qmap = "block.attn",
                ),
                attn_post_norm = RMSNorm(
                    config = config,
                    key = f"model.layers.{idx}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                    out_dtype = torch.float,
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"model.layers.{idx}.pre_feedforward_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                mlp = GatedMLP(
                    config = config,
                    key = f"model.layers.{idx}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = "up_proj",
                    key_gate = "gate_proj",
                    key_down = "down_proj",
                    qmap = "block.mlp",
                    activation_fn = "gelu"
                ),
                mlp_post_norm = RMSNorm(
                    config = config,
                    key = f"model.layers.{idx}.post_feedforward_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                    out_dtype = torch.float,
                ),
            )
            for idx in range(config.num_hidden_layers)
        ]

        self.last_kv_module_idx = len(self.modules) - 1

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
                alt_key = "model.embed_tokens",
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                softcap = config.final_logit_softcapping,
                caps = {"logits_output": True}
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        params["input_ids"] = input_ids
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids