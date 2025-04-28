from __future__ import annotations
from typing_extensions import override
import torch
from .config import Config, no_default
from .model import Model
from ..util.rope import RopeSettings, RopeStyle
from ..modules import LayerNorm, Embedding, ParallelDecoderBlock, Attention, GatedMLP, Linear
from ..modules.attn import prepare_for_attn

class Cohere2Config(Config):
    arch_string = "Cohere2ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            Cohere2Model,
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)

        self.sliding_window = self.read_cfg(int, "sliding_window", -1)
        self.sliding_window_pattern = self.read_cfg(int, "sliding_window_pattern", 1)
        self.assert_cfg(str, "order_of_interleaved_layers", "local_attn_first")

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)

        # Norms
        self.layernorm_eps = self.read_cfg(float, "layer_norm_eps", 1e-05)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", True)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.GPTJ)

        # Logit scale
        self.logit_scale = self.read_cfg(float, "logit_scale", 0.0625)


class Cohere2Model(Model):
    config_class = Cohere2Config

    def __init__(
        self,
        config: Cohere2Config,
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

        swa = [
            config.sliding_window if (idx + 1) % config.sliding_window_pattern != 0 else -1
            for idx in range(config.num_hidden_layers)
        ]

        self.modules += [
            ParallelDecoderBlock(
                config = config,
                key = f"model.layers.{idx}",
                input_norm = LayerNorm(
                    config = config,
                    key = f"model.layers.{idx}.input_layernorm",
                    layernorm_eps = config.layernorm_eps,
                ),
                attn = Attention(
                    config = config,
                    key = f"model.layers.{idx}.self_attn",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    head_dim = config.head_dim,
                    num_q_heads = config.num_q_heads,
                    num_kv_heads = config.num_kv_heads,
                    rope_settings = config.rope_settings if swa[idx] >= 0 else None,
                    sm_scale = None,
                    sliding_window = swa[idx],
                    key_q = "q_proj",
                    key_k = "k_proj",
                    key_v = "v_proj",
                    key_o = "o_proj",
                    qmap = "block.parallel",
                ),
                mlp = GatedMLP(
                    config = config,
                    key = f"model.layers.{idx}.mlp",
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    key_up = "up_proj",
                    key_gate = "gate_proj",
                    key_down = "down_proj",
                    qmap = "block.parallel",
                ),
            )
            for idx in range(config.num_hidden_layers)
        ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("lm_head"):
            head_alt_key = "model.embed_tokens"

        self.modules += [
            LayerNorm(
                config = config,
                key = "model.norm",
                layernorm_eps = config.layernorm_eps,
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
                post_scale = config.logit_scale
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        params["input_ids"] = input_ids
        input_ids = prepare_for_attn(input_ids, params)
        return input_ids