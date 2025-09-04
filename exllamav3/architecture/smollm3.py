from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear
from ..modules.attn import prepare_for_attn

class SmolLM3Config(Config):
    arch_string = "SmolLM3ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": SmolLM3Model},
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        self.layer_types = self.read_cfg(list, "layer_types", None)
        if self.layer_types:
            assert all(x == "full_attention" for x in self.layer_types)

        # MLP params
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", True)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)
        self.no_rope_layers = self.read_cfg(list, "no_rope_layers", no_default)


class SmolLM3Model(Model):
    config_class = SmolLM3Config

    def __init__(
        self,
        config: SmolLM3Config,
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
                    rope_settings = config.rope_settings if config.no_rope_layers[idx] == 1 else None,
                    sm_scale = None,
                    key_q = "q_proj",
                    key_k = "k_proj",
                    key_v = "v_proj",
                    key_o = "o_proj",
                    qmap = "block.attn",
                    out_dtype = torch.float,
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"model.layers.{idx}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
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
                    interm_dtype = torch.half,
                    out_dtype = torch.float,
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