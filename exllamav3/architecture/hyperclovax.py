from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle, RoPE
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear, DeepstackEmbed
from ..modules.attn import prepare_for_attn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .hcxvisionv2 import HCXVisionV2Config

class HyperClovaxConfig(Config):
    arch_string = "HyperCLOVAXForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": HyperClovaxModel},
            **kwargs
        )

        self.embedding_multiplier = self.read_cfg(float, "embedding_multiplier", 1.0)
        self.logits_scaling = self.read_cfg(float, "logits_scaling", no_default)

        self.assert_cfg(float, "residual_multiplier", 1.0, True)

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)
        self.attention_multiplier = self.read_cfg(float, "attention_multiplier", None)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(RopeStyle.NEOX)


class HyperClovaxModel(Model):
    config_class = HyperClovaxConfig

    def __init__(
        self,
        config: HyperClovaxConfig | HCXVisionV2Config,
        key_prefix: str = "model",
        head_key: str = "lm_head",
        post_norms: bool = True,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = f"{key_prefix}.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
                multiplier = config.embedding_multiplier
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}",
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = Attention(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.self_attn",
                        layer_idx = idx,
                        hidden_size = config.hidden_size,
                        head_dim = config.head_dim,
                        num_q_heads = config.num_q_heads,
                        num_kv_heads = config.num_kv_heads,
                        rope_settings = config.rope_settings,
                        sm_scale = config.attention_multiplier,
                        key_q = "q_proj",
                        key_k = "k_proj",
                        key_v = "v_proj",
                        key_o = "o_proj",
                        qmap = "block.attn",
                        out_dtype = torch.float
                    ),
                    attn_post_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_norm1",
                        rms_norm_eps = config.rms_norm_eps,
                        out_dtype = torch.float,
                    ) if post_norms else None,
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                    ),
                    mlp_post_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_norm2",
                        rms_norm_eps = config.rms_norm_eps,
                        out_dtype = torch.float,
                    ) if post_norms else None,
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
                key = head_key,
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True},
                post_scale = config.logits_scaling
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