from __future__ import annotations
from typing_extensions import override
import torch

from ..util.rope import RopeStyle, RoPE
from .qwen3_5 import read_qwen3_5_layer_types
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import (
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    Linear,
    GatedDeltaNet,
    GatedMLP,
)
from ..modules.attn import prepare_for_attn
from ..modules.gated_delta_net import prepare_for_recurrence

class OlmoHybridConfig(Config):
    arch_string = "OlmoHybridForCausalLM"

    def __init__(
        self,
        directory: str,
        text_cfg: str | None = None,
        **kwargs,
    ):
        super().__init__(
            directory,
            { "text": OlmoHybridModel },
            **kwargs
        )

        def pfx(key):
            nonlocal text_cfg
            return key if not text_cfg else f"{text_cfg}->{key}"

        # Attention params
        self.head_dim = self.read_cfg(int, pfx("head_dim"), None)
        self.hidden_size = self.read_cfg(int, pfx("hidden_size"), no_default)
        self.num_q_heads = self.read_cfg(int, pfx("num_attention_heads"), no_default)
        self.num_kv_heads = self.read_cfg(int, pfx("num_key_value_heads"), self.num_q_heads)
        self.full_attention_interval = self.read_cfg(int, pfx("full_attention_interval"), 4)
        self.linear_allow_neg_eigval = self.read_cfg(bool, pfx("linear_allow_neg_eigval"), False)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Linear attn params
        self.linear_conv_kernel_dim = self.read_cfg(int, pfx("linear_conv_kernel_dim"), 4)
        self.linear_num_key_heads = self.read_cfg(int, pfx("linear_num_key_heads"), 16)
        self.linear_num_value_heads = self.read_cfg(int, pfx("linear_num_value_heads"), 32)
        self.linear_key_head_dim = self.read_cfg(int, pfx("linear_key_head_dim"), 128)
        self.linear_value_head_dim = self.read_cfg(int, pfx("linear_value_head_dim"), 128)

        # MLP params
        self.assert_cfg(str, pfx("hidden_act"), "silu", True)
        self.intermediate_size = self.read_cfg(int, pfx("intermediate_size"), no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, pfx("rms_norm_eps"), no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, pfx("num_hidden_layers"), no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        self.layer_types = read_qwen3_5_layer_types(
            self,
            text_cfg,
            self.num_hidden_layers,
            self.full_attention_interval,
        )

        # RoPE
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 10000,
            config_dict = self.read_cfg(dict, text_cfg, no_default) if text_cfg else None
        )

class OlmoHybridModel(Model):
    config_class = OlmoHybridConfig

    def __init__(
        self,
        config: OlmoHybridConfig,
        key_prefix: str = "model",
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.key_prefix = key_prefix

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
            post_norms = config.layer_types[idx] != "linear_attention"
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}",
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.input_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ) if not post_norms else None,
                    attn_post_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ) if post_norms else None,
                    attn = (
                        GatedDeltaNet(
                            config = config,
                            key = f"{self.key_prefix}.layers.{idx}.linear_attn",
                            layer_idx = idx,
                            hidden_size = config.hidden_size,
                            k_head_dim = config.linear_key_head_dim,
                            v_head_dim = config.linear_value_head_dim,
                            num_k_heads = config.linear_num_key_heads,
                            num_v_heads = config.linear_num_value_heads,
                            rms_norm_eps = 1e-5,  # Hardcoded in reference
                            conv_kernel_size = config.linear_conv_kernel_dim,
                            beta_scale = 2.0 if config.linear_allow_neg_eigval else 1.0,
                            key_a_log = "A_log",
                            key_dt_bias = "dt_bias",
                            key_conv1d_q = "q_conv1d",
                            key_conv1d_k = "k_conv1d",
                            key_conv1d_v = "v_conv1d",
                            key_qkv = "in_proj_qkv",
                            key_qkv_alt = ["q_proj", "k_proj", "v_proj"],
                            key_z = "g_proj",
                            key_b = "b_proj",
                            key_a = "a_proj",
                            key_norm = "o_norm",
                            key_o = "o_proj",
                            qmap = "block.attn",
                            out_dtype = torch.float,
                        )
                        if config.layer_types[idx] == "linear_attention" else
                        (
                            Attention(
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
                                qmap = "block.attn",
                                q_norm = RMSNorm(
                                    config = config,
                                    key = f"{key_prefix}.layers.{idx}.self_attn.q_norm",
                                    rms_norm_eps = config.rms_norm_eps,
                                    span_heads = True,
                                ),
                                k_norm = RMSNorm(
                                    config = config,
                                    key = f"{key_prefix}.layers.{idx}.self_attn.k_norm",
                                    rms_norm_eps = config.rms_norm_eps,
                                    span_heads = True,
                                ),
                                out_dtype = torch.float,
                                interleaved_gate = False,
                            )
                        )
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_attention_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ) if not post_norms else None,
                    mlp_post_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.post_feedforward_layernorm",
                        rms_norm_eps = config.rms_norm_eps,
                    ) if post_norms else None,
                    mlp = (
                        GatedMLP(
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
                        )
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

        # Mark that we need recurrent cache for generation
        self.caps.update({"recurrent_states": True})

        # TP for this architecture is not implemented yet
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
