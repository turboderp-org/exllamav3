from __future__ import annotations
from typing_extensions import override
import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle, RoPE
from ..modules import (
    RMSNorm,
    Embedding,
    TransformerBlock,
    Attention,
    Linear,
    GatedMLP,
    ShortConv,
    ShortConvState,
)
from ..modules.attn import prepare_for_attn
from ..cache.recurrent_util import prepare_for_recurrence


def read_lfm2_layer_types(config: Config, num_layers: int) -> list[str]:
    layer_types = config.read_cfg(list, "layer_types", None)
    if layer_types is not None:
        assert len(layer_types) == num_layers, \
            "Length of layer_types key doesn't match number of hidden layers"
        for t in layer_types:
            if t not in ["full_attention", "conv"]:
                raise ValueError(f"Unknown layer type in layer_types: {t}")
        return layer_types

    return [
        "full_attention" if (idx + 1) % 4 == 0 else "conv"
        for idx in range(num_layers)
    ]


class Lfm2Config(Config):
    arch_string = "Lfm2ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Lfm2Model},
            **kwargs,
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "head_dim", None)
        self.hidden_size = self.read_cfg(int, "hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "num_key_value_heads", self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Short-conv params
        self.conv_kernel_size = self.read_cfg(int, "conv_L_cache", 3)

        # MLP params
        self.assert_cfg(str, "hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        self.layer_types = read_lfm2_layer_types(self, self.num_hidden_layers)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = self.read_cfg(float, "rope_theta", 1000000.0),
        )

        # Vision placeholders
        self.vision = None


class Lfm2Model(Model):
    config_class = Lfm2Config

    def __init__(
        self,
        config: Lfm2Config,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        key_prefix = "model"

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
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.operator_norm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    attn = (
                        ShortConv(
                            config = config,
                            key = f"{key_prefix}.layers.{idx}.conv",
                            layer_idx = idx,
                            hidden_size = config.hidden_size,
                            conv_kernel_size = config.conv_kernel_size,
                            key_in = "in_proj",
                            key_conv = "conv",
                            key_out = "out_proj",
                            qmap = "block.attn",
                            out_dtype = torch.float,
                            select_hq_bits = 2,
                        )
                        if config.layer_types[idx] == "conv" else
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
                            key_o = "out_proj",
                            qmap = "block.attn",
                            q_norm = RMSNorm(
                                config = config,
                                key = f"{key_prefix}.layers.{idx}.self_attn.q_layernorm",
                                rms_norm_eps = config.rms_norm_eps,
                            ),
                            k_norm = RMSNorm(
                                config = config,
                                key = f"{key_prefix}.layers.{idx}.self_attn.k_layernorm",
                                rms_norm_eps = config.rms_norm_eps,
                            ),
                            out_dtype = torch.float,
                            select_hq_bits = 2,
                        )
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.ffn_norm",
                        rms_norm_eps = config.rms_norm_eps,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.feed_forward",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "w3",
                        key_gate = "w1",
                        key_down = "w2",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        select_hq_bits = 1,
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
                key = f"{key_prefix}.embedding_norm",
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
                caps = {"logits_output": True},
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        self.caps.update({
            "supports_tp": False,
            "recurrent_states": True,
            "default_recurrent_checkpoint_interval": 2048,
            "linear_attn": True,
        })
        self.recurrent_state_cls = ShortConvState
        self.g_rope = RoPE("cpu", config.rope_settings)


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
