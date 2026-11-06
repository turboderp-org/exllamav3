from __future__ import annotations
from typing_extensions import override
import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeStyle
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


def read_lfm3_layer_types(config: Config, num_layers: int, prefix: str = "") -> list[str]:
    """
    LFM3 interleaves three layer types: short convolution, full attention and sliding-window
    attention. The pattern comes from the config; the fallback matches the released checkpoints
    (conv, conv, full attention, conv, sliding attention).
    """
    layer_types = config.read_cfg(list, f"{prefix}layer_types", None)
    if layer_types is not None:
        assert len(layer_types) == num_layers, \
            "Length of layer_types key doesn't match number of hidden layers"
        for t in layer_types:
            if t not in ["full_attention", "sliding_attention", "conv"]:
                raise ValueError(f"Unknown layer type in layer_types: {t}")
        return layer_types

    pattern = ["conv", "conv", "full_attention", "conv", "sliding_attention"]
    return [pattern[idx % len(pattern)] for idx in range(num_layers)]


class Lfm3Config(Config):
    arch_string = "Lfm3ForCausalLM"
    config_key = None

    # Prefix of the text-model config keys: empty for the standalone model, "text_config->" for the
    # multimodal variant, which keeps the language model's config in a nested dict
    prefix = ""

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            self.get_model_classes(),
            config_key = self.config_key,
            **kwargs,
        )

        self.read_text_config(self.prefix, self.get_text_config_dict())

    def get_model_classes(self) -> dict:
        return {"text": Lfm3Model}

    def get_text_config_dict(self) -> dict:
        return self.config_dict

    def read_text_config(self, prefix: str, config_dict: dict):
        # Attention params
        self.head_dim = self.read_cfg(int, f"{prefix}head_dim", None)
        self.hidden_size = self.read_cfg(int, f"{prefix}hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, f"{prefix}num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, f"{prefix}num_key_value_heads", self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Short-conv params
        self.conv_kernel_size = self.read_cfg(int, f"{prefix}conv_L_cache", 3)
        # exllamav3's linear layers have no bias, so only bias-free convolutions can be loaded
        self.assert_cfg(bool, f"{prefix}conv_bias", False, True)

        # MLP params
        self.assert_cfg(str, f"{prefix}hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, f"{prefix}intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, f"{prefix}norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, f"{prefix}num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        self.layer_types = read_lfm3_layer_types(self, self.num_hidden_layers, prefix)

        # Sandwich norm: norm after the attention/conv output and after the MLP output, before the
        # residual add
        self.use_sandwich_norm = self.read_cfg(bool, f"{prefix}use_sandwich_norm", True)

        # Sliding window, in tokens including the query (like the HF mask)
        self.sliding_window = self.read_cfg(int, f"{prefix}sliding_window", -1)

        # RoPE is given per layer type, nested in rope_parameters (transformers v5 style)
        rope_parameters = config_dict.get("rope_parameters", None)

        def rope_settings_for(layer_type: str):
            params = dict(rope_parameters[layer_type]) if rope_parameters and layer_type in rope_parameters else {}
            synth = dict(config_dict)
            synth["rope_parameters"] = params
            rope_type = params.get("rope_type", params.get("type", "default"))
            # LFM3's "default" rope keeps the frequency progression of the full head dim and zeroes
            # the tail of the rotated range (partial_rotary_factor) instead of shrinking it, which
            # is exllamav3's "proportional" layout
            override_type = "proportional" if rope_type == "default" else rope_type
            return self.read_rope_settings_default(
                RopeStyle.NEOX,
                config_dict = synth,
                override_type = override_type,
            )

        rope_settings_by_type = {lt: rope_settings_for(lt) for lt in set(self.layer_types)}
        self.rope_settings_list = [rope_settings_by_type[lt] for lt in self.layer_types]

        # Vision placeholders
        self.vision = None


class Lfm3Model(Model):
    config_class = Lfm3Config

    def __init__(
        self,
        config: Lfm3Config,
        key_prefix: str = "model",
        **kwargs,
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
            layer_type = config.layer_types[idx]

            attn_post_norm = None
            mlp_post_norm = None
            if config.use_sandwich_norm:
                attn_post_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.post_operator_norm",
                    rms_norm_eps = config.rms_norm_eps,
                )
                mlp_post_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.post_ffn_norm",
                    rms_norm_eps = config.rms_norm_eps,
                )

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
                        )
                        if layer_type == "conv" else
                        Attention(
                            config = config,
                            key = f"{key_prefix}.layers.{idx}.self_attn",
                            layer_idx = idx,
                            hidden_size = config.hidden_size,
                            head_dim = config.head_dim,
                            num_q_heads = config.num_q_heads,
                            num_kv_heads = config.num_kv_heads,
                            rope_settings = config.rope_settings_list[idx],
                            sm_scale = None,
                            # HF's sliding window counts the query itself
                            sliding_window = (
                                config.sliding_window - 1
                                if layer_type == "sliding_attention" and config.sliding_window > 0
                                else -1
                            ),
                            key_q = "q_proj",
                            key_k = "k_proj",
                            key_v = "v_proj",
                            key_o = "o_proj",
                            # Sigmoid gate, one value per head, applied to the attention output
                            key_g = "gate_proj",
                            qmap = "block.attn",
                            q_norm = RMSNorm(
                                config = config,
                                key = f"{key_prefix}.layers.{idx}.self_attn.q_norm",
                                rms_norm_eps = config.rms_norm_eps,
                            ),
                            k_norm = RMSNorm(
                                config = config,
                                key = f"{key_prefix}.layers.{idx}.self_attn.k_norm",
                                rms_norm_eps = config.rms_norm_eps,
                            ),
                            out_dtype = torch.float,
                        )
                    ),
                    attn_post_norm = attn_post_norm,
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
                    ),
                    mlp_post_norm = mlp_post_norm,
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


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        input_ids = prepare_for_attn(input_ids, params)
        prepare_for_recurrence(input_ids, params, self)
        return input_ids


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        # ChatML turns; the template opens the assistant turn with <think> (thinking on by default)
        p = ""
        if system_prompt:
            p += f"<|im_start|>system\n"
            p += f"{system_prompt}<|im_end|>\n"
        p += f"<|im_start|>user\n"
        p += f"{prompt}<|im_end|>\n"
        p += f"<|im_start|>assistant\n"
        p += f"<think>"
        return p
