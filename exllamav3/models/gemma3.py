from __future__ import annotations
from typing_extensions import override
import torch
from .config import Config, no_default
from .model import Model
from ..util.rope import RopeSettings, RopeStyle
from ..modules import RMSNorm, Embedding, TransformerBlock, Attention, GatedMLP, Linear
from ..modules.attn import prepare_for_attn

class Gemma3Config(Config):
    arch_string = "Gemma3ForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Gemma3Model, "vision": Gemma3VisionModel},
            **kwargs
        )

        # Gemma3 quirk, vocab size is implicit on HF versions
        if self.vocab_size is None:
            self.vocab_size = 262208

        # Attention params
        self.head_dim = self.read_cfg(int, "text_config->head_dim", 256)
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", 2304)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", 8)
        self.num_kv_heads = self.read_cfg(int, "text_config->num_key_value_heads", 4)

        self.query_pre_attn_scalar = self.read_cfg(float, "text_config->query_pre_attn_scalar", 256)
        self.attn_logit_softcapping = self.read_cfg(float, "text_config->attn_logit_softcapping", 0.0)
        self.sliding_window = self.read_cfg(int, "text_config->sliding_window", 4096)
        self.sliding_window_pattern = self.read_cfg(int, "text_config->sliding_window_pattern", 6)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.intermediate_size = self.read_cfg(int, "text_config->intermediate_size", 9216)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", 1e-6)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "text_config->tie_word_embeddings", True)

        # RoPE
        self.rope_settings_global = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 1e6,
            config_dict = self.read_cfg(dict, "text_config", no_default)
        )
        self.rope_settings_local = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 1e4,
            config_dict = {}
        )

        # Output softcap
        self.final_logit_softcapping = self.read_cfg(float, "text_config->final_logit_softcapping", 0.0)


class Gemma3Model(Model):
    config_class = Gemma3Config

    def __init__(
        self,
        config: Gemma3Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)

        self.modules += [
            Embedding(
                config = config,
                key = "language_model.model.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
                normalize = True,
            )
        ]

        self.first_block_idx = len(self.modules)

        is_local = [
            bool((idx + 1) % config.sliding_window_pattern)
            for idx in range(config.num_hidden_layers)
        ]

        self.modules += [
            TransformerBlock(
                config = config,
                key = f"language_model.model.layers.{idx}",
                attn_norm = RMSNorm(
                    config = config,
                    key = f"language_model.model.layers.{idx}.input_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                attn = Attention(
                    config = config,
                    key = f"language_model.model.layers.{idx}.self_attn",
                    layer_idx = idx,
                    hidden_size = config.hidden_size,
                    head_dim = config.head_dim,
                    num_q_heads = config.num_q_heads,
                    num_kv_heads = config.num_kv_heads,
                    rope_settings = config.rope_settings_local if is_local[idx] else config.rope_settings_global,
                    sm_scale = config.query_pre_attn_scalar ** (-0.5),
                    logit_softcapping = config.attn_logit_softcapping,
                    sliding_window = config.sliding_window if is_local[idx] else -1,
                    key_q = "q_proj",
                    key_k = "k_proj",
                    key_v = "v_proj",
                    key_o = "o_proj",
                    qmap = "block.attn",
                    q_norm = RMSNorm(
                        config = config,
                        key = f"language_model.model.layers.{idx}.self_attn.q_norm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                    k_norm = RMSNorm(
                        config = config,
                        key = f"language_model.model.layers.{idx}.self_attn.k_norm",
                        rms_norm_eps = config.rms_norm_eps,
                        constant_bias = 1.0,
                    ),
                ),
                attn_post_norm = RMSNorm(
                    config = config,
                    key = f"language_model.model.layers.{idx}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                    out_dtype = torch.float,
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"language_model.model.layers.{idx}.pre_feedforward_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    constant_bias = 1.0,
                ),
                mlp = GatedMLP(
                    config = config,
                    key = f"language_model.model.layers.{idx}.mlp",
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
                    key = f"language_model.model.layers.{idx}.post_feedforward_layernorm",
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
                key = "language_model.model.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
                constant_bias = 1.0,
            ),
            Linear(
                config = config,
                key = "language_model.lm_head",
                qbits_key = "head_bits",
                alt_key = "language_model.model.embed_tokens",
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


class Gemma3VisionModel(Model):
    def __init__(
        self,
        config: Gemma3Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        # TODO

    def load(self):
        raise NotImplementedError()  # TODO

    def unload(self):
        raise NotImplementedError()  # TODO

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: Gemma3Config) -> dict:
        vlm_tensors = config.stc.list_tensors(prefix = "vision_tower")
        mmp_tensors = config.stc.list_tensors(prefix = "multi_modal_projector")
        return vlm_tensors | mmp_tensors