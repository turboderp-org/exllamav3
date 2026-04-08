from __future__ import annotations

import json
import os
from types import SimpleNamespace

import torch
from typing_extensions import override

from ..model.config import Config
from ..model.model import Model
from ..modules import (
    Embedding,
    GatedMLP,
    Linear,
    RMSNorm,
    TransformerBlock,
    Gemma4Attention,
    Gemma4MoEFeedForward,
    Gemma4MoETransformerBlock,
    Gemma4TransformerBlock,
)
from ..modules.attn import prepare_for_attn
from ..util.file import no_default
from ..util.rope import RopeStyle
from .gemma4_mm import Gemma4VisionModel, set_gemma4_vision_groups


def _detect_gemma4_profile(config: "Gemma4Config") -> str:
    if (
        config.num_hidden_layers == 60 and
        not config.enable_moe_block and
        config.num_kv_heads == 16 and
        config.num_global_kv_heads == 4
    ):
        return "31b_dense"

    if (
        config.num_hidden_layers == 30 and
        config.enable_moe_block and
        config.num_kv_heads == 8 and
        config.num_global_kv_heads == 2
    ):
        return "26b_a4b_moe"

    return "generic"


def _filter_gemma4_indexed_embeddings(
    input_ids: torch.Tensor,
    params: dict,
) -> None:
    indexed_embeddings = params.get("indexed_embeddings") or []
    if not indexed_embeddings:
        params.pop("indexed_embeddings", None)
        return

    active_embeddings = [
        embedding
        for embedding in indexed_embeddings
        if ((input_ids >= embedding.first_index) & (input_ids < embedding.last_index)).any().item()
    ]

    if active_embeddings:
        params["indexed_embeddings"] = active_embeddings
    else:
        params.pop("indexed_embeddings", None)


def _update_gemma4_reconstruct_mode(
    params: dict,
    profile: str,
    has_mm_request: bool,
) -> None:
    temp_flag = "_gemma4_temp_reconstruct"

    if profile != "26b_a4b_moe":
        if params.pop(temp_flag, False):
            params.pop("reconstruct", None)
        return

    if has_mm_request:
        if "reconstruct" not in params:
            params["reconstruct"] = True
            params[temp_flag] = True
        else:
            params.pop(temp_flag, None)
    elif params.pop(temp_flag, False):
        params.pop("reconstruct", None)


class Gemma4Config(Config):
    arch_string = "Gemma4ForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Gemma4TextModel, "vision": Gemma4VisionModel},
            **kwargs
        )

        self.image_token_id = self.read_cfg(int, "image_token_id", None)
        self.boi_token_id = self.read_cfg(int, "boi_token_id", None)
        self.eoi_token_id = self.read_cfg(int, "eoi_token_id", None)
        self.vision_soft_tokens_per_image = self.read_cfg(int, "vision_soft_tokens_per_image", no_default)

        self.head_dim = self.read_cfg(int, "text_config->head_dim", no_default)
        self.global_head_dim = self.read_cfg(int, "text_config->global_head_dim", self.head_dim)
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "text_config->num_key_value_heads", self.num_q_heads)
        self.num_global_kv_heads = self.read_cfg(
            int,
            "text_config->num_global_key_value_heads",
            self.num_kv_heads
        )
        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "text_config->tie_word_embeddings", False)
        self.attention_k_eq_v = self.read_cfg(bool, "text_config->attention_k_eq_v", False)
        self.use_bidirectional_attention = self.read_cfg(str, "text_config->use_bidirectional_attention", None)

        self.layer_types = self.read_cfg(list, "text_config->layer_types", no_default)
        assert len(self.layer_types) == self.num_hidden_layers, \
            "Length of text_config->layer_types key doesn't match number of hidden layers"

        self.sliding_window = self.read_cfg(int, "text_config->sliding_window", -1)
        self.swa_pattern = []
        for layer_type in self.layer_types:
            match layer_type:
                case "sliding_attention":
                    self.swa_pattern.append(self.sliding_window)
                case "full_attention":
                    self.swa_pattern.append(-1)
                case _:
                    raise ValueError(f"Unknown layer type in layer_types: {layer_type}")

        self.assert_cfg(str, "text_config->hidden_activation", "gelu_pytorch_tanh", True)
        self.intermediate_size = self.read_cfg(int, "text_config->intermediate_size", no_default)

        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", no_default)
        self.attn_logit_softcapping = self.read_cfg(float, "text_config->attn_logit_softcapping", 0.0)
        self.final_logit_softcapping = self.read_cfg(float, "text_config->final_logit_softcapping", 0.0)

        self.hidden_size_per_layer_input = self.read_cfg(int, "text_config->hidden_size_per_layer_input", 0)
        if self.hidden_size_per_layer_input:
            raise NotImplementedError("Gemma4 per-layer inputs are not implemented yet")

        self.enable_moe_block = self.read_cfg(bool, "text_config->enable_moe_block", False)
        self.num_experts = self.read_cfg(int, "text_config->num_experts", 0)
        self.num_experts_per_tok = self.read_cfg(int, "text_config->top_k_experts", 0)
        self.moe_intermediate_size = self.read_cfg(int, "text_config->moe_intermediate_size", 0)
        if self.enable_moe_block:
            assert self.num_experts > 0, "Gemma4 MoE requires text_config->num_experts"
            assert self.num_experts_per_tok > 0, "Gemma4 MoE requires text_config->top_k_experts"
            assert self.moe_intermediate_size > 0, "Gemma4 MoE requires text_config->moe_intermediate_size"

        self.rope_settings_local = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 10000.0,
            config_dict = self.read_cfg(dict, "text_config->rope_parameters->sliding_attention", {}),
            override_type = self.read_cfg(
                str,
                "text_config->rope_parameters->sliding_attention->rope_type",
                None,
            )
        )
        self.rope_settings_full = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 1000000.0,
            config_dict = self.read_cfg(dict, "text_config->rope_parameters->full_attention", {}),
            override_type = self.read_cfg(
                str,
                "text_config->rope_parameters->full_attention->rope_type",
                None,
            )
        )
        self.rope_settings_full.head_dim = self.global_head_dim

        self.vision = SimpleNamespace()
        self.vision.hidden_size = self.read_cfg(int, "vision_config->hidden_size", no_default)
        self.vision.intermediate_size = self.read_cfg(int, "vision_config->intermediate_size", no_default)
        self.vision.num_hidden_layers = self.read_cfg(int, "vision_config->num_hidden_layers", no_default)
        self.vision.num_q_heads = self.read_cfg(int, "vision_config->num_attention_heads", no_default)
        self.vision.num_kv_heads = self.read_cfg(int, "vision_config->num_key_value_heads", self.vision.num_q_heads)
        self.vision.head_dim = self.read_cfg(int, "vision_config->head_dim", no_default)
        self.vision.patch_size = self.read_cfg(int, "vision_config->patch_size", no_default)
        self.vision.pooling_kernel_size = self.read_cfg(int, "vision_config->pooling_kernel_size", no_default)
        self.vision.position_embedding_size = self.read_cfg(int, "vision_config->position_embedding_size", no_default)
        self.vision.rms_norm_eps = self.read_cfg(float, "vision_config->rms_norm_eps", no_default)
        self.vision.standardize = self.read_cfg(bool, "vision_config->standardize", False)
        self.vision.rope_theta = self.read_cfg(float, "vision_config->rope_parameters->rope_theta", 100.0)
        self.vision.num_channels = 3
        self.vision.patch_dim = self.vision.num_channels * self.vision.patch_size ** 2

        processor_path = os.path.join(self.directory, "processor_config.json")
        with open(processor_path, encoding = "utf8") as f:
            processor_config = json.load(f)
        image_processor = processor_config["image_processor"]

        self.vision_pp = SimpleNamespace()
        self.vision_pp.do_convert_rgb = image_processor["do_convert_rgb"]
        self.vision_pp.do_rescale = image_processor["do_rescale"]
        self.vision_pp.do_normalize = image_processor["do_normalize"]
        self.vision_pp.image_mean = image_processor["image_mean"]
        self.vision_pp.image_std = image_processor["image_std"]
        self.vision_pp.resample = image_processor["resample"]
        self.vision_pp.rescale_factor = image_processor["rescale_factor"]
        self.vision_pp.max_soft_tokens = image_processor["max_soft_tokens"]
        self.vision_pp.patch_size = image_processor["patch_size"]
        self.vision_pp.pooling_kernel_size = image_processor["pooling_kernel_size"]
        self.vision.max_patches = self.vision_pp.max_soft_tokens * (self.vision_pp.pooling_kernel_size ** 2)


class Gemma4TextModel(Model):
    config_class = Gemma4Config

    def __init__(
        self,
        config: Gemma4Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        profile = _detect_gemma4_profile(config)
        is_31b_dense = profile == "31b_dense"
        self.caps.update({
            "supports_tp": is_31b_dense,
            "atomic_mm_prefill": True,
            "gemma4_profile": profile,
        })

        self.modules += [
            Embedding(
                config = config,
                key = "model.language_model.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
                multiplier = config.hidden_size ** 0.5,
            )
        ]

        self.first_block_idx = len(self.modules)

        for idx in range(config.num_hidden_layers):
            key = f"model.language_model.layers.{idx}"
            layer_is_full = config.layer_types[idx] == "full_attention"

            attn = Gemma4Attention(
                config = config,
                key = f"{key}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                head_dim = config.global_head_dim if layer_is_full else config.head_dim,
                num_q_heads = config.num_q_heads,
                num_kv_heads = (
                    config.num_global_kv_heads
                    if layer_is_full and config.attention_k_eq_v
                    else config.num_kv_heads
                ),
                use_k_as_v = layer_is_full and config.attention_k_eq_v,
                v_norm = RMSNorm(
                    config = config,
                    key = f"{key}.self_attn.v_norm",
                    rms_norm_eps = config.rms_norm_eps,
                    unweighted = True,
                ),
                rope_settings = config.rope_settings_full if layer_is_full else config.rope_settings_local,
                sm_scale = 1.0,
                sliding_window = config.swa_pattern[idx],
                key_q = "q_proj",
                key_k = "k_proj",
                key_v = "v_proj",
                key_o = "o_proj",
                qmap = "block.attn",
                logit_softcapping = config.attn_logit_softcapping,
                q_norm = RMSNorm(
                    config = config,
                    key = f"{key}.self_attn.q_norm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                k_norm = RMSNorm(
                    config = config,
                    key = f"{key}.self_attn.k_norm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
            )

            common_kwargs = dict(
                config = config,
                key = key,
                layer_idx = idx,
                attn_norm = RMSNorm(
                    config = config,
                    key = f"{key}.input_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                attn = attn,
                attn_post_norm = RMSNorm(
                    config = config,
                    key = f"{key}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    out_dtype = torch.float,
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"{key}.pre_feedforward_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
                mlp_post_norm = RMSNorm(
                    config = config,
                    key = f"{key}.post_feedforward_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                    out_dtype = torch.float,
                ),
            )

            if config.enable_moe_block:
                mlp = Gemma4MoEFeedForward(
                    config = config,
                    key = key,
                    hidden_size = config.hidden_size,
                    intermediate_size = config.intermediate_size,
                    moe_intermediate_size = config.moe_intermediate_size,
                    num_experts = config.num_experts,
                    num_experts_per_tok = config.num_experts_per_tok,
                    rms_norm_eps = config.rms_norm_eps,
                )
                block = Gemma4MoETransformerBlock(
                    mlp = mlp,
                    **common_kwargs,
                )
            else:
                block = Gemma4TransformerBlock(
                    mlp = GatedMLP(
                        config = config,
                        key = f"{key}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.intermediate_size,
                        key_up = "up_proj",
                        key_gate = "gate_proj",
                        key_down = "down_proj",
                        qmap = "block.mlp",
                        activation_fn = "gelu",
                    ),
                    **common_kwargs,
                )

            self.modules.append(block)

        self.last_kv_module_idx = len(self.modules) - 1

        self.modules += [
            RMSNorm(
                config = config,
                key = "model.language_model.norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = "lm_head",
                qbits_key = "head_bits",
                alt_key = "model.language_model.embed_tokens" if config.tie_word_embeddings else None,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                softcap = config.final_logit_softcapping,
                caps = {"logits_output": True},
            )
        ]

        self.logit_layer_idx = len(self.modules) - 1

        if config.enable_moe_block:
            self.calibration_all_experts = True


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        has_mm_request = bool(params.get("indexed_embeddings"))
        _filter_gemma4_indexed_embeddings(input_ids, params)
        _update_gemma4_reconstruct_mode(
            params,
            self.caps.get("gemma4_profile", "generic"),
            has_mm_request = has_mm_request,
        )
        set_gemma4_vision_groups(input_ids, params, self.config.boi_token_id, self.config.eoi_token_id)
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = "<bos>"
        if system_prompt:
            p += f"<|turn>system\n{system_prompt}<turn|>\n"
        p += f"<|turn>user\n{prompt}<turn|>\n"
        p += "<|turn>model\n"
        return p
