from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
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
    Gemma4QuantCacheLayer,
    Gemma4SingleQuantCacheLayer,
    Gemma4TransformerBlock,
    Gemma4VisionAttention,
    Gemma4VisionPatchEmbedder,
    Gemma4VisionPooler,
    Gemma4VisionProjector,
    Gemma4VisionStandardize,
)
from ..modules.attn import prepare_for_attn
from ..tokenizer import MMEmbedding, Tokenizer
from ..util.file import no_default
from ..util.rope import RopeStyle
from ..util.vision import convert_to_rgb


def _set_gemma4_vision_groups(
    input_ids: torch.Tensor,
    params: dict,
    boi_token_id: int | None,
    eoi_token_id: int | None,
) -> None:
    indexed_embeddings = params.get("indexed_embeddings") or []
    if not indexed_embeddings:
        params.pop("vision_group_ids", None)
        return

    vision_group_ids = torch.full(input_ids.shape, -1, dtype = torch.int32)
    group_index = 0
    for embedding in indexed_embeddings:
        mask = (input_ids >= embedding.first_index) & (input_ids < embedding.last_index)
        if not mask.any():
            continue

        for row in range(input_ids.shape[0]):
            row_mask = mask[row]
            if not row_mask.any():
                continue

            positions = torch.nonzero(row_mask, as_tuple = False).flatten()
            start = int(positions[0].item())
            end = int(positions[-1].item())

            vision_group_ids[row, start : end + 1] = group_index

        group_index += 1

    if group_index > 0:
        params["vision_group_ids"] = vision_group_ids
    else:
        params.pop("vision_group_ids", None)


def _get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
    total_px = height * width
    target_px = max_patches * (patch_size ** 2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to a 0 x 0 image. "
            f"Resized height should be divisible by pooling_kernel_size * patch_size = {side_mult}."
        )

    max_side_length = (max_patches // (pooling_kernel_size ** 2)) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)

    if target_height * target_width > target_px:
        raise ValueError(
            f"Resizing [{height}x{width}] to [{target_height}x{target_width}] exceeds patch budget"
        )

    return target_height, target_width


def _convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    num_channels, image_height, image_width = image.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = image.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patched_image = patched_image.transpose(1, 3, 2, 4, 0)
    patched_image = patched_image.reshape(num_patches_height * num_patches_width, -1)
    return patched_image


def _pad_patches_and_positions(
    patches: np.ndarray,
    positions: np.ndarray,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    current_length = patches.shape[0]
    if current_length > target_length:
        raise ValueError(
            f"Cannot pad Gemma4 patches from {current_length} down to target length {target_length}"
        )

    padding_length = target_length - current_length
    if padding_length == 0:
        return patches, positions

    patch_paddings = [(0, padding_length)] + [(0, 0)] * (patches.ndim - 1)
    position_paddings = [(0, padding_length), (0, 0)]
    patches = np.pad(patches, patch_paddings, mode = "constant", constant_values = 0)
    positions = np.pad(positions, position_paddings, mode = "constant", constant_values = -1)
    return patches, positions


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
        is_31b_dense = (
            config.num_hidden_layers == 60 and
            not config.enable_moe_block and
            config.num_kv_heads == 16 and
            config.num_global_kv_heads == 4
        )
        is_26b_a4b_moe = (
            config.num_hidden_layers == 30 and
            config.enable_moe_block and
            config.num_kv_heads == 8 and
            config.num_global_kv_heads == 2
        )
        use_single_quant_kv_cache = is_31b_dense or is_26b_a4b_moe
        self.caps.update({
            "supports_tp": False,
            "atomic_mm_prefill": True,
            "quantized_kv_cache_layer": Gemma4SingleQuantCacheLayer if use_single_quant_kv_cache else Gemma4QuantCacheLayer,
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
                force_quantized_fallback = is_31b_dense and not layer_is_full,
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
        _set_gemma4_vision_groups(input_ids, params, self.config.boi_token_id, self.config.eoi_token_id)
        return prepare_for_attn(input_ids, params)


    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = "<bos>"
        if system_prompt:
            p += f"<|turn>system\n{system_prompt}<turn|>\n"
        p += f"<|turn>user\n{prompt}<turn|>\n"
        p += "<|turn>model\n"
        return p


class Gemma4VisionModel(Model):

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: Gemma4Config) -> dict:
        return (
            config.stc.list_tensors(prefix = "model.vision_tower") |
            config.stc.list_tensors(prefix = "model.embed_vision")
        )


    def __init__(
        self,
        config: Gemma4Config,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.caps.update({
            "image_input": True,
            "supports_tp": False,
        })
        v = self.config.vision

        self.modules += [
            Gemma4VisionPatchEmbedder(
                config = config,
                key = "model.vision_tower.patch_embedder",
                hidden_size = v.hidden_size,
                patch_dim = v.patch_dim,
            )
        ]

        for idx in range(v.num_hidden_layers):
            key = f"model.vision_tower.encoder.layers.{idx}"
            self.modules.append(
                TransformerBlock(
                    config = config,
                    key = key,
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key}.input_layernorm",
                        rms_norm_eps = v.rms_norm_eps,
                    ),
                    attn = Gemma4VisionAttention(
                        config = config,
                        key = f"{key}.self_attn",
                        layer_idx = idx,
                        hidden_size = v.hidden_size,
                        head_dim = v.head_dim,
                        num_q_heads = v.num_q_heads,
                        num_kv_heads = v.num_kv_heads,
                        rope_theta = v.rope_theta,
                        rms_norm_eps = v.rms_norm_eps,
                    ),
                    attn_post_norm = RMSNorm(
                        config = config,
                        key = f"{key}.post_attention_layernorm",
                        rms_norm_eps = v.rms_norm_eps,
                        out_dtype = torch.float,
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key}.pre_feedforward_layernorm",
                        rms_norm_eps = v.rms_norm_eps,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"{key}.mlp",
                        hidden_size = v.hidden_size,
                        intermediate_size = v.intermediate_size,
                        key_up = "up_proj.linear",
                        key_gate = "gate_proj.linear",
                        key_down = "down_proj.linear",
                        activation_fn = "gelu",
                        qmap = "block.mlp",
                    ),
                    mlp_post_norm = RMSNorm(
                        config = config,
                        key = f"{key}.post_feedforward_layernorm",
                        rms_norm_eps = v.rms_norm_eps,
                        out_dtype = torch.float,
                    ),
                )
            )

        self.modules += [
            Gemma4VisionPooler(
                config = config,
                key = "model.vision_tower.pooler",
                hidden_size = v.hidden_size,
            )
        ]

        if v.standardize:
            self.modules += [
                Gemma4VisionStandardize(
                    config = config,
                    key = "model.vision_tower",
                )
            ]

        self.modules += [
            Gemma4VisionProjector(
                config = config,
                key = "model.embed_vision.embedding_projection",
                in_features = v.hidden_size,
                out_features = config.hidden_size,
                rms_norm_eps = v.rms_norm_eps,
            ),
        ]


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return input_ids


    def default_load_shape_dtype(self, chunk_size):
        return (
            1,
            self.config.vision.max_patches,
            self.config.vision.patch_dim,
        ), torch.half


    def default_load_params(self, chunk_size):
        h_patches = 45
        w_patches = self.config.vision.max_patches // h_patches
        grid_x, grid_y = np.meshgrid(np.arange(w_patches), np.arange(h_patches), indexing = "xy")
        position_ids = np.stack([grid_x, grid_y], axis = -1).reshape(self.config.vision.max_patches, 2)
        return {
            "input_ids": torch.zeros((1, self.config.vision.max_patches, self.config.vision.patch_dim), dtype = torch.half),
            "image_position_ids": torch.from_numpy(position_ids).unsqueeze(0).to(torch.long),
            "image_output_length": self.config.vision_pp.max_soft_tokens,
            "causal": False,
        }


    def preprocess(
        self,
        image: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor, int, tuple[int, int]]:
        vpp = self.config.vision_pp
        image = convert_to_rgb(image)
        old_size = image.size
        width, height = old_size
        target_height, target_width = _get_aspect_ratio_preserving_size(
            height = height,
            width = width,
            patch_size = vpp.patch_size,
            max_patches = self.config.vision.max_patches,
            pooling_kernel_size = vpp.pooling_kernel_size,
        )

        if (target_width, target_height) != old_size:
            image = image.resize((target_width, target_height), resample = Image.Resampling(vpp.resample))

        image_np = np.array(image).astype(np.float32)
        image_np = image_np.transpose(2, 0, 1)

        if vpp.do_rescale:
            image_np *= vpp.rescale_factor

        if vpp.do_normalize:
            image_mean = np.asarray(vpp.image_mean, dtype = np.float32).reshape(-1, 1, 1)
            image_std = np.asarray(vpp.image_std, dtype = np.float32).reshape(-1, 1, 1)
            image_np = (image_np - image_mean) / image_std

        patches = _convert_image_to_patches(image_np, vpp.patch_size)
        num_soft_tokens = patches.shape[0] // (vpp.pooling_kernel_size ** 2)

        patch_height = image_np.shape[-2] // vpp.patch_size
        patch_width = image_np.shape[-1] // vpp.patch_size
        grid_x, grid_y = np.meshgrid(np.arange(patch_width), np.arange(patch_height), indexing = "xy")
        positions = np.stack([grid_x, grid_y], axis = -1).reshape(patches.shape[0], 2)

        patches, positions = _pad_patches_and_positions(
            patches,
            positions,
            self.config.vision.max_patches,
        )

        pixel_values = torch.from_numpy(patches).half().unsqueeze(0)
        image_position_ids = torch.from_numpy(positions).to(torch.long).unsqueeze(0)
        return pixel_values, image_position_ids, num_soft_tokens, (target_width, target_height)


    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image.Image | list[Image.Image],
        text_alias: str | None = None,
    ):
        if isinstance(image, list):
            assert text_alias is None, "Cannot apply a single alias to a list of images"
            return [self.get_image_embeddings(tokenizer, i) for i in image]

        pixel_values, image_position_ids, _, prep_size = self.preprocess(image)
        params = {
            "causal": False,
            "image_position_ids": image_position_ids,
            "image_output_length": self.config.vision_pp.max_soft_tokens,
        }
        embedding_tensor = self.forward(
            pixel_values,
            params = params,
        ).cpu()
        pooler_mask = params.get("image_pooler_mask")
        if pooler_mask is not None:
            pooler_mask = pooler_mask.cpu()
            embedding_tensor = embedding_tensor[0][pooler_mask[0]].unsqueeze(0)
            num_soft_tokens = int(pooler_mask[0].sum().item())
        else:
            num_soft_tokens = embedding_tensor.shape[1]

        boi_token_id = tokenizer.single_id("<|image>")
        image_token_id = tokenizer.single_id("<|image|>")
        eoi_token_id = tokenizer.single_id("<image|>")
        token_string = torch.tensor(
            [[boi_token_id] + [image_token_id] * num_soft_tokens + [eoi_token_id]],
            dtype = torch.long,
        )
        token_string[:, 1:-1] = -1

        mme = MMEmbedding(
            embeddings = embedding_tensor[0],
            text_alias = text_alias,
            token_string = token_string,
        )
        mme.metadata.update({
            "original_size": image.size,
            "preprocessed_size": prep_size,
            "model_architecture": self.config.architecture,
        })
        return mme
