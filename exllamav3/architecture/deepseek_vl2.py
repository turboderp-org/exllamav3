from __future__ import annotations

from typing_extensions import override
from types import SimpleNamespace
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import (
    Attention,
    BlockSparseMLP,
    Embedding,
    GatedMLP,
    Linear,
    RMSNorm,
    TransformerBlock,
)
from ..modules.attn import prepare_for_attn
from ..tokenizer import MMEmbedding, Tokenizer
from ..util.file import read_dict
from ..util.rope import RopeStyle
from ..util.vision import convert_to_rgb, normalize_image


class DeepseekVLV2Config(Config):
    arch_string = "DeepseekVLV2ForCausalLM"
    model_type_string = "deepseek_vl_v2"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": DeepseekVLV2TextModel, "vision": DeepseekVLV2VisionModel},
            **kwargs,
        )

        # Text backbone (DeepSeek-MoE style, standard q/k/v attention)
        self.hidden_size = self.read_cfg(int, "language_config->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "language_config->num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "language_config->num_key_value_heads", self.num_q_heads)
        self.head_dim = self.hidden_size // self.num_q_heads
        self.max_position_embeddings = self.read_cfg(int, "language_config->max_position_embeddings", 4096)

        self.assert_cfg(str, "language_config->hidden_act", "silu", optional = True)
        self.intermediate_size = self.read_cfg(int, "language_config->intermediate_size", no_default)
        self.first_k_dense_replace = self.read_cfg(int, "language_config->first_k_dense_replace", 0)
        self.moe_intermediate_size = self.read_cfg(int, "language_config->moe_intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, "language_config->n_routed_experts", no_default)
        self.num_shared_experts = self.read_cfg(int, "language_config->n_shared_experts", 0)
        self.num_experts_per_tok = self.read_cfg(int, "language_config->num_experts_per_tok", no_default)
        self.routed_scaling_factor = self.read_cfg(float, "language_config->routed_scaling_factor", 1.0)
        self.n_group = self.read_cfg(int, "language_config->n_group", 1)
        self.topk_group = self.read_cfg(int, "language_config->topk_group", 1)
        self.topk_method = self.read_cfg(str, "language_config->topk_method", "greedy")
        self.norm_topk_prob = self.read_cfg(bool, "language_config->norm_topk_prob", False)

        self.rms_norm_eps = self.read_cfg(float, "language_config->rms_norm_eps", 1e-6)
        self.num_hidden_layers = self.read_cfg(int, "language_config->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "language_config->tie_word_embeddings", False)

        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            config_dict = self.read_cfg(dict, "language_config", no_default),
        )

        # Vision tower + projector
        read_vision_config = self.read_cfg(dict, "vision_config", no_default)
        self.vision = read_deepseek_vl2_vision_config(read_vision_config)

        read_projector_config = self.read_cfg(dict, "projector_config", {})
        self.projector = read_deepseek_vl2_projector_config(
            read_projector_config,
            self.vision.width,
            self.hidden_size,
        )

        proc_path = os.path.join(self.directory, "processor_config.json")
        with open(proc_path, encoding = "utf8") as f:
            read_proc_config = json.load(f)
        self.vision_pp = read_deepseek_vl2_processor_config(read_proc_config)

        self.tile_tag = self.read_cfg(str, "tile_tag", "2D")
        self.global_view_pos = self.read_cfg(str, "global_view_pos", "head")
        self.candidate_resolutions = [
            tuple(x) for x in self.read_cfg(list, "candidate_resolutions", [(self.vision.image_size, self.vision.image_size)])
        ]


def read_deepseek_vl2_vision_config(config_dict: dict):
    v = SimpleNamespace()
    raw_model_name = read_dict(config_dict, str, "model_name", "siglip_so400m_patch14_384")
    model_name_map = {
        "siglip_so400m_patch14_384": "vit_so400m_patch14_siglip_384.webli",
        "vit_so400m_patch14_siglip_384": "vit_so400m_patch14_siglip_384.webli",
    }
    v.model_name = model_name_map.get(raw_model_name, raw_model_name)
    v.model_type = read_dict(config_dict, str, "model_type", "vision")
    v.patch_size = read_dict(config_dict, int, "patch_size", 14)
    v.width = read_dict(config_dict, int, "width", 1152)
    v.layers = read_dict(config_dict, int, "layers", 27)
    v.mlp_ratio = read_dict(config_dict, float, "mlp_ratio", 3.7362)
    v.image_size = 384
    return v


def read_deepseek_vl2_projector_config(config_dict: dict, input_dim: int, n_embed: int):
    p = SimpleNamespace()
    p.projector_type = read_dict(config_dict, str, "projector_type", "downsample_mlp_gelu")
    p.input_dim = read_dict(config_dict, int, "input_dim", input_dim)
    p.n_embed = read_dict(config_dict, int, "n_embed", n_embed)
    p.depth = read_dict(config_dict, int, "depth", 2)
    p.mlp_ratio = read_dict(config_dict, int, "mlp_ratio", 1)
    p.downsample_ratio = read_dict(config_dict, int, "downsample_ratio", 2)
    return p


def read_deepseek_vl2_processor_config(config_dict: dict):
    pp = SimpleNamespace()
    pp.candidate_resolutions = [tuple(x) for x in read_dict(config_dict, list, "candidate_resolutions", [(384, 384)])]
    pp.downsample_ratio = read_dict(config_dict, int, "downsample_ratio", 2)
    pp.image_mean = tuple(read_dict(config_dict, list, "image_mean", [0.5, 0.5, 0.5]))
    pp.image_std = tuple(read_dict(config_dict, list, "image_std", [0.5, 0.5, 0.5]))
    pp.normalize = read_dict(config_dict, bool, "normalize", True)
    pp.image_token = read_dict(config_dict, str, "image_token", "<image>")
    pp.pad_token = read_dict(config_dict, str, "pad_token", "<｜▁pad▁｜>")
    pp.patch_size = read_dict(config_dict, int, "patch_size", 14)
    pp.processor_class = read_dict(config_dict, str, "processor_class", "DeepseekVLV2Processor")
    pp.add_special_token = read_dict(config_dict, bool, "add_special_token", False)
    pp.sft_format = read_dict(config_dict, str, "sft_format", "deepseek")
    pp.mask_prompt = read_dict(config_dict, bool, "mask_prompt", False)
    pp.ignore_id = read_dict(config_dict, int, "ignore_id", -100)
    return pp


class DeepseekVLV2TextModel(Model):
    config_class = DeepseekVLV2Config

    def __init__(
        self,
        config: DeepseekVLV2Config,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        key_prefix = "language.model"

        self.modules += [
            Embedding(
                config = config,
                key = f"{key_prefix}.embed_tokens",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            )
        ]

        self.first_block_idx = len(self.modules)

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
                    key_q = "q_proj",
                    key_k = "k_proj",
                    key_v = "v_proj",
                    key_o = "o_proj",
                    qmap = "block.attn",
                    out_dtype = torch.float,
                ),
                mlp_norm = RMSNorm(
                    config = config,
                    key = f"{key_prefix}.layers.{idx}.post_attention_layernorm",
                    rms_norm_eps = config.rms_norm_eps,
                ),
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
                    if idx < config.first_k_dense_replace else
                    BlockSparseMLP(
                        config = config,
                        key = f"{key_prefix}.layers.{idx}.mlp",
                        hidden_size = config.hidden_size,
                        intermediate_size = config.moe_intermediate_size,
                        num_experts = config.num_experts,
                        num_experts_per_tok = config.num_experts_per_tok,
                        key_up = "experts.{expert_idx}.up_proj",
                        key_gate = "experts.{expert_idx}.gate_proj",
                        key_down = "experts.{expert_idx}.down_proj",
                        key_gate_up_split = "experts.gate_up_proj",
                        key_down_split = "experts.down_proj",
                        key_routing_gate = "gate",
                        qmap = "block.mlp",
                        interm_dtype = torch.half,
                        out_dtype = torch.float,
                        router_type = "deepseek_v2",
                        routed_scaling_factor = config.routed_scaling_factor,
                        n_group = config.n_group,
                        topk_group = config.topk_group,
                        topk_method = config.topk_method,
                        norm_topk_prob = config.norm_topk_prob,
                        shared_experts = (
                            GatedMLP(
                                config = config,
                                key = f"{key_prefix}.layers.{idx}.mlp.shared_experts",
                                hidden_size = config.hidden_size,
                                intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
                                key_up = "up_proj",
                                key_gate = "gate_proj",
                                key_down = "down_proj",
                                qmap = "block.mlp",
                                interm_dtype = torch.half,
                                out_dtype = torch.float,
                            ) if config.num_shared_experts > 0 else None
                        ),
                    )
                ),
            )
            for idx in range(config.num_hidden_layers)
        ]

        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not self.config.stc.has_tensor("language.lm_head"):
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
                key = "language.lm_head",
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True},
            ),
        ]

        self.logit_layer_idx = len(self.modules) - 1
        self.calibration_all_experts = True
        self.caps.update({"supports_tp": False})

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return prepare_for_attn(input_ids, params)

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"<|User|>{system_prompt}\n"
        p += f"<|User|>{prompt}<|Assistant|>"
        return p


class DeepseekVL2Projector(nn.Module):

    def __init__(self, config: DeepseekVLV2Config):
        super().__init__()
        mlp_hidden = config.projector.n_embed * config.projector.mlp_ratio
        self.downsample_ratio = config.projector.downsample_ratio
        self.layers = nn.Sequential(
            nn.Linear(config.projector.input_dim * self.downsample_ratio * self.downsample_ratio, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, config.projector.n_embed),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, hw, input_dim = x.shape
        h = w = int(hw ** 0.5)

        pad = 0
        if h % self.downsample_ratio:
            pad = self.downsample_ratio - h % self.downsample_ratio

        x = x.reshape(bs, h, w, input_dim)
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)
        x = x.permute(0, 3, 1, 2)
        x = F.unfold(
            x,
            kernel_size = self.downsample_ratio,
            stride = self.downsample_ratio,
            padding = 0,
        )
        x = x.permute(0, 2, 1)
        return self.layers(x)


class DeepseekVLV2VisionModel:

    @staticmethod
    def get_additional_compiled_tensors(config: DeepseekVLV2Config) -> dict:
        vision_tensors = config.stc.list_tensors(prefix = "vision")
        projector_tensors = config.stc.list_tensors(prefix = "projector")
        newline_tensors = {
            "image_newline": {"n_bytes": config.stc.get_tensor_size("image_newline")}
        }
        separator_tensors = {
            "view_seperator": {"n_bytes": config.stc.get_tensor_size("view_seperator")}
        }
        return vision_tensors | projector_tensors | newline_tensors | separator_tensors

    def __init__(
        self,
        config: DeepseekVLV2Config,
        **kwargs,
    ):
        self.config = config
        self.device = None
        self.vision = None
        self.projector = None
        self.image_newline = None
        self.view_seperator = None

    def _iter_prefix_keys(self, prefix: str) -> list[str]:
        tensors = self.config.stc.list_tensors(prefix, only_serializable = True)
        return sorted(tensors.keys())

    def _load_prefixed_state_dict(
        self,
        prefix: str,
        strip_prefix: str,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        state = {}
        for key in self._iter_prefix_keys(prefix):
            short_key = key[len(strip_prefix):]
            state[short_key] = self.config.stc.get_tensor(key, device = device, float2half = True)
        return state

    def load_gen(
        self,
        reserve_per_device = None,
        callback = None,
        **kwargs,
    ):
        import timm

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        yield (0, 1)

        with torch.inference_mode():
            self.vision = timm.create_model(
                self.config.vision.model_name,
                pretrained = False,
                num_classes = 0,
                dynamic_img_size = True,
                dynamic_img_pad = True,
            )
            vision_state = self._load_prefixed_state_dict("vision", "vision.", "cpu")
            missing, unexpected = self.vision.load_state_dict(vision_state, strict = False)
            if unexpected:
                raise ValueError(f"Unexpected DeepSeek-VL2 vision keys: {unexpected}")
            if missing:
                raise ValueError(f"Missing DeepSeek-VL2 vision keys: {missing}")
            self.vision = self.vision.to(self.device, dtype = torch.float16).eval()

            self.projector = DeepseekVL2Projector(self.config)
            projector_state = self._load_prefixed_state_dict("projector", "projector.", "cpu")
            missing, unexpected = self.projector.load_state_dict(projector_state, strict = False)
            if unexpected:
                raise ValueError(f"Unexpected DeepSeek-VL2 projector keys: {unexpected}")
            if missing:
                raise ValueError(f"Missing DeepSeek-VL2 projector keys: {missing}")
            self.projector = self.projector.to(self.device, dtype = torch.float16).eval()

            self.image_newline = self.config.stc.get_tensor("image_newline", device = self.device, float2half = True)
            self.view_seperator = self.config.stc.get_tensor("view_seperator", device = self.device, float2half = True)

        if callback:
            callback(1, 1)
        yield (1, 1)

    def unload(self):
        self.vision = None
        self.projector = None
        self.image_newline = None
        self.view_seperator = None
        self.device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = convert_to_rgb(image)
        image = np.array(image).astype(np.float32)
        if self.config.vision_pp.normalize:
            image = image / 255.0
            image = normalize_image(
                image,
                self.config.vision_pp.image_mean,
                self.config.vision_pp.image_std,
            )
        else:
            image = image / 255.0
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image).half()

    def _select_best_resolution(self, image_size: tuple[int, int]) -> tuple[int, int]:
        original_width, original_height = image_size
        best_fit = None
        max_effective_resolution = 0
        min_wasted_resolution = float("inf")

        for width, height in self.config.candidate_resolutions:
            scale = min(width / original_width, height / original_height)
            downscaled_width = int(original_width * scale)
            downscaled_height = int(original_height * scale)
            effective_resolution = min(
                downscaled_width * downscaled_height,
                original_width * original_height,
            )
            wasted_resolution = (width * height) - effective_resolution

            if effective_resolution > max_effective_resolution or (
                effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
            ):
                max_effective_resolution = effective_resolution
                min_wasted_resolution = wasted_resolution
                best_fit = (width, height)

        return best_fit or (self.config.vision.image_size, self.config.vision.image_size)

    def _preprocess_images(self, images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        tiles = []
        crops = []
        image_size = self.config.vision.image_size
        fill_color = tuple(int(x * 255) for x in self.config.vision_pp.image_mean)
        crop_anyres = len(images) <= 2

        for image in images:
            if crop_anyres:
                best_width, best_height = self._select_best_resolution(image.size)
            else:
                best_width, best_height = image_size, image_size

            global_view = ImageOps.pad(image, (image_size, image_size), color = fill_color)
            tiles.append(self._image_to_tensor(global_view))

            local_view = ImageOps.pad(image, (best_width, best_height), color = fill_color)
            for top in range(0, best_height, image_size):
                for left in range(0, best_width, image_size):
                    tile = local_view.crop((left, top, left + image_size, top + image_size))
                    tiles.append(self._image_to_tensor(tile))

            crops.append([best_width // image_size, best_height // image_size])

        pixel_values = torch.stack(tiles, dim = 0).to(self.device)
        images_spatial_crop = torch.tensor(crops, dtype = torch.long, device = self.device)
        return pixel_values, images_spatial_crop

    def _pixel_values_to_embedding(
        self,
        pixel_values: torch.Tensor,
        images_spatial_crop: torch.Tensor,
    ) -> list[torch.Tensor]:
        with torch.inference_mode():
            image_features = self.vision.forward_features(pixel_values)
            image_embeds = self.projector(image_features)

        _, hw, n_dim = image_embeds.shape
        h = w = int(hw ** 0.5)
        tile_index = 0
        outputs = []

        for crop in images_spatial_crop:
            num_width_tiles, num_height_tiles = int(crop[0].item()), int(crop[1].item())
            num_tiles_in_image = num_width_tiles * num_height_tiles

            global_features = image_embeds[tile_index]
            local_features = image_embeds[tile_index + 1: tile_index + 1 + num_tiles_in_image]
            tile_index += num_tiles_in_image + 1

            global_features = global_features.view(h, w, n_dim)
            new_lines_in_global = self.image_newline.view(1, 1, -1).expand(h, 1, -1)
            global_features = torch.cat([global_features, new_lines_in_global], dim = 1).reshape(-1, n_dim)

            local_features = local_features.view(num_height_tiles, num_width_tiles, h, w, n_dim)
            local_features = local_features.permute(0, 2, 1, 3, 4).reshape(num_height_tiles * h, num_width_tiles * w, n_dim)
            new_lines_in_local = self.image_newline.view(1, 1, -1).expand(num_height_tiles * h, 1, -1)
            local_features = torch.cat([local_features, new_lines_in_local], dim = 1).reshape(-1, n_dim)

            if self.config.global_view_pos == "head":
                merged = torch.cat([global_features, self.view_seperator[None, :], local_features], dim = 0)
            else:
                merged = torch.cat([local_features, self.view_seperator[None, :], global_features], dim = 0)

            outputs.append(merged)

        return outputs

    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image.Image | list[Image.Image],
        text_alias: str | None = None,
    ):
        assert self.vision is not None and self.projector is not None, "DeepSeek-VL2 vision model is not loaded"

        if isinstance(image, list):
            images = image
            return_batch = True
        else:
            images = [image]
            return_batch = False

        pixel_values, images_spatial_crop = self._preprocess_images(images)
        embedding_tensors = self._pixel_values_to_embedding(pixel_values, images_spatial_crop)

        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        mmes = []
        for idx, emb in enumerate(embedding_tensors):
            emb_cpu = emb.cpu()
            token_string = torch.full((1, emb_cpu.shape[0]), -1, dtype = torch.long)
            mme = MMEmbedding(
                embeddings = emb_cpu,
                token_string = token_string,
                text_alias = None if text_alias is None else text_alias,
            )
            mme.metadata.update({
                "original_size": images[idx].size,
                "model_architecture": self.config.architecture,
                "preprocessed_tiles": images_spatial_crop[idx].tolist(),
            })
            mmes.append(mme)

        return mmes if return_batch else mmes[0]
