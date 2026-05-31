from __future__ import annotations
from typing_extensions import override
from types import SimpleNamespace
import numpy as np
import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..util.rope import RopeSettings, RopeStyle
from ..util.file import read_dict
from ..modules import Attention, Linear, LayerNorm, MLP, TransformerBlock, Conv
from ..modules.arch_specific.step3_7 import Step3_7Downsampler, Step3_7PosEmbedding
from ..tokenizer import Tokenizer, MMEmbedding
from PIL import Image
from .step3_5 import Step3_5Model
from .mm_processing.step3_7 import get_patches, get_vision_position_ids, image_to_tensor, step3_7_position_embedding_grid_2d

class Step3_7Config(Config):
    arch_string = "Step3p7ForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Step3_7Model, "vision": Step3_7VisionModel},
            **kwargs
        )

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "text_config->tie_word_embeddings", False)

        # Attention params
        self.head_dim = self.read_cfg(int, "text_config->head_dim", None)
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "text_config->num_attention_groups", no_default)  # !!!

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Sliding attn layers use alternative settings
        self.sliding_window = self.read_cfg(int, "text_config->sliding_window", -1)
        self.alt_head_dim = self.read_cfg(int, "text_config->attention_other_setting->head_dim", None)
        self.alt_num_q_heads = self.read_cfg(int, "text_config->attention_other_setting->num_attention_heads", no_default)
        self.alt_num_kv_heads = self.read_cfg(int, "text_config->attention_other_setting->num_attention_groups", no_default)  # !!!

        # Layer types
        self.layer_types = self.read_cfg(list, "text_config->layer_types", no_default)

        # RoPE
        text_config = self.read_cfg(dict, "text_config", no_default)
        yarn_only_types = self.read_cfg(list, "text_config->yarn_only_types", None)
        rope_theta = self.read_cfg(list, "text_config->rope_theta", no_default)
        partial_rotary_factors = self.read_cfg(list, "text_config->partial_rotary_factors", no_default)
        self.rope_settings_list = [
            self.read_rope_settings_default(
                RopeStyle.NEOX,
                default_rope_theta = rt,
                default_partial_rotary_factor = prf,
                config_dict = text_config,
                override_type = "default" if yarn_only_types and lt not in yarn_only_types else None,
            )
            for (rt, prf, lt) in zip(rope_theta, partial_rotary_factors, self.layer_types)
        ]

        # MLP params
        self.intermediate_size = self.read_cfg(int, "text_config->intermediate_size", no_default)

        self.assert_cfg(bool, "text_config->norm_expert_weight", True, True)
        self.moe_intermediate_size = self.read_cfg(int, "text_config->moe_intermediate_size", no_default)
        self.num_experts = self.read_cfg(int, "text_config->moe_num_experts", no_default)
        self.num_experts_per_tok = self.read_cfg(int, "text_config->moe_top_k", no_default)
        self.shared_expert_intermediate_size = self.read_cfg(int, "text_config->share_expert_dim", no_default)
        self.assert_cfg(bool, "text_config->use_moe_router_bias", True, True)
        self.routed_scaling_factor = self.read_cfg(float, "text_config->moe_router_scaling_factor", 3.0)

        moe_layers = self.read_cfg(str, "text_config->moe_layers_enum", no_default)
        self.moe_layers = set(int(l) for l in moe_layers.split(","))

        self.swiglu_limits = self.read_cfg(list, "text_config->swiglu_limits", no_default)
        self.swiglu_limits_shared = self.read_cfg(list, "text_config->swiglu_limits_shared", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", 1e-5)
        # Step 3.7 config currently says use_qk_norm=false, but the reference
        # modeling_step3p7.py constructs and applies q_norm/k_norm unconditionally.
        self.use_qk_norm = True

        # Vision model settings
        vision_config = self.read_cfg(dict, "vision_config", no_default)
        self.vision = SimpleNamespace(
            heads = read_dict(vision_config, int, "heads", no_default),
            hidden_act = read_dict(vision_config, str, "hidden_act", "quick_gelu"),
            image_size = read_dict(vision_config, int, "image_size", 728),
            layers = read_dict(vision_config, int, "layers", no_default),
            ls_init_value = read_dict(vision_config, float, "ls_init_value", 0.1),
            num_channels = read_dict(vision_config, int, "num_channels", 3),
            patch_size = read_dict(vision_config, int, "patch_size", 14),
            use_cls_token = read_dict(vision_config, bool, "use_cls_token", False),
            use_ln_post = read_dict(vision_config, bool, "use_ln_post", False),
            use_ln_pre = read_dict(vision_config, bool, "use_ln_pre", True),
            use_abs_posemb = read_dict(vision_config, bool, "use_abs_posemb", True),
            use_rope2d = read_dict(vision_config, bool, "use_rope2d", True),
            width = read_dict(vision_config, int, "width", no_default),
            layer_norm_eps = read_dict(vision_config, float, "layer_norm_eps", 1e-5),
            mlp_ratio = read_dict(vision_config, float, "mlp_ratio", 8960 / 1536),
            rope_theta = read_dict(vision_config, float, "rope_theta", 10000.0),
            rope_theta_rescale_factor = read_dict(vision_config, float, "rope_theta_rescale_factor", 1.0),
        )
        assert not self.vision.use_cls_token
        assert self.vision.use_ln_pre
        assert not self.vision.use_ln_post

        self.vision.head_dim = self.vision.width // self.vision.heads
        self.vision.base_grid = self.vision.image_size // self.vision.patch_size

        self.projector_bias = self.read_cfg(bool, "projector_bias", False)
        self.image_token_id = self.read_cfg(int, "image_token_id", None)
        self.im_start_token = self.read_cfg(str, "im_start_token", "<im_start>")
        self.im_end_token = self.read_cfg(str, "im_end_token", "<im_end>")
        self.im_patch_token = self.read_cfg(str, "im_patch_token", "<im_patch>")
        self.image_token_len = self.read_cfg(int, "image_token_len", 169)
        self.patch_token_len = self.read_cfg(int, "patch_token_len", 81)
        self.patch_size_pp = 504


    def get_tensor_name_fixes(self):
        return {
            ".in_proj_weight": ".in_proj.weight",
            ".in_proj_bias": ".in_proj.bias",
        }


class Step3_7Model(Step3_5Model):
    config_class = Step3_7Config

    def __init__(
        self,
        config: Step3_7Config,
        key_prefix: str = "model",
        swa_full: bool = False,
        **kwargs
    ):
        super().__init__(config, key_prefix, swa_full, **kwargs)


class Step3_7VisionModel(Model):
    config_class = Step3_7Config

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: Step3_7Config) -> dict:
        tensors = config.stc.list_tensors(prefix = "model.vision_model")
        tensors |= config.stc.list_tensors(prefix = "vision_model")
        tensors |= config.stc.list_tensors(prefix = "model.vit_large_projector")
        tensors |= config.stc.list_tensors(prefix = "vit_large_projector")
        return tensors


    def __init__(
        self,
        config: Step3_7Config,
        key_prefix: str = "model.vision_model",
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.caps.update({"image_input": True})

        if not config.stc.has_tensor(f"{key_prefix}.conv1.weight"):
            key_prefix = "vision_model"
        projector_key = "model.vit_large_projector"
        if not config.stc.has_tensor(f"{projector_key}.weight"):
            projector_key = "vit_large_projector"

        v = config.vision
        self.vision_key_prefix = key_prefix
        self.projector_key = projector_key
        self.modules += [
            Conv(
                config = config,
                key = f"{key_prefix}.conv1",
                in_channels = v.num_channels,
                out_channels = v.width,
                kernel_size = (v.patch_size, v.patch_size),
                flat = True,
                out_dtype = torch.float,
            ),
            Step3_7PosEmbedding(
                config = config,
                key = f"{key_prefix}.positional_embedding",
                hidden_size = v.width,
                base_grid = v.base_grid,
                out_dtype = torch.float,
            ),
            LayerNorm(
                config = config,
                key = f"{key_prefix}.ln_pre",
                layernorm_eps = v.layer_norm_eps
            ),
        ]

        for idx in range(v.layers):
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.transformer.resblocks.{idx}",
                    layer_idx = idx,
                    attn_norm = LayerNorm(
                        config = config,
                        key = f"{key_prefix}.transformer.resblocks.{idx}.ln_1",
                        layernorm_eps = v.layer_norm_eps,
                    ),
                    attn = Attention(
                        config = config,
                        key = f"{key_prefix}.transformer.resblocks.{idx}.attn",
                        layer_idx = idx,
                        hidden_size = v.width,
                        head_dim = v.head_dim,
                        num_q_heads = v.heads,
                        num_kv_heads = v.heads,
                        rope_settings = (
                            RopeSettings(
                                head_dim = v.head_dim // 2,
                                rope_theta = v.rope_theta * v.rope_theta_rescale_factor ** (v.head_dim / (v.head_dim - 2)),
                                rope_style = RopeStyle.GPTJ,
                            )
                            if v.use_rope2d else None
                        ),
                        key_fused_qkv = "in_proj",
                        key_o = "out_proj",
                        qmap = "block.attn",
                        out_dtype = torch.float,
                    ),
                    mlp_norm = LayerNorm(
                        config = config,
                        key = f"{key_prefix}.transformer.resblocks.{idx}.ln_2",
                        layernorm_eps = v.layer_norm_eps,
                    ),
                    mlp = MLP(
                        config = config,
                        key = f"{key_prefix}.transformer.resblocks.{idx}.mlp",
                        hidden_size = v.width,
                        intermediate_size = int(v.width * v.mlp_ratio),
                        key_up = "c_fc",
                        key_down = "c_proj",
                        activation_fn = v.hidden_act,
                        qmap = "block.mlp",
                        out_dtype = torch.float,
                        pad_to = 1,
                    ),
                    key_attn_resid_scalar = "ls_1.gamma",
                    key_mlp_resid_scalar = "ls_2.gamma",
                )
            ]

        self.modules += [
            Step3_7Downsampler(
                config = config,
                key = f"{key_prefix}.vit_downsampler",
                key_1 = f"{key_prefix}.vit_downsampler1",
                key_2 = f"{key_prefix}.vit_downsampler2",
                hidden_size = v.width,
                kernel_size = 3,
                stride = 2,
                padding = 1,
                out_dtype = torch.half
            ),
            Linear(
                config = config,
                key = projector_key,
                in_features = v.width * 4,
                out_features = config.hidden_size,
                qmap = "block",
                out_dtype = torch.half,
                pad_to = 1,
            )
        ]

        self.class_embedding = None
        self.positional_embedding = None


    @override
    def unload(self):
        self.class_embedding = None
        self.positional_embedding = None
        super().unload()


    def default_load_shape_dtype(self, chunk_size):
        return (
            (1, 3, self.config.vision.image_size, self.config.vision.image_size),
            torch.half
        )


    def default_load_params(self, max_chunk_size):
        v = self.config.vision
        hw = v.image_size // v.patch_size
        return { "grid_hw": (hw, hw) }


    def preprocess(self, image: Image.Image):
        raw_img, patches, patch_newline_mask = get_patches(image)
        pixel_values = image_to_tensor(raw_img, self.config.vision.image_size)
        patch_pixel_values = torch.cat(
            [image_to_tensor(p, self.config.patch_size_pp) for p in patches],
            dim = 0
        ) if patches else None
        return pixel_values, patch_pixel_values, len(patches), patch_newline_mask, raw_img.size


    def _get_hw(self, x):
        v = self.config.vision
        bsz, _, height, width = x.shape
        grid_h, grid_w = height // v.patch_size, width // v.patch_size
        return grid_h, grid_w


    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image.Image | list[Image.Image],
        text_alias: str | None = None,
    ):
        cfg = self.config
        v = self.config.vision
        id_start = tokenizer.single_id(cfg.im_start_token)
        id_end = tokenizer.single_id(cfg.im_end_token)
        id_patch_start = tokenizer.single_id("<patch_start>")
        id_patch_end = tokenizer.single_id("<patch_end>")
        id_patch_newline = tokenizer.single_id("<patch_newline>")

        if isinstance(image, list):
            assert text_alias is None, "Cannot apply single alias to list of images"
            return [self.get_image_embeddings(tokenizer, i) for i in image]

        (
            pixel_values,
            patch_pixel_values,
            num_patches,
            patch_newline_mask,
            prep_image_size,
        ) = self.preprocess(image)

        hw = self._get_hw(pixel_values)
        inv_freq = step3_7_position_embedding_grid_2d(hw, v.head_dim, v.rope_theta)
        params = { "causal": False, "grid_hw": hw, "inv_freq": inv_freq }
        image_features = self.forward(pixel_values, params).cpu()

        features = []
        token_ids = []
        if num_patches:
            hw = self._get_hw(patch_pixel_values)
            inv_freq = step3_7_position_embedding_grid_2d(hw, v.head_dim, v.rope_theta)
            inv_freq = inv_freq.repeat(num_patches, 1, 1)
            params = { "causal": False, "grid_hw": hw, "inv_freq": inv_freq }
            patch_features = self.forward(patch_pixel_values, params).cpu()

            for idx in range(num_patches):
                features.append(patch_features[idx])
                token_ids += [id_patch_start] + [-1] * self.config.patch_token_len + [id_patch_end]
                if patch_newline_mask and patch_newline_mask[idx]:
                    token_ids.append(id_patch_newline)

        features.append(image_features[0])
        token_ids += [id_start] + [-1] * self.config.image_token_len + [id_end]
        embedding_tensor = torch.cat(features, dim = 0)
        token_string = torch.tensor([token_ids], dtype = torch.long)

        mme = MMEmbedding(
            embeddings = embedding_tensor,
            text_alias = text_alias,
            token_string = token_string,
        )
        mme.metadata.update({
            "original_size": image.size,
            "preprocessed_size": prep_image_size,
            "num_patches": num_patches,
            "model_architecture": self.config.architecture,
        })
        return mme


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return input_ids
