from __future__ import annotations
from typing_extensions import override
import numpy as np
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..model.model import Model
from ..util.rope import RopeStyle, position_embedding_grid_2d, RopeSettings, RoPE
from ..util.vision import convert_to_rgb, normalize_image, smart_resize, get_qwen2_window_index
from ..util.file import read_dict,  no_default
from ..util.tensor import get_for_device
from ..modules import (
    Module,
    TransformerBlock,
    Attention,
    Linear,
    Conv,
    GatedMLP,
    RMSNorm,
    Qwen3VLPosEmbedding
)
from .hyperclovax import HyperClovaxModel
from ..tokenizer import Tokenizer, MMEmbedding
from types import SimpleNamespace
from PIL import Image
import os, json

class HCXVisionV2Config(Config):
    arch_string = "HCXVisionV2ForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": HCXVisionV2Model, "vision": HCXVisionV2VisionModel},
            **kwargs
        )

        self.embedding_multiplier = self.read_cfg(float, "text_config->embedding_multiplier", 1.0)
        self.logits_scaling = self.read_cfg(float, "text_config->logits_scaling", no_default)

        self.assert_cfg(float, "text_config->residual_multiplier", 1.0, True)

        # Attention params
        self.head_dim = self.read_cfg(int, "text_config->head_dim", None)
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "text_config->num_key_value_heads", self.num_q_heads)
        self.attention_multiplier = self.read_cfg(float, "text_config->attention_multiplier", None)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.assert_cfg(str, "text_config->hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "text_config->intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "text_config->tie_word_embeddings", False)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            config_dict = self.read_cfg(dict, "text_config", no_default)
        )

        # Vision model settings
        read_vision_config = self.read_cfg(dict, "vision_config", no_default)
        self.vision = read_hcxv2_vision_config(read_vision_config)

        prep_path = os.path.join(self.directory, "preprocessor_config.json")
        with open(prep_path, encoding = "utf8") as f:
            read_prep_config = json.load(f)
        self.vision_pp = read_hcxv2_pp_config(read_prep_config)

def read_hcxv2_vision_config(config_dict: dict):
    v = SimpleNamespace(**{
        k: read_dict(config_dict, t, k, no_default)
        for k, t in [
            ("depth", int),
            ("fullatt_block_indexes", list),
            ("hidden_act", str),
            ("hidden_size", int),
            ("intermediate_size", int),
            ("max_length", int),
            ("max_num_grids", int),
            ("min_length", int),
            ("num_heads", int),
            ("out_hidden_size", int),
            ("patch_size", int),
            ("spatial_merge_size", int),
            ("spatial_patch_size", int),
            ("temporal_patch_size", int),
            ("model_type", str),
            ("window_size", int),
        ]
    })
    v.num_channels = 3
    v.head_dim = v.hidden_size // v.num_heads
    v.rms_norm_eps = 1e-6
    v.rope_theta = 10000
    assert v.model_type in ["qwen2_5_vl"], \
        "Expected vision_config->model_type to be 'qwen2_5_vl'"
    return v


def read_hcxv2_pp_config(config_dict: dict):
    pp = SimpleNamespace(**{
        k: read_dict(config_dict, t, k, no_default)
        for k, t in [
            ("size", dict),
            ("patch_size", int),
            ("temporal_patch_size", int),
            ("merge_size", int),
            ("image_mean", list),
            ("image_std", list),
            ("image_processor_type", str),
        ]
    })
    pp.resample = 3
    pp.rescale_factor = 1 / 255
    pp.min_pixels = pp.size["shortest_edge"]  # Mislabeled in preprocessor_config
    pp.max_pixels = pp.size["longest_edge"]
    assert pp.image_processor_type in ["Qwen2VLImageProcessor", "Qwen2VLImageProcessorFast"], \
        "Expected image_processor_type to be 'Qwen2VLImageProcessor' or 'Qwen2VLImageProcessorFast'"
    return pp


class HCXVisionV2Model(HyperClovaxModel):
    config_class = HCXVisionV2Config

    def __init__(
        self,
        config: HCXVisionV2Config,
        **kwargs
    ):
        super().__init__(
            config,
            key_prefix = "model.language_model.model",
            post_norms = False,
            head_key = "model.language_model.lm_head",
            **kwargs
        )

        # Generator needs MRoPE freqs when using MMEmbeddings
        # self.caps.update({"mrope": True})
        # self.g_rope = RoPE("cpu", config.rope_settings)


class HCXVisionV2VisionPatchMerger(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        key_up: str,
        key_down: str,
        key_norm: str,
        hidden_size: int,
        merge_size: int,
        out_hidden_size: int,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.in_size = hidden_size * merge_size
        self.interm_size = hidden_size * merge_size
        self.out_size = out_hidden_size
        self.out_dtype = out_dtype

        self.up = Linear(
            config = config,
            key = f"{key}.{key_up}",
            in_features = self.in_size,
            out_features = self.interm_size,
            qmap = qmap + ".input",
            out_dtype = torch.half,
            pad_to = 1
        )
        self.down = Linear(
            config = config,
            key = f"{key}.{key_down}",
            in_features = self.interm_size,
            out_features = self.out_size,
            qmap = qmap + ".down",
            out_dtype = self.out_dtype,
            allow_input_padding = True,
            pad_to = 1
        )

        self.register_submodule(self.up)
        self.register_submodule(self.down)

        if key_norm:
            self.norm = RMSNorm(
                config = config,
                key = f"{key}.{key_norm}",
                rms_norm_eps = 1e-6,
                out_dtype = torch.half,
            )
            self.register_submodule(self.norm)

    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def weights_numel(self):
        numel = self.up.weights_numel() + self.down.weights_numel()
        if self.norm: numel += self.norm.weights_numel()
        return numel

    @override
    def forward(
        self,
        x: torch.Tensor,
        params,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        bsz, seqlen, dim = x.shape
        y = self.norm.forward(x, params).to(torch.half)
        y = y.view(-1, self.in_size)

        y = self.up.forward(y, params)
        y = F.gelu(y, approximate = "tanh")
        y = self.down.forward(y, params)
        y = y.view(bsz, -1, self.out_size)

        return y


class HCXVisionV2VisionSpatialMerger(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        spatial_merge_unit: int,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.spatial_merge_unit = spatial_merge_unit
        self.out_dtype = out_dtype

    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def weights_numel(self):
        return 0

    @override
    def forward(
        self,
        x: torch.Tensor,
        params,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        window_index = params.get("window_index")
        if window_index is None:
            return x

        bsz, seq_len, dim = x.shape
        x = x.reshape(bsz * seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(bsz, seq_len, -1)

        inv_freq = params.get("inv_freq")
        if inv_freq is not None:
            inv_freq = inv_freq.reshape(bsz * seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
            inv_freq = inv_freq[window_index, :, :]
            inv_freq = inv_freq.reshape(bsz, seq_len, -1)
            params["inv_freq"] = inv_freq

        return x

# TODO: Refactor into Qwen2.5-VL base class
class HCXVisionV2VisionModel(Model):

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: HCXVisionV2Config) -> dict:
        vlm_tensors = config.stc.list_tensors(prefix = "model.vision_model")
        mmp_tensors = config.stc.list_tensors(prefix = "model.mm_projector")
        return vlm_tensors | mmp_tensors

    def __init__(
        self,
        config: HCXVisionV2Config,
        key_prefix = "model.vision_model",
        mm_prefix = "model.mm_projector",
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.caps.update({
            "image_input": True,
        })
        v = self.config.vision

        self.modules += [
            Conv(
                config = config,
                key = f"{key_prefix}.patch_embed.proj",
                in_channels = v.num_channels,
                out_channels = v.hidden_size,
                kernel_size = (v.temporal_patch_size, v.patch_size, v.patch_size),
                flat = True,
                out_dtype = torch.float,
            ),
            HCXVisionV2VisionSpatialMerger(
                config = config,
                key = f"{key_prefix}.spatial_merger",
                spatial_merge_unit = v.spatial_merge_size**2,
            )
        ]

        for idx in range(v.depth):

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.blocks.{idx}",
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.blocks.{idx}.norm1",
                        rms_norm_eps = v.rms_norm_eps
                    ),
                    attn = Attention(
                        config = config,
                        key = f"{key_prefix}.blocks.{idx}.attn",
                        layer_idx = idx,
                        hidden_size = v.hidden_size,
                        head_dim = v.head_dim,
                        num_q_heads = v.num_heads,
                        num_kv_heads = v.num_heads,
                        rope_settings = RopeSettings(
                            head_dim = v.head_dim,
                            rope_style = RopeStyle.NEOX,
                        ),
                        key_fused_qkv = "qkv",
                        key_o = "proj",
                        qmap = "block.attn",
                        use_cu_seqlens = bool(idx not in v.fullatt_block_indexes)
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.blocks.{idx}.norm2",
                        rms_norm_eps = v.rms_norm_eps
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"{key_prefix}.blocks.{idx}.mlp",
                        hidden_size = v.hidden_size,
                        intermediate_size = v.intermediate_size,
                        key_gate = "gate_proj",
                        key_up = "up_proj",
                        key_down = "down_proj",
                        activation_fn = "silu",
                        qmap = "block.mlp",
                        pad_to = 1,
                    ),
                )
            ]

        self.modules += [
            HCXVisionV2VisionPatchMerger(
                config = config,
                key = f"{key_prefix}.merger",
                key_norm = "ln_q",
                key_up = "mlp.0",
                key_down = "mlp.2",
                hidden_size = v.hidden_size,
                merge_size = v.spatial_merge_size ** 2,
                out_hidden_size = v.out_hidden_size,
                out_dtype = torch.half,
                qmap = "block",
            ),
            Linear(
                config = config,
                key = mm_prefix,
                in_features = v.out_hidden_size,
                out_features = v.out_hidden_size,
                qmap = "block",
                out_dtype = torch.float,
            )
        ]


    def preprocess(
        self,
        images: Image | list[Image]
    ) -> (torch.Tensor, tuple):
        v = self.config.vision
        pp = self.config.vision_pp
        resample = Image.Resampling(pp.resample)
        image_mean, image_std = tuple(pp.image_mean), tuple(pp.image_std)

        # Make list and truncate to whole number of spatial patches
        if not isinstance(images, list):
            mode = "image"
            images = [images]
        else:
            mode = "video"
            g = pp.temporal_patch_size
            frames = len(images)
            if frames > 1:
                frames = frames // g * g
                images = images[:frames]

        # Convert to RGB and resize as necessary
        images = [convert_to_rgb(image) for image in images]

        old_size = images[0].size
        assert all(old_size == frame.size for frame in images), \
            "All frames in video must have same dimensions"

        new_size = smart_resize(
            old_size,
            pp.patch_size * v.spatial_merge_size,
            pp.min_pixels,
            pp.max_pixels,
        )
        if old_size != new_size:
            images = [image.resize(new_size, resample = resample) for image in images]

        # Convert to numpy array and normalize
        images = [np.array(image).astype(np.float32) for image in images]
        images = [image * pp.rescale_factor for image in images]
        images = [normalize_image(image, image_mean, image_std) for image in images]

        # Reshape and convert to tensor
        patches = np.array(images)
        patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] == 1:
            patches = np.tile(patches, (pp.temporal_patch_size, 1, 1, 1))
        channels = patches.shape[1]
        grid_t = patches.shape[0] // pp.temporal_patch_size
        grid_h = new_size[1] // pp.patch_size
        grid_w = new_size[0] // pp.patch_size
        patches = patches.reshape(
            grid_t,
            pp.temporal_patch_size,
            channels,
            grid_h // v.spatial_merge_size,
            v.spatial_merge_size,
            pp.patch_size,
            grid_w // v.spatial_merge_size,
            v.spatial_merge_size,
            pp.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channels * pp.temporal_patch_size * pp.patch_size ** 2
        )

        if mode == "image":
            image = torch.from_numpy(flatten_patches).half()
            return image, new_size, (grid_t, grid_h, grid_w)
        else:
            video = torch.from_numpy(flatten_patches).half()
            return video, new_size, (grid_t, grid_h, grid_w)


    def default_load_shape_dtype(self, chunk_size):
        return (
            (
                1,
                9216,
                1176,
            ),
            torch.half
        )


    def default_load_params(self):
        return { "grid_thw": torch.tensor([[1, 58, 46]]) }


    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image | list[Image],
        text_alias: str | None = None,
    ):
        v = self.config.vision

        if isinstance(image, list):
            assert text_alias is None, "Cannot apply single alias to list of images"
            image_tensor = []
            for i in image:
                t, prep_image_size, grid_thw = self.preprocess(i)
                image_tensor.append(t.unsqueeze(0))
            image_tensor = torch.cat(image_tensor, dim = 0)
            return_batch = True
        else:
            image_tensor, prep_image_size, grid_thw = self.preprocess(image)
            image = [image]
            image_tensor = image_tensor.unsqueeze(0)
            return_batch = False

        inv_freq = position_embedding_grid_2d(
            grid_thw,
            v.head_dim,
            v.spatial_merge_size,
            v.rope_theta
        )
        window_index, window_cu_seqlens = get_qwen2_window_index(
            [grid_thw],
            v.window_size,
            v.spatial_merge_size,
            v.patch_size
        )
        window_cu_seqlens = torch.unique_consecutive(torch.tensor(window_cu_seqlens, dtype = torch.int))
        max_seqlen = (window_cu_seqlens[1:] - window_cu_seqlens[:-1]).max().item()
        params = {
            "causal": False,
            "grid_thw": torch.tensor([grid_thw], dtype = torch.int),
            "inv_freq": inv_freq,
            "window_index": window_index,
            "cu_seqlens": window_cu_seqlens,
            "max_seqlen": max_seqlen
        }

        embedding_tensor = self.forward(
            image_tensor,
            params = params,
        ).cpu()

        # Qwen2.5 wants this, idk
        reverse_indices = torch.argsort(window_index)
        embedding_tensor = embedding_tensor[0][reverse_indices, :].unsqueeze(0)

        num_emb_tokens = embedding_tensor.shape[1]
        mmes = []
        for i in range(embedding_tensor.shape[0]):
            id_start = tokenizer.single_id("<|image_start|>")
            id_end = tokenizer.single_id("<|image_end|>")
            token_string = torch.tensor([[id_start] + [-1] * num_emb_tokens + [id_end]], dtype = torch.long)

            mme = MMEmbedding(
                embeddings = embedding_tensor[i],
                text_alias = text_alias,
                token_string = token_string,
                grid_thw = grid_thw,
                mrope_merge_size = v.spatial_merge_size
            )

            mme.metadata.update({
                "original_size": image[i].size,
                "preprocessed_size": prep_image_size,
                "model_architecture": self.config.architecture,
            })

            mmes.append(mme)

        return mmes if return_batch else mmes[0]


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return input_ids