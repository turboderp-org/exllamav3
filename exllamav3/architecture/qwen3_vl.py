from __future__ import annotations
from typing_extensions import override
import numpy as np
import torch
from ..model.config import Config
from ..model.model import Model
from ..util.rope import RopeStyle, RopeSettings
from .mm_processing.common import convert_to_rgb, normalize_image
from .mm_processing.qwen2 import qwen2_smart_resize, qwen2_position_embedding_grid_2d
from ..util.file import read_dict,  no_default
from ..modules import (
    TransformerBlock,
    Attention,
    Conv,
    MLP,
    LayerNorm,
)
from ..modules.arch_specific.qwen3_vl import Qwen3VLPosEmbedding, Qwen3VLVisionPatchMerger
from .qwen3 import Qwen3Model
from ..tokenizer import Tokenizer, MMEmbedding
from types import SimpleNamespace
from PIL import Image
import os, json

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .qwen3_vl_moe import Qwen3VLMoeConfig


class Qwen3VLConfig(Config):
    arch_string = "Qwen3VLForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Qwen3VLModel, "vision": Qwen3VLVisionModel},
            **kwargs
        )

        # Attention params
        self.head_dim = self.read_cfg(int, "text_config->head_dim", None)
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", no_default)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", no_default)
        self.num_kv_heads = self.read_cfg(int, "text_config->num_key_value_heads", self.num_q_heads)

        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # MLP params
        self.assert_cfg(str, "text_config->hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(int, "text_config->intermediate_size", no_default)

        # Norms
        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", no_default)

        # Layers
        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)

        # RoPE
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 5000000,
            config_dict = self.read_cfg(dict, "text_config", no_default)
        )

        # Vision model settings
        read_vision_config = self.read_cfg(dict, "vision_config", no_default)
        self.vision = read_qwen3_vl_vision_config(read_vision_config)

        prep_path = os.path.join(self.directory, "preprocessor_config.json")
        with open(prep_path, encoding = "utf8") as f:
            read_prep_config = json.load(f)
        self.vision_pp = read_qwen3_vl_pp_config(read_prep_config)

        self.vision_start_token_id = self.read_cfg(int, "vision_start_token_id", 151652)
        self.vision_end_token_id = self.read_cfg(int, "vision_end_token_id", 151653)


def read_qwen3_vl_vision_config(config_dict: dict):
    v = SimpleNamespace(**{
        k: read_dict(config_dict, t, k, no_default)
        for k, t in [
            ("deepstack_visual_indexes", list),
            ("depth", int),
            ("hidden_act", str),
            ("hidden_size", int),
            ("intermediate_size", int),
            ("num_heads", int),
            ("num_position_embeddings", int),
            ("out_hidden_size", int),
            ("patch_size", int),
            ("spatial_merge_size", int),
            ("temporal_patch_size", int),
            ("model_type", str),
        ]
    })
    v.num_channels = 3
    v.head_dim = v.hidden_size // v.num_heads
    v.rope_theta = 10000.0
    v.layernorm_eps = 1e-6
    model_types = [
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_vision",
        "qwen3_5_moe_vision",
    ]
    assert v.model_type in model_types, \
        f"Expected vision_config->model_type to be one of {model_types}"
    return v


def read_qwen3_vl_pp_config(config_dict: dict):
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
    assert pp.image_processor_type == "Qwen2VLImageProcessorFast", \
        "Expected image_processor_type to be 'Qwen2VLImageProcessorFast'"
    return pp


class Qwen3VLModel(Qwen3Model):
    config_class = Qwen3VLConfig

    def __init__(
        self,
        config: Qwen3VLConfig,
        **kwargs
    ):
        super().__init__(config, key_prefix = "model.language_model", **kwargs)


class Qwen3VLVisionModel(Model):

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: Qwen3VLConfig) -> dict:
        vlm_tensors = config.stc.list_tensors(prefix = "model.visual")
        return vlm_tensors

    def __init__(
        self,
        config: Qwen3VLConfig | Qwen3VLMoeConfig,
        key_prefix = "model.visual",
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
            Qwen3VLPosEmbedding(
                config = config,
                key = f"{key_prefix}.pos_embed",
                num_position_embeddings = v.num_position_embeddings,
                spatial_merge_size = v.spatial_merge_size,
                hidden_size = v.hidden_size,
                out_dtype = torch.float,
            ),
        ]

        for idx in range(v.depth):

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.blocks.{idx}",
                    layer_idx = idx,
                    attn_norm = LayerNorm(
                        config = config,
                        key = f"{key_prefix}.blocks.{idx}.norm1",
                        layernorm_eps = v.layernorm_eps
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
                        use_cu_seqlens = True,
                    ),
                    mlp_norm = LayerNorm(
                        config = config,
                        key = f"{key_prefix}.blocks.{idx}.norm2",
                        layernorm_eps = v.layernorm_eps
                    ),
                    mlp = MLP(
                        config = config,
                        key = f"{key_prefix}.blocks.{idx}.mlp",
                        hidden_size = v.hidden_size,
                        intermediate_size = v.intermediate_size,
                        key_up = "linear_fc1",
                        key_down = "linear_fc2",
                        activation_fn = "gelu",
                        qmap = "block.mlp",
                        pad_to = 1,
                    ),
                )
            ]

            if idx in v.deepstack_visual_indexes:
                merger_idx = v.deepstack_visual_indexes.index(idx)
                self.modules += [
                    Qwen3VLVisionPatchMerger(
                        config = config,
                        key = f"{key_prefix}.deepstack_merger_list.{merger_idx}",
                        key_norm = "norm",
                        key_up = "linear_fc1",
                        key_down = "linear_fc2",
                        hidden_size = v.hidden_size,
                        merge_size = v.spatial_merge_size ** 2,
                        out_hidden_size = v.out_hidden_size,
                        use_postshuffle_norm = True,
                        extract = merger_idx,
                        out_dtype = torch.float,
                        qmap = "block",
                    )
                ]

        self.modules += [
            Qwen3VLVisionPatchMerger(
                config = config,
                key = f"{key_prefix}.merger",
                key_norm = "norm",
                key_up = "linear_fc1",
                key_down = "linear_fc2",
                hidden_size = v.hidden_size,
                merge_size = v.spatial_merge_size ** 2,
                out_hidden_size = v.out_hidden_size,
                out_dtype = torch.float,
                qmap = "block",
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

        new_size = qwen2_smart_resize(
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
                11008,
                1536,
            ),
            torch.half
        )


    def default_load_params(self, max_chunk_size):
        return { "grid_thw": torch.tensor([[1, 86, 128]]) }


    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image | list[Image],
        text_alias: str | None = None,
    ):
        if isinstance(image, list):
            images = image
            return_batch = True
        else:
            images = [image]
            return_batch = False

        if not images:
            raise ValueError("No images provided")
        if return_batch:
            assert text_alias is None, "Cannot apply single alias to list of images"

        prepped = []
        seg_lens = []
        for img in images:
            tensor, prep_image_size, grid_thw = self.preprocess(img)
            prepped.append((tensor, prep_image_size, grid_thw))
            seg_lens.append(grid_thw[0] * grid_thw[1] * grid_thw[2])

        bounds = [0]
        for length in seg_lens:
            bounds.append(bounds[-1] + length)

        mmes = self._embed_packed(
            tokenizer,
            prepped,
            text_alias,
            torch.tensor(bounds, dtype = torch.int32),
            max(seg_lens),
            images,
        )
        return mmes if return_batch else mmes[0]


    def get_video_embeddings(
        self,
        tokenizer: Tokenizer,
        frames: list[Image],
        text_alias: str | None = None,
    ) -> MMEmbedding:
        if not frames:
            raise ValueError("No frames provided")

        # One attention segment per frame (HF convention: get_vision_cu_seqlens with
        # merge_temporal=False); vision RoPE is spatial-only over the whole clip.
        tensor, prep_image_size, grid_thw = self.preprocess(frames)
        t, h, w = grid_thw

        return self._embed_packed(
            tokenizer,
            [(tensor, prep_image_size, grid_thw)],
            text_alias,
            torch.tensor([i * h * w for i in range(t)] + [t * h * w], dtype = torch.int32),
            h * w,
            [frames[0]],
        )[0]


    def _embed_packed(
        self,
        tokenizer: Tokenizer,
        prepped: list[tuple[torch.Tensor, tuple, tuple]],
        text_alias: str | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        images: list[Image],
    ) -> list[MMEmbedding]:
        # Pack all inputs into one varlen sequence: tokens concatenated along the
        # sequence dim, segmented by cu_seqlens, with per-grid RoPE tables stacked.
        #
        # EXL3_VISION_PACK_CHUNK: optional upper bound on tokens fed through the tower in
        # one forward pass. When set (> 0) and more than one input unit is packed,
        # consecutive units are grouped so each pass stays within the budget and peak
        # activation RAM stays O(group) instead of O(total); group outputs are concatenated
        # afterwards and match the monolithic pass within normal fp tolerance (grouping
        # changes GEMM M dims, so bit equality is not expected). A single unit is never
        # split internally: one video clip larger than the budget runs as an oversized
        # group rather than being cut up (the flat temporal conv couples frames within a
        # clip). Default 0 keeps the monolithic pass exactly as before.
        v = self.config.vision
        m = v.spatial_merge_size

        token_counts = [g[0] * g[1] * g[2] for _, _, g in prepped]
        chunk_tokens = int(os.environ.get("EXL3_VISION_PACK_CHUNK", "0"))
        groups = None
        if chunk_tokens > 0 and len(prepped) > 1 and sum(token_counts) > chunk_tokens:
            groups = []
            group, group_tokens = [], 0
            for item, n in zip(prepped, token_counts):
                if group and group_tokens + n > chunk_tokens:
                    groups.append(group)
                    group, group_tokens = [], 0
                group.append(item)
                group_tokens += n
            if group:
                groups.append(group)

        def run_group(units, group_cu = None, group_max = None):
            grids_g = [g for _, _, g in units]
            packed_g = torch.cat([t.unsqueeze(0) for t, _, _ in units], dim = 1)
            inv_freq_g = torch.cat([
                qwen2_position_embedding_grid_2d(g, v.head_dim, m, v.rope_theta)
                for g in grids_g
            ], dim = 0)
            if group_cu is None:
                lens_g = [g[0] * g[1] * g[2] for g in grids_g]
                bounds_g = [0]
                for length in lens_g:
                    bounds_g.append(bounds_g[-1] + length)
                group_cu = torch.tensor(bounds_g, dtype = torch.int32)
                group_max = max(lens_g)
            params_g = {
                "causal": False,
                "grid_thw": torch.tensor(grids_g, dtype = torch.int),
                "inv_freq": inv_freq_g,
                "cu_seqlens": group_cu,
                "max_seqlen": group_max,
            }
            embedding_tensor_g = self.forward(packed_g, params = params_g).cpu()
            deepstack_g = \
                [de.squeeze(0) for de in params_g["deepstack"]] if "deepstack" in params_g else None
            return embedding_tensor_g, deepstack_g

        if groups is None:
            embedding_tensor, deepstack = run_group(prepped, cu_seqlens, max_seqlen)
        else:
            embeddings, deepstacks = [], []
            for grp in groups:
                emb, ds = run_group(grp)
                embeddings.append(emb)
                deepstacks.append(ds)
            embedding_tensor = torch.cat(embeddings, dim = 1)
            deepstack = None if deepstacks[0] is None else [
                torch.cat(parts, dim = 0)
                for parts in zip(*deepstacks)
            ]

        grids = [g for _, _, g in prepped]

        merged_lens = [(g[0] * g[1] * g[2]) // (m * m) for g in grids]
        id_start = self.config.vision_start_token_id
        id_end = self.config.vision_end_token_id

        mmes = []
        offset = 0
        for i, num_emb_tokens in enumerate(merged_lens):
            token_string = torch.tensor([[id_start] + [-1] * num_emb_tokens + [id_end]], dtype = torch.long)
            mme = MMEmbedding(
                embeddings = embedding_tensor[0, offset : offset + num_emb_tokens],
                text_alias = text_alias,
                token_string = token_string,
                deepstack_embeddings = None if deepstack is None else [
                    de[offset : offset + num_emb_tokens] for de in deepstack
                ],
                grid_thw = grids[i],
                mrope_merge_size = m,
            )
            mme.metadata.update({
                "original_size": images[i].size,
                "preprocessed_size": prepped[i][1],
                "model_architecture": self.config.architecture,
            })
            mmes.append(mme)
            offset += num_emb_tokens

        return mmes


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return input_ids