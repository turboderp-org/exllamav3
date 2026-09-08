from __future__ import annotations
from typing_extensions import override
import math
import numpy as np
import torch
from PIL import Image, ImageOps
from types import SimpleNamespace
from ..model.model import Model
from ..util.rope import RopeStyle, RopeSettings
from ..modules import (
    TransformerBlock,
    Attention,
    Linear,
    GatedMLP,
    RMSNorm,
)
from ..modules.arch_specific.deepseek_v4_vision import DeepseekV4VisionAligner
from ..tokenizer import Tokenizer, MMEmbedding
from .mm_processing.common import convert_to_rgb
from .mm_processing.qwen2 import qwen2_position_embedding_grid_2d
from .mm_processing.deepseek_v4 import safe_resize, COMPRESS_PAD_TO, build_image_block, IMAGE_START, IMAGE_PAD, \
    _MARKER_ROW, IMAGE

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .deepseek_v4 import DeepseekV4Config


def read_deepseek_v4_vision_config(config) -> SimpleNamespace | None:
    n_layers = config.read_cfg(int, "vision_n_layers", 0)
    if n_layers <= 0:
        return None
    v = SimpleNamespace(
        n_layers = n_layers,
        dim = config.read_cfg(int, "vision_dim", 1024),
        n_heads = config.read_cfg(int, "vision_n_heads", 16),
        inter_dim = config.read_cfg(int, "vision_inter_dim", 2816),
        patch_size = config.read_cfg(int, "vision_patch_size", 14),
        rope_theta = config.read_cfg(float, "vision_rope_theta", 10000.0),
        downsample_ratio = config.read_cfg(int, "vision_downsample_ratio", 3),
        max_n_token = config.read_cfg(int, "vision_max_n_token", 384),
        min_pixels = config.read_cfg(int, "vision_min_pixels", 147456),
        max_wh_ratio = config.read_cfg(int, "vision_max_wh_ratio", 8),
    )
    v.head_dim = v.dim // v.n_heads
    v.num_channels = 3
    v.rms_norm_eps = 1e-6
    return v


class DeepseekV4VisionModel(Model):

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: DeepseekV4Config) -> dict:
        t = {}
        for prefix in ("vision", "aligner", "image_start", "image_pad", "image_newline", "image_end"):
            t.update(config.stc.list_tensors(prefix = prefix))
        return t

    def __init__(
        self,
        config: DeepseekV4Config,
        key_prefix = "vision",
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.caps.update({
            "image_input": True,
            "default_vision_bits": 6,
        })
        v = self.config.vision
        assert v is not None, "DeepseekV4VisionModel: config has no vision tower"

        self.modules += [
            Linear(
                config = config,
                key = f"{key_prefix}.patch_embed.proj",
                in_features = v.num_channels * v.patch_size ** 2,
                out_features = v.dim,
                pad_to = 1,
                out_dtype = torch.float,
            ),
        ]

        for idx in range(v.n_layers):
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.blocks.{idx}",
                    layer_idx = idx,
                    attn_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.blocks.{idx}.norm1",
                        rms_norm_eps = v.rms_norm_eps,
                    ),
                    attn = Attention(
                        config = config,
                        key = f"{key_prefix}.blocks.{idx}.attn",
                        layer_idx = idx,
                        hidden_size = v.dim,
                        head_dim = v.head_dim,
                        num_q_heads = v.n_heads,
                        num_kv_heads = v.n_heads,
                        rope_settings = RopeSettings(
                            head_dim = v.head_dim,
                            rope_style = RopeStyle.NEOX,
                        ),
                        key_fused_qkv = "wqkv",
                        key_o = "wo",
                        qmap = "block.attn",
                    ),
                    mlp_norm = RMSNorm(
                        config = config,
                        key = f"{key_prefix}.blocks.{idx}.norm2",
                        rms_norm_eps = v.rms_norm_eps,
                    ),
                    mlp = GatedMLP(
                        config = config,
                        key = f"{key_prefix}.blocks.{idx}.mlp",
                        hidden_size = v.dim,
                        intermediate_size = v.inter_dim,
                        # The checkpoint stores gate|up fused as w1; the split projections
                        # need their own names for the quantized form (a fused key alone
                        # would give both Linears the same key and one would overwrite the
                        # other at conversion)
                        key_fused_gate_up = "w1",
                        key_gate = "gate_proj",
                        key_up = "up_proj",
                        key_down = "w2",
                        activation_fn = "silu",
                        qmap = "block.mlp",
                    ),
                )
            ]

        self.modules += [
            RMSNorm(
                config = config,
                key = f"{key_prefix}.norm",
                rms_norm_eps = v.rms_norm_eps,
                out_dtype = torch.half,
            ),
            DeepseekV4VisionAligner(
                config = config,
                key = "aligner",
                key_up = "w1",
                key_down = "w2",
                vision_dim = v.dim,
                downsample_ratio = v.downsample_ratio,
                out_hidden_size = config.hidden_size,
                out_dtype = torch.half,
                qmap = "block",
            ),
        ]

    @property
    def aligner(self) -> DeepseekV4VisionAligner:
        return self.modules[-1]

    def preprocess(self, image: Image, dtype: torch.dtype = torch.half):
        """
        PIL image -> (patches (n_vit_h * n_vit_w, 3 * p * p) half in raster order, n_vit_h,
        n_vit_w, n_llm_h, n_llm_w). Reference image_processor.load_image: aspect clamped to
        max_wh_ratio, upscaled to min_pixels, resized so the token block fits max_n_token, padded
        to the patch grid with gray, normalized to [-1, 1].
        """
        v = self.config.vision
        p = v.patch_size
        image = convert_to_rgb(image)
        width, height = image.size
        if v.max_wh_ratio is not None and width > height * v.max_wh_ratio:
            width = height * v.max_wh_ratio
        if 0 < width * height < v.min_pixels:
            ratio = (v.min_pixels / (width * height)) ** 0.5
            width = int(width * ratio)
            height = int(height * ratio)
        best_width = math.ceil(width / p) * p
        best_height = math.ceil(height / p) * p
        n_llm_h, n_llm_w, best_height, best_width = safe_resize(
            height, width, best_height, best_width, p, v.downsample_ratio, v.max_n_token)
        n_vit_h, n_vit_w = best_height // p, best_width // p
        if v.max_wh_ratio is not None and image.width >= v.max_wh_ratio * image.height:
            image = image.resize((best_width, best_height))
        else:
            image = ImageOps.pad(image, (best_width, best_height), color = (127, 127, 127))
        x = torch.from_numpy(np.asarray(image, dtype = np.float32)).permute(2, 0, 1) / 255
        x = (x - 0.5) / 0.5
        patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3 * p * p)
        return patches.to(dtype), n_vit_h, n_vit_w, n_llm_h, n_llm_w

    def default_load_shape_dtype(self, chunk_size):
        v = self.config.vision
        return ((1, 3456, v.num_channels * v.patch_size ** 2), torch.half)

    def default_load_params(self, max_chunk_size):
        v = self.config.vision
        n_h, n_w = 48, 72
        return {
            "causal": False,
            "grid_hw": (n_h, n_w),
            "inv_freq": qwen2_position_embedding_grid_2d((1, n_h, n_w), v.head_dim, 1, v.rope_theta),
        }

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        return input_ids

    def encode_patches(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        """(n_vit_h * n_vit_w, 3 * p * p) half patches -> aligner output (n_llm_h * n_llm_w, D)
        half, raster order over the LLM grid."""
        v = self.config.vision
        params = {
            "causal": False,
            "grid_hw": (n_vit_h, n_vit_w),
            # 2D rope: rotate-half over the full head with the h frequencies in the first half
            # of the rotary dims and the w frequencies in the second (raster order, no merge
            # window), which is the reference's [h, w] flattened frequency layout
            "inv_freq": qwen2_position_embedding_grid_2d((1, n_vit_h, n_vit_w), v.head_dim, 1, v.rope_theta),
        }
        return self.forward(patches.unsqueeze(0), params = params)[0]

    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image | list[Image],
        text_alias: str | None = None,
    ):
        """
        One MMEmbedding per image: the whole N-layout token block (markers, newline and pad rows
        from the learned vectors, image rows from the aligner) as embeddings. The block is built
        with its maximum of COMPRESS_PAD_TO - 1 leading pad rows and marked position-aligned, so
        the tokenizer emits just the suffix that puts IMAGE_START on the compressor-group phase
        the reference's build_image_block(start_pos) would (0-3 leading pads by prompt position).
        """
        if isinstance(image, list):
            assert text_alias is None, "Cannot apply single alias to list of images"
            return [self.get_image_embeddings(tokenizer, i, None) for i in image]

        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = self.preprocess(image)
        out = self.encode_patches(patches, n_vit_h, n_vit_w)
        assert out.shape[0] == n_llm_h * n_llm_w
        lead = COMPRESS_PAD_TO - 1
        types, perm = build_image_block(n_llm_h, n_llm_w, 0)     # start_pos 0 -> `lead` pads
        assert int(types[lead]) == IMAGE_START and all(int(t) == IMAGE_PAD for t in types[:lead])
        markers = self.aligner.markers
        rows = torch.tensor([_MARKER_ROW.get(int(t), 0) for t in types], dtype = torch.long, device = markers.device)
        block = markers[rows].clone()
        block[types.to(markers.device) == IMAGE] = out[perm.to(out.device)].to(block.device)
        block = block.cpu()
        token_string = torch.full((1, block.shape[0]), -1, dtype = torch.long)

        mme = MMEmbedding(
            embeddings = block,
            text_alias = text_alias,
            token_string = token_string,
            # IMAGE_START (row `lead`) must sit at a prompt position p with p % 4 == 3
            align = COMPRESS_PAD_TO,
            align_phase = COMPRESS_PAD_TO - 1,
            align_lead = lead,
        )
        mme.metadata.update({
            "original_size": image.size,
            "preprocessed_size": (n_vit_w * self.config.vision.patch_size, n_vit_h * self.config.vision.patch_size),
            "grid_vit": (n_vit_h, n_vit_w),
            "grid_llm": (n_llm_h, n_llm_w),
            "block_types": types.tolist(),
            "model_architecture": self.config.architecture,
        })
        return mme
