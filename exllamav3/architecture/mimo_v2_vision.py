from __future__ import annotations
from typing_extensions import override
import torch
from types import SimpleNamespace
from ..model.model import Model
from ..util.rope import RopeStyle, RopeSettings
from ..util.file import read_dict, no_default
from ..modules import TransformerBlock, Attention, Conv, RMSNorm, GatedMLP
from ..modules.arch_specific.qwen2_5_vl import Qwen2_5VLVisionPatchMerger
from ..modules.arch_specific.mimo_v2 import MiMoV2VisionReorder
from .mm_processing.qwen2 import qwen2_position_embedding_grid_2d
from .qwen2_5_vl import Qwen2_5VLVisionModel
from ..tokenizer import Tokenizer, MMEmbedding
from PIL import Image

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .mimo_v2 import MiMoV2Config


def read_mimo_v2_vision_config(config_dict: dict):
    v = SimpleNamespace(**{
        k: read_dict(config_dict, t, k, no_default)
        for k, t in [
            ("depth", int),
            ("fullatt_block_indexes", list),
            ("vit_window_attn_types", list),
            ("hidden_size", int),
            ("intermediate_size", int),
            ("num_heads", int),
            ("out_hidden_size", int),
            ("patch_size", int),
            ("spatial_merge_size", int),
            ("temporal_patch_size", int),
            ("visual_token_window_size", int),
        ]
    })
    v.num_kv_heads = read_dict(config_dict, int, "num_key_value_heads", v.num_heads)
    # The reference reads qk_channels with a default of 64; hidden_size / num_heads is 40 here
    v.head_dim = read_dict(config_dict, int, "qk_channels", 64)
    v.use_sink = read_dict(config_dict, bool, "use_sink", False)
    v.num_channels = read_dict(config_dict, int, "in_chans", 3)
    v.rms_norm_eps = read_dict(config_dict, float, "rms_norm_eps", 1e-6)
    v.rope_theta = 10000
    assert len(v.vit_window_attn_types) == v.depth, "MiMo-ViT: vit_window_attn_types must list every block"
    assert read_dict(config_dict, str, "hidden_act", "silu") == "silu", "MiMo-ViT: expected silu MLPs"
    return v


class MiMoV2VisionModel(Qwen2_5VLVisionModel):
    """
    MiMo-ViT (MiMo-V2.6): a Qwen2.5-VL-shaped tower (conv patch embed, 2D rotary, spatial
    merger) whose blocks are grouped-query attention with a symmetric sliding window of
    visual_token_window_size tokens and a learned per-head sink on the windowed blocks, full
    attention on fullatt_block_indexes. vit_window_attn_types gives each block's token order,
    -1 global, 0 row-major, 1 column-major: consecutive windowed blocks alternate so the window
    sweeps both spatial axes, and the tower re-serializes the tokens (and their rotary table)
    at every change of order. Preprocessing is the Qwen2-VL image pipeline (inherited).
    """

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: MiMoV2Config) -> dict:
        return config.stc.list_tensors(prefix = "visual")


    def __init__(
        self,
        config: MiMoV2Config,
        key_prefix = "visual",
        **kwargs
    ):
        Model.__init__(self, config, **kwargs)
        self.config = config
        self.caps.update({
            "image_input": True,
        })
        v = self.config.vision
        unit = v.spatial_merge_size ** 2
        window = v.visual_token_window_size

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
        ]

        prev_type = None
        for idx in range(v.depth):
            attn_type = v.vit_window_attn_types[idx]
            windowed = attn_type != -1 and idx not in v.fullatt_block_indexes
            # Column-major serialization on entry to a run of type-1 blocks, back to row-major
            # after it (the reference applies the same two index permutations)
            if attn_type == 1 and prev_type != 1:
                self.modules += [MiMoV2VisionReorder(config, f"{key_prefix}.blocks.{idx}.reorder", "mimo_vit_col_index", unit)]
            if idx > 0 and attn_type != 1 and prev_type == 1:
                self.modules += [MiMoV2VisionReorder(config, f"{key_prefix}.blocks.{idx}.reorder", "mimo_vit_row_index", unit)]
            prev_type = attn_type

            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.blocks.{idx}",
                    layer_idx = idx,
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
                        num_kv_heads = v.num_kv_heads,
                        rope_settings = RopeSettings(
                            head_dim = v.head_dim,
                            rope_style = RopeStyle.NEOX,
                        ),
                        key_fused_qkv = "qkv",
                        key_o = "proj",
                        # |i - j| <= window is visible, on both sides
                        sliding_window = window if windowed else -1,
                        window_right = window if windowed else 0,
                        key_sinks = "sinks" if (v.use_sink and windowed) else None,
                        sink_key0 = True,
                        qmap = "block.attn",
                        out_dtype = torch.float,
                        use_cu_seqlens = True,
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
                        out_dtype = torch.float,
                        pad_to = 1,
                        # The final block's MLP intermediate and output run past the fp16 range
                        # (its output feeds the merger's LayerNorm, which normalizes it away)
                        **({"interm_dtype": torch.float, "interm_div": 16.0} if idx == v.depth - 1 else {}),
                    ),
                )
            ]

        self.modules += [
            Qwen2_5VLVisionPatchMerger(
                config = config,
                key = f"{key_prefix}.merger",
                key_norm = "ln_q",
                key_up = "mlp.0",
                key_down = "mlp.2",
                hidden_size = v.hidden_size,
                merge_size = unit,
                out_hidden_size = v.out_hidden_size,
                out_dtype = torch.half,
                qmap = "block",
                norm_type = "layernorm",
                gelu_approximate = "none",
            )
        ]


    def default_load_shape_dtype(self, chunk_size):
        v = self.config.vision
        return (1, 9216, v.num_channels * v.temporal_patch_size * v.patch_size ** 2), torch.half


    def default_load_params(self, max_chunk_size):
        return {"grid_thw": torch.tensor([[1, 96, 96]])}


    def _grid_params(self, grid_thw: tuple) -> dict:
        """Everything the tower derives from one image's (t, h, w) patch grid: the rotary
        table, the two serialization indices (merge-unit granularity), and per-frame
        attention segments"""
        v = self.config.vision
        t, h, w = (int(x) for x in grid_thw)
        gh, gw = h // v.spatial_merge_size, w // v.spatial_merge_size
        inv_freq = qwen2_position_embedding_grid_2d((t, h, w), v.head_dim, v.spatial_merge_size, v.rope_theta)
        col_index = torch.arange(t * gh * gw).reshape(t, gh, gw).transpose(1, 2).reshape(-1)
        cu_seqlens = torch.arange(0, t + 1, dtype = torch.int32) * (h * w)
        return {
            "inv_freq": inv_freq.unsqueeze(0),
            "mimo_vit_col_index": col_index,
            "mimo_vit_row_index": torch.argsort(col_index),
            "cu_seqlens": cu_seqlens,
            "max_seqlen": h * w,
            "causal": False,
        }


    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        if "inv_freq" not in params:
            grid = params["grid_thw"]
            assert grid.shape[0] == 1, "MiMo-ViT: one image per forward"
            params.update(self._grid_params(tuple(grid[0].tolist())))
        return input_ids


    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image | list[Image],
        text_alias: str | None = None,
    ):
        if isinstance(image, list):
            assert text_alias is None, "Cannot apply single alias to list of images"
            return [self.get_image_embeddings(tokenizer, i) for i in image]

        image_tensor, prep_image_size, grid_thw = self.preprocess(image)
        params = {"grid_thw": torch.tensor([grid_thw], dtype = torch.int)}
        embedding_tensor = self.forward(image_tensor.unsqueeze(0), params = params).cpu()[0]

        id_start = self.config.vision_start_token_id
        id_end = self.config.vision_end_token_id
        token_string = torch.tensor([[id_start] + [-1] * embedding_tensor.shape[0] + [id_end]], dtype = torch.long)
        mme = MMEmbedding(
            embeddings = embedding_tensor,
            text_alias = text_alias,
            token_string = token_string,
            grid_thw = grid_thw,
        )
        mme.metadata.update({
            "original_size": image.size,
            "preprocessed_size": prep_image_size,
            "model_architecture": self.config.architecture,
        })
        return mme
