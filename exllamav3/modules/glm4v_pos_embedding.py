from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..model.config import Config
from . import Module
from ..ext import exllamav3_ext as ext
from ..model.model_tp_alloc import TPAllocation
from ..util.tensor import get_for_device, to2

class Glm4VPosEmbedding(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        num_position_embeddings: int,
        hidden_size: int,
        spatial_merge_size: int,
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "Glm4VPosEmbedding"
        self.qmap = qmap
        self.key = key

        self.num_position_embeddings = num_position_embeddings
        self.num_grid_per_side = num_position_embeddings ** 0.5
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size
        self.out_dtype = out_dtype

        self.pos_embed_2d = None


    @override
    def weights_numel(self):
        return self.num_position_embeddings + self.hidden_size


    def optimizer_targets(self):
        raise NotImplementedError()


    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        weight = self.config.stc.get_tensor(self.key + ".weight", self.device, allow_bf16 = True, no_defer = True)
        orig_size_sq = weight.shape[0]
        orig_size = int(orig_size_sq**0.5)
        self.pos_embed_2d = (
            weight.view(orig_size, orig_size, self.hidden_size)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device = device, dtype = torch.float32)
        )

    @override
    def unload(self):
        self.device = None
        self.pos_embed_2d = None


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ):
        grid_thw = get_for_device(params, "grid_thw", self.device)

        # Position IDs
        spm = self.spatial_merge_size
        t, h, w = grid_thw[0].tolist()
        hpos_ids = torch.arange(h, dtype = torch.float).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(h // spm, spm, w // spm, spm)
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()
        wpos_ids = torch.arange(w, dtype = torch.float).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(h // spm, spm, w // spm, spm)
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        h_coords = hpos_ids.to(self.device).float()
        w_coords = wpos_ids.to(self.device).float()
        length = x.shape[1]

        # Normalize coordinates to [-1, 1] range for grid_sample
        norm_w = (w_coords + 0.5) / (w / 2)
        norm_h = (h_coords + 0.5) / (h / 2)
        grid = torch.stack((norm_w, norm_h), dim = -1).unsqueeze(0).unsqueeze(2) - 1

        # Perform bicubic interpolation
        interpolated_embed_fp32 = F.grid_sample(
            self.pos_embed_2d,
            grid,
            mode = "bicubic",
            align_corners = False,
            padding_mode = "border"
        )

        # Reshape and convert back to original dtype
        adapted_pos_embed_fp32 = interpolated_embed_fp32.squeeze(0).squeeze(-1).T

        x += adapted_pos_embed_fp32
        return to2(x, out_dtype, self.out_dtype)
