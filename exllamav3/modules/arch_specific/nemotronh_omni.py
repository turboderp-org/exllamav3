from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from .. import Module, Linear
from ...model import Config
from ...util.tensor import to2


class NemotronHOmniPatchEmbedder(Module):
    """
    C-RADIO patch generator: linear patch projection (a separate projection for temporally packed
    video tubelets), learned position table resampled to the input grid, and the learned cls/register
    prefix tokens. The position table is interpolated to a max(rows, cols) square and cropped, as in
    the reference, so non-square inputs sample the table anisotropically.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        patch_dim: int,
        video_patch_dim: int,
        table_side: int,
        out_dtype: torch.dtype = torch.float,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.table_side = table_side
        self.out_dtype = out_dtype
        self.position_embedding_key = f"{key}.pos_embed"
        self.prefix_key = f"{key}.cls_token.token"
        self.position_embedding_table = None
        self.prefix_tokens = None
        self.direct_numel = 0

        self.patch_projection = Linear(
            config = config,
            key = f"{key}.embedder",
            in_features = patch_dim,
            out_features = hidden_size,
            qmap = "block",
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.register_submodule(self.patch_projection)
        self.video_projection = Linear(
            config = config,
            key = f"{key}.video_embedder",
            in_features = video_patch_dim,
            out_features = hidden_size,
            qmap = "block",
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.register_submodule(self.video_projection)


    @override
    def optimizer_targets(self):
        return []


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.position_embedding_table = self.config.stc.get_tensor(
            self.position_embedding_key, device, float2half = True, allow_bf16 = True
        )
        self.prefix_tokens = self.config.stc.get_tensor(
            self.prefix_key, device, float2half = True, allow_bf16 = True
        )
        self.direct_numel = self.position_embedding_table.numel() + self.prefix_tokens.numel()


    @override
    def unload(self):
        super().unload()
        self.position_embedding_table = None
        self.prefix_tokens = None
        self.direct_numel = 0


    @override
    def weights_numel(self):
        return super().weights_numel() + self.direct_numel


    @override
    def get_tensors(self):
        if self.position_embedding_table is None:
            return {}
        return {
            self.position_embedding_key: self.position_embedding_table.contiguous(),
            self.prefix_key: self.prefix_tokens.contiguous(),
        }


    def sample_position_embedding(self, rows: int, cols: int) -> torch.Tensor:
        side = self.table_side
        pos = self.position_embedding_table.reshape(1, side, side, self.hidden_size).permute(0, 3, 1, 2).float()
        max_dim = max(rows, cols)
        if max_dim != side:
            pos = F.interpolate(pos, size = (max_dim, max_dim), mode = "bilinear", align_corners = False)
        pos = pos[..., :rows, :cols]
        return pos.flatten(2).transpose(1, 2)


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:

        # x: (bsz, n_patches, patch_dim) flattened as (channel, patch row, patch column) per patch,
        # or (bsz, n_patches, temporal * patch_dim) for video tubelets
        rows, cols = params["grid_hw"]
        projection = self.video_projection if params.get("video", False) else self.patch_projection
        y = projection.forward(x.half(), params, out_dtype = torch.half).float()
        y += self.sample_position_embedding(rows, cols)
        prefix = self.prefix_tokens.unsqueeze(0).expand(y.shape[0], -1, -1).float()
        y = torch.cat((prefix, y), dim = 1)
        return to2(y, out_dtype, self.out_dtype)
