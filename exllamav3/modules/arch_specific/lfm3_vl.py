from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F

from ...model.config import Config
from .. import Module
from ..layernorm import LayerNorm
from ..linear import Linear
from ...util.tensor import to2


class Lfm3VlVisionConv(Module):
    """
    Gated depthwise 2D convolution over the patch grid of each frame. The patches arrive grouped by merge
    block, so they are reordered onto the full-resolution grid before the convolution and back into
    merge-block order afterwards.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        conv_kernel_size: int,
        spatial_merge_size: int,
        qmap: str | None = "block.attn",
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, qmap)
        self.hidden_size = hidden_size
        self.conv_kernel_size = conv_kernel_size
        self.spatial_merge_size = spatial_merge_size
        self.out_dtype = out_dtype
        self.conv_weight = None
        self.conv_weight_numel = 0

        self.in_proj = Linear(
            config = config,
            key = f"{key}.in_proj",
            in_features = hidden_size,
            out_features = 3 * hidden_size,
            qmap = qmap,
        )
        self.out_proj = Linear(
            config = config,
            key = f"{key}.out_proj",
            in_features = hidden_size,
            out_features = hidden_size,
            qmap = qmap,
        )
        self.register_submodule(self.in_proj)
        self.register_submodule(self.out_proj)

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.conv_weight = self.config.stc.get_tensor(
            f"{self.key}.spatial_conv.weight", device, float2half = True)
        self.conv_weight_numel = self.conv_weight.numel()

    @override
    def unload(self):
        super().unload()
        self.conv_weight = None

    @override
    def get_tensors(self):
        return {f"{self.key}.spatial_conv.weight": self.conv_weight.contiguous()}

    @override
    def weights_numel(self):
        return super().weights_numel() + self.conv_weight_numel

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        b, c, xx = self.in_proj.forward(x, params).chunk(3, dim = -1)
        gated = b * xx
        merge = self.spatial_merge_size
        outputs = []
        offset = 0
        for height, width in params["frame_shapes"]:
            length = height * width
            frame = gated[offset:offset + length]
            offset += length
            frame = frame.view(height // merge, width // merge, merge, merge, -1)
            frame = frame.permute(4, 0, 2, 1, 3).reshape(1, -1, height, width)
            frame = F.conv2d(
                frame,
                self.conv_weight,
                padding = self.conv_kernel_size // 2,
                groups = self.hidden_size,
            )
            frame = frame.view(-1, height // merge, merge, width // merge, merge).permute(1, 3, 2, 4, 0)
            outputs.append(frame.reshape(length, -1))
        conv_output = torch.cat(outputs, dim = 0)
        y = self.out_proj.forward(c * conv_output, params)
        return to2(y, out_dtype, self.out_dtype)


class Lfm3VlProjector(Module):
    """
    LayerNorm over the vision tokens, a merge of each downsample_factor x downsample_factor block into one
    token, then linear_1 -> gelu -> linear_2 into the language model's hidden size.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        in_features: int,
        hidden_size: int,
        out_features: int,
        merge_size: int,
        layernorm_eps: float,
        qmap: str | None = "block",
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, qmap)
        self.in_features = in_features
        self.merge_size = merge_size
        self.out_dtype = out_dtype

        self.layer_norm = LayerNorm(
            config = config,
            key = f"{key}.layer_norm",
            layernorm_eps = layernorm_eps,
        )
        self.linear_1 = Linear(
            config = config,
            key = f"{key}.linear_1",
            in_features = in_features,
            out_features = hidden_size,
            qmap = qmap,
        )
        self.linear_2 = Linear(
            config = config,
            key = f"{key}.linear_2",
            in_features = hidden_size,
            out_features = out_features,
            qmap = qmap,
        )
        self.register_submodule(self.layer_norm)
        self.register_submodule(self.linear_1)
        self.register_submodule(self.linear_2)

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        y = self.layer_norm.forward(x, params)
        y = y.reshape(-1, self.in_features)
        y = self.linear_1.forward(y, params)
        y = F.gelu(y)
        y = self.linear_2.forward(y, params)
        return to2(y, out_dtype, self.out_dtype)
