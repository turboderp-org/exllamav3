from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from ...model.config import Config
from ...modules import Module, RMSNorm
from ...util.tensor import to2

# Hy4 vision tower pieces around the plain ViT blocks. Reference: transformers models/hy_v4_vl
# (HYV4VLVisionPatchEmbed, HYV4VLVisionPatchMerger)


class HyV4VisionPosEmbedding(Module):
    """
    Learned position grid (a class slot followed by edge x edge patch slots), resampled to the image's
    patch grid and added to the patch embeddings. params["grid_hw"]: (rows, columns) of patches
    """

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        position_edge: int,
        interpolate_mode: str = "bilinear",
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.position_edge = position_edge
        self.interpolate_mode = interpolate_mode
        self.out_dtype = out_dtype
        self.grid = None

    @override
    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def weights_numel(self):
        return (self.position_edge ** 2 + 1) * self.hidden_size

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        weight = self.config.stc.get_tensor(f"{self.key}.weight", device, allow_bf16 = True, no_defer = True)
        assert weight.shape == (self.position_edge ** 2 + 1, self.hidden_size), \
            f"{self.key}: unexpected shape {tuple(weight.shape)}"
        # (1, C, edge, edge) fp32, the layout and precision the resampling runs in
        self.grid = weight[1:].float().view(1, self.position_edge, self.position_edge, self.hidden_size) \
            .permute(0, 3, 1, 2).contiguous()

    @override
    def unload(self):
        self.device = None
        self.grid = None

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        h, w = params["grid_hw"]
        # An explicit scale of (size + 0.1) / edge rather than a target size: the sampling grid follows
        # the scale, which shifts it slightly against the one a size alone would give
        pos = F.interpolate(
            self.grid,
            scale_factor = ((h + 0.1) / self.position_edge, (w + 0.1) / self.position_edge),
            mode = self.interpolate_mode,
            align_corners = False,
        )
        assert pos.shape[-2:] == (h, w)
        pos = pos.permute(0, 2, 3, 1).reshape(1, h * w, self.hidden_size)
        x = x.float() + pos
        return to2(x, out_dtype, self.out_dtype)


class HyV4VisionMerger(Module):
    """
    Patch states -> text-width image tokens: RMSNorm, a merge x merge strided convolution, GELU, a 1 x 1
    convolution, then a learned newline token closing every merged row. (1, rows * columns, C) in,
    (1, rows / merge * (columns / merge + 1), out_hidden_size) out. params["grid_hw"] as above
    """

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        out_hidden_size: int,
        merge_size: int,
        rms_norm_eps: float,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.out_hidden_size = out_hidden_size
        self.merge_size = merge_size
        self.out_dtype = out_dtype
        self.norm = RMSNorm(config, f"{key}.norm", rms_norm_eps, out_dtype = torch.half)
        self.register_submodule(self.norm)
        self.w0 = self.b0 = self.w2 = self.b2 = self.newline = None

    @override
    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def weights_numel(self):
        mid = self.hidden_size * self.merge_size ** 2
        return super().weights_numel() + mid * mid + mid + self.out_hidden_size * (mid + 2)

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        def raw(name):
            return self.config.stc.get_tensor(f"{self.key}.{name}", device, allow_bf16 = True, no_defer = True).half()
        self.w0, self.b0 = raw("proj.0.weight"), raw("proj.0.bias")
        self.w2, self.b2 = raw("proj.2.weight"), raw("proj.2.bias")
        self.newline = raw("image_newline")

    @override
    def unload(self):
        super().unload()
        self.w0 = self.b0 = self.w2 = self.b2 = self.newline = None

    @override
    def get_tensors(self):
        return {
            f"{self.key}.proj.0.weight": self.w0.contiguous(),
            f"{self.key}.proj.0.bias": self.b0.contiguous(),
            f"{self.key}.proj.2.weight": self.w2.contiguous(),
            f"{self.key}.proj.2.bias": self.b2.contiguous(),
            f"{self.key}.image_newline": self.newline.contiguous(),
        }

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        h, w = params["grid_hw"]
        m = self.merge_size
        x = self.norm.forward(x, params, out_dtype = torch.half)
        x = x.view(1, h, w, self.hidden_size).permute(0, 3, 1, 2)
        x = F.conv2d(x, self.w0, self.b0, stride = m)
        x = F.gelu(x)
        x = F.conv2d(x, self.w2, self.b2)
        c = x.shape[1]
        x = torch.cat((x, self.newline.view(1, c, 1, 1).expand(1, c, h // m, 1)), dim = -1)
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        return to2(x, out_dtype, self.out_dtype)
