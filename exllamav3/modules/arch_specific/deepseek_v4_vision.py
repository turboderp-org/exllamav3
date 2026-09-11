from __future__ import annotations
import torch
import torch.nn.functional as F
from typing_extensions import override
from ...model.config import Config
from ...modules import Module, Linear
from ...util.tensor import to2


class DeepseekV4VisionAligner(Module):
    """
    DeepSeek-V4 vision aligner: the ViT grid (n_h, n_w, C) is zero-padded to multiples of the
    downsample ratio r, unfolded r x r (channel-major, matching F.unfold) into C * r * r vectors,
    then projected w1 -> GELU -> w2 into the language model's hidden size. One row per LLM grid
    cell, raster order over the (ceil(n_h / r), ceil(n_w / r)) grid.

    Also owns the four learned image marker vectors (image_start/end/newline/pad) the token block
    is assembled from; they live at the checkpoint root, outside the vision.* namespace.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        key_up: str,
        key_down: str,
        vision_dim: int,
        downsample_ratio: int,
        out_hidden_size: int,
        marker_keys: tuple[str, str, str, str] = ("image_start", "image_pad", "image_newline", "image_end"),
        out_dtype: torch.dtype | None = None,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "DeepseekV4VisionAligner"
        self.vision_dim = vision_dim
        self.r = downsample_ratio
        self.in_size = vision_dim * downsample_ratio ** 2
        self.out_size = out_hidden_size
        self.out_dtype = out_dtype
        self.marker_keys = marker_keys
        self.markers = None

        self.up = Linear(
            config = config,
            key = f"{key}.{key_up}",
            in_features = self.in_size,
            out_features = self.out_size,
            qmap = qmap + ".input" if qmap else None,
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.down = Linear(
            config = config,
            key = f"{key}.{key_down}",
            in_features = self.out_size,
            out_features = self.out_size,
            qmap = qmap + ".down" if qmap else None,
            out_dtype = self.out_dtype,
            pad_to = 1,
        )
        self.register_submodule(self.up)
        self.register_submodule(self.down)

    @override
    def optimizer_targets(self):
        return []

    @override
    def weights_numel(self):
        return self.up.weights_numel() + self.down.weights_numel() + 4 * self.out_size

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        # (4, D) half: image_start, image_pad, image_newline, image_end. Read immediately
        # (no_defer): the stack copies the values, so a deferred read would fill the originals
        # after the copy was taken
        self.markers = torch.stack([
            self.config.stc.get_tensor(k, device, float2half = True, allow_bf16 = True, no_defer = True)
            for k in self.marker_keys
        ]).half()

    @override
    def unload(self):
        super().unload()
        self.markers = None

    @override
    def get_tensors(self):
        t = super().get_tensors()
        if self.markers is not None:
            for i, k in enumerate(self.marker_keys):
                t[k] = self.markers[i].contiguous()
        return t

    def unshuffle(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """(n_h * n_w, C) raster grid -> (ceil(n_h/r) * ceil(n_w/r), C * r * r), channel-major
        within each r x r cell (F.unfold order), zero padding on the bottom/right edges."""
        r = self.r
        C = x.shape[-1]
        g = x.view(n_h, n_w, C)
        ph, pw = -n_h % r, -n_w % r
        if ph or pw:
            g = F.pad(g, (0, 0, 0, pw, 0, ph))
        H, W = n_h + ph, n_w + pw
        g = g.view(H // r, r, W // r, r, C)          # (bh, kh, bw, kw, c)
        g = g.permute(0, 2, 4, 1, 3)                 # (bh, bw, c, kh, kw)
        return g.reshape((H // r) * (W // r), C * r * r)

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        assert bsz == 1, "DeepseekV4VisionAligner: one image per forward"
        n_h, n_w = params["grid_hw"]
        assert n_h * n_w == seqlen, f"grid {n_h}x{n_w} does not match {seqlen} patches"
        y = self.unshuffle(x.view(seqlen, dim).half(), n_h, n_w)
        y = self.up.forward(y, params)
        y = F.gelu(y.float()).half()                 # exact (erf) GELU, as the reference
        y = self.down.forward(y, params)
        y = y.view(1, -1, self.out_size)
        return to2(y, out_dtype, self.out_dtype)
