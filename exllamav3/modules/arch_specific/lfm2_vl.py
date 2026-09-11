"""
LFM2-VL multimodal projector module.

Implements the projector described in the LFM2-VL paper / reference impl:

    pixel_unshuffle(factor)
        -> optional LayerNorm
        -> Linear(in * f^2, projector_hidden_size) [bias]
        -> gelu (or other activation)
        -> Linear(projector_hidden_size, text_hidden_size) [bias]

The LFM2-VL model checkpoint stores the projector as:

    model.multi_modal_projector.linear_1.{weight, bias}
    model.multi_modal_projector.linear_2.{weight, bias}

with no explicit LayerNorm tensor when ``projector_use_layernorm`` is False
(the default for LFM2.5-VL).
"""

from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F

from ...model.config import Config
from .. import Module, Linear, LayerNorm


# ---------------------------------------------------------------------------
# Patch embedding + position embedding (fused)
# ---------------------------------------------------------------------------

class Lfm2VlPatchEmbedding(Module):
    """
    Linear patch projection followed by addition of the learned position
    embedding table.

    The LFM2-VL checkpoint stores:

        model.vision_tower.vision_model.embeddings.patch_embedding.{weight, bias}
        model.vision_tower.vision_model.embeddings.position_embedding.weight

    The position table has a fixed size of 256 (one row per 16x16 patch
    cell), and the active rows are simply the first ``N`` where ``N`` is the
    number of patches for the current tile.
    """

    def __init__(
        self,
        config: Config,
        key: str,
        in_features: int,
        hidden_size: int,
        num_positions: int,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, None)  # no quant
        self.module_name = "Lfm2VlPatchEmbedding"

        self.in_features = in_features
        self.hidden_size = hidden_size
        # Source grid is square: sqrt(num_positions) x sqrt(num_positions).
        # For LFM2.5-VL this is 16x16 (256 positions).
        self.num_positions = num_positions
        self.source_grid = int(round(num_positions ** 0.5))
        assert self.source_grid * self.source_grid == num_positions, (
            f"num_positions must be a perfect square, got {num_positions}"
        )
        self.out_dtype = out_dtype

        self.proj = Linear(
            config = config,
            key = f"{key}.patch_embedding",
            in_features = in_features,
            out_features = hidden_size,
            out_dtype = out_dtype or torch.float,
        )
        self.register_submodule(self.proj)

        self.position_weight = None
        self._numel = None

    def optimizer_targets(self):
        raise NotImplementedError()

    def load(self, device, **kwargs):
        self.device = device
        self.proj.load(device, **kwargs)
        # Load the (num_positions, hidden_size) position embedding directly.
        self.position_weight = self.config.stc.get_tensor(
            self.key + ".position_embedding.weight",
            device,
            allow_bf16 = True,
        )
        self._numel = self.proj.weights_numel() + self.position_weight.numel()

    def unload(self):
        self.proj.unload()
        self.position_weight = None
        self._numel = None

    def weights_numel(self):
        return self._numel

    def get_tensors(self):
        t = {}
        t.update(self.proj.get_tensors())
        if self.position_weight is not None:
            t[f"{self.key}.position_embedding.weight"] = self.position_weight
        return t

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        y = self.proj.forward(x, params, out_dtype = torch.float)
        N = y.shape[1]
        grid_h = params.get("grid_h")
        grid_w = params.get("grid_w")
        if grid_h is None or grid_w is None:
            # Backward-compat: assume a perfect-square grid and take the first
            # N rows of the position table (works for 256-patch tiles only).
            pos = self.position_weight[:N].unsqueeze(0)
        else:
            assert grid_h * grid_w == N, (
                f"Lfm2VlPatchEmbedding: grid {grid_h}x{grid_w} = {grid_h * grid_w} "
                f"doesn't match sequence length {N}"
            )
            pos = self._interpolate_position(grid_h, grid_w, y.device, y.dtype)
        y = y + pos.to(y.dtype)
        return y.to(out_dtype or self.out_dtype or torch.float)

    def _interpolate_position(
        self, grid_h: int, grid_w: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Bilinearly interpolate the learned (16, 16, H) position table to
        (grid_h, grid_w, H) and return shape (1, grid_h*grid_w, H)."""
        s = self.source_grid
        h = self.hidden_size
        if grid_h == s and grid_w == s:
            return self.position_weight.unsqueeze(0).to(device = device, dtype = dtype)
        # (S*S, H) -> (S, S, H) -> (1, H, S, S) for interpolate, then back.
        pw = self.position_weight.to(device = device, dtype = torch.float)
        pw = pw.view(s, s, h).permute(2, 0, 1).unsqueeze(0)  # (1, H, S, S)
        # bicubic requires 4D input; bilinear is fine and matches the
        # SigLIP2 NaFlex behaviour closely enough for inference.
        pw = torch.nn.functional.interpolate(
            pw, size = (grid_h, grid_w), mode = "bilinear", align_corners = False,
        )
        pw = pw.squeeze(0).permute(1, 2, 0).reshape(grid_h * grid_w, h)
        return pw.to(dtype = dtype)


class Lfm2VlMultiModalProjector(Module):
    """
    Multimodal projector for LFM2-VL.

    :param downsample_factor:
        Pixel-unshuffle factor. 2 for LFM2.5-VL. The projector takes an input
        of shape ``(B, N, C)`` where N is a perfect square, and reshapes to
        ``(B, N/f^2, C*f^2)`` before the first linear.

    :param use_layernorm:
        When True, an optional LayerNorm over the unshuffled dim is applied
        between the unshuffle and linear_1.

    :param hidden_act:
        Activation between linear_1 and linear_2. ``"gelu"`` corresponds to
        ``F.gelu(approximate='tanh')`` (gelu_pytorch_tanh).
    """

    def __init__(
        self,
        config: Config,
        key: str,
        in_features: int,
        hidden_size: int,
        out_features: int,
        downsample_factor: int = 2,
        use_layernorm: bool = False,
        hidden_act: str = "gelu",
        bias: bool = True,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, None)  # no quant on the projector
        self.module_name = "Lfm2VlMultiModalProjector"

        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.downsample_factor = downsample_factor
        self.use_layernorm = use_layernorm
        self.hidden_act = hidden_act
        self.bias = bias
        self.out_dtype = out_dtype

        unshuffled = in_features * (downsample_factor ** 2)
        self.unshuffled = unshuffled

        self.up = Linear(
            config = config,
            key = f"{key}.linear_1",
            in_features = unshuffled,
            out_features = hidden_size,
            out_dtype = torch.half,
        )
        self.down = Linear(
            config = config,
            key = f"{key}.linear_2",
            in_features = hidden_size,
            out_features = out_features,
            out_dtype = out_dtype or torch.half,
        )
        self.register_submodule(self.up)
        self.register_submodule(self.down)

        if use_layernorm:
            self.norm = LayerNorm(
                config = config,
                key = f"{key}.norm",  # not present in the released checkpoint
                layernorm_eps = 1e-6,
                out_dtype = torch.half,
            )
            self.register_submodule(self.norm)
        else:
            self.norm = None

        self._numel = None

    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def load(self, device, **kwargs):
        self.device = device
        for sub in (self.up, self.down) + ((self.norm,) if self.norm else ()):
            sub.load(device, **kwargs)
        self._numel = (
            self.up.weights_numel()
            + self.down.weights_numel()
            + (self.norm.weights_numel() if self.norm else 0)
        )

    @override
    def unload(self):
        for sub in (self.up, self.down) + ((self.norm,) if self.norm else ()):
            sub.unload()
        self._numel = None

    def weights_numel(self):
        return self._numel

    def get_tensors(self):
        t = {}
        t.update(self.up.get_tensors())
        t.update(self.down.get_tensors())
        if self.norm:
            t.update(self.norm.get_tensors())
        return t

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.hidden_act == "gelu":
            return F.gelu(x, approximate = "tanh")
        if self.hidden_act == "silu":
            return F.silu(x)
        if self.hidden_act == "relu":
            return F.relu(x)
        return F.gelu(x)

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        f = self.downsample_factor
        if f > 1:
            B, N, C = x.shape
            grid_h = params.get("grid_h")
            grid_w = params.get("grid_w")
            if grid_h is not None and grid_w is not None:
                assert grid_h * grid_w == N, (
                    f"Lfm2VlMultiModalProjector: grid {grid_h}x{grid_w} = "
                    f"{grid_h * grid_w} != N={N}"
                )
                # Pad each dim to a multiple of f if needed (shouldn't happen
                # in practice: tile dims are multiples of patch*f, and
                # smart_resize rounds to multiples of patch*f as well).
                pad_h = (f - grid_h % f) % f
                pad_w = (f - grid_w % f) % f
                if pad_h or pad_w:
                    x = torch.nn.functional.pad(
                        x, (0, 0, 0, pad_w, 0, pad_h)
                    )
                    grid_h += pad_h
                    grid_w += pad_w
                    N = grid_h * grid_w
                gh2, gw2 = grid_h // f, grid_w // f
                # (B, gh, gw, C) -> (B, gh/f, f, gw/f, f, C) -> (B, gh/f*gw/f, f*f*C)
                x = x.view(B, grid_h, grid_w, C)
                x = x.view(B, gh2, f, gw2, f, C)
                x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
                x = x.view(B, gh2 * gw2, f * f * C)
            else:
                # Backward-compat: assume square grid.
                assert N % (f * f) == 0, (
                    f"Lfm2VlMultiModalProjector: N={N} not divisible by f^2={f * f}"
                )
                side = int(round(N ** 0.5))
                assert side * side == N, (
                    f"Lfm2VlMultiModalProjector: N={N} is not a perfect square "
                    "and no grid_h/grid_w in params"
                )
                side2 = side // f
                x = x.view(B, side, side, C)
                x = x.view(B, side2, f, side2, f, C)
                x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
                x = x.view(B, side2 * side2, f * f * C)
        if self.norm is not None:
            x = self.norm.forward(x, params).to(torch.half)
        # The upstream layer (post_layernorm) can emit float; the inner
        # hgemm requires half. Cast explicitly.
        x = x.to(torch.half)
        y = self.up.forward(x, params, out_dtype = torch.half)
        y = self._activation(y)
        y = self.down.forward(y, params, out_dtype = out_dtype or self.out_dtype)
        return y
