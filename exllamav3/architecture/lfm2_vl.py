"""
Architecture support for LFM2-VL (LiquidAI / LFM2.5-VL).

Registers ``Lfm2VlForConditionalGeneration`` in the architecture registry.

The model has three components wired through the standard exllamav3 submodel
mechanism (text + vision):

    model.language_model.*              - LFM2 (text decoder)
    model.vision_tower.vision_model.*   - SigLIP2 (vision encoder)
    model.multi_modal_projector.*       - 2-layer MLP with pixel-unshuffle

The text component reuses the existing ``Lfm2Model`` unchanged, only
overriding the key prefix. The vision component implements a SigLIP2
transformer (patch+pos-embed -> N x [LayerNorm -> SelfAttn -> LayerNorm -> MLP]
-> final LayerNorm -> multimodal projector).
"""

from __future__ import annotations
from typing_extensions import override
import json
import math
import os
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image

from ..model.config import Config, no_default
from ..model.model import Model
from ..util.file import read_dict
from ..util.rope import RopeStyle
from ..modules import (
    TransformerBlock,
    Attention,
    MLP,
    LayerNorm,
)
from ..modules.arch_specific.lfm2_vl import (
    Lfm2VlPatchEmbedding,
    Lfm2VlMultiModalProjector,
)
from .lfm2 import Lfm2Model
from ..tokenizer import Tokenizer, MMEmbedding


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Lfm2VlConfig(Config):
    arch_string = "Lfm2VlForConditionalGeneration"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Lfm2VlTextModel, "vision": Lfm2VlVisionModel},
            **kwargs,
        )

        # ---- Text (LFM2) sub-config ----
        # Mirror the field set Lfm2Config reads; in HF's LFM2-VL the LFM2
        # parameters live under text_config and the rope uses rope_parameters.
        self.head_dim = self.read_cfg(
            int, ["text_config->head_dim", "head_dim"], None
        )
        self.hidden_size = self.read_cfg(
            int, ["text_config->hidden_size", "hidden_size"], no_default
        )
        self.num_q_heads = self.read_cfg(
            int,
            ["text_config->num_attention_heads", "num_attention_heads"],
            no_default,
        )
        self.num_kv_heads = self.read_cfg(
            int,
            ["text_config->num_key_value_heads", "num_key_value_heads"],
            self.num_q_heads,
        )
        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_q_heads

        # Short-conv params
        self.conv_kernel_size = self.read_cfg(int, "text_config->conv_L_cache", 3)

        # MLP
        self.assert_cfg(str, "text_config->hidden_act", "silu", True)
        self.intermediate_size = self.read_cfg(
            int, ["text_config->intermediate_size", "intermediate_size"], no_default
        )

        # Norms
        self.rms_norm_eps = self.read_cfg(
            float,
            [
                "text_config->norm_eps",
                "text_config->rms_norm_eps",
                "norm_eps",
                "rms_norm_eps",
            ],
            no_default,
        )

        # Layers
        self.num_hidden_layers = self.read_cfg(
            int, ["text_config->num_hidden_layers", "num_hidden_layers"], no_default
        )
        self.tie_word_embeddings = self.read_cfg(bool, "tie_word_embeddings", False)
        # Per-layer type ("conv" | "full_attention") - Lfm2Model branches on this.
        # In LFM2-VL this lives under text_config->layer_types.
        self.layer_types = self._read_layer_types(self.num_hidden_layers)

        # RoPE for the text decoder
        self.rope_settings = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = self.read_cfg(
                float,
                [
                    "text_config->rope_theta",
                    "text_config->rope_parameters->rope_theta",
                    "rope_theta",
                    "rope_parameters->rope_theta",
                ],
                1000000.0,
            ),
        )

        # ---- Vision (SigLIP2) sub-config ----
        self.vision = read_lfm2_vl_vision_config(
            self.read_cfg(dict, "vision_config", no_default)
        )

        # ---- Multimodal projector ----
        self.projector = read_lfm2_vl_projector_config(self)

        # ---- Preprocessor (for image transforms) ----
        prep_path = os.path.join(self.directory, "processor_config.json")
        if os.path.exists(prep_path):
            with open(prep_path, encoding = "utf8") as f:
                self.vision_pp = read_lfm2_vl_pp_config(json.load(f))
        else:
            prep_path = os.path.join(self.directory, "preprocessor_config.json")
            with open(prep_path, encoding = "utf8") as f:
                self.vision_pp = read_lfm2_vl_pp_config(json.load(f))

        # ---- Special token IDs ----
        # The LFM2-VL tokenizer uses <|image_start|> ... <|image_end|> to
        # surround one image's worth of projected embeddings. We try the
        # explicit config fields first, then fall back to looking up the
        # tokenizer's added-tokens.
        self.image_token_id = self.read_cfg(int, "image_token_id", None)
        self.vision_start_token_id = self.read_cfg(int, "vision_start_token_id", None)
        self.vision_end_token_id = self.read_cfg(int, "vision_end_token_id", None)

    # ------------------------------------------------------------------

    def _read_layer_types(self, num_layers: int) -> list[str]:
        """Read per-layer type list from the LFM2 sub-config.

        Mirrors the helper in ``Lfm2Config`` but reads from the LFM2-VL
        sub-config layout. LFM2.5-VL stores the list at
        ``text_config.layer_types``.
        """
        layer_types = self.read_cfg(list, "text_config->layer_types", None)
        if layer_types is not None:
            assert len(layer_types) == num_layers, (
                "Length of text_config.layer_types doesn't match num_hidden_layers"
            )
            for t in layer_types:
                if t not in ("full_attention", "conv"):
                    raise ValueError(f"Unknown layer type: {t}")
            return layer_types
        # Fall back to the default interleaving used by LFM2-1.2B / 2.6B.
        return [
            "full_attention" if (idx + 1) % 4 == 0 else "conv"
            for idx in range(num_layers)
        ]


def read_lfm2_vl_vision_config(config_dict: dict) -> SimpleNamespace:
    v = SimpleNamespace(**{
        k: read_dict(config_dict, t, k, no_default)
        for k, t in [
            ("hidden_size", int),
            ("intermediate_size", int),
            ("num_hidden_layers", int),
            ("num_attention_heads", int),
            ("num_channels", int),
            ("patch_size", int),
            ("layer_norm_eps", float),
            ("hidden_act", str),
        ]
    })
    v.model_type = read_dict(
        config_dict, str, "model_type", "siglip2_vision_model"
    )
    v.head_dim = v.hidden_size // v.num_attention_heads
    # SigLIP2 uses fixed learned position embeddings; the released 400M
    # variant has 256 (16x16) cells, matching the (256, hidden) weight shape
    # in the checkpoint.
    v.max_patches = 256
    return v


def read_lfm2_vl_projector_config(config_obj: "Lfm2VlConfig") -> SimpleNamespace:
    cfg_dict = config_obj.config_dict
    p = SimpleNamespace(
        downsample_factor = read_dict(cfg_dict, int, "downsample_factor", 2),
        hidden_size = read_dict(
            cfg_dict, int, "projector_hidden_size", config_obj.hidden_size
        ),
        bias = read_dict(cfg_dict, bool, "projector_bias", True),
        hidden_act = read_dict(cfg_dict, str, "projector_hidden_act", "gelu"),
        use_layernorm = read_dict(cfg_dict, bool, "projector_use_layernorm", False),
        in_features = config_obj.vision.hidden_size,
    )
    p.unshuffled = p.in_features * (p.downsample_factor ** 2)
    return p


def read_lfm2_vl_pp_config(config_dict: dict) -> SimpleNamespace:
    # The processor config nests an image_processor block.
    src = config_dict.get("image_processor", config_dict)
    pp = SimpleNamespace(
        image_mean = tuple(src.get("image_mean", (0.5, 0.5, 0.5))),
        image_std = tuple(src.get("image_std", (0.5, 0.5, 0.5))),
        rescale_factor = src.get("rescale_factor", 1.0 / 255),
        do_normalize = src.get("do_normalize", True),
        do_rescale = src.get("do_rescale", True),
        do_resize = src.get("do_resize", True),
        do_image_splitting = src.get("do_image_splitting", True),
        patch_size = src.get("encoder_patch_size", src.get("patch_size", 16)),
        downsample_factor = src.get("downsample_factor", 2),
        min_image_tokens = src.get("min_image_tokens", 64),
        max_image_tokens = src.get("max_image_tokens", 256),
        max_num_patches = src.get("max_num_patches", 1024),
        max_tiles = src.get("max_tiles", 10),
        min_tiles = src.get("min_tiles", 2),
        tile_size = src.get("tile_size", 512),
        use_thumbnail = src.get("use_thumbnail", True),
        use_image_special_tokens = src.get("use_image_special_tokens", True),
        max_pixels_tolerance = src.get("max_pixels_tolerance", 2.0),
        resample = src.get("resample", 3),
        size = src.get("size", {"height": 512, "width": 512}),
    )
    return pp


# ---------------------------------------------------------------------------
# Text component (LFM2 with VL key prefix)
# ---------------------------------------------------------------------------

class Lfm2VlTextModel(Lfm2Model):
    """LFM2 text decoder reused as the language backbone of LFM2-VL.

    All tensors live under ``model.language_model.`` instead of the top-level
    ``model.`` used by the standalone LFM2 model, so we override the key
    prefix and leave everything else intact.
    """
    config_class = Lfm2VlConfig

    def __init__(
        self,
        config: Lfm2VlConfig,
        **kwargs,
    ):
        super().__init__(
            config,
            key_prefix = "model.language_model",
            **kwargs,
        )
        # The inherited Embedding module sets `prefer_cpu: True` so its
        # lookup table is CPU-pinned and the output is uploaded per-step.
        # That conflicts with the multimodal splice, which assumes the
        # embedding buffer and the MM embeddings share a device. Override
        # the cap on this specific instance to keep the whole model on
        # the same device the vision tower uses.
        if self.modules:
            self.modules[0].caps["prefer_cpu"] = False


# ---------------------------------------------------------------------------
# Vision component (SigLIP2 + multimodal projector)
# ---------------------------------------------------------------------------

class Lfm2VlVisionModel(Model):
    """SigLIP2 vision tower + multimodal projector for LFM2-VL.

    Module order (forward = sequential application):

        1. ``Lfm2VlPatchEmbedding``     (patch linear + pos-embed add)
        2. ``num_hidden_layers`` x ``TransformerBlock``
               - LayerNorm -> SelfAttn (separate QKV) -> LayerNorm -> MLP (gelu)
        3. ``LayerNorm``               (post encoder)
        4. ``Lfm2VlMultiModalProjector`` (pixel-unshuffle + 2x Linear + gelu)
    """

    def __init__(
        self,
        config: Lfm2VlConfig,
        key_prefix: str = "model.vision_tower.vision_model",
        projector_prefix: str = "model.multi_modal_projector",
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.caps.update({
            "image_input": True,
            "default_vision_bits": 6,
        })

        v = self.config.vision
        p = self.config.projector

        # 1) Patch + position embedding
        in_dim = v.num_channels * v.patch_size * v.patch_size
        self.modules += [
            Lfm2VlPatchEmbedding(
                config = config,
                key = f"{key_prefix}.embeddings",
                in_features = in_dim,
                hidden_size = v.hidden_size,
                num_positions = v.max_patches,
                out_dtype = torch.float,
            )
        ]

        # 2) Encoder blocks
        for idx in range(v.num_hidden_layers):
            self.modules += [
                TransformerBlock(
                    config = config,
                    key = f"{key_prefix}.encoder.layers.{idx}",
                    layer_idx = idx,
                    attn_norm = LayerNorm(
                        config = config,
                        key = f"{key_prefix}.encoder.layers.{idx}.layer_norm1",
                        layernorm_eps = v.layer_norm_eps,
                        out_dtype = torch.float,
                    ),
                    attn = Attention(
                        config = config,
                        key = f"{key_prefix}.encoder.layers.{idx}.self_attn",
                        layer_idx = idx,
                        hidden_size = v.hidden_size,
                        head_dim = v.head_dim,
                        num_q_heads = v.num_attention_heads,
                        num_kv_heads = v.num_attention_heads,
                        rope_settings = None,  # SigLIP2 uses learned pos, not RoPE
                        sm_scale = None,
                        key_q = "q_proj",
                        key_k = "k_proj",
                        key_v = "v_proj",
                        key_o = "out_proj",
                        qmap = "block.attn",
                        out_dtype = torch.float,
                    ),
                    mlp_norm = LayerNorm(
                        config = config,
                        key = f"{key_prefix}.encoder.layers.{idx}.layer_norm2",
                        layernorm_eps = v.layer_norm_eps,
                        out_dtype = torch.float,
                    ),
                    mlp = MLP(
                        config = config,
                        key = f"{key_prefix}.encoder.layers.{idx}.mlp",
                        hidden_size = v.hidden_size,
                        intermediate_size = v.intermediate_size,
                        key_up = "fc1",
                        key_down = "fc2",
                        activation_fn = "gelu",  # gelu_pytorch_tanh
                        qmap = "block.mlp",
                        out_dtype = torch.float,
                    ),
                )
            ]

        # 3) Post encoder LayerNorm
        self.modules += [
            LayerNorm(
                config = config,
                key = f"{key_prefix}.post_layernorm",
                layernorm_eps = v.layer_norm_eps,
                out_dtype = torch.float,
            )
        ]

        # 4) Multimodal projector
        self.modules += [
            Lfm2VlMultiModalProjector(
                config = config,
                key = projector_prefix,
                in_features = p.in_features,
                hidden_size = p.hidden_size,
                out_features = config.hidden_size,
                downsample_factor = p.downsample_factor,
                use_layernorm = p.use_layernorm,
                hidden_act = p.hidden_act,
                bias = p.bias,
                out_dtype = torch.float,
            )
        ]

        self.caps.update({
            "supports_tp": False,
        })

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    # Mirrors the reference LFM2-VL processor (processing_lfm2_vl.py):

    #   1. _is_img_too_large      - is the image bigger than max_image_tokens
    #                                 permits for a single tile?
    #   2. _high_res_preprocessor - find an (n_cols, n_rows) grid in
    #                                 [min_tiles, max_tiles] that best matches
    #                                 the aspect ratio, resize the image to
    #                                 (n_cols*tile_size, n_rows*tile_size),
    #                                 crop into tiles of (tile_size, tile_size).
    #   3. _smart_resize          - resize a single image so its dims are
    #                                 multiples of patch*downsample and the
    #                                 total patch count sits in
    #                                 [min_image_tokens, max_image_tokens].
    #   4. Thumbnail               - when there are >1 tiles, also append a
    #                                 thumbnail of the original image, sized
    #                                 via smart_resize with [min, min] tokens.
    #   5. Per-tile normalize     - rescale + normalize + flatten into patches
    #                                 and run the vision tower + projector.

    def _is_img_too_large(
        self,
        image: Image.Image,
        max_image_tokens: int | None = None,
        encoder_patch_size: int | None = None,
        max_pixels_tolerance: float | None = None,
    ) -> bool:
        pp = self.config.vision_pp
        p = self.config.projector
        max_image_tokens = max_image_tokens or pp.max_image_tokens
        encoder_patch_size = encoder_patch_size or pp.patch_size
        max_pixels_tolerance = max_pixels_tolerance or pp.max_pixels_tolerance
        w, h = image.size
        h_bar = max(encoder_patch_size, self._round_by_factor(h, encoder_patch_size))
        w_bar = max(encoder_patch_size, self._round_by_factor(w, encoder_patch_size))
        return h_bar * w_bar > (
            max_image_tokens
            * encoder_patch_size ** 2
            * p.downsample_factor ** 2
            * max_pixels_tolerance
        )

    def _high_res_preprocessor(
        self,
        image: Image.Image,
        min_tiles: int,
        max_tiles: int,
        tile_size: int,
    ) -> tuple[list[Image.Image], int, int]:
        """Split ``image`` into a grid of ``tile_size`` x ``tile_size`` tiles.

        Returns (tiles, num_rows, num_cols). The grid is chosen to match the
        image's aspect ratio as closely as possible while keeping the
        number of tiles in [min_tiles, max_tiles].
        """
        orig_w, orig_h = image.size
        aspect = orig_w / orig_h
        target_ratios = sorted({
            (w, h)
            for n in range(min_tiles, max_tiles + 1)
            for w in range(1, n + 1)
            for h in range(1, n + 1)
            if min_tiles <= w * h <= max_tiles
        }, key = lambda r: r[0] * r[1])
        if not target_ratios:
            return [], 0, 0
        grid_w, grid_h = _find_closest_aspect_ratio(
            aspect, target_ratios, orig_w, orig_h, tile_size
        )
        target_w = tile_size * grid_w
        target_h = tile_size * grid_h
        total = grid_w * grid_h
        resized = image.resize((target_w, target_h))
        tiles: list[Image.Image] = []
        for i in range(total):
            col = i % grid_w
            row = i // grid_w
            box = (
                col * tile_size,
                row * tile_size,
                (col + 1) * tile_size,
                (row + 1) * tile_size,
            )
            tiles.append(resized.crop(box))
        return tiles, grid_h, grid_w

    def _smart_resize(
        self,
        image: Image.Image,
        downsample_factor: int | None = None,
        min_image_tokens: int | None = None,
        max_image_tokens: int | None = None,
        encoder_patch_size: int | None = None,
    ) -> Image.Image:
        pp = self.config.vision_pp
        p = self.config.projector
        downsample_factor = downsample_factor or p.downsample_factor
        min_image_tokens = min_image_tokens or pp.min_image_tokens
        max_image_tokens = max_image_tokens or pp.max_image_tokens
        encoder_patch_size = encoder_patch_size or pp.patch_size
        total_factor = encoder_patch_size * downsample_factor
        min_pixels = min_image_tokens * encoder_patch_size ** 2 * downsample_factor ** 2
        max_pixels = max_image_tokens * encoder_patch_size ** 2 * downsample_factor ** 2
        w, h = image.size
        h_bar = max(total_factor, self._round_by_factor(h, total_factor))
        w_bar = max(total_factor, self._round_by_factor(w, total_factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((h * w) / max_pixels)
            h_bar = max(total_factor, self._floor_by_factor(h / beta, total_factor))
            w_bar = max(total_factor, self._floor_by_factor(w / beta, total_factor))
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (h * w))
            h_bar = self._ceil_by_factor(h * beta, total_factor)
            w_bar = self._ceil_by_factor(w * beta, total_factor)
        return image.resize((w_bar, h_bar))

    def _get_tokens_num(self, image_height: int, image_width: int) -> int:
        pp = self.config.vision_pp
        p = self.config.projector
        n_ph = image_height // pp.patch_size
        n_pw = image_width // pp.patch_size
        dh = math.ceil(n_ph / p.downsample_factor)
        dw = math.ceil(n_pw / p.downsample_factor)
        return dh * dw

    @staticmethod
    def _round_by_factor(n: float, f: int) -> int:
        return round(n / f) * f

    @staticmethod
    def _ceil_by_factor(n: float, f: int) -> int:
        return math.ceil(n / f) * f

    @staticmethod
    def _floor_by_factor(n: float, f: int) -> int:
        return math.floor(n / f) * f

    # ------------------------------------------------------------------
    # Per-tile normalization
    # ------------------------------------------------------------------

    def _preprocess_tile(self, image: Image.Image) -> torch.Tensor:
        """Convert a (H, W) PIL image into a flat (num_patches, C*P*P) tensor.

        Mirrors the per-tile steps of ``Siglip2ImageProcessor``: rescale
        to ``[0, 1]``, optionally normalize with the configured mean/std,
        then reshape to flat patches. We do not resize the image here; the
        caller is expected to have already produced a tile whose dimensions
        are multiples of ``patch_size``.
        """
        pp = self.config.vision_pp
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.array(image).astype(np.float32)
        if pp.do_rescale:
            arr = arr * pp.rescale_factor
        if pp.do_normalize:
            mean = np.array(pp.image_mean, dtype = np.float32)
            std = np.array(pp.image_std, dtype = np.float32)
            arr = (arr - mean) / std
        ps = pp.patch_size
        h, w = arr.shape[:2]
        assert h % ps == 0 and w % ps == 0, (
            f"Tile size ({h}, {w}) is not a multiple of patch_size {ps}; "
            "caller must smart_resize or split first."
        )
        n_ph = h // ps
        n_pw = w // ps
        arr = arr.transpose(2, 0, 1)
        arr = arr.reshape(3, n_ph, ps, n_pw, ps)
        arr = arr.transpose(1, 3, 2, 4, 0)
        arr = arr.reshape(n_ph * n_pw, ps * ps * 3)
        return torch.from_numpy(arr.copy()).half()

    def _resize_and_maybe_split(
        self,
        image: Image.Image,
    ) -> tuple[list[Image.Image], int, int, int, int]:
        """Run the full resize/split pipeline on a single image.

        Returns
        -------
        tiles : list[Image.Image]
            PIL tiles (already in their final sizes).
        num_tokens_per_tile : int
            Number of image-token placeholders per tile.
        num_rows, num_cols : int
            Grid dimensions used for the tile list (excluding thumbnail).
        num_thumbnail_tokens : int
            Number of image-token placeholders for the thumbnail (0 if
            no thumbnail is added).
        """
        pp = self.config.vision_pp
        do_splitting = (
            not (pp.min_tiles == pp.max_tiles == 1) and pp.do_image_splitting
        )
        if do_splitting and self._is_img_too_large(image):
            tiles, num_rows, num_cols = self._high_res_preprocessor(
                image, pp.min_tiles, pp.max_tiles, pp.tile_size
            )
            if len(tiles) > 1:
                num_thumb_tokens = 0
                if pp.use_thumbnail:
                    thumb = self._smart_resize(
                        image,
                        min_image_tokens = pp.min_image_tokens,
                        max_image_tokens = pp.min_image_tokens,
                    )
                    num_thumb_tokens = self._get_tokens_num(thumb.height, thumb.width)
                    tiles.append(thumb)
                return (
                    tiles,
                    self._get_tokens_num(pp.tile_size, pp.tile_size),
                    num_rows,
                    num_cols,
                    num_thumb_tokens,
                )
        resized = self._smart_resize(image)
        return [resized], self._get_tokens_num(resized.height, resized.width), 1, 1, 0

    # ------------------------------------------------------------------
    # Inference entry points
    # ------------------------------------------------------------------

    def default_load_shape_dtype(self, chunk_size):
        v = self.config.vision
        return (
            (
                1,
                v.max_patches,
                v.num_channels * v.patch_size * v.patch_size,
            ),
            torch.half,
        )

    def default_load_params(self, max_chunk_size):
        return {}

    def _run_vision(self, tile: Image.Image) -> torch.Tensor:
        """Preprocess + run the vision tower + projector on a single tile.

        Returns a ``(num_tokens, text_hidden)`` tensor on the same device as
        the vision model.
        """
        patches = self._preprocess_tile(tile)
        x = patches.unsqueeze(0)
        pp = self.config.vision_pp
        grid_h = tile.height // pp.patch_size
        grid_w = tile.width // pp.patch_size
        params = {"causal": False, "grid_h": grid_h, "grid_w": grid_w}
        with torch.inference_mode():
            hidden = self.forward(x, params = params)
        return hidden[0]

    def get_image_embeddings(
        self,
        tokenizer: Tokenizer,
        image: Image.Image | list[Image.Image],
        text_alias: str | None = None,
    ):
        """Build ``MMEmbedding`` objects for ``image``.

        Each input image becomes one ``MMEmbedding`` whose ``token_string``
        encodes the full multi-tile sequence:

            [image_start]? (img_row_R_col_C? + image*num_tokens_per_tile)+
            (img_thumbnail? + image*num_thumbnail_tokens)?  [image_end]?

        with ``-1`` placeholders wherever an image embedding goes.

        Returns a single ``MMEmbedding`` for a single image (to match the
        exllamav3 contract used by TabbyAPI's ``get_image_embedding_exl3``)
        and a list of ``MMEmbedding`` for a list of images.
        """
        if isinstance(image, Image.Image):
            return self._build_image_mme(tokenizer, image, text_alias = text_alias)
        return [
            self._build_image_mme(tokenizer, img, text_alias = text_alias)
            for img in image
        ]

    def _build_image_mme(
        self,
        tokenizer: Tokenizer,
        image: Image.Image,
        text_alias: str | None = None,
    ) -> MMEmbedding:
        pp = self.config.vision_pp
        tiles, num_tokens_per_tile, num_rows, num_cols, num_thumb_tokens = (
            self._resize_and_maybe_split(image)
        )

        # Compute per-tile patch grids (used for grid_thw metadata).
        per_tile_grid: list[tuple[int, int]] = []
        for idx, tile in enumerate(tiles):
            is_thumb = (idx == len(tiles) - 1) and num_thumb_tokens > 0
            h_p = math.ceil(tile.height / pp.patch_size)
            w_p = math.ceil(tile.width / pp.patch_size)
            per_tile_grid.append((h_p, w_p))

        # Run vision + projector per tile.
        # The LFM2-VL text Embedding is kept on GPU (prefer_cpu disabled
        # in Lfm2VlTextModel.__init__), so MM embeddings stay on GPU too.
        tile_outputs: list[torch.Tensor] = [
            self._run_vision(t) for t in tiles
        ]

        use_special = bool(
            getattr(pp, "use_image_special_tokens", True)
        )
        id_image = _resolve_image_token_id(tokenizer, self.config)
        id_start = _resolve_special_id(
            tokenizer, self.config, "<|image_start|>", "vision_start_token_id"
        )
        id_end = _resolve_special_id(
            tokenizer, self.config, "<|image_end|>", "vision_end_token_id"
        )
        id_thumb = _resolve_special_id(
            tokenizer, self.config,
            "<|img_thumbnail|>", "image_thumbnail_token_id",
            required = False,
        )

        token_ids: list[int] = []
        if use_special and id_start is not None:
            token_ids.append(id_start)
        if num_rows * num_cols == 1:
            token_ids.extend([-1] * num_tokens_per_tile)
        else:
            for row in range(num_rows):
                for col in range(num_cols):
                    if use_special:
                        rc = f"<|img_row_{row + 1}_col_{col + 1}|>"
                        rc_id = _resolve_special_id(
                            tokenizer, self.config, rc, "row_col_token_id",
                            required = False,
                        )
                        if rc_id is not None:
                            token_ids.append(rc_id)
                    token_ids.extend([-1] * num_tokens_per_tile)
            if num_thumb_tokens > 0:
                if use_special and id_thumb is not None:
                    token_ids.append(id_thumb)
                token_ids.extend([-1] * num_thumb_tokens)
        if use_special and id_end is not None:
            token_ids.append(id_end)

        token_string = torch.tensor([token_ids], dtype = torch.long)
        all_emb = torch.cat(tile_outputs, dim = 0)
        num_mm = sum(1 for t in token_ids if t == -1)
        assert num_mm == all_emb.shape[0], (
            f"MM token count mismatch: token_string has {num_mm} -1 slots, "
            f"vision produced {all_emb.shape[0]} embeddings "
            f"(tiles={num_rows * num_cols}, thumb={num_thumb_tokens})"
        )

        mme = MMEmbedding(
            embeddings = all_emb,
            text_alias = text_alias,
            token_string = token_string,
            grid_thw = (
                1,
                sum(g[0] for g in per_tile_grid) if len(per_tile_grid) > 1 else per_tile_grid[0][0],
                sum(g[1] for g in per_tile_grid) if len(per_tile_grid) > 1 else per_tile_grid[0][1],
            ),
            mrope_merge_size = self.config.projector.downsample_factor,
        )
        mme.metadata.update({
            "model_architecture": self.config.architecture,
            "lfm2_vl_num_rows": num_rows,
            "lfm2_vl_num_cols": num_cols,
            "lfm2_vl_num_thumbnail_tokens": num_thumb_tokens,
            "lfm2_vl_tokens_per_tile": num_tokens_per_tile,
            "lfm2_vl_per_tile_grid": per_tile_grid,
        })
        return mme

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # Vision "tokens" are image patches; identity passthrough.
        return input_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Pick the (w, h) tile grid whose aspect ratio is closest to the image's.

    Mirrors the helper in qwen2.5-vl / LFM2-VL processing: ties on aspect
    error are broken in favour of the ratio that better preserves the image
    area.
    """
    best_diff = float("inf")
    best: tuple[int, int] = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_aspect)
        if diff < best_diff:
            best_diff = diff
            best = ratio
        elif diff == best_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best = ratio
    return best


def _resolve_image_token_id(tokenizer: Tokenizer, config) -> int:
    if config.image_token_id is not None:
        return int(config.image_token_id)
    if hasattr(tokenizer, "get_token_id"):
        tid = tokenizer.get_token_id("<image>")
        if tid is not None:
            return int(tid)
    if hasattr(tokenizer, "image_token_id"):
        return int(tokenizer.image_token_id)
    raise ValueError(
        "LFM2-VL: could not resolve the <image> token id. Either set "
        "image_token_id in config.json or load the tokenizer before "
        "calling get_image_embeddings."
    )


def _resolve_special_id(
    tokenizer: Tokenizer,
    config,
    token: str,
    config_field: str,
    *,
    required: bool = False,
) -> int | None:
    """Look up a special token id in this order: config field, then tokenizer.

    The tokenizer may be either an exllamav3 ``Tokenizer`` (with
    ``get_token_id``) or a transformers-style tokenizer (with
    ``convert_tokens_to_ids``). We try both.
    """
    cfg_val = getattr(config, config_field, None)
    if cfg_val is not None:
        return int(cfg_val)
    if hasattr(tokenizer, "get_token_id"):
        tid = tokenizer.get_token_id(token)
        if tid is not None:
            return int(tid)
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and getattr(tokenizer, "unk_token_id", None) != tid:
            return int(tid)
    if required:
        raise ValueError(
            f"LFM2-VL: could not resolve token {token!r}; pass a tokenizer "
            "with this token registered, or set the corresponding field "
            f"in config.json ({config_field})."
        )
    return None


# ---------------------------------------------------------------------------
# Public helper for chat-template integration
# ---------------------------------------------------------------------------

def expand_image_placeholders(
    tokenizer: Tokenizer,
    text: str,
    images: list[Image.Image],
    vision_model: "Lfm2VlVisionModel | None" = None,
) -> str:
    """Expand ``<image>`` placeholders in ``text`` into the full multi-tile
    sequence used by LFM2-VL.

    Parameters
    ----------
    tokenizer
        The loaded tokenizer (must contain ``<|image_start|>``,
        ``<|image_end|>``, ``<|img_thumbnail|>``, ``<image>``).
    text
        Input prompt containing one ``<image>`` placeholder per image.
    images
        List of PIL images, one per placeholder.
    vision_model
        Required, used to compute per-tile token counts.
    """
    if text.count("<image>") != len(images):
        raise ValueError(
            f"expand_image_placeholders: {text.count('<image>')} <image> "
            f"tokens in text but {len(images)} images passed"
        )
    if vision_model is None:
        raise ValueError(
            "expand_image_placeholders requires a vision_model to compute "
            "per-tile token counts"
        )

    out_parts: list[str] = []
    split = text.split("<image>")
    for idx, image in enumerate(images):
        out_parts.append(split[idx])
        (
            _tiles, num_tokens_per_tile, num_rows, num_cols, num_thumb_tokens,
        ) = vision_model._resize_and_maybe_split(image)
        out_parts.append("<|image_start|>")
        if num_rows * num_cols == 1:
            out_parts.append("<image>" * num_tokens_per_tile)
        else:
            for row in range(num_rows):
                for col in range(num_cols):
                    out_parts.append(f"<|img_row_{row + 1}_col_{col + 1}|>")
                    out_parts.append("<image>" * num_tokens_per_tile)
            if num_thumb_tokens > 0:
                out_parts.append("<|img_thumbnail|>")
                out_parts.append("<image>" * num_thumb_tokens)
        out_parts.append("<|image_end|>")
    out_parts.append(split[-1])
    return "".join(out_parts)
