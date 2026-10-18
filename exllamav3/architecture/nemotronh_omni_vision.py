from __future__ import annotations

import math
from typing_extensions import override

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..model.config import Config
from ..model.model import Model
from ..modules import Attention, LayerNorm, Linear, MLP, Module, RMSNorm, TransformerBlock
from ..tokenizer import MMEmbedding, Tokenizer


class RadioPatchEmbedding(Module):
    """C-RADIO patch projection, interpolated position table, and learned prefix tokens."""

    def __init__(self, config: Config, key: str, vision_config):
        super().__init__(config, key, None)
        self.vision_config = vision_config
        patch_dim = vision_config.num_channels * vision_config.patch_size ** 2
        self.patch_projection = Linear(
            config = config,
            key = f"{key}.embedder",
            in_features = patch_dim,
            out_features = vision_config.hidden_size,
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.video_projection = Linear(
            config = config,
            key = f"{key}.video_embedder",
            in_features = vision_config.video_temporal_patch_size * patch_dim,
            out_features = vision_config.hidden_size,
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.modules = [self.patch_projection, self.video_projection]
        self.position_embedding = None
        self.cls_register_token = None

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.position_embedding = self.config.stc.get_tensor(
            f"{self.key}.pos_embed", device, float2half = True, allow_bf16 = True
        )
        self.cls_register_token = self.config.stc.get_tensor(
            f"{self.key}.cls_token.token", device, float2half = True, allow_bf16 = True
        )

    @override
    def unload(self):
        super().unload()
        self.position_embedding = None
        self.cls_register_token = None

    @override
    def get_tensors(self):
        if self.position_embedding is None:
            return {}
        return {
            f"{self.key}.pos_embed": self.position_embedding.contiguous(),
            f"{self.key}.cls_token.token": self.cls_register_token.contiguous(),
        }

    @override
    def weights_numel(self):
        v = self.vision_config
        direct_numel = (
            (v.max_img_size // v.patch_size) ** 2 * v.hidden_size +
            (v.num_cls_tokens + v.num_registers) * v.hidden_size
        )
        return super().weights_numel() + direct_numel

    @override
    def forward(self, pixel_values: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        pixel_values = self.prepare_for_device(pixel_values, params).half()
        v = self.vision_config
        bsz, channels, height, width = pixel_values.shape
        ps = v.patch_size
        rows, cols = height // ps, width // ps

        # RADIO's checkpoint flattens each patch as C, patch-row, patch-column.
        patches = pixel_values.unfold(2, ps, ps).unfold(3, ps, ps)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(bsz, rows * cols, -1)
        use_video_projection = params.get("use_video_patch_projection", False)
        projection = self.video_projection if use_video_projection else self.patch_projection
        patches = projection.forward(patches, params, out_dtype = torch.half)

        max_rows = v.max_img_size // ps
        max_cols = v.max_img_size // ps
        pos = self.position_embedding.reshape(1, max_rows, max_cols, v.hidden_size)
        pos = pos.permute(0, 3, 1, 2).float()
        max_dim = max(rows, cols)
        pos = F.interpolate(pos, size = (max_dim, max_dim), mode = "bilinear", align_corners = False)
        pos = pos[..., :rows, :cols]
        if pos.shape[-2:] != (rows, cols):
            pos = F.interpolate(pos, size = (rows, cols), mode = "bilinear", align_corners = False)
        pos = pos.flatten(2).transpose(1, 2).to(patches.dtype)
        patches = patches + pos

        prefix = self.cls_register_token.unsqueeze(0).expand(bsz, -1, -1).to(patches.dtype)
        return torch.cat((prefix, patches), dim = 1)


class RadioPatchShuffle(Module):
    """Drop RADIO's cls/register tokens and pixel-shuffle spatial features by 2x2."""

    def __init__(self, config: Config, key: str, num_prefix_tokens: int, ratio: float, version: str):
        super().__init__(config, key, None)
        self.num_prefix_tokens = num_prefix_tokens
        self.ratio = ratio
        self.version = version

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        rows, cols = params["grid_hw"]
        x = x[:, self.num_prefix_tokens:, :].reshape(x.shape[0], rows, cols, x.shape[-1])
        scale = self.ratio
        n, h, w, c = x.shape
        x = x.reshape(n, h, int(w * scale), int(c / scale))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(n, int(w * scale), int(h * scale), int(c / (scale * scale)))
        if self.version != "v1":
            x = x.permute(0, 2, 1, 3).contiguous()
        return x.reshape(n, -1, x.shape[-1])


class NemotronHOmniVisionModel(Model):
    """C-RADIO vision tower and NemotronH Omni projector for image and video inputs."""

    config_class = None

    @staticmethod
    @override
    def get_additional_compiled_tensors(config):
        tensors = config.stc.list_tensors(prefix = "vision_model.")
        tensors |= config.stc.list_tensors(prefix = "mlp1.")
        return tensors

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.caps.update({"image_input": True, "video_input": True, "default_vision_bits": 16})
        v = config.vision
        key = "vision_model.radio_model.model"
        self.patch_size = v.patch_size
        self.modules.append(RadioPatchEmbedding(config, f"{key}.patch_generator", v))

        for idx in range(v.num_hidden_layers):
            layer_key = f"{key}.blocks.{idx}"
            self.modules.append(TransformerBlock(
                config = config,
                key = layer_key,
                layer_idx = idx,
                attn_norm = LayerNorm(
                    config = config,
                    key = f"{layer_key}.norm1",
                    layernorm_eps = v.layer_norm_eps,
                    out_dtype = torch.half,
                ),
                attn = Attention(
                    config = config,
                    key = f"{layer_key}.attn",
                    layer_idx = idx,
                    hidden_size = v.hidden_size,
                    head_dim = v.hidden_size // v.num_attention_heads,
                    num_q_heads = v.num_attention_heads,
                    num_kv_heads = v.num_attention_heads,
                    rope_settings = None,
                    sm_scale = None,
                    key_fused_qkv = "qkv",
                    key_o = "proj",
                    qmap = "block.attn",
                    out_dtype = torch.half,
                ),
                mlp_norm = LayerNorm(
                    config = config,
                    key = f"{layer_key}.norm2",
                    layernorm_eps = v.layer_norm_eps,
                    out_dtype = torch.half,
                ),
                mlp = MLP(
                    config = config,
                    key = f"{layer_key}.mlp",
                    hidden_size = v.hidden_size,
                    intermediate_size = int(v.hidden_size * v.mlp_ratio),
                    key_up = "fc1",
                    key_down = "fc2",
                    activation_fn = v.hidden_act,
                    qmap = "block.mlp",
                    out_dtype = torch.half,
                    pad_to = 1,
                ),
            ))

        self.modules += [
            RadioPatchShuffle(
                config, "vision_projector.pixel_shuffle", v.num_cls_tokens + v.num_registers,
                config.downsample_ratio, config.ps_version,
            ),
            RMSNorm(
                config = config,
                key = "mlp1.0",
                rms_norm_eps = 1e-5,
                out_dtype = torch.half,
            ),
            MLP(
                config = config,
                key = "mlp1",
                hidden_size = v.hidden_size * int(1 / config.downsample_ratio) ** 2,
                intermediate_size = config.projector_hidden_size,
                key_up = "1",
                key_down = "3",
                activation_fn = "relu2",
                out_size = config.hidden_size,
                qmap = "block.mlp",
                out_dtype = torch.half,
                pad_to = 1,
            ),
        ]

    def default_load_shape_dtype(self, chunk_size):
        return (1, 3, 512, 512), torch.half

    def default_load_params(self, max_chunk_size):
        return {"grid_hw": (32, 32), "causal": False}

    @override
    def prepare_inputs(self, pixel_values: torch.Tensor, params: dict) -> torch.Tensor:
        params["grid_hw"] = (pixel_values.shape[-2] // self.patch_size, pixel_values.shape[-1] // self.patch_size)
        return pixel_values

    def _preprocess(self, image: Image.Image, video: bool = False):
        pp = self.config.vision_pp
        image = image.convert("RGB")
        orig_w, orig_h = image.size
        divisor = int(round(1.0 / self.config.downsample_ratio))
        if video:
            target = pp.video_target_num_patches
            aspect = orig_w / max(orig_h, 1)
            rows = max(round(math.sqrt(target / aspect)), 1)
            cols = max(round(math.sqrt(target * aspect)), 1)
            rows_up = rows + (divisor - rows % divisor if rows % divisor else 0)
            cols_up = cols + (divisor - cols % divisor if cols % divisor else 0)
            if rows_up * cols_up <= target:
                rows, cols = rows_up, cols_up
            else:
                rows = max(divisor, rows - rows % divisor)
                cols = max(divisor, cols - cols % divisor)
        else:
            budget = (pp.max_model_len - 4) * divisor ** 2
            budget = min(max(budget, pp.min_num_patches), pp.max_num_patches)
            closest_h = round(orig_h / pp.patch_size + 0.5)
            closest_w = round(orig_w / pp.patch_size + 0.5)
            factor = min(math.sqrt(budget / (closest_h * closest_w)), 1.0)
            rows = math.floor(factor * closest_h)
            cols = math.floor(factor * closest_w)
            if budget > pp.min_num_patches and rows * cols < pp.min_num_patches:
                up = math.sqrt(pp.min_num_patches / max(rows * cols, 1))
                rows, cols = math.ceil(up * rows), math.ceil(up * cols)
            if rows % divisor:
                inc = divisor - rows % divisor
                rows = rows + inc if (rows + inc) * cols <= budget else max(divisor, rows - rows % divisor)
            if cols % divisor:
                inc = divisor - cols % divisor
                cols = cols + inc if rows * (cols + inc) <= budget else max(divisor, cols - cols % divisor)

        target_h, target_w = rows * pp.patch_size, cols * pp.patch_size
        arr = np.asarray(image, dtype = np.uint8)
        pixels = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        if pixels.shape[-2:] != (target_h, target_w):
            pixels = F.interpolate(
                pixels, size = (target_h, target_w), mode = "bicubic", align_corners = False, antialias = True
            )
        mean = torch.tensor(pp.norm_mean).view(1, 3, 1, 1)
        std = torch.tensor(pp.norm_std).view(1, 3, 1, 1)
        pixels = (pixels / 255.0 - mean) / std
        return pixels[0].half(), (target_w, target_h), (rows, cols)

    @torch.inference_mode()
    def _encode(self, pixels: torch.Tensor, grid_hw: tuple[int, int], video: bool = False):
        params = {"grid_hw": grid_hw, "use_video_patch_projection": video, "causal": False}
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
        return self.forward(pixels, params = params).cpu()

    def get_image_embeddings(self, tokenizer: Tokenizer, image: Image.Image | list[Image.Image], text_alias: str | None = None):
        if isinstance(image, list):
            assert text_alias is None, "Cannot apply one alias to multiple images"
            return [self.get_image_embeddings(tokenizer, img) for img in image]
        pixels, resized, grid = self._preprocess(image)
        embeddings = self._encode(pixels, grid).squeeze(0)
        id_start = tokenizer.single_id("<img>")
        id_end = tokenizer.single_id("</img>")
        token_string = torch.tensor([[id_start] + [-1] * embeddings.shape[0] + [id_end]], dtype = torch.long)
        result = MMEmbedding(embeddings = embeddings, token_string = token_string, text_alias = text_alias)
        result.metadata.update({
            "original_size": image.size,
            "preprocessed_size": resized,
            "model_architecture": self.config.architecture,
            "media_type": "image",
        })
        return result

    @torch.inference_mode()
    def get_video_embeddings(self, tokenizer: Tokenizer, frames: list[Image.Image], text_alias: str | None = None):
        if not frames:
            raise ValueError("Video input requires at least one frame")
        if len({frame.size for frame in frames}) != 1:
            raise ValueError("All video frames must have the same dimensions")
        prepared = [self._preprocess(frame, video = True) for frame in frames]
        if len({item[2] for item in prepared}) != 1:
            raise ValueError("Video frames did not resolve to a common patch grid")
        grid = prepared[0][2]
        pixels = torch.stack([item[0] for item in prepared])
        temporal = self.config.vision.video_temporal_patch_size
        if pixels.shape[0] % temporal:
            pixels = torch.cat((pixels, pixels[-1:].expand(temporal - pixels.shape[0] % temporal, -1, -1, -1)))
        groups = pixels.reshape(-1, temporal * 3, pixels.shape[-2], pixels.shape[-1])
        embeds = self._encode(groups, grid, video = True).reshape(-1, int(grid[0] * grid[1] * self.config.downsample_ratio ** 2), self.config.hidden_size)
        h = int(grid[0] * self.config.downsample_ratio)
        w = int(grid[1] * self.config.downsample_ratio)
        if h * w != embeds.shape[1]:
            raise ValueError("NemotronH Omni video projector returned an unexpected patch grid")
        keep_mask = torch.ones(embeds.shape[:2], dtype = torch.bool)
        if self.config.video_pruning_rate > 0 and embeds.shape[0] > 1:
            spatial = embeds.reshape(embeds.shape[0], h, w, embeds.shape[-1])
            similarity = F.cosine_similarity(spatial[1:], spatial[:-1], dim = -1)
            dissimilarity = torch.cat((255 * torch.ones_like(spatial[:1, :, :, 0]), 1 - similarity), dim = 0).reshape(-1)
            keep = max(h * w, int(embeds.shape[0] * h * w * (1 - self.config.video_pruning_rate)))
            indices = torch.argsort(dissimilarity, descending = True, stable = True)[:keep]
            keep_mask = torch.zeros_like(dissimilarity, dtype = torch.bool)
            keep_mask[indices] = True
            keep_mask = keep_mask.reshape(embeds.shape[:2])
        counts = keep_mask.sum(dim = 1).tolist()
        embeds = embeds[keep_mask]
        id_start = tokenizer.single_id("<img>")
        id_end = tokenizer.single_id("</img>")
        token_ids = []
        for count in counts:
            token_ids.extend([id_start] + [-1] * count + [id_end])
        token_string = torch.tensor([token_ids], dtype = torch.long)
        result = MMEmbedding(embeddings = embeds, token_string = token_string, text_alias = text_alias)
        result.metadata.update({
            "original_size": frames[0].size,
            "preprocessed_size": prepared[0][1],
            "frame_count": len(frames),
            "model_architecture": self.config.architecture,
            "media_type": "video",
        })
        return result
