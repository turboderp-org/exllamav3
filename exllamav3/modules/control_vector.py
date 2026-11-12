"""Opt-in, per-layer residual projection for Qwen4Exp inference."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re

from safetensors import safe_open
import torch
import triton
import triton.language as tl

from .module import Module
from .hyperconnections import hc_flush


@triton.jit
def _project_rows(X, V, WIDTH: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * WIDTH + col, col < WIDTH, other = 0).to(tl.float32)
    direction = tl.load(V + col, col < WIDTH, other = 0).to(tl.float32)
    dot = tl.sum(value * direction, axis = 0)
    tl.store(X + row * WIDTH + col, value - (SCALE * dot) * direction, col < WIDTH)


@dataclass(frozen = True)
class ControlVector:
    """A model-lifetime projection, loaded before constructing the model.

    Directions are stored as immutable F32 bytes. Each model gets its own tensors;
    changing the source file or retaining this object cannot mutate a live model.
    The first integration supports Qwen4Exp's FP32 hyperconnection streams only.
    """
    hidden_size: int
    num_hidden_layers: int
    hc_count: int
    # (zero-based decoder index, normalized F32 direction, effective F32 strength)
    directions: tuple[tuple[int, bytes, float], ...]
    fingerprint: str = field(init = False)

    def __post_init__(self):
        shape = (self.hidden_size, self.num_hidden_layers, self.hc_count)
        if any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError("Invalid control-vector model dimensions")
        directions = []
        previous = None
        for i, data, strength in self.directions:
            if type(i) is not int or not 0 <= i < self.num_hidden_layers or \
                    (previous is not None and i != previous + 1):
                raise ValueError("Control-vector layers must be contiguous, ordered and unique")
            if not isinstance(data, (bytes, bytearray, memoryview)) or len(data) != 4 * self.hidden_size:
                raise ValueError(f"Invalid control-vector direction at layer {i}")
            data = bytes(data)
            value = torch.frombuffer(bytearray(data), dtype = torch.float32)
            if not torch.isfinite(value).all() or abs(value.double().norm().item() - 1.0) > 1e-5:
                raise ValueError(f"Control-vector direction must be finite and normalized at layer {i}")
            strength = torch.tensor(float(strength), dtype = torch.float32).item()
            if not math.isfinite(strength) or strength <= 0:
                raise ValueError(f"Invalid control-vector strength at layer {i}")
            directions.append((i, data, strength))
            previous = i
        if not directions:
            raise ValueError("Control vector contains no directions")
        # Own immutable data even when constructed directly with mutable inputs.
        object.__setattr__(self, "directions", tuple(directions))
        # Hash effective math, not the pathname or caller-provided identity.
        digest = hashlib.sha256(json.dumps(["qwen4_exp.project.v1", *shape]).encode())
        for i, data, strength in directions:
            digest.update(json.dumps([i, strength]).encode())
            digest.update(data)
        object.__setattr__(self, "fingerprint", digest.hexdigest())

    @classmethod
    def from_file(cls, config, path, *, scale: float = 1.0,
                  layer_range: tuple[int, int] | None = None) -> ControlVector | None:
        """Load direction.N tensors; layer_range is zero-based and inclusive.

        None selects the entire contiguous range present in the file. Zero scale
        returns None after validation, preserving the unmodified model graph/cache.
        Scaling and normalization match llama.cpp's per-layer projection variant:
        scale the raw F32 vector, normalize using an FP64 norm, then store the unit
        direction and effective norm as F32.
        """
        if getattr(config, "architecture", None) != "Qwen4ExpForConditionalGeneration":
            raise ValueError("Control-vector projection currently supports Qwen4Exp only")
        scale = float(scale)
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("Control-vector scale must be finite and nonnegative")
        shape = (config.hidden_size, config.num_hidden_layers, config.hc_mult)
        if any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError("Invalid control-vector model dimensions")
        tensors = {}
        with safe_open(str(path), framework = "pt", device = "cpu") as reader:
            expected = {
                "source_architecture": "qwen4_exp", "operation": "project",
                "hidden_size": str(shape[0]), "num_hidden_layers": str(shape[1]),
                "hc_count": str(shape[2]),
            }
            metadata = reader.metadata() or {}
            if any(metadata.get(k) != v for k, v in expected.items()):
                raise ValueError("Control-vector metadata does not match the model")
            for name in reader.keys():
                match = re.fullmatch(r"direction\.(0|[1-9][0-9]*)", name)
                if match is None or not 0 <= int(match[1]) < shape[1]:
                    raise ValueError(f"Invalid control-vector tensor name: {name}")
                value = reader.get_tensor(name)
                if value.shape != (shape[0],) or value.dtype != torch.float32 \
                        or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid control-vector tensor: {name}")
                tensors[int(match[1])] = value
        if not tensors:
            raise ValueError("Control-vector file contains no directions")
        if layer_range is not None and (not isinstance(layer_range, (tuple, list)) or len(layer_range) != 2):
            raise ValueError("Control-vector range must contain two decoder indices")
        first, last = layer_range if layer_range is not None else (min(tensors), max(tensors))
        if type(first) is not int or type(last) is not int or not 0 <= first <= last < shape[1]:
            raise ValueError("Control-vector range must contain valid zero-based decoder indices")
        if any(i not in tensors for i in range(first, last + 1)):
            raise ValueError("Control-vector file is missing a selected layer")
        if scale == 0:
            return None
        directions = []
        for i in range(first, last + 1):
            weighted = tensors[i] * scale
            norm = weighted.double().norm().item()
            strength = torch.tensor(norm, dtype = torch.float32).item()
            if not math.isfinite(norm) or norm == 0 or not math.isfinite(strength):
                raise ValueError(f"Invalid control-vector norm at layer {i}")
            direction = (weighted.double() / norm).float().contiguous()
            directions.append((i, direction.numpy().tobytes(), strength))
        return cls(*shape, tuple(directions))

    def validate(self, config):
        if getattr(config, "architecture", None) != "Qwen4ExpForConditionalGeneration" \
                or (config.hidden_size, config.num_hidden_layers, config.hc_mult) != \
                   (self.hidden_size, self.num_hidden_layers, self.hc_count):
            raise ValueError("Control vector was loaded for a different model architecture or shape")
        if getattr(config, "layer_map", None) or getattr(config, "layer_map_str", None):
            raise ValueError("Control-vector projection does not support layer remapping")

    def modules(self, config, key_prefix):
        self.validate(config)
        return {
            i: ResidualProjection(
                config, f"{key_prefix}.layers.{i}.control_vector",
                torch.frombuffer(bytearray(data), dtype = torch.float32), strength, i,
            ) for i, data, strength in self.directions
        }


class ResidualProjection(Module):
    """Project each FP32 residual stream independently after a decoder block."""

    def __init__(self, config, key, direction, scale, steered_layer_idx):
        super().__init__(config, key, None)
        self.direction_cpu = direction.clone()
        self.direction = None
        self.scale = scale
        self.steered_layer_idx = steered_layer_idx
        self.width = config.hidden_size
        self.streams = config.hc_mult
        # layer_idx stays None: this module owns no attention or recurrent cache.

    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        self.direction = self.direction_cpu.to(device = device, dtype = torch.float32)

    def unload(self):
        self.direction = None
        super().unload()

    def weights_numel(self):
        return self.direction_cpu.numel()

    def optimizer_targets(self):
        return []

    def prepare_for_device(self, x, params):
        if "hc_pending" in params and x.device != self.device:
            hc_flush(params)
        return super().prepare_for_device(x, params)

    def forward(self, x, params, out_dtype = None):
        if x.ndim != 4 or x.shape[-2:] != (self.streams, self.width) or x.dtype != torch.float32:
            raise ValueError("Expected FP32 [batch, tokens, streams, hidden_size] residual")
        if out_dtype not in (None, torch.float32):
            raise ValueError("Projection must preserve FP32 residual streams")
        if "capture" in params or "quant_preserve" in params:
            raise ValueError("Control-vector projection is inference-only")
        hc_flush(params)
        # Inserting this module breaks GatedResidual.link_sites' weighted-copy
        # handoff. Also invalidate any copy a caller might have relinked manually.
        params.pop("gr_weighted", None)
        if not x.is_contiguous():
            x = x.contiguous()
        if self.direction is None or self.direction.device != x.device:
            raise RuntimeError("Projection vector is not loaded on the residual device")
        if x.numel() == 0:
            return x
        if x.is_cuda:
            _project_rows[(x.numel() // self.width,)](
                x, self.direction, WIDTH = self.width, SCALE = self.scale,
                BLOCK = triton.next_power_of_2(self.width), num_warps = 4, enable_fp_fusion = False,
            )
        else:
            dots = (x * self.direction).sum(dim = -1, keepdim = True)
            x.sub_((self.scale * dots) * self.direction)
        return x
