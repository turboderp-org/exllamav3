"""
LoRA (Low-Rank Adaptation) support for ExLlamaV3.

Loads PEFT-format LoRA adapters and applies them at runtime without
modifying base model weights. Multiple adapters can be loaded
simultaneously; all loaded adapters are applied during forward pass.

Usage::

    lora = LoRA.from_directory(model, "/path/to/peft-adapter")
    # All generation now includes this adapter's contribution
    response = generator.generate(prompt = "Hello", ...)
    # Unload to revert to base model
    lora.unload()

Compatible with adapters trained via PEFT/Unsloth on the full-precision
base model. LoRA weights are applied on top of the dequantized output
of each target linear layer.
"""

from __future__ import annotations
import os
import json
import math
import torch
from safetensors.torch import load_file as safe_load_file
from ..modules.linear import Linear

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model


class LoRA:
    """
    LoRA adapter loaded from PEFT format.

    Stores pre-transposed and pre-scaled A/B weight matrices on target
    Linear modules. During forward pass, Linear.forward() computes:
    ``output += input @ A @ B`` for each loaded adapter.
    """

    @staticmethod
    def from_directory(
            model: Model,
            directory: str,
            lora_scaling: float = 1.0,
    ) -> LoRA:
        """
        Load LoRA adapter from a PEFT directory.

        :param model:
            Loaded ExLlamaV3 model instance.

        :param directory:
            Path to directory containing adapter_config.json and
            adapter_model.safetensors (or .bin).

        :param lora_scaling:
            Additional scaling factor applied on top of alpha/r.
        """
        config_path = os.path.join(directory, "adapter_config.json")
        weights_st = os.path.join(directory, "adapter_model.safetensors")
        weights_bin = os.path.join(directory, "adapter_model.bin")

        if os.path.exists(weights_st):
            return LoRA(model, config_path, weights_st, lora_scaling)
        if os.path.exists(weights_bin):
            return LoRA(model, config_path, weights_bin, lora_scaling)
        raise FileNotFoundError(f"No LoRA adapter found in {directory}")

    @torch.inference_mode()
    def __init__(
            self,
            model: Model,
            config_path: str,
            weights_path: str,
            lora_scaling: float = 1.0,
    ):
        self.target_modules = {}
        self.name = os.path.basename(os.path.dirname(config_path))

        # Read adapter config
        with open(config_path, encoding="utf8") as f:
            config = json.load(f)

        self.lora_r = config["r"]
        self.lora_alpha = float(config["lora_alpha"])

        effective_alpha = self.lora_alpha
        if config.get("use_rslora", False):
            effective_alpha *= math.sqrt(self.lora_r)

        self.lora_scaling = lora_scaling * effective_alpha / self.lora_r

        if config.get("fan_in_fan_out", False):
            raise ValueError("fan_in_fan_out mode is not supported")

        # Build modules dict if needed
        if model.modules_dict is None:
            model.modules_dict = {m.key: m for m in model}

        # Load weights
        if weights_path.endswith(".safetensors"):
            raw_tensors = safe_load_file(weights_path, device="cpu")
        else:
            raw_tensors = torch.load(weights_path, map_location="cpu", weights_only=True)

        loaded = 0
        skipped_keys = []
        tp_skipped = []

        for key, tensor in raw_tensors.items():
            # Skip non-LoRA keys (e.g. modules_to_save, original_module)
            if ".lora_A." not in key and ".lora_B." not in key:
                continue

            # Extract full path and lora half from PEFT key
            full_path, lora_half = self._parse_key(key)
            if full_path is None:
                skipped_keys.append(key)
                continue

            # Match against model modules by suffix to handle any
            # PEFT key prefix (base_model.model.*, etc.)
            target = None
            module_key = None
            path_parts = full_path.split(".")
            for start in range(len(path_parts)):
                candidate = ".".join(path_parts[start:])
                t = model.modules_dict.get(candidate)
                if t is not None and isinstance(t, Linear):
                    target = t
                    module_key = candidate
                    break

            if target is None:
                skipped_keys.append(key)
                continue

            # Tensor-parallel sliced modules not supported
            if target.is_sliced:
                tp_skipped.append(key)
                continue

            if tensor.dtype in (torch.bfloat16, torch.float32):
                tensor = tensor.to(torch.float16)

            # Transpose for efficient matmul: x @ A @ B
            # PEFT stores lora_A as [rank, in_features] and
            # lora_B as [out_features, rank].
            # We want A as [in_features, rank] and B as [rank, out_features].
            tensor = tensor.T.contiguous()

            # Pre-scale B matrix
            if lora_half == "lora_B" and self.lora_scaling != 1.0:
                tensor.mul_(self.lora_scaling)

            # Pad to match target dimensions (quantized layers may pad features to multiples of block size)
            if lora_half == "lora_A" and tensor.shape[0] < target.in_features:
                padded = torch.zeros(target.in_features, tensor.shape[1], dtype=tensor.dtype)
                padded[:tensor.shape[0]] = tensor
                tensor = padded
            elif lora_half == "lora_B" and tensor.shape[1] < target.out_features:
                padded = torch.zeros(tensor.shape[0], target.out_features, dtype=tensor.dtype)
                padded[:, :tensor.shape[1]] = tensor
                tensor = padded

            tensor = tensor.to(target.device)

            # Register on target module
            if lora_half == "lora_A":
                target.lora_a_tensors[self] = tensor
            else:
                target.lora_b_tensors[self] = tensor

            self.target_modules[module_key] = target
            loaded += 1

        print(
            f" -- LoRA '{self.name}': loaded {loaded} tensors "
            f"(r={self.lora_r}, alpha={self.lora_alpha:.0f}, "
            f"scaling={self.lora_scaling:.4f})"
        )
        if skipped_keys:
            print(
                f" -- LoRA '{self.name}': skipped {len(skipped_keys)} "
                f"unmatched keys"
            )
        if tp_skipped:
            print(
                f" -- LoRA '{self.name}': skipped {len(tp_skipped)} tensors "
                f"on tensor-parallel sliced modules"
            )

    @staticmethod
    def _parse_key(key: str) -> tuple[str | None, str | None]:
        """
        Parse PEFT tensor key to (full_path, lora_half).

        Returns the full dotted path before lora_A/lora_B and the half
        name. The caller matches this path against model modules by
        suffix, so any PEFT key prefix format is handled automatically.

            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            -> ("base_model.model.model.layers.0.self_attn.q_proj", "lora_A")
        """
        parts = key.split(".")
        for j, p in enumerate(parts):
            if p in ("lora_A", "lora_B"):
                return ".".join(parts[:j]), p
        return None, None

    def unload(self):
        """Remove this adapter's tensors from all target modules."""
        for target in self.target_modules.values():
            target.lora_a_tensors.pop(self, None)
            target.lora_b_tensors.pop(self, None)

        self.target_modules = {}
