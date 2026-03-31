from __future__ import annotations
import torch


class TQ3ExpertOffloader:
    """
    Manages MoE expert weight compression for memory-efficient inference.

    In MoE models (e.g., Qwen3.5-35B-A3B with 64 experts, 4 active per token),
    most experts sit idle in VRAM. This class compresses idle experts to TQ3
    format (~6.4x compression) and decompresses them on-demand.

    Memory impact example:
    - 64 experts x 3 linear layers x (4096x14336) @ fp16 = ~21GB
    - With TQ3: ~3.3GB for idle experts + ~1.3GB for 4 active experts = ~4.6GB
    - Savings: ~16GB VRAM

    Usage:
        offloader = TQ3ExpertOffloader()

        # Compress all experts initially
        for i, expert in enumerate(experts):
            offloader.compress_expert(i, expert.weight)

        # During inference, decompress only selected experts
        for expert_idx in selected_experts:
            weight = offloader.get_expert(expert_idx, device='cuda')
            # ... use weight ...
            offloader.release_expert(expert_idx)  # Free GPU copy
    """

    def __init__(self, store_device: str = "cpu"):
        """
        Args:
            store_device: Device for compressed storage ('cpu' for max savings)
        """
        self.store_device = store_device
        self.experts: dict = {}  # expert_idx -> compressed data
        self.active_cache: dict = {}  # expert_idx -> decompressed weight

    def compress_expert(
        self,
        expert_idx: int,
        weight: torch.Tensor,  # (in_features, out_features) fp16
    ) -> None:
        """Compress an expert's weight to TQ3 and store.

        Args:
            expert_idx: Integer index identifying the expert.
            weight: (in_features, out_features) fp16 weight tensor.
                    in_features must be a multiple of 32.
        """
        in_features, out_features = weight.shape
        assert in_features % 32 == 0, \
            f"in_features must be multiple of 32, got {in_features}"

        num_blocks = in_features // 32
        w_blocks = weight.float().view(num_blocks, 32, out_features)

        # Per-block scale (max abs per 32-row block, per column)
        scales = w_blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
        scales_flat = scales.squeeze(1)  # (num_blocks, out_features)

        # Normalize
        w_norm = w_blocks / scales

        # Lloyd-Max ternary: boundary at +/- 0.5
        nonzero = (w_norm.abs() >= 0.5).int()
        positive = ((w_norm > 0) & (nonzero == 1)).int()

        # Pack bitplanes
        device = weight.device
        bit_indices = torch.arange(32, device=device).view(1, 32, 1)
        bp0 = (nonzero << bit_indices).sum(dim=1).to(torch.int32)
        bp1 = (positive << bit_indices).sum(dim=1).to(torch.int32)

        packed = torch.zeros(num_blocks * 2, out_features, dtype=torch.int32, device=device)
        packed[0::2] = bp0
        packed[1::2] = bp1

        # Store on target device
        self.experts[expert_idx] = {
            "packed": packed.to(self.store_device),
            "scales": scales_flat.half().to(self.store_device),
            "in_features": in_features,
            "out_features": out_features,
        }

    def get_expert(self, expert_idx: int, device: str = "cuda") -> torch.Tensor:
        """Decompress an expert's weight on-demand.

        Returns the cached copy if the expert is already decompressed.

        Args:
            expert_idx: Integer index of the expert to retrieve.
            device: Target device for the decompressed weight.

        Returns:
            (in_features, out_features) fp16 weight tensor on the target device.
        """
        if expert_idx in self.active_cache:
            return self.active_cache[expert_idx]

        data = self.experts[expert_idx]
        packed = data["packed"].to(device)
        scales = data["scales"].to(device)
        in_f = data["in_features"]
        out_f = data["out_features"]

        num_blocks = in_f // 32
        bp0 = packed[0::2]  # nonzero masks, (num_blocks, out_f)
        bp1 = packed[1::2]  # positive masks, (num_blocks, out_f)

        bit_indices = torch.arange(32, device=device).view(1, 32, 1)
        nonzero = ((bp0.unsqueeze(1) >> bit_indices) & 1).to(torch.float16)
        positive = ((bp1.unsqueeze(1) >> bit_indices) & 1).to(torch.float16)

        # Reconstruct: val = nonzero * (2*positive - 1) * scale
        ternary = nonzero * (2.0 * positive - 1.0)
        w = (ternary * scales.unsqueeze(1)).reshape(in_f, out_f)

        self.active_cache[expert_idx] = w
        return w

    def release_expert(self, expert_idx: int) -> None:
        """Free the decompressed GPU copy of an expert.

        Args:
            expert_idx: Integer index of the expert to release.
        """
        self.active_cache.pop(expert_idx, None)

    def release_all(self) -> None:
        """Free all decompressed GPU copies."""
        self.active_cache.clear()

    def num_experts(self) -> int:
        """Return the number of compressed experts stored."""
        return len(self.experts)

    def compressed_size(self) -> int:
        """Total compressed storage in bytes across all experts."""
        total = 0
        for data in self.experts.values():
            total += data["packed"].numel() * 4 + data["scales"].numel() * 2
        return total

    def uncompressed_size(self) -> int:
        """Total uncompressed storage in bytes (fp16) across all experts."""
        total = 0
        for data in self.experts.values():
            total += data["in_features"] * data["out_features"] * 2
        return total
