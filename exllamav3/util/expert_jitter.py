"""Seeded expert offsets shared by requests and direct-forward benchmarks."""
import math
import random
import torch


def expert_jitter_strength(value):
    strength = float(value)
    if not math.isfinite(strength) or strength < 1:
        raise ValueError("expert_jitter must be finite and >= 1")
    return strength


def expert_jitter_layout(model):
    from ..modules import BlockSparseMLP
    layout, size = {}, 0
    for module in model:
        if isinstance(module, BlockSparseMLP) and module.key not in layout:
            layout[module.key] = size
            size += module.num_experts
    return layout, size


def expert_jitter_offsets(strength, rng, size):
    if strength == 1 or not size:
        return None
    # Clone the sampling RNG; keep factors in Python precision until log() so even strengths
    # above the fp16 factor range produce finite, representable offsets.
    jitter_rng = random.Random()
    jitter_rng.setstate(rng.getstate())
    offsets = torch.tensor([
        math.log(jitter_rng.uniform(1 / strength, strength)) for _ in range(size)
    ], dtype = torch.half).unsqueeze(0)
    offsets._static_dev_cache = True
    return offsets
