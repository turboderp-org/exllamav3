"""
Generate the synthetic DFlash2 fixture: seeded fp16 tensors for every weight the
loader demands, derived by walking the live module tree (shapes come from the
module objects, not from hand enumeration). Run from the repo root:

    PYTHONPATH=. python tests/data/dflash2_synth/make_synth.py

Requires torch + safetensors. No GPU needed.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from exllamav3.architecture.dflash2 import DFlash2Config, DFlash2Model
from exllamav3.modules import Linear, RMSNorm
from exllamav3.modules.arch_specific.dflash2 import (
    CandidateSelector,
    GroupedDynamicCausalConv,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    torch.manual_seed(20260917)
    cfg = DFlash2Config(HERE)
    model = DFlash2Model(cfg)
    tensors = {}

    def emit(module):
        if isinstance(module, Linear):
            w = torch.randn(
                module.out_features_unpadded, module.in_features_unpadded,
                dtype = torch.float32,
            ) * 0.02
            tensors[module.key + ".weight"] = w.to(torch.float16)
        elif isinstance(module, RMSNorm):
            if not module.unweighted:
                tensors[module.tensor_key] = torch.ones(
                    cfg.hidden_size, dtype = torch.float16)
        elif isinstance(module, GroupedDynamicCausalConv):
            tensors[module.key + ".base_kernel"] = (
                torch.randn(2, module.kernel_size, module.hidden_size,
                            dtype = torch.float32) * 0.05
            ).to(torch.float16)
        elif isinstance(module, CandidateSelector):
            for name in ("predecessor_codebook", "successor_codebook"):
                tensors[f"{module.key}.{name}"] = (
                    torch.randn(cfg.vocab_size, module.rank,
                                dtype = torch.float32) * 0.05
                ).to(torch.float16)
        for sm in module.modules:
            emit(sm)

    for m in model.modules:
        emit(m)

    from safetensors.torch import save_file
    out = os.path.join(HERE, "model.safetensors")
    save_file(tensors, out)
    total = sum(t.numel() * 2 for t in tensors.values())
    print(f"wrote {out}: {len(tensors)} tensors, {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
