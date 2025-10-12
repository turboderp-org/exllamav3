from __future__ import annotations
import torch
from . import Linear

class MultiLinear:
    def __init__(
        self,
        device: torch.Device,
        linears: list[Linear]
    ):
        self.device = device
        self.linears = linears
        self.num_linears = len(linears)

        assert all(l.quant_type == "exl3" for l in linears)
        assert all(l.inner.bias is None for l in linears)
        assert all(not l.softcap for l in linears)
        assert all(l.post_scale == 1.0 for l in linears)

        self.in_features = linears[0].in_features
        self.out_features = linears[0].out_features
        self.K = linears[0].inner.K
        assert all(l.inner.K == self.K for l in linears)
        assert all(l.in_features == self.in_features for l in linears)
        assert all(l.out_features == self.out_features for l in linears)

        self.ptrs_suh = torch.tensor([l.inner.suh.data_ptr() for l in linears], dtype = torch.long, device = device)
        self.ptrs_svh = torch.tensor([l.inner.svh.data_ptr() for l in linears], dtype = torch.long, device = device)
        self.ptrs_trellis = torch.tensor([l.inner.trellis.data_ptr() for l in linears], dtype = torch.long, device = device)

        self.mcg = linears[0].inner.mcg
        assert all(l.inner.mcg == self.mcg for l in linears[1:])
        self.mul1 = linears[0].inner.mul1
        assert all(l.inner.mul1 == self.mul1 for l in linears[1:])

    def unload(self):
        pass
