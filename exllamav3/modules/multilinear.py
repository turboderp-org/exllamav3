from __future__ import annotations
import torch
from . import Linear

class MultiLinear:
    def __init__(
        self,
        device: torch.Device,
        linears: list[Linear],
        allow_bias: bool = False,
    ):
        self.device = device
        self.linears = linears
        self.num_linears = len(linears)

        assert all(l.quant_type == "exl3" for l in linears)
        # The mgemm kernels never apply biases; callers setting allow_bias handle them
        # separately (BC_BlockSparseMLP bias kernels)
        assert allow_bias or all(l.inner.bias is None for l in linears)
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

    def q_cb(self):
        return self.mcg, self.mul1

    def unload(self):
        pass


class SlicedMultiLinear:
    """
    Bundle of linears that share an input but differ in width (Q/K/V, optionally a full gate),
    cut into equal-width column slices and run as ONE exl3_mgemm in sliced mode. The kernel
    schedules every slice as its own concurrent z-group, so a wide Q can't leave the K/V groups
    idle the way per-matrix scheduling does, and the input transform runs once per source.

    Slice pointers index the trellis / svh at the slice's first column; the kernel gets the
    source's full width as the row stride and writes each slice in place into the source's
    output. c_ptrs() builds the per-slice output pointer table for a set of output tensors.
    """

    def __init__(
        self,
        device: torch.Device,
        linears: list[Linear],
        min_width: int = 256,
    ):
        import math
        self.device = device
        self.linears = linears
        self.num_src = len(linears)

        assert all(l.quant_type == "exl3" for l in linears)
        assert all(l.inner.bias is None for l in linears), "SlicedMultiLinear: biases unsupported"
        assert all(not l.softcap for l in linears)
        assert all(l.post_scale == 1.0 for l in linears)

        self.in_features = linears[0].in_features
        self.K = linears[0].inner.K
        assert all(l.inner.K == self.K for l in linears)
        assert all(l.in_features == self.in_features for l in linears)
        self.mcg = linears[0].inner.mcg
        assert all(l.inner.mcg == self.mcg for l in linears[1:])
        self.mul1 = linears[0].inner.mul1
        assert all(l.inner.mul1 == self.mul1 for l in linears[1:])

        # Equal slice width: the gcd of the widths, which has to be a whole number of 128-column
        # blocks and not so narrow that a slice can't fill its group
        widths = [l.out_features for l in linears]
        self.width = math.gcd(*widths)
        if self.width % 128 != 0 or self.width < min_width:
            raise ValueError(f"SlicedMultiLinear: unsuitable slice width {self.width} for widths {widths}")

        trellis_ptrs, svh_ptrs, targets, offsets, strides, srcs = [], [], [], [], [], []
        for i, l in enumerate(linears):
            trellis = l.inner.trellis     # (K_in / 16, N / 16, 16 * bits) int16
            assert trellis.dim() == 3 and trellis.shape[-1] == 16 * self.K and trellis.dtype == torch.int16
            assert trellis.is_contiguous() and l.inner.svh.is_contiguous()
            for n0 in range(0, l.out_features, self.width):
                trellis_ptrs.append(trellis.data_ptr() + (n0 // 16) * 16 * self.K * trellis.element_size())
                svh_ptrs.append(l.inner.svh.data_ptr() + n0 * l.inner.svh.element_size())
                targets.append(i)
                offsets.append(n0)
                strides.append(l.out_features)
                srcs.append(i)
        self.num_slices = len(targets)
        self.slice_targets = targets
        self.slice_offsets = offsets

        self.ptrs_trellis = torch.tensor(trellis_ptrs, dtype = torch.long, device = device)
        self.ptrs_svh = torch.tensor(svh_ptrs, dtype = torch.long, device = device)
        self.ptrs_suh = torch.tensor([l.inner.suh.data_ptr() for l in linears], dtype = torch.long, device = device)
        self.size_n_list = torch.full((self.num_slices,), self.width, dtype = torch.int32, device = device)
        self.n_stride_list = torch.tensor(strides, dtype = torch.int32, device = device)
        self.had_src_list = torch.tensor(srcs, dtype = torch.int32, device = device)
        # CPU descriptor for the C++ graph builders: target, column offset, width, row stride, source
        self.meta = torch.tensor([targets, offsets, [self.width] * self.num_slices, strides, srcs], dtype = torch.int32)

    def c_ptrs(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        """Per-slice output pointers: outputs[i] is linear i's contiguous (rows, out_features) output"""
        assert len(outputs) == self.num_src
        ptrs = []
        for t, n0 in zip(self.slice_targets, self.slice_offsets):
            o = outputs[t]
            assert o.is_contiguous() and o.shape[-1] == self.linears[t].out_features
            ptrs.append(o.data_ptr() + n0 * o.element_size())
        return torch.tensor(ptrs, dtype = torch.long, device = self.device)

    def q_cb(self):
        return self.mcg, self.mul1

    def unload(self):
        pass
