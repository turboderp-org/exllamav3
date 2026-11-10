from __future__ import annotations
import torch
from torch import nn
from ...ext import exllamav3_ext as ext
from ...util.tensor import to2
from ...util import first_not_none

class LinearFP16:

    quant_type: str = "fp16"

    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        full_in_features: int | None = None,
        full_out_features: int | None = None,
        first_in_feature: int | None = None,
        first_out_feature: int | None = None,
        out_dtype: torch.dtype | None = None,
        key: str | None = None
    ):
        self.weight = weight
        self._pinned_store = None
        if bias is not None and bias.dtype == torch.float: bias = bias.to(torch.half)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.swap_device = None
        self.full_in_features = full_in_features
        self.full_out_features = full_out_features
        self.first_in_feature = first_in_feature
        self.first_out_feature = first_out_feature
        self.out_dtype = out_dtype
        self.key = key

        if self.weight.shape[0] == full_in_features and self.weight.shape[0] < in_features:
            self.weight = self.weight[first_in_feature : first_in_feature + in_features, :]
        if self.weight.shape[1] == full_out_features and self.weight.shape[1] < out_features:
            self.weight = self.weight[:, first_out_feature : first_out_feature + out_features]
            if bias is not None:
                self.bias = self.bias[..., first_out_feature : first_out_feature + out_features]

        if in_features < full_in_features or out_features < full_out_features:
            w = torch.empty(self.weight.shape, dtype = self.weight.dtype, device = self.weight.device)
            w.copy_(self.weight)
            self.weight = w

        self.bc = ext.BC_LinearFP16(
            self.weight,
            self.bias,
        )

    def unload(self):
        pass

    def get_tensors(self, key: str):
        t = {}
        t[f"{key}.weight"] = self.weight.T.contiguous()
        if self.bias is not None:
            t[f"{key}.bias"] = self.bias
        return t

    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.view(-1, x.shape[-1])
        dtype = out_dtype or self.out_dtype or torch.half
        y = torch.empty(
            (x.shape[0], self.out_features),
            dtype = dtype,
            device = x.device
        )
        weight = self.weight
        pinned = self._pinned_store
        if pinned is not None:
            # Pinned-host storage (vision towers): stage the weight into a shared VRAM scratch
            # first. cuBLAS re-reads B once per m-tile, so computing straight from the zero-copy
            # alias multiplies the PCIe traffic by ~m/128 (measured 56x wall time on a large
            # tower at m ~4k); a single async H2D pass then a normal GEMM costs one transfer
            from ...util.tensor import g_tensor_cache
            weight = g_tensor_cache.get_bucketed(x.device, pinned.numel(), pinned.dtype, "fp16_pin_stage").view(pinned.shape)
            weight.copy_(pinned, non_blocking = True)
        if dtype == x.dtype:
            torch.matmul(x, weight, out = y)
        else:
            ext.hgemm(x, weight, y)
        if self.bias is not None:
            y += self.bias
        y = y.view(out_shape)
        return y

    def get_weight_tensor(self) -> torch.Tensor:
        return self.weight

    def staged_weight(self) -> torch.Tensor:
        """The fp16 (in, out) weight for ext kernels that read it directly; a transient view for a packed inner"""
        return self.weight

    def reconstruct_columns(self, cols: torch.Tensor, out: torch.Tensor):
        """Weight columns (in_features, len(cols)) for the given column indices, written into out"""
        torch.index_select(self.weight, 1, cols, out = out)

    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.bias

    def set_weight(self, w: torch.Tensor):
        self.weight = w.half()

    # Swap tensors to CPU (to free some space while quantizing)
    def swap_cpu(self):
        if self.swap_device is not None:
            return
        self.swap_device = self.weight.device
        self.weight = self.weight.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

    def unswap_cpu(self):
        if self.swap_device is None:
            return
        self.weight = self.weight.to(self.swap_device)
        if self.bias is not None:
            self.bias = self.bias.to(self.swap_device)
        self.swap_device = None

    def tp_export(self, plan, producer):
        return {
            "cls": LinearFP16,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "weight": producer.send(self.weight),
            "bias": producer.send(self.bias),
            "out_dtype": self.out_dtype,
        }

    @staticmethod
    def tp_import_split(local_context, exported, plan, split, dbg = False):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_w = exported["weight"]
        id_b = exported["bias"]

        if split is not None:
            split_out, first, last = split
        else:
            split_out, first, last = True, 0, exported["out_features"]

        if split_out:
            w = consumer.recv(id_w, cuda = True, slice_dim = 1, first = first, last = last)
            b = consumer.recv(id_b, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features = last - first
        else:
            w = consumer.recv(id_w, cuda = True, slice_dim = 0, first = first, last = last)
            b = consumer.recv(id_b, cuda = True) if (first == 0) else None
            in_features = last - first
            out_features = exported["out_features"]

        module = LinearFP16(
            in_features = in_features,
            out_features = out_features,
            weight = w,
            bias = b,
            full_in_features = in_features,
            full_out_features = out_features,
            first_in_feature = 0,
            first_out_feature = 0,
            out_dtype = exported["out_dtype"],
        )
        return module


    @staticmethod
    def tp_import_split_3(local_context, exported, plan, split_0, split_1, split_2, dbg = False):
        return LinearFP16.tp_import_split_n(local_context, exported, plan, [split_0, split_1, split_2], dbg)


    @staticmethod
    def tp_import_split_n(local_context, exported, plan, splits, dbg = False):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_w = exported["weight"]
        id_b = exported["bias"]

        w_ = []
        b_ = []
        in_features = 0
        out_features = 0

        for split in splits:
            assert split is not None
            split_out, first, last = split
            assert split_out
            w = consumer.recv(id_w, cuda = True, slice_dim = 1, first = first, last = last)
            b = consumer.recv(id_b, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features += last - first
            w_.append(w)
            b_.append(b)

        w = torch.cat(w_, dim = 1)
        b = torch.cat(b_, dim = 0) if b_[0] is not None else None

        module = LinearFP16(
            in_features = in_features,
            out_features = out_features,
            weight = w,
            bias = b,
            full_in_features = in_features,
            full_out_features = out_features,
            first_in_feature = 0,
            first_out_feature = 0,
            out_dtype = exported["out_dtype"],
        )
        return module


FMT_E4M3, FMT_E2M1, FMT_E5M2 = 0, 1, 2

# MXFP4 (e2m1): sign bit, 2 exponent bits, 1 mantissa bit
MXFP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def packed_fmt(weight: torch.Tensor) -> int | None:
    """Kernel format code of a natively quantized checkpoint tensor, None for anything else"""
    if weight.dtype in (torch.int8, torch.uint8): return FMT_E2M1
    if weight.dtype == torch.float8_e4m3fn: return FMT_E4M3
    if weight.dtype == torch.float8_e5m2: return FMT_E5M2
    return None


class PackedSegment:
    """One checkpoint tensor (or part) of a packed linear: `packed` bytes in checkpoint (out, in) orientation
    (or (in, out) when src_transposed, fp8 only), its block scale grid and the output-feature column it
    starts at in the layer's (in, out) weight"""

    def __init__(self, packed: torch.Tensor, scale: torch.Tensor | None, fmt: int, col0: int = 0,
                 src_transposed: bool = False, block: tuple[int, int] | None = None):
        """block: (out features, in features) per scale entry; None divides the weight by the grid exactly. Given
        explicitly for formats with a fixed block whose last block may be partial (128 x 128 weight_scale_inv)"""
        self.packed = packed.contiguous()
        self.fmt = fmt
        self.col0 = col0
        self.src_transposed = src_transposed
        if scale is not None:
            scale = scale.contiguous()
            if scale.dtype != torch.uint8:
                scale = scale.float()
            if scale.dim() == 0:
                scale = scale.view(1, 1)
            elif scale.dim() == 1:
                scale = scale.view(-1, 1)
        self.scale = scale
        if src_transposed:
            self.cols, self.rows = self.packed.shape
        else:
            self.rows = self.packed.shape[0]
            self.cols = self.packed.shape[1] * (2 if fmt == FMT_E2M1 else 1)
        self.br = self.bc = 0
        if self.scale is not None:
            sr, sc = self.scale.shape
            if block is not None:
                self.br, self.bc = block
            else:
                assert self.rows % sr == 0 and self.cols % sc == 0, \
                    f"packed weight {self.rows}x{self.cols} not divisible by scale grid {sr}x{sc}"
                self.br, self.bc = self.rows // sr, self.cols // sc
            assert self.br * sr >= self.rows and self.bc * sc >= self.cols, "scale grid does not cover the weight"

    def nbytes(self) -> int:
        return self.packed.numel() + (self.scale.numel() * self.scale.element_size() if self.scale is not None else 0)

    def to(self, device):
        self.packed = self.packed.to(device)
        if self.scale is not None:
            self.scale = self.scale.to(device)

    def dequant_torch(self, global_scale: float) -> torch.Tensor:
        """(rows, cols) fp16 in checkpoint orientation, the arithmetic of the eager loaders"""
        pk = self.packed
        if self.fmt == FMT_E2M1:
            u8 = pk.view(torch.uint8)
            lut = torch.tensor(MXFP4_LUT, dtype = torch.float, device = pk.device)
            w = torch.stack((lut[(u8 & 0x0F).long()], lut[(u8 >> 4).long()]), dim = -1).view(pk.shape[0], -1)
        else:
            w = pk.view(torch.float8_e4m3fn if self.fmt == FMT_E4M3 else torch.float8_e5m2).float()
            if self.src_transposed:
                w = w.T
        rows, cols = w.shape
        if self.scale is not None:
            s_ = (self.scale.float() - 127.0).exp2() if self.scale.dtype == torch.uint8 else self.scale.float()
            s_ = s_.repeat_interleave(self.br, 0)[:rows].repeat_interleave(self.bc, 1)[:, :cols]
            w = w * s_
        w = w.half()
        if global_scale != 1.0:
            w = (w.float() * global_scale).half()
        return w


class LinearFP16Packed(LinearFP16):
    """
    Conversion-time variant of LinearFP16 for natively quantized checkpoint tensors (fp8 e4m3 / e5m2 with
    per-tensor, per-row or block scales; packed fp4 e2m1 with E8M0 block scales, as DeepSeek-V4, MiMo and
    gpt-oss store them): the bytes stay packed as stored and are dequantized to fp16 only when a projection
    is used, one at a time, so a layer of hundreds of experts occupies a quarter (fp4) or half (fp8) of the
    VRAM its fp16 form would. Lossless: every consumer sees the fp16 weight the eager load produces. Not a
    runtime path (no graph object, no TP export); the converter enables it on the config.
    """

    quant_type: str = "fp16"

    def __init__(
        self,
        in_features: int,
        out_features: int,
        segments: list,
        bias: torch.Tensor | None,
        pad_out,
        global_scale: float = 1.0,
        out_dtype: torch.dtype | None = None,
        key: str | None = None,
    ):
        self.segments = segments
        self.global_scale = global_scale
        self.pad_out = pad_out
        self._pinned_store = None
        if bias is not None and bias.dtype == torch.float: bias = bias.to(torch.half)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.swap_device = None
        self.full_in_features = None
        self.full_out_features = None
        self.first_in_feature = None
        self.first_out_feature = None
        self.out_dtype = out_dtype
        self.key = key
        self.bc = None
        self.rows_real = max(seg.cols for seg in segments)                     # in features covered
        self.cols_real = max(seg.col0 + seg.rows for seg in segments)          # out features covered

    @property
    def weight(self) -> torch.Tensor:
        return self.get_weight_tensor()

    @weight.setter
    def weight(self, value):
        raise AttributeError("LinearFP16Packed holds packed weights; use set_weight on a plain LinearFP16")

    @property
    def device(self):
        return self.segments[0].packed.device

    def storage_bytes(self) -> int:
        return sum(seg.nbytes() for seg in self.segments) \
            + (self.bias.numel() * self.bias.element_size() if self.bias is not None else 0)

    def _dequant_into(self, out: torch.Tensor) -> torch.Tensor:
        """Write the fp16 weight into out, an (in_features, out_features) buffer on the segments' device
        whose pad rows/cols are already zero"""
        dev = self.device
        if dev.type == "cuda" and hasattr(ext, "dequant_blocks"):
            for seg in self.segments:
                ext.dequant_blocks(seg.packed.view(torch.uint8), seg.scale, out, seg.fmt, self.global_scale,
                                   seg.col0, seg.src_transposed, seg.br, seg.bc)
            return out
        for seg in self.segments:
            w = seg.dequant_torch(self.global_scale).T       # (in, out) part
            out[:w.shape[0], seg.col0:seg.col0 + w.shape[1]].copy_(w)
        return out

    def _zero_pads(self, w: torch.Tensor):
        if self.rows_real < self.in_features: w[self.rows_real:].zero_()
        if self.cols_real < self.out_features: w[:, self.cols_real:].zero_()

    def _scratch(self, device) -> torch.Tensor:
        # Through the caching allocator, rounded up to a power of two so the freed block serves the next
        # projection's dequant regardless of its exact shape (weight-sized, so not a kept static)
        numel = self.in_features * self.out_features
        w = torch.empty((1 << max(numel - 1, 0).bit_length(),), dtype = torch.half, device = device)
        w = w[:numel].view(self.in_features, self.out_features)
        self._zero_pads(w)
        return w

    def staged_weight(self) -> torch.Tensor:
        return self._dequant_into(self._scratch(self.device))

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.out_features,)
        x = x.view(-1, x.shape[-1])
        dtype = out_dtype or self.out_dtype or torch.half
        y = torch.empty((x.shape[0], self.out_features), dtype = dtype, device = x.device)
        weight = self._dequant_into(self._scratch(x.device))
        if dtype == x.dtype:
            torch.matmul(x, weight, out = y)
        else:
            ext.hgemm(x, weight, y)
        if self.bias is not None:
            y += self.bias
        return y.view(out_shape)

    def get_weight_tensor(self) -> torch.Tensor:
        """A fresh fp16 (in_features, out_features) weight on the device the packed data lives (or lived, when
        swapped: the packed bytes are moved back for the dequant and parked again)"""
        swapped = self.swap_device
        if swapped is not None:
            for seg in self.segments: seg.to(swapped)
        w = torch.zeros((self.in_features, self.out_features), dtype = torch.half, device = self.device)
        self._dequant_into(w)
        if swapped is not None:
            for seg in self.segments: seg.to("cpu")
        return w

    def reconstruct_columns(self, cols: torch.Tensor, out: torch.Tensor):
        torch.index_select(self.get_weight_tensor(), 1, cols, out = out)

    def get_tensors(self, key: str):
        t = {f"{key}.weight": self.get_weight_tensor().T.contiguous()}
        if self.bias is not None:
            t[f"{key}.bias"] = self.bias
        return t

    def set_weight(self, w: torch.Tensor):
        raise NotImplementedError("LinearFP16Packed holds packed weights")

    def swap_cpu(self):
        if self.swap_device is not None:
            return
        self.swap_device = self.device
        for seg in self.segments: seg.to("cpu")
        if self.bias is not None:
            self.bias = self.bias.cpu()

    def unswap_cpu(self):
        if self.swap_device is None:
            return
        for seg in self.segments: seg.to(self.swap_device)
        if self.bias is not None:
            self.bias = self.bias.to(self.swap_device)
        self.swap_device = None

    def tp_export(self, plan, producer):
        raise NotImplementedError("LinearFP16Packed is a conversion-time storage form")


class LinearFP16_torch:

    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ):
        self.nn_linear = nn.Linear(
            in_features,
            out_features,
            bias is not None,
            device = "meta"
        )
        self.nn_linear.weight = nn.Parameter(weight, requires_grad = False)
        if bias is not None:
            self.nn_linear.bias = nn.Parameter(bias, requires_grad = False)

    def get_tensors(self, key: str):
        t = {}
        t[f"{key}.weight"] = self.nn_linear.weight.data.contiguous()
        if self.nn_linear.bias is not None:
            t[f"{key}.bias"] = self.nn_linear.bias.data.contiguous()
        return t

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        x = self.nn_linear.forward(x)
        return to2(x, out_dtype)

    def get_weight_tensor(self) -> torch.Tensor:
        return self.nn_linear.weight.data.T

    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.nn_linear.bias.data if self.nn_linear.bias is not None else None

    def set_weight(self, w):
        self.nn_linear.weight = nn.Parameter(w.T.half(), requires_grad = False)
