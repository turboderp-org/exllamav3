"""Pure-PyTorch fallback implementations of C++ extension functions.

Used on ROCm where the CUDA-specific kernels (activation.cu, norm.cu, etc.) are
excluded from the build. Each function matches the signature of its C++ counterpart
so it can be monkey-patched onto the extension module transparently.

These are written for correctness, not performance — they use standard PyTorch ops
that compose naturally with CUDA graph capture and (potentially) torch.compile.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


# -- Activation fused ops (activation.cu) -------------------------------------

def _act_mul(
    activation: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float,
) -> None:
    """Match act_mul_kernel_{h,f}'s dtype and clamp contract."""
    if x.dtype not in (torch.float16, torch.float32) or y.dtype != x.dtype:
        raise TypeError("activation multiply expects matching float16 or float32 inputs")
    if z.dtype != torch.float16:
        raise TypeError("activation multiply expects float16 output")

    x_act = activation(x)
    if act_limit != 0.0:
        x_act = x_act.clamp(max = act_limit)
        y = y.clamp(min = -act_limit, max = act_limit)
    result = x_act * y
    # The float-input native kernels saturate before the float-to-half store.
    if x.dtype == torch.float32:
        result = result.clamp(min = -65504.0, max = 65504.0)
    z.copy_(result)


def silu_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    _act_mul(F.silu, x, y, z, act_limit)

def silu_oai_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    # GPT-OSS clamped SwiGLU: clamp the gate from above and the up projection
    # symmetrically before applying the 1.702-scaled sigmoid and up-path bias.
    if x.dtype not in (torch.float16, torch.float32) or y.dtype != x.dtype or z.dtype != torch.float16:
        raise TypeError("silu_oai_mul expects matching float16/float32 inputs and float16 output")
    gate = x.float()
    up = y.float()
    if act_limit != 0.0:
        gate = gate.clamp(max = act_limit)
        up = up.clamp(min = -act_limit, max = act_limit)
    r = (up + 1.0) * gate * torch.sigmoid(1.702 * gate)
    if x.dtype == torch.float32 and z.dtype == torch.float16:
        r = r.clamp(min = -65504.0, max = 65504.0)
    z.copy_(r)

def gelu_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    _act_mul(lambda t: F.gelu(t, approximate = "tanh"), x, y, z, act_limit)


def relu2_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    _act_mul(lambda t: torch.square(F.relu(t)), x, y, z, act_limit)


def relu_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    _act_mul(F.relu, x, y, z, act_limit)

def xielu(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha_p: torch.Tensor,
    alpha_n: torch.Tensor,
) -> None:
    if x.dtype != torch.float32 or y.dtype != torch.float16:
        raise TypeError("xielu expects float32 input and float16 output")
    if any(a.device.type != "cpu" or a.numel() < 1 for a in (alpha_p, alpha_n)):
        raise ValueError("alpha_p and alpha_n must be nonempty CPU tensors")
    if any(a.dtype not in (torch.float32, torch.float16, torch.bfloat16) for a in (alpha_p, alpha_n)):
        raise TypeError("unsupported alpha_p or alpha_n dtype")
    p = F.softplus(alpha_p.reshape(-1)[0].float(), threshold = 20.0).item()
    n = F.softplus(alpha_n.reshape(-1)[0].float(), threshold = 20.0).item() + 0.5
    eps = torch.scalar_tensor(-9.9838e-7, dtype = torch.float32, device = x.device)
    r = torch.where(
        x > 0,
        p * x.square() + 0.5 * x,
        (torch.expm1(torch.minimum(x, eps)) - x) * n + 0.5 * x,
    )
    y.copy_(r.clamp(min = -65504.0, max = 65504.0))


# -- Packed logit bitmask (generator/sampling_fused.cu) -----------------------

def apply_logit_bitmask(
    logits_in: torch.Tensor,
    logits_out: torch.Tensor,
    bitmask: torch.Tensor,
) -> None:
    """Copy allowed packed-mask logits and write ``-inf`` everywhere else."""
    if logits_in.ndim != 2 or not logits_in.is_contiguous():
        raise ValueError("apply_logit_bitmask: logits must be a contiguous 2-D tensor")
    if not logits_out.is_contiguous():
        raise ValueError("apply_logit_bitmask: output must be contiguous")
    if logits_out.shape != logits_in.shape:
        raise ValueError("apply_logit_bitmask: shape mismatch")
    if logits_out.dtype != logits_in.dtype:
        raise TypeError("apply_logit_bitmask: dtype mismatch")
    if logits_in.dtype not in (torch.float16, torch.float32):
        raise TypeError("apply_logit_bitmask: logits must be half or float")
    if bitmask.dtype != torch.int32 or bitmask.ndim != 2 or not bitmask.is_contiguous():
        raise ValueError("apply_logit_bitmask: bitmask must be a contiguous 2-D int32 tensor")
    if bitmask.shape[0] not in (1, logits_in.shape[0]):
        raise ValueError("apply_logit_bitmask: bad bitmask batch dim")
    if not (logits_in.is_cuda and logits_out.is_cuda and bitmask.is_cuda):
        raise ValueError("apply_logit_bitmask: tensors must be GPU-local")
    if logits_in.device != logits_out.device or logits_in.device != bitmask.device:
        raise ValueError("apply_logit_bitmask: tensors must be on the same device")

    rows, dim = logits_in.shape
    indices = torch.arange(dim, device = logits_in.device, dtype = torch.long)
    word_indices = indices >> 5
    bit_indices = indices & 31
    mask_rows = bitmask if bitmask.shape[0] == rows else bitmask.expand(rows, -1)
    in_width = word_indices < bitmask.shape[1]
    keep = torch.zeros((rows, dim), dtype = torch.bool, device = logits_in.device)
    if in_width.any():
        words = mask_rows[:, word_indices[in_width]].to(torch.int64)
        keep[:, in_width] = ((words >> bit_indices[in_width]) & 1).bool()
    logits_out.copy_(logits_in)
    logits_out.masked_fill_(~keep, float("-inf"))


# -- In-place gate ops (activation.cu) -----------------------------------------

def mul_sigmoid_(o: torch.Tensor, g: torch.Tensor) -> None:
    o.mul_(torch.sigmoid(g))

def _validate_broadcast_gate(o: torch.Tensor, g: torch.Tensor) -> None:
    if o.dtype != torch.float16 or g.dtype != torch.float16:
        raise TypeError("broadcast gate expects float16 tensors")
    if o.ndim != 4 or g.ndim != 3 or o.shape[:-1] != g.shape:
        raise ValueError("broadcast gate expects x [B, S, H, D] and gate [B, S, H]")


def mul_sigmoid_broadcast_(o: torch.Tensor, g: torch.Tensor) -> None:
    _validate_broadcast_gate(o, g)
    o.mul_(torch.sigmoid(g).unsqueeze(-1))


def mul_softplus_broadcast_(o: torch.Tensor, g: torch.Tensor) -> None:
    _validate_broadcast_gate(o, g)
    o.copy_((o.float() * F.softplus(g.float()).unsqueeze(-1)).to(o.dtype))


def add_sigmoid_gate(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> None:
    if any(t.dtype != torch.float32 for t in (x, y, z)):
        raise TypeError("add_sigmoid_gate expects float32 tensors")
    if y.shape[-1:] != (1,) or x.shape[:-1] != y.shape[:-1] or z.shape != x.shape:
        raise ValueError("add_sigmoid_gate expects x/z [..., D] and gate [..., 1]")
    z.add_(x * torch.sigmoid(y))


def add_sigmoid_gate_proj(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    w: torch.Tensor,
) -> None:
    if x.dtype != torch.float32 or z.dtype != torch.float32 or y.dtype != torch.float16 or w.dtype != torch.float16:
        raise TypeError("add_sigmoid_gate_proj expects float32 x/z and float16 y/w")
    if x.shape != y.shape or z.shape != x.shape or w.ndim != 2 or w.shape != (x.shape[-1], 1):
        raise ValueError("add_sigmoid_gate_proj expects x/y/z [..., D] and w [D, 1]")
    gate = torch.matmul(y.float(), w.float())
    z.add_(x * torch.sigmoid(gate))


# -- Attention helpers (activation.cu) ----------------------------------------

def deinterleave_qg(
    qg: torch.Tensor,
    q: torch.Tensor,
    g: torch.Tensor,
    head_dim: int,
) -> None:
    if any(t.dtype != torch.float16 for t in (qg, q, g)):
        raise TypeError("deinterleave_qg expects float16 tensors")
    if head_dim <= 0 or q.numel() != g.numel() or q.numel() * 2 != qg.numel():
        raise ValueError("deinterleave_qg size mismatch")
    if qg.shape[-1] % (2 * head_dim):
        raise ValueError("qg last dimension must contain whole interleaved heads")
    heads = qg.shape[-1] // (2 * head_dim)
    chunks = qg.reshape(*qg.shape[:-1], heads, 2 * head_dim)
    q.copy_(chunks[..., :head_dim].reshape_as(q))
    g.copy_(chunks[..., head_dim:].reshape_as(g))


# -- Norm ops (norm.cu) --------------------------------------------------------

def rms_norm(
    x: torch.Tensor,
    w: torch.Tensor | None,
    y: torch.Tensor,
    eps: float,
    constant_bias: float,
    constant_scale: float,
    span_heads: bool,
    add_residual: bool,
    w_groups: int = 1,
) -> None:
    if x.dtype not in (torch.float16, torch.float32) or y.dtype not in (torch.float16, torch.float32):
        raise TypeError("rms_norm expects float16/float32 input and output")
    if w is not None and w.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("rms_norm expects float16/bfloat16 weight")
    if x.shape != y.shape or w_groups < 1:
        raise ValueError("rms_norm input/output shape or w_groups mismatch")
    if span_heads and w_groups != 1:
        raise ValueError("rms_norm w_groups and span_heads are exclusive")
    xn = x.flatten(-2) if span_heads else x
    yn = y.flatten(-2) if span_heads else y
    dim = xn.shape[-1]
    rows = xn.numel() // dim
    xf = xn.reshape(rows, dim).float()
    if x.dtype == torch.float16:
        xf = xf.clamp(min = -65504.0, max = 65504.0)
    xf = xf * torch.rsqrt(xf.square().mean(dim = -1, keepdim = True) + eps)
    xf = xf * constant_scale
    if w is not None:
        wf = w.float()
        if constant_bias != 0.0:
            wf = wf + constant_bias
        if w_groups == 1:
            if w.numel() != dim:
                raise ValueError("rms_norm weight must have dim elements")
            xf = xf * wf.reshape(1, dim)
        else:
            if w.numel() != w_groups * dim:
                raise ValueError("rms_norm weight must have w_groups * dim elements")
            row_groups = torch.arange(rows, device = x.device) % w_groups
            xf = xf * wf.reshape(w_groups, dim).index_select(0, row_groups)
    result = xf.reshape_as(yn)
    if add_residual:
        yn.copy_((yn.float() + result).to(yn.dtype))
    else:
        yn.copy_(result.to(yn.dtype))

def rms_norm_res_in(
    x: torch.Tensor,
    w: torch.Tensor | None,
    y: torch.Tensor,
    r: torch.Tensor,
    eps: float,
    constant_bias: float,
    constant_scale: float,
) -> None:
    if x.dtype not in (torch.float16, torch.float32) or r.dtype not in (torch.float16, torch.float32):
        raise TypeError("rms_norm_res_in expects float16/float32 input and residual")
    if y.dtype != torch.float16 or (w is not None and w.dtype not in (torch.float16, torch.bfloat16)):
        raise TypeError("rms_norm_res_in expects float16 output and float16/bfloat16 weight")
    if x.shape != y.shape or x.shape != r.shape:
        raise ValueError("rms_norm_res_in tensor shapes must match")
    x_add = x.float()
    if x.dtype == torch.float16:
        x_add = x_add.clamp(min = -65504.0, max = 65504.0)
    r.copy_((r.float() + x_add).to(r.dtype))
    rf = r.float()
    if w is not None:
        if w.numel() != x.shape[-1]:
            raise ValueError("rms_norm_res_in weight must have dim elements")
        wf = w.float() + constant_bias if constant_bias != 0.0 else w.float()
    else:
        wf = None
    var = rf.pow(2).mean(dim = -1, keepdim = True) + eps
    rf = rf * torch.rsqrt(var) * constant_scale
    if wf is not None:
        rf = rf * wf
    y.copy_(rf.to(y.dtype))

def gated_rms_norm(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    eps: float,
    constant_bias: float,
    w_groups: int,
    gate_first: bool,
    gate_act: int = 0,
) -> None:
    if x.dtype != torch.bfloat16 or y.dtype not in (torch.float16, torch.float32):
        raise TypeError("gated_rms_norm expects bfloat16 input and float16/float32 output")
    if w.dtype not in (torch.bfloat16, torch.float32) or g.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("gated_rms_norm weight/gate dtype mismatch")
    if x.shape != y.shape or x.shape != g.shape or w_groups < 1:
        raise ValueError("gated_rms_norm input/output shape or w_groups mismatch")
    if gate_act not in (0, 1):
        raise ValueError("gated_rms_norm gate_act must be 0 (SiLU) or 1 (sigmoid)")
    dim = x.shape[-1]
    rows = x.numel() // dim
    if w.numel() != w_groups * dim:
        raise ValueError("gated_rms_norm weight must have w_groups * dim elements")

    xf = x.reshape(rows, dim).float()
    gf = g.reshape(rows, dim).float()
    gate = torch.sigmoid(gf) if gate_act == 1 else F.silu(gf)
    norm_input = xf * gate if gate_first else xf
    hidden = norm_input * torch.rsqrt(
        norm_input.square().mean(dim = -1, keepdim = True) + eps
    )
    wf = w.float().reshape(w_groups, dim)
    if constant_bias != 0.0:
        wf = wf + constant_bias
    row_groups = torch.arange(rows, device = x.device) % w_groups
    hidden = hidden * wf.index_select(0, row_groups)
    if not gate_first:
        hidden = hidden * gate
    y.copy_(hidden.reshape_as(y).to(y.dtype))


# -- Softcap (softcap.cu) ------------------------------------------------------

def softcap(x: torch.Tensor, y: torch.Tensor, cap: float) -> None:
    if x.dtype not in (torch.float16, torch.float32) or y.dtype != x.dtype:
        raise TypeError("softcap expects matching float16 or float32 tensors")
    if x.shape != y.shape:
        raise ValueError("softcap input and output shapes must match")
    y.copy_(torch.tanh(x.float() / cap).mul_(cap).to(y.dtype))


# -- Quantized cache (cache/q_cache.cu) ----------------------------------------

def _hadamard32(x: torch.Tensor) -> torch.Tensor:
    """Apply the unnormalised H32 used by the cache quantizer to its last axis."""
    for width in (1, 2, 4, 8, 16):
        x = x.reshape(*x.shape[:-1], -1, 2, width)
        lo, hi = x.unbind(dim = -2)
        x = torch.stack((lo + hi, lo - hi), dim = -2).reshape(*x.shape[:-3], -1)
    return x


_CACHE_GROUP_CHUNK = 4096
_CACHE_ROW_CHUNK = 8192
_DSA_TOPK_ROW_CHUNK = 8


def _f32_scalar(value: float, like: torch.Tensor) -> torch.Tensor:
    return torch.scalar_tensor(value, dtype = torch.float32, device = like.device)


def _cache_codes(values: torch.Tensor, bits: int, compand_a: float) -> torch.Tensor:
    """Encode normalized H32 values with the q-cache float32 operation order.

    This follows ``lmq.cuh``'s float32 FMA/Cardano path, but PyTorch does not promise
    the CUDA kernel's exact cbrt or FMA instruction selection across backends.
    """
    midpoint = _f32_scalar(1 << (bits - 1), values)
    if compand_a > 0.0:
        a = _f32_scalar(compand_a, values)
        b = _f32_scalar(1.0, values) - a
        inv_b = _f32_scalar(1.0, values) / b
        p3 = (a * inv_b) * _f32_scalar(1.0 / 3.0, values)
        p3_cub = (p3 * p3) * p3
        q_half = (values * inv_b) * _f32_scalar(0.5, values)
        delta = torch.addcmul(p3_cub, q_half, q_half)
        root_delta = torch.sqrt(delta)
        exponent = _f32_scalar(1.0 / 3.0, values)
        cbrt_plus = torch.sign(q_half + root_delta) * torch.abs(q_half + root_delta).pow(exponent)
        cbrt_minus = torch.sign(q_half - root_delta) * torch.abs(q_half - root_delta).pow(exponent)
        root = cbrt_plus + cbrt_minus
        codes = torch.floor(torch.addcmul(midpoint, root, midpoint))
    else:
        codes = torch.floor(torch.addcmul(midpoint, values, midpoint))
    return codes.clamp_(0, (1 << bits) - 1).to(torch.int64)


def _pack_cache_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack 32 codes as the cache's high-to-low power-of-two bit planes."""
    planes = []
    remaining = bits
    for width in (8, 4, 2, 1):
        if not bits & width:
            continue
        remaining -= width
        fields = (codes >> remaining) & ((1 << width) - 1)
        values_per_word = 32 // width
        fields = fields.reshape(*fields.shape[:-1], width, values_per_word)
        shifts = torch.arange(values_per_word, device = codes.device, dtype = torch.int64) * width
        planes.append(torch.sum(fields << shifts, dim = -1))
    return torch.cat(planes, dim = -1).to(torch.int32)


def _unpack_cache_codes(words: torch.Tensor, bits: int) -> torch.Tensor:
    codes = torch.zeros(*words.shape[:-1], 32, device = words.device, dtype = torch.int64)
    word_base = 0
    for width in (8, 4, 2, 1):
        if not bits & width:
            continue
        values_per_word = 32 // width
        plane = words[..., word_base : word_base + width]
        shifts = torch.arange(values_per_word, device = words.device, dtype = torch.int64) * width
        plane = (plane.unsqueeze(-1) >> shifts) & ((1 << width) - 1)
        codes = (codes << width) | plane.reshape(*plane.shape[:-2], 32)
        word_base += width
    return codes


def _cont_cache_layout(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor) -> tuple[int, int]:
    if x.ndim < 1 or packed.ndim < 1 or not (x.is_contiguous() and packed.is_contiguous() and scales.is_contiguous()):
        raise ValueError("quantized-cache tensors must be contiguous and have a last dimension")
    head_dim = x.shape[-1]
    if head_dim == 0 or head_dim % 32:
        raise ValueError("head_dim must be a nonzero multiple of 32")
    groups = head_dim // 32
    bits = packed.shape[-1] // groups
    if not 2 <= bits <= 8 or packed.numel() != x.numel() // 32 * bits or scales.numel() != x.numel() // 32:
        raise ValueError("invalid quantized-cache shape or bitrate")
    return groups, bits


def quant_cache_cont(
    x: torch.Tensor,
    out: torch.Tensor,
    out_scales: torch.Tensor,
    compand_a: float = 0.0,
) -> None:
    """H32 cache quantization with CUDA-compatible packed planes, in bounded chunks."""
    if x.dtype != torch.float16 or out.dtype != torch.int32 or out_scales.dtype != torch.float16:
        raise TypeError("quant_cache_cont expects fp16 input/scales and int32 output")
    _, bits = _cont_cache_layout(x, out, out_scales)
    flat_x = x.reshape(-1, 32)
    flat_out = out.reshape(-1, bits)
    flat_scales = out_scales.reshape(-1)
    r32 = _f32_scalar(0.17677669529663688110, x)
    epsilon = _f32_scalar(1.0e-10, x)
    for first in range(0, flat_x.shape[0], _CACHE_GROUP_CHUNK):
        last = min(first + _CACHE_GROUP_CHUNK, flat_x.shape[0])
        rotated = _hadamard32(flat_x[first:last].float()) * r32
        scales = rotated.abs().amax(dim = -1) + epsilon
        codes = _cache_codes(rotated * scales.reciprocal().unsqueeze(-1), bits, compand_a)
        flat_out[first:last].copy_(_pack_cache_codes(codes, bits))
        flat_scales[first:last].copy_(scales.to(torch.float16))


def dequant_cache_cont(
    x: torch.Tensor,
    in_scales: torch.Tensor,
    out: torch.Tensor,
    compand_a: float = 0.0,
) -> None:
    """Inverse of quant_cache_cont, in bounded chunks and without host transfers."""
    if x.dtype != torch.int32 or in_scales.dtype != torch.float16 or out.dtype != torch.float16:
        raise TypeError("dequant_cache_cont expects int32 input and fp16 scales/output")
    _, bits = _cont_cache_layout(out, x, in_scales)
    flat_x = x.reshape(-1, bits)
    flat_scales = in_scales.reshape(-1)
    flat_out = out.reshape(-1, 32)
    r32 = _f32_scalar(0.17677669529663688110, out)
    for first in range(0, flat_x.shape[0], _CACHE_GROUP_CHUNK):
        last = min(first + _CACHE_GROUP_CHUNK, flat_x.shape[0])
        words = flat_x[first:last].to(torch.int64) & 0xffffffff
        codes = _unpack_cache_codes(words, bits).float()
        if compand_a > 0.0:
            a = _f32_scalar(compand_a, codes)
            b = _f32_scalar(1.0, codes) - a
            inv_n = _f32_scalar(1.0 / (1 << bits), codes)
            t = torch.addcmul(_f32_scalar(-1.0, codes), codes * 2.0 + 1.0, inv_n)
            values = t * torch.addcmul(a, t * t, b)
        else:
            midpoint = _f32_scalar(1 << (bits - 1), codes)
            values = codes - (midpoint - _f32_scalar(0.5, codes))
            values = values * ((flat_scales[first:last].float() * r32) / midpoint).unsqueeze(-1)
        if compand_a > 0.0:
            values = values * (flat_scales[first:last].float() * r32).unsqueeze(-1)
        flat_out[first:last].copy_(_hadamard32(values).to(torch.float16))


def _paged_cache_dim(x: torch.Tensor) -> int:
    if x.ndim == 4:
        return x.shape[2] * x.shape[3]
    if x.ndim == 3:
        return x.shape[2]
    raise ValueError("paged cache tensors must be 3D or 4D")


def _validate_paged_cache(
    packed: torch.Tensor, scales: torch.Tensor, out: torch.Tensor | None,
    other_packed: torch.Tensor, other_scales: torch.Tensor, other_out: torch.Tensor | None,
    cache_seqlens: torch.Tensor, block_table: torch.Tensor, page_size: int,
) -> tuple[int, int]:
    tensors = tuple(t for t in (packed, scales, out, other_packed, other_scales, other_out, cache_seqlens, block_table) if t is not None)
    if any(t.device != packed.device for t in tensors) or any(not t.is_contiguous() for t in tensors):
        raise ValueError("paged-cache tensors must be contiguous and on one device")
    if packed.dtype != torch.int32 or other_packed.dtype != torch.int32 or scales.dtype != torch.float16 or other_scales.dtype != torch.float16 or (out is not None and (out.dtype != torch.float16 or other_out is None or other_out.dtype != torch.float16)):
        raise TypeError("paged cache expects int32 payloads and fp16 scales/output")
    if cache_seqlens.dtype != torch.int32 or block_table.dtype != torch.int32 or cache_seqlens.ndim != 1 or block_table.ndim != 2:
        raise TypeError("cache_seqlens and block_table must be int32 1D/2D tensors")
    if page_size != 256 or packed.ndim != 3 or other_packed.ndim != 3 or packed.shape[:2] != other_packed.shape[:2] or packed.shape[1] != page_size:
        raise ValueError("paged quantized cache must use 256-token pages")
    dim = scales.shape[-1] * 32
    if scales.shape != packed.shape[:2] + (scales.shape[-1],) or other_scales.shape != scales.shape:
        raise ValueError("invalid paged-cache scale shape")
    groups = dim // 32
    if packed.shape[2] % groups or other_packed.shape[2] % groups or not (2 <= packed.shape[2] // groups <= 8 and 2 <= other_packed.shape[2] // groups <= 8):
        raise ValueError("invalid paged-cache bitrate")
    if out is not None and (_paged_cache_dim(out) != dim or _paged_cache_dim(other_out) != dim or out.shape != other_out.shape or out.ndim not in (3, 4) or out.shape[1] != page_size):
        raise ValueError("paged K/V output shapes do not match the cache")
    if block_table.shape[0] != cache_seqlens.shape[0]:
        raise ValueError("block table and cache lengths have different batch sizes")
    return dim, block_table.shape[1]


def _paged_positions(cache_seqlens: torch.Tensor, block_table: torch.Tensor, page_size: int, first: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    bsz = block_table.shape[0]
    batch = torch.arange(bsz, device = block_table.device, dtype = torch.long).unsqueeze(1)
    logical = torch.arange(first, first + count, device = block_table.device, dtype = torch.long).unsqueeze(0)
    logical = logical.expand(bsz, -1)
    tokens = logical + cache_seqlens.to(torch.long).unsqueeze(1)
    pages = torch.div(tokens, page_size, rounding_mode = "floor")
    mapped = block_table[batch, pages]
    physical = mapped.to(torch.long) * page_size + tokens.remainder(page_size)
    return physical.reshape(-1), logical.reshape(-1)


def _quant_cache_paged_one(source: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, source_positions: torch.Tensor, destination_positions: torch.Tensor, dim: int, compand_a: float) -> None:
    rows = source.reshape(-1, dim).index_select(0, source_positions)
    groups = dim // 32
    bits = packed.shape[-1] // groups
    q_rows = torch.empty((rows.shape[0], groups * bits), dtype = torch.int32, device = rows.device)
    q_scales = torch.empty((rows.shape[0], groups), dtype = torch.float16, device = rows.device)
    quant_cache_cont(rows, q_rows, q_scales, compand_a)
    packed.reshape(-1, groups * bits).index_copy_(0, destination_positions, q_rows)
    scales.reshape(-1, groups).index_copy_(0, destination_positions, q_scales)


def quant_cache_paged(
    k_in: torch.Tensor, k_out: torch.Tensor, k_out_scales: torch.Tensor,
    v_in: torch.Tensor, v_out: torch.Tensor, v_out_scales: torch.Tensor,
    cache_seqlens: torch.Tensor, block_table: torch.Tensor, page_size: int,
    seq_len: int, compand_a: float = 0.0, in_contiguous: bool = False,
) -> None:
    """Append K/V rows to the mapped cache pages using bounded GPU-local staging."""
    dim, _ = _validate_paged_cache(k_out, k_out_scales, None, v_out, v_out_scales, None, cache_seqlens, block_table, page_size)
    if k_in.dtype != torch.float16 or v_in.dtype != torch.float16:
        raise TypeError("paged-cache append input must be fp16")
    if seq_len < 0 or k_in.shape != v_in.shape or k_in.device != k_out.device or v_in.device != k_out.device or not (k_in.is_contiguous() and v_in.is_contiguous()) or _paged_cache_dim(k_in) != dim or _paged_cache_dim(v_in) != dim:
        raise ValueError("invalid paged-cache append input")
    bsz = block_table.shape[0]
    if (in_contiguous and (k_in.numel() < bsz * seq_len * dim or v_in.numel() < bsz * seq_len * dim)) or (not in_contiguous and (k_in.numel() < k_out.shape[0] * page_size * dim or v_in.numel() < v_out.shape[0] * page_size * dim)):
        raise ValueError("paged-cache append input is too small")
    for first in range(0, seq_len, max(1, _CACHE_ROW_CHUNK // max(bsz, 1))):
        count = min(max(1, _CACHE_ROW_CHUNK // max(bsz, 1)), seq_len - first)
        destination, logical = _paged_positions(cache_seqlens, block_table, page_size, first, count)
        batch = torch.arange(bsz, device = k_in.device, dtype = torch.long).unsqueeze(1).expand(bsz, count).reshape(-1)
        source_positions = batch * seq_len + logical if in_contiguous else destination
        _quant_cache_paged_one(k_in, k_out, k_out_scales, source_positions, destination, dim, compand_a)
        _quant_cache_paged_one(v_in, v_out, v_out_scales, source_positions, destination, dim, compand_a)


def _dequant_cache_paged(
    k_in: torch.Tensor, k_in_scales: torch.Tensor, k_out: torch.Tensor,
    v_in: torch.Tensor, v_in_scales: torch.Tensor, v_out: torch.Tensor,
    cache_seqlens: torch.Tensor, block_table: torch.Tensor, page_size: int,
    compand_a: float, sliding_window: int, bonus_len: int, compact_out: bool,
) -> None:
    dim, pages_per_seq = _validate_paged_cache(k_in, k_in_scales, k_out, v_in, v_in_scales, v_out, cache_seqlens, block_table, page_size)
    if bonus_len < 0:
        raise ValueError("bonus_len must be nonnegative")
    groups = dim // 32
    bsz = block_table.shape[0]
    required_pages = bsz * pages_per_seq if compact_out else k_in.shape[0]
    if k_out.shape[0] < required_pages or v_out.shape[0] < required_pages:
        raise ValueError("paged-cache output is too small")
    max_tokens = pages_per_seq * page_size
    chunks_per_token = (groups + 3) // 4
    tokens_per_block = 256 // chunks_per_token
    limits = (cache_seqlens.to(torch.long) + bonus_len).clamp(min = 0, max = max_tokens)
    if sliding_window > 0:
        window_starts = ((limits - sliding_window).clamp_min(0) // tokens_per_block) * tokens_per_block
    else:
        window_starts = torch.zeros_like(limits)

    # This correctness fallback trades one bounded device-to-host synchronization for loop
    # bounds that cover only the required cache extent. The lower bound retains the native
    # launch block containing each sequence's window; compact destinations remain logical.
    host_bounds = torch.stack((window_starts.amin(), limits.amax())).cpu().tolist()
    first_required, last_required = host_bounds
    rows_per_chunk = max(1, _CACHE_ROW_CHUNK // max(bsz, 1))
    for first in range(first_required, last_required, rows_per_chunk):
        count = min(rows_per_chunk, last_required - first)
        source, logical = _paged_positions(torch.zeros_like(cache_seqlens), block_table, page_size, first, count)
        batch = torch.arange(bsz, device = k_in.device, dtype = torch.long).unsqueeze(1).expand(bsz, count).reshape(-1)
        limit = limits.repeat_interleave(count)
        valid = logical < limit
        if sliding_window > 0:
            valid &= logical >= window_starts.repeat_interleave(count)
        destination = (batch * pages_per_seq * page_size + logical) if compact_out else source
        for packed, scales, out in ((k_in, k_in_scales, k_out), (v_in, v_in_scales, v_out)):
            bits = packed.shape[-1] // groups
            q_rows = packed.reshape(-1, groups * bits).index_select(0, source)
            s_rows = scales.reshape(-1, groups).index_select(0, source)
            decoded = torch.empty((q_rows.shape[0], dim), dtype = torch.float16, device = packed.device)
            dequant_cache_cont(q_rows, s_rows, decoded, compand_a)
            flat_out = out.reshape(-1, dim)
            preserved = flat_out.index_select(0, destination)
            flat_out.index_copy_(0, destination, torch.where(valid.unsqueeze(1), decoded, preserved))


def dequant_cache_paged(
    k_in: torch.Tensor, k_in_scales: torch.Tensor, k_out: torch.Tensor,
    v_in: torch.Tensor, v_in_scales: torch.Tensor, v_out: torch.Tensor,
    cache_seqlens: torch.Tensor, block_table: torch.Tensor, page_size: int,
    sliding_window: int = -1, compand_a: float = 0.0,
) -> None:
    """Dequantize referenced cache rows in place; a positive window limits logical tokens."""
    _dequant_cache_paged(k_in, k_in_scales, k_out, v_in, v_in_scales, v_out, cache_seqlens, block_table, page_size, compand_a, sliding_window, 0, False)


def dequant_cache_paged_window(
    k_in: torch.Tensor, k_in_scales: torch.Tensor, k_out: torch.Tensor,
    v_in: torch.Tensor, v_in_scales: torch.Tensor, v_out: torch.Tensor,
    cache_seqlens: torch.Tensor, block_table: torch.Tensor, page_size: int,
    bonus_len: int, compand_a: float = 0.0,
) -> None:
    """Dequantize mapped pages into compact batch/page scratch storage in place."""
    _dequant_cache_paged(k_in, k_in_scales, k_out, v_in, v_in_scales, v_out, cache_seqlens, block_table, page_size, compand_a, -1, bonus_len, True)


# -- Quantization utilities (quant/util.cu) ------------------------------------


def count_inf_nan(x: torch.Tensor, counts: torch.Tensor) -> None:
    """Accumulate infinity and NaN counts into a two-element int64 tensor."""
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError("count_inf_nan expects fp16 or fp32 input")
    if counts.dtype != torch.int64 or counts.numel() != 2:
        raise TypeError("count_inf_nan expects a two-element int64 output")
    if counts.device != x.device:
        raise ValueError("count_inf_nan expects input and output on the same device")
    counts.add_(torch.stack((torch.isinf(x).sum(), torch.isnan(x).sum())))


# -- DSA top-k (dsa_topk.cu) ---------------------------------------------------


def dsa_topk_gfx12_supported(
    scores: torch.Tensor,
    indices: torch.Tensor,
    k: int,
    t_ptr: torch.Tensor | None = None,
    t_seq: int = 0,
) -> bool:
    """Whether this is the fixed-K QSA shape handled by the gfx12 HIP kernel."""
    if not torch.version.hip or t_ptr is not None or t_seq != 0:
        return False
    if not isinstance(scores, torch.Tensor) or not isinstance(indices, torch.Tensor):
        return False
    if scores.device.type != "cuda" or indices.device != scores.device:
        return False
    if scores.dtype != torch.float16 or indices.dtype != torch.int32:
        return False
    if scores.ndim != 2 or indices.ndim != 2 or scores.shape[0] != indices.shape[0]:
        return False
    if k != 512 or indices.shape[1] != 512 or scores.shape[1] < 512:
        return False
    if (scores.shape[1] > 2**31 - 1 or scores.stride(0) > 2**31 - 1 or
            scores.stride(1) != 1 or scores.stride(0) < scores.shape[1] or scores.stride(0) % 128):
        return False
    if not indices.is_contiguous():
        return False
    device = torch.cuda.current_device() if scores.device.index is None else scores.device.index
    props = torch.cuda.get_device_properties(device)
    arch = getattr(props, "gcnArchName", "").split(":", 1)[0]
    return arch in ("gfx1200", "gfx1201") and getattr(props, "warp_size", 0) == 32


def dsa_topk(
    scores: torch.Tensor,
    indices: torch.Tensor,
    k: int,
    t_ptr: torch.Tensor | None = None,
    t_seq: int = 0,
) -> None:
    """Write CUDA-ordered fp16 top-k indices and -1 padding in bounded row chunks."""
    if scores.dtype != torch.float16 or indices.dtype != torch.int32:
        raise TypeError("dsa_topk expects fp16 scores and int32 indices")
    if scores.ndim != 2 or indices.ndim != 2 or scores.shape[0] != indices.shape[0] or indices.device != scores.device:
        raise ValueError("dsa_topk expects scores and indices with matching rows on one device")
    if scores.stride(1) != 1 or not indices.is_contiguous() or not 0 <= k <= indices.shape[1]:
        raise ValueError("dsa_topk output shape or layout mismatch")
    if t_ptr is not None and (t_ptr.dtype != torch.int32 or t_ptr.device != scores.device or t_ptr.numel() < (1 if t_seq <= 0 else (scores.shape[0] + t_seq - 1) // t_seq)):
        raise ValueError("dsa_topk t_ptr must provide int32 bounds on the score device")

    rows, width = scores.shape
    indices.fill_(-1)
    if k == 0 or width == 0:
        return
    columns = torch.arange(width, device = scores.device)
    for first in range(0, rows, _DSA_TOPK_ROW_CHUNK):
        last = min(first + _DSA_TOPK_ROW_CHUNK, rows)
        chunk_scores = scores[first:last]
        if t_ptr is None:
            bounds = torch.full((last - first,), width, device = scores.device, dtype = torch.long)
        elif t_seq > 0:
            bounds = t_ptr.reshape(-1)[torch.arange(first, last, device = scores.device) // t_seq].long()
        else:
            bounds = t_ptr.reshape(-1)[0].expand(last - first).long()
        bounds = bounds.clamp_(0, width)
        # dsa_topk.cu compares the monotonic fp16 bit key: -inf and negative NaNs are
        # excluded (key <= 0x03ff); positive NaNs sort above +inf; +0 sorts above -0.
        raw = chunk_scores.view(torch.int16).to(torch.int32) & 0xffff
        keys = torch.where((raw & 0x8000) != 0, (~raw) & 0xffff, raw | 0x8000)
        valid = (columns.unsqueeze(0) < bounds.unsqueeze(1)) & (keys > 0x03ff)
        ranked = keys.masked_fill(~valid, -1)
        threshold = torch.topk(ranked, min(k, width), dim = 1).values[:, -1:]
        count = valid.sum(dim = 1, keepdim = True)
        above = valid & (keys > threshold)
        ties = valid & (keys == threshold)
        ties &= ties.cumsum(dim = 1) <= (k - above.sum(dim = 1, keepdim = True))
        selected = torch.where(count < k, valid, above | ties)
        rank = selected.cumsum(dim = 1) - 1
        row, column = selected.nonzero(as_tuple = True)
        indices[first + row, rank[row, column]] = column.to(torch.int32)


# -- Sentinel for missing BC_* classes -----------------------------------------

class _BCNone:
    """Callable that returns None, used as stand-in for missing BC_* constructors."""
    __slots__ = ()
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None
