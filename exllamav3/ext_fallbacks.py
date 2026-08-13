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

def silu_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    r = F.silu(x) * y
    if act_limit != 0.0:
        r = torch.clamp(r, min = -act_limit, max = act_limit)
    z.copy_(r)

def silu_oai_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    # OAI variant: silu(x * y) — see activation.cu
    r = F.silu(x * y)
    if act_limit != 0.0:
        r = torch.clamp(r, min = -act_limit, max = act_limit)
    z.copy_(r)

def gelu_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    r = F.gelu(x, approximate = "tanh") * y
    if act_limit != 0.0:
        r = torch.clamp(r, min = -act_limit, max = act_limit)
    z.copy_(r)

def relu2_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    r = torch.square(F.relu(x)) * y
    if act_limit != 0.0:
        r = torch.clamp(r, min = -act_limit, max = act_limit)
    z.copy_(r)

def relu_mul(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    r = F.relu(x) * y
    if act_limit != 0.0:
        r = torch.clamp(r, min = -act_limit, max = act_limit)
    z.copy_(r)

def xielu(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    act_limit: float = 0.0,
) -> None:
    r = (torch.tanh(x.clamp(min = -2.3562, max = 2.3562)) * x) * y
    if act_limit != 0.0:
        r = torch.clamp(r, min = -act_limit, max = act_limit)
    z.copy_(r)


# -- In-place gate ops (activation.cu) -----------------------------------------

def mul_sigmoid_(o: torch.Tensor, g: torch.Tensor) -> None:
    o.mul_(torch.sigmoid(g))

def mul_sigmoid_broadcast_(o: torch.Tensor, g: torch.Tensor) -> None:
    o.mul_(torch.sigmoid(g))

def mul_softplus_broadcast_(o: torch.Tensor, g: torch.Tensor) -> None:
    o.mul_(F.softplus(g.float(), threshold = 11).to(o.dtype))

def add_sigmoid_gate(g: torch.Tensor, o: torch.Tensor) -> None:
    o.add_(g).mul_(torch.sigmoid(o))

def add_sigmoid_gate_proj(x: torch.Tensor, g: torch.Tensor, o: torch.Tensor) -> None:
    o.copy_(x + g)
    o.mul_(torch.sigmoid(o))


# -- Attention helpers (activation.cu) ----------------------------------------

def deinterleave_qg(
    qg: torch.Tensor,
    q: torch.Tensor,
    g: torch.Tensor,
    head_dim: int,
) -> None:
    bsz, qlen = qg.shape[0], qg.shape[1]
    chunks = qg.view(bsz, qlen, -1, head_dim * 2)
    q.copy_(chunks[..., :head_dim].reshape(q.shape))
    g.copy_(chunks[..., head_dim:].reshape(g.shape))


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
) -> None:
    xf = x.float()
    if w is not None:
        wf = (w + constant_bias).float() if constant_bias != 0.0 else w.float()
    else:
        wf = None
    var = xf.pow(2).mean(dim = -1, keepdim = True) + eps
    xf = xf * torch.rsqrt(var) * constant_scale
    if wf is not None:
        xf = xf * wf
    y.copy_(xf.to(y.dtype))

def rms_norm_res_in(
    x: torch.Tensor,
    w: torch.Tensor | None,
    y: torch.Tensor,
    r: torch.Tensor,
    eps: float,
    constant_bias: float,
    constant_scale: float,
) -> None:
    r.add_(x)
    rf = r.float()
    if w is not None:
        wf = (w + constant_bias).float() if constant_bias != 0.0 else w.float()
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
) -> None:
    xf = x.float()
    gf = g.float()
    if gate_first:
        hidden = xf * F.silu(gf)
        if w_groups > 1:
            wf = w.view(w_groups, -1).float()
            hidden_2d = hidden.view(-1, wf.shape[1])
            var = hidden_2d.pow(2).mean(dim = -1, keepdim = True) + eps
            hidden_2d = hidden_2d * torch.rsqrt(var)
            hidden = (wf * hidden_2d).view(hidden.shape)
        else:
            var = hidden.pow(2).mean(-1, keepdim = True) + eps
            hidden = hidden * torch.rsqrt(var)
            hidden = w.float() * hidden
    else:
        var = xf.pow(2).mean(-1, keepdim = True) + eps
        xf = xf * torch.rsqrt(var)
        if w_groups > 1:
            hidden = w.view(w_groups, -1).float() * xf.view(-1, w.shape[-1] // w_groups if w.dim() > 1 else w.shape[0] // w_groups)
        else:
            hidden = w.float() * xf
        hidden = hidden * F.silu(gf)
    y.copy_(hidden.to(y.dtype))


# -- Softcap (softcap.cu) ------------------------------------------------------

def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    if cap == 0.0:
        return x
    return torch.tanh(x / cap) * cap


# -- Sentinel for missing BC_* classes -----------------------------------------

class _BCNone:
    """Callable that returns None, used as stand-in for missing BC_* constructors."""
    __slots__ = ()
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None
