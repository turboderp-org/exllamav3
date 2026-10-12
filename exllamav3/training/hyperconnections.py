"""
Differentiable gated-residual hyper-connections (Qwen3.8-Flash-Next /
``qwen4_exp``) for the native training forward.

On this architecture the residual between decoder blocks is not a ``[b, t, d]``
vector but a stack of ``hc_mult`` (4) parallel fp32 streams ``[b, t, H, d]``:
``ExpandStreams`` broadcasts the token embedding into the streams, every
sublayer site (attention / MLP of every block) reads a MIX of the streams and
writes its output back into each stream with its own gate, and a final
combine-less mixer collapses the stack before the LM head (there is no final
RMSNorm). exllamav3's inference module (``modules/hyperconnections.py``,
``GatedResidual``) runs this through fused CUDA kernels; this module is the
plain-torch, autograd-capable transcription of its ``_mix_ref`` -- the fp32
reference the inference parity tests compare the kernels against -- with no
exllamav3 imports, so the CPU test suite loads it standalone.

Site math (``GatedResidual._mix_ref`` / ``apply_``), all fp32::

    normed = rmsnorm_per_stream(x) * norm_w          # norm_w is (1 + w), [H, d]
    flat   = normed.flatten(H, d)                    # [b, t, H*d]
    t      = silu(flat @ down^T / H)                 # low-rank gate, [b, t, rank]
    w      = sigmoid(t @ up^T)                       # [b, t, H*d]
    mixed  = mean_over_streams(w * normed)           # the sublayer's input, [b, t, d]
    post   = 2 * sigmoid(flat @ inject^T / H)        # per-stream inject gate, [b, t, H]
    x'     = x + post[..., None] * y[..., None, :]   # y = sublayer output

The final mixer is the same ``mixed`` with no ``inject`` (``post`` None).
Inference feeds the sublayer ``mixed`` cast to half and keeps the stack fp32;
the training forward casts ``mixed`` to the compute dtype and keeps the stack
fp32 likewise (see ``native_llama._site_in`` / ``_site_out``).
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F


def expand_streams(hidden: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """``ExpandStreams``: broadcast ``[b, t, d]`` into the fp32 stream stack
    ``[b, t, hc_mult, d]`` (a real copy, like the inference module's
    ``.contiguous()``, so later residual adds don't alias the embedding)."""
    return hidden.float().unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()


def gated_residual_mix(streams: torch.Tensor, spec: dict):
    """One site's mix: ``[b, t, H, d]`` fp32 streams -> ``(post, mixed)`` where
    ``mixed`` is ``[b, t, d]`` fp32 (the sublayer input) and ``post`` is the
    ``[b, t, H]`` fp32 inject gate, or ``None`` for the final mixer (no
    ``inject`` in ``spec``). ``spec`` comes from ``backbone.gated_residual_spec``:
    ``hc_mult``, ``eps``, ``norm_w`` ``[H, d]`` (already ``1 + w``), ``down``
    ``[rank, H*d]``, ``up`` ``[H*d, rank]``, ``inject`` ``[H, H*d]`` or None.
    Transcribes ``GatedResidual._mix_ref`` op for op (the matmuls promote the
    fp16 tables to fp32 as the reference does)."""
    H = spec["hc_mult"]
    d = streams.shape[-1]
    # fp64 inputs (gradcheck) stay fp64; everything else runs fp32 like inference.
    x = streams if streams.dtype == torch.float64 else streams.float()
    norm_w = spec["norm_w"].to(x.dtype)
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + spec["eps"]) * norm_w
    flat = normed.flatten(-2)                                             # [b, t, H*d]
    t = F.silu(F.linear(flat, spec["down"].to(x.dtype)) / H)              # [b, t, rank]
    w = torch.sigmoid(F.linear(t, spec["up"].to(x.dtype)))                # [b, t, H*d]
    mixed = (w.unflatten(-1, (H, d)) * normed).mean(dim=-2)               # [b, t, d]
    inject = spec.get("inject")
    post = None
    if inject is not None:
        post = 2.0 * torch.sigmoid(F.linear(flat, inject.to(x.dtype)) / H)   # [b, t, H]
    return post, mixed


def gated_residual_apply(streams: torch.Tensor, y: torch.Tensor,
                         post: torch.Tensor) -> torch.Tensor:
    """The site's residual update ``x + post (x) y`` (``GatedResidual.apply_``,
    out of place): every stream ``h`` receives ``post[..., h] * y``. ``y`` is
    the sublayer output in the compute dtype; promoted to the stack's dtype
    exactly as the inference ``y.float()`` does."""
    return streams + post.unsqueeze(-1) * y.to(streams.dtype).unsqueeze(-2)


def streams_mean(streams: torch.Tensor) -> torch.Tensor:
    """The collapsed hidden state the inference block EXPORTS for a stream-stack
    residual (``TransformerBlock.forward``: ``x.mean(dim=2)``) -- what the
    EBFT feature taps read on this architecture."""
    return streams.mean(dim=2)
