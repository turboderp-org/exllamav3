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


# ---------------------------------------------------------------------------
# mHC (manifold-constrained hyper-connections): DeepSeek-V4 / GLM-5.3-Flash.
# ---------------------------------------------------------------------------
# A different hyper-connection kind from the GatedResidual above (same stream
# stack ``[b, t, H, d]`` fp32, different site math). exllamav3's inference
# module is ``modules/hyperconnections.py`` ``HyperConnection`` (fused
# ``ext.hc_mix`` / ``ext.hc_apply`` kernels with a torch fallback,
# ``_mix_unfused`` / ``apply_``) and ``HyperHead``; the functions below are
# the plain-torch, autograd-capable transcription of those fallbacks, which
# follow the transformers reference (``DeepseekV4HyperConnection``) op for op.
#
# Site math (``HyperConnection._mix_unfused`` / ``apply_``), all fp32::
#
#     flat   = rmsnorm_unweighted(x.flatten(H, d))           # [b, t, H*d]
#     mix    = flat @ fn^T                                    # [b, t, 2H + H*H]
#     pre_w, post_w, comb_w = split(mix, [H, H, H*H])
#     pre    = sigmoid(pre_w * scale[0] + base[:H]) + hc_eps  # [b, t, H]
#     post   = 2 * sigmoid(post_w * scale[1] + base[H:2H])    # [b, t, H]
#     comb   = softmax(comb_w.view(H, H) * scale[2] + base[2H:].view(H, H), -1) + hc_eps
#     comb   = comb / (comb.sum(-2) + hc_eps)                 # columns
#     repeat sinkhorn_iters - 1 times:
#         comb = comb / (comb.sum(-1) + hc_eps)               # rows
#         comb = comb / (comb.sum(-2) + hc_eps)               # columns
#     collapsed = sum_k pre[..., k] * x[..., k, :]            # the sublayer input, [b, t, d]
#     x'     = post[..., None] * y[..., None, :] + comb^T @ x  # y = sublayer output
#
# The block's own pre-norm (input_layernorm / post_attention_layernorm) then
# runs on ``collapsed`` (inference: ``mix_norm``; folded into the kernel when
# it can be, else the module's ``forward`` on the half collapsed rows). The
# final ``HyperHead`` collapses the stack with its own ``pre`` gate (DeepSeek-
# V4: ``hc_head_fn/base/scale``) or, on GLM-5.3-Flash, an unweighted mean,
# and the model's final RMSNorm follows.


def _unweighted_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def mhc_mix(streams: torch.Tensor, spec: dict):
    """One mHC site's mix: fp32 ``[b, t, H, d]`` streams -> ``(post, comb,
    collapsed)`` with ``post`` ``[b, t, H]``, ``comb`` ``[b, t, H, H]`` (the
    Sinkhorn-normalized combine matrix) and ``collapsed`` ``[b, t, d]`` (the
    sublayer's input BEFORE the block's pre-norm). ``spec`` comes from
    ``backbone.mhc_spec``: ``hc_mult``, ``eps`` (the unweighted norm's), ``hc_eps``,
    ``sinkhorn_iters``, ``fn`` ``[2H + H*H, H*d]``, ``base`` ``[2H + H*H]``,
    ``scale`` ``[3]`` (all fp32). Transcribes ``HyperConnection._mix_unfused``."""
    H = spec["hc_mult"]
    x = streams if streams.dtype == torch.float64 else streams.float()
    dt = x.dtype
    fn, base, scale = spec["fn"].to(dt), spec["base"].to(dt), spec["scale"].to(dt)
    eps = spec["hc_eps"]
    flat = _unweighted_rmsnorm(x.flatten(2), spec["eps"])            # [b, t, H*d]
    mix = F.linear(flat, fn)                                          # [b, t, 2H + H*H]
    pre_w, post_w, comb_w = mix.split([H, H, H * H], dim=-1)
    pre_b, post_b, comb_b = base.split([H, H, H * H])
    pre_s, post_s, comb_s = scale.unbind(0)
    pre = torch.sigmoid(pre_w * pre_s + pre_b) + eps                  # [b, t, H]
    post = 2.0 * torch.sigmoid(post_w * post_s + post_b)              # [b, t, H]
    comb = comb_w.view(*comb_w.shape[:-1], H, H) * comb_s + comb_b.view(H, H)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(int(spec["sinkhorn_iters"]) - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    collapsed = (pre.unsqueeze(-1) * x).sum(dim=2)                    # [b, t, d]
    return post, comb, collapsed


def mhc_apply(streams: torch.Tensor, y: torch.Tensor, post: torch.Tensor,
              comb: torch.Tensor) -> torch.Tensor:
    """The site's residual update ``x' = post (x) y + comb^T x``
    (``HyperConnection.apply_``'s torch fallback, out of place): stream ``k``
    receives ``post[..., k] * y + sum_j comb[..., j, k] * x[..., j, :]``. ``y``
    is the sublayer output in the compute dtype, promoted to the stack's."""
    yk = y.to(streams.dtype)
    return post.unsqueeze(-1) * yk.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), streams)


def mhc_head(streams: torch.Tensor, spec: dict) -> torch.Tensor:
    """The final ``HyperHead`` collapse ``[b, t, H, d]`` -> ``[b, t, d]`` fp32:
    an unweighted mean over the streams (``spec["mean"]``: GLM-5.3-Flash) or
    the DeepSeek-V4 gate ``sum_k (sigmoid(flat @ fn^T * scale + base) + hc_eps)_k
    * x_k`` with ``fn`` ``[H, H*d]``, ``base`` ``[H]``, ``scale`` ``[1]``
    (``HyperHead.forward``'s torch fallback)."""
    x = streams if streams.dtype == torch.float64 else streams.float()
    if spec.get("mean", False):
        return x.mean(dim=2)
    dt = x.dtype
    flat = _unweighted_rmsnorm(x.flatten(2), spec["eps"])
    mixes = F.linear(flat, spec["fn"].to(dt))                         # [b, t, H]
    pre = torch.sigmoid(mixes * spec["scale"].to(dt) + spec["base"].to(dt)) + spec["hc_eps"]
    return (pre.unsqueeze(-1) * x).sum(dim=2)
