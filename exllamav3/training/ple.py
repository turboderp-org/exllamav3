"""
Differentiable PLE (per-layer embedding) injection layer for the native
training forward (Qwen3.8-Flash-Next / ``qwen4_exp``).

The inference module (``modules/ple.py``, ``PLELayer``) sits between two early
decoder blocks and adds hashed n-gram features into every stream of the
hyper-connection stack. Its n-gram lookup is a frozen 51.2B-row table gathered
host-side (``NGramEmbedding``: CPU hashing, rows from RAM or disk); everything
AFTER the lookup is a few small projections, three grouped RMSNorms, a gate
and a dilated depthwise causal conv -- the differentiable part this module
transcribes from ``PLELayer.forward_streams_reference`` (the op-by-op form the
fused ``ple_forward_streams`` kernel mirrors) and the ``ple_gate`` kernel.
Plain torch, no exllamav3 imports, so the CPU suite loads it standalone; the
lookup itself happens in ``backbone.ple_ngram_embed`` (frozen, no grad).

Per token, with the stream stack ``streams`` ``[b, t, H, d]`` fp32 and the
looked-up n-gram embedding ``emb`` ``[b, t, ple_dim]``::

    key    = norm_key(key_proj(emb).view(b, t, H, d))          # per-stream grouped RMSNorm
    value  = value_proj(emb)                                   # [b, t, d], shared by the streams
    query  = norm_query(streams)                               # per-stream grouped RMSNorm
    gate   = <query_h, key_h> * gate_scale                     # [b, t, H]  (gate_scale = d^-0.5)
    gated  = sigmoid(signed_sqrt(gate)) * value                # [b, t, H, d]
    normed = norm_conv(gated).flatten(H, d)                    # [b, t, H*d]
    conv   = silu(depthwise_causal_conv1d(normed, k, dilation))
    delta  = gated + conv.view(b, t, H, d)
    streams <- streams + delta

``signed_sqrt(g) = sign(g) * sqrt(max(|g|, 1e-6))`` is the ``ple_gate``
kernel's form (``ple.cu``), zero at zero. The conv is causal and stateless
here: training always sees whole sequences from position 0, so the inference
layer's zero conv state (``conv_state_len = (kernel - 1) * dilation`` trailing
columns) is exactly a left pad of that many zeros, and the n-gram history for a
fresh sequence is ``ngram_size - 1`` eos tokens in front of the ids
(``PLELayer._history``), reproduced by ``backbone.ple_ngram_embed``.

Right-padding is safe: a pad position's gate/conv only ever feed later pad
positions, which are masked from the loss. Sample packing is NOT: the conv
would carry the previous document's values across the boundary and the n-gram
hash would span it (same rule as GatedDeltaNet; ``native_llama`` rejects it).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def ple_signed_sqrt_gate(gate: torch.Tensor, gate_scale: float) -> torch.Tensor:
    """``sigmoid(signed_sqrt(gate * gate_scale))`` in fp32 -- the ``ple_gate``
    kernel's scalar: ``a = sqrt(max(|g|, 1e-6))``, ``ss = sign(g) * a``,
    ``sigmoid(ss)``. ``sign`` has no gradient, ``sqrt`` sees the clamped
    magnitude, so this is finite everywhere (including at 0)."""
    g = gate if gate.dtype == torch.float64 else gate.float()
    g = g * gate_scale
    a = torch.sqrt(torch.clamp(g.abs(), min=1e-6))
    return torch.sigmoid(torch.sign(g) * a)


def ple_dilated_causal_conv_silu(x: torch.Tensor, conv_w: torch.Tensor,
                                 dilation: int) -> torch.Tensor:
    """``silu(conv1d(x))``, depthwise (``groups = channels``) and causal with
    ``dilation``: ``[b, t, ch] -> [b, t, ch]`` in ``x``'s dtype. ``conv_w`` is
    the module's ``[ch, 1, kernel]`` tensor. Left-pads ``(kernel - 1) *
    dilation`` zeros -- the fresh-sequence conv state of ``PLELayer._short_conv``
    (``conv_state is None`` -> ``new_zeros((b, ch, conv_state_len))``)."""
    kernel = conv_w.shape[-1]
    pad = (kernel - 1) * dilation
    xt = x.transpose(1, 2)                                            # [b, ch, t]
    w = conv_w.to(xt.dtype)
    if w.dim() == 2:
        w = w.unsqueeze(1)
    y = F.conv1d(F.pad(xt, (pad, 0)), w, groups=xt.shape[1], dilation=dilation)
    return F.silu(y).transpose(1, 2)


def ple_delta(streams: torch.Tensor, emb: torch.Tensor, spec: dict,
              key_proj, value_proj, norm, compute_dtype: torch.dtype) -> torch.Tensor:
    """The layer's additive delta ``[b, t, H, d]`` (stack dtype) for the stream
    stack ``streams`` and the frozen n-gram embedding ``emb`` ``[b, t,
    ple_dim]``. ``key_proj`` / ``value_proj`` are the (frozen) ``DiffLinear``
    wrappers of the module's projections; ``norm(x, spec)`` is the trainer's
    RMSNorm (``native_llama._norm``) -- the three ``spec["norm_*"]`` entries
    are grouped specs whose ``[H, d]`` weight broadcasts per stream; ``spec``
    also carries ``hc_mult``, ``hidden_size``, ``gate_scale``, ``conv_w`` and
    ``dilation`` (``backbone.ple_spec``).

    Dtypes follow the reference: the projections emit the compute dtype (half
    at inference), the key / query norms return fp32, the gate and the gated
    values are fp32, ``norm_conv`` returns the compute dtype (half at
    inference) and the conv runs in it, the delta is fp32."""
    b, t = streams.shape[:2]
    H, d = spec["hc_mult"], spec["hidden_size"]
    sdt = streams.dtype if streams.dtype == torch.float64 else torch.float32
    e = emb.to(compute_dtype)
    key = key_proj(e).view(b, t, H, d)
    key = norm(key.to(sdt), spec["norm_key"])                         # [b, t, H, d] fp32
    value = value_proj(e)                                              # [b, t, d]
    query = norm(streams.to(sdt), spec["norm_query"])                  # [b, t, H, d] fp32
    gate = (query * key).sum(dim=-1)                                   # [b, t, H]
    s = ple_signed_sqrt_gate(gate, spec["gate_scale"])                 # [b, t, H] fp32
    gated = s.unsqueeze(-1) * value.to(sdt).unsqueeze(-2)              # [b, t, H, d]
    # norm_conv: fp32 internals on the fp32 gated values, rounded once to the
    # compute dtype at the output (the inference out_dtype = half store).
    normed = norm(gated, spec["norm_conv"]).to(compute_dtype).flatten(-2)   # [b, t, H*d]
    conv = ple_dilated_causal_conv_silu(normed, spec["conv_w"], spec["dilation"])
    return gated + conv.to(sdt).view(b, t, H, d)
