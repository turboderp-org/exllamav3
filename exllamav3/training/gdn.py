"""
Differentiable Gated DeltaNet math for the native training forward.

Qwen3.5/3.6 build ~3/4 of their layers as ``GatedDeltaNet`` (linear/recurrent
attention: a causal depthwise conv + SiLU over the packed q/k/v projections,
then the gated delta rule with L2-normalized q/k and per-value-head exponential
decay, then a gated RMSNorm). exllamav3's inference implementation
(``modules/gated_delta_net.py``) runs under ``@torch.inference_mode`` through
CUDA/Triton kernels, so training needs its own autograd-capable forward. This
module holds the pure math -- plain torch, no exllamav3 imports, so the CPU
test suite can load it standalone (like ``fused_ce`` / ``qlora_linear``).

Two delta-rule paths mirror the flash/eager attention split in
``native_llama``:

* the **fla fast path** (``fla.ops.gated_delta_rule.chunk_gated_delta_rule``,
  dispatched from ``NativeLlamaQLoRA._gdn_delta_rule``) -- the same
  autograd-capable Triton kernel exllamav3's own inference prefill uses, so
  train-time numerics match serve-time numerics by construction. CUDA +
  fp16/bf16 only.
* :func:`gdn_delta_rule_reference` here -- a sequential, fully differentiable
  fp32 scan, transcribed from the validated inference reference
  (``torch_recurrent_gated_delta_rule``) but autograd-safe (no in-place writes
  into graph tensors). It is the CPU / fp32 / gradcheck / validate-gate path;
  it is O(t) sequential and keeps the per-step state in the graph, so it is
  for correctness work and short sequences, not long training runs.

Semantics (matching the inference module + its CUDA fused op, gdn.cu):

* ``beta = sigmoid(b_proj(x)) * beta_scale``
* ``g = -exp(a_log) * softplus(a_proj(x) + dt_bias)``  (per value head, fp32)
* q/k are L2-normalized per head inside the rule (``use_qk_l2norm_in_kernel``)
* GQA-style grouping: ``num_v_heads = G * num_k_heads``; value head ``j`` uses
  key head ``j // G`` (the natural repeat_interleave layout of the split
  ``in_proj_qkv``)
* recurrence per value head ``h`` with state ``S`` of shape ``[dk, dv]``::

      S_t = exp(g_t) * S_{t-1} + k_t ⊗ (beta_t * (v_t - (exp(g_t)*S_{t-1})^T k_t))
      o_t = (S_t^T q_t) * dk^-0.5

* output norm: ``rmsnorm(o) * (w + bias) * silu(z)`` (``GatedRMSNorm``), or
  ``* sigmoid(z)`` on a ``gate_activation = "sigmoid"`` norm
  (Qwen3.8-Flash-Next's ``output_gate_type``)
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F


def gdn_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-head L2 normalization, matching the inference reference / fla's
    in-kernel ``l2norm``: ``x / sqrt(sum(x^2) + eps)`` over the last dim."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def gdn_delta_rule_reference(
    q: torch.Tensor,        # [b, t, nk, dk]  raw (pre-L2-norm)
    k: torch.Tensor,        # [b, t, nk, dk]  raw (pre-L2-norm)
    v: torch.Tensor,        # [b, t, nv, dv]
    g: torch.Tensor,        # [b, t, nv]      log decay (<= 0), fp32
    beta: torch.Tensor,     # [b, t, nv]      update strength in (0, beta_scale)
    eps: float = 1e-6,
) -> torch.Tensor:
    """Sequential differentiable gated delta rule; returns ``[b, t, nv, dv]``
    in fp32. Faithful to ``torch_recurrent_gated_delta_rule`` (the reference
    the inference CUDA kernel is validated against) with the GQA q/k expansion
    applied here, and made autograd-safe: outputs are stacked, never written
    in place, and the state update is out of place so the graph stays valid.
    """
    b, t, nk, dk = q.shape
    nv, dv = v.shape[2], v.shape[3]
    assert k.shape == (b, t, nk, dk) and v.shape[:2] == (b, t)
    assert nv % nk == 0, "num_v_heads must be a multiple of num_k_heads"

    # Promote to >= fp32 (half dtypes upcast; fp64 kept for gradcheck).
    def up(x):
        return x.to(torch.promote_types(x.dtype, torch.float32))
    q = gdn_l2norm(up(q), eps)
    k = gdn_l2norm(up(k), eps)
    v = up(v)
    g = up(g)
    beta = up(beta)

    grp = nv // nk
    if grp > 1:
        # Value head j uses key head j // grp (the split in_proj_qkv layout).
        q = q.repeat_interleave(grp, dim=2)
        k = k.repeat_interleave(grp, dim=2)

    scale = dk ** -0.5
    state = q.new_zeros(b, nv, dk, dv)
    outs = []
    for i in range(t):
        g_t = g[:, i].exp().unsqueeze(-1)              # [b, nv, 1]
        beta_t = beta[:, i].unsqueeze(-1)              # [b, nv, 1]
        q_t, k_t, v_t = q[:, i], k[:, i], v[:, i]
        # Read of the decayed state: ((g*S)^T k) per head -> [b, nv, dv].
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem * g_t) * beta_t
        # S = g*S + k ⊗ delta  ==  (I - beta k k^T)(g S) + beta k v^T.
        state = state * g_t.unsqueeze(-1) + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((state * q_t.unsqueeze(-1)).sum(dim=-2) * scale)
    return torch.stack(outs, dim=1)                    # [b, t, nv, dv]


def gdn_causal_conv1d_silu(
    x: torch.Tensor,                    # [b, dim, t]  packed q|k|v channels
    weight: torch.Tensor,               # [dim, kernel]  depthwise conv weight
    bias: Optional[torch.Tensor] = None,  # [dim]
) -> torch.Tensor:
    """Stateless causal depthwise conv + SiLU, ``[b, dim, t] -> [b, dim, t]``.

    Training always sees whole sequences from position 0, so the inference
    path's zero-initialized conv state is exactly a left pad of ``kernel - 1``
    zeros (``causal_conv1d_update_function_torch`` concatenates ``kernel``
    state columns and drops the first output -- same thing). Runs in ``x``'s
    dtype (fp32 on the validate path, compute dtype in training).
    """
    kernel = weight.shape[-1]
    y = F.conv1d(F.pad(x, (kernel - 1, 0)), weight.unsqueeze(1).to(x.dtype),
                 bias.to(x.dtype) if bias is not None else None,
                 padding=0, groups=x.shape[1])
    return F.silu(y)


def gdn_gated_rmsnorm(x: torch.Tensor, spec: dict, gate: torch.Tensor) -> torch.Tensor:
    """Gated RMSNorm over the value-head dim: ``rmsnorm(x) * (w + bias) *
    act(gate)``, fp32 internals, returned in ``x``'s dtype. ``spec`` is a
    ``backbone.gdn_norm_spec`` dict (weight/eps/bias + ``gate_activation``);
    ``x``/``gate`` are ``[b, t, nv, dv]``. ``gate_activation`` is ``"silu"``
    (Qwen3.5/3.6 -- ``GatedRMSNorm.forward_torch``) or ``"sigmoid"``
    (Qwen3.8-Flash-Next's ``output_gate_type``, KDA -- the module's sigmoid
    path: weighted norm, then ``* sigmoid(gate)``, all fp32). Anything else
    is rejected by ``gdn_norm_spec`` at construction, never guessed here."""
    act = spec.get("gate_activation", "silu")
    assert act in ("silu", "sigmoid"), f"unknown GDN gate activation {act!r}"
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True) + spec["eps"]
    xn = xf * torch.rsqrt(var)
    w = spec["weight"]
    if w is not None:
        w = w.float()
        b = spec["bias"]
        xn = xn * (w + b) if b != 0.0 else xn * w
    g = gate.float()
    xn = xn * (torch.sigmoid(g) if act == "sigmoid" else F.silu(g))
    return xn.to(x.dtype)


def gdn_beta_g(
    b_raw: torch.Tensor,      # [b, t, nv]  b_proj output
    a_raw: torch.Tensor,      # [b, t, nv]  a_proj output
    a_log: torch.Tensor,      # [nv]
    dt_bias: torch.Tensor,    # [nv]
    beta_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The gate/strength nonlinearities of the inference fused op (gdn.cu):
    ``beta = sigmoid(b) * beta_scale``; ``g = -exp(a_log) * softplus(a +
    dt_bias)``. Computed in fp32 (g is kept fp32 through the rule, as the
    inference path does)."""
    beta = torch.sigmoid(b_raw.float()) * float(beta_scale)
    g = -a_log.float().exp() * F.softplus(a_raw.float() + dt_bias.float())
    return beta, g
