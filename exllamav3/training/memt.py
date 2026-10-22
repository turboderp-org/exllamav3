"""
MemT-1.0 native memory block: pure-torch, differentiable reference math.

MemT-1.0 (MemTensor, Oct 2026, arXiv 2610.26825) bolts a "native memory block"
onto each full-attention layer of a frozen backbone. MemT-1.0-Nano's backbone
is Qwen3.6-35B-A3B *as released* (layers 3, 7, ..., 39 -- the ten
``full_attention`` layers -- each get one memory block). The block keeps a
persistent per-session state ``M`` of shape ``[H, dk, dv]`` (16 x 256 x 256
fp32 per layer, 40 MiB for the whole model) that

* is READ during generation: every token of a turn reads the SAME snapshot
  ``M_t`` (the state as of the start of the turn), and the readout is added to
  the backbone attention context right before ``o_proj`` (after the
  backbone's own sigmoid attention gate);
* is WRITTEN at turn end ("commit"): one forward over the whole turn
  ``X_t || Y_t`` runs the stable token-level GDN-2 recurrence from ``M_t`` and
  publishes ``M_{t+1}``. Memory starts at zero for a new trajectory.

Tensor layout of one block, HF keys under
``model.language_model.layers.{i}.self_attn.memt.`` (from the released
SGLang implementation, ``memt_sglang/models/qwen/layer.py``):

    in_proj.weight    [5*H*D + 2*D, hidden]  rows: q | k | v | erase | write
                                             | decay_latent (D) | gate_latent (D)
    f_down.weight     [H*D, D]               forget-gate up-projection
    g_down.weight     [H*D, D]               output-gate up-projection
    decay_scale_log   [H]                    per-head log decay scale (s_h)
    decay_bias        [H*D]                  per-channel forget bias (b_lambda)
    o_norm.weight     [D]                    per-head RMSNorm gain (shared)

For Nano (hidden 2048, H 16, D 256) that is 45,093,136 parameters per block,
450,931,360 over the ten blocks -- exactly the paper's Table 6.

Math (paper sec. 2.2.2 + app. C, matching the released kernels):

    x = pre-attention (input-normed) hidden state, [b, t, hidden]
    q, k, v           = x W_q, x W_k, x W_v                       -> [b, t, H, D]
    q_n, k_n          = l2norm(q), l2norm(k)   (eps 1e-6, per head)
    erase_sqrt        = sqrt(sigmoid(x W_beta))                   -> [b, t, H, D]
    write             = sigmoid(x W_omega)                        -> [b, t, H, D]
    log_decay         = -exp(s_h) * softplus((x W_f) W_lambda + b_lambda)
                                                                   -> [b, t, H, D]  (<= 0)
    read R            = (q_n M_t) / sqrt(D)                       -> [b, t, H, D]
    gate G            = (x W_g/d) W_g/u                           -> [b, t, H, D]
    out               = concat_h[ RMSNorm_D(R_h) * w_o * softcap_silu(G_h) ]
                        softcap_silu(x) = tau * tanh(silu(x) / tau), tau = 2
    commit, per token i and head h, u = erase_sqrt * k_n (a row vector over dk):
        S <- diag(exp(log_decay_i)) S          (per-k-channel decay)
        S <- S - u^T (u S)                     (symmetrized erase, (I - u^T u) S)
        S <- S + u^T (write_i * v_i)           (gated write)

Plain torch, no exllamav3 imports (CPU test suite loads it standalone, like
``gdn.py``). The sequential scan is the correctness / gradcheck reference; a
chunked CUDA path (fla's ``gdn2`` kernels, which the MemT release vendors)
is the follow-up for long commits.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen = True)
class MemTSpec:
    hidden_size: int
    num_heads: int
    head_dim: int
    rms_norm_eps: float
    output_gate_softcap: float = 2.0
    l2norm_eps: float = 1e-6

    @property
    def value_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def in_proj_rows(self) -> int:
        return 5 * self.num_heads * self.head_dim + 2 * self.head_dim

    @property
    def state_shape(self) -> tuple[int, int, int]:
        """Per-trajectory state of one block: ``[H, dk, dv]`` (fp32)."""
        return (self.num_heads, self.head_dim, self.head_dim)


def memt_in_proj_slices(spec: MemTSpec) -> dict[str, slice]:
    """Row ranges of the fused ``in_proj`` (release layout, see module doc)."""
    hd, d = spec.value_dim, spec.head_dim
    names = [("q", hd), ("k", hd), ("v", hd), ("erase", hd), ("write", hd),
             ("decay_latent", d), ("gate_latent", d)]
    out, pos = {}, 0
    for name, n in names:
        out[name] = slice(pos, pos + n)
        pos += n
    assert pos == spec.in_proj_rows
    return out


def memt_param_shapes(spec: MemTSpec) -> dict[str, tuple[int, ...]]:
    """Checkpoint tensor shapes of one block, keyed by the HF suffix."""
    hd, d, h = spec.value_dim, spec.head_dim, spec.num_heads
    return {
        "in_proj.weight": (spec.in_proj_rows, spec.hidden_size),
        "f_down.weight": (hd, d),
        "g_down.weight": (hd, d),
        "decay_scale_log": (h,),
        "decay_bias": (hd,),
        "o_norm.weight": (d,),
    }


def memt_param_count(spec: MemTSpec) -> int:
    n = 0
    for shape in memt_param_shapes(spec).values():
        p = 1
        for s in shape:
            p *= s
        n += p
    return n


@dataclass
class MemTWeights:
    """The six tensors of one block (any float dtype; math runs in fp32)."""
    in_proj: torch.Tensor          # [rows, hidden]
    f_down: torch.Tensor           # [H*D, D]
    g_down: torch.Tensor           # [H*D, D]
    decay_scale_log: torch.Tensor  # [H]
    decay_bias: torch.Tensor       # [H*D]
    o_norm: torch.Tensor           # [D]

    @staticmethod
    def from_dict(t: dict[str, torch.Tensor]) -> "MemTWeights":
        return MemTWeights(
            in_proj = t["in_proj.weight"],
            f_down = t["f_down.weight"],
            g_down = t["g_down.weight"],
            decay_scale_log = t["decay_scale_log"],
            decay_bias = t["decay_bias"],
            o_norm = t["o_norm.weight"],
        )

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "in_proj.weight": self.in_proj,
            "f_down.weight": self.f_down,
            "g_down.weight": self.g_down,
            "decay_scale_log": self.decay_scale_log,
            "decay_bias": self.decay_bias,
            "o_norm.weight": self.o_norm,
        }

    def check(self, spec: MemTSpec):
        for name, shape in memt_param_shapes(spec).items():
            got = tuple(self.to_dict()[name].shape)
            assert got == shape, f"memt {name}: shape {got}, expected {shape}"


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

def _f32(x: torch.Tensor) -> torch.Tensor:
    """Promote to at least fp32 (fp64 is kept, for gradcheck)."""
    return x.to(torch.promote_types(x.dtype, torch.float32))


def memt_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """fla ``l2norm``: ``x / sqrt(sum(x^2) + eps)`` over the last dim."""
    return x * torch.rsqrt((x * x).sum(dim = -1, keepdim = True) + eps)


def memt_project(x: torch.Tensor, w: MemTWeights, spec: MemTSpec) -> dict[str, torch.Tensor]:
    """``x [b, t, hidden]`` through the fused ``in_proj``; heads split out.
    Runs in ``x``'s dtype (fp32 on the reference path)."""
    b, t, _ = x.shape
    H, D = spec.num_heads, spec.head_dim
    y = F.linear(x, w.in_proj.to(x.dtype))
    sl = memt_in_proj_slices(spec)
    out = {}
    for name in ("q", "k", "v", "erase", "write"):
        out[name] = y[..., sl[name]].reshape(b, t, H, D)
    out["decay_latent"] = y[..., sl["decay_latent"]]      # [b, t, D]
    out["gate_latent"] = y[..., sl["gate_latent"]]        # [b, t, D]
    return out


def memt_gates(
    erase_raw: torch.Tensor,       # [b, t, H, D]
    write_raw: torch.Tensor,       # [b, t, H, D]
    decay_latent: torch.Tensor,    # [b, t, D]
    w: MemTWeights,
    spec: MemTSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(erase_sqrt, write, log_decay)``, all ``[b, t, H, D]`` fp32.

    erase_sqrt = exp(0.5 * logsigmoid(erase_raw)) = sqrt(sigmoid(erase_raw))
    write      = sigmoid(write_raw)
    log_decay  = -exp(decay_scale_log)[h] * softplus(f_down(decay_latent) + decay_bias)
    """
    b, t = erase_raw.shape[:2]
    H, D = spec.num_heads, spec.head_dim
    erase_sqrt = torch.exp(0.5 * F.logsigmoid(_f32(erase_raw)))
    write = torch.sigmoid(_f32(write_raw))
    dl = _f32(decay_latent)
    f = F.linear(dl, w.f_down.to(dl.dtype)) + w.decay_bias.to(dl.dtype)   # [b, t, H*D]
    f = F.softplus(f).view(b, t, H, D)
    log_decay = -w.decay_scale_log.to(dl.dtype).exp().view(1, 1, H, 1) * f
    return erase_sqrt, write, log_decay


def memt_read(
    q_raw: torch.Tensor,     # [b, t, H, D]  raw query projection
    state: torch.Tensor,     # [b, H, D, D] or [H, D, D]  (fp32)
    spec: MemTSpec,
) -> torch.Tensor:
    """Memory readout ``R = l2norm(q) M / sqrt(D)``, ``[b, t, H, D]`` fp32."""
    q = memt_l2norm(_f32(q_raw), spec.l2norm_eps)
    s = state.to(q.dtype)
    if s.dim() == 3:
        s = s.unsqueeze(0).expand(q.shape[0], -1, -1, -1)
    assert s.shape[0] == q.shape[0] and s.shape[1:] == spec.state_shape, \
        f"memt state shape {tuple(state.shape)} does not fit [b={q.shape[0]}, {spec.state_shape}]"
    return torch.einsum("bthk,bhkv->bthv", q, s) * (spec.head_dim ** -0.5)


def memt_update_reference(
    k_raw: torch.Tensor,        # [b, t, H, D]  raw key projection
    v: torch.Tensor,            # [b, t, H, D]
    erase_sqrt: torch.Tensor,   # [b, t, H, D]
    write: torch.Tensor,        # [b, t, H, D]
    log_decay: torch.Tensor,    # [b, t, H, D]  (<= 0)
    state0: Optional[torch.Tensor],   # [b, H, D, D] / [H, D, D] / None (zeros)
    spec: MemTSpec,
) -> torch.Tensor:
    """Sequential stable GDN-2 scan; returns the final state ``[b, H, D, D]``
    in fp32 (fp64 kept for gradcheck). Autograd-safe (no in-place writes).

    Per token, per head (``u`` a row over dk):
        S = diag(exp(g)) S ;  S = S - u^T (u S) ;  S = S + u^T (write * v)
    """
    b, t, H, D = k_raw.shape

    up = _f32
    k = memt_l2norm(up(k_raw), spec.l2norm_eps)
    u_all = up(erase_sqrt) * k                      # [b, t, H, D]
    v = up(v)
    write = up(write)
    g = up(log_decay)
    if state0 is None:
        S = k.new_zeros(b, H, D, D)
    else:
        S = up(state0)
        if S.dim() == 3:
            S = S.unsqueeze(0).expand(b, -1, -1, -1)
        assert S.shape == (b, H, D, D), f"memt state0 shape {tuple(state0.shape)}"
    for i in range(t):
        S = S * g[:, i].exp().unsqueeze(-1)                       # decay rows (k channels)
        u = u_all[:, i]                                           # [b, H, D]
        uS = torch.einsum("bhk,bhkv->bhv", u, S)                  # u S
        S = S - u.unsqueeze(-1) * uS.unsqueeze(-2)                # (I - u^T u) S
        S = S + u.unsqueeze(-1) * (write[:, i] * v[:, i]).unsqueeze(-2)
    return S


# Long commits (documents, long replies) go through fla's chunked KDA kernel: the GDN-2 update
#   S = (I - u u^T) diag(exp g) S + u (w*v)^T
# is Kimi Delta Attention with beta = 1, key u = erase_sqrt * l2norm(k), value w * v and the per-k-channel
# log decay g. Checked against the sequential reference on random inputs (relative error ~1e-3 in fp32,
# the kernel's internal precision; the release commits through the same family of fla kernels). Short
# turns keep the exact fp32 reference.
MEMT_CHUNKED_MIN_TOKENS = int(os.environ.get("EXL3_MEMT_CHUNKED_MIN_TOKENS", "128"))
_memt_chunked_enable = os.environ.get("EXL3_MEMT_CHUNKED", "1") != "0"


def memt_update(
    k_raw: torch.Tensor,
    v: torch.Tensor,
    erase_sqrt: torch.Tensor,
    write: torch.Tensor,
    log_decay: torch.Tensor,
    state0: Optional[torch.Tensor],
    spec: MemTSpec,
) -> torch.Tensor:
    """``memt_update_reference`` for short or CPU inputs, fla ``chunk_kda`` for long CUDA ones."""
    b, t, H, D = k_raw.shape
    if not (_memt_chunked_enable and k_raw.is_cuda and t >= MEMT_CHUNKED_MIN_TOKENS):
        return memt_update_reference(k_raw, v, erase_sqrt, write, log_decay, state0, spec)
    try:
        from fla.ops.kda import chunk_kda
    except ImportError:
        return memt_update_reference(k_raw, v, erase_sqrt, write, log_decay, state0, spec)
    up = _f32
    u = memt_l2norm(up(k_raw), spec.l2norm_eps) * up(erase_sqrt)
    wv = up(write) * up(v)
    if state0 is None:
        S0 = k_raw.new_zeros(b, H, D, D, dtype = torch.float32)
    else:
        S0 = up(state0)
        if S0.dim() == 3:
            S0 = S0.unsqueeze(0).expand(b, -1, -1, -1)
        S0 = S0.contiguous()
    _, S = chunk_kda(
        torch.zeros_like(u), u, wv, up(log_decay), torch.ones(b, t, H, device = u.device, dtype = u.dtype),
        initial_state = S0, output_final_state = True,
    )
    return S.to(torch.float32)


def memt_output_gate(gate_latent: torch.Tensor, w: MemTWeights, spec: MemTSpec) -> torch.Tensor:
    """``G = g_down(gate_latent)`` viewed per head, ``[b, t, H, D]``."""
    b, t, _ = gate_latent.shape
    g = F.linear(gate_latent, w.g_down.to(gate_latent.dtype))
    return g.view(b, t, spec.num_heads, spec.head_dim)


def memt_gated_softcap_norm(
    x: torch.Tensor,         # [b, t, H, D]  readout
    gate: torch.Tensor,      # [b, t, H, D]
    w: MemTWeights,
    spec: MemTSpec,
) -> torch.Tensor:
    """Per-head ``RMSNorm(x) * o_norm * softcap_silu(gate)``, >= fp32 throughout.
    Transcribes the release's ``MemTRMSNormGated`` (norm over head_dim, one
    shared ``[D]`` gain, ``tau * tanh(silu(gate) / tau)``)."""
    xf = _f32(x)
    var = xf.pow(2).mean(dim = -1, keepdim = True)
    xn = xf * torch.rsqrt(var + spec.rms_norm_eps) * w.o_norm.to(xf.dtype)
    gv = F.silu(gate.to(xf.dtype))
    tau = float(spec.output_gate_softcap)
    gv = tau * torch.tanh(gv / tau)
    return xn * gv


# ---------------------------------------------------------------------------
# Whole block
# ---------------------------------------------------------------------------

def memt_block_forward(
    x: torch.Tensor,                       # [b, t, hidden]  pre-attention normed hidden
    w: MemTWeights,
    spec: MemTSpec,
    read_state: Optional[torch.Tensor],    # [b, H, D, D] / [H, D, D] / None (= zero state)
    commit: bool = False,
    write_state: Optional[torch.Tensor] = None,   # defaults to read_state
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """One memory block over a turn.

    Returns ``(out, next_state)``: ``out`` is ``[b, t, H*D]`` (fp32) to be ADDED
    to the backbone attention context right before ``o_proj``; ``next_state``
    is the committed ``[b, H, D, D]`` state when ``commit`` else ``None``.

    With ``read_state`` None (or all zeros) the readout is exactly zero -- the
    norm of a zero vector is zero -- so the block is a no-op on the backbone.
    Every token reads the same ``read_state`` (the release reads a fixed
    per-request snapshot; token-level updates only feed the commit).
    """
    b, t, _ = x.shape
    H, D = spec.num_heads, spec.head_dim
    p = memt_project(x, w, spec)
    gate = memt_output_gate(p["gate_latent"], w, spec)
    if read_state is None:
        read = x.new_zeros(b, t, H, D, dtype = torch.promote_types(x.dtype, torch.float32))
    else:
        read = memt_read(p["q"], read_state, spec)
    out = memt_gated_softcap_norm(read, gate, w, spec).reshape(b, t, H * D)
    next_state = None
    if commit:
        erase_sqrt, write, log_decay = memt_gates(
            p["erase"], p["write"], p["decay_latent"], w, spec)
        next_state = memt_update(
            p["k"], p["v"], erase_sqrt, write, log_decay,
            write_state if write_state is not None else read_state, spec)
    return out, next_state


# ---------------------------------------------------------------------------
# config.json "memt" block (release layout, validated the way the released
# server validates it: pinned algorithm switches, no unknown keys)
# ---------------------------------------------------------------------------

# Every field the released kernels implement exactly one variant of. A config
# that flips any of these describes a different algorithm, so it is refused
# rather than served with the wrong one (mirrors memt_sglang's
# FIXED_QWEN_MEMORY_CONFIG).
MEMT_FIXED_FIELDS = {
    "memory_update_mode": "token",
    "gdn_version": "stable_gdn2",
    "memory_use_short_conv": False,
    "memory_head_mode": "mha",
    "memory_read_scale": "k_inv_sqrt",
    "linear_attention_state_policy": "reset_per_chunk",
    "tail_policy": "commit",
    "output_gate_type": "silu",
    "output_gate_bias": False,
    "output_gate_softcap_enabled": True,
    "stable_gdn2_sqrt_erase_gate": True,
    "backbone_gdn_override": False,
}
MEMT_SIZE_FIELDS = ("num_memory_heads", "num_kv_heads", "chunk_tokens", "output_gate_softcap_limit")


@dataclass(frozen = True)
class MemTLayout:
    spec: MemTSpec
    layers: tuple[int, ...]     # decoder layers carrying a memory block
    chunk_tokens: int           # release's commit tiling unit (informational here)


def parse_memt_block(
    memt: dict,
    *,
    hidden_size: int,
    num_q_heads: int,
    head_dim: int,
    rms_norm_eps: float,
    layer_types: list[str],
) -> MemTLayout:
    """Validate the ``memt`` object of a MemT-1.0-Nano ``config.json`` against
    the backbone's attention geometry and return the memory layout. Memory
    blocks sit on every ``full_attention`` layer, one memory head per query
    head (``memory_head_mode: mha``)."""
    if not isinstance(memt, dict):
        raise ValueError("config.json: 'memt' must be an object")
    known = set(MEMT_FIXED_FIELDS) | set(MEMT_SIZE_FIELDS)
    missing = sorted(known - memt.keys())
    unknown = sorted(memt.keys() - known)
    if missing or unknown:
        raise ValueError(f"config.json 'memt': missing={missing}, unknown={unknown} "
                         f"(only the released MemT-1.0-Nano layout is supported)")
    for k, v in MEMT_FIXED_FIELDS.items():
        if type(memt[k]) is not type(v) or memt[k] != v:
            raise ValueError(f"config.json 'memt.{k}' = {memt[k]!r}: only {v!r} is supported")
    if memt["num_memory_heads"] != num_q_heads or memt["num_kv_heads"] != num_q_heads:
        raise ValueError("config.json 'memt': num_memory_heads and num_kv_heads must equal "
                         f"the backbone's num_attention_heads ({num_q_heads})")
    spec = MemTSpec(
        hidden_size = int(hidden_size),
        num_heads = int(num_q_heads),
        head_dim = int(head_dim),
        rms_norm_eps = float(rms_norm_eps),
        output_gate_softcap = float(memt["output_gate_softcap_limit"]),
    )
    layers = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    if not layers:
        raise ValueError("MemT: the backbone has no full_attention layers to carry memory")
    return MemTLayout(spec = spec, layers = layers, chunk_tokens = int(memt["chunk_tokens"]))
