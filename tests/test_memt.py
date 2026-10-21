"""
CPU tests for the MemT-1.0 native memory block math in
``exllamav3/training/memt.py`` (no GPU, no compiled extension, no model; the
module is loaded standalone like ``test_gdn.py``).

Checks:

  * the Nano geometry (hidden 2048, 16 heads x 256, 10 memory layers)
    reproduces the paper's Table 6 exactly: 450,931,360 trainable memory
    parameters and 10,485,760 state elements (40 MiB fp32);
  * a zero (or absent) memory state makes the block an exact no-op, so the
    backbone forward is untouched until a state is attached;
  * the vectorized sequential scan equals an explicit matrix-form
    transcription of the paper's update, eq. (22):
    ``S = (I - u^T u) D S + u^T (w * v)``, in fp64;
  * the whole block equals a port of the released SGLang ``QwenMemoryBlock``
    (its in_proj row slicing, gate nonlinearities, decay, gated softcap norm)
    with the Triton scan replaced by the matrix-form reference;
  * the symmetrized erase is non-expansive and a long random commit stays
    bounded (the paper's app. C stability argument);
  * every token of a turn reads the same snapshot: a token's readout does
    not depend on earlier tokens of the turn (read-only during generation);
  * fp64 gradcheck of read + commit through all six weight tensors and the
    incoming state.

Run:  python tests/test_memt.py
"""

from __future__ import annotations
import os
import sys
import types
import importlib.util
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.join(_ROOT, "exllamav3", "training")

_pkg = types.ModuleType("exl3train_memt")
_pkg.__path__ = [_TRAIN_DIR]
sys.modules["exl3train_memt"] = _pkg


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"exl3train_memt.{name}", os.path.join(_TRAIN_DIR, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"exl3train_memt.{name}"] = m
    spec.loader.exec_module(m)
    return m


_mt = _load("memt")
MemTSpec, MemTWeights = _mt.MemTSpec, _mt.MemTWeights


def _spec(hidden = 12, heads = 3, dim = 4, eps = 1e-6, softcap = 2.0):
    return MemTSpec(hidden_size = hidden, num_heads = heads, head_dim = dim,
                    rms_norm_eps = eps, output_gate_softcap = softcap)


def _weights(spec, dtype = torch.float32, scale = 0.3):
    shapes = _mt.memt_param_shapes(spec)
    t = {k: torch.randn(*s, dtype = dtype) * scale for k, s in shapes.items()}
    # Release init: decay_scale_log 0, decay_bias = softplus^-1(ln2 / half_life),
    # o_norm gain ~1. Random values around those keep the test in the regime
    # the real checkpoint lives in (decay < 1 per token, not vanishing).
    t["decay_scale_log"] = torch.randn(spec.num_heads, dtype = dtype) * 0.2
    t["decay_bias"] = torch.randn(spec.value_dim, dtype = dtype) * 0.5 - 1.0
    t["o_norm.weight"] = 1.0 + torch.randn(spec.head_dim, dtype = dtype) * 0.1
    return MemTWeights.from_dict(t)


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------

def _matrix_form_update(k_raw, v, erase_sqrt, write, log_decay, state0, spec):
    """Paper eq. (22) with explicit matrices, one head and token at a time:
    S = (I - u^T u) D S + u^T (w * v), u = sqrt(beta) * l2norm(k)."""
    b, t, H, D = k_raw.shape
    dt = torch.float64
    k = k_raw.to(dt)
    k = k / torch.sqrt((k * k).sum(-1, keepdim = True) + spec.l2norm_eps)
    u_all = erase_sqrt.to(dt) * k
    S = state0.to(dt).clone() if state0 is not None else torch.zeros(b, H, D, D, dtype = dt)
    if S.dim() == 3:
        S = S.unsqueeze(0).expand(b, -1, -1, -1).clone()
    I = torch.eye(D, dtype = dt)
    for bi in range(b):
        for h in range(H):
            for i in range(t):
                u = u_all[bi, i, h].unsqueeze(0)                  # [1, D] row vector
                Dm = torch.diag(log_decay[bi, i, h].to(dt).exp())
                wv = (write[bi, i, h].to(dt) * v[bi, i, h].to(dt)).unsqueeze(0)
                S[bi, h] = (I - u.T @ u) @ Dm @ S[bi, h] + u.T @ wv
    return S


class _SglangPort:
    """Port of memt_sglang/models/qwen/layer.py:QwenMemoryBlock (release
    code), with the Triton scan/read replaced by the matrix-form oracle.
    Keeps the release's own row bookkeeping (qkv_rows / gate_rows /
    weight[-dim:] / _write_gate_slice) so the slicing in memt.py is checked
    against an independent spelling of it."""

    def __init__(self, w: MemTWeights, spec: MemTSpec):
        self.w, self.spec = w, spec
        self.dt = w.in_proj.dtype      # release runs fp32 here; the test runs it in fp64
        H, D = spec.num_heads, spec.head_dim
        self.num_heads, self.head_dim = H, D
        self.value_dim = H * D
        self.read_scale = D ** -0.5
        self.qkv_rows = (H * D,) * 3
        erase_write_rows = (H * D,) * 2
        self.gate_rows = (*erase_write_rows, D, D)
        self.qkv_width = sum(self.qkv_rows)
        self.write_gate_rows = self.gate_rows[:-1]
        self._write_gate_slice = slice(self.qkv_width, self.qkv_width + sum(self.write_gate_rows))

    @staticmethod
    def _l2norm(x, eps = 1e-6):
        return x / torch.sqrt((x * x).sum(-1, keepdim = True) + eps)

    def _log_decay(self, decay_latent):
        w = self.w
        gate = F.softplus(F.linear(decay_latent, w.f_down).to(self.dt) + w.decay_bias.to(self.dt))
        return -w.decay_scale_log.to(self.dt).exp().view(self.num_heads, 1) * gate.view(
            *gate.shape[:-1], self.num_heads, -1)

    def _write_gates(self, gates, batch, seq_len):
        H, D = self.num_heads, self.head_dim
        *write_gates, decay_latent = torch.split(gates, self.write_gate_rows, dim = -1)
        log_decay = self._log_decay(decay_latent)
        erase = torch.exp(0.5 * F.logsigmoid(write_gates[0].to(self.dt))).view(batch, seq_len, H, D)
        write = write_gates[1].sigmoid().view(batch, seq_len, H, D)
        log_decay = log_decay.view(batch, seq_len, H, -1)
        return erase, write, log_decay

    def o_norm(self, hidden_states, gate):
        w, spec = self.w, self.spec
        hidden_states = hidden_states.to(self.dt)
        variance = hidden_states.pow(2).mean(-1, keepdim = True)
        hidden_states = hidden_states * torch.rsqrt(variance + spec.rms_norm_eps)
        hidden_states = w.o_norm * hidden_states
        gate_value = F.silu(gate.to(self.dt))
        limit = float(spec.output_gate_softcap)
        gate_value = limit * torch.tanh(gate_value / limit)
        return hidden_states * gate_value

    def forward(self, hidden_states, initial_state, commit):
        """hidden_states [tokens, hidden] (one sample), initial_state [H, D, D]."""
        weight, dim = self.w.in_proj, self.head_dim
        hb = hidden_states.unsqueeze(0)
        n = len(hidden_states)
        q, k, v = F.linear(hb, weight[: self.qkv_width]).split(self.qkv_rows, dim = -1)
        q, k, v = [p.view(1, n, self.num_heads, dim) for p in (q, k, v)]
        output_gate = F.linear(F.linear(hb, weight[-dim:]), self.w.g_down)
        qn = self._l2norm(q)
        read = torch.einsum("bthk,hkv->bthv", qn, initial_state.to(self.dt)) * self.read_scale
        output = self.o_norm(read.reshape(-1, dim), output_gate.reshape(-1, dim))
        output = output.view(n, self.value_dim)
        if not commit:
            return output, None
        erase, write, log_decay = self._write_gates(
            F.linear(hb, weight[self._write_gate_slice]), 1, n)
        final = _matrix_form_update(k, v, erase, write, log_decay,
                                    initial_state.unsqueeze(0), self.spec)[0]
        return output, final


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_nano_geometry_matches_paper():
    spec = MemTSpec(hidden_size = 2048, num_heads = 16, head_dim = 256, rms_norm_eps = 1e-6)
    layers = 10                                   # layers 3, 7, ..., 39 of 40
    shapes = _mt.memt_param_shapes(spec)
    assert shapes["in_proj.weight"] == (20992, 2048), shapes["in_proj.weight"]
    per_layer = _mt.memt_param_count(spec)
    assert per_layer == 45_093_136, per_layer
    assert per_layer * layers == 450_931_360, per_layer * layers          # Table 6
    h, k, v = spec.state_shape
    assert h * k * v * layers == 10_485_760                               # Table 6
    assert h * k * v * layers * 4 == 40 * 1024 * 1024                     # 40 MiB fp32
    sl = _mt.memt_in_proj_slices(spec)
    assert sl["q"] == slice(0, 4096) and sl["gate_latent"] == slice(20736, 20992)
    print("[memt] Nano geometry matches paper Table 6 PASSED")


def test_zero_state_is_noop():
    torch.manual_seed(0)
    spec = _spec()
    w = _weights(spec)
    x = torch.randn(2, 5, spec.hidden_size)
    out_none, st = _mt.memt_block_forward(x, w, spec, None)
    assert st is None
    out_zero, _ = _mt.memt_block_forward(x, w, spec, torch.zeros(spec.state_shape))
    assert out_none.shape == (2, 5, spec.value_dim)
    assert out_none.abs().max().item() == 0.0
    assert out_zero.abs().max().item() == 0.0
    # And a non-zero state does change the output (the gate is not degenerate)
    out_rand, _ = _mt.memt_block_forward(x, w, spec, torch.randn(spec.state_shape))
    assert out_rand.abs().max().item() > 1e-3
    print("[memt] zero / absent state is an exact no-op PASSED")


def test_update_matches_matrix_form():
    torch.manual_seed(1)
    spec = _spec()
    b, t, H, D = 2, 6, spec.num_heads, spec.head_dim
    k = torch.randn(b, t, H, D, dtype = torch.float64)
    v = torch.randn(b, t, H, D, dtype = torch.float64)
    erase_sqrt = torch.rand(b, t, H, D, dtype = torch.float64).sqrt()
    write = torch.rand(b, t, H, D, dtype = torch.float64)
    log_decay = -torch.rand(b, t, H, D, dtype = torch.float64) * 0.3
    s0 = torch.randn(b, H, D, D, dtype = torch.float64)
    out = _mt.memt_update_reference(k, v, erase_sqrt, write, log_decay, s0, spec)
    ref = _matrix_form_update(k, v, erase_sqrt, write, log_decay, s0, spec)
    err = (out - ref).abs().max().item()
    assert err < 1e-12, f"scan vs matrix form: max|Δ|={err}"
    # Broadcast [H, D, D] state and None state paths
    out3 = _mt.memt_update_reference(k, v, erase_sqrt, write, log_decay, s0[0], spec)
    ref3 = _matrix_form_update(k, v, erase_sqrt, write, log_decay, s0[0], spec)
    assert (out3 - ref3).abs().max().item() < 1e-12
    outz = _mt.memt_update_reference(k, v, erase_sqrt, write, log_decay, None, spec)
    refz = _matrix_form_update(k, v, erase_sqrt, write, log_decay, None, spec)
    assert (outz - refz).abs().max().item() < 1e-12
    print(f"[memt] sequential scan matches matrix-form eq. (22) (max|Δ|={err:.2e}) PASSED")


def test_block_matches_sglang_port():
    torch.manual_seed(2)
    spec = _spec(hidden = 16, heads = 4, dim = 8)
    w = _weights(spec, dtype = torch.float64)
    t = 7
    x = torch.randn(t, spec.hidden_size, dtype = torch.float64)
    s0 = torch.randn(*spec.state_shape, dtype = torch.float64) * 0.5
    port = _SglangPort(w, spec)
    ref_out, ref_state = port.forward(x, s0, commit = True)
    out, state = _mt.memt_block_forward(x.unsqueeze(0), w, spec, s0, commit = True)
    e_out = (out[0] - ref_out).abs().max().item()
    e_st = (state[0] - ref_state).abs().max().item()
    assert e_out < 1e-10, f"block output vs SGLang port: max|Δ|={e_out}"
    assert e_st < 1e-10, f"committed state vs SGLang port: max|Δ|={e_st}"
    # Read-only call: same output, no state
    ref_out2, _ = port.forward(x, s0, commit = False)
    out2, st2 = _mt.memt_block_forward(x.unsqueeze(0), w, spec, s0, commit = False)
    assert st2 is None and (out2[0] - ref_out2).abs().max().item() < 1e-10
    print(f"[memt] block matches SGLang QwenMemoryBlock port (out {e_out:.1e}, state {e_st:.1e}) PASSED")


def test_erase_nonexpansive_and_bounded():
    torch.manual_seed(3)
    spec = _spec(hidden = 8, heads = 2, dim = 6)
    H, D = spec.num_heads, spec.head_dim
    # One symmetrized erase step contracts (||u|| <= 1): ||(I - u^T u) S||_F <= ||S||_F
    for _ in range(50):
        u = torch.randn(D, dtype = torch.float64)
        u = u / u.norm() * torch.rand(1, dtype = torch.float64)      # ||u|| in [0, 1)
        S = torch.randn(D, D, dtype = torch.float64)
        A = torch.eye(D, dtype = torch.float64) - torch.outer(u, u)
        assert (A @ S).norm() <= S.norm() + 1e-12
    # A long random commit stays bounded: decay < 1 and |write * v| bounded
    # give the geometric bound of app. C, eq. (5).
    t = 400
    k = torch.randn(1, t, H, D)
    v = torch.randn(1, t, H, D).clamp(-3, 3)
    erase_sqrt = torch.rand(1, t, H, D).sqrt()
    write = torch.rand(1, t, H, D)
    log_decay = -torch.rand(1, t, H, D) * 0.1 - 0.01
    s0 = torch.zeros(1, H, D, D)
    state = _mt.memt_update_reference(k, v, erase_sqrt, write, log_decay, s0, spec)
    assert torch.isfinite(state).all()
    # per-token write norm <= ||u|| * ||w*v|| <= 1 * 3*sqrt(D); decay >= exp(-0.11)
    bound = 3 * (D ** 0.5) / (1 - float(torch.exp(torch.tensor(-0.01))))
    assert state.norm(dim = (-1, -2)).max().item() < bound
    print("[memt] symmetrized erase is non-expansive, long commit bounded PASSED")


def test_turn_reads_fixed_snapshot():
    torch.manual_seed(4)
    spec = _spec()
    w = _weights(spec)
    s0 = torch.randn(*spec.state_shape)
    x = torch.randn(1, 6, spec.hidden_size)
    out, _ = _mt.memt_block_forward(x, w, spec, s0)
    x2 = x.clone()
    x2[:, :3] = torch.randn(1, 3, spec.hidden_size)     # perturb earlier tokens
    out2, _ = _mt.memt_block_forward(x2, w, spec, s0)
    assert torch.equal(out[:, 3:], out2[:, 3:]), "later tokens must not see earlier tokens of the turn"
    assert not torch.equal(out[:, :3], out2[:, :3])
    print("[memt] within-turn readout reads a fixed snapshot PASSED")


def test_parse_memt_block_nano():
    memt = dict(_mt.MEMT_FIXED_FIELDS, num_memory_heads = 16, num_kv_heads = 16,
                chunk_tokens = 2048, output_gate_softcap_limit = 2.0)
    layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(40)]
    lay = _mt.parse_memt_block(memt, hidden_size = 2048, num_q_heads = 16, head_dim = 256,
                               rms_norm_eps = 1e-6, layer_types = layer_types)
    assert lay.layers == tuple(range(3, 40, 4)) and len(lay.layers) == 10          # Table 6
    assert lay.spec.output_gate_softcap == 2.0 and lay.chunk_tokens == 2048
    assert _mt.memt_param_count(lay.spec) * len(lay.layers) == 450_931_360
    # Pinned algorithm switches and geometry are enforced, unknown keys refused
    import copy
    for bad in (dict(memt, gdn_version = "gdn2"), dict(memt, num_kv_heads = 2),
                dict(memt, extra_switch = True), {k: v for k, v in memt.items() if k != "tail_policy"},
                dict(memt, output_gate_bias = 0)):
        try:
            _mt.parse_memt_block(bad, hidden_size = 2048, num_q_heads = 16, head_dim = 256,
                                 rms_norm_eps = 1e-6, layer_types = layer_types)
        except ValueError:
            continue
        raise AssertionError(f"accepted a bad memt block: {bad}")
    print("[memt] config 'memt' block parses to the Nano layout, variants refused PASSED")


def test_block_gradcheck():
    torch.manual_seed(5)
    spec = _spec(hidden = 6, heads = 2, dim = 3)
    w = _weights(spec, dtype = torch.float64)
    tensors = w.to_dict()
    for v_ in tensors.values():
        v_.requires_grad_(True)
    x = torch.randn(1, 3, spec.hidden_size, dtype = torch.float64, requires_grad = True)
    s0 = (torch.randn(*spec.state_shape, dtype = torch.float64) * 0.5).requires_grad_(True)
    names = list(tensors.keys())

    def fn(x_, s0_, *ws):
        ww = MemTWeights.from_dict(dict(zip(names, ws)))
        out, st = _mt.memt_block_forward(x_, ww, spec, s0_, commit = True)
        return out, st

    assert torch.autograd.gradcheck(fn, (x, s0, *[tensors[n] for n in names]),
                                    eps = 1e-6, atol = 1e-5, rtol = 1e-4)
    print("[memt] fp64 gradcheck of read + commit PASSED")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from util import run_timed
    run_timed([
        test_nano_geometry_matches_paper,
        test_zero_state_is_noop,
        test_update_matches_matrix_form,
        test_block_matches_sglang_port,
        test_erase_nonexpansive_and_bounded,
        test_turn_reads_fixed_snapshot,
        test_parse_memt_block_nano,
        test_block_gradcheck,
    ], label = "memt")
    print("\nALL MEMT TESTS PASSED")
