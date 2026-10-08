"""
Mamba2 decode kernels against float64 per-token torch recurrences.

ext.mamba2_dt_op(dt_raw, dt_bias, a_log, dt, g, dt_min, dt_max): per row and head h,
    dt = clamp(softplus(dt_raw + dt_bias[h]), dt_min, dt_max)     (softplus: beta 1, linear above 20)
    g  = -exp(a_log[h]) * dt                                        (fp32, from the unrounded dt)
with dt stored as bf16 (round to nearest). dt_raw [B, S, H] fp32, dt_bias / a_log [H] fp32, H <= 512; dtypes,
shapes and H are validated.

ext.cuda_recurrent_mamba2(mixed_xbc, g, dt, D, recurrent_state, out, Nk, Nv, Dk, Dv, slots, history): the SSD
recurrence over conv channel order [x (Nv * Dv), B (Nk * Dk), C (Nk * Dk)], v head h using group h // (Nv / Nk):
    S_t = exp(g_t[h]) * S_{t-1} + B_t[grp] (x) (dt_t[h] * x_t[h])        (Dk x Dv per head)
    y_t = C_t[grp] . S_t + D[h] * x_t[h]                                   (bf16, rounded toward zero)
The state of batch row b lives at recurrent_state[slots[b]] (b without slots), [num_slots, max_history + 1, Nv,
Dk, Dv] fp32, read from history entry 0. Without history the final state goes to entry 0; with history the state
after token t (t < seqlen - 1) goes to entry t + 1 and the final state to entry 0. Other entries and slots are
untouched. Dk must be a multiple of 32 and Dk, Dv <= 256; shapes, dtypes and the slots device are validated.
Reductions run in a fixed order, so results are bit-reproducible.
"""

import math

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


# mamba2_dt_op

def dt_reference(dt_raw, dt_bias, a_log, dt_min, dt_max):
    x = dt_raw.double() + dt_bias.double()
    sp = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))
    dt = sp.clamp(min = dt_min, max = dt_max)
    g = -torch.exp(a_log.double()) * dt
    return dt, g


def _dt_run(dt_raw, dt_bias, a_log, dt_min, dt_max):
    dt = torch.full(dt_raw.shape, 777.0, dtype = torch.bfloat16, device = dt_raw.device)
    g = torch.full(dt_raw.shape, 777.0, dtype = torch.float, device = dt_raw.device)
    ext.mamba2_dt_op(dt_raw, dt_bias, a_log, dt, g, dt_min, dt_max)
    torch.cuda.synchronize(dt_raw.device)
    return dt, g


def assert_dt_close(dt, g, ref_dt, ref_g, a_log):
    # dt: the fp32 softplus (log1pf of __expf: a few fp32 ulp, i.e. < 2^-20 relative) rounded once to bf16, so
    # within half a bf16 ulp (2^-8 relative) plus that. Values below the fp32 normal range may flush to zero.
    # g: fp32 product of __expf(a_log) and dt, each a few ulp: 2^-19 relative
    tiny = 2**-120
    err_dt = (dt.double() - ref_dt).abs()
    bad = err_dt > 2**-8 * (1 + 2**-10) * ref_dt.abs() + tiny
    assert not bad.any(), f"dt: {int(bad.sum())} elements outside half a bf16 ulp"
    err_g = (g.double() - ref_g).abs()
    bad = err_g > 2**-19 * ref_g.abs() + tiny * torch.exp(a_log.double()).max()
    assert not bad.any(), f"g: {int(bad.sum())} elements outside 2^-19 relative, worst {err_g.max().item():.3e}"


@pytest.mark.parametrize("H", [1, 24, 64, 80, 128, 512])
@pytest.mark.parametrize("B, S", [(1, 1), (3, 1), (2, 7), (1, 300)])
@pytest.mark.parametrize("dt_min, dt_max", [(0.0, math.inf), (1e-3, 0.1), (0.0, 1.0)])
@torch.inference_mode()
def test_mamba2_dt_op(device, H, B, S, dt_min, dt_max):
    dt_raw = torch.randn((B, S, H), device = device) * 4
    dt_bias = torch.randn((H,), device = device)
    a_log = torch.randn((H,), device = device)
    ref_dt, ref_g = dt_reference(dt_raw, dt_bias, a_log, dt_min, dt_max)
    dt, g = _dt_run(dt_raw, dt_bias, a_log, dt_min, dt_max)
    assert_dt_close(dt, g, ref_dt, ref_g, a_log)


@torch.inference_mode()
def test_mamba2_dt_op_extremes(device):
    # Around the softplus linear threshold (20), far into both tails, and infinities
    H = 8
    vals = torch.tensor([-1e4, -100.0, -30.0, -1.0, 0.0, 19.999, 20.0, 20.001, 30.0, 1e4], device = device)
    dt_raw = vals[:, None].expand(-1, H).contiguous().view(1, -1, H)
    dt_bias = torch.zeros((H,), device = device)
    a_log = torch.linspace(-3, 3, H, device = device)
    for dt_min, dt_max in [(0.0, math.inf), (1e-3, 0.1)]:
        ref_dt, ref_g = dt_reference(dt_raw, dt_bias, a_log, dt_min, dt_max)
        dt, g = _dt_run(dt_raw, dt_bias, a_log, dt_min, dt_max)
        assert_dt_close(dt, g, ref_dt, ref_g, a_log)
    inf_raw = torch.tensor([[[-math.inf, math.inf]]], device = device)
    dt, g = _dt_run(inf_raw, torch.zeros(2, device = device), torch.zeros(2, device = device), 0.0, 0.1)
    assert dt.float().tolist() == [[[0.0, pytest.approx(0.1, rel = 2**-8)]]]
    assert g[0, 0, 0].item() == 0.0 and g[0, 0, 1].item() == pytest.approx(-0.1, rel = 1e-6)


@torch.inference_mode()
def test_mamba2_dt_op_rejects(device):
    B, S, H = 1, 2, 4
    raw = torch.randn((B, S, H), device = device)
    bias = torch.randn((H,), device = device)
    a_log = torch.randn((H,), device = device)
    dt = torch.empty((B, S, H), dtype = torch.bfloat16, device = device)
    g = torch.empty((B, S, H), device = device)
    with pytest.raises(RuntimeError, match = "datatype"):
        ext.mamba2_dt_op(raw.half(), bias, a_log, dt, g, 0.0, 1.0)
    with pytest.raises(RuntimeError, match = "datatype"):
        ext.mamba2_dt_op(raw, bias, a_log, dt.float(), g, 0.0, 1.0)
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.mamba2_dt_op(raw, bias[:-1], a_log, dt, g, 0.0, 1.0)
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.mamba2_dt_op(raw, bias, a_log, dt, g[:, :1], 0.0, 1.0)
    big = torch.randn((1, 1, 513), device = device)
    with pytest.raises(RuntimeError, match = "too many heads"):
        ext.mamba2_dt_op(big, big[0, 0], big[0, 0], big.bfloat16(), big.clone(), 0.0, 1.0)


# cuda_recurrent_mamba2

def mamba2_reference(xbc, g, dt, D, state, slots, history, nk, nv, dk, dv):
    bsz, seqlen, _ = xbc.shape
    group = nv // nk
    x, Bm, Cm = torch.split(xbc.double(), [nv * dv, nk * dk, nk * dk], dim = -1)
    x = x.view(bsz, seqlen, nv, dv)
    Bm = Bm.view(bsz, seqlen, nk, dk).repeat_interleave(group, dim = 2)
    Cm = Cm.view(bsz, seqlen, nk, dk).repeat_interleave(group, dim = 2)
    g = g.double()
    dt = dt.double()
    D = D.double()
    out = torch.empty((bsz, seqlen, nv, dv), dtype = torch.float64, device = xbc.device)
    mag = torch.empty_like(out)
    new_state = state.double().clone()
    for b in range(bsz):
        s = int(slots[b]) if slots is not None else b
        S = state[s, 0].double()
        for t in range(seqlen):
            u = x[b, t] * dt[b, t][:, None]                                           # (nv, dv)
            S = S * g[b, t].exp()[:, None, None] + Bm[b, t][:, :, None] * u[:, None, :]
            out[b, t] = torch.einsum("hkv,hk->hv", S, Cm[b, t]) + D[:, None] * x[b, t]
            mag[b, t] = torch.einsum("hkv,hk->hv", S.abs(), Cm[b, t].abs()) + (D[:, None] * x[b, t]).abs()
            if history and t < seqlen - 1:
                new_state[s, t + 1] = S
        new_state[s, 0] = S
    return out, mag, new_state


def _mamba2_run(xbc, g, dt, D, state, slots, history, nk, nv, dk, dv):
    bsz, seqlen, _ = xbc.shape
    out = torch.full((bsz, seqlen, nv, dv), 777.0, dtype = torch.bfloat16, device = xbc.device)
    ext.cuda_recurrent_mamba2(xbc, g, dt, D, state, out, nk, nv, dk, dv, slots, history)
    torch.cuda.synchronize(xbc.device)
    return out


def _mamba2_inputs(device, bsz, seqlen, nk, nv, dk, dv, num_slots, hist, seed = 0):
    gen = torch.Generator(device = "cpu").manual_seed(seed)
    def rn(*shape):
        return torch.randn(shape, generator = gen).to(device)
    xbc = rn(bsz, seqlen, nv * dv + 2 * nk * dk).bfloat16()
    dt = (torch.rand((bsz, seqlen, nv), generator = gen).to(device) * 0.2 + 0.01).bfloat16()
    a = torch.rand((nv,), generator = gen).to(device) * 4
    g = -a[None, None, :] * dt.float()
    D = rn(nv)
    state = rn(num_slots, hist + 1, nv, dk, dv)
    return xbc, g, dt, D, state


def assert_mamba2_close(out, state, ref_out, mag, ref_state, seqlen):
    # Output: fp32 dot product of Dk terms plus the skip term, rounded toward zero to bf16 (up to one bf16 ulp,
    # 2^-7 relative). The fp32 state carries the rounding of one fma per step and __expf(g) (a few ulp) per
    # step, i.e. ~(seqlen + Dk) * 2^-23 of the summed term magnitudes; 2^-15 covers Dk <= 256, seqlen <= 32
    err = (out.double() - ref_out).abs()
    bound = 2**-7 * ref_out.abs() + 2**-15 * mag
    bad = err > bound
    assert not bad.any(), f"out: {int(bad.sum())} / {bad.numel()} outside bound, worst excess {(err - bound).max().item():.3e}"
    # State: each element is a linear recurrence; per step one decay multiply (fp32 __expf, a few ulp), one fma.
    # Error relative to the element's running magnitude stays below (seqlen + 1) * 2^-21
    tol = (seqlen + 1) * 2**-21
    scale = ref_state.abs().amax(dim = (-1, -2), keepdim = True) + 1e-30
    err = (state.double() - ref_state).abs() / scale
    assert err.max().item() < tol, f"state: relative error {err.max().item():.3e} >= {tol:.1e}"


MAMBA2_SHAPES = [
    # (num_k_heads = n_groups, num_v_heads = mamba heads, k_head_dim = ssm_state_size, v_head_dim = head_dim)
    (1, 1, 32, 32),
    (1, 4, 64, 64),
    (2, 8, 128, 64),
    (8, 64, 128, 64),
    (8, 128, 128, 64),
    (2, 6, 64, 96),
    (1, 2, 128, 200),
    (1, 2, 256, 64),
    (4, 4, 32, 256),
]


@pytest.mark.parametrize("nk, nv, dk, dv", MAMBA2_SHAPES)
@pytest.mark.parametrize("bsz, seqlen", [(1, 1), (3, 1), (2, 4), (1, 16)])
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("use_slots", [False, True])
@torch.inference_mode()
def test_recurrent_mamba2(device, nk, nv, dk, dv, bsz, seqlen, history, use_slots):
    hist = max(seqlen, 2)
    num_slots = bsz + 2
    xbc, g, dt, D, state = _mamba2_inputs(device, bsz, seqlen, nk, nv, dk, dv, num_slots, hist,
                                          seed = nk * 1000 + nv + dk + dv + seqlen)
    slots = torch.randperm(num_slots)[:bsz].to(torch.int32).to(device) if use_slots else None
    ref_out, mag, ref_state = mamba2_reference(xbc, g, dt, D, state, slots, history, nk, nv, dk, dv)
    out = _mamba2_run(xbc, g, dt, D, state, slots, history, nk, nv, dk, dv)
    assert_mamba2_close(out, state, ref_out, mag, ref_state, seqlen)
    # Entries and slots the call must not write are bit-identical to the input
    written = torch.zeros(state.shape[:2], dtype = torch.bool)
    for b in range(bsz):
        s = int(slots[b]) if slots is not None else b
        written[s, 0] = True
        if history:
            written[s, 1 : seqlen] = True
    keep = ~written.to(device)
    assert torch.equal(state[keep], ref_state[keep].float())


@pytest.mark.parametrize("history", [False, True])
@torch.inference_mode()
def test_recurrent_mamba2_deterministic(device, history):
    nk, nv, dk, dv, bsz, seqlen = 8, 64, 128, 64, 4, 4
    xbc, g, dt, D, state0 = _mamba2_inputs(device, bsz, seqlen, nk, nv, dk, dv, bsz, seqlen, seed = 7)
    results = []
    for _ in range(3):
        s = state0.clone()
        results.append((_mamba2_run(xbc, g, dt, D, s, None, history, nk, nv, dk, dv), s))
    for o, s in results[1:]:
        assert torch.equal(o, results[0][0]) and torch.equal(s, results[0][1])


@torch.inference_mode()
def test_recurrent_mamba2_history_matches_steps(device):
    # History entry n (1 <= n < seqlen) holds the state after the first n tokens: bit-identical to the final state
    # of a call over only those n tokens, whose outputs also match the full call position by position
    nk, nv, dk, dv, bsz, seqlen = 2, 8, 128, 64, 2, 6
    xbc, g, dt, D, state0 = _mamba2_inputs(device, bsz, seqlen, nk, nv, dk, dv, bsz, seqlen, seed = 3)
    full = state0.clone()
    out_full = _mamba2_run(xbc, g, dt, D, full, None, True, nk, nv, dk, dv)
    for n in range(1, seqlen):
        part = state0.clone()
        out_part = _mamba2_run(xbc[:, :n].contiguous(), g[:, :n].contiguous(), dt[:, :n].contiguous(), D, part,
                               None, False, nk, nv, dk, dv)
        assert torch.equal(out_part, out_full[:, :n])
        assert torch.equal(part[:, 0], full[:, n])


@torch.inference_mode()
def test_recurrent_mamba2_rejects(device):
    nk, nv, dk, dv, bsz, seqlen = 1, 2, 32, 32, 1, 2
    xbc, g, dt, D, state = _mamba2_inputs(device, bsz, seqlen, nk, nv, dk, dv, 1, 0)
    out = torch.empty((bsz, seqlen, nv, dv), dtype = torch.bfloat16, device = device)

    def call(xbc = xbc, g = g, dt = dt, D = D, state = state, out = out, nk = nk, nv = nv, dk = dk, dv = dv,
             slots = None, history = False):
        ext.cuda_recurrent_mamba2(xbc, g, dt, D, state, out, nk, nv, dk, dv, slots, history)

    with pytest.raises(RuntimeError, match = "divisible"):
        call(nk = 3)
    x48 = torch.zeros((bsz, seqlen, nv * dv + 2 * 48), dtype = torch.bfloat16, device = device)
    with pytest.raises(RuntimeError, match = "multiple of 32"):
        call(xbc = x48, dk = 48, state = torch.zeros((1, 2, nv, 48, dv), device = device))
    with pytest.raises(RuntimeError, match = "Max head dim"):
        call(xbc = torch.zeros((bsz, seqlen, nv * 288 + 2 * dk), dtype = torch.bfloat16, device = device), dv = 288,
             state = torch.zeros((1, 2, nv, dk, 288), device = device),
             out = torch.empty((bsz, seqlen, nv, 288), dtype = torch.bfloat16, device = device))
    with pytest.raises(RuntimeError, match = "mixed_xbc must be"):
        call(xbc = xbc[..., :-1].contiguous())
    with pytest.raises(RuntimeError, match = "g must be"):
        call(g = g[:, :1].contiguous())
    with pytest.raises(RuntimeError, match = "dt must be"):
        call(dt = dt[:, :1].contiguous())
    with pytest.raises(RuntimeError, match = "D must be"):
        call(D = D[:1].contiguous())
    with pytest.raises(RuntimeError, match = "recurrent_state must be"):
        call(history = True)  # history over 2 tokens needs at least 2 entries, state has 1
    with pytest.raises(RuntimeError, match = "core_attn_out must be"):
        call(out = out[:, :1].contiguous())
    with pytest.raises(RuntimeError, match = "datatype"):
        call(dt = dt.float())
    with pytest.raises(RuntimeError, match = "datatype"):
        call(xbc = xbc.half())
    with pytest.raises(RuntimeError, match = "slots must be"):
        call(slots = torch.zeros(2, dtype = torch.int32, device = device))
    with pytest.raises(RuntimeError, match = "same device"):
        call(slots = torch.zeros(1, dtype = torch.int32))


@pytest.mark.parametrize("history", [False, True])
@torch.inference_mode()
def test_recurrent_mamba2_guard_regions(device, history):
    # out and recurrent_state are contiguous views into larger sentinel-filled buffers: nothing outside is written
    nk, nv, dk, dv, bsz, seqlen, num_slots, hist = 2, 8, 128, 64, 2, 3, 3, 3
    xbc, g, dt, D, state0 = _mamba2_inputs(device, bsz, seqlen, nk, nv, dk, dv, num_slots, hist, seed = 11)
    slots = torch.tensor([2, 0], dtype = torch.int32, device = device)
    guard = 4096
    n_out = bsz * seqlen * nv * dv
    n_state = state0.numel()
    out_buf = torch.full((guard + n_out + guard,), 3.0e38, dtype = torch.bfloat16, device = device)
    st_buf = torch.full((guard + n_state + guard,), -1.0e30, dtype = torch.float, device = device)
    out = out_buf[guard : guard + n_out].view(bsz, seqlen, nv, dv)
    state = st_buf[guard : guard + n_state].view(state0.shape)
    state.copy_(state0)
    ref_out, mag, ref_state = mamba2_reference(xbc, g, dt, D, state0, slots, history, nk, nv, dk, dv)
    ext.cuda_recurrent_mamba2(xbc, g, dt, D, state, out, nk, nv, dk, dv, slots, history)
    torch.cuda.synchronize(device)
    assert_mamba2_close(out, state, ref_out, mag, ref_state, seqlen)
    assert (out_buf[:guard] == 3.0e38).all() and (out_buf[guard + n_out:] == 3.0e38).all()
    assert (st_buf[:guard] == -1.0e30).all() and (st_buf[guard + n_state:] == -1.0e30).all()


@pytest.mark.parametrize("H", [24, 128])
@torch.inference_mode()
def test_mamba2_dt_op_guard_regions(device, H):
    # dt and g are contiguous views into sentinel-filled buffers: nothing outside them is written
    B, S, guard = 3, 5, 1024
    n = B * S * H
    dt_buf = torch.full((guard + n + guard,), 777.0, dtype = torch.bfloat16, device = device)
    g_buf = torch.full((guard + n + guard,), 777.0, dtype = torch.float, device = device)
    dt = dt_buf[guard : guard + n].view(B, S, H)
    g = g_buf[guard : guard + n].view(B, S, H)
    dt_raw = torch.randn((B, S, H), device = device)
    dt_bias = torch.randn((H,), device = device)
    a_log = torch.randn((H,), device = device)
    ext.mamba2_dt_op(dt_raw, dt_bias, a_log, dt, g, 0.0, math.inf)
    torch.cuda.synchronize(device)
    ref_dt, ref_g = dt_reference(dt_raw, dt_bias, a_log, 0.0, math.inf)
    assert_dt_close(dt, g, ref_dt, ref_g, a_log)
    for buf in (dt_buf, g_buf):
        assert (buf[:guard] == 777.0).all() and (buf[guard + n:] == 777.0).all()
