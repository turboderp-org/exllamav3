"""
ext.cuda_causal_conv1d_update(x, conv_state, slots, weight, bias, out, activation, history): depthwise causal conv1d
decode step over a slot-indexed state, all bf16.

    x (bsz, dim, seqlen), conv_state (num_slots, dim, state_size >= K), weight (dim, K <= 16), bias (dim) | None,
    out (bsz, seqlen, dim), slots (bsz) int32 | None (identity, then num_slots >= bsz)

For batch row b with slot s = slots[b], the input sequence is seq = concat(conv_state[s, :, 0:K], x[b]) and
    out[b, t, d] = act(bias[d] + sum_k weight[d, k] * seq[d, t + 1 + k]),  act = silu if activation else identity
Without history the last K inputs are written to conv_state[s, :, 0:K] (columns >= K untouched); with history the
last min(state_size, K + seqlen) inputs go to the tail of conv_state[s] (the head stays until a rewind). Slots
not named in `slots` are untouched. State values are copies of bf16 inputs, so state updates are exact; out is
the fp32 sum rounded to bf16. Contiguity, dtypes, shapes, K and state_size are validated (TORCH_CHECK).

Reference: float64 torch over the same bf16 inputs.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


def reference(x, conv_state, slots, weight, bias, activation, history):
    bsz, dim, seqlen = x.shape
    K = weight.shape[1]
    state_size = conv_state.shape[2]
    state = conv_state.clone()
    out = torch.empty((bsz, seqlen, dim), dtype = torch.float64, device = x.device)
    mag = torch.empty_like(out)
    w = weight.double()
    for b in range(bsz):
        s = int(slots[b]) if slots is not None else b
        seq = torch.cat([conv_state[s, :, :K], x[b]], dim = -1).double()           # (dim, K + seqlen)
        win = seq.unfold(1, K, 1)[:, 1:]                                             # (dim, seqlen, K)
        terms = win * w[:, None, :]
        acc = terms.sum(-1)
        m = terms.abs().sum(-1)
        if bias is not None:
            acc = acc + bias.double()[:, None]
            m = m + bias.double().abs()[:, None]
        if activation:
            acc = acc * torch.sigmoid(acc)
        out[b] = acc.T
        mag[b] = m.T
        if history:
            n = min(state_size, K + seqlen)
            state[s, :, state_size - n:] = seq[:, -n:].bfloat16()
        else:
            state[s, :, :K] = seq[:, -K:].bfloat16()
    return out, mag, state


def assert_conv_out(out, ref, mag):
    # out is the fp32 fma chain (K + 1 terms) rounded once to bf16 (RN): half a bf16 ulp (2^-8 relative) of the
    # value, plus the fp32 accumulation error (<= (K + 1) * 2^-24 of the summed magnitudes, bounded by 2^-19 for
    # K <= 16, times silu's maximum slope 1.1) and the __expf error in silu (a few fp32 ulp of the value)
    err = (out.double() - ref).abs()
    bound = 2**-8 * ref.abs() * (1 + 2**-10) + 2**-18 * mag + 1e-30
    bad = err > bound
    assert not bad.any(), \
        f"{int(bad.sum())} / {bad.numel()} outside bound, worst excess {(err - bound).max().item():.3e}"


def _run(x, state, slots, w, bias, activation, history):
    out = torch.full((x.shape[0], x.shape[2], x.shape[1]), 12345.0, dtype = torch.bfloat16, device = x.device)
    ext.cuda_causal_conv1d_update(x, state, slots, w, bias, out, activation, history)
    torch.cuda.synchronize(x.device)
    return out


@pytest.mark.parametrize("dim", [8, 300, 8192, 10240])
@pytest.mark.parametrize("K", [2, 4, 16])
@pytest.mark.parametrize("seqlen", [1, 2, 5, 32])
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("activation", [True, False])
@torch.inference_mode()
def test_conv1d_update(device, dim, K, seqlen, history, activation):
    bsz = 3
    num_slots = 5
    state_size = K + (seqlen + 2 if history else 0)
    slots = torch.tensor([3, 0, 4], dtype = torch.int32, device = device)
    x = torch.randn((bsz, dim, seqlen), device = device).bfloat16()
    w = (torch.randn((dim, K), device = device) / K ** 0.5).bfloat16()
    bias = torch.randn((dim,), device = device).bfloat16()
    state = torch.randn((num_slots, dim, state_size), device = device).bfloat16()
    ref_out, mag, ref_state = reference(x, state, slots, w, bias, activation, history)
    out = _run(x, state, slots, w, bias, activation, history)
    assert_conv_out(out, ref_out, mag)
    assert torch.equal(state, ref_state)


@pytest.mark.parametrize("history", [False, True])
@torch.inference_mode()
def test_conv1d_update_no_slots_no_bias(device, history):
    # slots = None maps batch row b to slot b; slots beyond bsz untouched
    bsz, dim, K, seqlen = 2, 640, 4, 3
    x = torch.randn((bsz, dim, seqlen), device = device).bfloat16()
    w = torch.randn((dim, K), device = device).bfloat16()
    state = torch.randn((4, dim, K + 4), device = device).bfloat16()
    ref_out, mag, ref_state = reference(x, state, None, w, None, True, history)
    out = _run(x, state, None, w, None, True, history)
    assert_conv_out(out, ref_out, mag)
    assert torch.equal(state, ref_state)


@torch.inference_mode()
def test_conv1d_update_history_longer_than_state(device):
    # seqlen + K > state_size with history: only the last state_size inputs are kept, filling the whole buffer
    bsz, dim, K, seqlen = 1, 256, 4, 10
    x = torch.randn((bsz, dim, seqlen), device = device).bfloat16()
    w = torch.randn((dim, K), device = device).bfloat16()
    state = torch.randn((1, dim, K + 3), device = device).bfloat16()
    ref_out, mag, ref_state = reference(x, state, None, w, None, True, True)
    out = _run(x, state, None, w, None, True, True)
    assert_conv_out(out, ref_out, mag)
    assert torch.equal(state, ref_state)


@torch.inference_mode()
def test_conv1d_update_matches_sequential_steps(device):
    # Without history, one call over seqlen tokens == seqlen single-token calls (same window carried in state)
    bsz, dim, K, seqlen = 2, 1024, 4, 6
    slots = torch.tensor([1, 0], dtype = torch.int32, device = device)
    x = torch.randn((bsz, dim, seqlen), device = device).bfloat16()
    w = torch.randn((dim, K), device = device).bfloat16()
    bias = torch.randn((dim,), device = device).bfloat16()
    state0 = torch.randn((2, dim, K), device = device).bfloat16()
    s1 = state0.clone()
    out1 = _run(x, s1, slots, w, bias, True, False)
    s2 = state0.clone()
    out2 = torch.cat([_run(x[:, :, t : t + 1].contiguous(), s2, slots, w, bias, True, False) for t in range(seqlen)], dim = 1)
    assert torch.equal(out1, out2)
    assert torch.equal(s1, s2)


@torch.inference_mode()
def test_conv1d_update_deterministic(device):
    bsz, dim, K, seqlen = 4, 4096, 4, 8
    x = torch.randn((bsz, dim, seqlen), device = device).bfloat16()
    w = torch.randn((dim, K), device = device).bfloat16()
    state0 = torch.randn((bsz, dim, K + 8), device = device).bfloat16()
    outs = []
    for _ in range(3):
        s = state0.clone()
        outs.append((_run(x, s, None, w, None, True, True), s))
    for o, s in outs[1:]:
        assert torch.equal(o, outs[0][0]) and torch.equal(s, outs[0][1])


@torch.inference_mode()
def test_conv1d_update_rejects(device):
    bsz, dim, K, seqlen = 2, 64, 4, 2
    bf = torch.bfloat16
    x = torch.randn((bsz, dim, seqlen), device = device).to(bf)
    w = torch.randn((dim, K), device = device).to(bf)
    state = torch.randn((bsz, dim, K), device = device).to(bf)
    out = torch.empty((bsz, seqlen, dim), dtype = bf, device = device)

    def call(x = x, state = state, slots = None, w = w, bias = None, out = out):
        ext.cuda_causal_conv1d_update(x, state, slots, w, bias, out, True, False)

    with pytest.raises(RuntimeError, match = "CONV1D_MAX_K"):
        call(w = torch.randn((dim, 17), device = device).to(bf), state = torch.randn((bsz, dim, 17), device = device).to(bf))
    with pytest.raises(RuntimeError, match = "at least K"):
        call(state = torch.randn((bsz, dim, K - 1), device = device).to(bf))
    with pytest.raises(RuntimeError, match = "contiguous"):
        call(x = torch.randn((bsz, seqlen, dim), device = device).to(bf).transpose(1, 2))
    with pytest.raises(RuntimeError, match = "conv_state must be"):
        call(state = torch.randn((bsz, dim + 1, K), device = device).to(bf))
    with pytest.raises(RuntimeError, match = "weight must be"):
        call(w = torch.randn((dim + 1, K), device = device).to(bf))
    with pytest.raises(RuntimeError, match = "out must be"):
        call(out = torch.empty((bsz, dim, seqlen), dtype = bf, device = device))
    with pytest.raises(RuntimeError, match = "datatype"):
        call(x = x.half())
    with pytest.raises(RuntimeError, match = "datatype"):
        call(bias = torch.zeros(dim, dtype = torch.half, device = device))
    with pytest.raises(RuntimeError, match = "datatype"):
        call(slots = torch.zeros(bsz, dtype = torch.long, device = device))
    with pytest.raises(RuntimeError, match = "slots must be"):
        call(slots = torch.zeros(bsz + 1, dtype = torch.int32, device = device))
    with pytest.raises(RuntimeError, match = "too small"):
        call(state = torch.randn((bsz - 1, dim, K), device = device).to(bf))


@pytest.mark.parametrize("history", [False, True])
@torch.inference_mode()
def test_conv1d_update_guard_regions(device, history):
    # out and conv_state are contiguous views into larger sentinel-filled buffers: nothing outside them is written
    bsz, dim, K, seqlen, num_slots = 2, 300, 4, 3, 3
    state_size = K + (4 if history else 0)
    guard = 1024
    sentinel = 3.0e38
    n_out = bsz * seqlen * dim
    n_state = num_slots * dim * state_size
    out_buf = torch.full((guard + n_out + guard,), sentinel, dtype = torch.bfloat16, device = device)
    st_buf = torch.full((guard + n_state + guard,), sentinel, dtype = torch.bfloat16, device = device)
    out = out_buf[guard : guard + n_out].view(bsz, seqlen, dim)
    state = st_buf[guard : guard + n_state].view(num_slots, dim, state_size)
    state.copy_(torch.randn(state.shape, device = device))
    slots = torch.tensor([2, 0], dtype = torch.int32, device = device)
    x = torch.randn((bsz, dim, seqlen), device = device).bfloat16()
    w = torch.randn((dim, K), device = device).bfloat16()
    bias = torch.randn((dim,), device = device).bfloat16()
    ref_out, mag, ref_state = reference(x, state, slots, w, bias, True, history)
    ext.cuda_causal_conv1d_update(x, state, slots, w, bias, out, True, history)
    torch.cuda.synchronize(device)
    assert_conv_out(out, ref_out, mag)
    assert torch.equal(state, ref_state)
    for buf, n in ((out_buf, n_out), (st_buf, n_state)):
        assert (buf[:guard] == sentinel).all() and (buf[guard + n:] == sentinel).all()
