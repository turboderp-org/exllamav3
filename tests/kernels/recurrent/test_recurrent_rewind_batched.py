"""
Batched recurrent-layer rewind (speculative decoding draft rejection), against plain torch references:

ext.batched_conv_rewind(jobs, device_index): for each ConvRewindJob(src, dst, dim, cdim, stride), for every channel
d < dim, dst[d * stride + k] <- src[d * stride + k] for k < cdim, bf16, with the cdim source elements read before
any is written so overlapping windows (src - dst < cdim elements) copy like a memmove. cdim > CONV1D_MAX_K (16) is
rejected. Nothing outside the dst windows is written. Any number of jobs (the host splits them into launches of
64), an empty list is a no-op, jobs of one call may target different tensors with different sizes.

GDNLayerState: _rewind_layers (ext.rewind_recurrent_layers) must shift the conv ring (window <- last cdim columns
before the rejected tokens), and a conv update with history followed by a conv rewind of r tokens must leave the
window a plain update over the first seqlen - r tokens would leave. Over the layer's staged inputs (the views a
speculative pass writes into) it must rebuild the state a pass over the accepted prefix would have produced from the
base row, bit for bit, leaving the base row untouched. Layers of different geometries and state dtypes may share one
call, and the addresses follow the buffers through a free and reallocation.
"""


import types

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.gated_delta_net import _rewind_layers

CONV1D_MAX_K = 16


def _conv_job(t: torch.Tensor, slot: int, src_col: int, dst_col: int, cdim: int, dim: int | None = None):
    # t: (slots, dim, state_size) bf16, per-channel stride t.stride(1)
    return ext.ConvRewindJob(
        t[slot, 0, src_col:].data_ptr(),
        t[slot, 0, dst_col:].data_ptr(),
        t.shape[1] if dim is None else dim,
        cdim,
        t.stride(1),
    )


def _conv_ref(t: torch.Tensor, slot: int, src_col: int, dst_col: int, cdim: int, dim: int | None = None):
    dim = t.shape[1] if dim is None else dim
    src = t[slot, :dim, src_col : src_col + cdim].clone()
    t[slot, :dim, dst_col : dst_col + cdim] = src


# batched_conv_rewind

@pytest.mark.parametrize("dim", [1, 100, 256, 257, 8192, 10240])
@pytest.mark.parametrize("cdim", [2, 4, 16])
@pytest.mark.parametrize("max_history", [1, 3, 20])
@torch.inference_mode()
def test_batched_conv_rewind_windows(device, dim, cdim, max_history):
    # conv_state (slots, dim, cdim + max_history): every rejection count r in [0, max_history] copies the window
    # ending r columns before the end to the head. For max_history < cdim the windows overlap and the copy
    # must behave like a memmove (source read before written)
    num_slots = 2
    state_size = cdim + max_history
    for r in range(max_history + 1):
        t = torch.randn((num_slots, dim, state_size), device = device).bfloat16()
        e = t.clone()
        p = state_size - r
        slot = r % num_slots
        ext.batched_conv_rewind([_conv_job(t, slot, p - cdim, 0, cdim)], device.index)
        _conv_ref(e, slot, p - cdim, 0, cdim)
        torch.cuda.synchronize(device)
        assert torch.equal(t, e), f"r = {r}"


@pytest.mark.parametrize("num_layers", [2, 64, 65, 130])
@torch.inference_mode()
def test_batched_conv_rewind_many_layers(device, num_layers):
    # Many jobs over separate tensors with different dims and kernel sizes in one call (several launches past 64)
    gen = torch.Generator(device = "cpu").manual_seed(num_layers)
    tensors, expected, jobs = [], [], []
    for i in range(num_layers):
        dim = [64, 300, 1024, 2048][i % 4]
        cdim = [4, 2, 3, 16][i % 4]
        hist = [1, 2, 5][i % 3]
        t = torch.randn((3, dim, cdim + hist), generator = gen).bfloat16().to(device)
        e = t.clone()
        slot = i % 3
        r = i % (hist + 1)
        p = cdim + hist - r
        jobs.append(_conv_job(t, slot, p - cdim, 0, cdim))
        _conv_ref(e, slot, p - cdim, 0, cdim)
        tensors.append(t)
        expected.append(e)
    ext.batched_conv_rewind(jobs, device.index)
    torch.cuda.synchronize(device)
    for i, (t, e) in enumerate(zip(tensors, expected)):
        assert torch.equal(t, e), f"layer {i}"


@torch.inference_mode()
def test_batched_conv_rewind_partial_dim(device):
    # dim smaller than the tensor's channel count: channels >= dim are not touched
    t = torch.randn((1, 512, 8), device = device).bfloat16()
    e = t.clone()
    ext.batched_conv_rewind([_conv_job(t, 0, 4, 0, 4, dim = 300)], device.index)
    _conv_ref(e, 0, 4, 0, 4, dim = 300)
    torch.cuda.synchronize(device)
    assert torch.equal(t, e)


@torch.inference_mode()
def test_batched_conv_rewind_rejects_large_cdim(device):
    t = torch.randn((1, 64, 40), device = device).bfloat16()
    e = t.clone()
    with pytest.raises(RuntimeError, match = "CONV1D_MAX_K"):
        ext.batched_conv_rewind([_conv_job(t, 0, 20, 0, CONV1D_MAX_K + 1)], device.index)
    torch.cuda.synchronize(device)
    assert torch.equal(t, e)


# GDNLayerState: the conv rewind against a torch reference, the replay against a kernel pass over the prefix

def _layer_state(device, num_slots, max_history, fdim, k, nv, dk, dv, nk = None, state_dtype = torch.float):
    from exllamav3.modules.gated_delta_net import GDNLayerState
    module = types.SimpleNamespace(
        fdim_qkv = fdim,
        conv_kernel_size = k,
        num_k_heads = nk or nv,
        num_v_heads = nv,
        k_head_dim = dk,
        v_head_dim = dv,
    )
    ls = GDNLayerState(module, num_slots, max_history, cache_id = 0, state_dtype = state_dtype)
    ls.alloc(device)
    ls.conv_state.copy_(torch.randn(ls.conv_state.shape, device = device))
    ls.recurrent_state.normal_()
    return ls


def _conv_rewind_reference(ls, slot, last_history, num_tokens):
    cdim = ls.module.conv_kernel_size
    if last_history > 0:
        p = ls.conv_state.shape[-1] - num_tokens
        ls.conv_state[slot, :, :cdim] = ls.conv_state[slot, :, p - cdim : p].clone()


@pytest.mark.parametrize("max_history", [1, 3, 7])
@torch.inference_mode()
def test_gdn_layer_state_conv_rewind_matches_reference(device, max_history):
    num_slots = 3
    for slot in range(num_slots):
        for last_history in range(max_history + 1):
            for num_tokens in range(last_history + 1):
                a = _layer_state(device, num_slots, max_history, 384, 4, 4, 32, 64)
                ref = _layer_state(device, num_slots, max_history, 384, 4, 4, 32, 64)
                ref.conv_state.copy_(a.conv_state)
                a.spec_shape = (1, last_history + 1, 1)
                _rewind_layers([a], slot, last_history, num_tokens, 0, a.spec_shape, 2 * slot, 2 * slot + 1)
                torch.cuda.synchronize(device)
                _conv_rewind_reference(ref, slot, last_history, num_tokens)
                assert torch.equal(a.conv_state, ref.conv_state), (slot, last_history, num_tokens)
    a = _layer_state(device, 1, 3, 384, 4, 4, 32, 64)
    a.spec_shape = (1, 4, 1)
    with pytest.raises(RuntimeError, match = "outside the recorded history"):
        _rewind_layers([a], 0, 3, 4, 0, a.spec_shape, 0, 1)


@pytest.mark.parametrize("state_dtype", [torch.float, torch.half], ids = ["fp32_state", "fp16_state"])
@pytest.mark.parametrize("max_history", [3, 64])
@pytest.mark.parametrize("bsz", [1, 3])
@torch.inference_mode()
def test_gdn_layer_state_replay(device, max_history, bsz, state_dtype):
    # A speculative pass of max_history + 1 tokens writes its scan inputs into the staged views; every batch row
    # and every prefix then replays to the state a prefix-length pass from the same base row produces
    nk, nv, dk, dv, fdim = 2, 4, 64, 64, 2 * 2 * 64 + 4 * 64
    seqlen = max_history + 1
    ls = _layer_state(device, bsz, max_history, fdim, 4, nv, dk, dv, nk = nk, state_dtype = state_dtype)
    conv_out, beta, g = ls.staged_views(bsz, seqlen)
    conv_out.copy_((torch.randn(conv_out.shape, device = device) * 0.25).bfloat16())
    beta.copy_(torch.sigmoid(torch.randn(beta.shape, device = device)).bfloat16())
    g.copy_(torch.randn(g.shape, device = device) * 0.5 - 1.0)
    pool0 = ls.recurrent_state.clone()
    for row in range(bsz):
        for parity in (0, 1):
            base, scratch = 2 * row + parity, 2 * row + 1 - parity
            for prefix in sorted({1, seqlen // 2, seqlen - 1}):
                expect = pool0.clone()
                out = torch.empty((1, prefix, nv, dv), dtype = torch.bfloat16, device = device)
                ext.cuda_recurrent_gated_delta_rule(
                    conv_out[row : row + 1, :prefix].contiguous(), g[row : row + 1, :prefix].contiguous(),
                    beta[row : row + 1, :prefix].contiguous(), expect, out, nk, nv, dk, dv,
                    torch.tensor([scratch], dtype = torch.int32, device = device), True,
                    torch.tensor([base], dtype = torch.int32, device = device))
                ls.recurrent_state.copy_(pool0)
                _rewind_layers([ls], row, seqlen - 1, seqlen - prefix, row, (bsz, seqlen, 1), base, scratch)
                torch.cuda.synchronize(device)
                assert torch.equal(ls.recurrent_state, expect), (row, parity, prefix)
                assert torch.equal(ls.recurrent_state[base], pool0[base])
    with pytest.raises(AssertionError):
        _rewind_layers([ls], 0, seqlen - 2, 1, 0, (bsz, seqlen - 1, 1), 0, 1)   # staged shape mismatch
    with pytest.raises(AssertionError):
        ls.staged_views(bsz, seqlen + 1)                  # longer than max_history + 1


@pytest.mark.parametrize("bsz", [1, 3])
@torch.inference_mode()
def test_gdn_layer_state_replay_token_major(device, bsz):
    # The same pass staged one step at a time (token-major rows, a recurrent drafter's loop) replays to the same
    # states as the row-major staging of the whole pass, through the replay job's token stride
    nk, nv, dk, dv, fdim, max_history = 2, 4, 64, 64, 2 * 2 * 64 + 4 * 64, 5
    steps = max_history + 1
    conv_all = (torch.randn((bsz, steps, fdim), device = device) * 0.25).bfloat16()
    beta_all = torch.sigmoid(torch.randn((bsz, steps, nv), device = device)).bfloat16()
    g_all = torch.randn((bsz, steps, nv), device = device) * 0.5 - 1.0
    rm = _layer_state(device, bsz, max_history, fdim, 4, nv, dk, dv, nk = nk)
    c, b, g = rm.staged_views(bsz, steps)
    c.copy_(conv_all); b.copy_(beta_all); g.copy_(g_all)
    tm = _layer_state(device, bsz, max_history, fdim, 4, nv, dk, dv, nk = nk)
    tm.recurrent_state.copy_(rm.recurrent_state)
    for t in range(steps):
        c, b, g = tm.staged_step_views(bsz, t)
        c.copy_(conv_all[:, t : t + 1]); b.copy_(beta_all[:, t : t + 1]); g.copy_(g_all[:, t : t + 1])
    assert tm.spec_shape == (bsz, steps, bsz) and rm.spec_shape == (bsz, steps, 1)
    pool0 = rm.recurrent_state.clone()
    for row in range(bsz):
        for prefix in (1, steps // 2, steps - 1):
            rm.recurrent_state.copy_(pool0); tm.recurrent_state.copy_(pool0)
            for ls, shape in ((rm, (bsz, steps, 1)), (tm, (bsz, steps, bsz))):
                _rewind_layers([ls], row, steps - 1, steps - prefix, row, shape, 2 * row, 2 * row + 1)
            torch.cuda.synchronize(device)
            assert torch.equal(tm.recurrent_state, rm.recurrent_state), (row, prefix)
    with pytest.raises(AssertionError):
        tm.staged_step_views(bsz, 2)   # a step must continue the staged pass


def _staged_layer(device, num_slots, bsz, seqlen, nk, nv, dk, dv, state_dtype):
    # A layer state with random staged inputs of a (bsz, seqlen) pass
    ls = _layer_state(device, num_slots, seqlen - 1, 2 * nk * dk + nv * dv, 4, nv, dk, dv, nk = nk, state_dtype = state_dtype)
    for t, scale, shift in zip(ls.staged_views(bsz, seqlen), (0.25, 0.1, 0.5), (0.0, 0.5, -1.0)):
        t.copy_(torch.randn(t.shape, device = device) * scale + shift)
    return ls


@torch.inference_mode()
def test_rewind_layers_mixed_geometries(device):
    # One call over layers of two geometries and both state dtypes, more than a launch's worth of one of them, leaves
    # every layer exactly where a call over that layer alone does
    specs = [(2, 4, 64, 64, torch.half)] * 50 + [(2, 4, 64, 64, torch.float)] * 2 + [(4, 8, 128, 128, torch.half)] * 2
    layers = [_staged_layer(device, 2, 2, 5, *spec) for spec in specs[::-1]]
    alone = [_staged_layer(device, 2, 2, 5, *spec) for spec in specs[::-1]]
    for a, b in zip(layers, alone):
        for x, y in zip(a._tensors(), b._tensors()):
            y.copy_(x)
    before = [l.recurrent_state.clone() for l in layers]
    args = (1, 4, 2, 1, (2, 5, 1), 3, 2)   # slot 1 on odd parity: base row 3, scratch row 2
    _rewind_layers(layers, *args)
    for l in alone:
        _rewind_layers([l], *args)
    torch.cuda.synchronize(device)
    for i, (a, b, s) in enumerate(zip(layers, alone, before)):
        assert torch.equal(a.recurrent_state, b.recurrent_state) and torch.equal(a.conv_state, b.conv_state), i
        assert not torch.equal(a.recurrent_state[2], s[2]), i
        assert torch.equal(a.recurrent_state[[0, 1, 3]], s[[0, 1, 3]]), i


@torch.inference_mode()
def test_rewind_layers_follow_reallocation(device):
    # The stored addresses are dropped with the buffers: a rewind after free() and alloc() acts on the new ones. The
    # old buffers are held, so that the allocator cannot hand the same addresses out again
    ls = _staged_layer(device, 1, 1, 5, 2, 4, 64, 64, torch.half)
    _rewind_layers([ls], 0, 4, 2, 0, (1, 5, 1), 0, 1)
    held = ls._tensors()
    ls.free()
    ls.alloc(device)
    ls.staged_views(1, 5)
    for new, old in zip(ls._tensors(), held):
        new.copy_(old)
    ls.recurrent_state[1].zero_()
    expect = held[1][1].clone()
    held[1][1].zero_()
    _rewind_layers([ls], 0, 4, 2, 0, (1, 5, 1), 0, 1)
    torch.cuda.synchronize(device)
    assert torch.equal(ls.recurrent_state[1], expect) and not held[1][1].any()


def _conv_window_reference(state_head: torch.Tensor, x: torch.Tensor, k: int):
    # Window (last k inputs) after consuming x: concatenation of the old window and the new inputs
    seq = torch.cat([state_head[..., :k], x], dim = -1)
    return seq[..., -k:]


@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("seqlen, max_history", [(2, 1), (4, 3), (8, 7), (3, 7)])
@torch.inference_mode()
def test_conv_history_then_rewind(device, k, seqlen, max_history):
    # Draft/verify: an update over seqlen tokens with history, then a rewind of r rejected tokens, leaves the
    # window (head k columns) a non-history update over the first seqlen - r tokens would leave
    bsz, dim = 2, 320
    slots = torch.tensor([1, 0], dtype = torch.int32, device = device)
    w = torch.randn((dim, k), device = device).bfloat16()
    x = torch.randn((bsz, dim, seqlen), device = device).bfloat16()
    state0 = torch.randn((2, dim, k + max_history), device = device).bfloat16()
    for r in range(0, min(seqlen, max_history) + 1):
        state = state0.clone()
        out = torch.empty((bsz, seqlen, dim), dtype = torch.bfloat16, device = device)
        ext.cuda_causal_conv1d_update(x, state, slots, w, None, out, True, True)
        jobs = []
        for b in range(bsz):
            s = int(slots[b])
            p = state.shape[-1] - r
            jobs.append(_conv_job(state, s, p - k, 0, k))
        ext.batched_conv_rewind(jobs, device.index)
        torch.cuda.synchronize(device)
        for b in range(bsz):
            s = int(slots[b])
            ref = _conv_window_reference(state0[s], x[b, :, : seqlen - r], k)
            assert torch.equal(state[s, :, :k], ref), (r, b)
