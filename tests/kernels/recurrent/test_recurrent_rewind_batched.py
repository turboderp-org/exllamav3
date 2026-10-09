"""
Batched recurrent-layer rewind (speculative decoding draft rejection), against plain torch references:

ext.batched_conv_rewind(jobs, device_index): for each ConvRewindJob(src, dst, dim, cdim, stride), for every channel
d < dim, dst[d * stride + k] <- src[d * stride + k] for k < cdim, bf16, with the cdim source elements read before
any is written so overlapping windows (src - dst < cdim elements) copy like a memmove. cdim > CONV1D_MAX_K (16) is
rejected. Nothing outside the dst windows is written. Any number of jobs (the host splits them into launches of
64), an empty list is a no-op, jobs of one call may target different tensors with different sizes.

GDNLayerState: rewind_conv_job must describe the conv-ring shift (window <- last cdim columns before the rejected
tokens), and a conv update with history followed by a conv rewind of r tokens must leave the window a plain update
over the first seqlen - r tokens would leave. replay_job over the layer's staged inputs (the views a speculative
pass writes into) must rebuild, through ext.batched_scan_replay, the state a pass over the accepted prefix would
have produced from the base row, bit for bit, leaving the base row untouched.
"""


import types

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

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


# GDNLayerState: the conv job against a torch reference, the replay against a kernel pass over the prefix

def _layer_state(device, num_slots, max_history, fdim, k, nv, dk, dv, nk = None):
    from exllamav3.modules.gated_delta_net import GDNLayerState
    module = types.SimpleNamespace(
        fdim_qkv = fdim,
        conv_kernel_size = k,
        num_k_heads = nk or nv,
        num_v_heads = nv,
        k_head_dim = dk,
        v_head_dim = dv,
    )
    ls = GDNLayerState(module, num_slots, max_history, cache_id = 0)
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
def test_gdn_layer_state_conv_job_matches_reference(device, max_history):
    num_slots = 3
    for slot in range(num_slots):
        for last_history in range(max_history + 1):
            for num_tokens in range(last_history + 1):
                a = _layer_state(device, num_slots, max_history, 384, 4, 4, 32, 64)
                ref = _layer_state(device, num_slots, max_history, 384, 4, 4, 32, 64)
                ref.conv_state.copy_(a.conv_state)
                cj = a.rewind_conv_job(slot, last_history, num_tokens)
                assert (cj is None) == (last_history == 0)
                if cj is not None:
                    ext.batched_conv_rewind([cj], device.index)
                torch.cuda.synchronize(device)
                _conv_rewind_reference(ref, slot, last_history, num_tokens)
                assert torch.equal(a.conv_state, ref.conv_state), (slot, last_history, num_tokens)
    with pytest.raises(AssertionError):
        _layer_state(device, 1, 3, 384, 4, 4, 32, 64).rewind_conv_job(0, 3, 4)


@pytest.mark.parametrize("max_history", [3, 64])
@pytest.mark.parametrize("bsz", [1, 3])
@torch.inference_mode()
def test_gdn_layer_state_replay(device, max_history, bsz):
    # A speculative pass of max_history + 1 tokens writes its scan inputs into the staged views; every batch row
    # and every prefix then replays to the state a prefix-length pass from the same base row produces
    nk, nv, dk, dv, fdim = 2, 4, 64, 64, 2 * 2 * 64 + 4 * 64
    seqlen = max_history + 1
    ls = _layer_state(device, bsz, max_history, fdim, 4, nv, dk, dv, nk = nk)
    conv_out, beta, g = ls.staged_views(bsz, seqlen)
    conv_out.copy_((torch.randn(conv_out.shape, device = device) * 0.25).bfloat16())
    beta.copy_(torch.sigmoid(torch.randn(beta.shape, device = device)).bfloat16())
    g.copy_(torch.randn(g.shape, device = device) * 0.5 - 1.0)
    pool0 = ls.recurrent_state.clone()
    for row in range(bsz):
        for parity in (0, 1):
            base, scratch = 2 * row + parity, 2 * row + 1 - parity
            for prefix in sorted({1, seqlen // 2, seqlen}):
                expect = pool0.clone()
                out = torch.empty((1, prefix, nv, dv), dtype = torch.bfloat16, device = device)
                ext.cuda_recurrent_gated_delta_rule(
                    conv_out[row : row + 1, :prefix].contiguous(), g[row : row + 1, :prefix].contiguous(),
                    beta[row : row + 1, :prefix].contiguous(), expect, out, nk, nv, dk, dv,
                    torch.tensor([scratch], dtype = torch.int32, device = device), True,
                    torch.tensor([base], dtype = torch.int32, device = device))
                ls.recurrent_state.copy_(pool0)
                ext.batched_scan_replay([ls.replay_job(row, prefix, (bsz, seqlen), base, scratch)],
                                        device.index, *ls.scan_geometry())
                torch.cuda.synchronize(device)
                assert torch.equal(ls.recurrent_state, expect), (row, parity, prefix)
                assert torch.equal(ls.recurrent_state[base], pool0[base])
    with pytest.raises(AssertionError):
        ls.replay_job(0, 1, (bsz, seqlen - 1), 0, 1)   # staged shape mismatch
    with pytest.raises(AssertionError):
        ls.staged_views(bsz, seqlen + 1)               # longer than max_history + 1


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
