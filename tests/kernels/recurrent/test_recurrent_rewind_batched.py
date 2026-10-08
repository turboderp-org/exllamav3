"""
Batched recurrent-state rewind kernels (speculative decoding draft rejection), against plain torch copies:

ext.batched_state_rewind(jobs, device_index): for each StateRewindJob(src, dst, n), dst[0:n] <- src[0:n] as fp32
(n a multiple of 4, rejected otherwise; src/dst never overlap). Nothing outside [dst, dst + n) is written.

ext.batched_conv_rewind(jobs, device_index): for each ConvRewindJob(src, dst, dim, cdim, stride), for every channel
d < dim, dst[d * stride + k] <- src[d * stride + k] for k < cdim, bf16, with the cdim source elements read before
any is written so overlapping windows (src - dst < cdim elements) copy like a memmove. cdim > CONV1D_MAX_K (16) is
rejected. Nothing outside the dst windows is written.

Both process any number of jobs (the host splits them into launches of 64), and an empty list is a no-op. Jobs
of one call may target different tensors with different sizes. Copies are exact.

The GDNLayerState job builders (rewind_conv_job / rewind_state_job) must describe the same copy as the reference
GDNLayerState.rewind (torch views), and a conv update with history followed by a conv rewind of r tokens must
leave the window a plain update over the first seqlen - r tokens would leave.
"""

import types

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

CONV1D_MAX_K = 16


def _state_job(t: torch.Tensor, src_idx: tuple, dst_idx: tuple, n: int):
    return ext.StateRewindJob(t[src_idx].data_ptr(), t[dst_idx].data_ptr(), n)


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


# batched_state_rewind

@pytest.mark.parametrize("num_layers", [1, 3, 64, 65, 150])
@pytest.mark.parametrize("state_shape", [(4, 8, 8), (16, 128, 128), (3, 4, 12)])
@torch.inference_mode()
def test_batched_state_rewind(device, num_layers, state_shape):
    # One recurrent_state tensor per layer, (slots, max_history + 1, Hv, Dk, Dv); each job copies one history
    # entry of one slot into entry 0 of that slot. Every other element of every tensor is unchanged
    num_slots, history = 3, 5
    gen = torch.Generator(device = "cpu").manual_seed(num_layers)
    states = [torch.randn((num_slots, history + 1, *state_shape), generator = gen).to(device) for _ in range(num_layers)]
    expected = [s.clone() for s in states]
    n = states[0][0, 0].numel()

    jobs = []
    for i, (s, e) in enumerate(zip(states, expected)):
        slot = i % num_slots
        src = 1 + (i % history)
        jobs.append(_state_job(s, (slot, src), (slot, 0), n))
        e[slot, 0] = e[slot, src]

    ext.batched_state_rewind(jobs, device.index)
    torch.cuda.synchronize(device)
    for i, (s, e) in enumerate(zip(states, expected)):
        assert torch.equal(s, e), f"layer {i}"


@torch.inference_mode()
def test_batched_state_rewind_mixed_sizes(device):
    # Jobs of different lengths in one launch: the grid covers the longest, shorter jobs stop at their own end.
    # Each destination sits between sentinel guard regions that must not be written
    guard = 64
    sizes = [4, 8, 1024, 12, 4096 + 4, 32768, 4]
    bufs = []
    jobs = []
    for i, n in enumerate(sizes):
        src = torch.randn(n, device = device) + i
        dst = torch.full((guard + n + guard,), -777.0, device = device)
        bufs.append((src, dst, n))
        jobs.append(ext.StateRewindJob(src.data_ptr(), dst[guard:].data_ptr(), n))
    ext.batched_state_rewind(jobs, device.index)
    torch.cuda.synchronize(device)
    for src, dst, n in bufs:
        assert torch.equal(dst[guard : guard + n], src)
        assert (dst[:guard] == -777.0).all() and (dst[guard + n:] == -777.0).all()


@torch.inference_mode()
def test_batched_state_rewind_bit_exact(device):
    # A copy, not arithmetic: NaN payloads, infinities, -0.0 and denormals survive bit for bit
    n = 1024
    src_bits = torch.randint(-2**31, 2**31 - 1, (n,), dtype = torch.int32, device = device)
    src_bits[:4] = torch.tensor([0x7fc00001, 0x7f800000, -2**31, 1], dtype = torch.int32)  # NaN, inf, -0.0, denormal
    dst = torch.zeros(n, dtype = torch.int32, device = device)
    ext.batched_state_rewind([ext.StateRewindJob(src_bits.data_ptr(), dst.data_ptr(), n)], device.index)
    torch.cuda.synchronize(device)
    assert torch.equal(dst, src_bits)


@torch.inference_mode()
def test_batched_state_rewind_empty(device):
    ext.batched_state_rewind([], device.index)
    ext.batched_conv_rewind([], device.index)


@torch.inference_mode()
def test_batched_state_rewind_rejects_unaligned_count(device):
    src = torch.randn(16, device = device)
    dst = torch.zeros(16, device = device)
    with pytest.raises(RuntimeError, match = "multiple of 4"):
        ext.batched_state_rewind([ext.StateRewindJob(src.data_ptr(), dst.data_ptr(), 6)], device.index)
    torch.cuda.synchronize(device)
    assert (dst == 0).all()


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


# Job builders of GDNLayerState against its torch rewind()

def _layer_state(device, num_slots, max_history, fdim, k, nv, dk, dv):
    from exllamav3.modules.gated_delta_net import GDNLayerState
    module = types.SimpleNamespace(
        fdim_qkv = fdim,
        conv_kernel_size = k,
        num_v_heads = nv,
        k_head_dim = dk,
        v_head_dim = dv,
    )
    ls = GDNLayerState(module, num_slots, max_history, cache_id = 0)
    ls.alloc(device)
    ls.conv_state.copy_(torch.randn(ls.conv_state.shape, device = device))
    ls.recurrent_state.normal_()
    return ls


@pytest.mark.parametrize("max_history", [1, 3, 7])
@torch.inference_mode()
def test_gdn_layer_state_jobs_match_rewind(device, max_history):
    num_slots = 3
    for slot in range(num_slots):
        for last_history in range(max_history + 1):
            for num_tokens in range(last_history + 1):
                a = _layer_state(device, num_slots, max_history, 384, 4, 4, 32, 64)
                b = types.SimpleNamespace(
                    conv_state = a.conv_state.clone(),
                    recurrent_state = a.recurrent_state.clone(),
                )
                cj = a.rewind_conv_job(slot, last_history, num_tokens)
                sj = a.rewind_state_job(slot, last_history, num_tokens)
                if cj is not None:
                    ext.batched_conv_rewind([cj], device.index)
                if sj is not None:
                    ext.batched_state_rewind([sj], device.index)
                torch.cuda.synchronize(device)
                ref = _layer_state(device, num_slots, max_history, 384, 4, 4, 32, 64)
                ref.conv_state.copy_(b.conv_state)
                ref.recurrent_state.copy_(b.recurrent_state)
                ref.rewind(slot, last_history, num_tokens)
                assert torch.equal(a.conv_state, ref.conv_state), (slot, last_history, num_tokens)
                assert torch.equal(a.recurrent_state, ref.recurrent_state), (slot, last_history, num_tokens)


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
