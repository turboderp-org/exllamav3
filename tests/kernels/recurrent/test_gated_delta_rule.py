"""
Gated delta rule kernels against a step-by-step fp32 torch recurrence (l2-normalized q / k, exp gates, GQA over
v heads): the CUDA recurrent kernel (ext.cuda_recurrent_gated_delta_rule over a row-indexed state pool; a
speculative pass -- history -- reads each row's initial state from slots_in, writes slots and leaves slots_in
untouched) and the vendored fla chunked prefill kernel (exllamav3.vendor.fla.chunk_gated_delta_rule), plus
bit-reproducibility of the CUDA kernel and the rewind replay (ext.batched_scan_replay), which must reproduce the
kernel's own state after the accepted prefix of a speculative pass bit for bit.
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.gated_delta_net_fn.gated_delta_rule import (
    gated_delta_rule_fn,
    pack_state_row,
    unpack_state_row,
)
from exllamav3.util.hadamard import get_hadamard
from testlib.gdn import pool_from_state, state_from_pool


def _l2norm(x: torch.Tensor, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(dim = -1, keepdim = True) + eps)


def _rotate32(x: torch.Tensor) -> torch.Tensor:
    # The Sylvester H32 per 32-wide k block, normalized (H^2 = I): the basis the int8 pool is
    # stored in. get_hadamard(32) is the recursive Sylvester construction, H[i,j] = (-1)^popcount(i & j)
    h = (get_hadamard(32).float() * 32 ** -0.5).to(device = x.device)
    xb = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    return (xb @ h).reshape(*x.shape)


def _rotate_state(state: torch.Tensor) -> torch.Tensor:
    # _rotate32 over the k dim (dim -2) of a state (..., k, v): move k to the end, rotate, move back
    return _rotate32(state.movedim(-2, -1)).movedim(-2, -1)


def _torch_gated_delta_rule(
    mixed_qkv: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    slots: torch.Tensor | None,
    history: bool,
    num_k_heads: int,
    num_v_heads: int,
    k_head_dim: int,
    v_head_dim: int,
    rotate: bool = False,
):
    bsz, seqlen, _ = mixed_qkv.shape
    group = num_v_heads // num_k_heads
    k_dim = num_k_heads * k_head_dim
    v_dim = num_v_heads * v_head_dim
    scale = k_head_dim ** -0.5

    q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
    q = _l2norm(q.float().view(bsz, seqlen, num_k_heads, k_head_dim))
    k = _l2norm(k.float().view(bsz, seqlen, num_k_heads, k_head_dim))
    if rotate:
        # The int8 pool is stored in the rotated basis: q / k enter the scan through the same basis
        q = _rotate32(q)
        k = _rotate32(k)
    v = v.float().view(bsz, seqlen, num_v_heads, v_head_dim)
    g = g.float().exp()
    beta = beta.float()

    out = torch.empty((bsz, seqlen, num_v_heads, v_head_dim), dtype = torch.bfloat16, device = mixed_qkv.device)
    state_out = recurrent_state.clone()

    for bi in range(bsz):
        slot = int(slots[bi].item()) if slots is not None else bi
        # A speculative pass starts from the base row (slots_in = slot + 1 here) and leaves it as is
        state = state_out[slot + 1 if history else slot, 0].clone()

        for t in range(seqlen):
            next_state = torch.empty_like(state)

            for vh in range(num_v_heads):
                kh = vh // group
                kv_mem = (state[vh] * k[bi, t, kh].unsqueeze(-1)).sum(dim = -2)
                v_t = v[bi, t, vh] - kv_mem * g[bi, t, vh]
                next_state[vh] = state[vh] * g[bi, t, vh] + \
                    k[bi, t, kh].unsqueeze(-1) * v_t.unsqueeze(-2) * beta[bi, t, vh]
                out[bi, t, vh] = ((next_state[vh] * q[bi, t, kh].unsqueeze(-1)).sum(dim = -2) * scale).bfloat16()

            state = next_state

        state_out[slot, 0].copy_(state)

    return out, state_out


def _run_cuda_gated_delta_rule(
    mixed_qkv: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    slots: torch.Tensor | None,
    history: bool,
    num_k_heads: int,
    num_v_heads: int,
    k_head_dim: int,
    v_head_dim: int,
    recurrent_scales: torch.Tensor | None = None,
):
    out = torch.empty(
        (mixed_qkv.shape[0], mixed_qkv.shape[1], num_v_heads, v_head_dim),
        dtype = torch.bfloat16,
        device = mixed_qkv.device,
    )
    state = recurrent_state.clone()
    slots_in = slots + 1 if history else None
    ext.cuda_recurrent_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        state,
        out,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        slots,
        history,
        slots_in,
        recurrent_scales,
    )
    torch.cuda.synchronize()
    return out, state


def _run_chunk_gated_delta_rule(
    mixed_qkv: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    k_head_dim: int,
    v_head_dim: int,
):
    # The vendored kernels query the devices and Triton at import
    from exllamav3.vendor.fla import chunk_gated_delta_rule

    bsz, _, _ = mixed_qkv.shape
    k_dim = num_k_heads * k_head_dim
    v_dim = num_v_heads * v_head_dim

    q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
    q = q.view(bsz, -1, num_k_heads, k_head_dim)
    k = k.view(bsz, -1, num_k_heads, k_head_dim)
    v = v.view(bsz, -1, num_v_heads, v_head_dim)

    out, final_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g = g,
        beta = beta,
        initial_state = recurrent_state[:, 0],
        output_final_state = True,
        use_qk_l2norm_in_kernel = True,
    )
    torch.cuda.synchronize()
    return out.to(torch.bfloat16), final_state


RECURRENT_SHAPES = [
    (1, 1, 1, 1, 64, 64),
    (2, 5, 2, 4, 64, 64),
    (3, 7, 2, 4, 128, 128),
    (1, 15, 16, 32, 128, 128),
    (2, 17, 16, 32, 128, 128),
    (1, 128, 4, 8, 256, 256),
    (2, 1024, 4, 8, 128, 128),
    (1, 2047, 2, 8, 128, 128),
    (1, 2049, 2, 8, 128, 128),
]


def _recurrent_param(shape, history):
    return pytest.param(history, *shape)


@pytest.mark.parametrize("state_dtype", [torch.float, torch.half], ids = ["fp32_state", "fp16_state"])
@pytest.mark.parametrize(
    "history,bsz,seqlen,num_k_heads,num_v_heads,k_head_dim,v_head_dim",
    [_recurrent_param(shape, history) for history in (False, True) for shape in RECURRENT_SHAPES],
)
@torch.inference_mode()
def test_cuda_recurrent_gated_delta_rule_matches_torch(
    device,
    state_dtype,
    history,
    bsz,
    seqlen,
    num_k_heads,
    num_v_heads,
    k_head_dim,
    v_head_dim,
):
    torch.manual_seed(1234)

    qkv_dim = 2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim
    # Rows: slot s writes row 2s + 1 and (history) reads row 2s + 2, so neither overlaps another row
    num_slots = 2 * bsz + 2

    mixed_qkv = (torch.randn((bsz, seqlen, qkv_dim), dtype = torch.float, device = device) * 0.25).bfloat16()
    g = torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device) * 0.5 - 1.0
    beta = torch.sigmoid(torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device)).bfloat16()
    # fp16-representable initial values, so the fp32 reference and an fp16 pool start from the same state
    recurrent_state = (torch.randn(
        (num_slots, 1, num_v_heads, k_head_dim, v_head_dim),
        dtype = torch.float,
        device = device,
    ) * 0.05).half().float()
    pool_in = recurrent_state.to(state_dtype)
    slots = 2 * torch.arange(bsz, dtype = torch.int32, device = device) + 1

    ref_out, ref_state = _torch_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        recurrent_state,
        slots,
        history,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
    )
    cuda_out, cuda_state = _run_cuda_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        pool_in,
        slots,
        history,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
    )

    assert cuda_state.dtype == state_dtype
    cuda_state = cuda_state.float()
    torch.testing.assert_close(cuda_out, ref_out, rtol = 5e-2, atol = 5e-2)
    torch.testing.assert_close(cuda_state, ref_state, rtol = 5e-2, atol = 5e-2)
    if state_dtype == torch.half:
        # One rounding of the final state: within fp16 precision of the fp32 scan
        torch.testing.assert_close(cuda_state[slots.long()], ref_state[slots.long()], rtol = 2e-3, atol = 1e-4)
    if history:
        # The base rows are left exactly as they were
        assert torch.equal(cuda_state[slots.long() + 1], recurrent_state[slots.long() + 1])
    else:
        assert torch.equal(cuda_state[slots.long() + 1], recurrent_state[slots.long() + 1])   # untouched neighbours


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize(
    "bsz,seqlen,num_k_heads,num_v_heads,k_head_dim,v_head_dim",
    [
        (1, 17, 2, 4, 128, 128),
        (2, 9, 4, 8, 128, 128),
    ],
)
@torch.inference_mode()
def test_cuda_recurrent_gated_delta_rule_int8_matches_torch(
    device,
    history,
    bsz,
    seqlen,
    num_k_heads,
    num_v_heads,
    k_head_dim,
    v_head_dim,
):
    # The int8 pool rounds once per pass and stores the row rotated: out and the written pool rows
    # agree with the fp32 reference to int8 precision (5e-2 on out, 1e-2 on the pool rows). The same
    # pass over an fp16 pool holding that dequantized state rotated into the pool's basis pins that
    # the rotation is a change of basis of the scan and not of its result: the two agree within a
    # bf16 output quantum or two (6e-5 on these shapes), while a reference in the model basis is off
    # by ~1.2e-2, so that comparison is held at 2e-3
    torch.manual_seed(4321)
    qkv_dim = 2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim
    num_slots = 2 * bsz + 2
    mixed_qkv = (torch.randn((bsz, seqlen, qkv_dim), dtype = torch.float, device = device) * 0.25).bfloat16()
    g = torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device) * 0.5 - 1.0
    beta = torch.sigmoid(torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device)).bfloat16()
    state0 = torch.randn((num_slots, 1, num_v_heads, k_head_dim, v_head_dim), dtype = torch.float, device = device) * 0.05
    pool, scales = pool_from_state(state0)
    pool0, scales0 = pool.clone(), scales.clone()
    slots = 2 * torch.arange(bsz, dtype = torch.int32, device = device) + 1

    ref_out, ref_state = _torch_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        state_from_pool(pool, scales),
        slots,
        history,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        rotate = True,
    )
    cuda_out, cuda_state = _run_cuda_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        pool,
        slots,
        history,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        recurrent_scales = scales,
    )
    fp16_out, _ = _run_cuda_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        _rotate_state(state_from_pool(pool, scales0)).half(),
        slots,
        history,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
    )

    deq = state_from_pool(cuda_state, scales)
    torch.testing.assert_close(cuda_out, ref_out, rtol = 5e-2, atol = 5e-2)
    torch.testing.assert_close(cuda_out, fp16_out, rtol = 0.0, atol = 2e-3)
    # The pool rows the pass wrote: one quantization step from the reference
    torch.testing.assert_close(deq[slots.long()], ref_state[slots.long()], rtol = 1e-2, atol = 1e-2)
    # Every other row (values and scales alike) is untouched and bit-equal
    mask = torch.ones(num_slots, dtype = torch.bool, device = device)
    mask[slots.long()] = False
    assert torch.equal(cuda_state[mask], pool0[mask])
    assert torch.equal(scales[mask], scales0[mask])


@torch.inference_mode()
def test_int8_state_pack_unpack_round_trip(device):
    # The converter's pack/unpack round trip is one quantization step; the rotation is its own
    # inverse (an already-rotated state round-trips too) and a zero state packs to zero values and scales
    h, k, v = 4, 128, 128
    torch.manual_seed(31)
    state = torch.randn((h, k, v), dtype = torch.float, device = device) * 0.05
    values = torch.empty((h, k, v), dtype = torch.int8, device = device)
    scales = torch.empty((h, v, k // 32), dtype = torch.float16, device = device)

    pack_state_row(state, values, scales)
    back = unpack_state_row(values, scales)
    # atol = the max slice absmax of the rotated state / 127: twice the value-rounding bound
    # (amax / 254), with room for the fp16 scale rounding and the fp32 rotation
    amax = _rotate_state(state).view(h, k // 32, 32, v).abs().amax(dim = 2).max().item()
    torch.testing.assert_close(back, state, rtol = 0.0, atol = amax / 127.0)

    pack_state_row(_rotate_state(state), values, scales)
    torch.testing.assert_close(unpack_state_row(values, scales), _rotate_state(state),
                              rtol = 0.0, atol = amax / 127.0)

    pack_state_row(torch.zeros_like(state), values, scales)
    assert torch.equal(values, torch.zeros_like(values))
    assert torch.equal(scales, torch.zeros_like(scales))
    assert torch.equal(unpack_state_row(values, scales), torch.zeros_like(state))


@pytest.mark.parametrize(
    "bsz,seqlen,num_k_heads,num_v_heads,k_head_dim,v_head_dim",
    [
        (3, 9, 2, 4, 64, 64),        # generic kernel
        (2, 17, 4, 8, 128, 128),     # 128 kernel
        (1, 65, 2, 8, 128, 128),     # 128 kernel, a long (suffix-match class) pass
        (2, 9, 2, 4, 256, 256),      # generic, wide
    ],
)
@pytest.mark.parametrize("layers", [1, 3, 50])
@pytest.mark.parametrize("state_dtype", [torch.float, torch.int8], ids = ["fp32_state", "int8_state"])
@torch.inference_mode()
def test_scan_replay_matches_kernel_prefix(device, bsz, seqlen, num_k_heads, num_v_heads, k_head_dim, v_head_dim, layers, state_dtype):
    """A speculative pass advances scratch rows from base rows; batched_scan_replay over the first p staged
    tokens of one batch row must leave the scratch row exactly as a p-token pass from the same base row does,
    for every prefix, with the jobs of several layers (own pools and inputs) in one launch. 50 layers spans
    two launches (REPLAY_MAX_JOBS). The int8 pool replays through the char kernel: pass and replay
    quantize identically, so the bit-exactness carries over"""
    if state_dtype == torch.int8 and (k_head_dim != 128 or v_head_dim != 128):
        pytest.skip("int8 pools are 128x128 only")
    torch.manual_seed(7 + layers)
    qkv_dim = 2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim
    rr = bsz - 1   # the batch row replayed: the last one
    L = []
    for _ in range(layers):
        mixed_qkv = (torch.randn((bsz, seqlen, qkv_dim), dtype = torch.float, device = device) * 0.25).bfloat16()
        g = torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device) * 0.5 - 1.0
        beta = torch.sigmoid(torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device)).bfloat16()
        pool = torch.randn((2 * bsz, 1, num_v_heads, k_head_dim, v_head_dim), dtype = torch.float, device = device) * 0.05
        if state_dtype == torch.int8:
            pool, scales = pool_from_state(pool)
        else:
            scales = None
        L.append((mixed_qkv, g, beta, pool, scales))
    slots = 2 * torch.arange(bsz, dtype = torch.int32, device = device) + 1   # scratch rows
    slots_in = slots - 1                                                       # base rows
    for prefix in sorted({1, 2, seqlen // 2, seqlen - 1, seqlen}):
        if prefix < 1: continue
        expect = []
        for mixed_qkv, g, beta, pool, scales in L:
            ref = pool.clone()
            sref = scales.clone() if scales is not None else None
            out = torch.empty((bsz, prefix, num_v_heads, v_head_dim), dtype = torch.bfloat16, device = device)
            ext.cuda_recurrent_gated_delta_rule(mixed_qkv[:, :prefix].contiguous(), g[:, :prefix].contiguous(),
                                                beta[:, :prefix].contiguous(), ref, out, num_k_heads, num_v_heads,
                                                k_head_dim, v_head_dim, slots, True, slots_in, sref)
            expect.append((ref, sref))
        jobs = []
        pools = []
        scales2 = []
        for mixed_qkv, g, beta, pool, scales in L:
            p2 = pool.clone(); pools.append(p2)
            s2 = scales.clone() if scales is not None else None
            scales2.append(s2)
            rb = p2.stride(0) * p2.element_size()
            # Base pointers of the s2 rows; 0 for the scale-less (fp32) pool
            sb = s2.data_ptr() if s2 is not None else 0
            srb = s2.stride(0) * s2.element_size() if s2 is not None else 0
            jobs.append(ext.ScanReplayJob(
                mixed_qkv[rr].data_ptr(), g[rr].data_ptr(), beta[rr].data_ptr(),
                p2.data_ptr() + int(slots_in[rr]) * rb, p2.data_ptr() + int(slots[rr]) * rb, 0, prefix, 1,
                sb + int(slots_in[rr]) * srb, sb + int(slots[rr]) * srb))
        # The pool's dispatch index: 0 fp32, 2 int8 (see GDNLayerState.scan_geometry)
        ext.batched_scan_replay(jobs, torch.device(device).index, 0, num_k_heads, num_v_heads, k_head_dim, v_head_dim,
                               2 if state_dtype == torch.int8 else 0)
        torch.cuda.synchronize(device)
        for p2, s2, (ref, sref), (_, _, _, pool, scales) in zip(pools, scales2, expect, L):
            assert torch.equal(p2[int(slots[rr])], ref[int(slots[rr])]), f"prefix {prefix}: replayed row differs"
            # Nothing else in the pool moved
            mask = torch.ones(2 * bsz, dtype = torch.bool, device = device); mask[int(slots[rr])] = False
            assert torch.equal(p2[mask], pool[mask])
            if s2 is not None:
                assert torch.equal(s2[int(slots[rr])], sref[int(slots[rr])]), f"prefix {prefix}: replayed scales differ"
                assert torch.equal(s2[mask], scales[mask])


@pytest.mark.parametrize(
    "bsz,seqlen,num_k_heads,num_v_heads,k_head_dim,v_head_dim",
    [
        (1, 64, 1, 1, 64, 64),
        (2, 65, 2, 4, 64, 64),
        (3, 127, 2, 4, 128, 128),
    ],
)
@pytest.mark.parametrize("state_format", ["fp32", "int8"], ids = ["fp32_pool", "int8_pool"])
@torch.inference_mode()
def test_chunk_gated_delta_rule_matches_torch(
    device,
    bsz,
    seqlen,
    num_k_heads,
    num_v_heads,
    k_head_dim,
    v_head_dim,
    state_format,
):
    torch.manual_seed(5678)

    qkv_dim = 2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim
    mixed_qkv = (torch.randn((bsz, seqlen, qkv_dim), dtype = torch.float, device = device) * 0.25).bfloat16()
    g = torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device) * 0.5 - 1.0
    beta = torch.sigmoid(torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device)).bfloat16()
    recurrent_state = torch.randn(
        (bsz, 1, num_v_heads, k_head_dim, v_head_dim),
        dtype = torch.float,
        device = device,
    ) * 0.05

    if state_format == "int8":
        # The chunked path with a packed int8 pool (128x128 only) must agree with the fp32 reference
        # started from the same (quantized) state; a wrong-but-orthogonal converter matrix fails it
        if k_head_dim != 128 or v_head_dim != 128:
            pytest.skip("int8 pools are 128x128 only")
        pool, scales = pool_from_state(recurrent_state)
        init_state = _rotate_state(state_from_pool(pool, scales))
    else:
        pool, scales = None, None
        init_state = recurrent_state

    ref_out, ref_state = _torch_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        init_state,
        None,
        False,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
    )

    if state_format == "int8":
        out = gated_delta_rule_fn(
            mixed_qkv,
            beta,
            g,
            pool,
            None,
            False,
            True,
            num_k_heads,
            num_v_heads,
            num_k_heads * k_head_dim,
            num_v_heads * v_head_dim,
            k_head_dim,
            v_head_dim,
            recurrent_scales = scales,
        )
        torch.testing.assert_close(out, ref_out, rtol = 5e-2, atol = 5e-2)
        # The pass wrote the pool in place: decode the new rows back to the original basis, one
        # quantization step from the reference
        torch.testing.assert_close(_rotate_state(state_from_pool(pool, scales)), ref_state,
                                  rtol = 1e-2, atol = 1e-2)
        return

    chunk_out, chunk_state = _run_chunk_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        recurrent_state,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
    )
    cuda_out, cuda_state = _run_cuda_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        recurrent_state,
        None,
        False,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
    )

    torch.testing.assert_close(chunk_out, ref_out, rtol = 5e-2, atol = 5e-2)
    torch.testing.assert_close(chunk_state, ref_state[:, 0], rtol = 5e-2, atol = 5e-2)
    torch.testing.assert_close(chunk_out, cuda_out, rtol = 5e-2, atol = 5e-2)
    torch.testing.assert_close(chunk_state, cuda_state[:, 0], rtol = 5e-2, atol = 5e-2)


@pytest.mark.parametrize(
    "bsz,seqlen,num_k_heads,num_v_heads,k_head_dim,v_head_dim",
    [
        (2, 9, 2, 4, 64, 64),        # generic kernel
        (2, 9, 4, 8, 128, 128),      # 128 kernel
        (1, 9, 2, 4, 256, 256),      # generic, wide
    ],
)
@torch.inference_mode()
def test_cuda_recurrent_gated_delta_rule_is_bit_reproducible(device, bsz, seqlen, num_k_heads, num_v_heads, k_head_dim, v_head_dim):
    # The per-k-slice partial dot products used to be combined with shared-memory float atomics
    # in arrival order; they are now reduced in a fixed order, so identical inputs give identical
    # outputs and states
    torch.manual_seed(99)
    qkv_dim = 2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim
    mixed_qkv = (torch.randn((bsz, seqlen, qkv_dim), dtype = torch.float, device = device) * 0.25).bfloat16()
    g = torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device) * 0.5 - 1.0
    beta = torch.sigmoid(torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device)).bfloat16()
    recurrent_state = torch.randn((bsz + 1, 1, num_v_heads, k_head_dim, v_head_dim), dtype = torch.float, device = device) * 0.05
    slots = torch.arange(bsz, dtype = torch.int32, device = device) + 1
    runs = [_run_cuda_gated_delta_rule(mixed_qkv, g, beta, recurrent_state, slots, False,
                                       num_k_heads, num_v_heads, k_head_dim, v_head_dim) for _ in range(4)]
    for out, state in runs[1:]:
        assert torch.equal(out, runs[0][0]) and torch.equal(state, runs[0][1])


# Zero-size inputs: an empty batch, sequence or v-head axis (or v_head_dim = 0) takes no step: the state (all
# slots and history entries) and the empty output are untouched. num_k_heads sets the GQA ratio and must be
# positive; k_head_dim = 0 would l2-normalize empty q / k vectors and is rejected with the other k_head_dim rules

EMPTY_RULE_CASES = [
    # bsz, seqlen, nk, nv, dk, dv, history, slots, channelwise, error
    (0, 1, 2, 4, 32, 32, False, False, False, None),
    (0, 3, 2, 4, 32, 32, True, True, False, None),
    (2, 0, 2, 4, 32, 32, False, False, False, None),
    (2, 0, 2, 4, 32, 32, True, True, False, None),
    (1, 0, 2, 4, 128, 128, False, True, False, None),          # the 128x128 kernel (v_split 4 at bsz 1)
    (1, 0, 1, 2, 128, 128, True, True, True, None),            # channelwise (KDA) decay
    (2, 1, 2, 0, 32, 32, False, False, False, None),
    (2, 1, 2, 4, 32, 0, False, False, False, None),
    (2, 1, 0, 0, 32, 32, False, False, False, "num_k_heads must be positive"),
    (2, 1, 2, 4, 0, 32, False, False, False, "k_head_dim must be a positive multiple of 32"),
]


@pytest.mark.parametrize("bsz, seqlen, nk, nv, dk, dv, history, use_slots, channelwise, error", EMPTY_RULE_CASES)
@torch.inference_mode()
def test_empty(device, bsz, seqlen, nk, nv, dk, dv, history, use_slots, channelwise, error):
    mixed_qkv = torch.randn(bsz, seqlen, 2 * nk * dk + nv * dv, device = device).bfloat16()
    g = -torch.rand((bsz, seqlen, nv, dk) if channelwise else (bsz, seqlen, nv), device = device)
    beta = torch.rand(bsz, seqlen, nv, device = device).bfloat16()
    state = torch.randn(2 * max(bsz, 1) + 1, 1, nv, dk, dv, device = device)
    state0 = state.clone()
    out = torch.full((bsz, seqlen, nv, dv), 777.0, dtype = torch.bfloat16, device = device)
    slots = torch.arange(bsz, dtype = torch.int32, device = device) if use_slots else None
    slots_in = slots + 1 if (use_slots and history) else None
    call = lambda: ext.cuda_recurrent_gated_delta_rule(mixed_qkv, g, beta, state, out, nk, nv, dk, dv, slots, history, slots_in)
    if error:
        with pytest.raises(RuntimeError, match = error):
            call()
    else:
        call()
    torch.cuda.synchronize(device)
    assert torch.equal(state, state0), "state modified"
    assert (torch.ones(4, device = device) + 1).sum().item() == 8.0, "a later torch op failed after the empty call"


@pytest.mark.parametrize("which", ["mixed_qkv", "g", "beta", "state", "out"])
@torch.inference_mode()
def test_rejects_non_contiguous(device, which):
    """The kernel indexes every tensor densely: a strided view (here the token-major view of a channel-major
    buffer, as the conv emits it) would be misread rather than rejected"""
    bsz, seqlen, nk, nv, dk, dv = 1, 3, 2, 4, 32, 32
    t = dict(
        mixed_qkv = torch.randn(bsz, seqlen, 2 * nk * dk + nv * dv, device = device).bfloat16(),
        g = -torch.rand(bsz, seqlen, nv, device = device),
        beta = torch.rand(bsz, seqlen, nv, device = device).bfloat16(),
        state = torch.zeros(bsz, 1, nv, dk, dv, device = device),
        out = torch.empty(bsz, seqlen, nv, dv, dtype = torch.bfloat16, device = device),
    )
    x = t[which]
    t[which] = x.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not t[which].is_contiguous()
    with pytest.raises(RuntimeError, match = "must be contiguous"):
        ext.cuda_recurrent_gated_delta_rule(t["mixed_qkv"], t["g"], t["beta"], t["state"], t["out"], nk, nv, dk, dv,
                                            None, False)


@torch.inference_mode()
def test_int8_state_rejects_ineligible_calls(device):
    """The int8 dispatch is 128x128 GDN only: channelwise decay (KDA) does not commute with the rotation,
    and the generic (non-128) kernels have no char path. The scales tensor is int8-only."""
    bsz, seqlen, nk, nv, dk, dv = 1, 3, 2, 4, 128, 128
    mixed_qkv = torch.randn(bsz, seqlen, 2 * nk * dk + nv * dv, device = device).bfloat16()
    beta = torch.rand(bsz, seqlen, nv, device = device).bfloat16()
    out = torch.empty(bsz, seqlen, nv, dv, dtype = torch.bfloat16, device = device)
    pool = torch.randint(-127, 128, (2, 1, nv, dk, dv), dtype = torch.int8, device = device)
    scales = torch.rand((2, 1, nv, dv, dk // 32), device = device).half()

    # channelwise g (KDA) with an int8 state
    g_kda = -torch.rand(bsz, seqlen, nv, dk, device = device)
    with pytest.raises(RuntimeError, match = "not supported for channelwise"):
        ext.cuda_recurrent_gated_delta_rule(mixed_qkv, g_kda, beta, pool, out, nk, nv, dk, dv, None, False, None, scales)

    # int8 with non-128 head dims
    dk2, dv2 = 64, 64
    qkv2 = torch.randn(bsz, seqlen, 2 * nk * dk2 + nv * dv2, device = device).bfloat16()
    pool2 = torch.randint(-127, 128, (2, 1, nv, dk2, dv2), dtype = torch.int8, device = device)
    scales2 = torch.rand((2, 1, nv, dv2, dk2 // 32), device = device).half()
    out2 = torch.empty(bsz, seqlen, nv, dv2, dtype = torch.bfloat16, device = device)
    g2 = -torch.rand(bsz, seqlen, nv, device = device)
    with pytest.raises(RuntimeError, match = "requires 128x128"):
        ext.cuda_recurrent_gated_delta_rule(qkv2, g2, beta, pool2, out2, nk, nv, dk2, dv2, None, False, None, scales2)

    # an int8 pool without its per-slice scales
    g = -torch.rand(bsz, seqlen, nv, device = device)
    with pytest.raises(RuntimeError, match = "int8 recurrent state requires recurrent_scales"):
        ext.cuda_recurrent_gated_delta_rule(mixed_qkv, g, beta, pool, out, nk, nv, dk, dv, None, False, None, None)

    # a scales tensor with a non-int8 pool
    pool_fp16 = torch.zeros(2, 1, nv, dk, dv, dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "recurrent_scales is only used with the int8"):
        ext.cuda_recurrent_gated_delta_rule(mixed_qkv, g, beta, pool_fp16, out, nk, nv, dk, dv, None, False, None, scales)
