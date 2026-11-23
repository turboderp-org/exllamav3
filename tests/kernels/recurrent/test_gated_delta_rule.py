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
    _pack_bit_row,
    _pack_quantized_state_row,
    _unpack_bit_codes,
    _unpack_quantized_state_row,
    gated_delta_rule_fn,
)
from exllamav3.util.hadamard import get_hadamard
from testlib.gdn import _pack_row as reference_pack_row
from testlib.gdn import dequantize_pool, quantize_pool


def _l2norm(x: torch.Tensor, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(dim = -1, keepdim = True) + eps)


def _rotate32(x: torch.Tensor) -> torch.Tensor:
    hadamard = (get_hadamard(32).float() * 32 ** -0.5).to(device = x.device)
    slices = x.reshape(*x.shape[:-1], x.shape[-1] // 32, 32)
    return (slices @ hadamard).reshape(*x.shape)


def _rotate_state(state: torch.Tensor) -> torch.Tensor:
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
    rotate_qk: bool = False,
):
    bsz, seqlen, _ = mixed_qkv.shape
    group = num_v_heads // num_k_heads
    k_dim = num_k_heads * k_head_dim
    v_dim = num_v_heads * v_head_dim
    scale = k_head_dim ** -0.5

    q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
    q = _l2norm(q.float().view(bsz, seqlen, num_k_heads, k_head_dim))
    k = _l2norm(k.float().view(bsz, seqlen, num_k_heads, k_head_dim))
    if rotate_qk:
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
    recurrent_bits: int = 0,
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
        recurrent_bits,
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


def test_bit_codec_fixed_vector(device):
    # The -2 code at k = 3 crosses the first 16-bit boundary
    codes = torch.tensor(
        [[[1], [-1], [2], [-2], [3]] + [[0]] * 11], dtype = torch.int16, device = device
    )
    values = torch.empty((1, 1, 10), dtype = torch.int8, device = device)
    _pack_bit_row(codes, values, 5)
    expected_values = torch.tensor(
        [0xe1, 0x0b, 0x3f, 0, 0, 0, 0, 0, 0, 0], dtype = torch.uint8, device = device
    ).view(torch.int8).view(1, 1, 10)
    assert torch.equal(values, expected_values)

    units = torch.tensor([[[0x0be1, 0x003f, 0, 0, 0]]], dtype = torch.int32, device = device)
    expected_codes = codes.transpose(1, 2).to(torch.int64)
    assert torch.equal(_unpack_bit_codes(units, 5), expected_codes)


@pytest.mark.parametrize("bits", range(2, 13))
def test_bit_codec_matches_independent_reference(device, bits):
    torch.manual_seed(31)
    quantization_bound = (1 << (bits - 1)) - 1
    codes = torch.randint(
        -quantization_bound,
        quantization_bound + 1,
        (1, 1, 128, 2),
        dtype = torch.int16,
        device = device,
    )
    expected_values = reference_pack_row(codes, bits).view(1, 2, 128 * bits // 8)
    values = torch.empty_like(expected_values)
    _pack_bit_row(codes.view(1, 128, 2), values, bits)
    assert torch.equal(values, expected_values)

    byte_pairs = expected_values.view(1, 2, -1, 2).to(torch.int32)
    units = (byte_pairs[..., 0] & 0xff) | ((byte_pairs[..., 1] & 0xff) << 8)
    unpacked = _unpack_bit_codes(units, bits)
    expected_codes = codes.view(1, 128, 2).permute(0, 2, 1).to(torch.int64)
    assert torch.equal(unpacked, expected_codes)


@pytest.mark.parametrize("bits", [2, 9, 12])
def test_quantized_state_row_conversion(device, bits):
    torch.manual_seed(32)
    state = torch.randn((2, 128, 2), dtype = torch.float, device = device) * 0.05
    state[1].zero_()
    rotated_state = _rotate_state(state).unsqueeze(0).unsqueeze(0)
    expected_values, expected_scales = quantize_pool(rotated_state, bits)
    values = torch.empty((2, 2, 128 * bits // 8), dtype = torch.int8, device = device)
    scales = torch.empty_like(expected_scales[0, 0])

    _pack_quantized_state_row(state, values, scales, bits)
    assert torch.equal(values, expected_values[0, 0])
    torch.testing.assert_close(scales, expected_scales[0, 0], rtol = 2e-3, atol = 1e-6)
    assert not values[1].any() and not scales[1].any()

    unpacked = _unpack_quantized_state_row(values, scales, bits)
    expected_state = _rotate_state(dequantize_pool(
        values.unsqueeze(0).unsqueeze(0), scales.unsqueeze(0).unsqueeze(0), bits
    ))[0, 0]
    torch.testing.assert_close(unpacked, expected_state, rtol = 1e-5, atol = 1e-6)
    assert not unpacked[1].any()

    repacked_values = torch.empty_like(values)
    repacked_scales = torch.empty_like(scales)
    _pack_quantized_state_row(unpacked, repacked_values, repacked_scales, bits)
    assert torch.equal(repacked_values, values)
    assert torch.equal(repacked_scales, scales)


OUTPUT_TOLERANCE_BY_BITS = {2: 1e-1, 5: 5e-2, 8: 5e-2, 9: 1e-2, 12: 5e-3}

@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("recurrent_bits", [2, 5, 8, 9, 12])
@pytest.mark.parametrize(
    "bsz,seqlen,num_k_heads,num_v_heads,k_head_dim,v_head_dim",
    [
        (1, 17, 2, 4, 128, 128),
        (2, 9, 4, 8, 128, 128),
    ],
)
@torch.inference_mode()
def test_cuda_recurrent_gated_delta_rule_packed_matches_torch(
    device,
    history,
    recurrent_bits,
    bsz,
    seqlen,
    num_k_heads,
    num_v_heads,
    k_head_dim,
    v_head_dim,
):
    # Compare with an fp16 scan of the equivalent unrotated state.
    torch.manual_seed(4321)
    qkv_dim = 2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim
    num_slots = 2 * bsz + 2
    mixed_qkv = (torch.randn((bsz, seqlen, qkv_dim), dtype = torch.float, device = device) * 0.25).bfloat16()
    g = torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device) * 0.5 - 1.0
    beta = torch.sigmoid(torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device)).bfloat16()
    state0 = torch.randn((num_slots, 1, num_v_heads, k_head_dim, v_head_dim), dtype = torch.float, device = device) * 0.05
    original_values, original_scales = quantize_pool(state0, recurrent_bits)
    cuda_scales = original_scales.clone()
    slots = 2 * torch.arange(bsz, dtype = torch.int32, device = device) + 1

    ref_out, ref_state = _torch_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        dequantize_pool(original_values, original_scales, recurrent_bits),
        slots,
        history,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        rotate_qk = True,
    )
    cuda_out, cuda_state = _run_cuda_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        original_values,
        slots,
        history,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        recurrent_scales = cuda_scales,
        recurrent_bits = recurrent_bits,
    )
    fp16_out, _ = _run_cuda_gated_delta_rule(
        mixed_qkv,
        g,
        beta,
        _rotate_state(dequantize_pool(original_values, original_scales, recurrent_bits)).half().contiguous(),
        slots,
        history,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
    )

    output_state = dequantize_pool(cuda_state, cuda_scales, recurrent_bits)
    out_tol = OUTPUT_TOLERANCE_BY_BITS[recurrent_bits]
    torch.testing.assert_close(cuda_out, ref_out, rtol = out_tol, atol = out_tol)
    torch.testing.assert_close(cuda_out, fp16_out, rtol = 0.0, atol = 2e-3)
    quantization_bound = (1 << (recurrent_bits - 1)) - 1
    written_rows = slots.long()
    quantization_step = cuda_scales[written_rows].float().squeeze(1)[
        ..., torch.arange(k_head_dim, device = device) // 32
    ]
    quantization_step = quantization_step.permute(0, 1, 3, 2).unsqueeze(1) / quantization_bound
    assert ((output_state[written_rows] - ref_state[written_rows]).abs() <= 2 * quantization_step).all()

    untouched_rows = torch.ones(num_slots, dtype = torch.bool, device = device)
    untouched_rows[written_rows] = False
    assert torch.equal(cuda_state[untouched_rows], original_values[untouched_rows])
    assert torch.equal(cuda_scales[untouched_rows], original_scales[untouched_rows])


REPLAY_SHAPES = [
    (3, 9, 2, 4, 64, 64),        # generic kernel
    (2, 17, 4, 8, 128, 128),     # 128 kernel
    (1, 65, 2, 8, 128, 128),     # 128 kernel, a long (suffix-match class) pass
    (2, 9, 2, 4, 256, 256),      # generic, wide
]
REPLAY_PARAMS = [
    pytest.param(layers, torch.float, 0, *shape)
    for shape in REPLAY_SHAPES
    for layers in (1, 3, 50)
]
REPLAY_PARAMS += [
    # Packed V_SPLIT 1, V_SPLIT 4, and one 50-layer batching case
    pytest.param(1, torch.int8, 9, 2, 17, 4, 8, 128, 128),
    pytest.param(3, torch.int8, 9, 1, 65, 2, 8, 128, 128),
    pytest.param(50, torch.int8, 9, 2, 17, 4, 8, 128, 128),
]


@pytest.mark.parametrize(
    "layers,state_dtype,state_bits,bsz,seqlen,num_k_heads,num_v_heads,k_head_dim,v_head_dim",
    REPLAY_PARAMS,
)
@torch.inference_mode()
def test_scan_replay_matches_kernel_prefix(
    device, layers, state_dtype, state_bits, bsz, seqlen, num_k_heads, num_v_heads, k_head_dim, v_head_dim
):
    """Replay selected prefixes bit-exactly, including launches spanning 50 layers."""
    torch.manual_seed(7 + layers)
    qkv_dim = 2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim
    replayed_batch_row = bsz - 1
    layer_inputs = []
    for _ in range(layers):
        mixed_qkv = (torch.randn((bsz, seqlen, qkv_dim), dtype = torch.float, device = device) * 0.25).bfloat16()
        g = torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device) * 0.5 - 1.0
        beta = torch.sigmoid(torch.randn((bsz, seqlen, num_v_heads), dtype = torch.float, device = device)).bfloat16()
        values = torch.randn((2 * bsz, 1, num_v_heads, k_head_dim, v_head_dim), device = device) * 0.05
        if state_dtype == torch.int8:
            values, scales = quantize_pool(values, state_bits)
        else:
            scales = None
        layer_inputs.append((mixed_qkv, g, beta, values, scales))
    slots = 2 * torch.arange(bsz, dtype = torch.int32, device = device) + 1
    slots_in = slots - 1
    base_row = int(slots_in[replayed_batch_row])
    scratch_row = int(slots[replayed_batch_row])

    for prefix in sorted({1, 2, seqlen // 2, seqlen - 1, seqlen}):
        expected_states = []
        for mixed_qkv, g, beta, input_values, input_scales in layer_inputs:
            expected_values = input_values.clone()
            expected_scales = input_scales.clone() if input_scales is not None else None
            kernel_out = torch.empty(
                (bsz, prefix, num_v_heads, v_head_dim), dtype = torch.bfloat16, device = device
            )
            ext.cuda_recurrent_gated_delta_rule(
                mixed_qkv[:, :prefix].contiguous(), g[:, :prefix].contiguous(), beta[:, :prefix].contiguous(),
                expected_values, kernel_out, num_k_heads, num_v_heads, k_head_dim, v_head_dim,
                slots, True, slots_in, expected_scales, state_bits,
            )
            expected_states.append((expected_values, expected_scales))

        jobs = []
        replayed_values = []
        replayed_scales = []
        for mixed_qkv, g, beta, input_values, input_scales in layer_inputs:
            replay_values = input_values.clone()
            replay_scales = input_scales.clone() if input_scales is not None else None
            replayed_values.append(replay_values)
            replayed_scales.append(replay_scales)
            row_bytes = replay_values.stride(0) * replay_values.element_size()
            scale_ptr = replay_scales.data_ptr() if replay_scales is not None else 0
            scale_row_bytes = replay_scales.stride(0) * replay_scales.element_size() \
                if replay_scales is not None else 0
            jobs.append(ext.ScanReplayJob(
                mixed_qkv[replayed_batch_row].data_ptr(), g[replayed_batch_row].data_ptr(),
                beta[replayed_batch_row].data_ptr(), replay_values.data_ptr() + base_row * row_bytes,
                replay_values.data_ptr() + scratch_row * row_bytes, 0, prefix, 1,
                scale_ptr + base_row * scale_row_bytes, scale_ptr + scratch_row * scale_row_bytes,
            ))
        ext.batched_scan_replay(
            jobs, torch.device(device).index, 0, num_k_heads, num_v_heads, k_head_dim, v_head_dim,
            ext.STATE_PACKED if state_dtype == torch.int8 else ext.STATE_FP32, state_bits,
        )
        torch.cuda.synchronize(device)
        for actual_values, actual_scales, (expected_values, expected_scales), \
                (_, _, _, original_values, original_scales) in zip(
                    replayed_values, replayed_scales, expected_states, layer_inputs
                ):
            assert torch.equal(actual_values[scratch_row], expected_values[scratch_row]), \
                f"prefix {prefix}: state differs"
            untouched_rows = torch.ones(2 * bsz, dtype = torch.bool, device = device)
            untouched_rows[scratch_row] = False
            assert torch.equal(actual_values[untouched_rows], original_values[untouched_rows])
            if actual_scales is not None:
                assert torch.equal(actual_scales[scratch_row], expected_scales[scratch_row]), \
                    f"prefix {prefix}: scales differ"
                assert torch.equal(actual_scales[untouched_rows], original_scales[untouched_rows])


@pytest.mark.parametrize(
    "bsz,seqlen,num_k_heads,num_v_heads,k_head_dim,v_head_dim",
    [
        (1, 64, 1, 1, 64, 64),
        (2, 65, 2, 4, 64, 64),
        (3, 127, 2, 4, 128, 128),
    ],
)
@torch.inference_mode()
def test_chunk_gated_delta_rule_matches_torch(
    device,
    bsz,
    seqlen,
    num_k_heads,
    num_v_heads,
    k_head_dim,
    v_head_dim,
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

    ref_out, ref_state = _torch_gated_delta_rule(
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


@torch.inference_mode()
def test_chunk_gated_delta_rule_packed_state(device):
    bits = 9
    bsz, seqlen, num_k_heads, num_v_heads, k_head_dim, v_head_dim = 1, 65, 2, 4, 128, 128
    torch.manual_seed(5679)
    qkv_dim = 2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim
    mixed_qkv = (torch.randn((bsz, seqlen, qkv_dim), device = device) * 0.25).bfloat16()
    g = torch.randn((bsz, seqlen, num_v_heads), device = device) * 0.5 - 1.0
    beta = torch.sigmoid(torch.randn((bsz, seqlen, num_v_heads), device = device)).bfloat16()
    initial_state = torch.randn((bsz, 1, num_v_heads, k_head_dim, v_head_dim), device = device) * 0.05
    values, scales = quantize_pool(_rotate_state(initial_state), bits)
    dequantized_initial_state = _rotate_state(dequantize_pool(values, scales, bits))

    ref_out, ref_state = _torch_gated_delta_rule(
        mixed_qkv, g, beta, dequantized_initial_state, None, False,
        num_k_heads, num_v_heads, k_head_dim, v_head_dim,
    )
    out = gated_delta_rule_fn(
        mixed_qkv, beta, g, values, None, False, True,
        num_k_heads, num_v_heads, num_k_heads * k_head_dim, num_v_heads * v_head_dim,
        k_head_dim, v_head_dim, recurrent_scales = scales, recurrent_bits = bits,
    )

    torch.testing.assert_close(out, ref_out, rtol = 5e-2, atol = 5e-2)
    final_state = _rotate_state(dequantize_pool(values, scales, bits))
    torch.testing.assert_close(final_state, ref_state, rtol = 1e-2, atol = 1e-2)
    rotated_ref = _rotate_state(ref_state)
    expected_scales = quantize_pool(rotated_ref, bits)[1]
    torch.testing.assert_close(scales, expected_scales, rtol = 1e-2, atol = 1e-3)


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


@pytest.mark.parametrize("bits", [1, 13])
@torch.inference_mode()
def test_packed_state_rejects_invalid_width(device, bits):
    bsz, seqlen, nk, nv, dk, dv = 1, 3, 2, 4, 128, 128
    mixed_qkv = torch.randn(bsz, seqlen, 2 * nk * dk + nv * dv, device = device).bfloat16()
    g = -torch.rand(bsz, seqlen, nv, device = device)
    beta = torch.rand(bsz, seqlen, nv, device = device).bfloat16()
    out = torch.empty(bsz, seqlen, nv, dv, dtype = torch.bfloat16, device = device)
    values = torch.randint(-127, 128, (2, 1, nv, dv, dk * bits // 8), dtype = torch.int8, device = device)
    scales = torch.rand((2, 1, nv, dv, dk // 32), device = device).half()
    with pytest.raises(RuntimeError, match = "packed recurrent state requires 2..12 bits"):
        ext.cuda_recurrent_gated_delta_rule(
            mixed_qkv, g, beta, values, out, nk, nv, dk, dv, None, False, None, scales, bits
        )


@torch.inference_mode()
def test_packed_state_rejects_ineligible_operands(device):
    bits = 8
    bsz, seqlen, nk, nv, dk, dv = 1, 3, 2, 4, 128, 128
    mixed_qkv = torch.randn(bsz, seqlen, 2 * nk * dk + nv * dv, device = device).bfloat16()
    g = -torch.rand(bsz, seqlen, nv, device = device)
    beta = torch.rand(bsz, seqlen, nv, device = device).bfloat16()
    out = torch.empty(bsz, seqlen, nv, dv, dtype = torch.bfloat16, device = device)
    values = torch.randint(-127, 128, (2, 1, nv, dv, dk * bits // 8), dtype = torch.int8, device = device)
    scales = torch.rand((2, 1, nv, dv, dk // 32), device = device).half()

    channelwise_g = -torch.rand(bsz, seqlen, nv, dk, device = device)
    with pytest.raises(RuntimeError, match = "not supported for channelwise"):
        ext.cuda_recurrent_gated_delta_rule(
            mixed_qkv, channelwise_g, beta, values, out, nk, nv, dk, dv, None, False, None, scales, bits
        )

    small_dk = small_dv = 64
    small_qkv = torch.randn(bsz, seqlen, 2 * nk * small_dk + nv * small_dv, device = device).bfloat16()
    small_values = torch.randint(
        -127, 128, (2, 1, nv, small_dv, small_dk * bits // 8), dtype = torch.int8, device = device
    )
    small_scales = torch.rand((2, 1, nv, small_dv, small_dk // 32), device = device).half()
    small_out = torch.empty(bsz, seqlen, nv, small_dv, dtype = torch.bfloat16, device = device)
    with pytest.raises(RuntimeError, match = "requires 128x128"):
        ext.cuda_recurrent_gated_delta_rule(
            small_qkv, g, beta, small_values, small_out, nk, nv, small_dk, small_dv,
            None, False, None, small_scales, bits,
        )

    with pytest.raises(RuntimeError, match = "packed recurrent state requires recurrent_scales"):
        ext.cuda_recurrent_gated_delta_rule(
            mixed_qkv, g, beta, values, out, nk, nv, dk, dv, None, False, None, None, bits
        )

    fp16_state = torch.zeros(2, 1, nv, dk, dv, dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "recurrent_scales is only used with the packed recurrent state"):
        ext.cuda_recurrent_gated_delta_rule(
            mixed_qkv, g, beta, fp16_state, out, nk, nv, dk, dv, None, False, None, scales
        )

    int16_state = torch.zeros(2, 1, nv, dv, dk * bits // 8, dtype = torch.int16, device = device)
    with pytest.raises(RuntimeError, match = r"recurrent_state must be float32, float16 or int8 \(packed state\)"):
        ext.cuda_recurrent_gated_delta_rule(
            mixed_qkv, g, beta, int16_state, out, nk, nv, dk, dv, None, False, None, scales, bits
        )
