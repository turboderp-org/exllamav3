"""
int64 offset arithmetic in the Triton paged prefill kernel: Q / out row offsets and split-kv partial offsets
that exceed 2**31 elements must not wrap. Rows beyond the int32 boundary (and a below-boundary control) are
checked against closed-form expectations of hand-built inputs (two-key windows, uniform attention prefix means);
an empty query over a long cache-less key set must return an empty output. The two overflow tests need ~10-11 GiB
of free device memory.
"""
import math

import pytest
import torch

from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill

GiB = 1024**3


def require_free_memory(device, min_free_bytes: int):
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < min_free_bytes:
        pytest.skip(f"test requires at least {min_free_bytes / GiB:.1f} GiB free CUDA memory")


@pytest.mark.slow
@torch.inference_mode()
def test_prefill_q_and_out_offsets_past_int32(device):
    """Q/out element offsets are row * n_q_heads * head_dim + ...; at 32 heads x 256 dim the
    int32 boundary sits at row 2**31 / 8192 = 262144. Rows past it wrapped negative before the
    row64 widening: the q load reads far out of bounds and the out store never lands, leaving
    the NaN sentinel in place. ~9 GiB: q and out must each span > 2**31 elements (4 GiB fp16)."""
    require_free_memory(device, 10 * GiB)

    q_len = 262272  # two BLOCK_M = 64 blocks past the 262144 boundary
    n_q_heads = 32
    head_dim = 256

    q = torch.zeros((1, q_len, n_q_heads, head_dim), dtype = torch.half, device = device)
    k = torch.zeros((1, q_len, 1, head_dim), dtype = torch.half, device = device)
    v = torch.zeros_like(k)
    out = torch.full_like(q, torch.nan)

    token = torch.arange(q_len, dtype = torch.float32, device = device)
    head = torch.arange(n_q_heads, dtype = torch.float32, device = device)
    q[0, :, :, 0] = ((token.remainder(127) - 63)[:, None] / 16 + head[None, :] / 32).half()
    k[0, :, 0, 0] = torch.where(token.remainder(2) == 0, 1.0, -1.0).half()
    v[0, :, 0, 0] = (token.remainder(251) / 251).half()

    actual = paged_attn_triton_prefill(
        q, None, None, None, None, None, None,
        causal = True,
        window_size = 1,
        num_splits = 1,
        out = out,
        k_new = k,
        v_new = v,
    )

    # 262142 sits below the boundary as a control; the rest overflow without the fix
    for pos in (262142, 262144, 262206, 262271):
        scores = q[0, pos, :, 0].float()[:, None] * \
            k[0, pos - 1:pos + 1, 0, 0].float()[None, :] / math.sqrt(head_dim)
        weights = scores.softmax(dim = -1)
        expected = weights @ v[0, pos - 1:pos + 1, 0, 0].float()
        torch.testing.assert_close(actual[0, pos, :, 0].float(), expected, atol = 3e-4, rtol = 3e-4)
        torch.testing.assert_close(
            actual[0, pos, :, 1:],
            torch.zeros_like(actual[0, pos, :, 1:]),
        )

    del actual, out, v, k, q
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


@pytest.mark.slow
@torch.inference_mode()
def test_prefill_split_partial_offsets_past_int32(device):
    """partial_o element offsets are pid_lin * BLOCK_M * head_dim with pid_lin up to
    programs * num_splits; at 16384 tokens x 32 heads (programs = 8192, BLOCK_M = 64) the
    boundary of 2**31 / 16384 = 131072 falls at split count 16, so 17 splits push the tail
    programs past it. ~10 GiB: the fp32 partial buffer must span > 2**31 elements (8.5 GiB)."""
    require_free_memory(device, 11 * GiB)

    q_len = 16384
    n_q_heads = 32
    head_dim = 256

    q = torch.zeros((1, q_len, n_q_heads, head_dim), dtype = torch.half, device = device)
    k = torch.zeros((1, q_len, 1, head_dim), dtype = torch.half, device = device)
    v = torch.zeros_like(k)
    out = torch.full_like(q, torch.nan)

    # q = k = 0 gives uniform attention, so row `pos` must come out as the prefix mean of v.
    # v varies by position so that partials landing in (or read from) the wrong slot are
    # detectably wrong rather than interchangeable
    token = torch.arange(q_len, dtype = torch.float32, device = device)
    v[0, :, 0, 0] = (token.remainder(251) / 251).half()

    actual = paged_attn_triton_prefill(
        q, None, None, None, None, None, None,
        causal = True,
        window_size = None,
        num_splits = 17,
        out = out,
        k_new = k,
        v_new = v,
    )

    prefix_mean = v[0, :, 0, 0].float().cumsum(0) / (token + 1)

    # overflow starts at row 15360 for the high heads (pid_m = 240, bh >= 30) and covers all
    # heads from 15424; 1000 is a below-boundary control
    for pos in (1000, 15360, 16000, 16383):
        torch.testing.assert_close(
            actual[0, pos, :, 0].float(),
            prefix_mean[pos].expand(n_q_heads),
            atol = 1e-3,
            rtol = 1e-3,
        )
        torch.testing.assert_close(
            actual[0, pos, :, 1:],
            torch.zeros_like(actual[0, pos, :, 1:]),
        )

    del actual, out, v, k, q
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


@torch.inference_mode()
def test_prefill_empty_q_with_long_kv_returns_empty(device):
    require_free_memory(device, 1 * GiB)

    q = torch.empty((1, 0, 8, 128), dtype = torch.half, device = device)
    k = torch.empty((1, 8192, 1, 128), dtype = torch.half, device = device)
    v = torch.empty_like(k)

    actual = paged_attn_triton_prefill(
        q, None, None, None, None, None, None,
        causal = True,
        k_new = k,
        v_new = v,
    )

    assert actual.shape == q.shape
    assert actual.dtype == q.dtype
    assert actual.device == q.device
