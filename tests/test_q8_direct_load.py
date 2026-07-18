import os
import statistics
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from exllamav3.modules.attention_fn.triton_paged import (
    has_triton,
    paged_attn_triton_decode,
)


def _make_inputs(context: int, head_dim: int):
    device = torch.device("cuda:0")
    page_size = 256
    n_q_heads = 32
    n_kv_heads = 8
    num_pages = context // page_size
    token_dim = n_kv_heads * head_dim

    torch.manual_seed(0)
    q = torch.randn(
        (1, 1, n_q_heads, head_dim), dtype = torch.float16, device = device
    )
    k_bytes = torch.randint(
        0, 256, (num_pages, page_size, token_dim), dtype = torch.uint8,
        device = device,
    )
    v_bytes = torch.randint(
        0, 256, (num_pages, page_size, token_dim), dtype = torch.uint8,
        device = device,
    )
    k_cache = k_bytes.view(torch.int32)
    v_cache = v_bytes.view(torch.int32)
    scale_shape = (num_pages, page_size, token_dim // 32)
    k_scales = torch.rand(scale_shape, dtype = torch.float16, device = device)
    v_scales = torch.rand(scale_shape, dtype = torch.float16, device = device)
    block_table = torch.arange(
        num_pages, dtype = torch.int32, device = device
    ).unsqueeze(0)
    cache_seqlens = torch.tensor([context], dtype = torch.int32, device = device)

    return {
        "q": q,
        "k": None,
        "v": None,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_table": block_table,
        "cache_seqlens": cache_seqlens,
        "qc": (k_scales, v_scales, 8, 8),
        "max_kv_len": context,
        "n_kv_heads_override": n_kv_heads,
        "block_n": max(16, 8192 // head_dim),
        "num_splits": 16,
    }


def _run(inputs, q8_direct):
    return paged_attn_triton_decode(
        **inputs,
        _q8_direct = q8_direct,
    )


def _benchmark(inputs, q8_direct, warmups = 10, repetitions = 100, samples = 7):
    for _ in range(warmups):
        _run(inputs, q8_direct)
    torch.cuda.synchronize()

    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing = True)
        end = torch.cuda.Event(enable_timing = True)
        start.record()
        for _ in range(repetitions):
            _run(inputs, q8_direct)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / repetitions)
    return statistics.median(timings)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton,
    reason = "CUDA and Triton are required",
)
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@torch.inference_mode()
def test_q8_direct_load_matches_word_expansion(head_dim):
    inputs = _make_inputs(1024, head_dim)
    word = _run(inputs, False)
    direct = _run(inputs, True)
    torch.testing.assert_close(direct, word, rtol = 0, atol = 2e-5)


@pytest.mark.skipif(
    os.environ.get("EXL3_RUN_PERF_TESTS") != "1",
    reason = "set EXL3_RUN_PERF_TESTS=1 to run performance tests",
)
@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton,
    reason = "CUDA and Triton are required",
)
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@torch.inference_mode()
def test_q8_direct_load_benchmark(head_dim):
    inputs = _make_inputs(16384, head_dim)
    word_us = _benchmark(inputs, False)
    direct_us = _benchmark(inputs, True)
    speedup = word_us / direct_us
    print(
        f"Q8 decode on {torch.cuda.get_device_name()}, head_dim={head_dim}: "
        f"word={word_us:.3f} us, direct={direct_us:.3f} us, "
        f"speedup={speedup:.3f}x"
    )
    assert direct_us < word_us
