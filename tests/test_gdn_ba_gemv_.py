"""Native B/A GEMV arithmetic, bias, fallback and changing-input Graph checks."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "GPU required")


def inputs(rows, k, n, use_bias):
    generator = torch.Generator(device = "cuda").manual_seed(330)
    x = torch.randn(rows, k, generator = generator, device = "cuda", dtype = torch.half) * 0.1
    w = torch.randn(n, k, generator = generator, device = "cuda", dtype = torch.half) * 0.1
    bias = torch.randn(n, generator = generator, device = "cuda", dtype = torch.half) if use_bias else None
    return x, w, bias


def run(x, w, bias):
    y = torch.empty(x.shape[0], w.shape[0], device = x.device, dtype = torch.float)
    ext.gdn_ba_gemv(x, w, bias, y)
    return y


@pytest.mark.parametrize("rows,k,n", [
    (1, 5120, 96), (1, 5120, 95), (1, 5120, 97),
    (1, 5118, 96), (1, 5122, 96), (2, 5120, 96),
    (4, 64, 9), (1, 64, 1),
])
@pytest.mark.parametrize("use_bias", [False, True])
@torch.inference_mode()
def test_ba_gemv(rows, k, n, use_bias):
    x, w, bias = inputs(rows, k, n, use_bias)
    y = run(x, w, bias)
    reference = x.double() @ w.double().T
    if bias is not None:
        reference += bias.double()
    torch.testing.assert_close(y.double(), reference, rtol = 2e-5, atol = 5e-5)
    assert y.isfinite().all()

    # Duplicating the input forces the original eight-warp path (M > 1).
    # Each output keeps exactly the same FMA and reduction order.
    original = run(torch.cat((x, x)), w, bias)[:rows]
    assert torch.equal(y.view(torch.int32), original.view(torch.int32))


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("use_bias", [False, True])
@torch.inference_mode()
def test_ba_graph(rows, use_bias):
    x, w, bias = inputs(rows, 5120, 96, use_bias)
    y = torch.empty(rows, 96, device = x.device, dtype = torch.float)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            ext.gdn_ba_gemv(x, w, bias, y)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ext.gdn_ba_gemv(x, w, bias, y)
    source = x.clone()
    for factor in (1.0, 0.9, 1.1):
        x.copy_(source * factor)
        graph.replay()
        eager = run(x, w, bias)
        original = run(torch.cat((x, x)), w, bias)[:rows]
        assert torch.equal(y.view(torch.int32), eager.view(torch.int32))
        assert torch.equal(y.view(torch.int32), original.view(torch.int32))
        first = y.clone()
        graph.replay()
        assert torch.equal(y.view(torch.int32), first.view(torch.int32))


@pytest.mark.parametrize("use_bias", [False, True])
@torch.inference_mode()
def test_ba_cancellation(use_bias):
    x, w, bias = inputs(1, 5120, 96, use_bias)
    # Alternating products, including large and tiny terms, stress cancellation.
    x[:, ::2] = 16
    x[:, 1::2] = -16
    w[:, 1::2] = w[:, ::2]
    w[:, ::64] = 2 ** -12
    y = run(x, w, bias)
    original = run(torch.cat((x, x)), w, bias)[:1]
    assert y.isfinite().all()
    assert torch.equal(y.view(torch.int32), original.view(torch.int32))
