"""
Proxy error for weights quantized with CPU work buffers: block_trace_pair_streamed() streams column blocks of the
weight and its quantized version to the device and must match the dense block_trace() of the full error and weight.
"""

import pytest
import torch

from exllamav3.modules.quant.exl3_lib.quantize import block_trace, block_trace_pair_streamed


@pytest.mark.parametrize("k, n, col_block", [
    (256, 1024, 8192),   # single block
    (256, 1000, 128),    # ragged last block
    (512, 777, 1),       # one column at a time
])
def test_streamed_matches_dense(device, k, n, col_block):
    g = torch.Generator().manual_seed(0)
    W = torch.randn(k, n, generator = g)
    Q = W + 0.05 * torch.randn(k, n, generator = g)
    A = torch.randn(k, k, generator = g)
    H = (A @ A.T / k + torch.eye(k)).to(device)

    ref_num = block_trace((W - Q).to(device), H)
    ref_den = block_trace(W.to(device), H)
    num, den = block_trace_pair_streamed(W, Q, H, device, col_block = col_block)

    torch.testing.assert_close(num, ref_num, rtol = 1e-4, atol = 0)
    torch.testing.assert_close(den, ref_den, rtol = 1e-4, atol = 0)


def test_streamed_does_not_modify_inputs(device):
    g = torch.Generator().manual_seed(1)
    W = torch.randn(128, 300, generator = g)
    Q = torch.randn(128, 300, generator = g)
    H = torch.eye(128, device = device)
    W0, Q0 = W.clone(), Q.clone()
    block_trace_pair_streamed(W, Q, H, device, col_block = 64)
    assert torch.equal(W, W0) and torch.equal(Q, Q0)
