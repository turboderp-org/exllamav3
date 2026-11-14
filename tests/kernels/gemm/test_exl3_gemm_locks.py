"""
ext.exl3_gemm with a caller-owned lock buffer. By default the split-K counters live in one buffer per device, so two
GEMMs running at the same time on different streams use the same counters. With a buffer each, concurrent launches
(eager or replayed from a graph) match their serial results bit for bit, and every buffer is left zeroed for the next
launch.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import generator, rand_linear

pytestmark = pytest.mark.cuda_only


def _operands(m, k, n, K, seed, device):
    gen = generator(seed)
    w = rand_linear(k, n, K, gen, device)
    A = (torch.randn((m, k), generator = gen) * 0.5).half().to(device)
    C = torch.empty((m, n), dtype = torch.half, device = device)
    return A, w, C, torch.empty_like(A)


def _launch(op, shape, sms, locks = None):
    A, w, C, A_had = op
    return ext.exl3_gemm(A, w["trellis"], C, w["suh"], A_had, w["svh"], shape, False, True, sms, locks)


def _shape(m, k, n, K):
    return next(s for s in range(1, ext.exl3_gemm_num_kernel_shapes() + 1) if ext.exl3_gemm_shape_compat(s, m, k, n, K))


def _locks(n, device):
    return torch.zeros(ext.exl3_gemm_lock_elements(n), dtype = torch.int, device = device)


@torch.inference_mode()
def test_owned_locks_validation(device):
    for n in [0, -128, 127, 129]:
        with pytest.raises(RuntimeError, match = "positive multiple of 128"):
            ext.exl3_gemm_lock_elements(n)
    assert ext.exl3_gemm_lock_elements(1024) == 8
    op = _operands(8, 512, 1024, 4, 0, device)
    count = ext.exl3_gemm_lock_elements(1024)
    for locks, reason in [
        (torch.zeros(count, dtype = torch.int), "locks must be on A's device"),
        (torch.zeros(count, dtype = torch.long, device = device), "incorrect datatype"),
        (torch.zeros(count * 2, dtype = torch.int, device = device)[::2], "locks must be contiguous"),
        (torch.zeros(count - 1, dtype = torch.int, device = device), "locks buffer too small"),
    ]:
        with pytest.raises(RuntimeError, match = reason):
            _launch(op, 0, 0, locks)


@pytest.mark.parametrize("shape", [0, None])
@torch.inference_mode()
def test_owned_locks_match_shared(device, shape):
    """Same result with the device's buffer and with an owned one, autotuned or forced"""
    m, k, n, K = 64, 2048, 1024, 4
    op = _operands(m, k, n, K, 1, device)
    shape = _shape(m, k, n, K) if shape is None else shape
    sms = 0 if shape == 0 else 32
    _launch(op, shape, sms)
    ref = op[2].clone()
    locks = _locks(n, device)
    _launch(op, shape, sms, locks)
    torch.cuda.synchronize(device)
    assert torch.equal(op[2], ref)
    assert torch.count_nonzero(locks) == 0


@pytest.mark.parametrize("graph", [False, True])
@torch.inference_mode()
def test_owned_locks_concurrent_streams(device, graph):
    """A long GEMM and a short one on two streams, each on part of the SMs so both are resident at once"""
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if num_sms < 32:
        pytest.skip("needs at least 32 SMs")
    sms_a, sms_b = num_sms // 2, num_sms // 4
    big = _operands(2048, 4096, 1024, 4, 2, device)
    small = _operands(8, 4096, 1024, 4, 3, device)
    shape_a, shape_b = _shape(2048, 4096, 1024, 4), _shape(8, 4096, 1024, 4)
    locks_a, locks_b = _locks(1024, device), _locks(1024, device)

    _launch(big, shape_a, sms_a, locks_a)
    ref_a = big[2].clone()
    _launch(small, shape_b, sms_b, locks_b)
    ref_b = small[2].clone()
    torch.cuda.synchronize(device)

    stream_a, stream_b = torch.cuda.Stream(device), torch.cuda.Stream(device)
    g = None
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream = stream_a):
            _launch(big, shape_a, sms_a, locks_a)
    stream_a.wait_stream(torch.cuda.current_stream(device))
    stream_b.wait_stream(torch.cuda.current_stream(device))

    outs = []
    for _ in range(20):
        with torch.cuda.stream(stream_a):
            if g is None:
                _launch(big, shape_a, sms_a, locks_a)
            else:
                g.replay()
            out_a = big[2].clone()
        with torch.cuda.stream(stream_b):
            for _ in range(4):
                _launch(small, shape_b, sms_b, locks_b)
            out_b = small[2].clone()
        outs.append((out_a, out_b))
    torch.cuda.synchronize(device)
    for out_a, out_b in outs:
        assert torch.equal(out_a, ref_a)
        assert torch.equal(out_b, ref_b)
    assert torch.count_nonzero(locks_a) == 0 and torch.count_nonzero(locks_b) == 0
