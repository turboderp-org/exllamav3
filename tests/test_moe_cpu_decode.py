"""Single-token CPU MoE finishing agrees with the existing batched path.

Exercise partial/empty routing, repeated expert selections, biases, and more workers than
128-column output blocks. The two-token reference retains the separate transform and
accumulation phases, independently of the decode-only fused finish.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext


@pytest.fixture
def cpu_runtime():
    # The native pool pins its caller. Restore affinity before later tests start their
    # own PyTorch/OpenMP work, rather than leaving that work confined to worker zero.
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        if affinity is not None:
            os.sched_setaffinity(0, affinity)
        torch.set_num_threads(threads)


@pytest.mark.parametrize("activation", [0, 1, 2, 3])
@pytest.mark.parametrize("bias", [False, True])
def test_decode_finish_matches_batch(activation, bias, cpu_runtime):
    gen = torch.Generator().manual_seed(174)
    hidden, intermediate, experts = 384, 256, 3
    gated = activation != 2
    swizzled = ext.exl3_moe_cpu_has_avx512_bw()

    def matrices(k, n, enabled=True):
        ts, us, vs, bs = [], [], [], []
        if enabled:
            for e in range(experts):
                bits = (1, 3, 8)[e]
                t = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * bits),
                                  dtype=torch.int16, generator=gen)
                if swizzled and bits != 8:
                    t = t.view(k // 16, n // 128, 8, 16 * bits).permute(1, 0, 2, 3).contiguous().view_as(t)
                ts.append(t)
                us.append((torch.randn(k, generator=gen) * 0.015).half())
                vs.append(torch.randn(n, generator=gen).half())
                if bias:
                    bs.append((torch.randn(n, generator=gen) * 0.1).half())
        return ts, us, vs, bs

    g = matrices(hidden, intermediate, gated)
    u = matrices(hidden, intermediate)
    d = matrices(intermediate, hidden)
    handle = ext.exl3_moe_cpu_make_layer(
        *g[:3], *u[:3], *d[:3], g[3], u[3], d[3], activation, 1.0, int(swizzled))
    try:
        x = torch.randn(1, hidden, generator=gen).half()
        for selected in ([0, 1, 2], [-1, 1, -1], [-1, -1, -1], [1, 1, 1]):
            sel = torch.tensor([selected], dtype=torch.long)
            weights = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.half)
            reference = torch.empty(2, hidden)
            ext.exl3_moe_cpu_forward(handle, x.repeat(2, 1), sel.repeat(2, 1),
                                    weights.repeat(2, 1), reference, 1)
            for threads in (1, 4, 16):
                out = torch.full((1, hidden), float("nan"))
                ext.exl3_moe_cpu_forward(handle, x, sel, weights, out, threads)
                torch.testing.assert_close(out[0], reference[0], rtol=5e-4, atol=2e-5)
    finally:
        ext.exl3_moe_cpu_free_layer(handle)
