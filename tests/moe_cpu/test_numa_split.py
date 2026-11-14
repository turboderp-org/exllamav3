"""
NUMA column-split forward (exl3_moe_cpu_forward_numa) agrees with the single-pool path.

The numa driver partitions every expert by 128-column blocks across two node pools but keeps
the arithmetic identical (same per-row quantization scales, same per-block kernels), so outputs
must match the single-pool path exactly. Exercises the decode (single-token) and batched
dispatches, biases, uneven block splits (odd 128-block counts), and the K8 single-pool
fallback. Skipped when the host does not expose two NUMA nodes with a readable topology or the
active ISA tier packs no banded rate.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.moe_cpu import cpu_runtime, swizzle, swizzle_layouts   # noqa: F401 (fixture)

pytestmark = pytest.mark.nogpu


def numa_available():
    try:
        orders = ext.exl3_moe_cpu_numa_core_orders()
    except Exception:
        return False
    return len(orders) >= 2 and all(len(o) and n > 0 for o, n in orders)


requires_numa = pytest.mark.skipif(not numa_available(), reason = "needs two readable NUMA nodes")


def make_layer(gen, hidden, intermediate, experts, activation, bias, rates):
    gated = activation != 2

    def matrices(k, n, enabled):
        ts, us, vs, bs = [], [], [], []
        if enabled:
            for e in range(experts):
                bits = rates[e % len(rates)]
                t = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * bits),
                                  dtype = torch.int16, generator = gen)
                t = swizzle(t)
                ts.append(t)
                us.append((torch.randn(k, generator = gen) * 0.015).half())
                vs.append(torch.randn(n, generator = gen).half())
                if bias:
                    bs.append((torch.randn(n, generator = gen) * 0.1).half())
        return ts, us, vs, bs

    g = matrices(hidden, intermediate, gated)
    u = matrices(hidden, intermediate)
    d = matrices(intermediate, hidden)
    return ext.exl3_moe_cpu_make_layer(
        *g[:3], *u[:3], *d[:3], g[3], u[3], d[3], activation, 1.0, 1)


@pytest.mark.parametrize("activation", [0, 2])
@pytest.mark.parametrize("rows", [1, 5])
@requires_numa
def test_numa_matches_single_pool(activation, rows, cpu_runtime):
    gen = torch.Generator().manual_seed(1234)
    hidden, intermediate, experts = 512, 384, 4   # intermediate: 3 blocks -> uneven 2/1 split
    handle = make_layer(gen, hidden, intermediate, experts, activation, True, (2, 3))
    try:
        x = torch.randn(rows, hidden, generator = gen).half()
        sel = torch.tensor([[(e * 2 + 1) % experts for e in range(3)] + [experts - 1]] * rows, dtype = torch.long)
        sel[:, 0] = -1   # partial routing on the first slot
        weights = torch.rand(rows, 4, generator = gen).half() * 0.5
        for threads in (1, 4):
            ref = torch.full((rows, hidden), float("nan"))
            ext.exl3_moe_cpu_forward(handle, x, sel, weights, ref, threads)
            for tpn in (1, 3):
                out = torch.full((rows, hidden), float("nan"))
                ext.exl3_moe_cpu_forward_numa(handle, x, sel, weights, out, tpn)
                torch.testing.assert_close(out, ref, rtol = 0, atol = 0)
    finally:
        ext.exl3_moe_cpu_free_layer(handle)


@requires_numa
def test_numa_repeated_expert_falls_back_to_batched(cpu_runtime):
    # One token selecting the same expert twice: the decode path requires m == 1 chunks, so the
    # numa driver takes the batched dispatches -- results must still match exactly
    gen = torch.Generator().manual_seed(99)
    hidden, intermediate, experts = 384, 256, 2
    handle = make_layer(gen, hidden, intermediate, experts, 0, False, (2, 3))
    try:
        x = torch.randn(1, hidden, generator = gen).half()
        sel = torch.tensor([[1, 1]], dtype = torch.long)
        weights = torch.tensor([[0.3, 0.7]], dtype = torch.half)
        ref = torch.empty(1, hidden)
        ext.exl3_moe_cpu_forward(handle, x, sel, weights, ref, 4)
        out = torch.empty(1, hidden)
        ext.exl3_moe_cpu_forward_numa(handle, x, sel, weights, out, 4)
        torch.testing.assert_close(out, ref, rtol = 0, atol = 0)
    finally:
        ext.exl3_moe_cpu_free_layer(handle)


@requires_numa
def test_numa_k8_falls_back_to_single_pool(cpu_runtime):
    # A K8 expert has no banded packed layout and cannot be column-split: the numa driver must
    # fall back to the single-pool path for the whole layer (and still produce exact output)
    gen = torch.Generator().manual_seed(7)
    hidden, intermediate, experts = 384, 256, 2
    handle = make_layer(gen, hidden, intermediate, experts, 0, False, (8,))
    try:
        x = torch.randn(1, hidden, generator = gen).half()
        sel = torch.tensor([[0, 1]], dtype = torch.long)
        weights = torch.tensor([[0.5, 0.5]], dtype = torch.half)
        ref = torch.empty(1, hidden)
        ext.exl3_moe_cpu_forward(handle, x, sel, weights, ref, 4)
        out = torch.empty(1, hidden)
        ext.exl3_moe_cpu_forward_numa(handle, x, sel, weights, out, 4)
        torch.testing.assert_close(out, ref, rtol = 0, atol = 0)
    finally:
        ext.exl3_moe_cpu_free_layer(handle)
