"""
Multi-worker N-split, kernel level: the sum of W workers' partial outputs equals
the single-worker reference.

Each worker registers a slice of every expert along the intermediate dim -- gate/up
sliced along n, down along k, at 128-column Hadamard-block granularity (the same Megatron-style
decomposition the host's _slice_ranges produces) -- and computes the partial output over its
slice with the unmodified kernels; the host sums the per-worker partials on the GPU. This checks
the decomposition at the kernel level, without processes: register the same layer once per
slice, forward the identical tokens through each registration, and compare the summed partials
against the full layer. Uneven slices (remainder blocks), gateless layers, skipped experts
(-1), repeated experts, and the packed layout of the active tier are covered.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model.moe_cpu_host import _slice_ranges
from testlib.moe_cpu import cpu_runtime, swizzle, swizzle_layouts  # noqa: F401 (fixture)

pytestmark = pytest.mark.nogpu


def make_registration(projs, lo, hi, swizzled):
    """One worker's registration of a (gate, up, down) projection set: native-order tensors
    sliced to intermediate columns [lo, hi), then packed exactly like the child loader does
    (fetch native -> slice -> repack; the slice's whole 128-column blocks keep every packed
    band group legal). Gate/up: trellis dim 1 + svh + bias are intermediate-side. Down:
    trellis dim 0 + suh are intermediate-side; its n side (hidden) stays whole. The down
    bias lives on the unsliced output side, so only the worker holding [0, ...) applies it
    -- every worker adding it would count it (W - 1) extra times in the host's sum."""
    g, u, d = projs

    def proj(ms, n_sliced):
        ts, suhs, svhs, biases = ms
        out = []
        for i in range(len(ts)):
            tr = ts[i]
            t = tr[:, lo // 16:hi // 16, :] if n_sliced else tr[lo // 16:hi // 16, :, :]
            if swizzled:
                t = swizzle(t)
            out.append((
                t,
                suhs[i] if n_sliced else suhs[i][lo:hi],
                svhs[i][lo:hi] if n_sliced else svhs[i],
                (biases[i][lo:hi] if n_sliced else biases[i]) if biases[i] is not None else None,
            ))
        return out

    # The down projection is sliced along k, so its bias (n-side, hidden) goes to worker 0 only
    dg, du, dd = proj(g, True), proj(u, True), proj(d, False)
    if lo != 0:
        dd = [(tr, suh, svh, None) for tr, suh, svh, _ in dd]
    return dg, du, dd


def make_handle(reg, activation, swizzled):
    gs, us, ds = reg
    return ext.exl3_moe_cpu_make_layer(
        [t[0] for t in gs], [t[1] for t in gs], [t[2] for t in gs],
        [t[0] for t in us], [t[1] for t in us], [t[2] for t in us],
        [t[0] for t in ds], [t[1] for t in ds], [t[2] for t in ds],
        [t[3] for t in gs if t[3] is not None],
        [t[3] for t in us if t[3] is not None],
        [t[3] for t in ds if t[3] is not None],
        activation, 1.0, int(swizzled))


@pytest.mark.parametrize("activation", [0, 1, 2, 3])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("workers", [1, 2, 3, 4])
def test_worker_partials_sum_to_reference(activation, bias, workers, cpu_runtime):
    gen = torch.Generator().manual_seed(174)
    hidden, intermediate, experts = 384, 512, 3
    gated = activation != 2
    swizzled = len(swizzle_layouts()) > 1

    def native(k, n):
        ts, us, vs, bs = [], [], [], []
        for e in range(experts):
            bits = (1, 3, 8)[e]
            ts.append(torch.randint(-32768, 32767, (k // 16, n // 16, 16 * bits),
                                    dtype = torch.int16, generator = gen))
            us.append((torch.randn(k, generator = gen) * 0.015).half())
            vs.append(torch.randn(n, generator = gen).half())
            bs.append((torch.randn(n, generator = gen) * 0.1).half() if bias else None)
        return ts, us, vs, bs

    projs = (native(hidden, intermediate) if gated else ([], [], [], []),
             native(hidden, intermediate),
             native(intermediate, hidden))

    x = torch.randn(3, hidden, generator = gen).half()
    weights = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.9], [0.7, 0.2, 0.1]],
                           dtype = torch.half)
    handles = []
    try:
        # Full-expert reference (one worker holding the whole intermediate dim) plus one
        # registration per worker slice; identical tokens go through every registration
        ref_handle = make_handle(make_registration(projs, 0, intermediate, swizzled),
                                 activation, swizzled)
        handles.append(ref_handle)
        worker_handles = [make_handle(make_registration(projs, lo, hi, swizzled),
                                      activation, swizzled)
                          for lo, hi in _slice_ranges(intermediate, workers)]
        handles += worker_handles

        for sel_list in ([0, 1, 2], [-1, 1, -1], [-1, -1, -1], [2, 2, 0]):
            sel = torch.tensor([sel_list] * 3, dtype = torch.long)
            ref = torch.empty(3, hidden)
            ext.exl3_moe_cpu_forward(ref_handle, x, sel, weights, ref, 4)
            for threads in (1, 16):
                acc = torch.zeros(3, hidden)
                for h in worker_handles:
                    part = torch.empty(3, hidden)
                    ext.exl3_moe_cpu_forward(h, x, sel, weights, part, threads)
                    acc += part
                # Split vs. full differs only by output rounding: each worker's partial is
                # rounded independently on its path through the kernels, so summing W
                # partials reassociates the rounding (~1 ULP of the partial magnitude; up
                # to ~4e-3 here, and ISA-dependent on machines without AMX). Structural
                # slicing errors (misaligned block, double-counted output bias) produce
                # errors of the output's own magnitude -- 0.34 per worker for the bias
                # case -- so this tolerance still separates them by >30x.
                torch.testing.assert_close(acc, ref, rtol = 2e-3, atol = 1e-2)
    finally:
        for h in handles:
            ext.exl3_moe_cpu_free_layer(h)


def test_slice_ranges_rejects_partial_blocks():
    with pytest.raises(ValueError, match="not a multiple"):
        _slice_ranges(513, 1)


def test_slice_expert_proj():
    """The child loader's per-worker registration slice: trellis and the INTERMEDIATE-side
    scale vector are cut to [lo, hi), the hidden-side scale stays whole, and the output-side
    bias follows its projection (sliced for gate/up whose output side is the intermediate,
    whole for down -- and dropped entirely for workers > 0, whose host-side sum would count
    it (W - 1) extra times). Marker values (every vector the arange) make any missing or
    misplaced cut an exact-compare failure rather than a shape accident."""
    from exllamav3.model.moe_cpu_host import _slice_expert_proj

    hidden, interm, B = 2048, 1792, 64   # 16 * 4-bit trellis
    for (tr_shape, suh_len, svh_len, n_sliced) in [
        ((hidden // 16, interm // 16, B), hidden, interm, True),   # gate/up: n side sliced
        ((interm // 16, hidden // 16, B), interm, hidden, False),  # down: k side sliced
    ]:
        tr = torch.randint(0, 255, tr_shape, dtype = torch.uint8)
        suh = torch.arange(suh_len, dtype = torch.float).half()
        svh = torch.arange(svh_len, dtype = torch.float).half() * 2
        # Output-side bias: intermediate-length for gate/up, hidden-length for down
        bias = torch.arange(interm if n_sliced else hidden,
                            dtype = torch.float).half() * 3
        ranges = [(0, interm)] + _slice_ranges(interm, 4)
        for w, (lo, hi) in enumerate(ranges):
            t = _slice_expert_proj((tr, suh, svh, bias), n_sliced, lo, hi,
                                   drop_bias = (w != 0) and not n_sliced)
            t_tr, t_suh, t_svh, t_bias = t
            b16, c = slice(lo // 16, hi // 16), slice(lo, hi)
            torch.testing.assert_close(t_tr, tr[:, b16, :] if n_sliced else tr[b16, :, :])
            torch.testing.assert_close(t_suh, suh if n_sliced else suh[c])
            torch.testing.assert_close(t_svh, svh[c] if n_sliced else svh)
            if n_sliced:
                torch.testing.assert_close(t_bias, bias[c])
            else:
                assert (t_bias is None) == (w != 0)
                if w == 0:
                    torch.testing.assert_close(t_bias, bias)
