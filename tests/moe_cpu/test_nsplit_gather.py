"""
Multi-worker N-split: the copy plan that assembles a full expert block from the workers' slices.

Each worker stores its slice of every CPU expert (gate/up cut along n, down along k, at 128-column
blocks) packed in the CPU tier's layout, back to back as [gate | up | down]. Streamed prefill, the
promote-on-miss transfers and the verify path need the block a single worker would hold instead,
so moe_cpu_host._nsplit_pieces describes where each worker's slice lands in it. Checked here on
the host, byte for byte, against _copy_repacked over every packed layout the tiers use (native,
band groups, integer planar dwords, fractional halfword planes), for uneven slices and gateless
layers, together with the inverse scatter.
"""

import pytest
import torch

from exllamav3.model.moe_cpu_host import (_copy_repacked, _gather_plan, _nsplit_pieces, _scatter_plan,
                                           _slice_expert_proj, _slice_ranges)

pytestmark = pytest.mark.nogpu


def run_plan(plan, dst, src):
    """Host emulation of cuda_memcpy2d_async_batch over two byte tensors, addresses relative to
    each tensor's start"""
    for d, dpitch, s, spitch, width, height in plan.tolist():
        for r in range(height):
            dst[d + r * dpitch: d + r * dpitch + width] = src[s + r * spitch: s + r * spitch + width]


def packed(t, group, planar):
    out = torch.empty_like(t)
    _copy_repacked(out, t.contiguous(), group, planar)
    return out.view(torch.uint8).reshape(-1)


# (bits, group, planar): native, band-8, band-2 integer planar, band-2 fractional halfword planes
LAYOUTS = [(4, 0, 0), (4, 8, 0), (3, 2, 1), (2.5, 2, 1), (1.5, 2, 1), (8, 0, 0)]


@pytest.mark.parametrize("bits,group,planar", LAYOUTS)
@pytest.mark.parametrize("workers", [2, 3, 4])
@pytest.mark.parametrize("gated", [True, False])
def test_gather_reassembles_full_block(bits, group, planar, workers, gated):
    gen = torch.Generator().manual_seed(17)
    hidden, interm = 256, 640           # 5 blocks: uneven slices for 2..4 workers
    ps = int(16 * bits)
    dims = {"u": (hidden, interm, bits), "d": (interm, hidden, bits)}
    if gated:
        dims["g"] = (hidden, interm, bits)
    layout = tuple((group, planar) if nm in dims else (0, 0) for nm in ("g", "u", "d"))
    native = {nm: torch.randint(-32768, 32767, (k // 16, n // 16, ps), dtype = torch.int16, generator = gen)
              for nm, (k, n, _) in dims.items()}
    names = [nm for nm in ("g", "u", "d") if nm in dims]
    full = torch.cat([packed(native[nm], group, planar) for nm in names])

    worker_bytes, pieces = _nsplit_pieces(dims, layout, workers)
    blocks = []
    for lo, hi in _slice_ranges(interm, workers):
        parts = []
        for nm in names:
            k, n, _ = dims[nm]
            tr = _slice_expert_proj((native[nm], torch.zeros(k), torch.zeros(n), None), nm != "d", lo, hi)[0]
            parts.append(packed(tr, group, planar))
        blocks.append(torch.cat(parts))
    assert [b.numel() for b in blocks] == worker_bytes
    assert sum(worker_bytes) == full.numel()

    # Gather: worker blocks back to back in one source buffer, two experts' worth
    src = torch.cat(blocks * 2)
    starts = [sum(worker_bytes[:w]) for w in range(workers)]
    total = sum(worker_bytes)
    dst = torch.zeros(2 * full.numel(), dtype = torch.uint8)
    plan = _gather_plan(pieces, [0, full.numel()], [[s for s in starts], [total + s for s in starts]])
    run_plan(plan, dst, src)
    assert torch.equal(dst[:full.numel()], full)
    assert torch.equal(dst[full.numel():], full)

    # Scatter: the full block back into each worker's slice
    back = torch.zeros(total, dtype = torch.uint8)
    run_plan(_scatter_plan(pieces, 0, starts), back, full)
    assert torch.equal(back, torch.cat(blocks))


def test_slice_ranges():
    assert _slice_ranges(640, 2) == [(0, 384), (384, 640)]
    assert _slice_ranges(512, 4) == [(0, 128), (128, 256), (256, 384), (384, 512)]
    with pytest.raises(ValueError, match = "not a multiple"):
        _slice_ranges(513, 1)
    with pytest.raises(ValueError, match = "do not fit"):
        _slice_ranges(256, 3)
