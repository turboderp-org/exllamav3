"""
Multi-worker N-split, streamed prefill: ext.moe_unswizzle_trellis_nsplit reassembles every full native
expert projection from the workers' packed slices, exactly as streamed prefill lays them out in the VRAM
staging slot -- each expert's worker blocks back to back (pinned arena) or each worker's blocks of the whole
batch back to back (staged) -- for every packed layout (native, band-8, band-2 integer planar, fractional
halfword planes), uneven slices and gateless layers. The result must equal the native projections bitwise.
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model.moe_cpu_host import _copy_repacked, _nsplit_pieces, _slice_expert_proj, _slice_ranges

LAYOUTS = [(4, 0, 0), (4, 8, 0), (3, 2, 1), (2.5, 2, 1), (1.5, 2, 1), (8, 0, 0)]


def packed(t, group, planar):
    out = torch.empty_like(t)
    _copy_repacked(out, t.contiguous(), group, planar)
    return out.view(torch.uint8).reshape(-1)


@pytest.mark.parametrize("K,group,planar", LAYOUTS, ids = [f"K{K}-g{g}{'-p' if p else ''}" for K, g, p in LAYOUTS])
@pytest.mark.parametrize("workers", [2, 3, 4])
@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("staged", [False, True])
def test_restore_reassembles_full_native_experts(device, K, group, planar, workers, gated, staged):
    gen = torch.Generator().manual_seed(int(K * 10) + 7 * workers + gated + 2 * staged)
    hidden, interm, E = 256, 640, 3          # 5 blocks of 128: uneven slices
    ps = int(16 * K)
    dims = {"u": (hidden, interm, K), "d": (interm, hidden, K)}
    if gated:
        dims["g"] = (hidden, interm, K)
    names = [nm for nm in ("g", "u", "d") if nm in dims]
    layout = tuple((group, planar) if nm in dims else (0, 0) for nm in ("g", "u", "d"))
    native = [{nm: torch.randint(-32768, 32767, (k // 16, n // 16, ps), dtype = torch.int16, generator = gen)
               for nm, (k, n, _) in dims.items()} for _ in range(E)]
    wbytes, pieces = _nsplit_pieces(dims, layout, workers)
    ranges = _slice_ranges(interm, workers)
    blocks = [[torch.cat([packed(_slice_expert_proj((native[e][nm], torch.zeros(dims[nm][0]), torch.zeros(dims[nm][1]),
                                                     None), nm != "d", lo, hi)[0], group, planar) for nm in names])
               for lo, hi in ranges] for e in range(E)]
    exp_b = sum(wbytes)
    w_cum = [sum(wbytes[:w]) for w in range(workers)]
    if staged:
        src = torch.cat([blocks[e][w] for w in range(workers) for e in range(E)])
        w_off, w_stride = [E * o for o in w_cum], wbytes
    else:
        src = torch.cat([blocks[e][w] for e in range(E) for w in range(workers)])
        w_off, w_stride = w_cum, [exp_b] * workers
    src = src.view(torch.int16).to(device)
    dst = torch.zeros(E * exp_b // 2, dtype = torch.int16, device = device)
    split = [lo // 16 for lo, _ in ranges] + [ranges[-1][1] // 16]
    f_off = 0
    for i, nm in enumerate(names):
        k, n, _ = dims[nm]
        ext.moe_unswizzle_trellis_nsplit(src, dst, E, exp_b, f_off, k // 16, n // 16, K, group, planar,
                                         nm == "d", split, w_off, w_stride, [pieces[w][i][0] for w in range(workers)])
        f_off += (k // 16) * (n // 16) * ps * 2
    want = torch.cat([native[e][nm].reshape(-1) for e in range(E) for nm in names]).to(device)
    assert torch.equal(dst, want)
