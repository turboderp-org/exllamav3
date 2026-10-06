"""
Streamed single-row decode for CPU-offloaded MoE layers (model/moe_stream_decode.py,
EXL3_MOE_STREAM_DECODE): at decode, the selected experts' weight blocks are read straight out of
the CPU worker's pinned arena (EXL3_MOE_PINNED_ARENA, mapped into the device address space) into
a VRAM staging buffer by the moe_stream_gather kernel, and the resident fused MoE kernel runs
over them through pointer tables.

Kernel tests (no model needed; the arena is emulated with pinned host tensors, which get the same
zero-copy device mapping as the real memfd arena):
  - staged bytes == the host source blocks, with the experts spread over two arena chunks at
    64-byte aligned offsets with gaps; a slot selecting expert -1 (the CPU-share mask) is left
    untouched; the per-expert aux pointer tables are gathered into per-slot tables in the same
    launch
  - band-swizzled arena layout (the AVX-512 CPU tiers, EXL3_MOE_CPU_SWIZZLE): a swizzled
    [gate | up | down] block gathered verbatim and restored with moe_unswizzle_trellis equals the
    native tensors, per projection, K8 (never swizzled) included. This is the path a host with
    an AVX-512 CPU takes at every decode step.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3.ext import exllamav3_ext as ext

device = torch.device("cuda:0")
gen = torch.Generator().manual_seed(1234)


def _rand_i16(n):
    return torch.randint(-32768, 32767, (n,), dtype = torch.int16, generator = gen)


def _pinned_chunk(data):
    """A pinned copy of `data` and its CUDA alias (what _attach_chunk + pinned_cuda_view produce
    for a real arena chunk). The pinned tensor must outlive the alias."""
    host = data.pin_memory()
    assert host.is_pinned()
    return host, ext.pinned_cuda_view(host, device.index)


def _arena(blocks, chunks = 2, gap = 192):
    """Emulated arena: `blocks` (int16 tensors, one per expert) laid out round-robin over
    `chunks` pinned chunks at 64-byte aligned offsets with random filler between neighbours.
    Returns (host chunks, aliases, blk_chunk, blk_off, device tables)."""
    per_chunk = [[] for _ in range(chunks)]
    blk_chunk, blk_off = [], []
    for e, b in enumerate(blocks):
        c = e % chunks
        off = 64 + sum(x.numel() * 2 + gap for x in per_chunk[c])
        assert off % 64 == 0 and (b.numel() * 2) % 64 == 0
        blk_chunk.append(c)
        blk_off.append(off)
        per_chunk[c].append(b)
    host, alias = [], []
    for c in range(chunks):
        n = 64 + sum(x.numel() * 2 + gap for x in per_chunk[c])
        data = _rand_i16(n // 2)
        for e in range(len(blocks)):
            if blk_chunk[e] == c:
                o = blk_off[e] // 2
                data[o : o + blocks[e].numel()] = blocks[e]
        h, a = _pinned_chunk(data)
        host.append(h)
        alias.append(a)
    tables = (
        torch.tensor([a.data_ptr() for a in alias], dtype = torch.long, device = device),
        torch.tensor(blk_chunk, dtype = torch.int32, device = device),
        torch.tensor(blk_off, dtype = torch.long, device = device),
    )
    return host, alias, blk_chunk, blk_off, tables


def test_gather_matches_host():
    E, exp_b = 12, 3 * 40960                          # three K3 512x640 projections' worth of bytes
    blocks = [_rand_i16(exp_b // 2) for _ in range(E)]
    host, alias, bc, bo, (chunk_base, blk_chunk, blk_off) = _arena(blocks)
    sel_h = [7, 0, -1, 11, 4]                          # slot 2: nothing selected
    sel = torch.tensor(sel_h, dtype = torch.long, device = device)
    slots, n = len(sel_h), exp_b // 2
    stage = torch.full((slots * n,), 12345, dtype = torch.int16, device = device)
    rows = 6
    aux_ptrs = torch.randint(1 << 20, 1 << 40, (rows, E), dtype = torch.long, generator = gen).to(device)
    aux_out = torch.zeros((rows, slots), dtype = torch.long, device = device)
    ext.moe_stream_gather(stage, chunk_base, blk_chunk, blk_off, sel, exp_b, aux_ptrs, aux_out)
    torch.cuda.synchronize(device)
    staged, aux_out = stage.view(slots, n).cpu(), aux_out.cpu()
    for i, e in enumerate(sel_h):
        if e < 0:
            assert bool((staged[i] == 12345).all()), "a -1 slot must be left untouched"
            assert bool((aux_out[:, i] == 0).all()), "a -1 slot must not receive aux pointers"
            continue
        assert torch.equal(staged[i], blocks[e]), f"slot {i} (expert {e}) differs from the host block"
        assert torch.equal(staged[i], host[bc[e]][bo[e] // 2 : bo[e] // 2 + n])
        assert torch.equal(aux_out[:, i], aux_ptrs[:, e].cpu()), f"aux table entry for slot {i}"


def _swizzle(t, K):
    """The arena's band-contiguous layout of a native [k/16, n/16, 16K] trellis (the repack
    _HugeArena.rehome applies on the AVX-512 tiers, as in test_moe_cpu_tiers_); K8 stays native."""
    if K == 8:
        return t
    tk, tn, ps = t.shape
    return t.view(tk, tn // 8, 8, ps).permute(1, 0, 2, 3).contiguous().view(tk, tn, ps)


def test_gather_unswizzle_roundtrip():
    hid, inter, E = 512, 640, 3
    for K in (1, 3, 8):
        dims = dict(g = (hid, inter), u = (hid, inter), d = (inter, hid))
        native = [{p: _rand_i16((k // 16) * (n // 16) * 16 * K).view(k // 16, n // 16, 16 * K)
                   for p, (k, n) in dims.items()} for _ in range(E)]
        pb = {p: (k // 16) * (n // 16) * 16 * K * 2 for p, (k, n) in dims.items()}
        offs = dict(g = 0, u = pb["g"], d = pb["g"] + pb["u"])
        exp_b = sum(pb.values())
        blocks = [torch.cat([_swizzle(nt[p], K).reshape(-1) for p in ("g", "u", "d")]) for nt in native]
        host, alias, bc, bo, (chunk_base, blk_chunk, blk_off) = _arena(blocks, chunks = 1)
        sel_h = [2, 0, 1]
        sel = torch.tensor(sel_h, dtype = torch.long, device = device)
        raw = torch.zeros(len(sel_h) * exp_b // 2, dtype = torch.int16, device = device)
        nat = torch.zeros_like(raw)
        ext.moe_stream_gather(raw, chunk_base, blk_chunk, blk_off, sel, exp_b, None, None)
        for p, (k, n) in dims.items():
            ext.moe_unswizzle_trellis(raw, nat, len(sel_h), exp_b, offs[p], k // 16, n // 16, K, K != 8)
        torch.cuda.synchronize(device)
        raw_h, nat_h = raw.view(len(sel_h), -1).cpu(), nat.view(len(sel_h), -1).cpu()
        for i, e in enumerate(sel_h):
            assert torch.equal(raw_h[i], blocks[e]), f"K{K}: gathered block {e} differs from the arena"
            for p in ("g", "u", "d"):
                o = offs[p] // 2
                assert torch.equal(nat_h[i, o : o + pb[p] // 2], native[e][p].reshape(-1)), \
                    f"K{K}: {p} of expert {e} not restored to the native tile order"


def bandwidth_report(exp_b = 1843200, slots = 6, layers = 48, reps = 3):
    """Sustained gather rate for a Qwen3.8-Flash-Next-sized decode step (report, not a test)."""
    E = 32
    blocks = [_rand_i16(exp_b // 2) for _ in range(E)]
    host, alias, bc, bo, (chunk_base, blk_chunk, blk_off) = _arena(blocks)
    stage = torch.empty(slots * exp_b // 2, dtype = torch.int16, device = device)
    sels = [torch.randperm(E, generator = gen)[:slots].to(device) for _ in range(layers)]
    for _ in range(2):
        ext.moe_stream_gather(stage, chunk_base, blk_chunk, blk_off, sels[0], exp_b, None, None)
    torch.cuda.synchronize(device)
    best = 0.0
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        e0.record()
        for s in sels:
            ext.moe_stream_gather(stage, chunk_base, blk_chunk, blk_off, s, exp_b, None, None)
        e1.record()
        e1.synchronize()
        best = max(best, layers * slots * exp_b / e0.elapsed_time(e1) / 1e6)
    print(f"gather: {slots} x {exp_b >> 10} KiB experts x {layers} layers: {best:.1f} GB/s, "
          f"{layers * slots * exp_b / best / 1e6:.1f} ms per token")


if __name__ == "__main__":
    test_gather_matches_host(); print("PASS test_gather_matches_host")
    test_gather_unswizzle_roundtrip(); print("PASS test_gather_unswizzle_roundtrip")
    bandwidth_report()
