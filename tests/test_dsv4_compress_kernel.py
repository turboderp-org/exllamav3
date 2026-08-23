"""
Unit tests for the fused DSv4 compressor kernels (exllamav3_ext/dsv4_compress.cu) against
the torch DSV4Compressor path. The fused path (used by the cached forward) writes emitted
entries straight into pool tensors and keeps ring/snapshot state; the torch path with
DSV4CompressorState is the reference. Covers CSA-shaped (overlapping, m = 4), indexer-shaped
(overlapping, narrow) and HCA-shaped (non-overlapping, m = 128) compressors, uneven chunk
schedules, and bitwise chunked-vs-whole for the fused path itself.

    python tests/test_dsv4_compress_kernel.py --device cuda:1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3.ext import exllamav3_ext as ext


def torch_reference(kv_rows, gate_rows, ape, norm_w, eps, inv_freq, m, overlapping, hd):
    """All-at-once torch reference over the full row history, replicating
    DSV4Compressor.forward stateless math + norm + rope. Returns (T, hd) fp32."""
    total = kv_rows.shape[0]
    nw = total // m
    if nw == 0:
        return torch.zeros((0, hd), dtype = torch.float, device = kv_rows.device)
    W = kv_rows.shape[-1]
    kv = kv_rows[:nw * m].float().view(nw, m, W)
    gate = gate_rows[:nw * m].float().view(nw, m, W) + ape.unsqueeze(0)
    if overlapping:
        new_kv = kv.new_zeros((nw, 2 * m, hd))
        new_gate = gate.new_full((nw, 2 * m, hd), -float("inf"))
        new_kv[:, m:] = kv[..., hd:]
        new_gate[:, m:] = gate[..., hd:]
        if nw > 1:
            new_kv[1:, :m] = kv[:-1, :, :hd]
            new_gate[1:, :m] = gate[:-1, :, :hd]
        kv, gate = new_kv, new_gate
    comp = (kv * gate.softmax(dim = 1)).sum(dim = 1)
    comp = comp * torch.rsqrt(comp.square().mean(-1, keepdim = True) + eps) * norm_w.float()
    rd = inv_freq.shape[0] * 2
    wpos = torch.arange(nw, device = kv.device).float() * m
    theta = wpos[:, None] * inv_freq[None, :]
    cos, sin = theta.cos(), theta.sin()
    rope = comp[:, hd - rd:]
    e, o = rope[:, 0::2], rope[:, 1::2]
    comp[:, hd - rd:] = torch.stack((e * cos - o * sin, o * cos + e * sin), dim = -1).flatten(-2)
    return comp


def run_fused(kv_rows, gate_rows, ape, norm_w, eps, inv_freq, m, overlapping, hd, chunks,
              buf_rows, ovl_depth, split):
    """Drive ext.dsv4_compress chunk by chunk with fresh ring state. Returns pool (T, hd)."""
    device = kv_rows.device
    total = kv_rows.shape[0]
    W = kv_rows.shape[-1]
    cap = total // m + 8
    ring_kv = torch.zeros((buf_rows, W), dtype = torch.half, device = device)
    ring_gate = torch.zeros((buf_rows, W), dtype = torch.half, device = device)
    ovl = torch.zeros((ovl_depth, 2, m, hd), dtype = torch.float, device = device) \
        if overlapping else None
    if split:
        wa = hd - inv_freq.shape[0] * 2
        dest_a = torch.zeros((cap, wa), dtype = torch.half, device = device)
        dest_b = torch.zeros((cap, hd - wa), dtype = torch.half, device = device)
    else:
        dest_a = torch.zeros((cap, hd), dtype = torch.half, device = device)
        dest_b = None
    pos = 0
    for c in chunks:
        ext.dsv4_compress(
            kv_rows[pos:pos + c], gate_rows[pos:pos + c], ring_kv, ring_gate, ovl,
            ape, norm_w, eps, inv_freq, dest_a, dest_b, pos, None, m)
        pos += c
    nw = total // m
    out = torch.cat([dest_a[:nw], dest_b[:nw]], dim = -1) if split else dest_a[:nw].clone()
    return out


def check(device, tag, hd, W, m, overlapping, total, chunks, split, seed, tol = 2e-2):
    torch.manual_seed(seed)
    kv_rows = (torch.randn((total, W), device = device) * 0.7).half()
    gate_rows = (torch.randn((total, W), device = device) * 1.5).half()
    ape = (torch.randn((m, W), device = device) * 0.8).float()
    norm_w = (torch.randn((hd,), device = device) * 0.3 + 1.0).half()
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, device = device).float() / 64))
    if hd < 64:
        inv_freq = inv_freq[:hd // 2]
    eps = 1e-6

    ref = torch_reference(kv_rows, gate_rows, ape, norm_w, eps, inv_freq, m, overlapping, hd)
    got = run_fused(kv_rows, gate_rows, ape, norm_w, eps, inv_freq, m, overlapping, hd,
                    chunks, buf_rows = 256 + m, ovl_depth = 256 // m + 2, split = split)
    got_whole = run_fused(kv_rows, gate_rows, ape, norm_w, eps, inv_freq, m, overlapping, hd,
                          [total], buf_rows = max(256 + m, total + m),
                          ovl_depth = max(256 // m + 2, total // m + 2), split = split)

    assert got.shape[0] == ref.shape[0], f"{tag}: {got.shape[0]} windows vs ref {ref.shape[0]}"
    scale = ref.abs().max().item() + 1e-6
    err = (got.float() - ref).abs().max().item() / scale
    bitwise = torch.equal(got, got_whole) if len(chunks) > 1 else True
    ok = err < tol and bitwise and got.shape[0] > 0
    print(f"  {'PASS' if ok else 'FAIL'} {tag}: windows {got.shape[0]} rel {err:.2e} "
          f"chunked==whole {bitwise}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default = "cuda:0")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    ok = True

    #                      tag                hd   W     m   ovl    total  chunks
    cases = [
        ("csa split",      512, 1024, 4,   True,  317, [37, 1, 1, 128, 150], True),
        ("csa whole",      512, 1024, 4,   True,  64,  [64], True),
        ("csa singles",    512, 1024, 4,   True,  23,  [1] * 23, True),
        ("csa tiny-first", 512, 1024, 4,   True,  9,   [2, 1, 6], True),
        ("idx",            128, 256,  4,   True,  317, [37, 1, 1, 128, 150], False),
        ("idx singles",    128, 256,  4,   True,  17,  [1] * 17, False),
        ("hca split",      512, 512,  128, False, 517, [200, 56, 1, 260], True),
        ("hca whole",      512, 512,  128, False, 384, [384], True),
        ("big chunk",      512, 1024, 4,   True,  600, [600], True),        # > buf_rows
        ("big + tail",     512, 1024, 4,   True,  700, [650, 50], True),    # store clamp
    ]
    for i, (tag, hd, W, m, ovl, total, chunks, split) in enumerate(cases):
        ok &= check(device, tag, hd, W, m, ovl, total, chunks, split, seed = 400 + i)

    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
