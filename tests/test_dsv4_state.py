"""
Stateful-path consistency tests for DeepSeek-V4 components: the chunked (cached) compressor
must reproduce the single-shot result exactly -- sub-window buffering, Ca-overlap carry and
entry positioning included. Uses the tiny random checkpoint from compare_deepseek_v4_hf_.

    python tests/test_dsv4_state.py --device cuda:1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from compare_deepseek_v4_hf_ import make_checkpoint, TINY
from exllamav3 import Config, Model
from exllamav3.modules.dsv4 import DSV4CompressorState


def chunk_splits(seq, pattern):
    out, a = [], 0
    for p in pattern:
        b = min(a + p, seq)
        if b > a:
            out.append((a, b))
        a = b
    if a < seq:
        out.append((a, seq))
    return out


def check_compressor(comp, inv_freq, x, pattern, tag, tol = 2e-3):
    # tol: the wkv/wgate projections are fp16 GEMMs whose tiling depends on the row count, so
    # chunked and whole runs see ulp-level projection differences (same mechanism documented
    # for the MLA chunked-vs-whole tests). The buffering logic itself is exact: small-seq
    # cases where cuBLAS uses a single tile compare bitwise (0.0).
    with torch.inference_mode():
        whole_state = DSV4CompressorState()
        whole = comp.forward(x, {}, inv_freq, whole_state)
        chunk_state = DSV4CompressorState()
        parts = []
        for a, b in chunk_splits(x.shape[1], pattern):
            e = comp.forward(x[:, a:b], {}, inv_freq, chunk_state)
            if e.shape[1]:
                parts.append(e)
        chunked = torch.cat(parts, dim = 1) if parts else whole[:, :0]
        stateless = comp.forward(x, {}, inv_freq, None)
    ok = True
    if whole.shape != chunked.shape:
        print(f"  FAIL {tag}: shape {whole.shape} vs chunked {chunked.shape}")
        return False
    err = (whole.float() - chunked.float()).abs().max().item() if whole.numel() else 0.0
    ok &= err < tol
    print(f"  {'PASS' if err < tol else 'FAIL'} {tag} chunked-vs-whole ({pattern}): maxerr {err:.2e}, "
          f"{whole.shape[1]} entries, count {chunk_state.entry_count}")
    # stateless == stateful-from-empty on the shared prefix of complete windows
    n = stateless.shape[1]
    err2 = (whole[:, :n].float() - stateless.float()).abs().max().item() if n else 0.0
    ok &= err2 < tol
    if err2 >= tol:
        print(f"  FAIL {tag} stateless-vs-stateful prefix: {err2:.2e}")
    ok &= chunk_state.entry_count == whole.shape[1]
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default = "cuda:0")
    args = p.parse_args()
    device = args.device
    out_dir = "/tmp/dsv4_tiny_state"
    make_checkpoint(out_dir, seed = 11)

    config = Config.from_directory(out_dir)
    model = Model.from_config(config)
    model.load(device)

    # layers: [embed, expand, block0(sliding), block1(sliding), block2(csa), block3(hca), ...]
    csa = model.modules[2 + 2].attn
    hca = model.modules[2 + 3].attn
    assert csa.layer_type == "csa" and hca.layer_type == "hca"

    torch.manual_seed(5)
    ok = True
    for seq, pattern in [
        (315, [100, 107, 108]),      # uneven chunks, non-aligned ends
        (313, [1] * 9 + [304]),      # decode-like single-token steps then bulk
        (64, [3, 5, 7, 49]),         # sub-window chunks (buffer must carry)
        (7, [2, 2, 3]),              # shorter than HCA window entirely
    ]:
        x = torch.randn((1, seq, TINY["hidden_size"]), dtype = torch.half, device = device)
        ok &= check_compressor(csa.compressor, csa.inv_freq_compress, x, pattern, f"csa m4 seq{seq}")
        ok &= check_compressor(csa.indexer, csa.inv_freq_compress, x, pattern, f"idx m4 seq{seq}")
        ok &= check_compressor(hca.compressor, hca.inv_freq_compress, x, pattern, f"hca m8 seq{seq}")

    print("ALL PASS" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
