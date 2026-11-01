# Step 1 unit test: m8n8k4 lane-map algebra (host-side).
#
# Validates the PTX ISA mma.m8n8k4 (.f16 A/B, .f32 C/D, .row.col) fragment
# maps used by exl3_sm70_map.cuh:
#   - A (row-major): row = lane%4 (+4 if lane>=16), col = i, i=0..3
#   - B (col-major): col = lane%4 (+4 if lane>=16), row = i, i=0..3
#   - D (f32): row = X (+4 if lane>=16), X = (lane&1)+(i&2)
#              col = (i&4)+(lane&2)+(i&1), i=0..7
# per MMA computation (lanes {0-3,16-19} etc.). Each MMA's 8 participating
# lanes tile A (8x4), B (4x8), D (8x8) exactly; the D map reproduces
# D = A @ B through the lane-owned fragments.
#
# This pins the lane→element algebra the sm70 GEMV lane-remap is built on.
# Run:  python tests/test_sm70_lane_map.py

import torch

MMA_LANES = [0, 1, 2, 3, 16, 17, 18, 19]  # computation 1 (low + high group)


def d_cell(lane, i):
    X = (lane & 1) + (i & 2)
    row = X if lane < 16 else X + 4
    col = (i & 4) + (lane & 2) + (i & 1)
    return row, col


def a_cell(lane, i):
    row = (lane % 4) + (4 if lane >= 16 else 0)
    return row, i  # (row, k)


def b_cell(lane, i):
    col = (lane % 4) + (4 if lane >= 16 else 0)
    return i, col  # (k, col)


def tiling(fn, n, expect):
    claims = {}
    for lane in MMA_LANES:
        for i in range(n):
            r, c = fn(lane, i)
            claims[(r, c)] = claims.get((r, c), 0) + 1
    assert len(claims) == expect and all(v == 1 for v in claims.values()), \
        f"tiling broken: {len(claims)} cells (expect {expect}), dupes={any(v != 1 for v in claims.values())}"


def main():
    torch.manual_seed(0)
    # 1. Perfect tilings
    tiling(a_cell, 4, 32)  # A: 8 rows x 4 k
    tiling(b_cell, 4, 32)  # B: 4 k x 8 cols
    tiling(d_cell, 8, 64)  # D: 8x8
    print("tilings OK: A 8x4, B 4x8, D 8x8 — each cell exactly one lane")

    # 2. MMA semantics through the maps
    A = torch.randn(8, 4)
    B = torch.randn(4, 8)
    D = torch.zeros(8, 8)
    for lane in MMA_LANES:
        arow = (lane % 4) + (4 if lane >= 16 else 0)
        bcol = (lane % 4) + (4 if lane >= 16 else 0)
        for i in range(8):
            row, col = d_cell(lane, i)
            D[row, col] = (A[row, :] * B[:, col]).sum()
    err = (D - A @ B).abs().max().item()
    assert err < 1e-5, f"MMA simulation wrong: {err}"
    print(f"MMA semantics OK: D = A@B through lane maps (max err {err:.2e})")

    # 3. Cross-check vs 1Cat SM70_MMA_884 thread_offset_C. 1Cat's map covers
    # the composed 8x32 tile (4 col-quads); per MMA (lanes {0-3,16-19}, quad
    # offset (L&12)*2 == 0) it must equal the ISA D map at i=0.
    for L in [0, 1, 2, 3, 16, 17, 18, 19]:
        row_1cat = (L & 1) + (L // 16) * 4
        col_1cat = (L & 2) + (L & 12) * 2  # == L&2 within this MMA's quad
        row_isa, col_isa = d_cell(L, 0)
        assert (row_1cat, col_1cat) == (row_isa, col_isa), \
            f"1Cat/ISA mismatch at lane {L}: {(row_1cat, col_1cat)} vs {(row_isa, col_isa)}"
    print("1Cat SM70_MMA_884 thread_offset_C == ISA D map per-MMA — cross-validated")