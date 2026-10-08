"""
Tiled DSA top-k (ext.dsa_topk_tile + ext.dsa_topk_merge_tiles, as modules/qsa_indexer.py drives them) and the
single-pass ext.dsa_topk against an exact NumPy reference of the selection AND its emission order.

Total order: fp16 scores ranked by their 16-bit ordered key (sign-flip radix key: -0 ranks below +0, -inf is never
selected), ties broken by ascending index. Emission order of a row with n finite entries:
  n < k:  every finite index, ascending;
  n >= k: with v* the k-th largest key, the indices with key > v* ascending, then the first k - #(> v*) indices with
          key == v* ascending; -1 padding to k_pad.
dsa_topk_tile writes that order for its tile (indices offset by idx_offset, scores alongside, count) into one
workspace slot and touches no other slot. dsa_topk_merge_tiles over slots that hold tiles of ascending index ranges
(or a running set from a previous merge in slot 0, written through strided slot views with out_scr / out_cnt) must
produce exactly what dsa_topk produces on the whole row -- bitwise, order included (the qsa_indexer docstring and the
kernel comment claim this) -- which is also checked directly against dsa_topk.
"""
import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

NEG_INF_KEY = 0x03ff
MAX_G = 32


def keys_of(scores: torch.Tensor) -> np.ndarray:
    u = scores.cpu().contiguous().view(torch.int16).numpy().astype(np.int64) & 0xffff
    return np.where(u & 0x8000, (~u) & 0xffff, u | 0x8000)


def ref_order(key_row: np.ndarray, k: int) -> list[int]:
    fin = np.nonzero(key_row > NEG_INF_KEY)[0]
    if len(fin) < k:
        return fin.tolist()
    vstar = np.sort(key_row[fin])[::-1][k - 1]
    gt = np.nonzero(key_row > vstar)[0].tolist()
    eq = np.nonzero(key_row == vstar)[0].tolist()
    return gt + eq[: k - len(gt)]


def ref_topk(scores: torch.Tensor, k: int, k_pad: int) -> torch.Tensor:
    keys = keys_of(scores)
    out = np.full((scores.shape[0], k_pad), -1, dtype = np.int32)
    for r in range(scores.shape[0]):
        o = ref_order(keys[r], k)
        out[r, : len(o)] = o
    return torch.from_numpy(out)


def make_scores(mode: str, R: int, T: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    if mode == "randn":
        s = torch.randn((R, T), generator = g) * 2
    elif mode == "ties":
        s = torch.randint(0, 8, (R, T), generator = g).float() * 0.25
    elif mode == "zeros":
        # +0 / -0 / 1: equal as floats, distinct in the key order
        s = torch.tensor([0.0, -0.0, 1.0])[torch.randint(0, 3, (R, T), generator = g)]
    elif mode == "causal":
        # -inf past a per-row bound, as the indexer scores rows near the causal edge
        s = torch.randn((R, T), generator = g)
        bounds = torch.randint(1, T + 1, (R,), generator = g)
        s[torch.arange(T)[None, :] >= bounds[:, None]] = -float("inf")
    elif mode == "sparse":
        # Few finite entries scattered over the row
        s = torch.full((R, T), -float("inf"))
        for r in range(R):
            n = int(torch.randint(1, 40, (1,), generator = g))
            idx = torch.randperm(T, generator = g)[:n]
            s[r, idx] = torch.randn(n, generator = g)
    elif mode == "allinf":
        s = torch.full((R, T), -float("inf"))
    else:
        raise ValueError(mode)
    return s.half()


def run_tiled(scores: torch.Tensor, k: int, t_tile: int, kp: int) -> torch.Tensor:
    """The qsa_indexer._select_rows tiled loop: per tile a local top-k into slot 1, merged with the running set in
    slot 0 into slot 0 of the other workspace; the last merge writes the output"""
    rows, T = scores.shape
    dev = scores.device
    out = torch.empty((rows, kp), dtype = torch.int32, device = dev)
    ws = [(torch.empty((rows, 2, kp), dtype = torch.int32, device = dev),
           torch.empty((rows, 2, kp), dtype = torch.half, device = dev),
           torch.zeros((rows, 2), dtype = torch.int32, device = dev)) for _ in range(2)]
    cur = 0
    tiles = list(range(0, T, t_tile))
    for n, t0 in enumerate(tiles):
        t1 = min(t0 + t_tile, T)
        sc = scores[:, t0 : t1].contiguous()
        w_idx, w_scr, w_cnt = ws[cur]
        ext.dsa_topk_tile(sc, w_idx, w_scr, w_cnt, 1, min(k, t1 - t0), t0)
        if n == len(tiles) - 1:
            ext.dsa_topk_merge_tiles(w_idx, w_scr, w_cnt, out, None, None, k)
        else:
            n_idx, n_scr, n_cnt = ws[cur ^ 1]
            ext.dsa_topk_merge_tiles(w_idx, w_scr, w_cnt, n_idx[:, 0], n_scr[:, 0], n_cnt[:, 0], k)
            cur ^= 1
    return out


def check_rows(got: torch.Tensor, expect: torch.Tensor, scores: torch.Tensor, what: str):
    if torch.equal(got.cpu(), expect.cpu()):
        return
    for r in range(got.shape[0]):
        if not torch.equal(got[r].cpu(), expect[r].cpu()):
            g = got[r].cpu()
            e = expect[r].cpu()
            same_set = set(g[g >= 0].tolist()) == set(e[e >= 0].tolist())
            n_fin = int((scores[r] > -float("inf")).sum())
            pytest.fail(f"{what}: row {r} differs ({'same set, different order' if same_set else 'different set'};"
                        f" {n_fin} finite): got {g[:12].tolist()}..., expected {e[:12].tolist()}...")


# --------------------------------------------------------------------------------------------------------------
# Single-pass kernel: exact order (the property the tiled path is defined against)

# (R, T, k, mode); R <= 16 with T >= 32768 takes dsa_topk's internal split/merge path
TOPK_CASES = [
    (4, 2048, 512, "randn"), (3, 1000, 64, "ties"), (2, 4096, 256, "zeros"), (8, 3000, 128, "causal"),
    (3, 5000, 64, "sparse"), (2, 512, 16, "allinf"), (5, 33, 3, "ties"), (1, 40, 40, "randn"),
    (2, 40000, 512, "randn"), (2, 40000, 512, "ties"), (3, 65536, 1024, "causal"),
]


@pytest.mark.parametrize("R,T,k,mode", TOPK_CASES, ids = [f"R{c[0]}-T{c[1]}-k{c[2]}-{c[3]}" for c in TOPK_CASES])
@torch.inference_mode()
def test_topk_exact_order(device, R, T, k, mode):
    scores = make_scores(mode, R, T, 11).to(device)
    kp = -(-k // 32) * 32
    out = torch.empty((R, kp), dtype = torch.int32, device = device)
    ext.dsa_topk(scores, out, k, None, 0)
    check_rows(out, ref_topk(scores, k, kp), scores, "dsa_topk")


# --------------------------------------------------------------------------------------------------------------
# dsa_topk_tile: one slot, other slots untouched

@pytest.mark.parametrize("R,T,k,mode,slot,G,offset,dense", [
    (4, 2048, 128, "randn", 1, 2, 8192, True),
    (3, 700, 64, "ties", 0, 4, 0, True),
    (2, 1000, 512, "causal", 3, 4, 123456, True),
    (5, 100, 64, "sparse", 2, 3, 77, True),           # fewer finite than k
    (2, 513, 32, "zeros", 1, 2, 0, False),            # unaligned rows: scalar path
    (2, 2048, 2048, "randn", 0, 1, 0, False),         # k == T
])
@torch.inference_mode()
def test_tile(device, R, T, k, mode, slot, G, offset, dense):
    scores = make_scores(mode, R, T, 12)
    if dense:
        sc = scores.to(device)
    else:
        backing = torch.zeros((R, T + 3), dtype = torch.half, device = device)
        backing[:, :T] = scores.to(device)
        sc = backing[:, :T]
        assert sc.stride(0) % 8 != 0
    kp = -(-k // 32) * 32
    ws_idx = torch.full((R, G, kp), -7, dtype = torch.int32, device = device)
    ws_scr = torch.full((R, G, kp), 3.5, dtype = torch.half, device = device)
    ws_cnt = torch.full((R, G), -9, dtype = torch.int32, device = device)
    idx0, scr0, cnt0 = ws_idx.clone(), ws_scr.clone(), ws_cnt.clone()
    ext.dsa_topk_tile(sc, ws_idx, ws_scr, ws_cnt, slot, k, offset)

    keys = keys_of(scores)
    for r in range(R):
        o = ref_order(keys[r], k)
        n = len(o)
        assert ws_cnt[r, slot].item() == n, f"row {r}: count {ws_cnt[r, slot].item()} != {n}"
        assert ws_idx[r, slot, :n].tolist() == [i + offset for i in o], f"row {r}: indices / order"
        assert torch.equal(ws_scr[r, slot, :n].cpu().view(torch.int16), scores[r, o].view(torch.int16))
        # Beyond the count the slot is not written
        assert torch.equal(ws_idx[r, slot, n:], idx0[r, slot, n:])
    others = [g for g in range(G) if g != slot]
    assert torch.equal(ws_idx[:, others], idx0[:, others])
    assert torch.equal(ws_scr[:, others], scr0[:, others])
    assert torch.equal(ws_cnt[:, others], cnt0[:, others])


# --------------------------------------------------------------------------------------------------------------
# Tiled pipeline == single pass

# (R, T, t_tile, k, mode)
TILED_CASES = [
    (4, 20000, 8192, 512, "randn"),
    (3, 17000, 4096, 256, "ties"),
    (2, 9000, 1024, 128, "zeros"),
    (8, 30000, 8192, 1024, "causal"),
    (4, 12000, 2048, 2048, "ties"),        # k == t_tile: each tile contributes everything finite
    (2, 5000, 1000, 64, "allinf"),
    (3, 8193, 8192, 512, "randn"),         # last tile of one entry
    (2, 6000, 1024, 64, "sparse"),         # few scattered finite entries: candidate totals at or below k
    (16, 40000, 8192, 512, "causal"),      # dsa_topk reference takes its own split path here
]


@pytest.mark.parametrize("R,T,t_tile,k,mode", TILED_CASES,
                         ids = [f"R{c[0]}-T{c[1]}-tile{c[2]}-k{c[3]}-{c[4]}" for c in TILED_CASES])
@torch.inference_mode()
def test_tiled_equals_single_pass(device, R, T, t_tile, k, mode):
    scores = make_scores(mode, R, T, 13).to(device)
    kp = -(-k // 32) * 32
    got = run_tiled(scores, k, t_tile, kp)
    check_rows(got, ref_topk(scores, k, kp), scores, "tiled vs reference")
    single = torch.empty((R, kp), dtype = torch.int32, device = device)
    ext.dsa_topk(scores, single, k, None, 0)
    check_rows(got, single, scores, "tiled vs dsa_topk")
    # Deterministic
    assert torch.equal(run_tiled(scores, k, t_tile, kp), got)


@torch.inference_mode()
def test_merge_total_equals_k(device):
    """Exactly k finite candidates spread over two tiles, the smallest not last in index order: the single pass
    emits [> v*] then [== v*]; the merge sees total == k and must not just concatenate its slots"""
    k = 4
    s = torch.full((1, 64), -float("inf"))
    s[0, 3] = 1.0      # smallest, tile 0
    s[0, 10] = 5.0     # tile 0
    s[0, 40] = 3.0     # tile 1
    s[0, 50] = 2.0     # tile 1
    scores = s.half().to(device)
    kp = 32
    expect = ref_topk(scores, k, kp)
    assert expect[0, :4].tolist() == [10, 40, 50, 3]
    single = torch.empty((1, kp), dtype = torch.int32, device = device)
    ext.dsa_topk(scores, single, k, None, 0)
    check_rows(single, expect, scores, "dsa_topk")
    check_rows(run_tiled(scores, k, 32, kp), expect, scores, "tiled")


@torch.inference_mode()
def test_topk_split_total_equals_k(device):
    """Same situation inside dsa_topk's own split path (R <= 16, T >= 32768: span-local selections + the merge
    kernel), which its comment declares identical to the single-block kernel"""
    k = 4
    s = torch.full((1, 32768), -float("inf"))
    s[0, 3], s[0, 10], s[0, 2000], s[0, 3000] = 1.0, 5.0, 3.0, 2.0
    scores = s.half().to(device)
    kp = 32
    expect = ref_topk(scores, k, kp)
    assert expect[0, :4].tolist() == [10, 2000, 3000, 3]
    out = torch.empty((1, kp), dtype = torch.int32, device = device)
    ext.dsa_topk(scores, out, k, None, 0)
    check_rows(out, expect, scores, "dsa_topk split path")


@pytest.mark.parametrize("G,t_tile,k,mode", [(4, 1024, 256, "randn"), (32, 256, 64, "ties"),
                                             (8, 512, 512, "causal"), (32, 128, 100, "zeros")])
@torch.inference_mode()
def test_merge_many_slots(device, G, t_tile, k, mode):
    """One merge over G tile slots of ascending index ranges, scores and counts emitted"""
    R = 3
    T = G * t_tile - 5
    scores = make_scores(mode, R, T, 14).to(device)
    kp = -(-k // 32) * 32
    ws_idx = torch.empty((R, G, kp), dtype = torch.int32, device = device)
    ws_scr = torch.empty((R, G, kp), dtype = torch.half, device = device)
    ws_cnt = torch.zeros((R, G), dtype = torch.int32, device = device)
    for g in range(G):
        t0, t1 = g * t_tile, min((g + 1) * t_tile, T)
        ext.dsa_topk_tile(scores[:, t0 : t1].contiguous(), ws_idx, ws_scr, ws_cnt, g, min(k, t1 - t0), t0)
    out = torch.full((R, kp), -5, dtype = torch.int32, device = device)
    out_scr = torch.zeros((R, kp), dtype = torch.half, device = device)
    out_cnt = torch.full((R,), -1, dtype = torch.int32, device = device)
    ext.dsa_topk_merge_tiles(ws_idx, ws_scr, ws_cnt, out, out_scr, out_cnt, k)
    expect = ref_topk(scores, k, kp)
    check_rows(out, expect, scores, "merge")
    for r in range(R):
        n = int((expect[r] >= 0).sum())
        assert out_cnt[r].item() == n
        sel = expect[r, :n].long()
        assert torch.equal(out_scr[r, :n].cpu().view(torch.int16), scores[r].cpu()[sel].view(torch.int16))


@torch.inference_mode()
def test_tile_merge_rejects(device):
    sc = torch.zeros((2, 64), dtype = torch.half, device = device)
    idx = torch.zeros((2, 2, 32), dtype = torch.int32, device = device)
    scr = torch.zeros((2, 2, 32), dtype = torch.half, device = device)
    cnt = torch.zeros((2, 2), dtype = torch.int32, device = device)
    with pytest.raises(RuntimeError, match = "bad slot"):
        ext.dsa_topk_tile(sc, idx, scr, cnt, 2, 8, 0)
    with pytest.raises(RuntimeError, match = "bad slot"):
        ext.dsa_topk_tile(sc, idx, scr, cnt, 0, 33, 0)
    with pytest.raises(RuntimeError, match = "rows/slots mismatch"):
        ext.dsa_topk_tile(sc[:1], idx, scr, cnt, 0, 8, 0)
    with pytest.raises(RuntimeError, match = "bad workspace"):
        ext.dsa_topk_tile(sc, idx, scr[:, :, :16], cnt, 0, 8, 0)
    with pytest.raises(RuntimeError, match = "dense rows"):
        ext.dsa_topk_tile(sc.t(), idx, scr, cnt, 0, 8, 0)
    out = torch.zeros((2, 32), dtype = torch.int32, device = device)
    with pytest.raises(RuntimeError, match = "shape mismatch"):
        ext.dsa_topk_merge_tiles(idx, scr, cnt, out[:, :16], None, None, 8)
    with pytest.raises(RuntimeError, match = "shape mismatch"):
        ext.dsa_topk_merge_tiles(idx, scr, cnt, out, None, None, 33)
    big = torch.zeros((2, MAX_G + 1, 32), dtype = torch.int32, device = device)
    with pytest.raises(RuntimeError, match = "shape mismatch"):
        ext.dsa_topk_merge_tiles(big, big.half(), torch.zeros((2, MAX_G + 1), dtype = torch.int32, device = device),
                                 out, None, None, 8)
    with pytest.raises(RuntimeError, match = "out_scr must match"):
        ext.dsa_topk_merge_tiles(idx, scr, cnt, out, torch.zeros((2, 16), dtype = torch.half, device = device), None, 8)
    with pytest.raises(RuntimeError, match = "bad out_cnt"):
        ext.dsa_topk_merge_tiles(idx, scr, cnt, out, None, torch.zeros((3,), dtype = torch.int32, device = device), 8)
