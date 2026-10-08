"""
Contracts of fused_sampler and apply_logit_bitmask (generator/sampling_fused.cu), against a NumPy composition of
the sampling steps they replace and the Philox reference noise of testlib.sampling.

fused_sampler(logits (bsz, dim) half/float, logit_mask, logit_bitmask, out (bsz, 1) long, workspace, size,
inv_temp, minp_log, seed, mode, filters, top_k, top_p, inv_temp_filter, histogram) writes one token per row:

    x_i  = logits[i] (+ half mask[i] in fp32 | -inf where the bitmask bit is clear); i >= size never chosen
    mode 0: first index of max x_i
    mode 1: argmax over finite x of fp32(x_i * inv_temp) + G(u_{row * dim + i})       (Gumbel-max sampling)
    mode 2: the same restricted to x_i >= fp32(max x + minp_log)                       (min-P)
    mode 3: restricted to the min-P set (F_MINP), then the top_k largest (F_TOPK, ties at the k-th value all
            kept), then the top-P prefix (F_TOPP) of the descending order under softmax((x - max) *
            inv_temp_filter) over that set: tokens before the one whose inclusive cumulative probability
            first exceeds top_p are kept, the top token always

The noise is the per-element stream of gumbel_noise_f32 (Philox keyed on (seed, row * dim + i)), so the
expected token is computable exactly; rows whose winner is within the __logf noise tolerance of a runner-up
are not decided by the reference and skipped (a floor on the decided fraction keeps the tests meaningful).
Mode 3's kept set is located by a fixed-point histogram with sub-buckets of 1/32768 nat over 32 nats below the
max; the reference is the exact sort-based truncation, compared at a very high sampling temperature
(inv_temp = 1e-3) where the winner is the max noise over the kept set, so boundary tokens are exercised.

apply_logit_bitmask(logits_in, logits_out, bitmask): out[r, i] = in[r, i] where bit i of the row's (or the
single broadcast row's) packed int32 bitmask is set and i < 32 * words, else -inf; exact, out of place.
"""

import math

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.sampling import certain_argmax, element_noise, gumbel_noise_tolerance

F_TOPK, F_TOPP, F_MINP = 1, 2, 4
U_TIGHT = 1.0 - 1e-6


def run_fused(
    logits, mode, *, size = None, mask = None, bits = None, inv_temp = 1.0, minp_log = 0.0, seed = 0,
    filters = 0, top_k = 0, top_p = 1.0, inv_temp_filter = 1.0, workspace = None, histogram = None, out = None,
):
    bsz, dim = logits.shape
    dev = logits.device
    if workspace is None:
        workspace = torch.empty(bsz * ext.FUSED_SAMPLER_MAX_BLOCKS * 3, dtype = torch.float, device = dev)
    if histogram is None and mode == 3:
        histogram = torch.empty(bsz * ext.FUSED_SAMPLER_HIST_STRIDE, dtype = torch.uint8, device = dev)
    if out is None:
        out = torch.full((bsz, 1), -1, dtype = torch.long, device = dev)
    ext.fused_sampler(
        logits, mask, bits, out, workspace, size or dim, inv_temp, minp_log, seed, mode, filters, top_k, top_p,
        inv_temp_filter, histogram,
    )
    return out


# References

def unpack_bits(bits: torch.Tensor, width: int) -> np.ndarray:
    """(rows, words) int32 -> (rows, width) bool, bit j of word w = token 32 w + j; tokens past 32 * words are
    clear"""
    b = bits.cpu().numpy().view(np.uint32)
    rows, words = b.shape
    out = np.zeros((rows, max(width, 32 * words)), dtype = bool)
    for j in range(32):
        out[:, j: 32 * words: 32] = (b >> np.uint32(j)) & np.uint32(1)
    return out[:, :width]


def masked_inputs(logits, size, mask = None, bits = None) -> np.ndarray:
    """The x_i of the contract in float32, -inf at and beyond size"""
    x = logits.float().cpu().numpy().copy()
    bsz, dim = x.shape
    if mask is not None:
        m = mask.float().cpu().numpy()
        x[:, :size] = (x[:, :size] + m[:, :size]).astype(np.float32)
    if bits is not None:
        keep = unpack_bits(bits, dim)
        keep = np.broadcast_to(keep, (bsz, dim))
        x = np.where(keep, x, np.float32(-np.inf)).astype(np.float32)
    x[:, size:] = -np.inf
    return x


def kept_set(x: np.ndarray, mode, minp_log = 0.0, filters = 0, top_k = 0, top_p = 1.0, inv_temp_filter = 1.0):
    """Boolean kept set of one row (float32 x) under SS_Fused's documented rules, or None where the reference is
    ambiguous (a cumulative probability within 4e-6 of top_p, or a token within 1e-3 nat of the 32-nat tail
    boundary). Top-K keeps cutoff ties, and a cutoff 32 or more filter-temperature nats below the max drops
    every token that deep; top-P drops the crossing token together with all its ties"""
    finite = np.isfinite(x)
    if mode == 1:
        return finite
    m = x[finite].max() if finite.any() else np.float32(-np.inf)
    keep = finite.copy()
    if mode == 2 or (mode == 3 and filters & F_MINP):
        thr = np.float32(np.float32(m) + np.float32(minp_log))
        keep &= x >= thr
    if mode == 2:
        return keep
    if filters & F_TOPK and keep.sum() > top_k:
        vals = np.sort(x[keep])[::-1]
        kth = vals[top_k - 1]
        depth = (np.float64(m) - x.astype(np.float64)) * inv_temp_filter
        if (np.float64(m) - kth) * inv_temp_filter >= 32.0 - 1e-3:
            if np.any(keep & (np.abs(depth - 32.0) < 1e-3)):
                return None
            keep &= depth < 32.0
        else:
            keep &= x >= kth
    if filters & F_TOPP:
        idx = np.nonzero(keep)[0]
        v = x[idx].astype(np.float64)
        order = np.argsort(-v, kind = "stable")
        idx, v = idx[order], v[order]
        w = np.exp((v - np.float64(m)) * inv_temp_filter)
        c = np.cumsum(w) / w.sum()
        # The kernel's mass is __expf in fp32 (relative error ~2 ulp plus |arg| * 2^-24 for args down to -32),
        # summed exactly in 2^40 fixed point: normalized cumulative sums carry up to ~3e-6 of error. Tie groups
        # are kept or dropped whole, so only the cumulative mass at the end of each group decides
        group_end = np.append(v[1:] != v[:-1], True)
        if np.min(np.abs(c[group_end] - top_p)) < 4e-6:
            return None
        cross = int(np.argmax(c > top_p)) if (c > top_p).any() else len(c)
        if cross < len(c):
            # The crossing token's whole tie group goes, including ties sorted before it
            cross = int(np.argmax(v == v[cross]))
        nkeep = max(cross, 1)
        # The top token keeps its exact ties
        top_ties = int((v == v[0]).sum())
        nkeep = max(nkeep, top_ties)
        keep = np.zeros_like(keep)
        keep[idx[:nkeep]] = True
    return keep


def expected_tokens(logits, mode, seed, *, size = None, mask = None, bits = None, inv_temp = 1.0, **kw):
    """Per row: the reference token, or None where the reference cannot decide"""
    bsz, dim = logits.shape
    size = size or dim
    x = masked_inputs(logits, size, mask, bits)
    if mode == 0:
        return [int(np.argmax(x[r])) if np.isfinite(x[r]).any() else 0 for r in range(bsz)]
    u, g = element_noise(seed, bsz * dim)
    u, g = u.reshape(bsz, dim), g.reshape(bsz, dim)
    out = []
    for r in range(bsz):
        keep = kept_set(x[r], mode, **kw)
        if keep is None:
            out.append(None)
            continue
        if not keep.any():
            out.append(0)
            continue
        with np.errstate(invalid = "ignore"):
            obj = (x[r] * np.float32(inv_temp)).astype(np.float32).astype(np.float64) + g[r]
        obj = np.where(keep, obj, -np.inf)
        # fp32 rounding of the scaled logit and of the sum (an FMA may merge them) plus the noise bound
        tol = gumbel_noise_tolerance(u[r]) + 2.0 ** -22 * np.abs(np.where(keep, obj, 0.0)) + 1e-7
        tol[u[r] >= U_TIGHT] = 4.0
        out.append(certain_argmax(obj, tol))
    return out


class Tally:
    """Compares sampled tokens with the reference where it decides, and counts decided rows, so a test can
    require that the reference was not vacuous"""
    def __init__(self):
        self.rows = 0
        self.decided = 0

    def check(self, got: torch.Tensor, ref: list, msg: str):
        for r, (a, b) in enumerate(zip(got.view(-1).tolist(), ref)):
            self.rows += 1
            if b is not None:
                assert a == b, f"{msg} row {r}: sampled {a}, reference {b}"
                self.decided += 1

    def require(self, fraction: float):
        assert self.decided >= fraction * self.rows, f"reference decided only {self.decided}/{self.rows} rows"


def snapped_logits(bsz, dim, scale, seed, dtype = torch.half):
    """Logits on a 1/128 grid (exact in fp16 below 16 in magnitude), so distinct values are far apart compared to
    the histogram's sub-bucket width"""
    g_ = torch.Generator().manual_seed(seed)
    x = (torch.randn(bsz, dim, generator = g_) * scale).clamp(-15.9, 15.9)
    return (torch.round(x * 128) / 128).to(dtype)


def distinct_logits(bsz, dim, scale, seed):
    """fp32 logits whose distinct values are at least 4e-5 apart: more than one histogram sub-bucket (1/32768
    nat at inv_temp_filter <= 1.33), so the kept set is exact without exact ties"""
    g_ = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(bsz):
        v = np.sort((torch.randn(dim, generator = g_, dtype = torch.float64) * scale).numpy())
        v = v + np.arange(dim) * 4e-5
        v = v - v.max() + 8.0
        perm = torch.randperm(dim, generator = g_).numpy()
        rows.append(v[perm])
    return torch.tensor(np.stack(rows), dtype = torch.float)


# Mode 0

@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("bsz, dim", [(1, 1), (1, 7), (3, 1025), (2, 32000), (4, 151936), (1, 262147)])
@torch.inference_mode()
def test_fused_greedy(bsz, dim, dtype, device):
    logits = torch.randn(bsz, dim, generator = torch.Generator().manual_seed(dim)).to(dtype).to(device)
    out = run_fused(logits, 0)
    assert out.view(-1).tolist() == expected_tokens(logits, 0, 0)


@torch.inference_mode()
def test_fused_greedy_ties_pick_first(device):
    dim = 151936
    for positions in ([3, 70000, 150000], [131071, 131072], [dim - 2, dim - 1], [0, dim - 1]):
        logits = torch.zeros(2, dim, dtype = torch.half, device = device)
        logits[:, positions] = 7.0
        logits[1, positions[0]] = 0.0     # row 1: the second tie position is first
        out = run_fused(logits, 0)
        assert out.view(-1).tolist() == [positions[0], positions[1]], positions


@torch.inference_mode()
def test_fused_all_masked_row(device):
    logits = torch.full((2, 3000), float("-inf"), device = device)
    logits[1, 17] = 0.0
    for mode, kw in [(0, {}), (1, {}), (2, dict(minp_log = math.log(0.1))),
                     (3, dict(filters = F_TOPK | F_TOPP, top_k = 5, top_p = 0.9))]:
        out = run_fused(logits, mode, **kw)
        assert out.view(-1).tolist() == [0, 17], f"mode {mode}"


# Modes 1 and 2

@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("inv_temp", [1.0, 1 / 0.7, 1 / 1.6])
@pytest.mark.parametrize("bsz, dim", [(1, 7), (4, 1025), (8, 32000), (4, 151936)])
@torch.inference_mode()
def test_fused_sample(bsz, dim, inv_temp, dtype, device):
    tally = Tally()
    for seed in (0, 1, 2, 3, 0xFFFFFFFF, 123456789):
        logits = (torch.randn(bsz, dim, generator = torch.Generator().manual_seed(seed % 1000 + dim)) * 3)
        logits = logits.to(dtype).to(device)
        logits[:, ::5] = float("-inf")
        out = run_fused(logits, 1, inv_temp = inv_temp, seed = seed)
        tally.check(out, expected_tokens(logits, 1, seed, inv_temp = inv_temp), f"seed {seed}")
    tally.require(0.5)


@pytest.mark.parametrize("min_p", [0.02, 0.1, 0.5])
@pytest.mark.parametrize("temp_first", [False, True])
@pytest.mark.parametrize("bsz, dim", [(1, 9), (4, 4099), (4, 151936)])
@torch.inference_mode()
def test_fused_sample_minp(bsz, dim, min_p, temp_first, device):
    temperature = 1.3
    inv_temp = 1 / temperature
    minp_log = (temperature if temp_first else 1.0) * math.log(min_p)
    tally = Tally()
    for seed in range(6):
        logits = (torch.randn(bsz, dim, generator = torch.Generator().manual_seed(seed + dim)) * 3).half().to(device)
        # High sampling temperature on top of the real one makes boundary tokens win often
        for it in (inv_temp, 1e-3):
            out = run_fused(logits, 2, inv_temp = it, minp_log = minp_log, seed = seed)
            ref = expected_tokens(logits, 2, seed, inv_temp = it, minp_log = minp_log)
            tally.check(out, ref, f"seed {seed} inv_temp {it}")
    tally.require(0.5)


# Mode 3

MODE3_CASES = [
    dict(filters = F_TOPK, top_k = 1),
    dict(filters = F_TOPK, top_k = 5),
    dict(filters = F_TOPK, top_k = 50),
    dict(filters = F_TOPP, top_p = 0.5),
    dict(filters = F_TOPP, top_p = 0.95),
    dict(filters = F_TOPK | F_TOPP, top_k = 40, top_p = 0.8),
    dict(filters = F_TOPK | F_TOPP, top_k = 200, top_p = 0.3),
    dict(filters = F_MINP | F_TOPK, top_k = 30, min_p = 0.05),
    dict(filters = F_MINP | F_TOPP, top_p = 0.9, min_p = 0.01),
    dict(filters = F_MINP | F_TOPK | F_TOPP, top_k = 64, top_p = 0.9, min_p = 0.02),
]


@pytest.mark.parametrize("kind", ["half_grid", "float_distinct"])
@pytest.mark.parametrize("temp_first", [False, True], ids = ["filters_untempered", "filters_tempered"])
@pytest.mark.parametrize("case", MODE3_CASES, ids = lambda c: "-".join(f"{k}{v}" for k, v in c.items()))
@pytest.mark.parametrize("bsz, dim", [(4, 1000), (4, 32003), (2, 151936)])
@torch.inference_mode()
def test_fused_filters(bsz, dim, case, temp_first, kind, device):
    case = dict(case)
    min_p = case.pop("min_p", None)
    temperature = 0.75
    inv_temp = 1 / temperature
    itf = inv_temp if temp_first else 1.0
    minp_log = (temperature if temp_first else 1.0) * math.log(min_p) if min_p else 0.0
    tally = Tally()
    for seed in range(12):
        if kind == "half_grid":
            logits = snapped_logits(bsz, dim, 2.5, seed * 7 + dim).to(device)
        else:
            logits = distinct_logits(bsz, dim, 4.0, seed * 7 + dim).to(device)
        for it in (inv_temp, 1e-3):
            kw = dict(minp_log = minp_log, inv_temp_filter = itf, **case)
            out = run_fused(logits, 3, inv_temp = it, seed = seed, **kw)
            ref = expected_tokens(logits, 3, seed, inv_temp = it, **kw)
            tally.check(out, ref, f"seed {seed} inv_temp {it}")
    tally.require(0.5)


@torch.inference_mode()
def test_fused_topk_exact_kept_set(device):
    """At inv_temp -> 0 the winner is the max noise over the kept set; over many seeds every kept token and no
    other appears"""
    dim = 4096
    logits = snapped_logits(1, dim, 3.0, 5).to(device)
    x = logits.float().cpu().numpy()[0]
    for k in (1, 3, 17):
        ref = kept_set(x, 3, filters = F_TOPK, top_k = k)
        seen = set()
        for seed in range(40 * k):
            seen.add(run_fused(logits, 3, inv_temp = 1e-4, seed = seed, filters = F_TOPK, top_k = k).item())
        assert seen == set(np.nonzero(ref)[0].tolist()), f"k={k}"


@torch.inference_mode()
def test_fused_topk_ties_kept(device):
    """Tokens tied with the k-th value are all kept (documented divergence from sort-order truncation)"""
    logits = torch.full((1, 300), -3.0, dtype = torch.half, device = device)
    logits[0, 10] = 2.0
    logits[0, [20, 30, 40, 250]] = 1.0       # k = 3 cuts inside this tie group
    seen = set()
    for seed in range(200):
        seen.add(run_fused(logits, 3, inv_temp = 1e-4, seed = seed, filters = F_TOPK, top_k = 3).item())
    assert seen == {10, 20, 30, 40, 250}


@torch.inference_mode()
def test_fused_topp_ties_at_cutoff(device):
    """Top-P crossing inside a group of tied values: the crossing token is dropped together with all its exact
    ties (SS_Fused docstring). Probabilities e/(e+4), 1/(e+4) x 4: cumsums 0.405, 0.554, 0.702, ... cross
    top_p = 0.6 inside the tied group, so only the top token remains (sort-order truncation would keep one of the
    tied tokens as well)"""
    logits = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype = torch.half, device = device)
    seen = set()
    for seed in range(300):
        seen.add(run_fused(logits, 3, inv_temp = 1e-4, seed = seed, filters = F_TOPP, top_p = 0.6).item())
    assert seen == {0}, f"kept {sorted(seen)}"


@pytest.mark.parametrize("depth", [20.0, 31.0, 31.9, 33.0, 40.0])
@torch.inference_mode()
def test_fused_topk_deep_cutoff(depth, device):
    """Top-K keeps the k largest values among the tokens less than 32 nats below the max (at the filter
    temperature) and drops the tokens beyond that when the cutoff falls among them (SS_Fused docstring), never
    keeping more than k. The sampling temperature applied after the filter (near-uniform here) makes every kept
    token show up"""
    dim = 64
    logits = (-depth - torch.arange(dim).float() * 0.25).half().view(1, dim).to(device)
    logits[0, 5] = 0.0
    k = 3
    ref = logits[0].float().cpu()
    top = torch.topk(ref, k).indices.tolist()
    expect = {i for i in top if ref.max().item() - ref[i].item() < 32.0}
    seen = set()
    for seed in range(300):
        seen.add(run_fused(logits, 3, inv_temp = 1e-4, seed = seed, filters = F_TOPK, top_k = k).item())
    assert seen == expect, f"depth {depth}: kept {sorted(seen)}, expected {sorted(expect)}"


@torch.inference_mode()
def test_fused_topk_tail_then_topp(device):
    """Top-K cutting into the tail, then top-P over what remains: the three in-range tokens (probabilities 0.506,
    0.307, 0.186 among themselves; the deep tokens weigh nothing at the filter temperature) cross top_p = 0.9 at
    the third, so the first two remain"""
    deep = -40.0 - 0.01 * torch.arange(1000).float()
    logits = torch.cat((torch.tensor([0.0, -0.5, -1.0]), deep)).half().view(1, -1).to(device)
    seen = set()
    for seed in range(300):
        seen.add(run_fused(logits, 3, inv_temp = 1e-4, seed = seed, filters = F_TOPK | F_TOPP, top_k = 5,
                           top_p = 0.9).item())
    assert seen == {0, 1}, f"kept {sorted(seen)}"


@torch.inference_mode()
def test_fused_histogram_and_workspace_garbage(device):
    """The kernel clears the histogram and fully writes the workspace before reading them"""
    logits = snapped_logits(3, 20000, 2.5, 1).to(device)
    kw = dict(filters = F_TOPK | F_TOPP | F_MINP, top_k = 50, top_p = 0.9, minp_log = math.log(0.01), seed = 4)
    clean = run_fused(logits, 3, **kw)
    ws = torch.full((3 * ext.FUSED_SAMPLER_MAX_BLOCKS * 3,), float("nan"), device = device)
    hist = torch.full((3 * ext.FUSED_SAMPLER_HIST_STRIDE,), 0xA5, dtype = torch.uint8, device = device)
    dirty = run_fused(logits, 3, workspace = ws, histogram = hist, **kw)
    assert torch.equal(clean, dirty)


# Determinism, geometry, masks, size bound

@pytest.mark.parametrize("mode, kw", [
    (1, {}),
    (2, dict(minp_log = math.log(0.05))),
    (3, dict(filters = F_TOPK | F_TOPP, top_k = 50, top_p = 0.9)),
])
@torch.inference_mode()
def test_fused_deterministic_and_geometry_independent(mode, kw, device):
    """Same seed, same token; and the token does not depend on the block split: row 0's noise index is i alone,
    so padding the row with -inf (more blocks) must not change its sample"""
    dim = 5000
    logits = (torch.randn(1, dim, generator = torch.Generator().manual_seed(3)) * 3).half().to(device)
    padded = torch.full((1, 151936), float("-inf"), dtype = torch.half, device = device)
    padded[:, :dim] = logits
    for seed in range(20):
        a = run_fused(logits, mode, seed = seed, **kw)
        b = run_fused(logits, mode, seed = seed, **kw)
        c = run_fused(padded, mode, seed = seed, **kw)
        d = run_fused(padded, mode, seed = seed, size = dim + 3, **kw)
        assert a.item() == b.item() == c.item() == d.item(), f"seed {seed}: {a.item()} {b.item()} {c.item()} {d.item()}"


@pytest.mark.parametrize("mode, kw", [
    (0, {}),
    (1, {}),
    (2, dict(minp_log = math.log(0.05))),
    (3, dict(filters = F_TOPK | F_TOPP | F_MINP, top_k = 20, top_p = 0.9, minp_log = math.log(0.01))),
])
@pytest.mark.parametrize("mask_kind", ["half_bcast", "half_rows", "half_narrow", "bits_bcast", "bits_rows", "bits_narrow"])
@torch.inference_mode()
def test_fused_masks_and_size(mode, kw, mask_kind, device):
    bsz, dim = 3, 33003
    g_ = torch.Generator().manual_seed(11)
    logits = snapped_logits(bsz, dim, 2.5, 12).to(device)
    mask = bits = None
    size = dim
    if mask_kind.startswith("half"):
        rows = bsz if mask_kind == "half_rows" else 1
        width = dim if mask_kind != "half_narrow" else dim - 1000
        m = torch.where(torch.rand(rows, width, generator = g_) < 0.6, 0.0, float("-inf"))
        m += torch.where(torch.rand(rows, width, generator = g_) < 0.1, 1.5, 0.0)     # some additive bias too
        mask = m.half().to(device)
        size = min(dim, width)
    else:
        rows = bsz if mask_kind == "bits_rows" else 1
        words = (dim + 31) // 32 if mask_kind != "bits_narrow" else (dim - 1000) // 32
        bits = torch.randint(-2 ** 31, 2 ** 31 - 1, (rows, words), generator = g_, dtype = torch.int32).to(device)
        size = min(dim, words * 32)
    if mask_kind.endswith("narrow"):
        size -= 5       # size bound below the mask width
    tally = Tally()
    for seed in range(4):
        for it in (1.0, 1e-3):
            out = run_fused(logits, mode, size = size, mask = mask, bits = bits, seed = seed, inv_temp = it, **kw)
            ref = expected_tokens(logits, mode, seed, size = size, mask = mask, bits = bits, inv_temp = it, **kw)
            tally.check(out, ref, f"{mask_kind} seed {seed} inv_temp {it}")
            x = masked_inputs(logits, size, mask, bits)
            for r, t in enumerate(out.view(-1).tolist()):
                assert t < size and np.isfinite(x[r, t]), f"row {r}: sampled masked/out-of-bound token {t}"
    tally.require(0.4)


@torch.inference_mode()
def test_fused_rejects_invalid(device):
    logits = torch.randn(2, 1000, device = device)
    ws = torch.empty(2 * ext.FUSED_SAMPLER_MAX_BLOCKS * 3, device = device)
    hist = torch.empty(2 * ext.FUSED_SAMPLER_HIST_STRIDE, dtype = torch.uint8, device = device)
    out = torch.empty(2, 1, dtype = torch.long, device = device)

    def call(lg = logits, mask = None, bits = None, o = out, w = ws, size = 1000, mode = 1, h = hist):
        ext.fused_sampler(lg, mask, bits, o, w, size, 1.0, 0.0, 0, mode, F_TOPK, 5, 1.0, 1.0, h)

    call()
    bad = [
        dict(mode = 4),
        dict(size = 0),
        dict(size = 1001),
        dict(o = torch.empty(3, 1, dtype = torch.long, device = device)),
        dict(o = torch.empty(2, 1, dtype = torch.int, device = device)),
        dict(w = ws[:-1]),
        dict(lg = torch.randn(1000, 2, device = device).t()),
        dict(lg = logits.bfloat16()),
        dict(mask = torch.zeros(1, 999, dtype = torch.half, device = device)),
        dict(mask = torch.zeros(2, 1500, dtype = torch.half, device = device)),
        dict(mask = torch.zeros(3, 1000, dtype = torch.half, device = device)),
        dict(mask = torch.zeros(1, 1000, device = device)),
        dict(mask = torch.zeros(1, 1000, dtype = torch.half, device = device),
             bits = torch.zeros(1, 32, dtype = torch.int, device = device)),
        dict(bits = torch.zeros(1, 31, dtype = torch.int, device = device)),
        dict(bits = torch.zeros(3, 32, dtype = torch.int, device = device)),
        dict(mode = 3, h = None),
        dict(mode = 3, h = hist[:-1]),
        dict(mode = 3, h = torch.empty(2 * ext.FUSED_SAMPLER_HIST_STRIDE + 8, dtype = torch.uint8, device = device)[4:]),
    ]
    for kw in bad:
        with pytest.raises(RuntimeError):
            call(**kw)


# apply_logit_bitmask

@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("rows", ["bcast", "per_row"])
@pytest.mark.parametrize("bsz, dim, words", [(1, 1, 1), (1, 31, 1), (2, 32, 1), (3, 33, 2), (2, 32000, 1000),
                                             (2, 151936, 4748), (1, 151936, 4000), (4, 100003, 3200)])
@torch.inference_mode()
def test_apply_logit_bitmask(bsz, dim, words, rows, dtype, device):
    g_ = torch.Generator().manual_seed(dim + words)
    logits = torch.randn(bsz, dim, generator = g_).to(dtype).to(device)
    logits[:, ::9] = float("-inf")
    nrows = bsz if rows == "per_row" else 1
    bits = torch.randint(-2 ** 31, 2 ** 31 - 1, (nrows, words), generator = g_, dtype = torch.int32).to(device)
    out_full = torch.full((bsz * dim + 64,), float("nan"), dtype = dtype, device = device)
    out = out_full[:bsz * dim].view(bsz, dim)
    src = logits.clone()
    ext.apply_logit_bitmask(logits, out, bits)
    keep = torch.from_numpy(np.broadcast_to(unpack_bits(bits, dim), (bsz, dim)).copy()).to(device)
    ref = torch.where(keep, logits, torch.full_like(logits, float("-inf")))
    assert torch.equal(out, ref)
    assert torch.equal(logits, src), "input modified"
    assert torch.isnan(out_full[bsz * dim:]).all(), "wrote past the output"


@torch.inference_mode()
def test_apply_logit_bitmask_rejects(device):
    x = torch.randn(2, 64, device = device)
    bits = torch.zeros(1, 2, dtype = torch.int, device = device)
    ext.apply_logit_bitmask(x, torch.empty_like(x), bits)
    for args in [
        (x, torch.empty(2, 63, device = device), bits),
        (x, torch.empty(2, 64, dtype = torch.half, device = device), bits),
        (x, torch.empty_like(x), torch.zeros(3, 2, dtype = torch.int, device = device)),
        (x, torch.empty_like(x), torch.zeros(1, 2, dtype = torch.long, device = device)),
        (x, torch.empty_like(x), torch.zeros(1, 4, dtype = torch.int, device = device)[:, ::2]),
        (x.bfloat16(), torch.empty_like(x.bfloat16()), bits),
        (torch.randn(64, 2, device = device).t(), torch.empty_like(x), bits),
    ]:
        with pytest.raises(RuntimeError):
            ext.apply_logit_bitmask(*args)
