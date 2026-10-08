"""
CPU MoE expert kernel (cpu/moe_mul1.cpp): every ISA tier the CPU can run must agree. The int8 tiers (avx2, bw,
vnni, vbmi) compute the same integer dot products from the same int8 activations, so they agree to float
rounding of the epilogue; the scalar tier is an fp32 reference without the int8 activation quantization and
agrees to about a percent. Any state-extraction or accumulate bug in a tier shows up as O(1) error, far outside
both tolerances. Reference: the avx2 tier (native layout).

The tier is fixed per process by EXL3_MOE_CPU_MAX_ISA (read once at static init), so each tier runs in a
subprocess; on a VBMI machine that exercises vbmi, vnni, bw, avx2 and scalar. Covers K1-8, gated and gateless
experts, 1..5 tokens (m = 1..4 rows per expert chunk), the swizzled layout, and a 256-token case that takes the
GEMV phases' many-GEMV (strided) regime.

The second test repeats the comparison on real expert weights from the lfm2.5-8b-a1b mul1 ladder (one K per
model, layer 2, single-threaded so the accumulation order is fixed): there the int8 tiers must be bit-identical,
since real trellis statistics can expose extraction bugs that random states miss.
"""

import json
import os

import pytest
import torch

from testlib.moe_cpu import run_per_tier, swizzle, swizzle_layouts

pytestmark = [pytest.mark.nogpu, pytest.mark.cpu_flags("avx2")]

INT8_TOL = 5e-4     # rel. L2 between int8 tiers
SCALAR_TOL = 0.05   # rel. L2 int8 tiers vs the fp32 scalar reference

LADDER_DIR = "lfm2.5-8b-a1b/exl3"   # relative to the model root
LADDER = {1: "1.10bpw_mul1", 2: "2.10bpw_mul1", 3: "3.10bpw_mul1", 4: "4.10bpw_mul1",
          5: "5.10bpw_mul1", 6: "6.10bpw_mul1", 7: "7.06bpw_mul1", 8: "8.00bpw_mul1"}
LADDER_LAYER, LADDER_E, LADDER_TOPK = 2, 4, 2


def random_weight_outputs() -> dict:
    """{(K, gated, swizzled, tokens, threads): output} on random states, at this process's tier"""
    from exllamav3.ext import exllamav3_ext as ext
    g = torch.Generator().manual_seed(1234)
    threads = max(2, min(16, (os.cpu_count() or 2) // 2))

    def trellis(k, n, K):   # native [k/16, n/16, 16K] packed layout, random states
        return torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype = torch.int16, generator = g)

    def suh(n):   # real EXL3 suh: random signs x ~0.015
        s = torch.randint(0, 2, (n,), generator = g).float() * 2 - 1
        return (s * (0.015 + 0.004 * torch.randn(n, generator = g))).half().contiguous()

    def svh(n):   # real EXL3 svh: random signs x ~1.0
        s = torch.randint(0, 2, (n,), generator = g).float() * 2 - 1
        return (s * (1.0 + 0.1 * torch.randn(n, generator = g))).half().contiguous()

    # The swizzled layout is only ever handed to the tiers that consume it (scalar and avx2 read native only)
    swz_capable = True in swizzle_layouts()

    results = {}
    hid, inter, E, topk = 512, 640, 6, 3
    for K in range(1, 9):
        for gated in (True, False):
            # one set of logical weights per (K, gated); the swizzled layer is a repack of the same
            gt, gs, gv, ut, us, uv, dt, ds, dv = ([] for _ in range(9))
            for _ in range(E):
                if gated:
                    gt.append(trellis(hid, inter, K)); gs.append(suh(hid)); gv.append(svh(inter))
                ut.append(trellis(hid, inter, K)); us.append(suh(hid)); uv.append(svh(inter))
                dt.append(trellis(inter, hid, K)); ds.append(suh(inter)); dv.append(svh(hid))
            layers = [(False, ext.exl3_moe_cpu_make_layer(gt, gs, gv, ut, us, uv, dt, ds, dv, [], [], [],
                                                          0 if gated else 2, 0.0, 0))]
            if swz_capable:
                sw = lambda ts: [swizzle(t) for t in ts]
                layers.append((True, ext.exl3_moe_cpu_make_layer(sw(gt), gs, gv, sw(ut), us, uv, sw(dt), ds, dv,
                                                                 [], [], [], 0 if gated else 2, 0.0, 1)))
            cases = []
            for tokens in (1, 2, 3, 4, 5, 256):
                x = torch.randn(tokens, hid, generator = g).half()
                # 1..5 tokens on the same experts -> chunks of m = 1..4 rows; 256 tokens spread over all experts
                # -> many chunks per expert -> the strided GEMV assignment regime
                if tokens <= 5:
                    sel = torch.randperm(E, generator = g)[:topk].unsqueeze(0).repeat(tokens, 1)
                else:
                    sel = torch.stack([torch.randperm(E, generator = g)[:topk] for _ in range(tokens)])
                w = torch.rand(tokens, topk, generator = g)
                w = (w / w.sum(-1, keepdim = True)).half()
                cases.append((tokens, x, sel.contiguous(), w.contiguous()))
            for swz, h in layers:
                for tokens, x, sel, w in cases:
                    for th in ((1, threads) if tokens <= 5 else (threads,)):
                        out = torch.zeros(tokens, hid, dtype = torch.float32)
                        ext.exl3_moe_cpu_forward(h, x, sel, w, out, th)
                        results[(K, gated, swz, tokens, th)] = out.clone()
                ext.exl3_moe_cpu_free_layer(h)
    return results


def ladder_outputs(dirs: dict) -> dict:
    """{(K, m): output} on real experts of layer LADDER_LAYER of each {K: model dir}, m = 1..4 rows, one thread,
    native layout, at this process's tier"""
    from safetensors import safe_open
    from exllamav3.ext import exllamav3_ext as ext
    torch.manual_seed(0)
    results = {}
    for bits, d in dirs.items():
        idx = os.path.join(d, "model.safetensors.index.json")
        wm = None
        if os.path.exists(idx):
            with open(idx) as f:
                wm = json.load(f)["weight_map"]
        handles = {}

        def get(k):
            fn = wm[k] if wm else "model.safetensors"
            if fn not in handles:
                handles[fn] = safe_open(os.path.join(d, fn), "pt")
            return handles[fn].get_tensor(k)

        def mats(name):
            out = []
            for e in range(LADDER_E):
                k = f"model.layers.{LADDER_LAYER}.feed_forward.experts.{e}.{name}"
                out.append((get(k + ".trellis").contiguous(), get(k + ".suh").half().contiguous(),
                            get(k + ".svh").half().contiguous()))
            return out

        g, u, dn = mats("w1"), mats("w3"), mats("w2")
        handles.clear()
        assert g[0][0].shape[2] // 16 == bits, (d, g[0][0].shape)
        H = g[0][1].numel()
        h = ext.exl3_moe_cpu_make_layer(
            [t[0] for t in g], [t[1] for t in g], [t[2] for t in g],
            [t[0] for t in u], [t[1] for t in u], [t[2] for t in u],
            [t[0] for t in dn], [t[1] for t in dn], [t[2] for t in dn],
            [], [], [], 0, 0.0, 0)
        for m in range(1, 5):
            x = torch.randn(m, H).half()
            sel = torch.stack([torch.randperm(LADDER_E)[:LADDER_TOPK] for _ in range(m)]).int()
            w = torch.rand(m, LADDER_TOPK).half()
            out = torch.zeros(m, H, dtype = torch.float)
            ext.exl3_moe_cpu_forward(h, x, sel, w, out, 1)
            results[(bits, m)] = out.clone()
        ext.exl3_moe_cpu_free_layer(h)
    return results


def rel_l2(a, b):
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def test_tiers_agree_random_weights():
    outs = run_per_tier(random_weight_outputs)
    assert "avx2" in outs
    ref = outs["avx2"]   # native layout only; swizzled results compare against the same weights natively
    native = lambda key: (key[0], key[1], False, key[3], key[4])
    if any(t in outs for t in ("bw", "vnni", "vbmi")):
        assert any(k[2] for t in ("bw", "vnni", "vbmi") if t in outs for k in outs[t]), "no swizzled case ran"
    for tier, res in outs.items():
        assert {native(k) for k in res} == set(ref), f"{tier}: case set differs from avx2"
        tol = SCALAR_TOL if tier == "scalar" else INT8_TOL
        for key, out in res.items():
            assert torch.isfinite(out).all(), f"{tier} {key}: non-finite output"
            rel = rel_l2(out, ref[native(key)])
            assert rel <= tol, \
                f"tier {tier} vs avx2 differs on (K, gated, swizzled, tokens, threads) = {key}: rel {rel:.3e} > {tol}"


@pytest.fixture(scope = "module")
def ladder_dirs(model_registry) -> dict:
    """{K: dir} of the ladder checkpoints present under the model root"""
    if not model_registry.root:
        return {}
    dirs = {K: os.path.join(model_registry.root, LADDER_DIR, sub) for K, sub in LADDER.items()}
    return {K: d for K, d in dirs.items() if os.path.isfile(os.path.join(d, "config.json"))}


@pytest.fixture(scope = "module")
def ladder_tier_outputs(ladder_dirs) -> dict:
    if not ladder_dirs:
        pytest.skip(f"no lfm2.5-8b-a1b mul1 ladder checkpoint under <model root>/{LADDER_DIR}")
    return run_per_tier(ladder_outputs, ladder_dirs)


@pytest.mark.model
@pytest.mark.parametrize("K", sorted(LADDER))
def test_tiers_agree_real_weights(K, ladder_dirs, ladder_tier_outputs):
    if K not in ladder_dirs:
        pytest.skip(f"ladder checkpoint {LADDER_DIR}/{LADDER[K]} not present")
    outs = ladder_tier_outputs
    ref = outs["avx2"]
    for tier, res in outs.items():
        assert res.keys() == ref.keys(), f"{tier}: case set differs from avx2"
        for m in range(1, 5):
            out, r = res[(K, m)], ref[(K, m)]
            assert torch.isfinite(out).all() and r.abs().max() > 0, (tier, K, m)
            if tier == "scalar":
                rel = ((out - r).abs().max() / r.abs().max()).item()
                assert rel <= SCALAR_TOL, f"scalar vs avx2 at (K, m) = {(K, m)}: rel {rel:.3e}"
            else:
                assert torch.equal(out, r), \
                    f"{tier} differs from avx2 at (K, m) = {(K, m)}: max abs diff {(out - r).abs().max().item():.3e}"
