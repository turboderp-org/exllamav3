"""
CPU MoE expert kernel (cpu/moe_mul1.cpp): every ISA tier the CPU can run must agree. The int8
tiers (avx2, bw, vnni, vbmi) compute the same integer dot products from the same int8 activations,
so they agree to float rounding of the epilogue; the scalar tier is an fp32 reference without the
int8 activation quantization and agrees to ~1%. Any state-extraction or accumulate bug in a tier
shows up as O(1) error, far outside both tolerances.

The tier is fixed per process by EXL3_MOE_CPU_MAX_ISA (read once at static init), so each tier
runs in a subprocess; on a VBMI machine that exercises vbmi, vnni, bw, avx2 and scalar in one run.
Covers K1-8, gated and gateless experts, 1..5 tokens (m = 1..4 rows per expert chunk), the
swizzled layout, and a 256-token case that takes the GEMV phases' many-GEMV (strided) regime.

A second test repeats the comparison on real expert weights from the lfm2.5-8b-a1b mul1 ladder
(one K per model, layer 2, single-threaded so the accumulation order is fixed): there the int8
tiers must be bit-identical, since real trellis statistics can expose extraction bugs that
random states miss. Skipped when the ladder is not present.
"""
import os, sys, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INT8_TOL = 5e-4     # rel. L2 between int8 tiers (observed <= 1e-4 under -Ofast; 0 with strict FP)
SCALAR_TOL = 0.05   # rel. L2 int8 tiers vs the fp32 scalar reference (observed <= 0.02)


def _worker(tier, out_path):
    os.environ["EXL3_MOE_CPU_MAX_ISA"] = tier
    import torch
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

    # The swizzled (band-contiguous) layout is only ever handed to the tiers that consume it (the
    # host gates it on has_avx512_vbmi/has_avx512_bw); scalar and avx2 read the native layout only
    swz_capable = tier in ("bw", "vnni", "vbmi", "avx512", "avx512bw")

    def swizzled(t, K):
        if K == 8: return t   # K8 tensors stay native (make_matrix exempts them too)
        tk, tn, ps = t.shape
        return t.view(tk, tn // 8, 8, ps).permute(1, 0, 2, 3).contiguous().view(tk, tn, ps)

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
                sw = lambda ts: [swizzled(t, K) for t in ts]
                layers.append((True, ext.exl3_moe_cpu_make_layer(sw(gt), gs, gv, sw(ut), us, uv, sw(dt), ds, dv,
                                                                 [], [], [], 0 if gated else 2, 0.0, 1)))
            cases = []
            for tokens in (1, 2, 3, 4, 5, 256):
                x = torch.randn(tokens, hid, generator = g).half()
                # 1..5 tokens on the same experts -> chunks of m = 1..4 rows; 256 tokens spread over
                # all experts -> many chunks per expert -> the strided GEMV assignment regime
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
    torch.save(results, out_path)


LADDER_ROOT = os.environ.get("EXL3_MOE_TIER_LADDER", "/mnt/str/models/lfm2.5-8b-a1b/exl3")
LADDER = {1: "1.10bpw_mul1", 2: "2.10bpw_mul1", 3: "3.10bpw_mul1", 4: "4.10bpw_mul1",
          5: "5.10bpw_mul1", 6: "6.10bpw_mul1", 7: "7.06bpw_mul1", 8: "8.00bpw_mul1"}
LADDER_LAYER, LADDER_E, LADDER_TOPK = 2, 4, 2


def _ladder_present():
    return all(os.path.isdir(os.path.join(LADDER_ROOT, sub)) for sub in LADDER.values())


def _worker_real(tier, out_path):
    """Real-weight cases: K1-8 from the ladder, m = 1..4 rows, single thread, native layout."""
    os.environ["EXL3_MOE_CPU_MAX_ISA"] = tier
    import json, torch
    from safetensors import safe_open
    from exllamav3.ext import exllamav3_ext as ext
    torch.manual_seed(0)
    results = {}
    for bits, sub in LADDER.items():
        d = os.path.join(LADDER_ROOT, sub)
        idx = os.path.join(d, "model.safetensors.index.json")
        wm = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None
        handles = {}
        def get(k):
            fn = wm[k] if wm else "model.safetensors"
            if fn not in handles: handles[fn] = safe_open(os.path.join(d, fn), "pt")
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
        assert g[0][0].shape[2] // 16 == bits, (sub, g[0][0].shape)
        H = g[0][1].numel()
        h = ext.exl3_moe_cpu_make_layer(
            [t[0] for t in g], [t[1] for t in g], [t[2] for t in g],
            [t[0] for t in u], [t[1] for t in u], [t[2] for t in u],
            [t[0] for t in dn], [t[1] for t in dn], [t[2] for t in dn],
            [], [], [], 0, 0.0, 0)
        for m in range(1, 5):
            x = torch.randn(m, H).half()
            sel = torch.stack([torch.randperm(LADDER_E)[:LADDER_TOPK] for _ in range(m)]).int()
            w = torch.rand(m, LADDER_TOPK).float()
            out = torch.zeros(m, H, dtype = torch.float)
            ext.exl3_moe_cpu_forward(h, x, sel, w, out, 1)
            results[(bits, m)] = out.clone()
        ext.exl3_moe_cpu_free_layer(h)
    torch.save(results, out_path)


def _supported_tiers():
    from exllamav3.ext import exllamav3_ext as ext
    tiers = ["scalar"]
    if ext.exl3_moe_cpu_has_avx2(): tiers.append("avx2")
    if getattr(ext, "exl3_moe_cpu_has_avx512_bw", lambda: False)(): tiers.append("bw")
    if ext.exl3_moe_cpu_has_avx512_vnni(): tiers.append("vnni")
    if ext.exl3_moe_cpu_has_avx512_vbmi(): tiers.append("vbmi")
    return tiers


def test_cpu_moe_tiers_agree():
    import torch
    tiers = _supported_tiers()
    assert "avx2" in tiers, "no vector tier available; nothing to compare"
    with tempfile.TemporaryDirectory() as td:
        outs = {}
        for tier in tiers:
            path = os.path.join(td, f"{tier}.pt")
            env = dict(os.environ, EXL3_MOE_CPU_MAX_ISA = tier)
            subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", tier, path],
                           env = env, check = True)
            outs[tier] = torch.load(path)
        ref = outs["avx2"]   # native layout only; swizzled results compare against the same weights natively
        native = lambda key: (key[0], key[1], False, key[3], key[4])
        for tier, res in outs.items():
            assert all(native(k) in ref for k in res), f"{tier}: case set differs from avx2"
            tol = SCALAR_TOL if tier == "scalar" else INT8_TOL
            worst, exact = 0.0, 0
            for key, out in res.items():
                assert torch.isfinite(out).all(), f"{tier} {key}: non-finite output"
                r = ref[native(key)]
                rel = ((out - r).norm() / (r.norm() + 1e-12)).item()
                worst = max(worst, rel)
                exact += torch.equal(out, r)
                assert rel <= tol, f"tier {tier} vs avx2 differs on (K, gated, swizzled, tokens, threads) = {key}: rel {rel:.3e} > {tol}"
            nswz = sum(1 for k in res if k[2])
            print(f" -- {tier:6}: {len(res)} cases ({nswz} swizzled), worst rel {worst:.2e} vs avx2, {exact} bit-exact")


def test_cpu_moe_tiers_real_weights():
    import pytest, torch
    if not _ladder_present():
        pytest.skip(f"lfm2.5 mul1 ladder not found under {LADDER_ROOT}")
    tiers = _supported_tiers()
    assert "avx2" in tiers, "no vector tier available; nothing to compare"
    with tempfile.TemporaryDirectory() as td:
        outs = {}
        for tier in tiers:
            path = os.path.join(td, f"{tier}.pt")
            env = dict(os.environ, EXL3_MOE_CPU_MAX_ISA = tier)
            subprocess.run([sys.executable, os.path.abspath(__file__), "--worker-real", tier, path],
                           env = env, check = True)
            outs[tier] = torch.load(path)
        ref = outs["avx2"]
        worst_scalar = 0.0
        for tier, res in outs.items():
            assert res.keys() == ref.keys(), f"{tier}: case set differs from avx2"
            for key, out in res.items():
                r = ref[key]
                assert torch.isfinite(out).all() and r.abs().max() > 0, (tier, key)
                if tier == "scalar":
                    rel = ((out - r).abs().max() / r.abs().max()).item()
                    worst_scalar = max(worst_scalar, rel)
                    assert rel <= SCALAR_TOL, f"scalar vs avx2 at (K, m) = {key}: rel {rel:.3e}"
                else:
                    assert torch.equal(out, r), \
                        f"{tier} differs from avx2 at (K, m) = {key}: max abs diff {(out - r).abs().max().item():.3e}"
        print(f" -- real weights: {[t for t in tiers if t != 'scalar']} bit-identical over K1-8 x m1-4; "
              f"scalar worst rel {worst_scalar:.2e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1 and sys.argv[1] == "--worker-real":
        _worker_real(sys.argv[2], sys.argv[3])
    else:
        test_cpu_moe_tiers_agree()
        if _ladder_present():
            test_cpu_moe_tiers_real_weights()
        print("PASS")
