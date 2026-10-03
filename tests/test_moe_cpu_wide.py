"""
CPU MoE WIDE 15-bit two-row activation quantization (cpu/moe_mul1.cpp).

A row goes WIDE when more than 1/16 of its activations would round to zero
under per-row amax int8 quantization (a dominant component + small rest --
the MiMo saturated-layer case; upstream commit 0f62eedd). The path is
DEFAULT-ON (EXL3_MOE_CPU_WIDE=0 disables), yet Gaussian test inputs never
trigger it: 'zeros * 16 > k' needs a row whose amax dwarfs >15/16 of entries,
which randn never produces. This test fabricates exactly such rows and checks
the wide output against the scalar tier (fp32, no activation quantization --
the function all tiers approximate), plus that the fixture actually engaged
the path (wide differs from plain int8 on poisoned rows) and left clean rows
bit-identical.

All EXL3_MOE_CPU_* knobs are read at static init, so each variant runs in a
subprocess. CPU-only: no GPU required.

    PYTHONPATH=. python -m pytest tests/test_moe_cpu_wide.py -q
"""
import os
import subprocess
import sys
import textwrap

import pytest

import exllamav3
import torch

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(exllamav3.__file__)))

HID, INTER, E, TOPK = 512, 640, 4, 2
KS = [4, 8]           # K8 also covers the "K8 tensors stay native" exemption
POISON = [0, 1]       # rows with a dominant component
CLEAN = [2, 3]

_WORKER = textwrap.dedent(f"""
    import os, sys
    import torch

    def main():
        wide_env = os.environ["EXL3_MOE_CPU_WIDE"]
        isa = os.environ.get("EXL3_MOE_CPU_MAX_ISA", "avx2")
        import exllamav3  # ensure the package dir (and its ext) resolves
        from exllamav3.ext import exllamav3_ext as ext
        g = torch.Generator().manual_seed(1234)

        def trellis(k, n, K):
            return torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K),
                                 dtype=torch.int16, generator=g)

        def suh(n):
            s = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
            return (s * (0.015 + 0.004 * torch.randn(n, generator=g))).half().contiguous()

        def svh(n):
            s = torch.randint(0, 2, (n,), generator=g).float() * 2 - 1
            return (s * (1.0 + 0.1 * torch.randn(n, generator=g))).half().contiguous()

        results = {{}}
        tokens = 4
        # route every token to the same expert pair with uniform weights
        sel = torch.tensor([[0, 1]] * tokens, dtype=torch.long)
        for K in {KS}:
            ut = [trellis({HID}, {INTER}, K) for _ in range({E})]
            us = [suh({HID}) for _ in range({E})]
            uv = [svh({INTER}) for _ in range({E})]
            dt = [trellis({INTER}, {HID}, K) for _ in range({E})]
            ds = [suh({INTER}) for _ in range({E})]
            dv = [svh({HID}) for _ in range({E})]
            h = ext.exl3_moe_cpu_make_layer([], [], [], ut, us, uv, dt, ds, dv,
                                            [], [], [], 2, 0.0, 0)
            # poisoned rows: one dominant element, everything else ~amax/10^4
            # -> nearly all entries round to zero under amax/254 int8 quant
            x = torch.randn(tokens, {HID}, generator=g).half() * 0.01
            x[{POISON}, 0] = 100.0
            # uniform weights over the (identical-output) expert pair
            w = torch.full((tokens, {TOPK}), 1.0 / {TOPK}).half()
            out = torch.zeros(tokens, {HID}, dtype=torch.float32)
            ext.exl3_moe_cpu_forward(h, x.contiguous(), sel, w.contiguous(), out, 1)
            ext.exl3_moe_cpu_free_layer(h)
            results[K] = out
        torch.save(results, sys.argv[1])

    main()
""")


def _run(tag, extra_env):
    out = f"/tmp/moe_cpu_wide_{tag}.pt"
    env = dict(os.environ)
    env.update(extra_env)
    env["PYTHONPATH"] = _PKG_PARENT
    r = subprocess.run([sys.executable, "-c", _WORKER, out], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"{tag} worker failed:\n{r.stdout}\n{r.stderr[-2000:]}"
    import torch
    return torch.load(out, weights_only=False)


@pytest.mark.parametrize("K", KS)
def test_wide_rows_match_scalar_oracle(K):
    """Wide output on poisoned rows must approximate the fp32 scalar-tier
    reference; plain int8 on the same rows collapses (that's why wide exists)."""
    wide = _run("wide", {"EXL3_MOE_CPU_WIDE": "1", "EXL3_MOE_CPU_MAX_ISA": "avx2"})
    plain = _run("plain", {"EXL3_MOE_CPU_WIDE": "0", "EXL3_MOE_CPU_MAX_ISA": "avx2"})
    scal = _run("scalar", {"EXL3_MOE_CPU_WIDE": "1", "EXL3_MOE_CPU_MAX_ISA": "scalar"})

    def rel(a, b):
        d = (a.double() - b.double()).square().sum()
        den = b.double().square().sum().clamp_min(1e-12)
        return float((d / den).sqrt())

    for K in KS:
        for t in POISON:
            rw = rel(wide[K][t], scal[K][t])
            rp = rel(plain[K][t], scal[K][t])
            assert rw < 0.05, f"K={K} row{t}: wide vs scalar rel {rw:.4f} (should approximate the oracle)"
            assert rw < rp, f"K={K} row{t}: wide ({rw:.4f}) no better than plain int8 ({rp:.4f}) - fixture may not trigger the wide path"
            print(f"K={K} row{t}: wide-vs-scalar {rw:.5f}, plain-vs-scalar {rp:.5f}")
        # clean rows: wide and plain quantize identically (row not wide) -> bit-identical
        for t in CLEAN:
            assert torch.equal(wide[K][t], plain[K][t]), \
                f"K={K} clean row{t}: wide path must not touch non-wide rows (outputs differ)"
