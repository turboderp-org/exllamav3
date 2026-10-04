"""
Decode-GEMV coverage at K=7/K=8 and above-cap K.

The RDNA3 port compiles int8 GEMV instances for k7/k8 that are unreachable at the
default EXL3_INT8_GEMV_MAX_K=6, and K above the cap dispatches small-m rows to
the fp16 GEMV path whose m16n8k16 MMA is __shfl-emulated on ROCm (ptx.cuh) --
none of which any existing test or gate exercises (models on hand are K<=~4).
This test builds synthetic quantized tensors (random trellis + reconstruct
reference, the test_moe_coop recipe -- no model files) and checks run_alloc
against the hadamard-sandwich fp32-accumulated reference.

The int8 K-cap is a static, env-cached C++ value, so each case runs in a
subprocess with EXL3_INT8_GEMV_MAX_K set explicitly; the child verifies via
ext.exl3_gemv_int8_max_k(0) that the override took effect, so a silently
ignored env cannot vacuously pass.

    PYTHONPATH=. python -m pytest tests/test_exl3_gemv_k78.py -q
"""
import os
import subprocess
import sys
import textwrap

import pytest
import torch

import exllamav3
# The child must import the same exllamav3 the test process uses (installed package
# with the compiled extension, or a source tree) -- not whatever cwd happens to be.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(exllamav3.__file__)))

DEV = os.environ.get("EXL3_TEST_DEVICE", "cuda:0")
REL_TOL = 0.02  # int8 GEMV family measures ~0.8% RMS; 2% gates regressions, not quantization


def _case(K, cap, m_values):
    code = textwrap.dedent(f"""
        import sys
        import torch
        from exllamav3.ext import exllamav3_ext as ext

        cap = ext.exl3_gemv_int8_max_k(0)
        if cap != {cap}:
            print(f"CAP-MISMATCH: exl3_gemv_int8_max_k(0) = {{cap}}, expected {cap} "
                  f"(env EXL3_INT8_GEMV_MAX_K not honored?)", flush=True)
            sys.exit(3)

        gen = torch.Generator().manual_seed(4242)
        k, n, K = 512, 768, {K}
        trellis = torch.randint(0, 65536, (k // 16, n // 16, 16 * K), dtype=torch.int32,
                                generator=gen).to(torch.int16).to("{DEV}")
        suh = ((torch.rand((k,), generator=gen) * 0.2 + 0.9) * torch.where(
            torch.rand((k,), generator=gen) < 0.5, -1.0, 1.0)).half().to("{DEV}")
        svh = ((torch.rand((n,), generator=gen) * 0.2 + 0.9) * torch.where(
            torch.rand((n,), generator=gen) < 0.5, -1.0, 1.0)).half().to("{DEV}")
        W = torch.empty((k, n), dtype=torch.half, device="{DEV}")
        ext.reconstruct(W, trellis, K, False, False)

        from exllamav3.modules.quant.exl3 import LinearEXL3
        mod = LinearEXL3(config=None, in_features=k, out_features=n,
                         suh=suh, svh=svh, trellis=trellis, key="synthetic_k{K}")
        torch.manual_seed(99)
        fails = []
        for m in {m_values}:
            x = (torch.randn((m, k), device="{DEV}", dtype=torch.half) * 0.5).contiguous()
            y = mod.bc.run_alloc(x, n, False)
            # reference: the dense-path evaluation (hadamard sandwich, fp32 accumulate)
            xh = torch.empty_like(x)
            ext.had_r_128(x, xh, suh, None, 1.0)
            yref = torch.empty((m, n), dtype=torch.half, device="{DEV}")
            ext.hgemm(xh, W, yref)
            ext.had_r_128(yref, yref, None, svh, 1.0)
            num = (y.float() - yref.float()).square().mean()
            den = yref.float().square().mean().clamp_min(1e-12)
            rel = float((num / den).sqrt())
            finite = bool(torch.isfinite(y.float()).all())
            print(f"K={K} cap={cap} m={{m}} rel={{rel:.5f}} finite={{finite}}", flush=True)
            if rel > {REL_TOL} or not finite:
                fails.append(m)
        sys.exit(1 if fails else 0)
    """)
    env = dict(os.environ)
    env["EXL3_INT8_GEMV_MAX_K"] = str(cap)
    env["PYTHONPATH"] = _PKG_PARENT
    return subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True, timeout=900)


@pytest.mark.parametrize("K", [7, 8])
def test_int8_gemv_k7_k8(K):
    """K=7/8 on the int8 path: forces the compiled-but-unreachable int8 GEMV instances
    (single-matrix calls go through the cooperative kernel, not msq)."""
    r = _case(K, cap=K, m_values=[1, 2, 3, 4])
    assert r.returncode == 0, f"exit {r.returncode}\n{r.stdout}\n{r.stderr[-2000:]}"


def test_above_cap_k7_fp16_path():
    """K=7 with the cap at 6: small-m rows dispatch off the int8 path (fp16 GEMV /
    emulated MMA on ROCm). Whatever the fallback is, it must match the reference."""
    r = _case(7, cap=6, m_values=[1, 2, 4])
    assert r.returncode == 0, f"exit {r.returncode}\n{r.stdout}\n{r.stderr[-2000:]}"
