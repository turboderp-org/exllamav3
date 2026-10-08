"""
Streamed prefill with the packed expert layouts (tile-order swizzle, plus the planar dword
order on the AVX2 tier; the CUDA kernels read them directly, HIP restores native order during
staging) must produce the same perplexity as the native layout (EXL3_MOE_CPU_SWIZZLE=0,
verbatim staging) within run-to-run noise, on a model whose experts all sit on the CPU.
Reference: the native-layout run. Each configuration is a separate eval/ppl.py process (the
layout is fixed when the expert worker loads).
"""

import os
import re
import subprocess
import sys

import pytest

from testlib.isolated import device_env

# Any banded tier qualifies: AVX2 packs integer rates, the AVX-512 tiers pack band-8
pytestmark = [pytest.mark.slow, pytest.mark.moe_cpu, pytest.mark.cpu_flags("avx2")]

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CPU_LAYERS = 22     # every MoE layer of the model
REL_TOL = 2e-3


def perplexity(model_dir, device, env_extra):
    env = dict(os.environ, **device_env(device), **env_extra)
    r = subprocess.run([sys.executable, "eval/ppl.py", "-m", model_dir, "-mcl", str(CPU_LAYERS), "-r", "4", "-l", "2048"],
                       cwd = REPO_DIR, env = env, capture_output = True, text = True)
    m = re.search(r"-- Perplexity:\s*([0-9.]+)", r.stdout)
    assert r.returncode == 0 and m, r.stdout[-1500:] + r.stderr[-3000:]
    return float(m.group(1))


@pytest.mark.model("moe-hybrid")
def test_swizzled_streaming_matches_native_perplexity(model_dir, device):
    swizzled = perplexity(model_dir, device, {"EXL3_MOE_CPU_SWIZZLE": "1"})    # swizzled arena + GPU unswizzle
    native = perplexity(model_dir, device, {"EXL3_MOE_CPU_SWIZZLE": "0"})      # native arena, verbatim staging
    assert abs(swizzled - native) / native < REL_TOL, (swizzled, native)
