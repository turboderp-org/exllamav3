"""
Helpers for the CPU expert-offload kernels (cpu/moe_mul1.cpp).

ISA tiers: the kernel picks the highest instruction-set tier the CPU supports, capped by EXL3_MOE_CPU_MAX_ISA,
which is read once per process. Comparing tiers therefore means one child process per tier:

    outs = run_per_tier(worker, *args)           # {tier: worker(*args) as run under that cap}

Every tier this CPU can run is covered ("scalar" always; the int8 tiers "avx2", "bw", "vnni", "vbmi" when the
CPU has them), and each child's tier is checked against the cap it ran under. `worker` follows the rules of
testlib.isolated.run_isolated (module-level function, picklable arguments, torch.save-able result).

Layout: each tier's packed trellis layout is decided per rate by the kernel rules
(`swizzle`, mirroring the child loader through exl3_moe_cpu_swizzle_group /
exl3_moe_cpu_planar_layout); rates the active tier leaves native are returned unchanged.

Runtime: the native worker pool can pin its caller's thread and the tests limit torch's own threads around it;
the `cpu_runtime` fixture restores the process affinity and torch thread count afterwards. Import it into the
test module to use it (`from testlib.moe_cpu import cpu_runtime`).
"""

import importlib.util
import inspect
import os

import pytest
import torch

from testlib.isolated import run_isolated
from testlib.moe import repack_trellis

INT8_TIERS = ("avx2", "bw", "vnni", "vbmi")
TIERS = ("scalar",) + INT8_TIERS


def active_tier() -> str:
    """The tier the kernels run at in this process"""
    from exllamav3.ext import exllamav3_ext as ext
    if ext.exl3_moe_cpu_has_avx512_vbmi(): return "vbmi"
    if ext.exl3_moe_cpu_has_avx512_vnni(): return "vnni"
    if ext.exl3_moe_cpu_has_avx512_bw(): return "bw"
    if ext.exl3_moe_cpu_has_avx2(): return "avx2"
    return "scalar"


def supported_tiers() -> list[str]:
    """Tiers this process can run, lowest first (up to the hardware's, or a cap set in the test environment)"""
    return list(TIERS[: TIERS.index(active_tier()) + 1])


def _tier_and_result(func_path, func_name, args, kwargs):
    spec = importlib.util.spec_from_file_location("_exl3_tier_module", func_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return active_tier(), getattr(module, func_name)(*args, **kwargs)


def run_per_tier(func, *args, tiers = None, env: dict | None = None, **kwargs) -> dict:
    """{tier: func(*args, **kwargs)} with each tier in its own process (default: every supported tier)"""
    path = os.path.abspath(inspect.getsourcefile(func))
    out = {}
    for tier in tiers or supported_tiers():
        got, result = run_isolated(_tier_and_result, path, func.__name__, args, kwargs,
                                   env = dict(env or {}, EXL3_MOE_CPU_MAX_ISA = tier))
        assert got == tier, f"EXL3_MOE_CPU_MAX_ISA = {tier} ran at tier {got}"
        out[tier] = result
    return out


ALL_PACK_RATES = (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)


def swizzle(t: torch.Tensor) -> torch.Tensor:
    """Packed layout of a [k/16, n/16, 16K] trellis for this process's tier and this tensor's
    rate, exactly what the child loader applies (rates the tier leaves native come back
    contiguous)"""
    from exllamav3.ext import exllamav3_ext as ext
    K = t.shape[-1] / 16.0
    g = ext.exl3_moe_cpu_swizzle_group(K)
    return repack_trellis(t, g, bool(ext.exl3_moe_cpu_planar_layout(K))) if g else t.contiguous()


def swizzle_layouts() -> tuple[bool, ...]:
    """(native,) or (native, packed): the layouts the active tier takes"""
    from exllamav3.ext import exllamav3_ext as ext
    return (False, True) if any(ext.exl3_moe_cpu_swizzle_group(K) for K in ALL_PACK_RATES) else (False,)


@pytest.fixture
def cpu_runtime():
    """Single-threaded torch around the native pool; process affinity and torch threads restored afterwards"""
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        if affinity is not None:
            os.sched_setaffinity(0, affinity)
        torch.set_num_threads(threads)
