"""
Parity of the vendored forward-only flash-linear-attention kernels (exllamav3/vendor/fla) against the installed
fla package, on the shapes exllamav3 uses for prefill: bsz 1 per recurrent slot, bf16 q/k/v, fp32 gates and
states.

The kernel sources are identical, so results are bit-exact whenever both sides run the same tile config. The two
copies are separate kernel objects and tune independently, and near-tied candidates can land differently (on
Blackwell the gate cumsum picks BS 16 on one side and 32 on the other, which reorders the fp32 cumulative sums and
moves the state by ~5e-4 relative through the exponentials), so each test runs the vendored op first and pins the
installed kernels to the vendored picks before running the reference. A different BK/BV pick would otherwise
change the fp32 accumulation order (e.g. chunk_fwd_kernel_o BK 128 vs 64 gives one bf16 ulp in the output); the
check allows that much and no more. Skips when flash-linear-attention is not installed.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope = "module")
def ops():
    """(reference fla ops, vendored ops); both query the devices and Triton at import"""
    pytest.importorskip("fla")
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.kda import chunk_kda
    from fla.ops.simple_gla import chunk_simple_gla
    import exllamav3.vendor.fla as vend
    ref = SimpleNamespace(chunk_gated_delta_rule = chunk_gated_delta_rule, chunk_kda = chunk_kda,
                          chunk_simple_gla = chunk_simple_gla)
    return ref, vend


def _unwrap(o):
    from triton.runtime.autotuner import Autotuner
    from exllamav3.vendor.fla.utils import DeviceAutotune
    for _ in range(6):
        if isinstance(o, (Autotuner, DeviceAutotune)) or not hasattr(o, "fn"):
            break
        o = o.fn
    return o


def _kernel_name(o):
    # autotune and heuristics wrap each other in either order; the raw Python function is at the end of the chain
    for _ in range(8):
        if not hasattr(o, "fn"):
            break
        o = o.fn
    return getattr(o, "__name__", None)


def pin_reference_to_vendored():
    """Give every installed fla kernel the tile config its vendored twin just tuned to, so the comparison is between
    the kernels and not between two autotune runs"""
    import sys
    from triton.runtime.autotuner import Autotuner
    from exllamav3.vendor.fla.utils import DeviceAutotune
    picks = {}
    for name, mod in list(sys.modules.items()):
        if name.startswith("exllamav3.vendor.fla"):
            for o in list(vars(mod).values()):
                o = _unwrap(o)
                if isinstance(o, DeviceAutotune):
                    for tuner in o._tuners.values():
                        if tuner.cache:
                            picks[_kernel_name(o)] = list(tuner.cache.values())[-1]
    for name, mod in list(sys.modules.items()):
        if name.startswith("fla.ops"):
            for o in list(vars(mod).values()):
                o = _unwrap(o)
                if isinstance(o, Autotuner) and _kernel_name(o) in picks:
                    o.configs = [picks[_kernel_name(o)]]
                    o.cache.clear()


def check(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    if torch.equal(a, b):
        return
    # One bf16 ulp of the output scale for o (bf16), fp32 accumulation noise for the state
    d = (a.float() - b.float()).abs().max().item()
    lim = 2 ** -8 * b.float().abs().max().item() if a.dtype == torch.bfloat16 else 1e-5 * b.float().abs().max().item()
    assert d <= lim, f"max |diff| {d:.3e} > {lim:.3e}"


@pytest.mark.parametrize("T", [64, 100, 1000, 2048, 4097])
@pytest.mark.parametrize("with_state", [False, True])
def test_gated_delta_rule(device, ops, T, with_state):
    ref, vend = ops
    torch.manual_seed(T)
    B, H, HV, K, V = 1, 16, 32, 128, 128
    q = torch.randn(B, T, H, K, device = device, dtype = torch.bfloat16)
    k = torch.randn(B, T, H, K, device = device, dtype = torch.bfloat16)
    v = torch.randn(B, T, HV, V, device = device, dtype = torch.bfloat16)
    g = F.logsigmoid(torch.randn(B, T, HV, device = device, dtype = torch.float32)) * 2
    beta = torch.rand(B, T, HV, device = device, dtype = torch.bfloat16)
    h0 = torch.randn(B, HV, K, V, device = device, dtype = torch.float32) * 0.1 if with_state else None
    kw = dict(g = g, beta = beta, initial_state = h0, output_final_state = True, use_qk_l2norm_in_kernel = True)
    o2, s2 = vend.chunk_gated_delta_rule(q, k, v, **kw)
    pin_reference_to_vendored()
    o1, s1 = ref.chunk_gated_delta_rule(q, k, v, **kw)
    check(o2, o1)
    check(s2, s1)


@pytest.mark.parametrize("T", [64, 333, 2048])
@pytest.mark.parametrize("with_state", [False, True])
def test_kda(device, ops, T, with_state):
    ref, vend = ops
    torch.manual_seed(T)
    B, H, HV, K, V = 1, 32, 32, 128, 128
    q = torch.randn(B, T, H, K, device = device, dtype = torch.bfloat16)
    k = torch.randn(B, T, H, K, device = device, dtype = torch.bfloat16)
    v = torch.randn(B, T, HV, V, device = device, dtype = torch.bfloat16)
    g = F.logsigmoid(torch.randn(B, T, HV, K, device = device, dtype = torch.float32)) * 2
    beta = torch.rand(B, T, HV, device = device, dtype = torch.bfloat16)
    h0 = torch.randn(B, HV, K, V, device = device, dtype = torch.float32) * 0.1 if with_state else None
    kw = dict(g = g, beta = beta, initial_state = h0, output_final_state = True, use_qk_l2norm_in_kernel = True)
    o2, s2 = vend.chunk_kda(q, k, v, **kw)
    pin_reference_to_vendored()
    o1, s1 = ref.chunk_kda(q, k, v, **kw)
    check(o2, o1)
    check(s2, s1)


@pytest.mark.parametrize("T", [64, 1000, 2048])
@pytest.mark.parametrize("with_state", [False, True])
def test_simple_gla(device, ops, T, with_state):
    ref, vend = ops
    torch.manual_seed(T)
    B, H, K, V = 1, 64, 128, 64
    q = torch.randn(B, T, H, K, device = device, dtype = torch.bfloat16)
    k = torch.randn(B, T, H, K, device = device, dtype = torch.bfloat16) * 0.1
    v = torch.randn(B, T, H, V, device = device, dtype = torch.bfloat16)
    g = F.logsigmoid(torch.randn(B, T, H, device = device, dtype = torch.float32)) * 2
    h0 = torch.randn(B, H, K, V, device = device, dtype = torch.float32) * 0.1 if with_state else None
    kw = dict(g = g, scale = 1.0, initial_state = h0, output_final_state = True)
    o2, s2 = vend.chunk_simple_gla(q, k, v, **kw)
    pin_reference_to_vendored()
    o1, s1 = ref.chunk_simple_gla(q, k, v, **kw)
    check(o2, o1)
    check(s2, s1)
