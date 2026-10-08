"""
CPU expert kernels (ext.exl3_moe_cpu_*) at every trellis rate, half-integer (1.5 / 2.5 / 3.5 bpw, mul1) and integer,
native and band-swizzled layouts: the rate decode of a trellis width (trellis_rate), each layer's forward against
the GPU dense reconstruct path (testlib.moe.expert_mlp_ref), and agreement between the instruction set tiers (the
integer tiers bit-identical, the scalar fp32-activation tier to the int8 rounding). The GPU fused kernels at these
rates are covered in tests/kernels/moe/test_moe_half_rates.py.
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.block_sparse_mlp_cpu import trellis_rate
from testlib.exl3 import rand_experts
from testlib.moe import expert_mlp_ref
from testlib.moe_cpu import INT8_TIERS, run_per_tier, swizzle, swizzle_layouts

ALL_RATES = (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)

# Expert e gets e + 1 tokens: the CPU kernels take a group of 1 to 4 rows per call
CPU_E, CPU_H, CPU_I = 4, 512, 256
CPU_SEL = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]


def cpu_case(K, swizzled, gen):
    """One random layer through the CPU forward. Returns (experts, x, sel, w, out)"""
    ex = rand_experts(CPU_E, CPU_H, CPU_I, K, gen)
    T = len(CPU_SEL)
    x = (torch.randn((T, CPU_H), generator = gen) * 0.05).half()
    sel = torch.tensor(CPU_SEL, dtype = torch.long).view(T, 1)
    w = (torch.rand((T, 1), generator = gen) * 0.5 + 0.5).half()

    def packed(t):
        return swizzle(t) if swizzled else t.contiguous()

    lists = []
    for p in ("g", "u", "d"):
        lists += [[packed(e[0]) for e in ex[p]], [e[1].contiguous() for e in ex[p]], [e[2].contiguous() for e in ex[p]]]
    h = ext.exl3_moe_cpu_make_layer(*lists, [], [], [], 0, 0.0, 1 if swizzled else 0)
    out = torch.empty((T, CPU_H), dtype = torch.float32)
    ext.exl3_moe_cpu_forward(h, x.contiguous(), sel.contiguous(), w.contiguous(), out, 4)
    ext.exl3_moe_cpu_free_layer(h)
    return ex, x, sel, w, out


def cpu_outputs():
    """Every rate and layout, same seeds in every process"""
    outs = {}
    for K in ALL_RATES:
        for swz in swizzle_layouts():
            gen = torch.Generator().manual_seed(int(K * 10))
            outs[f"{K}_{int(swz)}"] = cpu_case(K, swz, gen)[4]
    return outs


@pytest.mark.nogpu
def test_trellis_rate():
    assert [trellis_rate(16 * k) for k in range(1, 9)] == list(range(1, 9))
    assert [trellis_rate(w) for w in (24, 40, 56)] == [1.5, 2.5, 3.5]
    for w in (0, 8, 72, 136, 144, 20, 60):
        assert trellis_rate(w) is None


@pytest.mark.parametrize("K", ALL_RATES)
def test_cpu_kernel_matches_gpu_reference(device, K):
    for swz in swizzle_layouts():
        gen = torch.Generator().manual_seed(int(K * 10))
        ex, x, sel, w, out = cpu_case(K, swz, gen)
        ref = torch.zeros((len(CPU_SEL), CPU_H), dtype = torch.float, device = device)
        for e in range(CPU_E):
            tk = (sel[:, 0] == e).nonzero().flatten()
            ref[tk.to(device)] = expert_mlp_ref(ex, e, x[tk].to(device), K) * w[tk].float().to(device)
        err = ((out.to(device) - ref).norm(dim = 1) / ref.norm(dim = 1)).max().item()
        # int8 activations: about a percent per call; a wrong state extraction gives order one
        assert err < 0.04, f"K {K} swizzled {swz}: row error {err}"


@pytest.mark.nogpu
@pytest.mark.cpu_flags("avx2")
def test_cpu_tiers_agree():
    """The integer tiers (AVX2, AVX-512 BW / VNNI / VBMI) are bit-identical, the scalar tier (fp32 activations)
    agrees to the int8 rounding. Each tier runs in its own process: the ISA cap is read once"""
    outs = run_per_tier(cpu_outputs)
    tiers = [t for t in INT8_TIERS if t in outs]
    if not tiers:
        pytest.skip("no vector tier in this build")
    base = outs[tiers[0]]
    assert any(k.startswith("3.5_") for k in base)
    # Each case against the lowest tier that ran it (packed cases exist from AVX2 up).
    first = {}
    for t in tiers:
        for k, v in outs[t].items():
            if k in first:
                assert torch.equal(v, outs[first[k]][k]), f"tier {t} differs from {first[k]} at {k}"
            else:
                first[k] = t
    for k, v in outs["scalar"].items():
        err = ((v - base[k]).norm(dim = 1) / v.norm(dim = 1)).max().item()
        assert err < 0.04, f"scalar tier vs {tiers[0]} at {k}: {err}"
