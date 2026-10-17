"""
CPU expert kernels on inputs that int8 activations cannot hold: a component shared by every token, large and
confined to a few dimensions, over a small part that differs between tokens. The rotation keeps the large
component inside the 128-blocks it lives in, the row's int8 scale follows it, and most of the other elements
round to zero. The kernels carry such a row as two int8 rows (high and low part of a 15-bit value).

The test layer is built the way such layers are: the shared component puts every gate deep into saturation
(here through a gate bias that cancels its response and leaves -8), so an error of the pre-activation
shows in the output as its exponential.

Also here: gates so far into saturation that the exponential of the pre-activation overflows.

Reference: an fp64 expert forward on the GPU from the dequantized trellis, and the same cases with every row
as plain int8 (EXL3_MOE_CPU_WIDE=0, a child process: the switch is read once).
"""
import pytest
import torch
import torch.nn.functional as F

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.util.hadamard import get_hadamard_dt
from testlib.env import get_test_device
from testlib.exl3 import rand_experts
from testlib.isolated import run_isolated
from testlib.moe_cpu import INT8_TIERS, cpu_runtime, run_per_tier, supported_tiers, swizzle, swizzle_layouts   # noqa: F401 (fixture)

pytestmark = pytest.mark.usefixtures("cpu_runtime")

DEV = get_test_device()
E, H, I = 4, 512, 256
RATES = (2, 3.5, 4, 8)
# Expert e gets e + 1 tokens: with every row wide that is 2 to 8 GEMV rows per chunk, the last ones in a
# second pass over the weights
SEL = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]


def shared_component():
    """Three large elements in two blocks"""
    c = torch.zeros(H)
    c[5], c[70], c[300] = 60.0, -35.0, 20.0
    return c


def inputs(gen, shared):
    """Token rows, on top of the shared component or on their own"""
    x = torch.randn((len(SEL), H), generator = gen) * 0.05
    return (x + shared_component() if shared else x).half()


def linear(ex, p, e, v, K):
    """A projection of one expert in fp64: dense weights from the trellis, rotations and scales around them"""
    Hd = get_hadamard_dt(128, DEV, torch.float64, 128 ** -0.5)
    def rot(t):
        return (t.reshape(t.shape[0], -1, 128) @ Hd).reshape(t.shape[0], -1)
    trellis, suh, svh = ex[p][e][:3]
    W = torch.empty((trellis.shape[0] * 16, trellis.shape[1] * 16), dtype = torch.half, device = DEV)
    ext.reconstruct(W, trellis.to(DEV), K, False, True)
    return rot(rot(v * suh.to(DEV).double()) @ W.double()) * svh.to(DEV).double()


def reference(ex, e, x, K):
    xd = x.to(DEV).double()
    g = linear(ex, "g", e, xd, K) + ex["gb"][e].to(DEV).double()
    u = linear(ex, "u", e, xd, K) + ex["ub"][e].to(DEV).double()
    return linear(ex, "d", e, F.silu(g) * u, K)


def make_experts(K, gen, saturate = True, peaked = False):
    """
    Random experts. With `saturate`, gate biases that leave -8 of the shared component's response. With
    `peaked`, three neurons that are always open and large, which puts the down projection's input in the
    same position as the shared component puts the input of the other two
    """
    ex = rand_experts(E, H, I, K, gen)
    c = shared_component().half().to(DEV).double().view(1, -1)
    ex["gb"] = [
        (-8.0 - linear(ex, "g", e, c, K))[0].half().cpu() if saturate else torch.zeros(I, dtype = torch.half)
        for e in range(E)
    ]
    ex["ub"] = [torch.zeros(I, dtype = torch.half) for _ in range(E)]
    if peaked:
        for e in range(E):
            # (all in the first of the two blocks of the down projection's input)
            ex["gb"][e][[3, 40, 100]] = 6.0
            ex["ub"][e][[3, 40, 100]] = torch.tensor([300.0, -200.0, 100.0], dtype = torch.half)
    return ex


def run(K, shared, per_call, swizzled = False, peaked = False):
    gen = torch.Generator().manual_seed(int(K * 10) + 1000)
    ex = make_experts(K, gen, saturate = shared, peaked = peaked)
    x = inputs(gen, shared)
    T = len(SEL)
    sel = torch.tensor(SEL, dtype = torch.long).view(T, 1)
    w = (torch.rand((T, 1), generator = gen) * 0.5 + 0.5).half()
    packed = swizzle if swizzled else torch.Tensor.contiguous
    lists = []
    for p in ("g", "u", "d"):
        lists += [[packed(v[0]) for v in ex[p]], [v[1].contiguous() for v in ex[p]], [v[2].contiguous() for v in ex[p]]]
    h = ext.exl3_moe_cpu_make_layer(*lists, ex["gb"], ex["ub"], [], 0, 0.0, 1 if swizzled else 0)
    out = torch.empty((T, H), dtype = torch.float32)
    for a in range(0, T, per_call):
        o = torch.empty((min(per_call, T - a), H), dtype = torch.float32)
        ext.exl3_moe_cpu_forward(h, x[a : a + per_call].contiguous(), sel[a : a + per_call].contiguous(),
                                 w[a : a + per_call].contiguous(), o, 4)
        out[a : a + per_call] = o
    ext.exl3_moe_cpu_free_layer(h)
    return ex, x, sel, w, out


def row_errors(K, ex, x, sel, w, out):
    ref = torch.zeros((len(SEL), H), dtype = torch.float64, device = DEV)
    for e in range(E):
        tk = (sel[:, 0] == e).nonzero().flatten()
        ref[tk.to(DEV)] = reference(ex, e, x[tk], K) * w[tk].double().to(DEV)
    return ((out.to(DEV).double() - ref).norm(dim = 1) / ref.norm(dim = 1)).cpu()


def outputs():
    """What the kernels return for every case, same seeds in every process"""
    outs = {}
    for K in RATES:
        for swz in swizzle_layouts():
            for per_call in (1, len(SEL)):
                for shared in (False, True):
                    outs[f"{K}_{int(swz)}_{int(shared)}_{per_call}"] = run(K, shared, per_call, swz)[4]
                outs[f"{K}_{int(swz)}_peaked_{per_call}"] = run(K, False, per_call, swz, peaked = True)[4]
    return outs


@pytest.fixture(scope = "module")
def plain():
    """The same cases with every row as plain int8"""
    return run_isolated(outputs, env = {"EXL3_MOE_CPU_WIDE": "0"})


@pytest.mark.cpu_flags("avx2")   # (the scalar tier takes float activations)
@pytest.mark.parametrize("K", RATES)
@pytest.mark.parametrize("per_call", (1, len(SEL)))
def test_shared_component(K, per_call, plain):
    for swz in swizzle_layouts():
        ex, x, sel, w, out = run(K, True, per_call, swz)
        err = row_errors(K, ex, x, sel, w, out)
        assert torch.isfinite(out).all()
        assert err.max() < 0.05, f"K {K} swizzled {swz}: row error {err.max():.4f}"
        # The inputs are of the kind that plain int8 rows get wrong
        before = row_errors(K, ex, x, sel, w, plain[f"{K}_{int(swz)}_1_{per_call}"])
        assert before.median() > 0.2, f"K {K}: plain rows {before.median():.4f}, wide rows {err.max():.4f}"


@pytest.mark.cpu_flags("avx2")   # (the scalar tier takes float activations)
@pytest.mark.parametrize("K", RATES)
@pytest.mark.parametrize("per_call", (1, len(SEL)))
def test_peaked_intermediate(K, per_call, plain):
    """The down projection's input goes wide as well. Its large elements carry the output, so plain rows are
    not far off here, and the high row alone is as good as a plain row: what is checked is that both rows of
    a token come back as one, which puts the error well below the plain rows'"""
    for swz in swizzle_layouts():
        ex, x, sel, w, out = run(K, False, per_call, swz, peaked = True)
        err = row_errors(K, ex, x, sel, w, out)
        before = row_errors(K, ex, x, sel, w, plain[f"{K}_{int(swz)}_peaked_{per_call}"])
        assert err.max() < 0.7 * before.max(), f"K {K}: wide rows {err.max():.2e}, plain rows {before.max():.2e}"
        assert err.median() < 0.5 * before.median()


@pytest.mark.parametrize("K", RATES)
def test_ordinary_rows_unchanged(K, plain):
    """Rows that int8 holds well are not touched: bit for bit what plain rows give"""
    for swz in swizzle_layouts():
        for per_call in (1, len(SEL)):
            ex, x, sel, w, out = run(K, False, per_call, swz)
            assert torch.equal(out, plain[f"{K}_{int(swz)}_0_{per_call}"])
            assert row_errors(K, ex, x, sel, w, out).max() < 0.04


@pytest.mark.cpu_flags("avx2")   # (the scalar tier takes float activations)
@pytest.mark.parametrize("K", (3.5, 4))
def test_mixed_chunks(K):
    """Wide and plain rows side by side in a chunk come out as they do on their own"""
    gen = torch.Generator().manual_seed(77)
    ex = make_experts(K, gen)
    wide_x, plain_x = inputs(gen, True), inputs(gen, False)
    T = len(SEL)
    pick = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1, 1, 0], dtype = torch.bool)
    x = torch.where(pick[:, None], wide_x, plain_x).contiguous()
    sel = torch.tensor(SEL, dtype = torch.long).view(T, 1)
    w = (torch.rand((T, 1), generator = gen) * 0.5 + 0.5).half()
    lists = []
    for p in ("g", "u", "d"):
        lists += [[v[0].contiguous() for v in ex[p]], [v[1].contiguous() for v in ex[p]], [v[2].contiguous() for v in ex[p]]]
    h = ext.exl3_moe_cpu_make_layer(*lists, ex["gb"], ex["ub"], [], 0, 0.0, 0)
    batch = torch.empty((T, H), dtype = torch.float32)
    ext.exl3_moe_cpu_forward(h, x, sel, w, batch, 4)
    single = torch.empty((T, H), dtype = torch.float32)
    for t in range(T):
        o = torch.empty((1, H), dtype = torch.float32)
        ext.exl3_moe_cpu_forward(h, x[t : t + 1].contiguous(), sel[t : t + 1].contiguous(), w[t : t + 1].contiguous(), o, 4)
        single[t] = o[0]
    ext.exl3_moe_cpu_free_layer(h)
    assert torch.allclose(batch, single, rtol = 1e-4, atol = 1e-6 * single.abs().max().item())
    # (the rows without the shared component are far outside the range this layer's biases are for)
    assert row_errors(K, ex, x, sel, w, batch)[pick].max() < 0.05


@pytest.mark.cpu_flags("avx2")   # (the scalar tier takes float activations)
def test_tiers_agree():
    """The integer tiers are bit-identical on wide rows too. Each tier needs its own process"""
    results = run_per_tier(outputs, tiers = [t for t in supported_tiers() if t in INT8_TIERS])
    tiers = list(results)
    base = results[tiers[0]]
    for t in tiers[1:]:
        for k, v in results[t].items():
            if k in base:
                assert torch.equal(v, base[k]), f"tier {t} differs from {tiers[0]} at {k}"


@pytest.mark.parametrize("activation, act_limit", [(0, 0.0), (3, 7.0)])
@pytest.mark.parametrize("per_call", (1, len(SEL)))
def test_deep_saturation_is_finite(activation, act_limit, per_call):
    """Pre-activations past -88.7, where exp() of their negative is infinite in fp32, give zero"""
    K = 4
    gen = torch.Generator().manual_seed(5)
    ex = make_experts(K, gen, saturate = False)
    for e in range(E):
        ex["gb"][e][::2] = -400.0
    x = inputs(gen, False)
    T = len(SEL)
    sel = torch.tensor(SEL, dtype = torch.long).view(T, 1)
    w = (torch.rand((T, 1), generator = gen) * 0.5 + 0.5).half()
    lists = []
    for p in ("g", "u", "d"):
        lists += [[v[0].contiguous() for v in ex[p]], [v[1].contiguous() for v in ex[p]], [v[2].contiguous() for v in ex[p]]]
    h = ext.exl3_moe_cpu_make_layer(*lists, ex["gb"], ex["ub"], [], activation, act_limit, 0)
    out = torch.empty((T, H), dtype = torch.float32)
    for a in range(0, T, per_call):
        o = torch.empty((min(per_call, T - a), H), dtype = torch.float32)
        ext.exl3_moe_cpu_forward(h, x[a : a + per_call].contiguous(), sel[a : a + per_call].contiguous(),
                                 w[a : a + per_call].contiguous(), o, 4)
        out[a : a + per_call] = o
    ext.exl3_moe_cpu_free_layer(h)
    assert torch.isfinite(out).all()
    ref = torch.zeros((T, H), dtype = torch.float64, device = DEV)
    for e in range(E):
        tk = (sel[:, 0] == e).nonzero().flatten()
        xd = x[tk].to(DEV).double()
        g = linear(ex, "g", e, xd, K) + ex["gb"][e].to(DEV).double()
        u = linear(ex, "u", e, xd, K)
        if activation == 0:
            a_ = F.silu(g) * u
        else:
            g = g.clamp(max = act_limit)
            a_ = (u.clamp(-act_limit, act_limit) + 1.0) * g * torch.sigmoid(1.702 * g)
        ref[tk.to(DEV)] = linear(ex, "d", e, a_, K) * w[tk].double().to(DEV)
    err = (out.to(DEV).double() - ref).norm(dim = 1) / ref.norm(dim = 1)
    assert err.max() < 0.04

