"""
Gated delta net input/gate preparation kernels, against a float64 torch formulation of the module's math.

ext.gated_delta_net_fused_op (Qwen3-Next fused projections, eager path and BC_GatedDeltaNet bsz-1 decode):
    mixed_qkvz [B, S, Nk * (2*Hk + 2*Ng*Hv)] fp32, per k-head segment [q(Hk), k(Hk), v(Ng*Hv), z(Ng*Hv)],
    mixed_ba [B, S, Nk * 2*Ng] fp32, per k-head segment [b(Ng), a(Ng)], Ng = Nv / Nk. Writes
        mixed_qkv [B, 2*Nk*Hk + Nv*Hv, S] bf16 = cat(q, k, v) over heads, channel-major (transposed)
        z [B, S, Nv, Hv] bf16
        beta [B, S, Nv] bf16 = sigmoid(b) * beta_scale
        g [B, S, Nv] fp32 = -exp(a_log) * softplus(a + dt_bias)    (softplus linear above 20, as F.softplus)
    dt_bias, a_log [Nv] bf16. The casts are exact round-to-nearest-even bf16 conversions.
ext.gated_delta_net_fused_op_2 (split projections, Qwen3.5 / Olmo-hybrid eager path): beta and g as above from
    separate b, a [B, S, H] fp32; dt_bias [H] bf16, a_log [H] fp32 or bf16; H <= 512, shapes validated.
ext.kda_gate_op (KDA, GLM5.3 / Kimi Linear; BC_GatedDeltaNetSplit graph path): qkv [B, S, F] fp32 ->
    mixed_qkv [B, F, S] bf16 (exact cast/transpose), beta [B, S, H] bf16 = sigmoid(b) * beta_scale, and the
    per-k-channel decay g [B, S, H, Dk] fp32 from f [B, S, H*Dk] with dt_bias [H*Dk] bf16, a_log [H]:
        lower_bound != 0: g = lower_bound * sigmoid(exp(a_log[h]) * (f + dt_bias))   ("safe gate")
        lower_bound == 0: g = -exp(a_log[h]) * softplus(f + dt_bias)

Every output is checked over its full extent from sentinel-filled buffers that carry guard regions on both
sides, which must come back untouched.

Tolerances: the extension builds with --use_fast_math, so exp is __expf with a documented error of
2 + floor(1.173 |x|) ulp; the remaining fp32 steps (add, log1p, approximate division, product) add a few ulp.
g is checked against a per-element relative bound derived from those terms. beta is rounded to bf16 after the
fp32 sigmoid, so the kernel may land one bf16 step from the correctly rounded value at a rounding boundary.
"""

import math

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

GUARD = 256
SENTINEL = -1232.0                                   # exact in bf16, fp16 and fp32
ULP32 = 2.0 ** -23


def guarded(shape, dtype, device):
    """Sentinel-filled tensor of `shape` inside a flat buffer with GUARD elements on both sides"""
    n = math.prod(shape)
    buf = torch.full((n + 2 * GUARD,), SENTINEL, dtype = dtype, device = device)
    return buf, buf[GUARD : GUARD + n].view(shape)


def assert_guards(buf, name):
    expect = torch.full((GUARD,), SENTINEL, dtype = buf.dtype, device = buf.device)
    assert torch.equal(buf[:GUARD], expect) and torch.equal(buf[-GUARD:], expect), f"{name}: write outside the output"


def expf_rel(x: torch.Tensor) -> torch.Tensor:
    """Relative error bound of __expf(x)"""
    return (2.0 + torch.floor(1.173 * x.abs())) * ULP32


def softplus64(x: torch.Tensor) -> torch.Tensor:
    x = x.double()
    return torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))


def assert_rel_bound(actual: torch.Tensor, expected: torch.Tensor, rel: torch.Tensor, name: str):
    assert actual.shape == expected.shape
    err = (actual.double() - expected).abs()
    bound = rel.double() * expected.abs()
    bad = err > bound
    if bad.any():
        i = bad.nonzero()[0].tolist()
        raise AssertionError(f"{name}: {int(bad.sum())} elements outside the bound, first at {i}: "
                             f"{actual[tuple(i)].item()} vs {expected[tuple(i)].item()} (bound {bound[tuple(i)].item():.3e})")


def assert_bf16_within_one_step(actual: torch.Tensor, expected64: torch.Tensor, name: str):
    """actual (bf16, positive) is the correctly rounded bf16 of expected64 or an adjacent bf16 value"""
    exp_bf = expected64.float().to(torch.bfloat16)
    assert actual.dtype == torch.bfloat16
    assert (actual.float() >= 0).all() and (exp_bf.float() >= 0).all()
    steps = (actual.view(torch.int16).int() - exp_bf.view(torch.int16).int()).abs()
    assert steps.max().item() <= 1, f"{name}: off by {steps.max().item()} bf16 steps"
    exact = (steps == 0).float().mean().item()
    assert exact > 0.97, f"{name}: only {exact:.3f} of values are correctly rounded"


def g_softplus_ref(a32: torch.Tensor, dt_bias: torch.Tensor, a_log: torch.Tensor):
    """g = -exp(a_log) * softplus(a + dt_bias) in float64 from the fp32 sum, and its relative error bound"""
    x = a32 + dt_bias.float()                          # the kernel's fp32 add, reproduced exactly
    al = a_log.double()
    g = -torch.exp(al) * softplus64(x)
    # softplus error from a relative exp error d: <= d (x < 0) or <= d / log(2) (x >= 0, softplus >= log 2)
    rel = 1.5 * expf_rel(x) * (x <= 20).double() + expf_rel(al) + 8 * ULP32
    return g, rel


def fused_op_inputs(B, S, Nk, Nv, Hk, Hv, device):
    Ng = Nv // Nk
    qkvz = torch.randn(B, S, Nk * (2 * Hk + 2 * Ng * Hv), device = device)
    ba = torch.randn(B, S, Nk * 2 * Ng, device = device) * 4.0
    ba[..., ::5] += 20.0                               # exercise softplus' linear branch and saturated sigmoid
    dt_bias = (torch.randn(Nv, device = device) * 2.0).bfloat16()
    a_log = (torch.rand(Nv, device = device) * 4.0 - 2.0).bfloat16()
    return qkvz, ba, dt_bias, a_log


def fused_op_reference(qkvz, ba, dt_bias, a_log, Nk, Nv, Hk, Hv, beta_scale):
    B, S, _ = qkvz.shape
    Ng = Nv // Nk
    seg = qkvz.view(B, S, Nk, 2 * Hk + 2 * Ng * Hv)
    q = seg[..., :Hk].reshape(B, S, Nk * Hk)
    k = seg[..., Hk : 2 * Hk].reshape(B, S, Nk * Hk)
    v = seg[..., 2 * Hk : 2 * Hk + Ng * Hv].reshape(B, S, Nv * Hv)
    z = seg[..., 2 * Hk + Ng * Hv :].reshape(B, S, Nv, Hv)
    mixed_qkv = torch.cat((q, k, v), dim = -1).transpose(1, 2).to(torch.bfloat16)
    bas = ba.view(B, S, Nk, 2 * Ng)
    b = bas[..., :Ng].reshape(B, S, Nv)
    a = bas[..., Ng:].reshape(B, S, Nv)
    beta = torch.sigmoid(b.double()) * beta_scale
    g, g_rel = g_softplus_ref(a, dt_bias, a_log)
    return mixed_qkv, z.to(torch.bfloat16), beta, g, g_rel


# (Nk, Nv, Hk, Hv): Qwen3-Next, a TP shard of it, Ng = 1, wide v heads, Hk != Hv, the 256 kernel instance and a
# non-power-of-two head dim
FUSED_SHAPES = [
    (16, 32, 128, 128),
    (4, 8, 128, 128),
    (8, 8, 128, 128),
    (2, 8, 64, 64),
    (4, 8, 64, 128),
    (2, 4, 128, 256),
    (2, 6, 96, 96),
]


@pytest.mark.parametrize("Nk, Nv, Hk, Hv", FUSED_SHAPES)
@pytest.mark.parametrize("B, S", [(1, 1), (1, 7), (3, 5), (1, 130)])
@pytest.mark.parametrize("beta_scale", [1.0, 2.0])
@torch.inference_mode()
def test_gated_delta_net_fused_op(device, Nk, Nv, Hk, Hv, B, S, beta_scale):
    torch.manual_seed(0)
    qkvz, ba, dt_bias, a_log = fused_op_inputs(B, S, Nk, Nv, Hk, Hv, device)
    F = 2 * Nk * Hk + Nv * Hv
    buf_qkv, mixed_qkv = guarded((B, F, S), torch.bfloat16, device)
    buf_z, z = guarded((B, S, Nv, Hv), torch.bfloat16, device)
    buf_beta, beta = guarded((B, S, Nv), torch.bfloat16, device)
    buf_g, g = guarded((B, S, Nv), torch.float, device)
    qkvz_in, ba_in = qkvz.clone(), ba.clone()

    ext.gated_delta_net_fused_op(qkvz, ba, dt_bias, a_log, mixed_qkv, z, beta, g, Nk, Nv, Hk, Hv, beta_scale)

    ref_qkv, ref_z, ref_beta, ref_g, g_rel = fused_op_reference(qkvz, ba, dt_bias, a_log, Nk, Nv, Hk, Hv, beta_scale)
    assert torch.equal(mixed_qkv, ref_qkv), "mixed_qkv: not the exact bf16 cast/transpose"
    assert torch.equal(z, ref_z), "z: not the exact bf16 cast"
    assert_bf16_within_one_step(beta, ref_beta, "beta")
    assert_rel_bound(g, ref_g, g_rel, "g")
    for buf, name in [(buf_qkv, "mixed_qkv"), (buf_z, "z"), (buf_beta, "beta"), (buf_g, "g")]:
        assert_guards(buf, name)
    assert torch.equal(qkvz, qkvz_in) and torch.equal(ba, ba_in), "inputs modified"


@torch.inference_mode()
def test_gated_delta_net_fused_op_rejects(device):
    Nk, Nv, Hk, Hv, B, S = 2, 4, 64, 64, 1, 2
    qkvz, ba, dt_bias, a_log = fused_op_inputs(B, S, Nk, Nv, Hk, Hv, device)
    F = 2 * Nk * Hk + Nv * Hv
    mk = lambda shape, dt: torch.empty(shape, dtype = dt, device = device)
    outs = lambda: (mk((B, F, S), torch.bfloat16), mk((B, S, Nv, Hv), torch.bfloat16),
                    mk((B, S, Nv), torch.bfloat16), mk((B, S, Nv), torch.float))
    with pytest.raises(RuntimeError, match = "divisible"):
        ext.gated_delta_net_fused_op(qkvz, ba, dt_bias, a_log, *outs(), Nk, 5, Hk, Hv, 1.0)
    with pytest.raises(RuntimeError, match = "mixed_qkvz"):
        ext.gated_delta_net_fused_op(qkvz[..., :-1], ba, dt_bias, a_log, *outs(), Nk, Nv, Hk, Hv, 1.0)
    with pytest.raises(RuntimeError, match = "mixed_ba"):
        ext.gated_delta_net_fused_op(qkvz, ba[..., :-1], dt_bias, a_log, *outs(), Nk, Nv, Hk, Hv, 1.0)
    mq, z, beta, g = outs()
    with pytest.raises(RuntimeError, match = "mixed_qkv"):
        ext.gated_delta_net_fused_op(qkvz, ba, dt_bias, a_log, mq[:, :-1], z, beta, g, Nk, Nv, Hk, Hv, 1.0)
    with pytest.raises(RuntimeError):
        ext.gated_delta_net_fused_op(qkvz, ba, dt_bias, a_log.float(), mq, z, beta, g, Nk, Nv, Hk, Hv, 1.0)
    with pytest.raises(RuntimeError):
        ext.gated_delta_net_fused_op(qkvz, ba, dt_bias, a_log, mq, z, beta, g.bfloat16(), Nk, Nv, Hk, Hv, 1.0)
    big_qkvz = torch.randn(B, S, 1 * (2 * 64 + 2 * 1 * 320), device = device)
    with pytest.raises(RuntimeError, match = "Max head dim"):
        ext.gated_delta_net_fused_op(big_qkvz, torch.randn(B, S, 2, device = device), dt_bias[:1], a_log[:1],
                                     mk((B, 2 * 64 + 320, S), torch.bfloat16), mk((B, S, 1, 320), torch.bfloat16),
                                     mk((B, S, 1), torch.bfloat16), mk((B, S, 1), torch.float), 1, 1, 64, 320, 1.0)


# fused_op_2: H = num_v_heads (Qwen3.5 variants and TP shards); 48 and 96 leave idle threads (512 % H != 0), 512
# is the largest accepted (one row per block)
@pytest.mark.parametrize("H", [1, 8, 16, 32, 48, 64, 96, 512])
@pytest.mark.parametrize("B, S", [(1, 1), (2, 3), (1, 257)])
@pytest.mark.parametrize("a_log_dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("beta_scale", [1.0, 2.0])
@torch.inference_mode()
def test_gated_delta_net_fused_op_2(device, H, B, S, a_log_dtype, beta_scale):
    torch.manual_seed(1)
    b = torch.randn(B, S, H, device = device) * 4.0
    a = torch.randn(B, S, H, device = device) * 4.0
    a[..., ::7] += 20.0
    dt_bias = (torch.randn(H, device = device) * 2.0).bfloat16()
    a_log = (torch.rand(H, device = device) * 4.0 - 2.0).to(a_log_dtype)
    buf_beta, beta = guarded((B, S, H), torch.bfloat16, device)
    buf_g, g = guarded((B, S, H), torch.float, device)

    ext.gated_delta_net_fused_op_2(b, a, dt_bias, a_log, beta, g, beta_scale)

    ref_g, g_rel = g_softplus_ref(a, dt_bias, a_log)
    assert_bf16_within_one_step(beta, torch.sigmoid(b.double()) * beta_scale, "beta")
    assert_rel_bound(g, ref_g, g_rel, "g")
    assert_guards(buf_beta, "beta")
    assert_guards(buf_g, "g")


@torch.inference_mode()
def test_gated_delta_net_fused_op_2_rejects(device):
    B, S, H = 1, 2, 16
    b = torch.randn(B, S, H, device = device)
    a = torch.randn(B, S, H, device = device)
    dt_bias = torch.randn(H, device = device).bfloat16()
    a_log = torch.randn(H, device = device)
    beta = torch.empty(B, S, H, dtype = torch.bfloat16, device = device)
    g = torch.empty(B, S, H, device = device)
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.gated_delta_net_fused_op_2(b, a[:, :1], dt_bias, a_log, beta, g, 1.0)
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.gated_delta_net_fused_op_2(b, a, dt_bias[:-1], a_log, beta, g, 1.0)
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.gated_delta_net_fused_op_2(b, a, dt_bias, a_log[:-1], beta, g, 1.0)
    with pytest.raises(RuntimeError, match = "incompatible shapes"):
        ext.gated_delta_net_fused_op_2(b, a, dt_bias, a_log, beta[:, :1], g, 1.0)
    with pytest.raises(RuntimeError, match = "unsupported dtype"):
        ext.gated_delta_net_fused_op_2(b, a, dt_bias, a_log.half(), beta, g, 1.0)
    with pytest.raises(RuntimeError):
        ext.gated_delta_net_fused_op_2(b, a, dt_bias.float(), a_log, beta, g, 1.0)
    H = 513
    with pytest.raises(RuntimeError, match = "too many heads"):
        ext.gated_delta_net_fused_op_2(
            torch.randn(1, 1, H, device = device), torch.randn(1, 1, H, device = device),
            torch.randn(H, device = device).bfloat16(), torch.randn(H, device = device),
            torch.empty(1, 1, H, dtype = torch.bfloat16, device = device), torch.empty(1, 1, H, device = device), 1.0
        )


def kda_reference(qkv, b, f, dt_bias, a_log, H, Dk, lower_bound, beta_scale):
    B, S, _ = qkv.shape
    mixed_qkv = qkv.transpose(1, 2).to(torch.bfloat16)
    beta = torch.sigmoid(b.double()) * beta_scale
    fv = (f + dt_bias.float().view(1, 1, -1)).view(B, S, H, Dk)      # the kernel's fp32 add, reproduced exactly
    al = a_log.double().view(1, 1, H, 1)
    decay = torch.exp(al)
    if lower_bound != 0.0:
        # sigmoid(arg) with arg rounded to fp32 in the kernel (one more ulp on the argument)
        arg = decay * fv.double()
        g = lower_bound * torch.sigmoid(arg)
        rel = expf_rel(arg) + expf_rel(al) * arg.abs() * torch.sigmoid(-arg) + arg.abs() * ULP32 + 8 * ULP32
    else:
        g = -decay * softplus64(fv)
        rel = 1.5 * expf_rel(fv) * (fv <= 20).double() + expf_rel(al) + 8 * ULP32
    return mixed_qkv, beta, g, rel


# (H, Dk, F): GLM5.3 / Kimi Linear (32 heads x 128, qkv = 3 * 4096), a TP shard, a small odd case
KDA_SHAPES = [(32, 128, 3 * 32 * 128), (8, 128, 3 * 8 * 128), (3, 32, 3 * 3 * 32 + 5)]


@pytest.mark.parametrize("H, Dk, F", KDA_SHAPES)
@pytest.mark.parametrize("B, S", [(1, 1), (4, 3), (8, 16), (1, 77)])
@pytest.mark.parametrize("lower_bound", [0.0, -5.0])
@pytest.mark.parametrize("a_log_dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("beta_scale", [1.0, 2.0])
@torch.inference_mode()
def test_kda_gate_op(device, H, Dk, F, B, S, lower_bound, a_log_dtype, beta_scale):
    torch.manual_seed(2)
    qkv = torch.randn(B, S, F, device = device)
    b = torch.randn(B, S, H, device = device) * 4.0
    f = torch.randn(B, S, H * Dk, device = device) * 3.0
    f[..., ::11] += 20.0
    dt_bias = (torch.randn(H * Dk, device = device)).bfloat16()
    a_log = (torch.rand(H, device = device) * 2.0 - 1.0).to(a_log_dtype)
    buf_qkv, mixed_qkv = guarded((B, F, S), torch.bfloat16, device)
    buf_beta, beta = guarded((B, S, H), torch.bfloat16, device)
    buf_g, g = guarded((B, S, H, Dk), torch.float, device)
    inputs = [t.clone() for t in (qkv, b, f)]

    ext.kda_gate_op(qkv, b, f, dt_bias, a_log, mixed_qkv, beta, g, lower_bound, beta_scale)

    ref_qkv, ref_beta, ref_g, g_rel = kda_reference(qkv, b, f, dt_bias, a_log, H, Dk, lower_bound, beta_scale)
    assert torch.equal(mixed_qkv, ref_qkv), "mixed_qkv: not the exact bf16 cast/transpose"
    assert_bf16_within_one_step(beta, ref_beta, "beta")
    assert_rel_bound(g, ref_g, g_rel, "g")
    for buf, name in [(buf_qkv, "mixed_qkv"), (buf_beta, "beta"), (buf_g, "g")]:
        assert_guards(buf, name)
    assert all(torch.equal(x, y) for x, y in zip((qkv, b, f), inputs)), "inputs modified"


@torch.inference_mode()
def test_kda_gate_op_rejects(device):
    B, S, H, Dk, F = 1, 2, 4, 32, 3 * 4 * 32
    args = dict(
        qkv = torch.randn(B, S, F, device = device),
        b = torch.randn(B, S, H, device = device),
        f = torch.randn(B, S, H * Dk, device = device),
        dt_bias = torch.randn(H * Dk, device = device).bfloat16(),
        a_log = torch.randn(H, device = device),
        mixed_qkv = torch.empty(B, F, S, dtype = torch.bfloat16, device = device),
        beta = torch.empty(B, S, H, dtype = torch.bfloat16, device = device),
        g = torch.empty(B, S, H, Dk, device = device),
    )
    call = lambda **kw: ext.kda_gate_op(*{**args, **kw}.values(), 0.0, 1.0)
    with pytest.raises(RuntimeError, match = "g must be"):
        call(g = torch.empty(B, S, H, Dk - 1, device = device))
    for name, bad in [("qkv", args["qkv"].half()), ("b", args["b"].bfloat16()), ("f", args["f"].half()),
                      ("dt_bias", args["dt_bias"].float()), ("mixed_qkv", args["mixed_qkv"].float()),
                      ("beta", args["beta"].float()), ("g", args["g"].bfloat16())]:
        with pytest.raises(RuntimeError):
            call(**{name: bad})


# Zero-size inputs: an empty batch, sequence or head axis is an elementwise no-op (nothing written, no launch). The
# fused op's head counts and head dims define its segment layout (Ng = Nv / Nk) and must be positive. kda_gate_op
# with H = 0 still casts qkv (only beta and g are empty)

def _assert_device_usable(device):
    torch.cuda.synchronize(device)
    assert (torch.ones(4, device = device) + 1).sum().item() == 8.0, "a later torch op failed after the empty call"


def _empty_fused_op(device, B, S, Nk = 2, Nv = 4, Hk = 32, Hv = 32):
    Ng = Nv // Nk if Nk else 0
    qkvz = torch.randn(B, S, Nk * (2 * Hk + 2 * Ng * Hv), device = device)
    ba = torch.randn(B, S, Nk * 2 * Ng, device = device)
    bufs = [guarded(s, t, device) for s, t in [((B, 2 * Nk * Hk + Nv * Hv, S), torch.bfloat16),
                                              ((B, S, Nv, Hv), torch.bfloat16), ((B, S, Nv), torch.bfloat16),
                                              ((B, S, Nv), torch.float)]]
    ext.gated_delta_net_fused_op(qkvz, ba, torch.zeros(Nv, dtype = torch.bfloat16, device = device),
                                 torch.zeros(Nv, dtype = torch.bfloat16, device = device),
                                 *[t for _, t in bufs], Nk, Nv, Hk, Hv, 1.0)
    for (buf, _), name in zip(bufs, ["mixed_qkv", "z", "beta", "g"]):
        assert_guards(buf, name)


def _empty_fused_op_2(device, B, S, H):
    (buf_beta, beta), (buf_g, g) = guarded((B, S, H), torch.bfloat16, device), guarded((B, S, H), torch.float, device)
    ext.gated_delta_net_fused_op_2(torch.randn(B, S, H, device = device), torch.randn(B, S, H, device = device),
                                   torch.zeros(H, dtype = torch.bfloat16, device = device),
                                   torch.zeros(H, device = device), beta, g, 1.0)
    assert_guards(buf_beta, "beta")
    assert_guards(buf_g, "g")


def _empty_kda(device, B, S, H, Dk = 4, F = 24):
    qkv = torch.randn(B, S, F, device = device)
    (buf_qkv, mixed_qkv), (buf_beta, beta), (buf_g, g) = guarded((B, F, S), torch.bfloat16, device), \
        guarded((B, S, H), torch.bfloat16, device), guarded((B, S, H, Dk), torch.float, device)
    ext.kda_gate_op(qkv, torch.randn(B, S, H, device = device), torch.randn(B, S, H * Dk, device = device),
                    torch.zeros(H * Dk, dtype = torch.bfloat16, device = device), torch.zeros(H, device = device),
                    mixed_qkv, beta, g, 0.0, 1.0)
    assert torch.equal(mixed_qkv, qkv.bfloat16().transpose(1, 2)), "mixed_qkv: not the exact bf16 cast/transpose"
    for buf, name in [(buf_qkv, "mixed_qkv"), (buf_beta, "beta"), (buf_g, "g")]:
        assert_guards(buf, name)


EMPTY_CASES = {
    "fused_op B=0": lambda d: _empty_fused_op(d, 0, 3),
    "fused_op S=0": lambda d: _empty_fused_op(d, 2, 0),
    "fused_op heads=0": ("head counts and head dims must be positive", lambda d: _empty_fused_op(d, 1, 1, Nk = 0, Nv = 0)),
    "fused_op Hk=0": ("head counts and head dims must be positive", lambda d: _empty_fused_op(d, 1, 1, Hk = 0)),
    "fused_op_2 B=0": lambda d: _empty_fused_op_2(d, 0, 3, 8),
    "fused_op_2 S=0": lambda d: _empty_fused_op_2(d, 2, 0, 8),
    "fused_op_2 H=0": lambda d: _empty_fused_op_2(d, 2, 3, 0),
    "kda B=0": lambda d: _empty_kda(d, 0, 2, 4),
    "kda S=0": lambda d: _empty_kda(d, 2, 0, 4),
    "kda H=0": lambda d: _empty_kda(d, 2, 3, 0),
}


@pytest.mark.parametrize("case", list(EMPTY_CASES))
@torch.inference_mode()
def test_empty(device, case):
    spec = EMPTY_CASES[case]
    if isinstance(spec, tuple):
        with pytest.raises(RuntimeError, match = spec[0]):
            spec[1](device)
    else:
        spec(device)
    _assert_device_usable(device)
