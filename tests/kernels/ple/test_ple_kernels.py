"""
PLE (Qwen3.8-Flash-Next per-layer embedding) extension functions, ple.cu:

ext.ple_gate(gate, value, out, gate_scale)
    out[b, s, h, :] = sigmoid(ss(gate[b, s, h] * gate_scale)) * value[b, s, :], ss(g) = sign(g) * sqrt(max(|g|, 1e-6))
    with sign(0) = 0. gate (B, S, H) fp32, value (B, S, D) fp16, out (B, S, H, D) fp32, all contiguous, D % 4 == 0.
    Writes exactly out; validates dtypes, ranks, shapes, contiguity and D % 4.

ext.ple_forward_streams(streams, emb, key_w, value_w, norm_key_w, norm_query_w, norm_conv_w, conv_w, conv_state,
                        eps, gate_scale, dilation, delta, conv_stream)
    The PLELayer.forward_streams sequence as one call (modules/ple.py, forward_streams_reference):
        key    = fp16(emb @ key_w)                           (rows, H * D)
        key_n  = rms(key per stream of D) * (1 + norm_key_w[h])            fp32
        query  = rms(streams) * (1 + norm_query_w[h])                       fp32
        gated  = sigmoid(ss(<query, key_n> * gate_scale)) * fp16(emb @ value_w)   per stream
        normed = fp16(rms(gated) * (1 + norm_conv_w[h]))
        conv_stream = [conv_state (or zeros) | normed^T]     (bsz, H * D, state_len + seq) fp16, written in full
        delta  = gated + silu(depthwise conv1d(conv_stream, conv_w, dilation))^T   (bsz, seq, H, D) fp32
    state_len = conv_stream.size(2) - seq must equal (ksize - 1) * dilation (the conv then yields exactly seq
    columns) and conv_state, when given, must be (bsz, H * D, state_len). Inputs and outputs must be contiguous.

References: float64 torch transcriptions of the formulas above (not the module's ext-based reference path),
rounding to fp16 where the contract stores fp16 (key, value, normed, conv output).
"""

import math

import pytest
import torch
import torch.nn.functional as F

from exllamav3.ext import exllamav3_ext as ext

EPS32 = torch.finfo(torch.float32).eps
ULP16 = 2.0 ** -10              # relative ulp of fp16


def signed_sqrt(g):
    return torch.sign(g) * torch.sqrt(torch.clamp(g.abs(), min = 1e-6))


def ref_ple_gate(gate, value, gate_scale):
    g = gate.double().cpu() * gate_scale
    s = torch.sigmoid(signed_sqrt(g))
    return s.unsqueeze(-1) * value.double().cpu().unsqueeze(2)


@pytest.mark.parametrize("shape", [(1, 1, 4, 4), (1, 1, 4, 2048), (2, 7, 4, 1024), (3, 5, 1, 12), (1, 33, 3, 260),
                                   (1, 1, 8, 4096)])
@pytest.mark.parametrize("gate_scale", [1.0, 1.0 / math.sqrt(2048)])
@torch.inference_mode()
def test_ple_gate(device, shape, gate_scale):
    B, S, H, D = shape
    torch.manual_seed(0)
    gate = torch.randn(B, S, H, device = device) * 40.0
    value = torch.randn(B, S, D, dtype = torch.half, device = device)
    # out as a contiguous slice of a sentinel-filled buffer: nothing outside it may be written
    sentinel = -777.0
    buf = torch.full((B + 2, S, H, D), sentinel, device = device)
    out = buf[1 : B + 1]
    ext.ple_gate(gate, value, out, gate_scale)
    ref = ref_ple_gate(gate, value, gate_scale)
    # fp32 gate * scale and sqrt (correctly rounded), __expf ((2 + 1.16 |x|) ulp) and __fdividef (2 ulp): the
    # sigmoid carries a relative error of a few tens of fp32 ulps at |ss| <= ~13, value * s one more rounding
    torch.testing.assert_close(out.double().cpu(), ref, rtol = 64 * EPS32, atol = 1e-30)
    assert (buf[0] == sentinel).all() and (buf[B + 1] == sentinel).all(), "ple_gate wrote outside out"


@torch.inference_mode()
def test_ple_gate_special_values(device):
    # sign(0) = 0 gives sigmoid(0) = 0.5 for +-0; |g| below 1e-6 clamps the magnitude to sqrt(1e-6) but keeps the
    # sign; saturated gates give exactly 0 / 1 (exp overflow must not turn into NaN)
    vals = [0.0, -0.0, 1e-7, -1e-7, 1e-6, 4.0, -4.0, 1e4, -1e4, 1e30, -1e30, float("inf"), float("-inf")]
    gate = torch.tensor(vals, device = device).view(1, 1, -1)
    value = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype = torch.half, device = device).view(1, 1, 4)
    out = torch.empty(1, 1, len(vals), 4, device = device)
    ext.ple_gate(gate, value, out, 1.0)
    ref = ref_ple_gate(gate, value, 1.0)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.double().cpu(), ref, rtol = 64 * EPS32, atol = 1e-30)
    assert (out[0, 0, 0] == value[0, 0].float() * 0.5).all()
    assert (out[0, 0, 1] == value[0, 0].float() * 0.5).all()


@torch.inference_mode()
def test_ple_gate_deterministic(device):
    gate = torch.randn(2, 9, 4, device = device)
    value = torch.randn(2, 9, 512, dtype = torch.half, device = device)
    a = torch.empty(2, 9, 4, 512, device = device)
    b = torch.empty_like(a)
    ext.ple_gate(gate, value, a, 0.3)
    ext.ple_gate(gate, value, b, 0.3)
    assert torch.equal(a, b)


def _gate_invalid(device):
    g = lambda *s, dt = torch.float: torch.zeros(*s, dtype = dt, device = device)
    return {
        "gate_fp16": (g(1, 2, 4, dt = torch.half), g(1, 2, 8, dt = torch.half), g(1, 2, 4, 8)),
        "value_fp32": (g(1, 2, 4), g(1, 2, 8), g(1, 2, 4, 8)),
        "out_fp16": (g(1, 2, 4), g(1, 2, 8, dt = torch.half), g(1, 2, 4, 8, dt = torch.half)),
        "out_rank3": (g(1, 2, 4), g(1, 2, 8, dt = torch.half), g(1, 2, 32)),
        "gate_rank2": (g(2, 4), g(1, 2, 8, dt = torch.half), g(1, 2, 4, 8)),
        "value_rank2": (g(1, 2, 4), g(2, 8, dt = torch.half), g(1, 2, 4, 8)),
        "gate_streams": (g(1, 2, 3), g(1, 2, 8, dt = torch.half), g(1, 2, 4, 8)),
        "value_dim": (g(1, 2, 4), g(1, 2, 12, dt = torch.half), g(1, 2, 4, 8)),
        "value_seq": (g(1, 2, 4), g(1, 3, 8, dt = torch.half), g(1, 2, 4, 8)),
        "dim_not_mult4": (g(1, 2, 4), g(1, 2, 6, dt = torch.half), g(1, 2, 4, 6)),
        "out_noncontig": (g(1, 2, 4), g(1, 2, 8, dt = torch.half), g(1, 2, 4, 16)[..., :8]),
        "value_noncontig": (g(1, 2, 4), g(1, 2, 16, dt = torch.half)[..., :8], g(1, 2, 4, 8)),
        "gate_noncontig": (g(1, 2, 8)[..., ::2], g(1, 2, 8, dt = torch.half), g(1, 2, 4, 8)),
    }


GATE_INVALID = ["gate_fp16", "value_fp32", "out_fp16", "out_rank3", "gate_rank2", "value_rank2", "gate_streams",
                "value_dim", "value_seq", "dim_not_mult4", "out_noncontig", "value_noncontig", "gate_noncontig"]


@pytest.mark.parametrize("case", GATE_INVALID)
@torch.inference_mode()
def test_ple_gate_rejects_invalid(device, case):
    cases = _gate_invalid(device)
    assert set(cases) == set(GATE_INVALID)
    gate, value, out = cases[case]
    with pytest.raises(RuntimeError):
        ext.ple_gate(gate, value, out, 1.0)


# ple_forward_streams

def make_ple_inputs(device, bsz, seq, H, D, ple_dim, ksize, dilation, with_state, norm_dtype, seed = 0):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    r = lambda *s, scale = 1.0: (torch.randn(*s, generator = g) * scale)
    t = dict(
        streams = r(bsz, seq, H, D, scale = 2.0).float(),
        emb = r(bsz, seq, ple_dim).half(),
        key_w = r(ple_dim, H * D, scale = ple_dim ** -0.5).half(),
        value_w = r(ple_dim, D, scale = ple_dim ** -0.5).half(),
        norm_key_w = r(H * D, scale = 0.3).to(norm_dtype),
        norm_query_w = r(H * D, scale = 0.3).to(norm_dtype),
        norm_conv_w = r(H * D, scale = 0.3).to(norm_dtype),
        conv_w = r(H * D, 1, ksize, scale = 0.5).half(),
        conv_state = r(bsz, H * D, (ksize - 1) * dilation).half() if with_state else None,
    )
    return {k: (v.to(device) if v is not None else None) for k, v in t.items()}


def ref_forward_streams(t, eps, gate_scale, dilation):
    """float64 transcription; fp16 at the contract's fp16 storage points. The two fp16 projections are taken
    from torch's own fp16 matmul (the op the binding calls for them, not code under test), so the reference
    starts from the same fp16 key/value rows instead of ones that differ by an fp16 ulp at rounding boundaries"""
    d = {k: (v.double().cpu() if v is not None else None) for k, v in t.items()}
    emb2 = t["emb"].flatten(0, -2)
    key16 = (emb2 @ t["key_w"]).double().cpu()
    value16 = (emb2 @ t["value_w"]).double().cpu()
    bsz, seq, H, D = d["streams"].shape
    hc = H * D
    state_len = (d["conv_w"].shape[-1] - 1) * dilation

    def rms(x, w):
        x = x * torch.rsqrt(x.pow(2).mean(dim = -1, keepdim = True) + eps)
        return x * (1.0 + w.view(H, D))

    key = key16.view(bsz, seq, H, D)
    key_n = rms(key, d["norm_key_w"])
    query_n = rms(d["streams"], d["norm_query_w"])
    gate = (key_n * query_n).sum(dim = -1)
    value = value16.view(bsz, seq, D)
    gated = torch.sigmoid(signed_sqrt(gate * gate_scale)).unsqueeze(-1) * value.unsqueeze(2)
    normed = rms(gated, d["norm_conv_w"]).half().double()
    state = d["conv_state"] if d["conv_state"] is not None else torch.zeros(bsz, hc, state_len, dtype = torch.double)
    stream = torch.cat((state, normed.view(bsz, seq, hc).transpose(1, 2)), dim = -1)
    y = F.conv1d(stream, d["conv_w"], groups = hc, dilation = dilation)
    conv_abs = F.conv1d(stream.abs(), d["conv_w"].abs(), groups = hc, dilation = dilation)
    conv_out = F.silu(y).transpose(1, 2).reshape(bsz, seq, H, D)
    delta = gated + conv_out
    return dict(delta = delta, conv_stream = stream, gated = gated, conv_out = conv_out,
                conv_abs = conv_abs.transpose(1, 2).reshape(bsz, seq, H, D), normed = normed)


def run_forward_streams(t, eps, gate_scale, dilation, conv_stream = None, delta = None):
    bsz, seq, H, D = t["streams"].shape
    state_len = (t["conv_w"].shape[-1] - 1) * dilation
    dev = t["streams"].device
    if delta is None:
        delta = torch.empty(bsz, seq, H, D, dtype = torch.float, device = dev)
    if conv_stream is None:
        conv_stream = torch.empty(bsz, H * D, state_len + seq, dtype = torch.half, device = dev)
    ext.ple_forward_streams(t["streams"], t["emb"], t["key_w"], t["value_w"], t["norm_key_w"], t["norm_query_w"],
                            t["norm_conv_w"], t["conv_w"], t["conv_state"], eps, gate_scale, dilation, delta,
                            conv_stream)
    return delta, conv_stream


def check_forward_streams(t, delta, conv_stream, ref):
    bsz, seq, H, D = t["streams"].shape
    state_len = conv_stream.shape[-1] - seq
    cs = conv_stream.double().cpu()
    # State columns: an exact copy of conv_state (or zeros)
    if t["conv_state"] is not None:
        assert torch.equal(conv_stream[..., :state_len], t["conv_state"]), "conv_stream state columns != conv_state"
    else:
        assert (conv_stream[..., :state_len] == 0).all(), "conv_stream state columns not zeroed"
    # New columns: the fp16 normed gate output. The kernel's fp32 norms and gate dot can land on the other side
    # of an fp16 rounding boundary than the float64 reference: one fp16 ulp either way, i.e. up to 2^-10 of the
    # value's magnitude, twice that when the two straddle a binade boundary
    new = cs[..., state_len:]
    ref_new = ref["conv_stream"][..., state_len:]
    err = (new - ref_new).abs()
    assert (err <= 2 * ULP16 * ref_new.abs() + 2.0 ** -24).all(), f"normed columns off by {err.max().item():.3g}"
    # delta = gated (fp32; relative error dominated by an fp16-ulp disagreement in key/value inputs, which moves
    # the gate dot and value by at most an fp16 ulp) + silu(conv) of fp16 operands: a one-ulp difference in any
    # conv input, the fp16 conv result and the fp16 silu output each contribute an fp16 ulp of the tap magnitude
    bound = 2 * ULP16 * ref["gated"].abs() + 3 * ULP16 * (ref["conv_abs"] + ref["conv_out"].abs()) + 1e-6
    err = (delta.double().cpu() - ref["delta"]).abs()
    bad = err > bound
    assert not bad.any(), f"{int(bad.sum())} delta elements out of bound, max excess {(err - bound).max().item():.3g}"


@pytest.mark.parametrize("bsz, seq", [(1, 1), (1, 5), (2, 3), (1, 40), (3, 1)])
@pytest.mark.parametrize("ksize, dilation", [(4, 3), (4, 2), (2, 1), (1, 3)])
@pytest.mark.parametrize("with_state", [False, True], ids = ["nostate", "state"])
@torch.inference_mode()
def test_ple_forward_streams(device, bsz, seq, ksize, dilation, with_state):
    H, D, ple_dim = 4, 256, 64
    eps, gate_scale = 1e-6, 1.0 / math.sqrt(D)
    t = make_ple_inputs(device, bsz, seq, H, D, ple_dim, ksize, dilation, with_state, torch.half)
    snapshot = {k: v.clone() for k, v in t.items() if v is not None}
    delta, conv_stream = run_forward_streams(t, eps, gate_scale, dilation)
    ref = ref_forward_streams(t, eps, gate_scale, dilation)
    check_forward_streams(t, delta, conv_stream, ref)
    for k, v in snapshot.items():
        assert torch.equal(t[k], v), f"ple_forward_streams modified input {k}"


@pytest.mark.parametrize("norm_dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("H, D, ple_dim", [(4, 2048, 256), (2, 12, 8), (1, 1024, 128)])
@torch.inference_mode()
def test_ple_forward_streams_geometry(device, norm_dtype, H, D, ple_dim):
    bsz, seq, ksize, dilation = 2, 6, 4, 3
    eps, gate_scale = 1e-6, 1.0 / math.sqrt(D)
    t = make_ple_inputs(device, bsz, seq, H, D, ple_dim, ksize, dilation, True, norm_dtype, seed = 1)
    delta, conv_stream = run_forward_streams(t, eps, gate_scale, dilation)
    check_forward_streams(t, delta, conv_stream, ref_forward_streams(t, eps, gate_scale, dilation))


@torch.inference_mode()
def test_ple_forward_streams_chunked_equals_whole(device):
    # The module splits long prefills into slabs carrying the conv state (trailing state_len columns of the
    # previous slab's conv_stream). The carried state is a copy, so it is bit-exact. Delta is not: the key/value
    # projections are cuBLAS GEMMs over all rows of the call, and cuBLAS picks its kernel (and so the fp16 rounding
    # of each projected element, 0.5 ulp ~ 4.9e-4 relative) by row count; through the norms and gate that is
    # about 1e-3 relative
    bsz, seq, H, D, ple_dim, ksize, dilation = 1, 23, 4, 128, 64, 4, 3
    eps, gate_scale = 1e-6, 1.0 / math.sqrt(D)
    state_len = (ksize - 1) * dilation
    t = make_ple_inputs(device, bsz, seq, H, D, ple_dim, ksize, dilation, True, torch.half, seed = 2)
    whole, whole_cs = run_forward_streams(t, eps, gate_scale, dilation)
    parts = []
    cs = t["conv_state"]
    for t0, t1 in ((0, 7), (7, 8), (8, 23)):
        sub = dict(t)
        sub["streams"] = t["streams"][:, t0:t1]
        sub["emb"] = t["emb"][:, t0:t1]
        sub["conv_state"] = cs
        d, c = run_forward_streams(sub, eps, gate_scale, dilation)
        parts.append(d)
        cs = c[:, :, -state_len:].contiguous()
    torch.testing.assert_close(torch.cat(parts, dim = 1), whole, rtol = 2e-3, atol = 2e-4)
    assert torch.equal(cs, whole_cs[:, :, -state_len:])


@torch.inference_mode()
def test_ple_forward_streams_outputs_only(device):
    # Outputs are contiguous slices of sentinel buffers: only delta and conv_stream themselves are written
    bsz, seq, H, D, ple_dim, ksize, dilation = 2, 4, 4, 64, 32, 4, 2
    state_len = (ksize - 1) * dilation
    t = make_ple_inputs(device, bsz, seq, H, D, ple_dim, ksize, dilation, True, torch.half, seed = 3)
    dbuf = torch.full((bsz + 2, seq, H, D), -55.0, device = device)
    cbuf = torch.full((bsz + 2, H * D, state_len + seq), -55.0, dtype = torch.half, device = device)
    delta, conv_stream = run_forward_streams(t, 1e-6, 0.125, dilation, conv_stream = cbuf[1 : bsz + 1],
                                             delta = dbuf[1 : bsz + 1])
    for buf in (dbuf, cbuf):
        assert (buf[0] == -55.0).all() and (buf[bsz + 1] == -55.0).all(), "wrote outside the outputs"
    check_forward_streams(t, delta, conv_stream, ref_forward_streams(t, 1e-6, 0.125, dilation))


@pytest.mark.parametrize("which", ["streams", "emb", "delta", "conv_stream", "state_len_short", "state_len_long",
                                   "conv_state_bsz"])
@torch.inference_mode()
def test_ple_forward_streams_rejects(device, which):
    bsz, seq, H, D, ple_dim, ksize, dilation = 2, 4, 2, 64, 32, 4, 2
    state_len = (ksize - 1) * dilation
    t = make_ple_inputs(device, bsz, seq, H, D, ple_dim, ksize, dilation, which == "conv_state_bsz", torch.half)
    delta = torch.empty(bsz, seq, H, D, device = device)
    conv_stream = torch.empty(bsz, H * D, state_len + seq, dtype = torch.half, device = device)
    match = None
    if which in ("state_len_short", "state_len_long"):
        # The conv would yield more or fewer than seq columns (or, at one column, broadcast silently)
        extra = -1 if which == "state_len_short" else 1
        conv_stream = torch.empty(bsz, H * D, state_len + extra + seq, dtype = torch.half, device = device)
        match = "state columns"
    elif which == "conv_state_bsz":
        # A one-row state would broadcast over the batch in the state copy
        t["conv_state"] = t["conv_state"][:1].contiguous()
        match = "conv_state must be"
    elif which == "streams":
        t["streams"] = torch.empty(bsz, seq, H, 2 * D, device = device)[..., :D]
    elif which == "emb":
        t["emb"] = torch.empty(bsz, seq, 2 * ple_dim, dtype = torch.half, device = device)[..., :ple_dim]
    elif which == "delta":
        delta = torch.empty(bsz, seq, H, 2 * D, device = device)[..., :D]
    else:
        conv_stream = torch.empty(bsz, H * D, 2 * (state_len + seq), dtype = torch.half, device = device)[..., ::2]
    with pytest.raises(RuntimeError, match = match):
        run_forward_streams(t, 1e-6, 0.1, dilation, conv_stream = conv_stream, delta = delta)
    assert_device_ok(device)


def assert_device_ok(device):
    # A failed launch would leave an error for the next op on the device
    torch.cuda.synchronize(device)
    assert torch.ones(8, device = device).sum().item() == 8


@pytest.mark.parametrize("shape", [(0, 3, 4, 64), (2, 0, 4, 64), (2, 3, 0, 64), (2, 3, 4, 0)])
@torch.inference_mode()
def test_ple_gate_empty(device, shape):
    # Elementwise gate: an empty output is a no-op (nothing written); dtype validation still applies
    B, S, H, D = shape
    buf = torch.full((64,), 7.0, device = device)
    gate = torch.zeros(B, S, H, device = device)
    value = torch.zeros(B, S, D, dtype = torch.half, device = device)
    ext.ple_gate(gate, value, buf[8:8].view(shape), 1.0)
    assert (buf == 7.0).all()
    with pytest.raises(RuntimeError, match = "incorrect datatype"):
        ext.ple_gate(gate.half(), value, buf[8:8].view(shape), 1.0)
    assert_device_ok(device)


@pytest.mark.parametrize("bsz, seq, H", [(1, 0, 4), (2, 0, 4), (0, 3, 4), (2, 3, 0)])
@pytest.mark.parametrize("with_state", [False, True], ids = ["nostate", "state"])
@torch.inference_mode()
def test_ple_forward_streams_empty(device, bsz, seq, H, with_state):
    # No tokens (or no streams): delta is empty and untouched, and the conv stream is just the carried state (the
    # caller keeps its trailing columns as the next state)
    D, ple_dim, ksize, dilation = 64, 32, 4, 2
    state_len = (ksize - 1) * dilation
    t = make_ple_inputs(device, bsz, seq, H, D, ple_dim, ksize, dilation, with_state, torch.half)
    dbuf = torch.full((64,), 7.0, device = device)
    delta = dbuf[8:8].view(bsz, seq, H, D)
    conv_stream = torch.full((bsz, H * D, state_len + seq), 7.0, dtype = torch.half, device = device)
    run_forward_streams(t, 1e-6, 0.125, dilation, conv_stream = conv_stream, delta = delta)
    assert (dbuf == 7.0).all()
    if with_state:
        assert torch.equal(conv_stream[..., :state_len], t["conv_state"])
    else:
        assert (conv_stream[..., :state_len] == 0).all()
    assert_device_ok(device)


@torch.inference_mode()
def test_ple_forward_streams_empty_dims(device):
    # D == 0: the per-stream RMS norms are over an empty vector, undefined -> raises
    t = make_ple_inputs(device, 1, 3, 4, 1, 32, 4, 2, True, torch.half)
    t = {k: (v[..., :0] if k in ("streams", "value_w") else v.view(-1)[:0] if k.startswith("norm") else v)
         for k, v in t.items()}
    t["key_w"], t["conv_w"], t["conv_state"] = t["key_w"][:, :0], t["conv_w"][:0], t["conv_state"][:, :0]
    with pytest.raises(RuntimeError, match = "ple_forward_streams: norm over an empty stream dimension"):
        run_forward_streams(t, 1e-6, 0.125, 2)
    # ple_dim == 0: key and value are K = 0 products, i.e. zeros, and the rest follows the formulas
    t = make_ple_inputs(device, 2, 3, 4, 64, 1, 4, 2, True, torch.half)
    t["emb"], t["key_w"], t["value_w"] = t["emb"][..., :0], t["key_w"][:0], t["value_w"][:0]
    delta, conv_stream = run_forward_streams(t, 1e-6, 0.125, 2)
    check_forward_streams(t, delta, conv_stream, ref_forward_streams(t, 1e-6, 0.125, 2))
    assert_device_ok(device)
