"""
ext.rope, the raw fused rotary-embedding kernel (rope.cu), tested at the binding level. modules/rope/test_rope.py
covers RoPE.apply against RoPE.apply_torch; this file pins the kernel contract on its own:

- q (bsz, seq, Hq, hd) and optional k (bsz, seq, Hk, hd), fp16, innermost dim dense, dims 0..2 dense over the head
  stride (stride(2) may exceed hd: trailing-slice views of wider heads are read and written in place through it).
  out_q must share q's strides; out_q == q / out_k == k is the in-place form.
- Each rotated pair (NEOX: (i, i + rd/2); GPT-J: (2i, 2i + 1), within [rotate_offset, rotate_offset + rd)) is turned
  by angle = fp32(inv_freq[i] * fp32(pos)) (or the angle read from an inv_freq table [b, pos, i]), with sin/cos
  scaled by attn_factor. rd = 2 * inv_freq.size(-1). Dimensions outside the rotated span pass through unchanged.
- Position of token t in batch row b: t + position, t + positions[b], or position_ids[b, t] (position_ids[b, t, r]
  for rotate span r when rotate_dims > 1, spans laid out back to back, hd == rd * rotate_dims).
- Optional per-head RMS norm (q_norm for q heads, k_norm for k heads, + norm_constant_bias) before the rotation:
  normalized in fp32 and rounded to fp16, then the weight applied (fp16 weight: fp16 multiply; bf16: fp32 multiply).
- Llama-4 scaling: after the rotation q heads only are scaled by 1 + beta * ln(1 + floor(pos / original)).
- Argument validation (TORCH_CHECK) for dtypes, ranks, shapes, layout and the rotate/position options.

Reference: an independent float64 torch rotation of the fp16 inputs. sin/cos come from __sinf/__cosf, whose
range reduction leaves an absolute error of about one fp32 ulp of the angle; the per-element bounds below are built
from that and from the fp16 output rounding. test_rope_long_positions_match_accurate_sincos asks whether the kernel
should instead match accurately computed sin/cos of the same fp32 angle (the HF formulation) at long context.
"""

import math

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

GPTJ, NEOX = 1, 2
EPS32 = torch.finfo(torch.float32).eps
HALF_ULP16 = 2.0 ** -11          # relative half-ulp of fp16 (round to nearest)


def default_inv_freq(rd, device, theta = 10000.0):
    return 1.0 / (theta ** (torch.arange(0, rd, 2, device = device).float() / rd))


def call_rope(q, out_q, inv_freq, k = None, out_k = None, position = 0, positions = None, position_ids = None,
              mode = NEOX, attn_factor = 1.0, q_norm = None, k_norm = None, norm_eps = 1e-6, norm_bias = 0.0,
              l4_beta = 0.0, l4_orig = 1, rotate_dims = 1, rotate_offset = 0):
    ext.rope(q, out_q, k, out_k, inv_freq, position, positions, position_ids, mode, attn_factor, q_norm, k_norm,
             norm_eps, norm_bias, l4_beta, l4_orig, rotate_dims, rotate_offset)


def ref_positions(bsz, seq, position, positions, position_ids, rotate_dims):
    """(bsz, seq, rotate_dims) int64 positions per token and rotate span"""
    if position_ids is not None:
        p = position_ids.long().cpu()
        if p.dim() == 2:
            p = p.unsqueeze(-1).expand(bsz, seq, rotate_dims)
        return p
    t = torch.arange(seq).view(1, seq)
    base = positions.long().cpu().view(bsz, 1) if positions is not None else torch.full((bsz, 1), position)
    return (t + base).unsqueeze(-1).expand(bsz, seq, rotate_dims)


def ref_angles(inv_freq, pos, bsz):
    """fp32 angles (bsz, seq, rotate_dims, rd/2): the kernel's single fp32 product, or the table entries"""
    inv = inv_freq.cpu()
    if inv.dim() == 1:
        return inv.view(1, 1, 1, -1) * pos.float().unsqueeze(-1)
    tab = inv.view(-1, inv.shape[-2], inv.shape[-1])
    if tab.shape[0] == 1:
        tab = tab.expand(bsz, -1, -1)
    b = torch.arange(bsz).view(bsz, 1, 1)
    return tab[b, pos]


def ref_norm(x, w, eps, bias):
    """Per-head RMS norm in float64 (no intermediate rounding; the bound accounts for the kernel's)"""
    x = x.double().cpu()
    x = x * torch.rsqrt(x.pow(2).mean(dim = -1, keepdim = True) + eps)
    if w is not None:
        x = x * (w.double().cpu() + bias)
    return x


def ref_rope(x, angles, mode, attn_factor, rotate_offset, rd):
    """
    x: (bsz, seq, H, hd) float64 (CPU), angles: (bsz, seq, rotate_dims, rd/2) fp32. Returns the rotated float64
    tensor and the per-element |v1| + |v2| (the magnitude the sin/cos error scales with; 0 outside the span)
    """
    out = x.clone()
    mag = torch.zeros_like(x)
    a = angles.double().unsqueeze(2)                               # (bsz, seq, 1, rdims, rd/2)
    sin = torch.sin(a) * attn_factor
    cos = torch.cos(a) * attn_factor
    for r in range(angles.shape[2]):
        o = rotate_offset + rd * r
        span = x[..., o : o + rd]
        if mode == NEOX:
            v1, v2 = span[..., : rd // 2], span[..., rd // 2 :]
            i1, i2 = slice(o, o + rd // 2), slice(o + rd // 2, o + rd)
        else:
            v1, v2 = span[..., 0::2], span[..., 1::2]
            i1, i2 = slice(o, o + rd, 2), slice(o + 1, o + rd, 2)
        s, c = sin[..., r, :], cos[..., r, :]
        out[..., i1] = v1 * c - v2 * s
        out[..., i2] = v2 * c + v1 * s
        m = v1.abs() + v2.abs()
        mag[..., i1] = m
        mag[..., i2] = m
    return out, mag


def sincos_err(angles, attn_factor):
    """Absolute sin/cos error bound of __sinf/__cosf on the fp32 angle: the hardware reduces the argument with an
    fp32 multiply by 1/(2 pi), so the reduced argument carries about one ulp of the angle (twice that as margin);
    inside [-pi, pi] the intrinsic's own error is ~2^-21.4"""
    return (2.0 * EPS32 * max(angles.abs().max().item(), 1.0) + 2.0 ** -21) * attn_factor


def assert_within(actual, expected, bound, msg = ""):
    """|actual - expected| <= bound elementwise, actual fp16 (or anything), expected/bound float64 on CPU"""
    a = actual.double().cpu()
    assert a.shape == expected.shape, f"{msg} shape {tuple(a.shape)} vs {tuple(expected.shape)}"
    assert torch.isfinite(a).all(), f"{msg} non-finite output"
    err = (a - expected).abs()
    bad = err > bound
    if bad.any():
        i = bad.nonzero()[0].tolist()
        raise AssertionError(f"{msg} {int(bad.sum())} elements out of bound, first at {i}: "
                             f"got {a[tuple(i)].item():.6g}, expected {expected[tuple(i)].item():.6g}, "
                             f"bound {bound[tuple(i)].item():.3g}, max excess {(err - bound).max().item():.3g}")


def rope_bound(ref, mag, angles, attn_factor, extra_rel = 0.0, pre_mag = None):
    """Per-element bound: fp16 rounding of the result, sin/cos error times the pair magnitude, plus extra_rel
    relative error carried in from rounded inputs to the rotation (pre_mag: their magnitude, default mag)"""
    e = sincos_err(angles, attn_factor)
    pm = mag if pre_mag is None else pre_mag
    return HALF_ULP16 * ref.abs() + e * mag + extra_rel * pm * attn_factor + 2.0 ** -24


# (bsz, seq, Hq, Hk, hd): decode-shaped grids split heads over gridDim.z (bsz * seq < 32), larger ones loop
# head rounds inside one block; odd head counts leave idle thread rows in the last round
geometries = [
    (1, 1, 8, 8, 128),
    (1, 1, 64, 8, 64),
    (1, 1, 28, 7, 96),
    (1, 1, 28, 7, 80),
    (1, 1, 16, 2, 32),
    (2, 3, 13, 3, 256),
    (1, 40, 32, 8, 128),
    (3, 17, 5, 1, 64),
    (1, 7, 4, 1, 512),
    (1, 3, 2, 1, 1024),
    (1, 2, 3, 0, 2048),
]


@pytest.mark.parametrize("geom", geometries, ids = lambda g: "x".join(map(str, g)))
@pytest.mark.parametrize("mode", [NEOX, GPTJ], ids = ["neox", "gptj"])
@pytest.mark.parametrize("in_place", [False, True], ids = ["oop", "inplace"])
@torch.inference_mode()
def test_rope_full_rotation(device, geom, mode, in_place):
    bsz, seq, hq, hk, hd = geom
    torch.manual_seed(0)
    q = torch.randn(bsz, seq, hq, hd, dtype = torch.half, device = device)
    k = torch.randn(bsz, seq, hk, hd, dtype = torch.half, device = device) if hk else None
    q0 = q.clone()
    k0 = k.clone() if k is not None else None
    inv_freq = default_inv_freq(hd, device)
    position = 37
    out_q = q if in_place else torch.empty_like(q)
    out_k = (k if in_place else torch.empty_like(k)) if k is not None else None
    call_rope(q, out_q, inv_freq, k, out_k, position = position, mode = mode)

    pos = ref_positions(bsz, seq, position, None, None, 1)
    ang = ref_angles(inv_freq, pos, bsz)
    for name, x0, out in (("q", q0, out_q), ("k", k0, out_k)):
        if x0 is None:
            continue
        ref, mag = ref_rope(x0.double().cpu(), ang, mode, 1.0, 0, hd)
        assert_within(out, ref, rope_bound(ref, mag, ang, 1.0), name)
    if not in_place:
        assert torch.equal(q, q0), "out-of-place rope modified q"
        if k is not None:
            assert torch.equal(k, k0), "out-of-place rope modified k"


@pytest.mark.parametrize("mode", [NEOX, GPTJ], ids = ["neox", "gptj"])
@pytest.mark.parametrize("variant", ["position", "positions", "position_ids"])
@torch.inference_mode()
def test_rope_position_sources(device, mode, variant):
    # Batched rows at different offsets; position_ids also non-monotonic and repeated
    bsz, seq, hq, hk, hd = 4, 9, 6, 2, 64
    torch.manual_seed(1)
    q = torch.randn(bsz, seq, hq, hd, dtype = torch.half, device = device)
    k = torch.randn(bsz, seq, hk, hd, dtype = torch.half, device = device)
    q0, k0 = q.clone(), k.clone()
    inv_freq = default_inv_freq(hd, device)
    position, positions, position_ids = 0, None, None
    if variant == "position":
        position = 1000
    elif variant == "positions":
        positions = torch.tensor([0, 5, 1999, 123], dtype = torch.int, device = device)
    else:
        position_ids = torch.randint(0, 2000, (bsz, seq), dtype = torch.int, device = device)
        position_ids[1, 3] = position_ids[1, 4]
    call_rope(q, q, inv_freq, k, k, position = position, positions = positions, position_ids = position_ids,
              mode = mode)
    pos = ref_positions(bsz, seq, position, positions, position_ids, 1)
    ang = ref_angles(inv_freq, pos, bsz)
    for name, x0, out in (("q", q0, q), ("k", k0, k)):
        ref, mag = ref_rope(x0.double().cpu(), ang, mode, 1.0, 0, hd)
        assert_within(out, ref, rope_bound(ref, mag, ang, 1.0), name)


@torch.inference_mode()
def test_rope_position_overrides_scalar(device):
    # positions / position_ids replace the scalar position (they are not added to it)
    q = torch.randn(2, 3, 2, 64, dtype = torch.half, device = device)
    inv_freq = default_inv_freq(64, device)
    positions = torch.tensor([4, 9], dtype = torch.int, device = device)
    a, b = torch.empty_like(q), torch.empty_like(q)
    call_rope(q, a, inv_freq, positions = positions, position = 0)
    call_rope(q, b, inv_freq, positions = positions, position = 500)
    assert torch.equal(a, b)
    pid = torch.tensor([[4, 5, 6], [9, 10, 11]], dtype = torch.int, device = device)
    c = torch.empty_like(q)
    call_rope(q, c, inv_freq, position_ids = pid, position = 777)
    assert torch.equal(a, c)


@pytest.mark.parametrize("mode", [NEOX, GPTJ], ids = ["neox", "gptj"])
@pytest.mark.parametrize("hd, rd, offset", [(128, 64, 0), (128, 32, 0), (96, 64, 0), (192, 64, 128), (512, 64, 448),
                                            (128, 64, 32)])
@torch.inference_mode()
def test_rope_partial_rotation(device, mode, hd, rd, offset):
    # Partial rotary: inv_freq of rd / 2 entries rotates [offset, offset + rd); every other dim passes through
    # bit-exact (no norm)
    bsz, seq, hq, hk = 2, 5, 4, 1
    torch.manual_seed(2)
    q = torch.randn(bsz, seq, hq, hd, dtype = torch.half, device = device)
    k = torch.randn(bsz, seq, hk, hd, dtype = torch.half, device = device)
    out_q, out_k = torch.empty_like(q), torch.empty_like(k)
    inv_freq = default_inv_freq(rd, device)
    positions = torch.tensor([3, 700], dtype = torch.int, device = device)
    call_rope(q, out_q, inv_freq, k, out_k, positions = positions, mode = mode, rotate_offset = offset)
    pos = ref_positions(bsz, seq, 0, positions, None, 1)
    ang = ref_angles(inv_freq, pos, bsz)
    keep = torch.ones(hd, dtype = torch.bool)
    keep[offset : offset + rd] = False
    for name, x0, out in (("q", q, out_q), ("k", k, out_k)):
        ref, mag = ref_rope(x0.double().cpu(), ang, mode, 1.0, offset, rd)
        assert_within(out, ref, rope_bound(ref, mag, ang, 1.0), name)
        assert torch.equal(out[..., keep], x0[..., keep]), f"{name}: dims outside the rotated span changed"


@pytest.mark.parametrize("mode", [NEOX, GPTJ], ids = ["neox", "gptj"])
@torch.inference_mode()
def test_rope_strided_head_view_in_place(device, mode):
    # Trailing / leading slice views of wider heads (dsv4, MLA indexer): the kernel works through stride(2) and
    # must leave the rest of each head (and of the buffer) untouched
    bsz, seq, h, full, rd = 2, 6, 3, 192, 64
    torch.manual_seed(3)
    buf = torch.randn(bsz, seq, h, full, dtype = torch.half, device = device)
    orig = buf.clone()
    inv_freq = default_inv_freq(rd, device)
    for sl in (slice(full - rd, full), slice(0, rd)):
        buf.copy_(orig)
        v = buf[..., sl]
        assert not v.is_contiguous()
        call_rope(v, v, inv_freq, position = 11, mode = mode)
        pos = ref_positions(bsz, seq, 11, None, None, 1)
        ang = ref_angles(inv_freq, pos, bsz)
        ref, mag = ref_rope(orig[..., sl].double().cpu(), ang, mode, 1.0, 0, rd)
        assert_within(buf[..., sl], ref, rope_bound(ref, mag, ang, 1.0), str(sl))
        rest = torch.ones(full, dtype = torch.bool)
        rest[sl] = False
        assert torch.equal(buf[..., rest], orig[..., rest]), "elements outside the view changed"


@torch.inference_mode()
def test_rope_padded_out_untouched(device):
    # Out-of-place into a padded-stride view of a sentinel buffer: only the view's elements are written
    bsz, seq, h, hd, pad = 1, 4, 5, 64, 16
    sentinel = -1234.0
    qbuf = torch.randn(bsz, seq, h, hd + pad, dtype = torch.half, device = device)
    obuf = torch.full_like(qbuf, sentinel)
    q, out = qbuf[..., :hd], obuf[..., :hd]
    inv_freq = default_inv_freq(hd, device)
    call_rope(q, out, inv_freq, position = 3)
    assert (obuf[..., hd:] == sentinel).all(), "padding of out_q written"
    ang = ref_angles(inv_freq, ref_positions(bsz, seq, 3, None, None, 1), bsz)
    ref, mag = ref_rope(q.double().cpu(), ang, NEOX, 1.0, 0, hd)
    assert_within(out, ref, rope_bound(ref, mag, ang, 1.0))


@pytest.mark.parametrize("mode", [NEOX, GPTJ], ids = ["neox", "gptj"])
@pytest.mark.parametrize("variant", ["shared_ids", "per_span_ids"])
@torch.inference_mode()
def test_rope_rotate_dims_2(device, mode, variant):
    # rotate_dims = 2 (2-D vision positions): hd = 2 * rd, span r uses position_ids[..., r] (or the same 2-D id)
    bsz, seq, hq, hk, rd = 1, 12, 4, 4, 32
    hd = 2 * rd
    torch.manual_seed(4)
    q = torch.randn(bsz, seq, hq, hd, dtype = torch.half, device = device)
    k = torch.randn(bsz, seq, hk, hd, dtype = torch.half, device = device)
    q0, k0 = q.clone(), k.clone()
    inv_freq = default_inv_freq(rd, device)
    if variant == "per_span_ids":
        pid = torch.randint(0, 64, (bsz, seq, 2), dtype = torch.int, device = device)
    else:
        pid = torch.randint(0, 64, (bsz, seq), dtype = torch.int, device = device)
    call_rope(q, q, inv_freq, k, k, position_ids = pid, mode = mode, rotate_dims = 2)
    pos = ref_positions(bsz, seq, 0, None, pid, 2)
    ang = ref_angles(inv_freq, pos, bsz)
    for name, x0, out in (("q", q0, q), ("k", k0, k)):
        ref, mag = ref_rope(x0.double().cpu(), ang, mode, 1.0, 0, rd)
        assert_within(out, ref, rope_bound(ref, mag, ang, 1.0), name)


@pytest.mark.parametrize("attn_factor", [1.0, 0.5, 1.3])
@torch.inference_mode()
def test_rope_attn_factor(device, attn_factor):
    # attn_factor scales sin and cos, i.e. the rotated span only
    bsz, seq, h, hd, rd = 1, 5, 4, 128, 64
    q = torch.randn(bsz, seq, h, hd, dtype = torch.half, device = device)
    out = torch.empty_like(q)
    inv_freq = default_inv_freq(rd, device)
    call_rope(q, out, inv_freq, position = 20, attn_factor = attn_factor)
    ang = ref_angles(inv_freq, ref_positions(bsz, seq, 20, None, None, 1), bsz)
    ref, mag = ref_rope(q.double().cpu(), ang, NEOX, attn_factor, 0, rd)
    assert_within(out, ref, rope_bound(ref, mag, ang, attn_factor))
    assert torch.equal(out[..., rd:], q[..., rd:])


@pytest.mark.parametrize("table_dim", [2, 3])
@pytest.mark.parametrize("mode", [NEOX, GPTJ], ids = ["neox", "gptj"])
@torch.inference_mode()
def test_rope_inv_freq_table(device, table_dim, mode):
    # A 2-D (L, rd/2) or 3-D (bsz, L, rd/2) inv_freq is an angle table indexed by [batch, pos]; a 2-D table is
    # only addressed per batch row 0 (the kernel strides batch rows by a whole table), so it is used with bsz 1
    hd, rd, L = 128, 128, 40
    bsz = 1 if table_dim == 2 else 3
    seq, h = 6, 3
    torch.manual_seed(5)
    tab = (torch.rand(bsz, L, rd // 2, device = device) * 8.0 - 4.0)
    if table_dim == 2:
        tab = tab[0]
    q = torch.randn(bsz, seq, h, hd, dtype = torch.half, device = device)
    out = torch.empty_like(q)
    for kw in (dict(position = 7), dict(position_ids = torch.randint(0, L, (bsz, seq), dtype = torch.int, device = device)),
               dict(positions = torch.randint(0, L - seq, (bsz,), dtype = torch.int, device = device))):
        call_rope(q, out, tab, mode = mode, **kw)
        pos = ref_positions(bsz, seq, kw.get("position", 0), kw.get("positions"), kw.get("position_ids"), 1)
        ang = ref_angles(tab, pos, bsz)
        ref, mag = ref_rope(q.double().cpu(), ang, mode, 1.0, 0, rd)
        assert_within(out, ref, rope_bound(ref, mag, ang, 1.0), str(list(kw)))


@pytest.mark.parametrize("norm_dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize("mode", [NEOX, GPTJ], ids = ["neox", "gptj"])
@pytest.mark.parametrize("hd, rd, offset, norm_bias", [(128, 128, 0, 0.0), (128, 128, 0, 1.0), (80, 80, 0, 0.0),
                                                       (512, 64, 448, 0.0), (256, 64, 0, 1.0)])
@torch.inference_mode()
def test_rope_qk_norm(device, norm_dtype, mode, hd, rd, offset, norm_bias):
    # Per-head RMS norm over the full head (q heads with q_norm, k heads with k_norm), then the rotation
    bsz, seq, hq, hk = 2, 3, 6, 2
    eps = 1e-6
    torch.manual_seed(6)
    q = torch.randn(bsz, seq, hq, hd, dtype = torch.half, device = device) * 3
    k = torch.randn(bsz, seq, hk, hd, dtype = torch.half, device = device) * 0.2
    qw = (torch.randn(hd, device = device) * 0.5 + (0.0 if norm_bias else 1.0)).to(norm_dtype)
    kw = (torch.randn(hd, device = device) * 0.5 + (0.0 if norm_bias else 1.0)).to(norm_dtype)
    out_q, out_k = torch.empty_like(q), torch.empty_like(k)
    inv_freq = default_inv_freq(rd, device)
    positions = torch.tensor([2, 1500], dtype = torch.int, device = device)
    call_rope(q, out_q, inv_freq, k, out_k, positions = positions, mode = mode, q_norm = qw, k_norm = kw,
              norm_eps = eps, norm_bias = norm_bias, rotate_offset = offset)
    ang = ref_angles(inv_freq, ref_positions(bsz, seq, 0, positions, None, 1), bsz)
    for name, x0, w, out in (("q", q, qw, out_q), ("k", k, kw, out_k)):
        xn = ref_norm(x0, w, eps, norm_bias)
        ref, mag = ref_rope(xn, ang, mode, 1.0, offset, rd)
        # The normed head is rounded to fp16 once (and with an fp16 weight the bias add and the product are fp16
        # roundings too): up to three half-ulps of relative error on each input of the rotation. Unrotated dims
        # carry that error directly
        pre = 3 * HALF_ULP16
        bound = rope_bound(ref, mag, ang, 1.0, extra_rel = pre)
        bound = bound + pre * xn.abs() * (mag == 0)
        assert_within(out, ref, bound, name)


@pytest.mark.parametrize("variant", ["position", "positions", "position_ids"])
@torch.inference_mode()
def test_rope_llama4_scaling(device, variant):
    # Llama-4 scale on q heads only, after the rotation: 1 + beta * ln(1 + floor(pos / original))
    bsz, seq, hq, hk, hd = 2, 6, 4, 2, 64
    beta, orig = 0.1, 4
    torch.manual_seed(7)
    q = torch.randn(bsz, seq, hq, hd, dtype = torch.half, device = device)
    k = torch.randn(bsz, seq, hk, hd, dtype = torch.half, device = device)
    out_q, out_k = torch.empty_like(q), torch.empty_like(k)
    inv_freq = default_inv_freq(hd, device)
    kwargs = {}
    if variant == "position":
        kwargs["position"] = 3
    elif variant == "positions":
        kwargs["positions"] = torch.tensor([0, 30], dtype = torch.int, device = device)
    else:
        kwargs["position_ids"] = torch.randint(0, 100, (bsz, seq), dtype = torch.int, device = device)
    call_rope(q, out_q, inv_freq, k, out_k, l4_beta = beta, l4_orig = orig, **kwargs)
    pos = ref_positions(bsz, seq, kwargs.get("position", 0), kwargs.get("positions"), kwargs.get("position_ids"), 1)
    ang = ref_angles(inv_freq, pos, bsz)
    scale = (1.0 + beta * torch.log1p(torch.div(pos[..., 0], orig, rounding_mode = "floor").double()))
    scale = scale.view(bsz, seq, 1, 1)
    ref_q, mag_q = ref_rope(q.double().cpu(), ang, NEOX, 1.0, 0, hd)
    ref_k, mag_k = ref_rope(k.double().cpu(), ang, NEOX, 1.0, 0, hd)
    # q: the rotated value is rounded to fp16, then scaled and rounded again (two half-ulps), __logf ~2^-21
    bq = (rope_bound(ref_q, mag_q, ang, 1.0) * scale + (HALF_ULP16 + 2.0 ** -20) * (ref_q * scale).abs())
    assert_within(out_q, ref_q * scale, bq, "q")
    assert_within(out_k, ref_k, rope_bound(ref_k, mag_k, ang, 1.0), "k")


@pytest.mark.parametrize("mode", [NEOX, GPTJ], ids = ["neox", "gptj"])
@torch.inference_mode()
def test_rope_deterministic_and_inverse(device, mode):
    # Repeat calls are bit-identical; rotating by a negated inv_freq (dsv4 de-rotation) undoes the rotation
    q = torch.randn(1, 33, 8, 128, dtype = torch.half, device = device)
    inv_freq = default_inv_freq(128, device)
    a, b = torch.empty_like(q), torch.empty_like(q)
    call_rope(q, a, inv_freq, position = 100, mode = mode)
    call_rope(q, b, inv_freq, position = 100, mode = mode)
    assert torch.equal(a, b)
    back = torch.empty_like(q)
    call_rope(a, back, -inv_freq, position = 100, mode = mode)
    ang = ref_angles(inv_freq, ref_positions(1, 33, 100, None, None, 1), 1)
    _, mag = ref_rope(q.double().cpu(), ang, mode, 1.0, 0, 128)
    # Two rotations, each adding an fp16 rounding (relative to the pair magnitude) and a sin/cos error; the
    # first rotation's error is carried through the second unchanged in size
    e = sincos_err(ang, 1.0)
    bound = 2 * (HALF_ULP16 + e) * mag * 1.01 + 2.0 ** -24
    assert_within(back, q.double().cpu(), bound)


@pytest.mark.xfail(strict = True, reason = "sin/cos via __sinf/__cosf: error grows with the angle (about one fp32 "
                                           "ulp of it) and exceeds fp16 resolution at long context")
@torch.inference_mode()
def test_rope_long_positions_match_accurate_sincos(device):
    # The HF formulation computes the same fp32 angle and takes accurate sin/cos of it. Against that, the result
    # should be off only by the fp16 output rounding (plus a few fp32 ulps of the rotation arithmetic)
    hd = 128
    inv_freq = default_inv_freq(hd, device)
    pid = torch.randint(32768, 262144, (1, 256), dtype = torch.int, device = device)
    q = torch.randn(1, 256, 4, hd, dtype = torch.half, device = device)
    out = torch.empty_like(q)
    call_rope(q, out, inv_freq, position_ids = pid)
    ang = ref_angles(inv_freq, ref_positions(1, 256, 0, None, pid, 1), 1)
    ref, mag = ref_rope(q.double().cpu(), ang, NEOX, 1.0, 0, hd)
    bound = HALF_ULP16 * ref.abs() + 8 * EPS32 * mag + 2.0 ** -24
    assert_within(out, ref, bound)


# Validation: every TORCH_CHECK in rope_gr raises before launching

def _args(device, **over):
    q = torch.zeros(1, 2, 2, 64, dtype = torch.half, device = device)
    a = dict(q = q, out_q = torch.empty_like(q), k = None, out_k = None, inv_freq = default_inv_freq(64, device),
             position = 0, positions = None, position_ids = None, mode = NEOX, attn_factor = 1.0, q_norm = None,
             k_norm = None, norm_eps = 1e-6, norm_bias = 0.0, l4_beta = 0.0, l4_orig = 1, rotate_dims = 1,
             rotate_offset = 0)
    a.update(over)
    return a


def _call(a):
    ext.rope(a["q"], a["out_q"], a["k"], a["out_k"], a["inv_freq"], a["position"], a["positions"],
             a["position_ids"], a["mode"], a["attn_factor"], a["q_norm"], a["k_norm"], a["norm_eps"],
             a["norm_bias"], a["l4_beta"], a["l4_orig"], a["rotate_dims"], a["rotate_offset"])


def _invalid_cases(device):
    z = lambda *s, dt = torch.half: torch.zeros(*s, dtype = dt, device = device)
    q = z(1, 2, 2, 64)
    i32 = torch.int
    return {
        "q_fp32": dict(q = z(1, 2, 2, 64, dt = torch.float), out_q = z(1, 2, 2, 64, dt = torch.float)),
        "q_rank3": dict(q = z(2, 2, 64), out_q = z(2, 2, 64)),
        "q_inner_strided": dict(q = z(1, 2, 2, 128)[..., ::2], out_q = z(1, 2, 2, 128)[..., ::2]),
        "out_q_layout": dict(out_q = z(1, 2, 2, 80)[..., :64]),
        "k_fp32": dict(k = z(1, 2, 1, 64, dt = torch.float), out_k = z(1, 2, 1, 64, dt = torch.float)),
        "k_seq": dict(k = z(1, 3, 1, 64), out_k = z(1, 3, 1, 64)),
        "k_head_dim": dict(k = z(1, 2, 1, 32), out_k = z(1, 2, 1, 32)),
        "inv_freq_fp16": dict(inv_freq = default_inv_freq(64, device).half()),
        "inv_freq_rank4": dict(inv_freq = torch.zeros(1, 1, 4, 32, device = device)),
        "rotate_dims_0": dict(rotate_dims = 0),
        "rotate_dims_5": dict(rotate_dims = 5),
        "rotate_dims_inconsistent": dict(rotate_dims = 2),
        "rotate_offset_negative": dict(rotate_offset = -2),
        "rotate_offset_overflow": dict(inv_freq = default_inv_freq(32, device), rotate_offset = 34),
        "positions_and_ids": dict(positions = torch.zeros(1, dtype = i32, device = device),
                                  position_ids = torch.zeros(1, 2, dtype = i32, device = device)),
        "positions_int64": dict(positions = torch.zeros(1, dtype = torch.long, device = device)),
        "positions_shape": dict(positions = torch.zeros(2, dtype = i32, device = device)),
        "positions_rank2": dict(positions = torch.zeros(1, 1, dtype = i32, device = device)),
        "position_ids_shape": dict(position_ids = torch.zeros(1, 3, dtype = i32, device = device)),
        "position_ids_rank3_mismatch": dict(position_ids = torch.zeros(1, 2, 3, dtype = i32, device = device)),
        "position_ids_noncontig": dict(position_ids = torch.zeros(1, 4, dtype = i32, device = device)[:, ::2]),
        "q_norm_size": dict(q_norm = z(32)),
        "norm_dtype_mismatch": dict(k = z(1, 2, 1, 64), out_k = z(1, 2, 1, 64), q_norm = z(64),
                                    k_norm = z(64, dt = torch.bfloat16)),
        "norm_fp32": dict(q_norm = z(64, dt = torch.float)),
        "mode_none": dict(mode = 0),
    }


INVALID_CASES = [
    "q_fp32", "q_rank3", "q_inner_strided", "out_q_layout", "k_fp32", "k_seq", "k_head_dim", "inv_freq_fp16",
    "inv_freq_rank4", "rotate_dims_0", "rotate_dims_5", "rotate_dims_inconsistent", "rotate_offset_negative",
    "rotate_offset_overflow", "positions_and_ids", "positions_int64", "positions_shape", "positions_rank2",
    "position_ids_shape", "position_ids_rank3_mismatch", "position_ids_noncontig", "q_norm_size",
    "norm_dtype_mismatch", "norm_fp32", "mode_none",
]


@pytest.mark.parametrize("case", INVALID_CASES)
@torch.inference_mode()
def test_rope_rejects_invalid(device, case):
    cases = _invalid_cases(device)
    assert set(cases) == set(INVALID_CASES)
    a = _args(device, **cases[case])
    # A rank-3 q fails on q.size(3) (IndexError) before the explicit rank check is reached
    with pytest.raises((RuntimeError, IndexError)):
        _call(a)
