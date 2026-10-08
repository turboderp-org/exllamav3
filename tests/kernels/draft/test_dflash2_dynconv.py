"""
ext.dflash2_dynconv(x, dyn, base, out, group_size, accumulate), the DFlash2 grouped dynamic causal convolution
(dflash2.cu), tested at the binding level (tests/modules/draft/test_dflash2.py tests the module wrapper
_grouped_dynamic_convolve against the module's torch path):

    out[b, t, c] (+)= sum_{k = 0}^{min(taps - 1, t)} (base[k, c] + dyn[b, t, k, c // group_size]) * x[b, t - k, c]

- x (bsz, seq, hidden) fp16/fp32 contiguous; dyn (bsz, seq, taps, hidden / group_size) fp16 with any strides;
  base (taps, hidden) fp16/bf16 contiguous; out (bsz, seq, hidden) fp16/fp32 contiguous, x's shape.
- accumulate adds into out (fp32 only); otherwise out is overwritten. Taps beyond the start of the sequence are
  skipped (causal, no state). Nothing outside out is written; empty batch/sequence is a no-op.
- Validation (TORCH_CHECK): ranks, contiguity, dtypes, group_size dividing hidden, base/dyn shapes, accumulate
  requiring fp32 out.

Reference: float64 torch over the formula (the taps loop vectorized over batch/time/channel). The kernel sums
the taps in fp32 (weights formed by one fp32 add), so its error is a few fp32 ulps of sum |w * x|, plus the
output rounding (fp16 half-ulp) or the accumulate add.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(not getattr(ext, "HAS_DFLASH2", False), reason = "build without the DFlash2 kernels")

EPS32 = torch.finfo(torch.float32).eps


def ref_dynconv(x, dyn, base, group_size):
    """float64 (out, sum of |terms|), both (bsz, seq, hidden) on CPU"""
    x = x.double().cpu()
    dyn = dyn.double().cpu()
    base = base.double().cpu()
    bsz, seq, hidden = x.shape
    taps = base.shape[0]
    dyn_c = dyn.repeat_interleave(group_size, dim = -1)              # (bsz, seq, taps, hidden)
    out = torch.zeros_like(x)
    mag = torch.zeros_like(x)
    for k in range(min(taps, seq)):
        w = base[k].view(1, 1, hidden) + dyn_c[:, k:, k]
        term = w * x[:, : seq - k]
        out[:, k:] += term
        mag[:, k:] += term.abs()
    return out, mag


def bound_for(ref, mag, taps, out_dtype):
    # fp32 weight add and up to `taps` fp32 multiply-adds: (taps + 2) ulps of the magnitude sum is generous
    b = (taps + 2) * EPS32 * mag + 1e-30
    if out_dtype == torch.half:
        b = b + 2.0 ** -11 * ref.abs() + 2.0 ** -25
    return b


def assert_within(actual, expected, bound, msg = ""):
    a = actual.double().cpu()
    assert a.shape == expected.shape
    assert torch.isfinite(a).all(), f"{msg} non-finite output"
    err = (a - expected).abs()
    bad = err > bound
    assert not bad.any(), f"{msg} {int(bad.sum())} out of bound, max excess {(err - bound).max().item():.3g}"


def make(device, bsz, seq, hidden, taps, group_size, x_dtype, base_dtype, dyn_layout = "contig", seed = 0):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    groups = hidden // group_size
    x = torch.randn(bsz, seq, hidden, generator = g).to(device = device, dtype = x_dtype)
    base = torch.randn(taps, hidden, generator = g).to(device = device, dtype = base_dtype)
    d = torch.randn(bsz, seq, taps, groups, generator = g).half().to(device)
    if dyn_layout == "contig":
        dyn = d
    elif dyn_layout == "packed":
        # the module's layout: one slice of a (bsz, seq, 2, taps, groups) projection
        packed = torch.randn(bsz, seq, 2, taps, groups, generator = g).half().to(device)
        packed[:, :, 1] = d
        dyn = packed[:, :, 1]
    elif dyn_layout == "permuted":
        dyn = d.permute(3, 2, 1, 0).contiguous().permute(3, 2, 1, 0)
    elif dyn_layout == "broadcast_t":
        dyn = d[:, :1].expand(bsz, seq, taps, groups)
    else:
        raise ValueError(dyn_layout)
    return x, dyn, base


@pytest.mark.parametrize("bsz, seq, hidden, taps, group_size", [
    (1, 1, 96, 3, 16),
    (2, 8, 96, 3, 16),
    (1, 17, 192, 5, 24),
    (3, 4, 2048, 4, 128),
    (1, 2, 64, 6, 64),           # more taps than positions
    (2, 16, 4100, 2, 4),         # hidden not a multiple of the block width
    (1, 5, 8, 1, 1),
])
@pytest.mark.parametrize("x_dtype, out_dtype", [(torch.half, torch.half), (torch.float, torch.float),
                                                (torch.half, torch.float), (torch.float, torch.half)])
@pytest.mark.parametrize("base_dtype", [torch.half, torch.bfloat16])
@torch.inference_mode()
def test_dynconv(device, bsz, seq, hidden, taps, group_size, x_dtype, out_dtype, base_dtype):
    x, dyn, base = make(device, bsz, seq, hidden, taps, group_size, x_dtype, base_dtype)
    sentinel = -321.0
    buf = torch.full((bsz + 2, seq, hidden), sentinel, dtype = out_dtype, device = device)
    out = buf[1 : bsz + 1]
    x0, dyn0, base0 = x.clone(), dyn.clone(), base.clone()
    ext.dflash2_dynconv(x, dyn, base, out, group_size, False)
    ref, mag = ref_dynconv(x, dyn, base, group_size)
    assert_within(out, ref, bound_for(ref, mag, taps, out_dtype))
    assert (buf[0] == sentinel).all() and (buf[bsz + 1] == sentinel).all(), "wrote outside out"
    assert torch.equal(x, x0) and torch.equal(dyn, dyn0) and torch.equal(base, base0), "inputs modified"


@pytest.mark.parametrize("dyn_layout", ["packed", "permuted", "broadcast_t"])
@torch.inference_mode()
def test_dynconv_dyn_strides(device, dyn_layout):
    bsz, seq, hidden, taps, group_size = 2, 7, 96, 3, 16
    x, dyn, base = make(device, bsz, seq, hidden, taps, group_size, torch.half, torch.bfloat16, dyn_layout, seed = 1)
    assert not dyn.is_contiguous()
    out = torch.empty_like(x)
    ext.dflash2_dynconv(x, dyn, base, out, group_size, False)
    ref, mag = ref_dynconv(x, dyn, base, group_size)
    assert_within(out, ref, bound_for(ref, mag, taps, torch.half), dyn_layout)


@pytest.mark.parametrize("x_dtype", [torch.half, torch.float])
@torch.inference_mode()
def test_dynconv_accumulate(device, x_dtype):
    bsz, seq, hidden, taps, group_size = 2, 9, 256, 3, 32
    x, dyn, base = make(device, bsz, seq, hidden, taps, group_size, x_dtype, torch.bfloat16, seed = 2)
    residual = torch.randn(bsz, seq, hidden, device = device) * 100
    prev = residual.double().cpu()
    ext.dflash2_dynconv(x, dyn, base, residual, group_size, True)
    ref, mag = ref_dynconv(x, dyn, base, group_size)
    # one more fp32 rounding for the residual add
    bound = bound_for(ref, mag, taps, torch.float) + EPS32 * (prev + ref).abs()
    assert_within(residual, prev + ref, bound)


@torch.inference_mode()
def test_dynconv_fp32_range(device):
    # fp32 in and out keeps values beyond the fp16 range
    x = torch.full((1, 8, 96), 1.5e5, device = device)
    dyn = torch.zeros(1, 8, 2, 6, dtype = torch.half, device = device)
    base = torch.ones(2, 96, dtype = torch.bfloat16, device = device)
    out = torch.empty_like(x)
    ext.dflash2_dynconv(x, dyn, base, out, 16, False)
    assert (out[:, 0] == 1.5e5).all() and (out[:, 1:] == 3.0e5).all()


@torch.inference_mode()
def test_dynconv_deterministic(device):
    x, dyn, base = make(device, 2, 12, 1024, 4, 64, torch.half, torch.half, seed = 3)
    a, b = torch.empty_like(x), torch.empty_like(x)
    ext.dflash2_dynconv(x, dyn, base, a, 64, False)
    ext.dflash2_dynconv(x, dyn, base, b, 64, False)
    assert torch.equal(a, b)


@pytest.mark.parametrize("shape", [(0, 4, 96), (1, 0, 96)])
@torch.inference_mode()
def test_dynconv_empty_is_noop(device, shape):
    bsz, seq, hidden = shape
    x = torch.zeros(shape, dtype = torch.half, device = device)
    dyn = torch.zeros(bsz, seq, 2, 6, dtype = torch.half, device = device)
    base = torch.zeros(2, 96, dtype = torch.half, device = device)
    out = torch.empty_like(x)
    ext.dflash2_dynconv(x, dyn, base, out, 16, False)
    torch.cuda.synchronize(device)


def _invalid(device):
    h = lambda *s, dt = torch.half: torch.zeros(*s, dtype = dt, device = device)
    x, dyn, base, out = h(1, 4, 96), h(1, 4, 2, 6), h(2, 96), h(1, 4, 96)
    return {
        "x_rank2": (h(4, 96), dyn, base, h(4, 96), 16, False),
        "dyn_rank3": (x, h(1, 4, 12), base, out, 16, False),
        "base_rank1": (x, dyn, h(192), out, 16, False),
        "x_noncontig": (h(1, 4, 192)[..., :96], dyn, base, out, 16, False),
        "out_noncontig": (x, dyn, base, h(1, 4, 192)[..., :96], 16, False),
        "base_noncontig": (x, dyn, h(2, 192)[:, :96], out, 16, False),
        "out_shape": (x, dyn, base, h(1, 5, 96), 16, False),
        "dyn_fp32": (x, h(1, 4, 2, 6, dt = torch.float), base, out, 16, False),
        "x_bf16": (h(1, 4, 96, dt = torch.bfloat16), dyn, base, out, 16, False),
        "out_bf16": (x, dyn, base, h(1, 4, 96, dt = torch.bfloat16), 16, False),
        "base_fp32": (x, dyn, h(2, 96, dt = torch.float), out, 16, False),
        "accumulate_fp16_out": (x, dyn, base, out, 16, True),
        "group_not_dividing": (x, dyn, base, out, 7, False),
        "group_zero": (x, dyn, base, out, 0, False),
        "base_width": (x, dyn, h(2, 64), out, 16, False),
        "dyn_taps": (x, h(1, 4, 3, 6), base, out, 16, False),
        "dyn_groups": (x, h(1, 4, 2, 5), base, out, 16, False),
        "dyn_seq": (x, h(1, 3, 2, 6), base, out, 16, False),
        # seq and batch index grid.y / grid.z, capped at 65535
        "seq_grid": (h(1, 65536, 16), h(1, 65536, 1, 1), h(1, 16), h(1, 65536, 16), 16, False),
        "bsz_grid": (h(65536, 1, 16), h(65536, 1, 1, 1), h(1, 16), h(65536, 1, 16), 16, False),
    }


INVALID = ["x_rank2", "dyn_rank3", "base_rank1", "x_noncontig", "out_noncontig", "base_noncontig", "out_shape",
           "dyn_fp32", "x_bf16", "out_bf16", "base_fp32", "accumulate_fp16_out", "group_not_dividing",
           "group_zero", "base_width", "dyn_taps", "dyn_groups", "dyn_seq", "seq_grid", "bsz_grid"]
INVALID_MATCH = {"seq_grid": "at most 65535", "bsz_grid": "at most 65535"}


@pytest.mark.parametrize("case", INVALID)
@torch.inference_mode()
def test_dynconv_rejects_invalid(device, case):
    cases = _invalid(device)
    assert set(cases) == set(INVALID)
    with pytest.raises(RuntimeError, match = INVALID_MATCH.get(case)):
        ext.dflash2_dynconv(*cases[case])


# Zero-size inputs of the DFlash2 kernels (this is the only kernel-level DFlash2 file; tests/modules/draft covers
# dflash2_selector_walk and dflash2_topk through the module):
# - dflash2_dynconv: empty batch, sequence or hidden is a no-op; taps = 0 is the empty sum (out = 0, or unchanged
#   when accumulating)
# - dflash2_selector_walk: empty batch is a no-op; rows = 0 still writes the anchor column (out[:, 0] = anchor,
#   conf[:, 0] = 0); rank = 0 makes every pairwise term the empty dot product, so each row picks the argmax of its
#   unary scores; an empty candidate list (k = 0) is an argmax over nothing and raises
# - dflash2_topk: empty batch or rows is a no-op; vocab = 0 is a top-k over nothing and raises

def _assert_device_usable(device):
    torch.cuda.synchronize(device)
    assert (torch.ones(4, device = device) + 1).sum().item() == 8.0, "a later torch op failed after the empty call"


@pytest.mark.parametrize("bsz, seq, hidden, taps", [(0, 4, 96, 2), (2, 0, 96, 2), (2, 4, 0, 2), (2, 4, 96, 0)])
@pytest.mark.parametrize("accumulate", [False, True])
@torch.inference_mode()
def test_dynconv_empty(device, bsz, seq, hidden, taps, accumulate):
    x = torch.randn(bsz, seq, hidden, device = device)
    dyn = torch.randn(bsz, seq, taps, hidden // 16, device = device).half()
    base = torch.randn(taps, hidden, device = device).half()
    out = torch.full((bsz, seq, hidden), 777.0, device = device)
    ext.dflash2_dynconv(x, dyn, base, out, 16, accumulate)
    expect = torch.full_like(out, 777.0) if accumulate or taps > 0 else torch.zeros_like(out)
    assert torch.equal(out, expect)
    _assert_device_usable(device)


def _walk(device, bsz, rows, k, rank, vocab = 50, seed = 0):
    g = torch.Generator().manual_seed(seed)
    unary = torch.randn(bsz, rows, k, generator = g).to(device)
    cands = torch.randint(0, vocab, (bsz, rows, k), generator = g).to(device)
    gate = torch.randn(bsz, rows, rank, generator = g).half().to(device)
    cb = torch.randn(vocab, rank, generator = g).half().to(device)
    anchor = torch.randint(0, vocab, (bsz,), generator = g).to(device)
    out = torch.full((bsz, rows + 1), -7, dtype = torch.long, device = device)
    conf = torch.full((bsz, rows + 1), 777.0, device = device)
    ext.dflash2_selector_walk(unary, cands, gate, cb, cb.clone(), anchor, out, conf)
    return unary, cands, anchor, out, conf


@pytest.mark.parametrize("bsz, rows, k, rank, error", [
    (0, 3, 4, 8, None),
    (2, 0, 4, 8, None),
    (2, 3, 4, 0, None),
    (2, 3, 0, 8, "argmax over an empty candidate list"),
    (2, 0, 0, 8, "argmax over an empty candidate list"),
    (0, 3, 0, 8, "argmax over an empty candidate list"),
])
@torch.inference_mode()
def test_selector_walk_empty(device, bsz, rows, k, rank, error):
    if error:
        with pytest.raises(RuntimeError, match = error):
            _walk(device, bsz, rows, k, rank)
    else:
        unary, cands, anchor, out, conf = _walk(device, bsz, rows, k, rank)
        best = unary.argmax(dim = -1, keepdim = True)
        assert torch.equal(out[:, 0], anchor)
        assert torch.equal(out[:, 1:], cands.gather(-1, best).squeeze(-1))
        assert torch.equal(conf[:, 0], torch.zeros_like(conf[:, 0]))
        assert torch.equal(conf[:, 1:], unary.gather(-1, best).squeeze(-1))
    _assert_device_usable(device)


@pytest.mark.parametrize("bsz, rows, vocab, error", [(0, 2, 100, None), (2, 0, 100, None), (2, 2, 0, "top-k over an empty vocab")])
@torch.inference_mode()
def test_topk_empty(device, bsz, rows, vocab, error):
    logits = torch.randn(bsz, rows, max(vocab, 64), device = device).half()
    values = torch.full((bsz, rows, 8), 777.0, device = device)
    indices = torch.full((bsz, rows, 8), -7, dtype = torch.long, device = device)
    call = lambda: ext.dflash2_topk(logits, vocab, 1.0, 0.0, values, indices)
    if error:
        with pytest.raises(RuntimeError, match = error):
            call()
    else:
        call()
    assert (values == 777.0).all() and (indices == -7).all()
    _assert_device_usable(device)
