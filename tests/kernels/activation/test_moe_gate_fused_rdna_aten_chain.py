"""
Parity matrix for moe_gate_fused_rdna against the ORIGINAL ATen chain it replaced on the
RDNA block-sparse MLP decode route -- not just an fp32 model (test_moe_gate_fused_rdna.py
holds that base). The reference here reconstructs the pre-fusion ops verbatim:

    dst.copy_(routed)
    g = torch.sigmoid(torch.matmul(y.to(Float), w.reshape(-1, 1).to(Float)))
    dst.addcmul_(g, sh)                      # gated merge
    dst.add_(sh)                             # ungated shared-expert merge
    (no sh: dst is just the copy)

so both sides round through ATen's fp16 opmath (fp32 arithmetic, one round to the
accumulator dtype on store). The only intended difference is the gate logit: the kernel's
dot accumulates fp32 in a fixed strided + warp-tree order, torch.matmul uses its own
reduction -- numerically equivalent, not bit-identical.

Axes: rows (the kernel sees rows x H, rows being batch * slots -- NOT a sequence axis,
none is simulated here) over 1, 2, 4, 5, 8, 16, 64, where 5 is the MTP verify regression
case (draft window 4 + 1) from the scratch-sizing history, and H over 256/512/1000/1023/
4096 -- 1023 is the non-multiple-of-32 width, so blockDim-strided sweeps must handle
ragged tails. Output dtype half and float; mixed input dtypes are reject-only.

Tolerances are MEASURED, not guessed: every case records |kernel - chain| into a module
summary (max absolute at |ref| ~ 0, max relative at |ref| >= 1), printed as [gate-tol]
lines under pytest -s after the module finishes, and asserts against BUDGET below.
BUDGET is derived from the one semantic difference:

  - half output: both sides round once to fp16; a differing fp32 gate scalar can flip that
    rounding decision, moving the result by at most 1 ulp of the half result (~1e-3
    relative; 1 ulp at 60000 is 32, so only the relative term is meaningful at scale).
    Budget: rtol 2e-3 (2 ulp), atol 1e-3 (ulp-scale noise where |ref| < 1).
  - float output: fp32 vs fp32, no store-rounding difference at all. The logit gap between
    two fp32 reductions over W <= 4096 well-scaled terms is ~1e-7 absolute (worst-case
    bound n*eps*sum|terms| ~ 6e-6), sigmoid attenuates by <= 1/4, so |dst diff| ~ |sh| *
    1e-6 plus fp32 rounding ~ 1 ulp. Budget: rtol 5e-5, atol 1e-5 (generous headroom over
    the bound; the [gate-tol] report shows the true maximum for later tightening).

Edge cases pinned exactly: all-zero operands, negative operands, a gate logit of exactly
0.5 (y == 0 -> both sides compute the same sigmoid, results must be BIT-identical), sums
near fp16 max (64000, representable) without overflowing the target, and bit-identical
output across repeated launches on identical input (determinism of the block reduction).

Skips cleanly on non-ROCm builds (the binding is compiled only for ROCm; the attribute
check covers CUDA builds) -- the CUDA build path is untouched either way.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(
    not getattr(ext, "HAS_RDNA_GATE_FUSED", False),
    reason = "moe_gate_fused_rdna is compiled for ROCm (RDNA) builds only",
)

# --------------------------------------------------------------------------- tolerances

# See the module docstring for the derivation. The [gate-tol] report shows what was
# actually observed across this matrix on the run's device.
BUDGET = {
    torch.half:  {"rtol": 2e-3,  "atol": 1e-3},   # 1-ulp flip margin of the final fp16 round
    torch.float: {"rtol": 5e-5,  "atol": 1e-5},   # pure fp32 reduction-order gap
}

_OBSERVED = {}   # output dtype -> [max_abs, max_rel]

@pytest.fixture(scope = "module", autouse = True)
def _report_tolerances():
    yield
    for dt, (a, r) in sorted(_OBSERVED.items(), key = lambda kv: str(kv[0])):
        b = BUDGET[dt]
        print(f"[gate-tol] out={str(dt):16s} max_abs={a:.3e} max_rel={r:.3e} "
              f"budget(atol={b['atol']:.1e}, rtol={b['rtol']:.1e})")

def _record(dtype, got, ref):
    d = (got.float() - ref.float()).abs()
    _OBSERVED.setdefault(dtype, [0.0, 0.0])
    _OBSERVED[dtype][0] = max(_OBSERVED[dtype][0], d.max().item())
    _OBSERVED[dtype][1] = max(_OBSERVED[dtype][1], (d / ref.float().abs().clamp_min(1.0)).max().item())

def _assert_budget(dtype, got, ref):
    _record(dtype, got, ref)
    b = BUDGET[dtype]
    torch.testing.assert_close(got, ref, rtol = b["rtol"], atol = b["atol"])

# ----------------------------------------------------------------------- ATen reference

def aten_chain(routed, sh, y, w):
    """The unfused original, op for op, as the pre-fusion host route dispatched it."""
    dst = torch.empty_like(routed)
    dst.copy_(routed)
    if sh is None:
        return dst
    if y is not None:
        g = torch.sigmoid(torch.matmul(y.to(torch.float), w.reshape(-1, 1).to(torch.float)))
        dst.addcmul_(g, sh)
    else:
        dst.add_(sh)
    return dst

def _inputs(rows, H, W, dtype, device, seed):
    gen = torch.Generator(device = device).manual_seed(seed)
    # small-scale inputs: logits land near 0, gate near 0.5, no fp16 saturation
    y  = ((torch.randn(rows, W, generator = gen, device = device) / 8)).half()
    w  = ((torch.randn(W, generator = gen, device = device) / W ** 0.5)).half()
    routed = torch.randn(rows, H, generator = gen, dtype = dtype, device = device)
    sh     = torch.randn(rows, H, generator = gen, dtype = dtype, device = device)
    return y, w, routed, sh

# -------------------------------------------------------------------------- parity matrix

ROWS = [1, 2, 4, 5, 8, 16, 64]          # 5 = MTP verify (draft window 4 + 1)
WIDTHS = [256, 512, 1000, 1023, 4096]   # 1023: not divisible by 32 (ragged sweep tail)

@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("H", WIDTHS)
@pytest.mark.parametrize("rows", ROWS)
@torch.inference_mode()
def test_against_aten_chain_gated(device, rows, H, dtype):
    # rows > 8 (MAX_BSZN of the host route) probe kernel generality beyond the route's
    # bsz contract; the route itself clamps earlier
    y, w, routed, sh = _inputs(rows, H, 2048, dtype, device, seed = rows * 131 + H)
    ref = aten_chain(routed, sh, y, w)
    dst = torch.empty_like(routed)
    ext.moe_gate_fused_rdna(y, w, routed, sh, dst)
    _assert_budget(dtype, dst, ref)


@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("H", WIDTHS)
@pytest.mark.parametrize("rows", ROWS)
@torch.inference_mode()
def test_against_aten_chain_ungated(device, rows, H, dtype):
    # The ungated shared-expert merge: the kernel's W == 0 form vs the copy_/add_ pair
    y, w, routed, sh = _inputs(rows, H, 2048, dtype, device, seed = rows * 977 + H)
    ref = aten_chain(routed, sh, None, None)
    dst = torch.empty_like(routed)
    ext.moe_gate_fused_rdna(None, None, routed, sh, dst)
    # no gate means no reduction-order freedom: fp32 opmath both sides, one round to the
    # accumulator dtype -- the two paths are bit-identical
    _record(dtype, dst, ref)
    assert torch.equal(dst, ref)


@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("rows", [1, 5, 64])
@torch.inference_mode()
def test_against_aten_chain_copy(device, rows, dtype):
    # Plain copy of the routed sum: bit-exact by construction
    _, _, routed, _ = _inputs(rows, 1023, 2048, dtype, device, seed = rows)
    ref = aten_chain(routed, None, None, None)
    dst = torch.empty_like(routed)
    ext.moe_gate_fused_rdna(None, None, routed, None, dst)
    assert torch.equal(dst, ref)


@pytest.mark.parametrize("W", [1, 37])   # dots shorter than one stride, and off-by-one tails
@torch.inference_mode()
def test_against_aten_chain_short_gate(device, W):
    rows, H = 5, 1000   # MTP regression row count again
    y, w, routed, sh = _inputs(rows, H, W, torch.half, device, seed = W)
    ref = aten_chain(routed, sh, y, w)
    dst = torch.empty_like(routed)
    ext.moe_gate_fused_rdna(y, w, routed, sh, dst)
    _assert_budget(torch.half, dst, ref)

# --------------------------------------------------------------------------- edge cases

@torch.inference_mode()
def test_zeros_and_negative_values(device):
    # Zeros: gate(0) = 0.5 exactly on both sides -> bit-identical. Negatives ride along.
    rows, H, W = 5, 512, 2048
    y = torch.zeros(rows, W, dtype = torch.half, device = device)
    w = torch.zeros(W, dtype = torch.half, device = device)
    routed = torch.full((rows, H), -1.5, dtype = torch.half, device = device)
    sh = torch.full((rows, H), 4.0, dtype = torch.half, device = device)
    ref = aten_chain(routed, sh, y, w)          # -1.5 + 0.5 * 4.0 = 0.5
    dst = torch.empty_like(routed)
    ext.moe_gate_fused_rdna(y, w, routed, sh, dst)
    assert torch.equal(dst, ref)
    assert torch.equal(dst, torch.full_like(dst, 0.5))
    # all-zero operands everywhere
    z = torch.zeros(rows, H, dtype = torch.half, device = device)
    dst.zero_()
    ext.moe_gate_fused_rdna(y, w, z, z, dst)
    assert torch.equal(dst, z)


@torch.inference_mode()
def test_near_fp16_max_without_overflow(device):
    # 60000 + sigmoid(0) * 8000 = 64000: inside fp16 max (65504) and exactly
    # representable (ulp there is 32); with the logit fixed at 0 both sides round the
    # same fp32 sum once, so the result must be bit-identical -- an overflow guard, not
    # a tolerance case
    rows, H, W = 2, 256, 2048
    y = torch.zeros(rows, W, dtype = torch.half, device = device)
    w = torch.zeros(W, dtype = torch.half, device = device)
    routed = torch.full((rows, H), 60000.0, dtype = torch.half, device = device)
    sh = torch.full((rows, H), 8000.0, dtype = torch.half, device = device)
    ref = aten_chain(routed, sh, y, w)
    dst = torch.empty_like(routed)
    ext.moe_gate_fused_rdna(y, w, routed, sh, dst)
    assert torch.equal(dst, torch.full_like(dst, 64000.0))
    assert torch.equal(dst, ref)
    # and in float output the same sum has no fp16 rounding at all
    routed_f = routed.float(); sh_f = sh.float(); dst_f = torch.empty_like(routed_f)
    ref_f = aten_chain(routed_f, sh_f, y, w)
    ext.moe_gate_fused_rdna(y, w, routed_f, sh_f, dst_f)
    assert torch.equal(dst_f, ref_f)


@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@torch.inference_mode()
def test_repeated_launches_are_bit_identical(device, dtype):
    # Determinism of the block reduction: same buffers, same input, 10 launches -- any
    # nondeterministic deposition order or atomic in the gate dot shows up here
    rows, H, W = 5, 1023, 4096   # MTP rows, ragged width, longest dot
    y, w, routed, sh = _inputs(rows, H, W, dtype, device, seed = 424242)
    dst = torch.empty_like(routed)
    ext.moe_gate_fused_rdna(y, w, routed, sh, dst)
    first = dst.clone()
    for _ in range(10):
        ext.moe_gate_fused_rdna(y, w, routed, sh, dst)
        assert torch.equal(dst, first)

# ----------------------------------------------------------------------- reject matrix

def assert_device_ok(device):
    # A failed launch would leave an error for the next op on the device
    torch.cuda.synchronize(device)
    assert torch.ones(8, device = device).sum().item() == 8

@pytest.mark.parametrize("bad", [
    "sh_dtype", "y_dtype", "w_dtype", "dst_dtype", "routed_stride", "sh_stride", "y_without_w", "shape",
])
@torch.inference_mode()
def test_mixed_dtypes_and_layouts_rejected(device, bad):
    # The main matrix only ever pairs uniform dtypes; every mixed combination must be a
    # TORCH_CHECK reject, not a silently reinterpreted buffer
    routed = torch.randn(5, 256, dtype = torch.half, device = device)
    sh = torch.randn(5, 256, dtype = torch.half, device = device)
    y = torch.randn(5, 64, dtype = torch.half, device = device)
    w = torch.randn(64, dtype = torch.half, device = device)
    dst = torch.empty_like(routed)
    if bad == "sh_dtype":
        with pytest.raises(RuntimeError):
            ext.moe_gate_fused_rdna(None, None, routed, sh.float(), dst)
    elif bad == "y_dtype":
        with pytest.raises(RuntimeError):
            ext.moe_gate_fused_rdna(y.float(), w, routed, sh, dst)
    elif bad == "w_dtype":
        with pytest.raises(RuntimeError):
            ext.moe_gate_fused_rdna(y, w.float(), routed, sh, dst)
    elif bad == "dst_dtype":
        with pytest.raises(RuntimeError):
            ext.moe_gate_fused_rdna(None, None, routed, sh, torch.empty_like(routed.float()))
    elif bad == "routed_stride":
        with pytest.raises(RuntimeError):
            ext.moe_gate_fused_rdna(None, None, routed.t(), None, torch.empty(256, 5, dtype = torch.half, device = device))
    elif bad == "sh_stride":
        big = torch.randn(5, 512, dtype = torch.half, device = device)
        with pytest.raises(RuntimeError):
            ext.moe_gate_fused_rdna(None, None, routed, big[:, :256], dst)  # right shape, strided rows
    elif bad == "y_without_w":
        with pytest.raises(RuntimeError):
            ext.moe_gate_fused_rdna(y, None, routed, sh, dst)
    else:
        with pytest.raises(RuntimeError):
            ext.moe_gate_fused_rdna(None, None, routed, torch.randn(5, 257, dtype = torch.half, device = device), dst)
    assert_device_ok(device)
