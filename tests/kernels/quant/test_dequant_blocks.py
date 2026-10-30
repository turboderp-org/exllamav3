"""
ext.dequant_blocks (exllamav3_ext/dequant_blocks.cu): natively quantized checkpoint bytes to fp16, written transposed
into a LinearFP16 (in, out) buffer, against the torch arithmetic of the eager loaders (PackedSegment.dequant_torch,
which mirrors Linear.apply_fp8_scales_ / apply_fp8_scales_inv_ / dequant_e8m0_blocks_ / _mxfp4_dequant):

- formats: fp8 e4m3fn (incl. subnormals and the 0x7f/0xff NaNs), fp8 e5m2 (incl. infinities), packed e2m1 nibbles
  (low nibble first)
- scale grids: E8M0 exponent bytes, float32 multipliers, or none; whole blocks and partial last blocks (ceil
  indexing, as the eager repeat-and-crop)
- a global multiplier applied after the fp16 rounding (the loader's weight_scale), an output column offset for
  concatenated parts, and an (in, out)-oriented fp8 source
- argument checks: output too small, fp4 with a transposed source, fp4 blocks not covering whole bytes
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.fp16 import PackedSegment, FMT_E4M3, FMT_E2M1, FMT_E5M2

pytestmark = pytest.mark.cuda_only

FMTS = {"e4m3": FMT_E4M3, "e2m1": FMT_E2M1, "e5m2": FMT_E5M2}


def random_packed(fmt: int, rows: int, cols: int, device, src_transposed: bool = False) -> torch.Tensor:
    """Random bytes in the checkpoint layout (every bit pattern, so the special encodings occur)"""
    gen = torch.Generator(device = "cpu").manual_seed(rows * 7919 + cols)
    if fmt == FMT_E2M1:
        shape = (rows, cols // 2)
    elif src_transposed:
        shape = (cols, rows)
    else:
        shape = (rows, cols)
    return torch.randint(0, 256, shape, dtype = torch.uint8, generator = gen).to(device)


def random_scale(kind: str, sr: int, sc: int, device):
    gen = torch.Generator(device = "cpu").manual_seed(sr * 31 + sc)
    if kind == "none":
        return None
    if kind == "e8m0":
        return torch.randint(100, 140, (sr, sc), dtype = torch.uint8, generator = gen).to(device)
    return (torch.rand((sr, sc), generator = gen) * 2 + 0.01).to(device)


def check(seg: PackedSegment, out_pad_rows: int, out_pad_cols: int, global_scale: float, device):
    ref = seg.dequant_torch(global_scale).T                    # (in, out) part
    out = torch.full((out_pad_rows, out_pad_cols), 7.0, dtype = torch.half, device = device)
    ext.dequant_blocks(seg.packed, seg.scale, out, seg.fmt, global_scale, seg.col0, seg.src_transposed, seg.br, seg.bc)
    got = out[:ref.shape[0], seg.col0:seg.col0 + ref.shape[1]]
    # NaN patterns compare by bit; everything else must be bit-identical as well
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), \
        f"max |diff| {(got.float() - ref.float()).abs().nan_to_num(0).max().item()}"
    # Outside the segment's rectangle the buffer is untouched
    mask = torch.ones_like(out, dtype = torch.bool)
    mask[:ref.shape[0], seg.col0:seg.col0 + ref.shape[1]] = False
    assert torch.all(out[mask] == 7.0)


@pytest.mark.parametrize("fmt", list(FMTS))
@pytest.mark.parametrize("scale_kind", ["none", "e8m0", "f32"])
@pytest.mark.parametrize("rows,cols,sr,sc,block", [
    (256, 512, 2, 4, None),          # whole blocks (128 x 128), derived from the grid
    (96, 160, 3, 5, None),           # 32 x 32 blocks over a tile-unaligned tensor
    (320, 200, 3, 2, (128, 128)),    # fixed 128-blocks with partial last blocks (320 = 2.5 blocks: ceil(320 / 3) would be wrong)
    (200, 320, 2, 3, (128, 128)),    # partial blocks on both sides
    (64, 2048, 64, 64, (1, 32)),     # per-row x 32-wide (MXFP4 geometry)
    (130, 70, 1, 1, None),           # per-tensor scale, odd tile coverage
])
def test_formats_and_grids(device, fmt, scale_kind, rows, cols, sr, sc, block):
    fmt = FMTS[fmt]
    if fmt == FMT_E2M1 and scale_kind != "none" and (block[1] if block else cols // sc) % 2:
        pytest.skip("fp4 blocks must cover whole bytes")
    packed = random_packed(fmt, rows, cols, device)
    seg = PackedSegment(packed, random_scale(scale_kind, sr, sc, device), fmt, block = block if scale_kind != "none" else None)
    check(seg, cols + 16, rows + 32, 1.0, device)


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
def test_transposed_source(device, fmt):
    fmt = FMTS[fmt]
    packed = random_packed(fmt, 192, 256, device, src_transposed = True)
    seg = PackedSegment(packed, random_scale("f32", 192, 1, device), fmt, src_transposed = True)
    assert (seg.rows, seg.cols) == (192, 256)
    check(seg, 256, 192, 1.0, device)


@pytest.mark.parametrize("global_scale", [0.5, 1.7, 3.0e-3])
def test_global_scale_after_rounding(device, global_scale):
    packed = random_packed(FMT_E4M3, 128, 256, device)
    seg = PackedSegment(packed, random_scale("f32", 1, 2, device), FMT_E4M3)
    check(seg, 256, 128, global_scale, device)


def test_column_offset_parts(device):
    """Two parts concatenated along the output dimension land side by side (Ling-style fused parts)"""
    a = PackedSegment(random_packed(FMT_E4M3, 128, 256, device), random_scale("f32", 1, 2, device), FMT_E4M3, col0 = 0)
    b = PackedSegment(random_packed(FMT_E5M2, 64, 256, device), random_scale("e8m0", 1, 2, device), FMT_E5M2, col0 = 128)
    out = torch.zeros((256, 192), dtype = torch.half, device = device)
    for seg in (a, b):
        ext.dequant_blocks(seg.packed, seg.scale, out, seg.fmt, 1.0, seg.col0, seg.src_transposed, seg.br, seg.bc)
    ref = torch.cat((a.dequant_torch(1.0).T, b.dequant_torch(1.0).T), dim = 1)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


def test_argument_checks(device):
    packed = random_packed(FMT_E4M3, 128, 256, device)
    small = torch.zeros((256, 100), dtype = torch.half, device = device)
    out = torch.zeros((256, 128), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError):
        ext.dequant_blocks(packed, None, small, FMT_E4M3, 1.0, 0, False, 0, 0)
    with pytest.raises(RuntimeError):
        ext.dequant_blocks(random_packed(FMT_E2M1, 128, 256, device), None, out, FMT_E2M1, 1.0, 0, True, 0, 0)
    with pytest.raises(RuntimeError):
        # 37-wide fp4 blocks split nibble pairs
        ext.dequant_blocks(random_packed(FMT_E2M1, 128, 256, device), random_scale("e8m0", 1, 7, device), out, FMT_E2M1, 1.0, 0, False, 1, 37)
    with pytest.raises(RuntimeError):
        # 320 rows over 3 scale rows without an explicit block size: not an exact division
        ext.dequant_blocks(random_packed(FMT_E4M3, 320, 256, device), random_scale("f32", 3, 1, device), out.new_zeros((256, 320)), FMT_E4M3, 1.0, 0, False, 0, 0)
    with pytest.raises(RuntimeError):
        # explicit blocks too small for the weight
        ext.dequant_blocks(random_packed(FMT_E4M3, 320, 256, device), random_scale("f32", 2, 1, device), out.new_zeros((256, 320)), FMT_E4M3, 1.0, 0, False, 128, 256)
