"""
ext.exl3_mgemm on gfx11 through the RDNA3 split-k matmul (rocm/quant/exl3_rdna3.cu, return code 95): a shared-input
bundle of equal-width matrices (gate/up-like, outputs stacked in C) and a sliced bundle of different widths
(Q/K/V/G-like, equal-width column slices written in place into each source's output), each against
testlib.exl3.linear_ref per matrix, for fp16 and fp32 output and the rows-per-pass instances.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import codebook_flags, generator, linear_ref, rand_linear

pytestmark = [pytest.mark.rocm_only]

RDNA3_PATH = 95


@pytest.fixture(autouse = True)
def _gfx11(device):
    if not torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx11"):
        pytest.skip("RDNA3 (gfx11) only")


def _ptrs(values, device):
    return torch.tensor([v.data_ptr() if isinstance(v, torch.Tensor) else v for v in values], dtype = torch.long,
                        device = device)


def _close(got, ref, tag):
    ref = ref.float()
    rel = ((got.float() - ref).square().mean().sqrt() / ref.square().mean().sqrt()).item()
    assert rel < 2e-3, f"{tag}: relative RMS error {rel:.2e}"


@pytest.mark.parametrize("m", [1, 3, 8, 13, 16, 40])
@pytest.mark.parametrize("c_dtype", [torch.half, torch.float])
@pytest.mark.parametrize("K,codebook", [(4, "mul1"), (3, "mcg"), (2.5, "mul1")])
@torch.inference_mode()
def test_shared_input_bundle(device, m, c_dtype, K, codebook):
    k, n, E = 2048, 1024, 2
    gen = generator(m * 10 + int(K * 2))
    ws = [rand_linear(k, n, K, gen, device) for _ in range(E)]
    x = (torch.randn((m, k), generator = gen) * 0.5).half().to(device)
    mcg, mul1 = codebook_flags(codebook)
    C = torch.full((E, m, n), float("nan"), dtype = c_dtype, device = device)
    a_had = torch.empty((E, m, k), dtype = torch.half, device = device)
    path = ext.exl3_mgemm(
        x.view(1, m, k), _ptrs([w["trellis"] for w in ws], device), C, _ptrs([w["suh"] for w in ws], device), a_had,
        _ptrs([w["svh"] for w in ws], device), None, None, K, -1, mcg, mul1, -1, -1, 0,
        1, None, None, None, None, 0,
    )
    assert path == RDNA3_PATH, f"exl3_mgemm took path {path}"
    for e, w in enumerate(ws):
        _close(C[e], linear_ref(x, w["trellis"], w["suh"], w["svh"], K, codebook), f"entry {e}")


@pytest.mark.parametrize("m", [1, 4, 9, 16])
@pytest.mark.parametrize("widths", [(4096, 1024, 1024), (3072, 1024, 1024, 3072)])
@torch.inference_mode()
def test_sliced_bundle(device, m, widths):
    k, K, W = 4096, 4, min(widths)
    gen = generator(m + len(widths))
    ws = [rand_linear(k, n, K, gen, device) for n in widths]
    x = (torch.randn((m, k), generator = gen) * 0.5).half().to(device)
    outs = [torch.full((m, n), float("nan"), dtype = torch.half, device = device) for n in widths]
    tr, sv, cp, src, stride = [], [], [], [], []
    for i, (w, n) in enumerate(zip(ws, widths)):
        for n0 in range(0, n, W):
            tr.append(w["trellis"].data_ptr() + (n0 // 16) * 16 * K * 2)
            sv.append(w["svh"].data_ptr() + n0 * 2)
            cp.append(outs[i].data_ptr() + n0 * 2)
            src.append(i)
            stride.append(n)
    S = len(tr)
    i32 = lambda v: torch.tensor(v, dtype = torch.int32, device = device)
    a_had = torch.empty((len(ws), m, k), dtype = torch.half, device = device)
    carrier = torch.empty((S, 1, W), dtype = torch.half, device = device).expand(S, m, W)
    path = ext.exl3_mgemm(
        x.view(1, m, k), _ptrs(tr, device), carrier, _ptrs([w["suh"] for w in ws], device), a_had, _ptrs(sv, device),
        None, None, K, -1, False, True, -1, -1, 0,
        1, i32([W] * S), _ptrs(cp, device), i32(stride), i32(src), len(ws),
    )
    assert path == RDNA3_PATH, f"exl3_mgemm took path {path}"
    for i, (o, w) in enumerate(zip(outs, ws)):
        _close(o, linear_ref(x, w["trellis"], w["suh"], w["svh"], K, "mul1"), f"source {i}")
