import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.arch_specific.olala_bf16 import _group_gemm, _row_gemv, codebook
from exllamav3.util.backend import gfx_arch
from testlib.exl3 import generator, rand_trellis
from testlib.moe import ptr_table


@pytest.mark.rocm_only
@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
def test_packed_bf16_expert_products(device, bits):
    if gfx_arch(device) != "gfx1100":
        pytest.skip("Olala BF16 kernels dispatch on gfx1100")
    k, n = 384, 128
    gen = generator(bits)
    packed = [rand_trellis(k, n, bits, gen, device) for _ in range(2)]
    pointers = ptr_table(packed, device)
    table = codebook(device)
    # Include activations beyond the FP16 range, two expert IDs and a partial row tile.
    x = (torch.randn((3, k), generator = gen) * 2**17).to(device, torch.bfloat16)
    ids = torch.tensor([1, 0, 1], device = device)
    reference = torch.empty((3, n), device = device)
    for expert, trellis in enumerate(packed):
        w = torch.empty((k, n), dtype = torch.half, device = device)
        ext.reconstruct(w, trellis, bits, False, True)
        reference[ids == expert] = x[ids == expert].float() @ w.bfloat16().float()

    result = torch.empty_like(reference)
    _row_gemv[(n // 16, 3)](x, pointers, table, ids, result, k, n, bits, 32)
    torch.testing.assert_close(result, reference, atol = 2.0, rtol = 1e-5)
    ends = torch.tensor([1, 3], device = device)
    tiles = torch.tensor([1, 2], device = device)
    result.fill_(float("nan"))
    _group_gemm[(n // 32, 2)](x, pointers, table, ids.argsort(), ends, tiles, result, k, n, bits, 2, 32)
    torch.testing.assert_close(result, reference, atol = 2.0, rtol = 1e-5)
