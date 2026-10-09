"""
The tile order of the quantizer's output: the default sm80 order the ExLlamaV3 kernels decode, and colmajor, written
for engines that decode with Volta's mma.m8n8k4 (position p of a tile holds element k = p % 16, n = p // 16).

A Linear is converted in each order from a synthetic checkpoint; its packed tensors must reconstruct the quantizer's
weight_q through the kernels (plus the tile gather for colmajor), every forward route must agree with weight_q, and a
colmajor tensor reloaded through the real loader becomes an FP16 layer holding the decoded weight.
"""

import pytest
import torch

from exllamav3.modules.linear import Linear
from exllamav3.modules.quant import LinearEXL3, LinearFP16
from exllamav3.modules.quant.exl3 import colmajor_gather_index
from exllamav3.modules.quant.exl3_lib.quantize import tensor_core_perm, tensor_core_perm_i
from testlib.checkpoint import module_config
from testlib.compare import assert_close_mr
from testlib.exl3 import generator

K_IN, N_OUT = 512, 384
WEIGHT_STD = 0.02


@pytest.mark.nogpu
def test_tile_order_perms():
    # Both orders are permutations of the 256 tile elements
    for order in ("sm80", "colmajor"):
        perm = tensor_core_perm("cpu", order).long()
        assert torch.equal(perm.sort().values, torch.arange(256))
        assert torch.equal(perm[tensor_core_perm_i("cpu", order).long()], torch.arange(256))
    p = torch.arange(256)
    assert torch.equal(tensor_core_perm("cpu", "colmajor").long(), (p % 16) * 16 + p // 16)
    with pytest.raises(AssertionError):
        tensor_core_perm("cpu", "sm75")


@pytest.mark.nogpu
def test_colmajor_gather_index():
    # The kernels put position p's value at element perm80[p]; the gather must move every value to the element
    # the colmajor encoder took it from
    values = torch.randn(256, generator = generator(0))
    perm80 = tensor_core_perm("cpu", "sm80").long()
    permcol = tensor_core_perm("cpu", "colmajor").long()
    true_tile = torch.empty(256)
    true_tile[permcol] = values
    sm80_tile = torch.empty(256)
    sm80_tile[perm80] = values
    assert torch.equal(sm80_tile[colmajor_gather_index(torch.device("cpu"))], true_tile)


@pytest.fixture(scope = "module")
def checkpoint(tmp_path_factory):
    weight = (torch.randn((N_OUT, K_IN), generator = generator(0)) * WEIGHT_STD).half()
    return module_config({"proj.weight": weight}, tmp_path_factory.mktemp("tile_order"))


@pytest.mark.parametrize("tile_order", ["sm80", "colmajor"])
@pytest.mark.parametrize("K", [2, 3, 4])
@pytest.mark.parametrize("codebook", ["mcg", "mul1"])
@torch.inference_mode()
def test_tile_order_convert(device, checkpoint, tmp_path, tile_order, K, codebook):
    linear = Linear(checkpoint, "proj", K_IN, N_OUT, qmap = "proj")
    linear.load(device)
    capture = {}
    torch.manual_seed(0)
    linear.forward(torch.randn((1, 2048, K_IN), dtype = torch.float16, device = device), {"capture": capture})

    quant_args = {"K": K, "seed": 1, "apply_out_scales": None, "devices": [device], codebook: True}
    if tile_order != "sm80":
        quant_args["tile_order"] = tile_order
    _, weight_q = linear.convert_exl3(capture["proj"], quant_args, return_weight_q = True)
    weight_q = weight_q.half()
    assert isinstance(linear.inner, LinearEXL3) and linear.inner.colmajor == (tile_order == "colmajor")
    tensors = linear.get_tensors()
    assert ("proj.tile_order" in tensors) == (tile_order == "colmajor")

    # Packed tensors -> kernels (+ tile gather) -> the quantizer's weight
    assert_close_mr(weight_q, linear.inner.get_weight_tensor(), rtol = 1e-3, atol = 1e-3, mismatch_ratio = 0.001)

    # Every forward route (GEMV for few rows, reconstruct + GEMM for many) agrees with the quantized weight
    for rows in (1, 4, 300):
        x = torch.randn((rows, K_IN), dtype = torch.half, device = device)
        ref = (x.float() @ weight_q.float()).half()
        assert_close_mr(linear.forward(x, {}), ref, rtol = 2e-2, atol = 2e-2, mismatch_ratio = 0.001)
    linear.unload()

    # Through the real loader: an sm80 tensor stays EXL3, a colmajor one becomes FP16 with the decoded weight
    reloaded = Linear(module_config({k: v.cpu() for k, v in tensors.items()}, tmp_path), "proj", K_IN, N_OUT)
    reloaded.load(device)
    assert isinstance(reloaded.inner, LinearFP16 if tile_order == "colmajor" else LinearEXL3)
    assert_close_mr(weight_q, reloaded.inner.get_weight_tensor(), rtol = 1e-3, atol = 1e-3, mismatch_ratio = 0.001)
    reloaded.unload()
