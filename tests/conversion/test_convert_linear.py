"""
Linear.convert_exl3 on an unquantized (LinearFP16) layer, for every K and codebook: the proxy error and the
quantized weight's distance from the original stay within per-K bounds, and the packed tensors the layer is left
with (LinearEXL3) reconstruct the returned quantized weight.

The layers are synthetic: Gaussian weights at the scale of a trained LLM's projections, written to a safetensors
file and loaded through the real loader (testlib.checkpoint), in Llama-3.2-1B projection shapes. The Hessian is
captured by the layer's own forward from random activations.
"""

import pytest
import torch

from exllamav3.modules.linear import Linear
from exllamav3.modules.quant import LinearFP16
from testlib.checkpoint import module_config
from testlib.compare import assert_close_mr
from testlib.exl3 import generator

# (in_features, out_features) per projection
SHAPES = {
    "q_proj": (2048, 2048),
    "k_proj": (2048, 512),
    "v_proj": (2048, 512),
    "o_proj": (2048, 2048),
    "up_proj": (2048, 8192),
    "gate_proj": (2048, 8192),
    "down_proj": (8192, 2048),
}
WEIGHT_STD = 0.02

max_proxy_err_per_K = {
    1: 0.5,
    2: 0.1,
    3: 0.05,
    4: 0.01,
    5: 0.005,
    6: 0.005,
    7: 0.005,
    8: 0.005,
}

w_tol_per_K = {
    1: (0.5, 0.5),
    2: (0.1, 0.1),
    3: (0.08, 0.08),
    4: (0.06, 0.06),
    5: (0.04, 0.04),
    6: (0.03, 0.03),
    7: (0.02, 0.02),
    8: (0.02, 0.02),
}


@pytest.fixture(scope = "module")
def checkpoint(tmp_path_factory):
    """Config over one safetensors file holding every projection's fp16 weight, (out, in) as stored by HF"""
    gen = generator(0)
    tensors = {
        f"{name}.weight": (torch.randn((n, k), generator = gen) * WEIGHT_STD).half()
        for name, (k, n) in SHAPES.items()
    }
    return module_config(tensors, tmp_path_factory.mktemp("convert_linear"))


@pytest.mark.parametrize("codebook", ["3inst", "mcg", "mul1"])
@pytest.mark.parametrize("K", range(1, 9))
@pytest.mark.parametrize("name", SHAPES)
@torch.inference_mode()
def test_convert_linear(device, checkpoint, name, K, codebook):
    k, n = SHAPES[name]
    linear = Linear(checkpoint, name, k, n, qmap = name)
    linear.load(device)
    assert isinstance(linear.inner, LinearFP16)

    # Capture the Hessian by forwarding random activations through the layer
    capture = {}
    torch.manual_seed(0)
    state = torch.randn((1, 2048, k), dtype = torch.float16, device = device)
    linear.forward(state, {"capture": capture})

    # The layer is quantized in place
    weight_orig = linear.inner.get_weight_tensor().clone()
    # The quantizer selects the codebook by the presence of the "mcg" / "mul1" key, as convert_model.py sets it
    quant_args = {
        "K": K,
        "seed": 1,
        "apply_out_scales": None,
        "devices": [device],
    }
    if codebook != "3inst":
        quant_args[codebook] = True
    proxy_err, weight_q = linear.convert_exl3(capture[name], quant_args, return_weight_q = True)
    weight_q = weight_q.half()

    assert proxy_err < max_proxy_err_per_K[K], f"proxy error {proxy_err:.5f}"

    # Distance from the original weight, allowing for 1% outliers
    rtol, atol = w_tol_per_K[K]
    assert_close_mr(weight_q, weight_orig, rtol = rtol, atol = atol, mismatch_ratio = 0.01)

    # Reconstruct from the encoded/packed tensors. Some tolerance is needed because the quantizer works in fp32
    # while reconstruction reverses the regularization in fp16
    weight_recons = linear.inner.get_weight_tensor()
    assert_close_mr(weight_q, weight_recons, rtol = 1e-3, atol = 1e-3, mismatch_ratio = 0.001)

    linear.unload()
