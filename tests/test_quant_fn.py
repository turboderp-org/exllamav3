import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3 import Config, Model
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3_lib.quantize import quantize_tiles
from util import assert_close_mr
import torch.nn.functional as F
import torch.testing
import math

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

device = "cuda:2"
test_model = "/mnt/str/eval_models/llama3.1-8b-instruct/hf/"
test_keys = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.down_proj",
]

config = Config.from_directory(test_model)
model = Model.from_config(config)

max_mse_per_K = {
    1: 0.3,
    2: 0.1,
    3: 0.1,
    4: 0.1,
    5: 0.1,
    6: 0.07,
    7: 0.05,
    8: 0.04,
}

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


@pytest.mark.parametrize("batch_size", [1, 16, 17, 128])
@pytest.mark.parametrize("K", [1])
@torch.inference_mode()
def test_encode(batch_size, K):

    torch.manual_seed(0)
    scale = 1.491
    in_tile = torch.randn((batch_size, 256), device = device) * scale
    out_tile, out_idx = quantize_tiles(in_tile, {"K": K})

    # Test tail-biting
    first_col = out_idx[:, 0].to(torch.int32) & 0xFFFF
    last_col = out_idx[:, 255].to(torch.int32) & 0xFFFF
    first_col = first_col >> K
    last_col = last_col & ((1 << (16 - K)) - 1)
    assert torch.equal(first_col, last_col)

    # Test MSE
    mse = F.mse_loss(in_tile / scale, out_tile / scale).item()
    assert mse < max_mse_per_K[K]


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6, 7, 8])
@torch.inference_mode()
def test_encode_ideal(batch_size, K):

    # Create random, valid, tail-biting encoding
    torch.manual_seed(0)
    encoded = torch.randint(low = 0, high = 65535, size = (batch_size, 256), device = device)
    for i in range(256):
        x = encoded[:, i]
        x = x & ((1 << K) - 1)
        for shift in range(1, int(math.ceil(16 / K))):
            j = (i + 256 - shift) % 256
            y = encoded[:, j]
            y = y & ((1 << K) - 1)
            x = x | (y << (K * shift))
        encoded[:, i] = x & 0xffff
    encoded = encoded.to(torch.short)

    # Decode
    decoded = torch.empty_like(encoded, dtype = torch.float)
    ext.decode(encoded, decoded)

    # Should quantize with zero loss
    out_tile, out_idx = quantize_tiles(decoded, {"K": K})
    torch.testing.assert_close(out_tile, decoded, rtol = 1e-6, atol = 1e-6)


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("test_key", test_keys)
@torch.inference_mode()
def test_quant_dequant(K, test_key):

    # Grab unquantized linear layer from model
    linear = model.find_module(test_key)
    linear.load(device = device)

    # Forward some random data through the layer to capture Hessian
    bsz = 2048
    torch.manual_seed(0)
    state = torch.randn((1, bsz, linear.in_features), dtype = torch.float16, device = device)
    capture_H = {}
    params = {
        "attn_mode": "flash_attn_nc",
        "capture": capture_H
    }
    rs = linear.prepare_for_device(state, params)
    ref_out = linear.forward(rs, params)

    # Copy the original weight since layer will be quantized in-place
    weight_orig = linear.inner.get_weight_tensor().clone()

    # Quantize the layer
    quant_args = {
        "K": K,
        "seed": 1,
    }
    proxy_err, weight_q = linear.convert_exl3(capture_H[linear.qmap], quant_args, return_weight_q = True)
    weight_q = weight_q.half()

    # Test proxy_err
    assert proxy_err < max_proxy_err_per_K[K]

    # Test max absolute weight difference from original, allow for 1% outliers
    rtol, atol = w_tol_per_K[K]
    assert_close_mr(weight_q, weight_orig, rtol = rtol, atol = atol, mismatch_ratio = 0.01)

    # Reconstruct from encoded/packed tensors. Some tolerance needed because the quantizer works in float32
    # while reconstruction reverses the regularization in float16
    weight_recons = linear.inner.get_weight_tensor()
    assert_close_mr(weight_q, weight_recons, rtol = 1e-3, atol = 1e-3, mismatch_ratio = 0.001)

    # Cleanup
    linear.unload()