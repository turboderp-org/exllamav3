import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3 import Config, Model

torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 200)

test_model = "/mnt/str/eval_models/llama3.1-8b-instruct/exl3/3.0bpw/"

test_keys = [
    ("model.layers.0.self_attn.q_proj", "model.layers.0.input_layernorm"),
    ("model.layers.0.self_attn.k_proj", "model.layers.0.input_layernorm"),
    ("model.layers.0.self_attn.v_proj", "model.layers.0.input_layernorm"),
    ("model.layers.0.self_attn.o_proj", "model.layers.0.input_layernorm"),
    ("model.layers.0.mlp.up_proj", "model.layers.0.post_attention_layernorm"),
    ("model.layers.0.mlp.gate_proj", "model.layers.0.post_attention_layernorm"),
    ("model.layers.0.mlp.down_proj", None),
    ("lm_head", "model.norm"),
]

devices = [
    "cuda:2"
]

batch_sizes = [1, 2, 8, 16, 17, 31, 32, 33, 256, 2048]

config = Config.from_directory(test_model)
model = Model.from_config(config)

@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("test_key", test_keys)
@pytest.mark.parametrize("batch_size", batch_sizes)
@torch.inference_mode()
def test_qgemm(device, test_key, batch_size):

    if test_key[1]:
        norm = model.find_module(test_key[1])
        norm.load(device = device)

    linear = model.find_module(test_key[0])
    linear.load(device = device)
    
    torch.manual_seed(0)
    x = torch.randn((1, batch_size, linear.in_features), dtype = torch.float16, device = device)

    if test_key[1]:
        x = norm.forward(x, {})
    
    x_qgemm = linear.forward(x, {"reconstruct": False})
    x_hgemm = linear.forward(x, {"reconstruct": True})
    tol = 0.05
    torch.testing.assert_close(x_qgemm, x_hgemm, rtol = tol, atol = tol)

    linear.unload()

    if test_key[1]:
        norm.unload()