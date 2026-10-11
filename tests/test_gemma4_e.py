"""E-series parity against Transformers, including shared-KV cached/chunked paths.

Run with: python -m pytest tests/test_gemma4_e.py -q
Requires Transformers with Gemma4 and a CUDA device.
"""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from exllamav3 import Config, Model, Cache
from exllamav3.conversion.preserved_state import save_preserved_state, load_preserved_state


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    torch.manual_seed(123)
    cfg = Gemma4TextConfig(
        vocab_size=256, hidden_size=256, intermediate_size=512,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, global_head_dim=64, num_kv_shared_layers=2,
        layer_types=["sliding_attention", "full_attention"] * 2,
        hidden_size_per_layer_input=128, vocab_size_per_layer_input=256,
        sliding_window=8, max_position_embeddings=1024,
        attention_k_eq_v=False, final_logit_softcapping=30.0,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0,
                               "partial_rotary_factor": 0.25},
        },
    )
    cfg._attn_implementation = "eager"
    hf = Gemma4ForCausalLM(cfg).eval().cuda()
    # Nontrivial layer scalars test that PLE is injected before scalar multiplication.
    for i, block in enumerate(hf.model.layers):
        block.layer_scalar.fill_(0.8 + i * 0.1)
    root = tmp_path_factory.mktemp("gemma4_e")
    full = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": cfg.to_dict(), "vision_soft_tokens_per_image": 1,
        "vision_config": {"hidden_size": 128, "intermediate_size": 256,
                          "num_hidden_layers": 1, "num_attention_heads": 2,
                          "head_dim": 64, "patch_size": 16, "pooling_kernel_size": 1,
                          "position_embedding_size": 64, "rms_norm_eps": 1e-6},
    }
    (root / "config.json").write_text(json.dumps(full))
    (root / "processor_config.json").write_text(json.dumps({"image_processor": {
        "do_convert_rgb": True, "do_rescale": True, "do_normalize": True,
        "image_mean": [0.5] * 3, "image_std": [0.5] * 3, "resample": 2,
        "rescale_factor": 1 / 255, "max_soft_tokens": 1, "patch_size": 16,
        "pooling_kernel_size": 1,
    }}))
    weights = {k.replace("model.", "model.language_model.", 1): v.detach().cpu().clone()
               for k, v in hf.state_dict().items() if k != "lm_head.weight"}
    save_file(weights, str(root / "model.safetensors"))
    ex = Model.from_config(Config.from_directory(str(root)))
    cache = Cache(ex, max_num_tokens=256)
    ex.load(device="cuda:0")
    yield hf, ex, cache
    ex.unload()


def check(got, want):
    got, want = got.float().cpu(), want.float().cpu()
    assert torch.isfinite(got).all()
    rel = ((got - want).square().mean() / want.square().mean()).sqrt().item()
    assert rel < 0.01, rel
    assert (got.argmax(-1) == want.argmax(-1)).float().mean() >= 0.95


@torch.inference_mode()
def test_full_and_chunked_shared_kv(pair):
    hf, ex, cache = pair
    ids = torch.tensor([[2, 14, 22, 73, 51, 7, 48, 32, 99, 16, 34, 61, 13, 44, 82, 39, 5]])
    ref = hf(ids.cuda(), use_cache=False).logits
    full = ex.forward(ids)
    check(full, ref)
    chunks = []
    for begin, end in [(0, 5), (5, 12), (12, 16), (16, 17)]:
        chunks.append(ex.forward(ids[:, begin:end], {
            "attn_mode": "flash_attn", "cache": cache,
            "past_len": begin, "batch_shape": (1, 256),
        }).cpu())
    check(torch.cat(chunks, dim=1), ref)


@torch.inference_mode()
def test_cache_only_prefill_skips_shared_tail(pair):
    hf, ex, cache = pair
    ids = torch.tensor([[2, 18, 92, 5, 6, 71, 3, 9, 19, 56, 102]])
    assert ex.last_kv_module_idx == ex.first_block_idx + 1
    assert len(ex.get_cache_layers()) == 2
    ex.prefill(ids[:, :-1], {
        "attn_mode": "flash_attn", "cache": cache,
        "past_len": 0, "batch_shape": (1, 256),
    })
    got = ex.forward(ids[:, -1:], {
        "attn_mode": "flash_attn", "cache": cache,
        "past_len": ids.shape[1] - 1, "batch_shape": (1, 256),
    })
    check(got, hf(ids.cuda(), use_cache=False).logits[:, -1:])


def test_calibration_side_inputs_checkpoint(tmp_path):
    rows = [{"gemma4_ple": torch.randn(1, 3, 4, 128),
             "gemma4_k_1": torch.randn(1, 3, 2, 64)}, {}]
    path = tmp_path / "preserved.safetensors"
    save_preserved_state(rows, path)
    restored = load_preserved_state(path, len(rows))
    assert restored[1] == {}
    for key in rows[0]:
        assert torch.equal(rows[0][key], restored[0][key])
    with pytest.raises(ValueError):
        load_preserved_state(path, 3)
