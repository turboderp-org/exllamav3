"""
DeepSeek-V3 / MLAttention full-model parity against HF transformers' native DeepseekV3 on real weights (registry
role moonlight-hf: Moonlight-16B-A3B, unquantized). exllamav3 runs the cached path (paged MLA latent cache, fp16 or
8-bit, chunked prefill) and is judged against HF's own eager-vs-sdpa noise floor (testlib.parity.FLOOR_K) on the
logits. HF runs first and is freed before exllamav3 loads, so the two copies never coexist.

Moonlight ships a tiktoken vocab that exllamav3's Tokenizer does not read, and its bundled tokenizer / modeling code
does not import under transformers 5 (transformers has DeepseekV3 natively, so no remote code). The comparison is
over token ids: real text through tiktoken when it is importable (in-distribution activations), fixed random ids
otherwise (which the floor run then calibrates).
"""

import base64
import os

import pytest
import torch

from testlib.env import has_module
from testlib.parity import Gate, hf_reference, logits_floor_gate

pytestmark = [pytest.mark.hf, pytest.mark.slow, pytest.mark.model("moonlight-hf")]

TEXT = (
    "The Antikythera mechanism is an ancient Greek hand-powered orrery, described as the oldest "
    "known example of an analogue computer. It could be used to predict astronomical positions "
    "and eclipses decades in advance, and to track the four-year cycle of athletic games. "
    "It was recovered in 1901 from a shipwreck off the coast of the Greek island of Antikythera."
)
SEQ_LEN = 256
CHUNK = 256
MAX_TOKENS = 2048


@pytest.fixture(scope = "module")
def input_ids(model_registry):
    model_dir = model_registry.get("moonlight-hf").path
    if has_module("tiktoken") and os.path.exists(os.path.join(model_dir, "tiktoken.model")):
        import tiktoken
        ranks = {}
        with open(os.path.join(model_dir, "tiktoken.model"), "rb") as f:
            for line in f:
                if line.strip():
                    token, rank = line.split()
                    ranks[base64.b64decode(token)] = int(rank)
        enc = tiktoken.Encoding(
            name = "moonshot",
            pat_str = r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks = ranks, special_tokens = {},
        )
        ids = enc.encode(TEXT)
        while len(ids) < SEQ_LEN:
            ids = ids + ids
        return torch.tensor([ids[:SEQ_LEN]], dtype = torch.long)
    return torch.randint(0, 100000, (1, SEQ_LEN), generator = torch.Generator().manual_seed(1234))


@torch.inference_mode()
def exl3_cached_logits(model_dir, ids, device, cache_bits):
    from exllamav3 import Cache, Config, Model
    from exllamav3.cache import CacheLayer_quant
    from testlib.parity import free_cuda
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    ckw = dict(layer_type = CacheLayer_quant, k_bits = cache_bits, v_bits = cache_bits) if cache_bits else {}
    cache = Cache(model, max_num_tokens = MAX_TOKENS, **ckw)
    model.load(device)
    try:
        out = []
        for a in range(0, ids.shape[1], CHUNK):
            params = {"attn_mode": "flash_attn", "cache": cache, "past_len": a, "batch_shape": (1, MAX_TOKENS)}
            out.append(model.forward(ids[:, a:a + CHUNK].to(device), params)[0].float().cpu())
        return torch.cat(out, dim = 0)
    finally:
        model.unload()
        del cache
        free_cuda()


@pytest.mark.parametrize("cache_bits", [0, 8], ids = ["fp16_cache", "q8_cache"])
def test_logits(model_dir, input_ids, device, cache_bits):
    from transformers import AutoModelForCausalLM
    ref = hf_reference(AutoModelForCausalLM, model_dir, input_ids, device, dtype = torch.bfloat16, attn_impl = "eager")
    floor = hf_reference(AutoModelForCausalLM, model_dir, input_ids, device, dtype = torch.bfloat16, attn_impl = "sdpa")
    got = exl3_cached_logits(model_dir, input_ids, device, cache_bits)
    gate = Gate(f"DeepSeek-V3 (Moonlight) cached, {'Q%d' % cache_bits if cache_bits else 'fp16'} latent cache, "
                f"vs HF bf16 eager, floor HF sdpa")
    logits_floor_gate(gate, got, ref["logits"], floor["logits"])
    gate.assert_passes()
