"""
Qwen3.8-Flash-Next hashed n-gram embedding (NGramEmbedding) against the HF reference module
(Qwen4ExpTextNGramEmbedding) on the unquantized 4-layer stub (registry role qwen4-exp-stub-hf), plus the stub's
K=3 trellis table (registry entry qwen4-exp-stub-ngram-k3, a directory holding ngram_k3.safetensors):

  - the checkpoint's hashing buffers equal what HF derives from the config (validates the stub builder's prime /
    offset recomputation and the multiplier copy)
  - all four load modes: fp16 RAM / disk must match HF bitwise; trellis RAM / disk must match each other bitwise and
    the source at the K=3 quantization error
  - EOS segmentation: n-grams never span an EOS, so moving one changes exactly the positions it reaches
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from testlib.parity import rfn

pytestmark = [pytest.mark.hf, pytest.mark.model("qwen4-exp-stub-hf")]

PREFIX = "model.language_model.layers.1.ple.ple_embedding"
EOS = 248044
B, S = 3, 57


@pytest.fixture(scope = "module")
def setup(model_registry):
    from safetensors import safe_open
    from transformers import AutoConfig
    stub = model_registry.get("qwen4-exp-stub-hf").path
    cfg = AutoConfig.from_pretrained(stub).text_config
    with open(os.path.join(stub, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]

    def get_tensor(key):
        with safe_open(os.path.join(stub, wm[key]), framework = "pt") as f:
            return f.get_tensor(key)

    # random ids with EOS boundaries sprinkled in
    input_ids = torch.randint(0, 248320, (B, S), generator = torch.Generator().manual_seed(0))
    input_ids[0, 20] = EOS
    input_ids[1, 0] = EOS
    input_ids[2, 30:33] = EOS
    return SimpleNamespace(stub = stub, cfg = cfg, get_tensor = get_tensor, input_ids = input_ids)


@pytest.fixture(scope = "module")
def trellis_dir(model_registry):
    path = model_registry.get("qwen4-exp-stub-ngram-k3").path
    if path is None or not os.path.isfile(os.path.join(path, "ngram_k3.safetensors")):
        pytest.skip(f"K=3 trellis n-gram table not available ({path})")
    return path


@pytest.fixture(scope = "module")
def hf_module(setup):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding
    cfg = setup.cfg
    return Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, layer_idx = 1, ple_layer_index = 0)


@pytest.fixture(scope = "module")
def reference(setup, hf_module):
    table = torch.cat([setup.get_tensor(f"{PREFIX}.ngram_embedding.shard_{s}.weight") for s in range(8)])
    mod = hf_module.to(torch.bfloat16)
    mod.ngram_embedding.weight.data = table
    with torch.inference_mode():
        return mod(setup.input_ids, past_key_values = None).float()


def _history(ids):
    # HF fills the previous context with EOS when there is no cache
    return torch.cat([torch.full((ids.shape[0], 2), EOS), ids], dim = 1)


@torch.inference_mode()
def _run(setup, directory, stream, device, ids = None):
    from exllamav3.loader.safetensors import SafetensorsCollection
    from exllamav3.modules import NGramEmbedding
    cfg = setup.cfg
    stc = SafetensorsCollection(directory)
    try:
        mod = NGramEmbedding(
            config = SimpleNamespace(stc = stc),
            key = f"{PREFIX}.ngram_embedding",
            ngram_size = cfg.ngram_size,
            heads_per_ngram = cfg.heads_per_ngram,
            ple_embed_dim = cfg.ple_embed_dim,
            eos_token_id = EOS,
            stream_from_disk = stream,
        )
        mod.load(device)
        out = mod.forward(_history(setup.input_ids if ids is None else ids), {}, out_dtype = torch.float).cpu()
        return mod.mode, out
    finally:
        stc.close()


@pytest.mark.parametrize("name", ["ngram_heads_offsets", "ngram_heads_vocab_sizes", "layer_multipliers"])
def test_hashing_buffers(setup, hf_module, name):
    assert torch.equal(getattr(hf_module, name), setup.get_tensor(f"{PREFIX}.{name}")), f"buffer mismatch: {name}"


def test_fp16_modes(setup, reference, device):
    mode_ram, ram = _run(setup, setup.stub, False, device)
    mode_disk, disk = _run(setup, setup.stub, True, device)
    assert (mode_ram, mode_disk) == ("fp16_ram", "fp16_disk")
    assert torch.equal(ram, disk), "fp16 RAM/disk mismatch"
    assert torch.equal(ram, reference), "fp16 path does not match the HF reference"


def test_trellis_modes(setup, reference, trellis_dir, device):
    mode_ram, ram = _run(setup, trellis_dir, False, device)
    mode_disk, disk = _run(setup, trellis_dir, True, device)
    assert (mode_ram, mode_disk) == ("trellis_ram", "trellis_disk")
    assert torch.equal(ram, disk), "trellis RAM/disk mismatch"
    err = rfn(ram, reference)
    assert 0.10 < err < 0.16, f"K=3 trellis rfn vs source {err:.5f}"


def test_eos_segmentation(setup, device):
    # Replacing the EOS at (0, 20): an n-gram must not span the boundary, so exactly positions 20-22 change
    _, base = _run(setup, setup.stub, False, device)
    ids2 = setup.input_ids.clone()
    ids2[0, 20] = 123
    _, out2 = _run(setup, setup.stub, False, device, ids2)
    diff_pos = (out2[0] != base[0]).any(dim = -1).nonzero().flatten().tolist()
    assert diff_pos == [20, 21, 22], f"EOS-boundary influence should span positions 20-22, got {diff_pos}"
