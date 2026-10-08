"""
Qwen3.8-Flash-Next PLE injection layer (PLELayer) against the HF reference (Qwen4ExpTextPLELayer) on the
unquantized 4-layer stub (registry role qwen4-exp-stub-hf): the full layer (n-gram embed -> key / value projections
-> per-stream signed-sqrt gate -> dilated depthwise conv) against fp32 HF with an fp16-HF control calibrating the
expected precision gap, an incremental (conv state + carried token context) consistency check, and the stub's K=3
trellis table (registry entry qwen4-exp-stub-ngram-k3) wired in through a stacked tensor collection.

forward_streams' fast path adds the layer's delta into the stream stack in place (and returns None for it), so the
delta is measured as the change of the streams.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from testlib.parity import control_check, Gate, rfn

pytestmark = [pytest.mark.hf, pytest.mark.model("qwen4-exp-stub-hf")]

KEY = "model.language_model.layers.1.ple"
EOS = 248044
B, S = 2, 41
T1 = 17         # incremental split point


@pytest.fixture(scope = "module")
def setup(model_registry, device):
    from safetensors import safe_open
    from transformers import AutoConfig
    stub = model_registry.get("qwen4-exp-stub-hf").path
    cfg = AutoConfig.from_pretrained(stub).text_config
    with open(os.path.join(stub, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]

    def gt(key):
        with safe_open(os.path.join(stub, wm[key]), framework = "pt") as f:
            return f.get_tensor(key)

    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, cfg.vocab_size, (B, S), generator = g)
    input_ids[0, 11] = EOS
    input_ids[1, 30] = EOS
    hidden = torch.randn(B, S, cfg.hc_count * cfg.hidden_size, generator = g) * 0.6
    history = torch.cat([torch.full((B, cfg.ngram_size - 1), EOS), input_ids], dim = 1).to(device)
    streams = hidden.view(B, S, cfg.hc_count, cfg.hidden_size).to(device).float().contiguous()
    return SimpleNamespace(stub = stub, cfg = cfg, gt = gt, input_ids = input_ids, hidden = hidden,
                           history = history, streams = streams)


def _hf(setup, dtype):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer
    hf = Qwen4ExpTextPLELayer(setup.cfg, layer_idx = 1, ple_layer_index = 0).to(dtype)
    sd = {n: setup.gt(f"{KEY}.{n}").to(dtype) for n in (
        "key_proj.weight", "value_proj.weight", "conv1d.weight",
        "norm_key.weight", "norm_query.weight", "norm_conv.weight")}
    sd["ple_embedding.ngram_embedding.weight"] = torch.cat(
        [setup.gt(f"{KEY}.ple_embedding.ngram_embedding.shard_{s}.weight") for s in range(8)]).to(dtype)
    for n in ("ngram_heads_offsets", "ngram_heads_vocab_sizes", "layer_multipliers"):
        sd[f"ple_embedding.{n}"] = setup.gt(f"{KEY}.ple_embedding.{n}")
    hf.load_state_dict(sd)
    return hf


@pytest.fixture(scope = "module")
def reference(setup, device):
    """(fp32 HF output on the CPU, fp16-HF control's rfn from it)"""
    with torch.inference_mode():
        ref = _hf(setup, torch.float32)(setup.hidden, setup.input_ids, past_key_values = None).float()
        hf16 = _hf(setup, torch.float16).to(device)
        ctrl = hf16(setup.hidden.half().to(device), setup.input_ids.to(device), past_key_values = None).float().cpu()
    del hf16
    torch.cuda.empty_cache()
    return ref, rfn(ctrl, ref)


def _make_ple(setup, device, trellis_dir = None):
    from exllamav3.loader.safetensors import SafetensorsCollection
    from exllamav3.modules import PLELayer
    cfg = setup.cfg
    stc = SafetensorsCollection(setup.stub)
    if trellis_dir is not None:
        stc.add_tensor_files(trellis_dir)
    ple = PLELayer(
        config = SimpleNamespace(stc = stc),
        key = KEY,
        layer_idx = -2,
        hidden_size = cfg.hidden_size,
        hc_mult = cfg.hc_count,
        ple_embed_dim = cfg.ple_embed_dim,
        ngram_size = cfg.ngram_size,
        heads_per_ngram = cfg.heads_per_ngram,
        eos_token_id = EOS,
        conv_kernel_size = cfg.ple_conv_kernel_size,
        rms_norm_eps = cfg.rms_norm_eps,
        stream_from_disk = True,
    )
    ple.load(device)
    return ple, stc


@torch.inference_mode()
def _delta(ple, streams, history, conv_state = None):
    """(delta the layer adds to the streams, conv column stream)"""
    s = streams.clone()
    delta, conv_stream = ple.forward_streams(s, history, {}, conv_state = conv_state)
    return (s - streams if delta is None else delta), conv_stream


@pytest.fixture(scope = "module")
def ple_fp16(setup, device):
    ple, stc = _make_ple(setup, device)
    yield ple
    ple.unload()
    stc.close()


def test_vs_hf(setup, reference, ple_fp16):
    ref, ctrl_err = reference
    assert ple_fp16.ple_embedding.mode == "fp16_disk"
    delta, _ = _delta(ple_fp16, setup.streams, setup.history)
    gate = Gate("PLE fp16 pipeline vs fp32 HF")
    control_check(gate, "rfn", rfn(delta.flatten(-2).cpu(), ref), ctrl_err)
    gate.assert_passes()


def test_incremental(setup, ple_fp16):
    # conv state + the caller-carried n-gram context must reproduce the full forward. The residual difference is 1-2
    # fp16 ulps spread uniformly over all positions (shape-dependent GEMM kernels at different lengths); a state bug
    # puts O(value) errors at the conv_state_len positions after the boundary
    cfg = setup.cfg
    full, _ = _delta(ple_fp16, setup.streams, setup.history)
    d1, cs1 = _delta(ple_fp16, setup.streams[:, :T1].contiguous(), setup.history[:, :cfg.ngram_size - 1 + T1])
    ids = setup.input_ids.to(setup.history.device)
    hist2 = torch.cat([ids[:, T1 - (cfg.ngram_size - 1):T1], ids[:, T1:]], dim = 1)
    d2, _ = _delta(ple_fp16, setup.streams[:, T1:].contiguous(), hist2,
                   conv_state = cs1[..., -ple_fp16.conv_state_len:])
    diff = (torch.cat([d1, d2], dim = 1) - full).abs()
    ierr = diff.max().item()
    assert ierr < 5e-4, f"incremental mismatch {ierr}"
    bnd = diff[:, T1:T1 + ple_fp16.conv_state_len + 1].max().item()
    assert bnd <= ierr + 1e-12, "boundary positions worse than global GEMM noise"


def test_trellis_table(setup, ple_fp16, model_registry, device):
    trellis_dir = model_registry.get("qwen4-exp-stub-ngram-k3").path
    if trellis_dir is None or not os.path.isfile(os.path.join(trellis_dir, "ngram_k3.safetensors")):
        pytest.skip(f"K=3 trellis n-gram table not available ({trellis_dir})")
    ple_q, stc_q = _make_ple(setup, device, trellis_dir)
    try:
        assert ple_q.ple_embedding.mode == "trellis_disk"
        delta_q, _ = _delta(ple_q, setup.streams, setup.history)
    finally:
        ple_q.unload()
        stc_q.close()
    delta, _ = _delta(ple_fp16, setup.streams, setup.history)
    qerr = rfn(delta_q, delta)
    assert qerr < 0.25, f"K=3 trellis table through the layer: output rfn {qerr:.4f} vs the fp16 table"
