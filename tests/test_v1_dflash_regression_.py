"""
v1 regression pins: the shared-file touches for DFlash2 (parent hook points,
block_cls line, input-layer scale) must leave v1 behavior bit-identical.
Runs in the default suite (CPU only, synth fixture where needed).
"""
import os

import pytest
import torch

SYNTH_DIR = os.path.join(os.path.dirname(__file__), "data", "dflash2_synth")


def _v1_config_dir(tmp_path, name):
    import json
    import shutil

    d = tmp_path / name
    shutil.copytree(SYNTH_DIR, d)
    with open(d / "config.json", encoding = "utf8") as f:
        c = json.load(f)
    c["architectures"] = ["DFlashDraftModel"]
    with open(d / "config.json", "w", encoding = "utf8") as f:
        json.dump(c, f)
    return str(d)


def test_parent_hooks_are_identity_by_default():
    # The four TransformerBlock wrap points must return their inputs
    # untouched, so v1 blocks (which never override them) compute exactly
    # what they did before the hooks existed.
    from exllamav3.modules.transformer import TransformerBlock
    y = torch.randn(2, 5, 32)
    params = {}
    y2, ctx = TransformerBlock._pre_attn(object(), y, params)
    assert y2 is y and ctx is None
    assert TransformerBlock._post_attn(object(), y, "ctx", params) is y
    y3, ctx3 = TransformerBlock._pre_mlp(object(), y, params)
    assert y3 is y and ctx3 is None
    assert TransformerBlock._post_mlp(object(), y, "ctx", params) is y


def test_v1_config_has_no_embedding_scale(tmp_path):
    # The input-layer scale reads input_embedding_scale with a 1.0 default;
    # v1 configs must not define it, which is what makes the v1 path a no-op
    # without a clone.
    from exllamav3 import Config
    cfg = Config.from_directory(_v1_config_dir(tmp_path, "v1cfg"))
    assert cfg.arch_string == "DFlashDraftModel"
    assert not hasattr(cfg, "input_embedding_scale")


def test_v1_model_loads_synth_weights_with_plain_blocks(tmp_path):
    # End-to-end v1 construction + load on the synth fixture: blocks must be
    # plain TransformerBlocks (block_cls default), and every weight must load
    # (proves the loader walk is intact for v1).
    from exllamav3 import Config, Model
    from exllamav3.modules.transformer import TransformerBlock
    cfg = Config.from_directory(_v1_config_dir(tmp_path, "v1load"))
    model = Model.from_config(cfg)
    try:
        model.load(device = torch.device("cpu"), progressbar = False)
        blocks = model.modules[model.first_block_idx : model.last_kv_module_idx + 1]
        assert len(blocks) == cfg.num_hidden_layers
        for b in blocks:
            assert type(b) is TransformerBlock
            assert not hasattr(b, "attention_conv")
        assert not hasattr(model, "candidate_selector")
    finally:
        model.unload()
