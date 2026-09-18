"""
End-to-end DFlash2 tests on the synthetic fixture (tests/data/dflash2_synth):
config, CPU load, propose shapes, and draft_conf export. No Hub, no GPU, no
EXL3_TEST_DFLASH2 needed — this is the default-suite coverage.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

SYNTH_DIR = os.path.join(os.path.dirname(__file__), "data", "dflash2_synth")


@pytest.fixture(scope = "session", autouse = True)
def _ensure_synth_weights():
    # The fixture weights are gitignored and built on demand: the first test
    # needing them runs make_synth.py (already in tree, needs torch).
    st = Path(__file__).parent / "data" / "dflash2_synth" / "model.safetensors"
    if not st.exists():
        subprocess.check_call([sys.executable, "make_synth.py"], cwd = str(st.parent))


def test_synth_config_reads_dflash2_fields():
    from exllamav3.architecture.dflash2 import DFlash2Config
    cfg = DFlash2Config(SYNTH_DIR)
    assert cfg.arch_string == "DFlash2DraftModel"
    assert cfg.conv_kernel_size == 2
    assert cfg.conv_group_size == 8
    assert cfg.selector_rank == 16
    assert cfg.selector_top_k == 4
    assert cfg.block_size == 8
    assert cfg.target_layer_ids == [0, 1]
    assert cfg.mask_token_id == 7
    assert cfg.num_hidden_layers == 2
    assert cfg.hidden_size == 128
    assert cfg.vocab_size == 512
    # logit scalings default to identity when the checkpoint omits them
    assert cfg.output_multiplier == 1.0
    assert cfg.final_logit_softcapping == 0.0
    assert cfg.input_embedding_scale == 1.0


def _load_synth():
    from exllamav3 import Config, Model
    cfg = Config.from_directory(SYNTH_DIR)
    assert cfg.arch_string == "DFlash2DraftModel"
    model = Model.from_config(cfg)
    model.load(device = torch.device("cpu"), progressbar = False)
    return cfg, model


def test_synth_loads_on_cpu_with_convs_and_selector():
    cfg, model = _load_synth()
    try:
        assert model.caps.get("dflash2_draft") is True
        assert model.caps.get("dflash_draft") is True
        assert model.caps.get("default_draft_size") == cfg.block_size - 1
        blocks = model.modules[model.first_block_idx : model.last_kv_module_idx + 1]
        assert len(blocks) == cfg.num_hidden_layers
        for b in blocks:
            assert b.attention_conv.base_kernel.shape == (2, cfg.conv_kernel_size, cfg.hidden_size)
            assert b.mlp_conv.base_kernel.shape == (2, cfg.conv_kernel_size, cfg.hidden_size)
        sel = model.candidate_selector
        assert sel.predecessor_codebook.shape == (cfg.vocab_size, cfg.selector_rank)
        assert sel.successor_codebook.shape == (cfg.vocab_size, cfg.selector_rank)
    finally:
        model.unload()


class _StubHead:
    def __init__(self, vocab, hidden):
        torch.manual_seed(0)
        self.w = torch.randn(vocab, hidden, dtype = torch.float16) * 0.02

    def prepare_for_device(self, x, params):
        return x

    def forward(self, x, params):
        return F.linear(x.float(), self.w.float())


class _StubTarget:
    loaded_tp = False

    def __init__(self, cfg):
        self.config = cfg
        self.logit_layer_idx = 0
        self.modules = [_StubHead(cfg.vocab_size, cfg.hidden_size)]


def _propose(cfg, model, temperature = 0.0, export_conf = False):
    B, T, H = 1, cfg.block_size, cfg.hidden_size
    torch.manual_seed(1)
    out_state = torch.randn(B, T, H, dtype = torch.float16) * 0.1
    anchor = torch.tensor([3], dtype = torch.long)
    model.attached_model = lambda: _StubTarget(cfg)
    params = {"export_draft_conf": True} if export_conf else {}
    path, candidates, q = model.propose(out_state, anchor, params, temperature)
    return path, candidates, q, params, out_state


def test_synth_propose_shapes_greedy():
    cfg, model = _load_synth()
    try:
        path, candidates, q, _, _ = _propose(cfg, model)
        assert path.shape == (1, cfg.block_size - 1)
        assert candidates.shape == (1, cfg.block_size - 1, cfg.selector_top_k)
        assert q is None
        assert path.dtype == torch.long
        assert ((0 <= path) & (path < cfg.vocab_size)).all()
    finally:
        model.unload()


def test_synth_propose_sampled_returns_q_distribution():
    cfg, model = _load_synth()
    try:
        torch.manual_seed(7)
        path, candidates, q, _, _ = _propose(cfg, model, temperature = 0.8)
        assert q is not None
        assert q.shape == candidates.shape
        assert torch.allclose(q.sum(-1), torch.ones_like(q.sum(-1)), atol = 1e-5)
    finally:
        model.unload()


def test_synth_propose_exports_draft_conf():
    # draft_conf is [anchor_placeholder, per-position selector scores...]; the
    # generator's -dds/-dc crop reads [:, 1:].
    cfg, model = _load_synth()
    try:
        path, candidates, q, params, out_state = _propose(cfg, model, export_conf = True)
        conf = params.get("draft_conf")
        assert conf is not None
        assert conf.shape == (1, cfg.block_size)
        assert torch.equal(conf[:, :1], torch.zeros_like(conf[:, :1]))
        # the exported scores must match a direct selector run over the same states
        sel = model.candidate_selector
        head = model.attached_model().modules[0]
        hidden = out_state[:, 1:, :].contiguous()
        logits = head.forward(head.prepare_for_device(hidden, {}), {})
        _, _, _, conf_direct = sel.select(
            hidden, logits, torch.tensor([3]), 0.0, return_confidence = True)
        assert torch.allclose(conf[:, 1:].float(), conf_direct.float(), atol = 1e-4)
    finally:
        model.unload()


def test_synth_propose_no_conf_by_default():
    cfg, model = _load_synth()
    try:
        _, _, _, params, _ = _propose(cfg, model, export_conf = False)
        assert "draft_conf" not in params
    finally:
        model.unload()


def _synth_dir_with(tmp_path, name, **overrides):
    """Copy of the synth fixture with config.json keys overridden."""
    d = tmp_path / name
    shutil.copytree(SYNTH_DIR, d)
    with open(d / "config.json", encoding = "utf8") as f:
        c = json.load(f)
    c.update(overrides)
    with open(d / "config.json", "w", encoding = "utf8") as f:
        json.dump(c, f)
    return str(d)


def test_v1_construct_builds_no_selector(tmp_path):
    # The v1 DFlash path must not gain DFlash2 modules: same directory,
    # v1 arch string, no CandidateSelector and no dflash2_draft cap.
    from exllamav3 import Config, Model
    d = _synth_dir_with(tmp_path, "v1", architectures = ["DFlashDraftModel"])
    cfg = Config.from_directory(d)
    assert cfg.arch_string == "DFlashDraftModel"
    model = Model.from_config(cfg)
    assert type(model).__name__ == "DFlashModel"
    assert not hasattr(model, "candidate_selector")
    assert model.caps.get("dflash2_draft") is None
    assert model.caps.get("dflash_draft") is True


@pytest.mark.parametrize("overrides", [
    {"conv_kernel_size": 9},      # larger than block_size 8
    {"conv_group_size": 12},      # 128 % 12 != 0
    {"selector_rank": 0},
    {"selector_top_k": 513},      # larger than vocab_size 512
])
def test_invalid_dflash2_config_fails_fast(tmp_path, overrides):
    from exllamav3.architecture.dflash2 import DFlash2Config
    with pytest.raises(AssertionError):
        DFlash2Config(_synth_dir_with(tmp_path, "badcfg", **overrides))


def test_raw_fp16_convert_contract():
    # Convert gate contract, pinned without running convert: the selector
    # advertises retain_raw_fp16, carries no quantization role (empty qmaps),
    # and neither DFlash2 projection has a qmap, so nothing can ever enter
    # the EXL3 path while get_tensors() still flows through the gate.
    from exllamav3 import Config, Model
    cfg = Config.from_directory(SYNTH_DIR)
    model = Model.from_config(cfg)
    sel = model.candidate_selector
    assert sel.caps.get("retain_raw_fp16") is True
    assert sel.get_qmaps() == set()
    assert sel.hidden_projection.qmap is None
    blocks = model.modules[model.first_block_idx : model.last_kv_module_idx + 1]
    for b in blocks:
        assert b.attention_conv.kernel_projection.qmap is None
        assert b.mlp_conv.kernel_projection.qmap is None
    # the gate expression convert uses, evaluated on the unloaded modules:
    # only the selector passes via the flag, everything else via qmaps
    flagged = [m.key for m in model.modules
               if not m.get_qmaps() and m.caps.get("retain_raw_fp16", False)]
    assert flagged == ["candidate_selector"]


def test_convert_emission_covers_raw_tensors():
    # What convert collects per module (get_tensors on loaded modules) must
    # include every raw fp16 tensor: codebooks, hidden projection, both conv
    # base kernels and kernel projections per block. A full convert.py run on
    # this fixture confirmed the same keys land fp16 in the output with no
    # EXL3 tensors under the raw prefixes (see doc/dflash2.md).
    import torch

    from exllamav3 import Config, Model
    cfg = Config.from_directory(SYNTH_DIR)
    model = Model.from_config(cfg)
    model.load(device = torch.device("cpu"), progressbar = False)
    try:
        emitted = {}
        for m in model.modules:
            for sm in m:
                emitted.update(sm.get_tensors())
        want = ["candidate_selector.hidden_projection.weight",
                "candidate_selector.predecessor_codebook.weight",
                "candidate_selector.successor_codebook.weight"]
        for idx in range(cfg.num_hidden_layers):
            for conv in ("attention_conv", "mlp_conv"):
                want += [f"layers.{idx}.{conv}.base_kernel",
                         f"layers.{idx}.{conv}.kernel_projection.weight"]
        missing = [k for k in want if k not in emitted]
        assert not missing, f"convert would drop: {missing}"
        bad = [(k, str(emitted[k].dtype)) for k in want
               if emitted[k].dtype != torch.float16]
        assert not bad, f"non-fp16 raw tensors: {bad}"
    finally:
        model.unload()


def test_propose_rejects_tp_target():
    cfg, model = _load_synth()
    try:
        B, T, H = 1, cfg.block_size, cfg.hidden_size
        out_state = torch.randn(B, T, H, dtype = torch.float16) * 0.1
        anchor = torch.tensor([3], dtype = torch.long)
        target = _StubTarget(cfg)
        target.loaded_tp = True
        model.attached_model = lambda: target
        with pytest.raises(NotImplementedError, match = "tensor-parallel targets"):
            model.propose(out_state, anchor, {}, 0.0)
    finally:
        model.unload()


def test_propose_applies_output_multiplier():
    # Same block states, multiplier 1.0 vs 2.0: the scaled unary must move the
    # exported selector scores (greedy walk is deterministic, states reseeded).
    cfg, model = _load_synth()
    try:
        _, _, _, params_plain, _ = _propose(cfg, model, export_conf = True)
        model.config.output_multiplier = 2.0
        _, _, _, params_scaled, _ = _propose(cfg, model, export_conf = True)
        assert not torch.equal(params_plain["draft_conf"], params_scaled["draft_conf"])
    finally:
        model.config.output_multiplier = 1.0
        model.unload()


def test_propose_applies_logit_softcapping():
    # A tight softcap squashes the draft logits, which must move the exported
    # selector scores versus the uncapped run over identical states.
    cfg, model = _load_synth()
    try:
        _, _, _, params_plain, _ = _propose(cfg, model, export_conf = True)
        model.config.final_logit_softcapping = 0.5
        _, _, _, params_capped, _ = _propose(cfg, model, export_conf = True)
        assert not torch.equal(params_plain["draft_conf"], params_capped["draft_conf"])
    finally:
        model.config.final_logit_softcapping = 0.0
        model.unload()


class _StubEmbedding:
    """Deterministic embedding: token id broadcast over hidden."""

    def __init__(self, hidden):
        self.hidden = hidden

    def forward(self, ids, params):
        return ids.float().unsqueeze(-1).expand(-1, -1, self.hidden).contiguous()


def test_input_embedding_scale_applies_to_mask_only():
    # input_embedding_scale=2 must double the mask (noise) embedding columns
    # while the anchor column (a real token embedding) stays unscaled.
    from types import SimpleNamespace

    from exllamav3.architecture.dflash2 import DFlash2Config
    from exllamav3.modules.arch_specific.dflash import DFlashInputLayer
    cfg = DFlash2Config(SYNTH_DIR)
    layer = DFlashInputLayer(
        config = cfg,
        key = "fc",
        key_norm = "hidden_norm",
        hidden_size = 8,
        target_state_size = 16,
        mask_token_id = 7,
        rms_norm_eps = cfg.rms_norm_eps,
        native_draft_len = 8,
    )
    layer.attached_model = lambda: SimpleNamespace(
        loaded_tp = False, modules = [_StubEmbedding(8)])

    out_plain = layer.forward(torch.tensor([[3]]), {})
    assert out_plain.shape == (1, 8, 8)
    assert torch.equal(out_plain[:, 0], torch.full((1, 8), 3.0))
    assert torch.equal(out_plain[:, 1:], torch.full((1, 7, 8), 7.0))

    cfg.input_embedding_scale = 2.0
    try:
        out_scaled = layer.forward(torch.tensor([[3]]), {})
        assert torch.equal(out_scaled[:, 0], torch.full((1, 8), 3.0))
        assert torch.equal(out_scaled[:, 1:], torch.full((1, 7, 8), 14.0))
    finally:
        cfg.input_embedding_scale = 1.0
