import os

import pytest
import torch

DFLASH2_HF_DIR = os.environ.get("EXL3_TEST_DFLASH2", "")

requires_dflash2_hf = pytest.mark.skipif(
    not DFLASH2_HF_DIR or not os.path.exists(os.path.join(DFLASH2_HF_DIR, "config.json")),
    reason = "DFlash2 draft checkpoint not available (set EXL3_TEST_DFLASH2 to its directory)",
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "CUDA device required",
)

requires_weights = pytest.mark.skipif(
    not os.path.isdir(DFLASH2_HF_DIR)
    or not any(f.endswith(".safetensors") for f in os.listdir(DFLASH2_HF_DIR)),
    reason = "DFlash2 safetensors not present (set EXL3_TEST_DFLASH2 to the checkpoint directory)",
)


@requires_cuda
@requires_dflash2_hf
@requires_weights
def test_load_real_dflash2_checkpoint():
    from exllamav3 import Config, Model
    cfg = Config.from_directory(DFLASH2_HF_DIR)
    assert cfg.arch_string == "DFlash2DraftModel"
    model = Model.from_config(cfg)
    model.load(progressbar = False)

    assert model.caps.get("dflash2_draft") is True
    assert model.caps.get("dflash_draft") is True
    assert model.caps.get("default_draft_size") == 7

    # every block must carry both convs with loaded tensors
    blocks = model.modules[model.first_block_idx : model.last_kv_module_idx + 1]
    assert len(blocks) == 5
    for b in blocks:
        assert b.attention_conv.base_kernel is not None
        assert b.attention_conv.base_kernel.shape == (2, 2, 5120)
        assert b.mlp_conv.base_kernel is not None
        assert b.attention_conv.kernel_projection.device is not None

    # selector codebooks loaded
    sel = model.candidate_selector
    assert sel.predecessor_codebook.shape == (248320, 256)
    assert sel.successor_codebook.shape == (248320, 256)

    # parameter count in the expected ~2B band
    n = sum(m.weights_numel() for m in model.modules)
    assert 1.8e9 < n < 2.4e9

    model.unload()


@requires_cuda
@requires_dflash2_hf
@requires_weights
def test_propose_shapes_with_stub_target_head():
    """
    Selector integration smoke: attach a stub target whose lm_head is an identity-ish
    projection, run propose() on fixed block states, and check output shapes and the
    greedy path degenerating to per-position argmax of the head logits when the
    codebooks are zeroed.
    """
    from exllamav3 import Config, Model
    cfg = Config.from_directory(DFLASH2_HF_DIR)
    model = Model.from_config(cfg)
    model.load(progressbar = False)

    # zero the codebooks and projection so edge scores vanish: the greedy walk must
    # then pick the head-logit argmax candidate at every position
    with torch.inference_mode():
        sel = model.candidate_selector
        sel.predecessor_codebook.zero_()
        sel.successor_codebook.zero_()

    class _StubTarget:
        class _Cfg:
            vocab_size = 248320
        config = _Cfg()
        loaded_tp = False
        logit_layer_idx = 0

        def __init__(self, vocab, hidden):
            self.vocab = vocab
            self.hidden = hidden
            self.modules = [_StubHead(vocab, hidden)]

    class _StubHead:
        # vocab == hidden for the shape trick below is too restrictive; instead map
        # hidden -> vocab with a fixed random projection and fp32 output
        def __init__(self, vocab, hidden):
            torch.manual_seed(0)
            self.w = torch.randn(vocab, hidden, dtype = torch.float16, device = "cuda") * 0.02
            self.device = self.w.device

        def prepare_for_device(self, x, params):
            return x

        def forward(self, x, params):
            return torch.nn.functional.linear(x.float(), self.w.float())

    model.attached_model = lambda: _StubTarget(248320, 5120)

    B, T, H = 1, 8, 5120
    torch.manual_seed(1)
    out_state = torch.randn(B, T, H, dtype = torch.float16, device = "cuda")
    anchor = torch.tensor([123], dtype = torch.long, device = "cuda")

    path, candidates, q = model.propose(out_state, anchor, {}, 0.0)
    assert path.shape == (B, T - 1)
    assert candidates.shape == (B, T - 1, cfg.selector_top_k)
    assert q is None

    # with zeroed codebooks, path must equal the argmax of the stub head over the
    # candidate set at each position
    head = model.attached_model().modules[0]
    logits = torch.nn.functional.linear(out_state[:, 1:].float(), head.w.float())
    for pos in range(T - 1):
        _, cand_pos = torch.topk(logits[:, pos], cfg.selector_top_k, dim = -1, sorted = False)
        # greedy walk with zero edge scores picks the highest-unary candidate
        best = cand_pos[0, torch.argmax(logits[0, pos, cand_pos[0]])]
        assert path[0, pos].item() == best.item()

    model.unload()
