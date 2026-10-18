"""
Qwen3.8-Flash-Next QSA indexer (QSAIndexer) against the HF reference (Qwen4ExpTextQSAIndexer) with real weights from
the unquantized 4-layer stub (registry role qwen4-exp-stub-hf; layer 3, the full-attention layer). The HF loop is
O(seq^2) per query, so the test uses a reduced token budget on a moderate sequence to make the top-k selection
binding. Both sides run fp16 (fp32 scoring, as in the references); the selection masks must match exactly. Also:
the incremental form (past raw keys) reproduces the full mask, and with the real budget everything visible is
selected at this length.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.hf, pytest.mark.model("qwen4-exp-stub-hf")]

KEY = "model.language_model.layers.3.self_attn.indexer"
BUDGET = 64         # block_topk 16: selection is non-trivial at the test length
B, S = 2, 220
SPLIT = 137


@pytest.fixture(scope = "module")
def setup(model_registry, device):
    import json
    import os
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding
    from exllamav3 import Config
    from exllamav3.loader.safetensors import SafetensorsCollection
    from exllamav3.util.rope import RoPE
    stub = model_registry.get("qwen4-exp-stub-hf").path
    cfg = AutoConfig.from_pretrained(stub).text_config
    with open(os.path.join(stub, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]

    def gt(key):
        with safe_open(os.path.join(stub, wm[key]), framework = "pt") as f:
            return f.get_tensor(key)

    # shared inputs and rope tables (from the HF rotary, so table construction is out of scope here)
    hidden = (torch.randn(B, S, cfg.hidden_size, generator = torch.Generator().manual_seed(0)) * 0.3).half().to(device)
    position_ids = torch.arange(S).view(1, 1, -1).expand(3, B, -1)
    cos, sin = Qwen4ExpTextRotaryEmbedding(cfg)(hidden.float().cpu(), position_ids)
    causal = torch.tril(torch.ones(S, S, dtype = torch.bool, device = device)).view(1, 1, S, S).expand(B, 1, -1, -1)
    stc = SafetensorsCollection(stub)
    yield SimpleNamespace(cfg = cfg, gt = gt, hidden = hidden, cos = cos.to(device, torch.float16),
                          sin = sin.to(device, torch.float16), causal = causal, stc = stc,
                          rope = RoPE(device, Config.from_directory(stub).rope_settings))
    stc.close()


def _ex(setup, budget, device):
    from exllamav3.modules.qsa_indexer import QSAIndexer
    cfg = setup.cfg
    ex = QSAIndexer(
        config = SimpleNamespace(stc = setup.stc), key = KEY,
        hidden_size = cfg.hidden_size, n_heads = cfg.indexer_n_heads, kv_heads = cfg.indexer_kv_heads,
        head_dim = cfg.indexer_head_dim, token_budget = budget,
        compress_ratio = cfg.indexer_compress_ratio, rms_norm_eps = cfg.rms_norm_eps,
    )
    ex.load(device)
    return ex


@pytest.fixture(scope = "module")
def ex_budget(setup, device):
    return _ex(setup, BUDGET, device)


@torch.inference_mode()
def test_mask_vs_hf(setup, ex_budget, device):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextQSAIndexer
    cfg = type(setup.cfg).from_dict(setup.cfg.to_dict())
    cfg.indexer_budget = BUDGET
    hf = Qwen4ExpTextQSAIndexer(cfg, layer_idx = 3).half().to(device)
    hf.load_state_dict({n: setup.gt(f"{KEY}.{n}").half()
                        for n in ("index_qk_proj.weight", "q_layernorm.weight", "k_layernorm.weight")})
    hf_mask = hf(setup.hidden, (setup.cos, setup.sin), setup.causal, past_key_values = None)
    hf_mask = (hf_mask & setup.causal).squeeze(1)                    # (B, S, S), incl. causality
    ex_mask, _ = ex_budget.build_mask(setup.hidden, None, setup.rope, {})
    mismatch = (hf_mask != ex_mask).sum().item()
    assert mismatch == 0, f"selection mask: {mismatch} mismatched entries of {hf_mask.numel()}"


@torch.inference_mode()
def test_incremental(setup, ex_budget):
    full, raw_k = ex_budget.build_mask(setup.hidden, None, setup.rope, {})
    _, raw_k1 = ex_budget.build_mask(setup.hidden[:, :SPLIT], None, setup.rope, {})
    mask2, raw_k2 = ex_budget.build_mask(setup.hidden[:, SPLIT:], raw_k1, setup.rope, {})
    # raw keys may differ by one fp16 ulp (shape-dependent GEMM reduction kernels at different lengths)
    assert torch.allclose(raw_k2.float(), raw_k.float(), atol = 2e-3), "raw key cache mismatch"
    assert torch.equal(mask2, full[:, SPLIT:]), "incremental mask mismatch"


@torch.inference_mode()
def test_full_budget_is_causal(setup, device):
    ex = _ex(setup, setup.cfg.indexer_budget, device)
    full_mask, _ = ex.build_mask(setup.hidden, None, setup.rope, {})
    assert torch.equal(full_mask, setup.causal.squeeze(1)), "full-budget mask should equal the causal mask"
