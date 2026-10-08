"""
BC_BlockSparseMLP's bsz-1..N CUDA graph embeds the shared expert's BC_GatedMLP. That inner module records different
patch sites depending on how it was built: one fused gate+up mgemm, or two separate GEMVs (the configuration
use_mgemm() picks for wide mul1 tensors). The outer launcher must patch the shared expert's input through the
matching site type, otherwise graph replay throws "Graph update failed".

Runs over every registry model tagged "shared_experts", once with the default (fused) build and once with the
separate-GEMV build forced for every GatedMLP (use_mgemm patched to False). Decode logits are checked against the
plain no-cache forward, teacher-forced on the same tokens (testlib.graph.assert_decode_matches_forward).
"""

import pytest

from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

from testlib.graph import assert_decode_matches_forward, generate_with_logits, load_with_caches, unload_model

NEW_TOKENS = 12
PROMPT = "The quick brown fox jumps over the lazy dog because"


@pytest.fixture
def loaded(model_dir, device, mode):
    def hook(config):
        if mode == "separate":
            config.infer_params.use_mgemm = lambda *a, **k: False
    _, model, (cache,), tokenizer = load_with_caches(model_dir, device, config_hook = hook)
    yield model, cache, tokenizer
    unload_model(model)


@pytest.mark.models("shared_experts")
@pytest.mark.parametrize("mode", ["fused", "separate"])
def test_shared_expert_graph(model_id, mode, loaded):
    model, cache, tok = loaded
    bc_layers = [m for m in model if isinstance(m, BlockSparseMLP) and getattr(m, "bc_sh_exp", False)]
    assert bc_layers, "model has no shared experts on the BC graph path; test is vacuous"
    fused = [m.shared_experts.multi_gu[0] is not None for m in bc_layers]
    assert all(f == (mode == "fused") for f in fused), f"shared expert build mode mismatch: fused={fused[:4]}"

    ((tokens, logits),) = generate_with_logits(model, cache, tok, [PROMPT], NEW_TOKENS, max_batch_size = 1)
    # the job may stop early on EOS
    assert_decode_matches_forward(model, tok.encode(PROMPT, add_bos = True), tokens, logits, 0.05,
                                  f"{model_id} [{mode}]", min_steps = 4)
