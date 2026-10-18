"""
BC_BlockSparseMLP's bsz-1..N CUDA graph embeds the shared expert's BC_GatedMLP. That inner module records different
patch sites depending on how it was built: one fused gate+up mgemm, or two separate GEMVs (the configuration
use_mgemm() picks for wide mul1 tensors). The outer launcher must patch the shared expert's input through the
matching site type, otherwise graph replay throws "Graph update failed".

Runs over every registry model tagged "shared_experts", once with the default (fused) build and once with the
separate-GEMV build forced for every GatedMLP (use_mgemm patched to False). The generation runs in child processes
with graph capture enabled and disabled; the graphed decode logits must match the eager run
(testlib.graph.assert_matches_eager).
"""

import pytest

from testlib.env import get_test_device
from testlib.graph import assert_matches_eager, generate_with_logits, load_with_caches, run_with_and_without_graphs, \
    unload_model

NEW_TOKENS = 12
PROMPT = "The quick brown fox jumps over the lazy dog because"


def scenario(model_dir: str, mode: str) -> list:
    """One greedy generation (child process) with the shared experts built as `mode`; [(label, tokens, logits)]"""
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

    def hook(config):
        if mode == "separate":
            config.infer_params.use_mgemm = lambda *a, **k: False
    _, model, (cache,), tok = load_with_caches(model_dir, get_test_device(), config_hook = hook)
    bc_layers = [m for m in model if isinstance(m, BlockSparseMLP) and getattr(m, "bc_sh_exp", False)]
    assert bc_layers, "model has no shared experts on the BC graph path; test is vacuous"
    fused = [m.shared_experts.multi_gu[0] is not None for m in bc_layers]
    assert all(f == (mode == "fused") for f in fused), f"shared expert build mode mismatch: fused={fused[:4]}"
    ((tokens, logits),) = generate_with_logits(model, cache, tok, [PROMPT], NEW_TOKENS, max_batch_size = 1)
    unload_model(model)
    return [(mode, tokens, [l.cpu() for l in logits])]


@pytest.mark.models("shared_experts")
@pytest.mark.parametrize("mode", ["fused", "separate"])
def test_shared_expert_graph(model_id, model_dir, mode, device):
    graphed, eager = run_with_and_without_graphs(scenario, model_dir, mode, device = device)
    # the job may stop early on EOS
    assert_matches_eager([(f"{model_id} [{l}]", t, g) for l, t, g in graphed],
                         [(f"{model_id} [{l}]", t, g) for l, t, g in eager], min_steps = 4)
