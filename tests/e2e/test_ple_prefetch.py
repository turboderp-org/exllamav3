"""
Generator-level check of the PLE n-gram prefetch (Qwen3.8-Flash-Next 4-layer quantized stub): with the job
staging the next prefill chunk ahead and the model staging the current one before its first layers, every
prefill-sized chunk must take a prefetched staging set (no misses beyond decode-sized forwards), and the n-gram
embeddings of every forward must be bit-identical to a run with prefetching disabled (the reference). Each run
is a separate process (the prompt cache would otherwise skip the second prefill).
"""

import os

import pytest

from testlib.isolated import device_env, run_isolated

pytestmark = [pytest.mark.model("qwen4-exp-stub"), pytest.mark.slow]

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNK = 512
PROMPT_TOKENS = 3000 + 137      # several chunks + a partial one
NEW_TOKENS = 12


def generate(model_dir: str) -> dict:
    """Greedy generation over a long prompt, recording the n-gram embedding forwards (child process)"""
    import argparse
    from exllamav3 import model_init, Generator, Job, ArgmaxSampler
    from exllamav3.modules import NGramEmbedding
    from exllamav3.modules.row_table import PREFETCH_MIN_TOKENS

    p = argparse.ArgumentParser()
    model_init.add_args(p)
    model, config, cache, tok = model_init.init(p.parse_args(["-m", model_dir, "-cs", "8192"]))[:4]
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = 1, max_chunk_size = CHUNK)
    ng = [m for m in model if isinstance(m, NGramEmbedding)]
    assert len(ng) == 1
    ng = ng[0]
    sizes, sums = [], []
    orig = ng.forward

    def logged(x, params, out_dtype = None):
        sizes.append(x.shape[0] * (x.shape[1] - ng.context_len))
        out = orig(x, params, out_dtype)
        sums.append(float(out.cpu().double().sum()))    # exact fingerprint of the staged rows
        return out

    ng.forward = logged
    ng.table.prefetch_stats = {"hit": 0, "miss": 0, "retired": 0}     # load-time forwards don't count

    with open(os.path.join(REPO_DIR, "README.md"), encoding = "utf8") as f:
        text = f.read()
    ids = tok.encode(text * 3, add_bos = True)[:, :PROMPT_TOKENS]
    assert ids.shape[1] == PROMPT_TOKENS
    out = []
    gen.enqueue(Job(input_ids = ids, max_new_tokens = NEW_TOKENS, sampler = ArgmaxSampler()))
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("token_ids") is not None:
                out += r["token_ids"].flatten().tolist()
    return {
        "tokens": out,
        "stats": dict(ng.prefetch_stats),
        "large": sum(1 for s in sizes if s >= PREFETCH_MIN_TOKENS),
        "small": sum(1 for s in sizes if s < PREFETCH_MIN_TOKENS),
        "sizes": sizes,
        "sums": sums,
    }


@pytest.fixture(scope = "module")
def runs(request, model_registry):
    model_dir = model_registry.get("qwen4-exp-stub").path
    device = request.config.getoption("--device")
    on = run_isolated(generate, model_dir, env = {**device_env(device), "EXL3_NGRAM_PREFETCH": "1"})
    off = run_isolated(generate, model_dir, env = {**device_env(device), "EXL3_NGRAM_PREFETCH": "0"})
    return on, off


def test_embeddings_identical_with_and_without_prefetch(runs):
    # The sampled tokens need not be identical past the first: the stub's MoE kernels are not deterministic
    # run to run
    a, b = runs
    assert a["sizes"] == b["sizes"]
    assert a["sums"] == b["sums"], [(x, y) for x, y in zip(a["sums"], b["sums"]) if x != y]
    assert len(a["tokens"]) == NEW_TOKENS and a["tokens"][0] == b["tokens"][0], (a["tokens"], b["tokens"])


def test_every_prefill_chunk_is_prefetched(runs):
    a, _ = runs
    assert a["large"] >= 5, a["sizes"]
    # every prefill-sized chunk was staged ahead (job prediction or the model's own hook), nothing else was
    assert a["stats"]["hit"] == a["large"], a
    assert a["stats"]["miss"] == a["small"], a
    assert a["stats"]["retired"] == 0, a


def test_prefetch_disabled_stages_nothing(runs):
    _, b = runs
    assert b["stats"]["hit"] == 0 and b["stats"]["miss"] == len(b["sizes"]), b
