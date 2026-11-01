"""
Prompt caching: a job whose prompt shares a prefix with an earlier job must reuse the earlier job's cache pages
(and, on recurrent models, its state checkpoints) and continue exactly as a fresh prefill would. With the CPU
page tier (-ccs), the same must hold after the shared pages were evicted from VRAM and are restored from system
RAM.

Reference: the first job's greedy continuation. The reused-prefix run only prefills the uncached tail, so the
arithmetic differs and the continuation may part ways at a near-tie (testlib.e2e.assert_greedy_equivalent).
"""

import pytest
import torch

from testlib.e2e import REFERENCE_TEXT, assert_greedy_equivalent, greedy, load_model

pytestmark = [pytest.mark.models(), pytest.mark.slow]

PAGE = 256
NEW_TOKENS = 32
CHECKPOINT_INTERVAL = PAGE     # recurrent models: a state checkpoint at every page, so prefixes are resumable


def _prompt(lm, tokens):
    ids = lm.tokenizer.encode(REFERENCE_TEXT * 8, add_bos = True)
    assert ids.shape[-1] >= tokens
    return ids[:, :tokens]


def _filler(lm, tokens):
    text = " ".join(f"entry{i} value{i * 7 % 13}" for i in range(tokens))
    return lm.tokenizer.encode(text, add_bos = True)[:, :tokens]


def test_prefix_reuse(model_id, model_dir, device):
    with load_model(model_dir, device, cache_tokens = 8 * PAGE) as lm:
        gen = lm.generator(recurrent_checkpoint_interval = CHECKPOINT_INTERVAL)
        ids = _prompt(lm, 3 * PAGE + 100)
        ref_tokens, ref_logits, first = greedy(gen, ids, NEW_TOKENS, return_logits = True)
        tokens, _, second = greedy(gen, ids, NEW_TOKENS)

    assert first["eos"]["cached_pages"] == 0
    reused = second["eos"]["cached_pages"]
    assert reused >= 3, f"second job reused {reused} pages of a {ids.shape[-1]}-token prompt"
    assert_greedy_equivalent(ref_tokens, ref_logits, tokens, tag = f"{reused} pages reused")


def test_followup_resumes_reply(model_id, model_dir, device):
    """A prompt extending a job's output resumes from the last page boundary of that output, with no regular
    checkpoint anywhere near it, and continues as a fresh prefill would. The sequence runs past the state rings of
    SWA models, where the last page boundary of a job lies just behind a ring shift"""
    with load_model(model_dir, device, cache_tokens = 32 * PAGE) as lm:
        gen = lm.generator(recurrent_checkpoint_interval = 64 * PAGE)
        ids = _prompt(lm, 4 * PAGE + 100)
        reply, _, _ = greedy(gen, ids, 4 * PAGE + 60)
        suffix = lm.tokenizer.encode("\n\nAnd then?\n", add_bos = False)
        followup = torch.cat((ids, torch.tensor([reply]), suffix), dim = -1)
        tokens, _, second = greedy(gen, followup, NEW_TOKENS)
        ref_tokens, ref_logits, _ = greedy(lm.generator(), followup, NEW_TOKENS, return_logits = True)

    boundary = (ids.shape[-1] + len(reply) - 1) // PAGE * PAGE
    cached = second["eos"]["cached_tokens"]
    assert cached >= boundary, f"resumed at {cached} of {followup.shape[-1]} tokens, the reply's last page ends at {boundary}"
    assert_greedy_equivalent(ref_tokens, ref_logits, tokens, tag = f"resumed at {cached}")


def test_cpu_tier_restore(model_id, model_dir, device):
    with load_model(model_dir, device, "-ccs", "1", cache_tokens = 8 * PAGE) as lm:
        gen = lm.generator(recurrent_checkpoint_interval = CHECKPOINT_INTERVAL)
        assert gen.cpu_page_cache is not None, "-ccs did not create the CPU page tier"
        ids = _prompt(lm, 3 * PAGE + 100)
        ref_tokens, ref_logits, _ = greedy(gen, ids, NEW_TOKENS, return_logits = True)
        # A distinct prompt filling the whole cache evicts every page of the first one
        greedy(gen, _filler(lm, 7 * PAGE), 8)
        pushes = gen.cpu_page_cache.metrics["pushes"]
        restores = gen.cpu_page_cache.metrics["restores"]
        tokens, _, again = greedy(gen, ids, NEW_TOKENS)
        restored = gen.cpu_page_cache.metrics["restores"] - restores

    assert pushes >= 3, f"only {pushes} pages were pushed to the CPU tier on eviction"
    assert restored >= 3 and again["eos"]["cached_pages"] >= 3, \
        f"{restored} pages restored, {again['eos']['cached_pages']} reused"
    assert_greedy_equivalent(ref_tokens, ref_logits, tokens, tag = f"{restored} pages restored from the CPU tier")
