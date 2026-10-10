"""
Speculative decoding is lossless: greedy generation with a drafter (MTP head, DFlash draft model, n-gram matching)
must produce the plain greedy continuation. Verification runs the target over several positions at once, which can
flip a genuine near-tie, so the sequences may diverge only at a position where the plain run's top two logits are
within NEAR_TIE of each other; everything before the first divergence must match exactly. Each case must also
actually accept drafted tokens, or it would not be testing the drafter.

Cases (DRAFT_CASES): registry role, model_init drafting arguments, and the prompt. The n-gram case uses a
repetitive prompt so the matcher has something to draft from. The hybrid cases (long suffix-match drafts over
the regular drafter, or alone) use a prompt the model continues by reciting its own context, so the suffix
automaton finds a match of at least the hybrid length; they must take at least one hybrid round, whose long
verification pass goes through the recurrent-state replay on the recurrent models.
"""

import pytest
import torch

from testlib.e2e import REFERENCE_TEXT, assert_greedy_equivalent, greedy, load_model

pytestmark = [pytest.mark.model(), pytest.mark.slow]

NEW_TOKENS = 96
PROMPT_TOKENS = 160
NEAR_TIE = 0.25             # max logit gap between the two candidates at a permitted divergence

_REPEAT_TEXT = "\n".join(f"item_{i:03d} = load_item({i}, cache = True)" for i in range(24))
# Verbatim recitation: the text, then its first half again; the greedy continuation is the second half
_RECITE_TEXT = REFERENCE_TEXT[:600] + "\n\n" + REFERENCE_TEXT[:300]
HYBRID = 24

DRAFT_CASES = {
    "mtp": ("mtp", lambda dirs: ["-mtp"], REFERENCE_TEXT),
    "dflash": ("dense-8b", lambda dirs: ["-dm", dirs["dflash-8b"]], REFERENCE_TEXT),
    "ngram": ("dense", lambda dirs: ["-ngram", "2"], _REPEAT_TEXT),
    "ngram-recurrent": ("recurrent", lambda dirs: ["-ngram", "2"], _REPEAT_TEXT),
    "hybrid": ("dense", lambda dirs: ["-hdt", str(HYBRID)], _RECITE_TEXT),
    "hybrid-mtp": ("mtp", lambda dirs: ["-mtp", "-hdt", str(HYBRID)], _RECITE_TEXT),
    "hybrid-ngram-recurrent": ("recurrent", lambda dirs: ["-ngram", "2", "-hdt", str(HYBRID)], _RECITE_TEXT),
}


@pytest.mark.parametrize("case", list(DRAFT_CASES))
@torch.inference_mode()
def test_drafting_is_lossless(case, model_registry, device):
    role, draft_args, text = DRAFT_CASES[case]
    entry = model_registry.get(role)
    if not entry.available:
        pytest.skip(f"test model '{role}' not available")
    dirs = {mid: model_registry.get(mid).path for mid in ([entry.draft] if entry.draft else [])}
    for mid, d in dirs.items():
        if not model_registry.get(mid).available:
            pytest.skip(f"draft model '{mid}' not available")

    hybrid = case.startswith("hybrid")
    with load_model(entry.path, device, *draft_args(dirs)) as lm:
        ids = lm.tokenizer.encode(text, add_bos = True)
        if not hybrid:
            ids = ids[:, :PROMPT_TOKENS]
        plain = lm.generator(draft_model = None, draft_cache = None, ngram_match_min = 0, hybrid_draft_tokens = 0)
        ref_tokens, ref_logits, _ = greedy(plain, ids, NEW_TOKENS, return_logits = True)
        del plain
        gen = lm.generator()
        tokens, _, events = greedy(gen, ids, NEW_TOKENS)
        hybrid_rounds = gen.hybrid_rounds

    final = events.get("eos")
    assert final is not None and final.get("accepted_draft_tokens", 0) > 0, \
        f"{case}: no drafted tokens were accepted ({final and final.get('rejected_draft_tokens')} rejected)"
    if hybrid:
        assert hybrid_rounds > 0, f"{case}: no hybrid draft round fired"
    assert_greedy_equivalent(ref_tokens, ref_logits, tokens, NEAR_TIE, tag = case)
