"""
Streamed text must equal the whole-sequence decode for token sequences whose byte-level pieces split
multibyte characters. Each generated token's text comes from a per-token piece table, so a character
spanning two tokens shows up as replacement characters in one or both pieces; the job has to decode the
held tokens together before emitting, both in the normal streaming path and when the job ends with pieces
still held (max_new_tokens). Issue #171: " 描述" is " �" + "�述" in Qwen3's vocabulary, which only a check
for a replacement character anywhere in the held text catches, not one for a trailing one.

Needs a small Qwen3-family model (role "dense", shares the vocabulary with the reported Qwen3-32B); tokens
are forced, so the model's own output does not matter. The reference is Tokenizer.decode of the whole sequence.
"""

import pytest
import torch

from exllamav3 import Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler
from testlib.e2e import load_model

pytestmark = pytest.mark.model("dense")

TEXTS = [
    "| 项目 | 描述 | 状态 | 说明 |",          # two-token words whose second piece starts with a replacement char
    " 描述 状态",
    "한국어 문장을 바이트 단위로 쪼개면 어떻게 될까요",   # Hangul (issue #255)
    "日本語のテキストと絵文字 🎉🚀 と記号 ∑∞≠",
    "Plain ASCII with no multibyte characters at all.",
]


@pytest.fixture(scope = "module")
def gen(model_registry, device):
    with load_model(model_registry.get("dense").path, device, cache_tokens = 2048) as lm:
        yield Generator(model = lm.model, cache = lm.cache, tokenizer = lm.tokenizer), lm.tokenizer


def stream(gen, tok, ids, max_new_tokens):
    job = Job(input_ids = tok.encode("Repeat:", add_bos = True), max_new_tokens = max_new_tokens, sampler = ArgmaxSampler())
    gen.enqueue(job)
    job.constrain_output_now(torch.tensor([ids], dtype = torch.long))
    streamed, out_ids = "", []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            streamed += r.get("text", "")
            if "token_ids" in r:
                out_ids += r["token_ids"].flatten().tolist()
    return streamed, out_ids


@pytest.mark.parametrize("text", TEXTS)
def test_streamed_text_matches_whole_decode(gen, text):
    gen, tok = gen
    ids = tok.encode(text, add_bos = False)[0].tolist()
    streamed, out_ids = stream(gen, tok, ids, len(ids))
    assert out_ids == ids
    assert streamed == tok.decode(torch.tensor([ids]))[0]
    assert "�" not in streamed


@pytest.mark.parametrize("text", TEXTS[:2])
def test_truncation_inside_the_sequence(gen, text):
    # Ending at every token count: the streamed text must equal the whole decode of the tokens emitted,
    # including the cases where the cut lands right after a completed two-token character (issue #171,
    # max_tokens 6 and 9) and the ones where it lands inside one (a genuine trailing replacement char)
    gen, tok = gen
    ids = tok.encode(text, add_bos = False)[0].tolist()
    for n in range(1, len(ids) + 1):
        streamed, out_ids = stream(gen, tok, ids, n)
        assert out_ids == ids[:n], n
        assert streamed == tok.decode(torch.tensor([ids[:n]]))[0], n
