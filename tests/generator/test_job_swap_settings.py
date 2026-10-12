"""
Mid-generation replacement of a job's sampler, filters and banned strings (Job.set_sampler / set_filters /
set_banned_strings and the AsyncJob wrappers). Every change must take effect from the next token sampled after
the call, including for a job that has been requeued in between, and from inside an `async for` over an
AsyncJob. Checked on the generated token sequence (forced tokens, allowed sets) and the streamed text against
the whole-sequence decode. Model role "dense-llama".
"""

import asyncio

import pytest
import torch

from exllamav3 import Generator, Job, AsyncGenerator, AsyncJob, Filter
from exllamav3.generator.sampler import ArgmaxSampler, CustomSampler, SS_LogitBias, SS_Argmax
from testlib.e2e import load_model

pytestmark = pytest.mark.model("dense-llama")

PROMPT = "Write a long story about a fox who lives in the forest and learns to read. Once upon a time, the"


class AllowSetFilter(Filter):
    """Allows only a fixed set of tokens, forever"""
    def __init__(self, tokenizer, allowed, background):
        super().__init__(tokenizer, None, None, False)
        self.allowed = list(allowed)
        self.background = background
    def reset(self): pass
    def accept_token(self, token): pass
    def is_completed(self): return False
    def use_background_worker(self): return self.background
    def get_next_logit_mask(self):
        m = torch.full((1, self.vocab_size), float("-inf"), dtype = torch.half)
        m[:, self.allowed] = 0
        return m


def forced_sampler(token):
    return CustomSampler([SS_LogitBias({token: 100.0}), SS_Argmax()])


def run(gen, job, on_step):
    """Iterate until the job ends, calling on_step(job, step) after every iteration; returns the streamed text"""
    gen.enqueue(job)
    step = 0
    text = ""
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            text += r.get("text", "")
        on_step(job, step)
        step += 1
    return text


def sequence(job):
    return job.sequences[0].sequence_ids.torch()[0].tolist()


@pytest.fixture(scope = "module")
def lm(model_registry, device):
    with load_model(model_registry.get("dense-llama").path, device, cache_tokens = 8192) as lm:
        yield lm


@pytest.fixture(scope = "module")
def tok(lm):
    return lm.tokenizer


@pytest.fixture(scope = "module")
def ids(tok):
    return tok.encode(PROMPT, add_bos = True)


@pytest.fixture
def gen(lm):
    gen = Generator(model = lm.model, cache = lm.cache, tokenizer = lm.tokenizer, max_batch_size = 1)
    yield gen
    gen.close()


def test_set_sampler(gen, tok, ids):
    target = tok.single_id(" fox")
    mark = {}
    def on_step(job, step):
        if step == 6:
            mark["len"] = len(job.sequences[0].sequence_ids)
            job.set_sampler(forced_sampler(target))
    job = Job(input_ids = ids, max_new_tokens = 24, sampler = ArgmaxSampler())
    run(gen, job, on_step)
    seq = sequence(job)
    before, after = seq[ids.shape[-1]:mark["len"]], seq[mark["len"]:]
    assert after and all(t == target for t in after), f"tokens after set_sampler: {after}"
    assert not all(t == target for t in before), "forced token appears before the switch"


@pytest.mark.parametrize("background", [False, True], ids = ["inline", "background"])
def test_set_filters(gen, tok, ids, background):
    allowed = [tok.single_id(" yes"), tok.single_id(" no")]
    outside = tok.single_id(" fox")
    mark = {}
    def on_step(job, step):
        if step == 5:
            mark["a"] = len(job.sequences[0].sequence_ids)
            job.set_filters([AllowSetFilter(tok, allowed, background)])
        if step == 15:
            # Removal check: a token outside the allowed set, forced by the sampler, which a filter that
            # is still active would mask out
            mark["b"] = len(job.sequences[0].sequence_ids)
            job.set_filters(None)
            job.set_sampler(forced_sampler(outside))
    job = Job(input_ids = ids, max_new_tokens = 30, sampler = ArgmaxSampler())
    run(gen, job, on_step)
    seq = sequence(job)
    constrained, free = seq[mark["a"]:mark["b"]], seq[mark["b"]:]
    assert constrained and all(t in allowed for t in constrained), f"constrained span: {constrained}"
    assert free and all(t == outside for t in free), f"span after removing the filter: {free}"


def test_set_filters_after_requeue(gen, tok, ids):
    allowed = [tok.single_id(" yes"), tok.single_id(" no")]
    mark = {}
    def on_step(job, step):
        if "a" not in mark and job.is_requeued and job.new_tokens > 2:
            mark["a"] = len(job.sequences[0].sequence_ids)
            job.set_filters([AllowSetFilter(tok, allowed, False)])
    job = Job(input_ids = ids, max_new_tokens = 300, max_rq_tokens = 16, sampler = ArgmaxSampler())
    run(gen, job, on_step)
    assert "a" in mark, "job never requeued"
    after = sequence(job)[mark["a"]:]
    assert after and all(t in allowed for t in after), f"tokens after set_filters on a requeued job: {after[:20]}"


def test_set_banned_strings_releases_held_text(gen, tok, ids):
    # The banned strings start with a very common word, so a partial match holds text almost immediately
    banned = [" the zqxj", " a zqxj", " and zqxj", " of zqxj", " to zqxj"]
    state = {}
    def on_step(job, step):
        if "released" not in state and job.checkpoint is not None and job.checkpoint["offset"] > 0:
            state["held"] = job.held_text
            job.set_banned_strings(None)
            state["checkpoint_after"] = job.checkpoint
            state["released"] = step
    job = Job(input_ids = ids, max_new_tokens = 40, sampler = ArgmaxSampler(), banned_strings = banned)
    text = run(gen, job, on_step)
    assert "released" in state, "no banned-string hold occurred"
    assert state["checkpoint_after"] is None, "set_banned_strings(None) left a checkpoint"
    full = tok.decode(torch.tensor([sequence(job)[ids.shape[-1]:]]))[0]
    assert text == full, f"streamed text differs from the generated sequence:\n{text!r}\n{full!r}"
    assert state["held"] in text, f"held text {state['held']!r} was not emitted"


def test_async_set_sampler(lm, tok, ids):
    target = tok.single_id(" fox")
    async def main():
        agen = AsyncGenerator(model = lm.model, cache = lm.cache, tokenizer = tok, max_batch_size = 1)
        job = AsyncJob(agen, input_ids = ids, max_new_tokens = 24, sampler = ArgmaxSampler())
        mark = None
        n = 0
        async for r in job:
            n += 1
            if n == 6 and mark is None:
                mark = len(job.job.sequences[0].sequence_ids)
                job.set_sampler(forced_sampler(target))
        await agen.close()
        return mark, sequence(job.job)
    mark, seq = asyncio.run(main())
    after = seq[mark:]
    assert after and all(t == target for t in after), f"AsyncJob tokens after set_sampler: {after}"
