"""
A job never asks the page table for more than the cache holds. The response is capped at the room left past
the prompt (one token per step plus the drafter's speculative window), so a chunked job that writes its way to
the end of the cache ends with max_new_tokens instead of asserting at the requeue that finds no pages left, and
the final requeue segment is cut to the cache rather than aligned past it, so a prompt that leaves only part of
a chunk is still accepted (a 61k-token prompt in a 64k cache with a 31-token hybrid draft window used to be
refused outright).
"""
import argparse
import inspect

import pytest

from exllamav3 import model_init, Generator, Job, ArgmaxSampler

CACHE = 4096
CHUNK = 2048


def init_model(model_dir, device):
    p = argparse.ArgumentParser()
    model_init.add_args(p, **{n: True for n in inspect.signature(model_init.add_args).parameters if "draft" in n})
    return model_init.init(p.parse_args(["-m", model_dir, "-cs", str(CACHE), "-hdt", "31"]), quiet = True, progress = False, device = device)


@pytest.fixture(scope = "module")
def loaded(model_registry, device):
    r = init_model(model_registry.get("recurrent").path, device)
    yield r
    r[0].unload()


def run(gen, job):
    gen.enqueue(job)
    last = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming" and r.get("eos"):
                last = r
    return last


@pytest.mark.model("recurrent")
@pytest.mark.parametrize("hybrid", [0, 31])
def test_chunked_job_ends_at_the_cache_edge(loaded, hybrid):
    model, config, cache, tok, *_ = loaded
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = 1, max_chunk_size = CHUNK,
                    hybrid_draft_tokens = hybrid)
    try:
        ids = tok.encode("Write an endless story about a lighthouse keeper. Never stop.", add_bos = True)
        job = Job(ids, max_new_tokens = 10 * CACHE, max_rq_tokens = CHUNK, stop_conditions = [],
                  sampler = ArgmaxSampler())
        last = run(gen, job)
        assert last["eos_reason"] == "max_new_tokens"
        assert last["new_tokens"] == CACHE - ids.shape[-1] - 1 - gen.max_draft_tokens
    finally:
        gen.close()


@pytest.mark.model("recurrent")
def test_prompt_leaving_part_of_a_chunk_is_accepted(loaded):
    model, config, cache, tok, *_ = loaded
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = 1, max_chunk_size = CHUNK,
                    hybrid_draft_tokens = 31)
    try:
        # A prompt past the midpoint: the first chunk's aligned budget reaches the end of the cache, where the
        # 31-token window does not fit, so the segment has to be cut rather than the job refused
        text = " ".join(f"line {i} of the lighthouse log, wind from the west, lamp lit at dusk." for i in range(400))
        ids = tok.encode(text, add_bos = True)
        ids = ids[:, : CACHE - 1500]
        room = CACHE - ids.shape[-1] - 1 - gen.max_draft_tokens
        assert 1 <= room < CHUNK
        job = Job(ids, max_new_tokens = 10 * CACHE, max_rq_tokens = CHUNK, stop_conditions = [],
                  sampler = ArgmaxSampler())
        last = run(gen, job)
        assert last["eos_reason"] == "max_new_tokens"
        assert last["new_tokens"] == room
    finally:
        gen.close()
