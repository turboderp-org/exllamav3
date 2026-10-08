"""
max_new_tokens must be exact: a job with max_new_tokens = k emits exactly k tokens and ends with eos_reason
"max_new_tokens" when nothing else stops it, with and without speculative decoding (the stop check used to be
off by one, and off by another num_draft_tokens with a draft model). Models are loaded through model_init, as
the example scripts do: a small recurrent model without drafting, and an MTP model with and without its MTP
draft head.
"""

import argparse
import inspect

import pytest

from exllamav3 import model_init, Generator, Job, ArgmaxSampler

PROMPT = "Count slowly: one, two, three, four, five, six, seven, eight, nine, ten. Again: one, two,"
LIMITS = (1, 2, 3, 5, 8, 13)


def init_model(model_dir, device, mtp):
    p = argparse.ArgumentParser()
    model_init.add_args(p, **{n: True for n in inspect.signature(model_init.add_args).parameters if "draft" in n})
    argv = ["-m", model_dir, "-cs", "4096"] + (["-mtp", "-ndt", "4"] if mtp else [])
    return model_init.init(p.parse_args(argv), quiet = True, progress = False, device = device)


@pytest.fixture(scope = "module")
def loaded(model_registry, device):
    """loaded(role, mtp) -> model_init.init result, loaded once per (role, mtp) and unloaded at module end"""
    models = {}
    def get(role, mtp):
        if (role, mtp) not in models:
            models[(role, mtp)] = init_model(model_registry.get(role).path, device, mtp)
        return models[(role, mtp)]
    yield get
    for r in models.values():
        for m in (r[0], r[4]):
            if m is not None:
                m.unload()


@pytest.mark.parametrize("k", LIMITS)
@pytest.mark.parametrize("role, draft", [
    pytest.param("recurrent", None, marks = pytest.mark.model("recurrent"), id = "recurrent"),
    # The 9B MTP model takes longer to load and warm up than the rest of the module together
    pytest.param("mtp", None, marks = [pytest.mark.model("mtp"), pytest.mark.slow], id = "mtp-model-no-draft"),
    pytest.param("mtp", "mtp", marks = [pytest.mark.model("mtp"), pytest.mark.slow], id = "mtp-draft"),
])
def test_max_new_tokens_exact(loaded, role, draft, k):
    model, config, cache, tok, draft_model, draft_config, draft_cache = loaded(role, role == "mtp")
    kw = {"draft_model": draft_model, "draft_cache": draft_cache, "num_draft_tokens": 4} if draft == "mtp" else {}
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = 1, **kw)
    try:
        job = Job(input_ids = tok.encode(PROMPT, add_bos = True), max_new_tokens = k, sampler = ArgmaxSampler())
        gen.enqueue(job)
        n = 0
        reason = None
        while gen.num_remaining_jobs():
            for res in gen.iterate():
                if res.get("token_ids") is not None:
                    n += res["token_ids"].numel()
                if res.get("eos"):
                    reason = res.get("eos_reason")
        assert (n, reason) == (k, "max_new_tokens"), f"emitted {n} tokens, eos_reason = {reason}"
    finally:
        gen.close()
