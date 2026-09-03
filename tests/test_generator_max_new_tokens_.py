"""
max_new_tokens must be exact: a job with max_new_tokens = k emits exactly k tokens when nothing else stops it,
with and without speculative decoding (the stop check used to be off by one, and off by another
num_draft_tokens with a draft model). GPU test; pass a model dir as argv[1] (default: a small qwen3.5 quant).
Set EXL3_TEST_DRAFT=mtp to also test with an MTP draft (model must have an MTP head).
"""
import os, sys, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import model_init, Generator, Job, ArgmaxSampler

DEFAULT_MODEL = "/mnt/str/models/qwen3.5-0.8b/exl3/2.00bpw/"
PROMPT = "Count slowly: one, two, three, four, five, six, seven, eight, nine, ten. Again: one, two,"


def main(model_dir, draft):
    p = __import__("argparse").ArgumentParser()
    model_init.add_args(p, **{n: True for n in inspect.signature(model_init.add_args).parameters if "draft" in n})
    argv = ["-m", model_dir, "-cs", "4096"] + (["-mtp", "-ndt", "4"] if draft == "mtp" else [])
    r = model_init.init(p.parse_args(argv)); model, config, cache, tok = r[:4]
    kw = {"draft_model": r[4], "draft_cache": r[6], "num_draft_tokens": 4} if draft == "mtp" else {}
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = 1, **kw)
    for k in (1, 2, 3, 5, 8, 13):
        j = Job(input_ids = tok.encode(PROMPT, add_bos = True), max_new_tokens = k, sampler = ArgmaxSampler())
        gen.enqueue(j); n = 0; reason = None
        while gen.num_remaining_jobs():
            for res in gen.iterate():
                if res.get("token_ids") is not None: n += res["token_ids"].numel()
                if res.get("eos"): reason = res.get("eos_reason")
        assert n == k and reason == "max_new_tokens", f"max_new_tokens={k} (draft={draft}): emitted {n} tokens, eos_reason={reason}"
    print(f"PASS max_new_tokens exact for draft={draft}")


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EXL3_TEST_MODEL", DEFAULT_MODEL)
    main(md, None)
    if os.environ.get("EXL3_TEST_DRAFT") == "mtp": main(md, "mtp")
