"""
Generator-level check of the PLE n-gram prefetch (Qwen3.8-Flash-Next 4-layer quantized stub): with
the job staging the next prefill chunk ahead and the model staging the current one before its first
layers, every prefill-sized chunk must take a prefetched staging set (no misses beyond decode-sized
forwards), and the greedy output must be identical to a run with prefetching disabled. Each run is
a separate process (the prompt cache would otherwise skip the second prefill).
"""
import os, sys, json, subprocess, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "/mnt/str/exl3temp/qwen38_stub4_q"
CHUNK = 512


def child():
    import torch
    from exllamav3 import model_init, Generator, Job, ArgmaxSampler
    from exllamav3.modules import NGramEmbedding
    p = __import__("argparse").ArgumentParser()
    model_init.add_args(p)
    model, config, cache, tok = model_init.init(p.parse_args(["-m", MODEL, "-cs", "8192"]))[:4]
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
    ng.prefetch_stats = {"hit": 0, "miss": 0, "retired": 0}     # load-time forwards don't count

    text = open(os.path.join(os.path.dirname(__file__), "..", "README.md")).read()
    ids = tok.encode(text * 3, add_bos = True)[:, :3000 + 137]     # several chunks + a partial one
    out = []
    j = Job(input_ids = ids, max_new_tokens = 12, sampler = ArgmaxSampler())
    gen.enqueue(j)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("token_ids") is not None:
                out += r["token_ids"].flatten().tolist()
    from exllamav3.modules.ngram_embedding import PREFETCH_MIN_TOKENS
    print("RESULT " + json.dumps({
        "tokens": out,
        "stats": ng.prefetch_stats,
        "large": sum(1 for s in sizes if s >= PREFETCH_MIN_TOKENS),
        "small": sum(1 for s in sizes if s < PREFETCH_MIN_TOKENS),
        "sizes": sizes,
        "sums": sums,
    }))


def run(prefetch):
    env = dict(os.environ, EXL3_NGRAM_PREFETCH = "1" if prefetch else "0")
    r = subprocess.run([sys.executable, __file__, "--child"], env = env, capture_output = True, text = True)
    lines = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, r.stdout[-2000:] + r.stderr[-4000:]
    return json.loads(lines[-1][7:])


if __name__ == "__main__":
    if "--child" in sys.argv:
        child()
        sys.exit(0)
    a = run(True)
    b = run(False)
    print("prefetch on :", a["stats"], "chunk sizes", a["sizes"])
    print("prefetch off:", b["stats"])
    # the n-gram embeddings of every forward are bit-identical either way (the sampled tokens
    # need not be: the stub's MoE kernels are not deterministic run to run)
    assert a["sizes"] == b["sizes"]
    assert a["sums"] == b["sums"], [(x, y) for x, y in zip(a["sums"], b["sums"]) if x != y]
    assert len(a["tokens"]) == 12 and a["tokens"][0] == b["tokens"][0], (a["tokens"], b["tokens"])
    assert a["large"] >= 5, a["sizes"]
    # every prefill-sized chunk was staged ahead (job prediction or the model's own hook), nothing else was
    assert a["stats"]["hit"] == a["large"], a
    assert a["stats"]["miss"] == a["small"], a
    assert a["stats"]["retired"] == 0, a
    assert b["stats"]["hit"] == 0 and b["stats"]["miss"] == len(b["sizes"]), b
    print("ok")
