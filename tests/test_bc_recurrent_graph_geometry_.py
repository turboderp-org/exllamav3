"""
BC_GatedDeltaNetSplit / BC_Mamba2 capture one CUDA graph per (bsz, seqlen, history) slot, and each graph bakes
in the geometry of the recurrent state buffers it was captured against (conv-state width, history stride).
A process can hold several caches with different max_history (e.g. one with and one without speculative
history). Regression: an older slot must fall back to eager when it is replayed against a cache whose
geometry differs from the one it captured, even after another slot's eager run against the new cache.

Runs on a real small model (GPU). Pass a model dir as argv[1] or via EXL3_TEST_MODEL; defaults to a
qwen3.5-0.8b quant (GatedDeltaNet). For BC_Mamba2 point it at a NemotronH quant.
"""
import sys, os, gc
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

DEFAULT_MODEL = "/mnt/str/models/qwen3.5-0.8b/exl3/2.00bpw/"
NEW_TOKENS = 12
PROMPTS = [
    "The quick brown fox jumps over the lazy dog because",
    "In 1969, the first humans landed on the Moon. The mission was called",
    "Water boils at 100 degrees Celsius at sea level, but on a mountain",
]


def teacher_forced_rfn(model, tok, prompt, tokens, step_logits):
    """Max relative error between the generator's per-step logits and the plain (no-cache) forward scored
    on the same continuation. Cache-mode decode differs from the no-cache path only at kernel precision;
    a graph replayed with wrong state strides is off by orders of magnitude."""
    ids = torch.cat((tok.encode(prompt, add_bos = True), torch.tensor([tokens[:-1]])), dim = 1)
    n = len(tokens)
    ref = model.forward(ids, params = {"last_tokens_only": n})[0].float()          # (n, V')
    worst = 0.0
    for k in range(n):
        a = step_logits[k].float().to(ref.device).view(-1); b = ref[k].view(-1)
        m = torch.isfinite(a) & torch.isfinite(b); a = a[m]; b = b[m]
        worst = max(worst, ((a - b).norm() / b.norm()).item())
    return worst


def generate(model, cache, tok, prompts, n, max_batch_size = 4):
    """Greedy decode; returns per job (tokens, [per-step logits before sampling])."""
    gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = max_batch_size)
    jobs = [Job(input_ids = tok.encode(p, add_bos = True), max_new_tokens = n, sampler = ArgmaxSampler(), return_logits = True) for p in prompts]
    toks = {j: [] for j in jobs}; lg = {j: [] for j in jobs}
    for j in jobs: gen.enqueue(j)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                toks[r["job"]] += r["token_ids"].view(-1).tolist()
                lg[r["job"]].append(r["logits"].view(-1, r["logits"].shape[-1])[0].clone())
    return [(toks[j][:n], lg[j][:n]) for j in jobs]


def main(model_dir):
    torch.manual_seed(0)
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache_a = Cache(model, max_num_tokens = 4096, max_history = 0)   # decode without speculative history
    cache_b = Cache(model, max_num_tokens = 4096, max_history = 4)   # e.g. a generator with 4 draft tokens
    model.load()
    tok = Tokenizer.from_config(config)
    for m in model.modules:
        for sm in m:
            if isinstance(sm, BlockSparseMLP): sm.fused_mode_buffers = None   # deterministic MoE accumulation
    TOL = 2e-2
    def check(label, prompt, res):
        toks, lg = res
        err = teacher_forced_rfn(model, tok, prompt, toks, lg)
        assert err < TOL, f"{label}: decode logits deviate from the no-cache reference (max rfn {err:.4f})"
        return err
    with torch.inference_mode():
        errs = []
        # 1) capture the bsz-1 decode slot against cache A's geometry
        (a0,) = generate(model, cache_a, tok, PROMPTS[:1], NEW_TOKENS)
        errs.append(check("cache A bsz-1", PROMPTS[0], a0))
        # 2) first use of the bsz-2 slot against cache B (eager run) - with an instance-level guard this
        #    re-armed the bsz-1 slot for cache B's geometry
        b1, b2 = generate(model, cache_b, tok, PROMPTS[1:3], NEW_TOKENS)
        errs.append(check("cache B bsz-2", PROMPTS[1], b1)); errs.append(check("cache B bsz-2", PROMPTS[2], b2))
        # 3) bsz-1 decode against cache B must not replay the graph captured against cache A
        (c0,) = generate(model, cache_b, tok, PROMPTS[:1], NEW_TOKENS)
        errs.append(check("cache B bsz-1 after slot re-arm", PROMPTS[0], c0))
        # 4) and cache A still works
        (a1,) = generate(model, cache_a, tok, PROMPTS[:1], NEW_TOKENS)
        errs.append(check("cache A bsz-1 after cache B", PROMPTS[0], a1))
    print(f"PASS {os.path.basename(model_dir.rstrip('/'))}: max rfn vs no-cache reference {max(errs):.5f} (tol {TOL})")
    model.unload(); gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EXL3_TEST_MODEL", DEFAULT_MODEL))
