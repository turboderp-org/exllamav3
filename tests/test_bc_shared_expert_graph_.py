"""
BC_BlockSparseMLP's bsz-1..N CUDA graph embeds the shared expert's BC_GatedMLP. That inner module records
different patch sites depending on how it was built: one fused gate+up mgemm, or two separate GEMVs (the
configuration use_mgemm() picks for wide mul1 tensors). The outer launcher must patch the shared expert's
input through the matching site type, otherwise graph replay throws "Graph update failed".

Runs on a real MoE model with shared experts (GPU). argv[1] / EXL3_TEST_MODEL selects the model dir;
argv[2] = "separate" forces the separate-GEMV build for every GatedMLP (monkeypatching use_mgemm), "fused"
keeps the default. Decode logits are checked against the plain forward, teacher-forced on the same tokens.
"""
import sys, os, gc
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

DEFAULT_MODEL = "/mnt/str/models/glm4.6v/exl3/2.13bpw/"
NEW_TOKENS = 12
PROMPT = "The quick brown fox jumps over the lazy dog because"


def main(model_dir, mode):
    config = Config.from_directory(model_dir)
    if mode == "separate":
        config.infer_params.use_mgemm = lambda *a, **k: False
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 4096)
    model.load()
    tok = Tokenizer.from_config(config)
    n_sh = 0
    for m in model.modules:
        for sm in m:
            if isinstance(sm, BlockSparseMLP):
                sm.fused_mode_buffers = None                       # deterministic eager MoE accumulation
                if getattr(sm, "bc_sh_exp", False): n_sh += 1
    assert n_sh > 0, "model has no shared experts on the BC graph path; test is vacuous"
    fused = [sm.shared_experts.multi_gu[0] is not None for m in model.modules for sm in m
             if isinstance(sm, BlockSparseMLP) and getattr(sm, "bc_sh_exp", False)]
    assert all(f == (mode == "fused") for f in fused), f"shared expert build mode mismatch: fused={fused[:4]} mode={mode}"
    with torch.inference_mode():
        gen = Generator(model = model, cache = cache, tokenizer = tok, max_batch_size = 1)
        ids = tok.encode(PROMPT, add_bos = True)
        job = Job(input_ids = ids, max_new_tokens = NEW_TOKENS, sampler = ArgmaxSampler(), return_logits = True)
        gen.enqueue(job); toks, lg = [], []
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                    toks += r["token_ids"].view(-1).tolist(); lg.append(r["logits"].view(-1, r["logits"].shape[-1])[0].clone())
        n = min(len(toks), len(lg))                      # the job may stop early on EOS
        assert n >= 4, f"too few decode steps to check ({n})"
        full = torch.cat((ids, torch.tensor([toks[:n - 1]])), dim = 1)
        ref = model.forward(full, params = {"last_tokens_only": n})[0].float()
        errs = []
        for k in range(n):
            a = lg[k].float().to(ref.device).view(-1); b = ref[k].view(-1); m = torch.isfinite(a) & torch.isfinite(b)
            errs.append(((a[m] - b[m]).norm() / b[m].norm()).item())
    worst = max(errs)
    assert worst < 0.05, f"{mode}: decode logits deviate from the no-cache reference (per-step rfn {[round(e, 4) for e in errs]})"
    print(f"PASS {os.path.basename(model_dir.rstrip('/'))} [{mode}] shared-expert BC layers {n_sh}: max rfn {worst:.5f}; text {tok.decode(torch.tensor([toks]))[0]!r}")
    model.unload(); gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("EXL3_TEST_MODEL", DEFAULT_MODEL), sys.argv[2] if len(sys.argv) > 2 else "separate")
