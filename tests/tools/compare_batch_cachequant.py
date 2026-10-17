"""
Batched-generator check of a quantized cache: N copies of the same long prompt run
concurrently through the Generator (exercising the batched graph paths: BC_DSV4BatchAttention
on DeepSeek-V4, the bsz > 1 slots elsewhere), greedy, with return_logits. Checks that every job
produced identical first-step logits (batch symmetry) and compares the batched first-step
logits against a single-job run of the same cache type, for fp16 and quantized caches.

    python tests/compare_batch_cachequant_.py -m /mnt/str/models/deepseek-v4-flash-0731/exl3/2.04bpw -p 5000
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "eval"))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ArgmaxSampler
from ppl import get_dataset_text

p = argparse.ArgumentParser()
p.add_argument("-m", "--model_dir", required = True)
p.add_argument("-p", "--prompt_len", type = int, default = 5000)
p.add_argument("-n", "--new_tokens", type = int, default = 8)
p.add_argument("-b", "--bits", default = "8")
p.add_argument("-j", "--jobs", type = int, default = 3)
p.add_argument("-d", "--device", default = "cuda:0")
args = p.parse_args()

config = Config.from_directory(args.model_dir)
model = Model.from_config(config)
tok = Tokenizer.from_config(config)
per_job = ((args.prompt_len + args.new_tokens + 255) // 256 + 1) * 256
caches = {"fp16": Cache(model, max_num_tokens = per_job * args.jobs, max_batch_size = args.jobs)}
for b in [int(x) for x in args.bits.split(",")]:
    caches[f"q{b}"] = Cache(model, max_num_tokens = per_job * args.jobs, max_batch_size = args.jobs,
                            layer_type = CacheLayer_quant, k_bits = b, v_bits = b)
model.load(args.device, progressbar = True)
text = get_dataset_text({"dataset": "wiki2"})
ids = tok.encode(text, add_bos = True)[:, :args.prompt_len]

def run(cache, n_jobs):
    gen = Generator(model, cache, tok, max_batch_size = n_jobs, max_chunk_size = 2048)
    jobs = [Job(input_ids = ids, max_new_tokens = args.new_tokens, return_logits = True,
                sampler = ArgmaxSampler(), decode_special_tokens = True) for _ in range(n_jobs)]
    for j in jobs:
        gen.enqueue(j)
    logits = {id(j): [] for j in jobs}
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if "logits" in r:
                logits[id(r["job"])].append(r["logits"].float().cpu().view(-1, r["logits"].shape[-1]))
    outs = [torch.cat(logits[id(j)], dim = 0) for j in jobs]
    for i, o in enumerate(outs):
        bad = (~torch.isfinite(o)).sum(-1)
        if bad.any():
            print(f"    job {i}: non-finite logits per step: {bad.tolist()}")
    return outs

def _fin(t):
    # Masked vocabulary columns come back as -inf from the generator; keep them out of the
    # comparison (they are identical in every run)
    return t.masked_fill(~torch.isfinite(t), -1e4)

def kl(ref, got):
    lr = torch.log_softmax(_fin(ref).double(), -1)
    lg = torch.log_softmax(_fin(got).double(), -1)
    return (lr.exp() * (lr - lg)).sum(-1)

single = {}
for name, cache in caches.items():
    single[name] = run(cache, 1)[0]
    print(f" -- {name} single: {single[name].shape[0]} steps")
ref = single["fp16"]
if os.environ.get("EXL3_CMP_DUMP"):
    torch.save({k: v.cpu() for k, v in single.items()}, os.environ["EXL3_CMP_DUMP"])
print(f"\n{'config':<8} {'jobs':>4} | {'sym max|d|':>10} | {'KL vs single':>12} {'KL vs fp16 single':>17} {'argmax':>7}")
for name, cache in caches.items():
    outs = run(cache, args.jobs)
    sym = max((_fin(o) - _fin(outs[0])).abs().max().item() for o in outs[1:])
    k1 = kl(single[name], outs[0]).mean().item()
    k2 = kl(ref, outs[0]).mean().item()
    am = (ref.argmax(-1) == outs[0].argmax(-1)).float().mean().item() * 100
    print(f"{name:<8} {args.jobs:>4} | {sym:>10.2e} | {k1:>12.2e} {k2:>17.2e} {am:>6.1f}%")
