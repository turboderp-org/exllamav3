"""
MemT serving step 1 (doc/memt_serving_plan.md): per-job memory state through the generator.

Loads the MemT sidecar, commits one turn into a memory state with model.memt_commit, then runs the SAME
follow-up prompt as three generator jobs -- A reads the committed state, Z reads an all-zero state, P has
no memory -- first each alone, then all three concurrently (batched decode, stacked states), then the
batch again (every prompt page is now in the cache under three different hash seeds). Finally a long
job with a tiny requeue budget checks that the state survives the requeue.

Pass criteria, all greedy: every job's tokens are the same alone, batched, and from cache; Z == P
(zero state is the plain backbone); A recalls the committed fact and P does not; the requeued job
matches the un-requeued one.

    python util/memt_gen_test.py /mnt/two/weights/MemT-1.0-Nano-exl3-6.00bpw [--split 20,23]
"""

import sys, argparse, time
import torch
from exllamav3 import Config, Model, Tokenizer, Cache, Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler

ap = argparse.ArgumentParser()
ap.add_argument("memt_dir")
ap.add_argument("--split", default = "20,23")
ap.add_argument("--new-tokens", type = int, default = 40)
ap.add_argument("--long-tokens", type = int, default = 700)
args = ap.parse_args()

NOTHINK = "<think>\n\n</think>\n\n"
FACT = "<|im_start|>user\nPlease remember this for later: my cat is named Quillon and my locker code is 4817.<|im_end|>\n<|im_start|>assistant\n" + NOTHINK
FACT_REPLY = "Got it, I'll remember that.<|im_end|>\n"
ASK = "<|im_start|>user\nWhat is my cat's name, and what is my locker code?<|im_end|>\n<|im_start|>assistant\n" + NOTHINK

t0 = time.time()
cfg = Config.from_directory(args.memt_dir)
model = Model.from_config(cfg)
cache = Cache(model, max_num_tokens = 16384)   # before load: the cache tensors are allocated with the layers
model.load(progressbar = True, use_per_device = [float(v) for v in args.split.split(",")])
tok = Tokenizer.from_config(cfg)
# 256-token checkpoint interval so the requeue test below actually requeues
gen = Generator(model, cache, tok, recurrent_checkpoint_interval = 256, recurrent_checkpoint_interval_pp = 256)
print(f"loaded in {time.time() - t0:.0f}s", flush = True)
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i} allocated {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB", flush = True)

enc = lambda s: tok.encode(s, add_bos = False, encode_special_tokens = True)
zero = model.memt_zero_state()
t0 = time.time()
committed = model.memt_commit(enc(FACT + FACT_REPLY), zero)
print(f"committed {enc(FACT + FACT_REPLY).shape[1]} tokens in {time.time() - t0:.2f}s; "
      f"state |.| per layer: " + " ".join(f"{s.norm().item():.2f}" for s in committed.values()), flush = True)

STATES = {"A": (committed, b"A" * 16), "Z": (zero, b"Z" * 16), "P": (None, None)}

def make_job(name, max_new = args.new_tokens, **kw):
    st, key = STATES[name]
    return Job(input_ids = enc(ASK), max_new_tokens = max_new, sampler = ArgmaxSampler(),
               decode_special_tokens = True, identifier = name, memt_state = st, memt_key = key, **kw)

def run(jobs):
    out = {j.identifier: [] for j in jobs}
    for j in jobs:
        gen.enqueue(j)
    t0 = time.time()
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r["stage"] == "streaming" and "token_ids" in r:
                out[r["identifier"]].append(r["token_ids"])
    dt = time.time() - t0
    toks = {k: (torch.cat(v, dim = -1)[0].tolist() if v else []) for k, v in out.items()}
    return toks, dt

def show(label, toks):
    for k, t in toks.items():
        print(f"  [{label}:{k}] {len(t):3d} tok  {tok.decode(torch.tensor([t]), decode_special_tokens = True)[0]!r}", flush = True)

ok = True
def check(cond, msg):
    global ok
    print(("  PASS " if cond else "  FAIL ") + msg, flush = True)
    ok &= bool(cond)

print("== alone", flush = True)
alone = {}
for name in "AZP":
    t, dt = run([make_job(name)])
    alone[name] = t[name]
    print(f"  {name}: {dt:.2f}s", flush = True)
show("alone", alone)

print("== batched (3 jobs, same prompt, different states)", flush = True)
batched, dt = run([make_job(n) for n in "AZP"])
print(f"  {dt:.2f}s", flush = True)
show("batch", batched)

print("== batched again (prompt pages cached under each seed)", flush = True)
cached, dt = run([make_job(n) for n in "AZP"])
print(f"  {dt:.2f}s", flush = True)
show("cache", cached)

print("== checks", flush = True)
for n in "AZP":
    check(alone[n] == batched[n], f"{n}: alone == batched")
    check(alone[n] == cached[n], f"{n}: alone == from cache")
check(alone["Z"] == alone["P"], "Z (zero state) == P (no memory)")
a_txt = tok.decode(torch.tensor([alone["A"]]), decode_special_tokens = True)[0]
p_txt = tok.decode(torch.tensor([alone["P"]]), decode_special_tokens = True)[0]
check("Quillon" in a_txt and "4817" in a_txt, "A recalls Quillon and 4817 from memory")
check("Quillon" not in p_txt and "4817" not in p_txt, "P does not know the fact")

print(f"== requeue: {args.long_tokens} tokens, max_rq_tokens 256 vs unbounded (no stop conditions)", flush = True)
long_a, dt1 = run([make_job("A", max_new = args.long_tokens)])
long_rq, dt2 = run([make_job("A", max_new = args.long_tokens, max_rq_tokens = 256)])
print(f"  unbounded {dt1:.1f}s ({len(long_a['A'])} tok), requeued {dt2:.1f}s ({len(long_rq['A'])} tok)", flush = True)
n = min(len(long_a["A"]), len(long_rq["A"]))
first_diff = next((i for i in range(n) if long_a["A"][i] != long_rq["A"][i]), None)
check(long_a["A"] == long_rq["A"], f"requeued job == unbounded job (first diff at {first_diff}, lengths {len(long_a['A'])}/{len(long_rq['A'])})")
print("  tail:", repr(tok.decode(torch.tensor([long_rq["A"][-60:]]), decode_special_tokens = True)[0]), flush = True)

print("ALL PASS" if ok else "SOME CHECKS FAILED", flush = True)
sys.exit(0 if ok else 1)
