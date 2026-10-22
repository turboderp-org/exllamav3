"""
MemT commit policy (doc/memt_serving_plan.md, section 5): commit_every, the typed "/commit" and
"/commit every N" commands, and the template_vars route (a "memt" control turn emitted by the chat template).

Same stateless-client setup as memt_chat_test.py, generator started with commit_every = 0 (never commit on
its own) so every commit here has to come from a command. Greedy decoding.

    python util/memt_commit_test.py /mnt/two/weights/MemT-1.0-Nano-exl3-6.00bpw [--split 20,23]
"""

import sys, argparse, time
import torch
from exllamav3 import Config, Model, Tokenizer, Cache, Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler

ap = argparse.ArgumentParser()
ap.add_argument("memt_dir")
ap.add_argument("--split", default = "20,23")
ap.add_argument("--new-tokens", type = int, default = 80)
args = ap.parse_args()

NOTHINK = "<think>\n\n</think>\n\n"

def render(system, turns, live_user, control = None):
    s = f"<|im_start|>system\n{system}<|im_end|>\n"
    for u, a in turns:
        s += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
    s += f"<|im_start|>user\n{live_user}<|im_end|>\n"
    if control:
        s += f"<|im_start|>memt\n{control}\n<|im_end|>\n"   # what the sidecar template emits from template_vars
    s += f"<|im_start|>assistant\n{NOTHINK}"
    return s

cfg = Config.from_directory(args.memt_dir)
model = Model.from_config(cfg)
cache = Cache(model, max_num_tokens = 16384)
model.load(progressbar = True, use_per_device = [float(v) for v in args.split.split(",")])
tok = Tokenizer.from_config(cfg)
gen = Generator(model, cache, tok, memt_memory = True, memt_commit_every = 0, memt_verbose = True)
mem = gen.memt_memory
enc = lambda s: tok.encode(s, add_bos = False, encode_special_tokens = True)

def chat(system, turns, user, control = None):
    prompt = render(system, turns, user, control)
    job = Job(input_ids = enc(prompt), max_new_tokens = args.new_tokens, sampler = ArgmaxSampler(),
              stop_conditions = [tok.eos_token_id, "<|im_end|>"])
    gen.enqueue(job)
    text = ""
    t0 = time.time()
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r["stage"] == "streaming":
                text += r.get("text", "")
    reply = text.strip()
    print(f"USER: {user!r}\nBOT:  {reply}   [{time.time() - t0:.2f}s, {mem.last['short_prompt_tokens']}/{mem.last['prompt_tokens']} tok]\n", flush = True)
    turns.append((user, reply))
    return reply

def seen():
    return tok.decode(torch.tensor(mem.last_prompt), decode_special_tokens = True)

ok = True
def check(cond, msg):
    global ok
    print(("  PASS " if cond else "  FAIL ") + msg, flush = True)
    ok &= bool(cond)

SYSTEM = "You are a helpful assistant. Answer briefly."
print("== A. commit_every 0: nothing is committed, the history rides raw", flush = True)
turns = []
chat(SYSTEM, turns, "Please remember this: my cat is named Quillon and my locker code is 4817.")
chat(SYSTEM, turns, "Also, my favourite colour is teal and I live in Tromsø.")
chat(SYSTEM, turns, "One more: my sister's name is Beatrix.")
r = chat(SYSTEM, turns, "What is my cat's name, my favourite colour, and my sister's name?")
check(mem.last["in_memory"] == 0 and mem.last["committed"] == 0, f"nothing committed: {mem.last}")
check(mem.last["short_prompt_tokens"] == mem.last["prompt_tokens"], "prompt passed through whole")
check("Quillon" in r and "teal" in r.lower() and "Beatrix" in r, "recalls all three from the raw context")

print("== B. typed /commit in the live message: everything before it goes into memory", flush = True)
r = chat(SYSTEM, turns, "What is my sister's name?\n/commit")
check(mem.last["in_memory"] == 4 and mem.last["committed"] == 4, f"4 units committed: {mem.last}")
check("/commit" not in seen(), "the command is stripped from what the model sees")
check(mem.last["short_prompt_tokens"] < mem.last["prompt_tokens"], "live prompt shorter than the history")
check("Beatrix" in r, "recalls the sister from memory")

print("== C. next turn: the old /commit still pins the floor, newer turns stay raw", flush = True)
chat(SYSTEM, turns, "New fact: my car is a green Saab.")
r = chat(SYSTEM, turns, "What is my cat's name and what car do I drive?")
check(mem.last["in_memory"] == 4 and mem.last["matched"] == 4 and mem.last["committed"] == 0, f"floor 4, no new commit: {mem.last}")
check(mem.last["units"] == 6, f"6 completed units: {mem.last}")
check("/commit" not in seen(), "the historical command is stripped too")
check("Quillon" in r, "recalls the cat from memory")
check("Saab" in r, "recalls the car from the raw turns")

print("== D. '/commit every 2' in the system prompt", flush = True)
SYS2 = "You are a helpful assistant. Answer briefly.\n/commit every 2"
turns2 = []
chat(SYS2, turns2, "Remember: my dog is called Pretzel.")
chat(SYS2, turns2, "Remember: my bike is yellow.")
check(mem.last["in_memory"] == 0, f"after 1 unit: {mem.last}")
chat(SYS2, turns2, "Remember: my street is Birch Lane.")
check(mem.last["in_memory"] == 2 and mem.last["committed"] == 2 and mem.last["commit_every"] == 2, f"after 2 units: {mem.last}")
check("/commit" not in seen(), "system-prompt command stripped")
r = chat(SYS2, turns2, "What is my dog called, what colour is my bike, and what street do I live on?")
check(mem.last["in_memory"] == 2 and mem.last["committed"] == 0, f"after 3 units, still 2: {mem.last}")
check("Pretzel" in r and "yellow" in r.lower() and "Birch" in r, "recalls two from memory, one raw")
r = chat(SYS2, turns2, "And my dog again?")
check(mem.last["in_memory"] == 4 and mem.last["committed"] == 2, f"after 4 units: {mem.last}")
check("Pretzel" in r, "recalls the dog")

print("== E. template_vars route: a memt control turn on this request only", flush = True)
turns3 = []
chat(SYSTEM, turns3, "Remember: my password hint is 'otter'.")
chat(SYSTEM, turns3, "Remember: I was born in Ghent.")
r = chat(SYSTEM, turns3, "Where was I born?", control = "/commit")
check(mem.last["in_memory"] == 2 and mem.last["committed"] == 2, f"memt turn commits: {mem.last}")
check("memt" not in seen() and "/commit" not in seen(), "control turn removed from the prompt")
check("Ghent" in r, "recalls from memory")
r = chat(SYSTEM, turns3, "And my password hint?")
check(mem.last["in_memory"] == 2 and mem.last["committed"] == 0, f"next request without the var: floor stays at the matched 2: {mem.last}")
check("otter" in r.lower(), "recalls the hint")
r = chat(SYSTEM, turns3, "Say hi.", control = "/commit every 1")
check(mem.last["in_memory"] == 4 and mem.last["committed"] == 2 and mem.last["commit_every"] == 1, f"'every 1' for one request: {mem.last}")

print("== G. a message that is only /commit: canned reply, no generation, exchange skipped later", flush = True)
turns4 = []
chat(SYSTEM, turns4, "Remember: my lucky number is 42.")
chat(SYSTEM, turns4, "Remember: my hometown is Bergen.")
r = chat(SYSTEM, turns4, "/commit")
check(r == "Memorized 2 exchanges." and mem.last["canned"] == r, f"canned reply: {r!r}")
check(mem.last["in_memory"] == 2 and mem.last["committed"] == 2, f"committed on the bare command: {mem.last}")
r = chat(SYSTEM, turns4, "/commit")
check(r == "Nothing new to memorize (3 exchanges in memory).", f"second bare /commit: {r!r}")
check(mem.last["skipped"] == 1 and mem.last["committed_tokens"] == 0, f"the bare exchange was skipped, not committed: {mem.last}")
r = chat(SYSTEM, turns4, "/commit every 3")
check(r == "Memorizing every 3 exchanges from now.", f"bare 'every': {r!r}")
r = chat(SYSTEM, turns4, "What is my lucky number and my hometown?")
check("42" in r and "Bergen" in r, "recall after the bare commands")
check("Memorized" not in seen() and "Nothing new" not in seen(), "canned exchanges kept out of the raw context")

print("== F. regenerate after a typed /commit: nothing new committed, same answer", flush = True)
t = list(turns)
t.pop()   # drop the last exchange and resend the same question
r1 = turns[-1][1]
r2 = chat(SYSTEM, t, turns[-1][0])
check(r2 == r1 and mem.last["committed"] == 0, f"identical reply, no commit: {mem.last}")

print(f"snapshots held: {len(mem.snapshots)} ({len(mem.snapshots) * mem.snapshot_bytes / 2**20:.0f} MiB)")
print("ALL PASS" if ok else "SOME CHECKS FAILED", flush = True)
sys.exit(0 if ok else 1)
