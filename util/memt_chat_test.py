"""
MemT serving step 2 (doc/memt_serving_plan.md): content-addressed conversation memory in the generator.

A scripted multi-turn chat, resending the full Qwen-format history each turn like a stateless client
(TabbyAPI / its web UI). The generator keeps the completed turns in memory and runs each request on
system turn + live turn only. Then the history is edited mid-way and resent: memory must roll back to
the edited point. Greedy decoding throughout.

    python util/memt_chat_test.py /mnt/two/weights/MemT-1.0-Nano-exl3-6.00bpw [--split 20,23] [--context-turns 0]
"""

import sys, argparse, time
import torch
from exllamav3 import Config, Model, Tokenizer, Cache, Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler

ap = argparse.ArgumentParser()
ap.add_argument("memt_dir")
ap.add_argument("--split", default = "20,23")
ap.add_argument("--context-turns", type = int, default = 0)
ap.add_argument("--new-tokens", type = int, default = 80)
args = ap.parse_args()

SYSTEM = "You are a helpful assistant. Answer briefly."
NOTHINK = "<think>\n\n</think>\n\n"

def render(system, turns, live_user):
    """Qwen chat format, reasoning dropped from past replies (like the UI), no-think generation prompt"""
    s = f"<|im_start|>system\n{system}<|im_end|>\n"
    for u, a in turns:
        s += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
    s += f"<|im_start|>user\n{live_user}<|im_end|>\n<|im_start|>assistant\n{NOTHINK}"
    return s

cfg = Config.from_directory(args.memt_dir)
model = Model.from_config(cfg)
cache = Cache(model, max_num_tokens = 16384)
model.load(progressbar = True, use_per_device = [float(v) for v in args.split.split(",")])
tok = Tokenizer.from_config(cfg)
gen = Generator(model, cache, tok, memt_memory = True, memt_context_turns = args.context_turns, memt_verbose = True)
mem = gen.memt_memory
enc = lambda s: tok.encode(s, add_bos = False, encode_special_tokens = True)

def chat(system, turns, user):
    prompt = render(system, turns, user)
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
    print(f"USER: {user}\nBOT:  {reply}   [{time.time() - t0:.2f}s, {mem.last['short_prompt_tokens']}/{mem.last['prompt_tokens']} tok]\n", flush = True)
    return reply

ok = True
def check(cond, msg):
    global ok
    print(("  PASS " if cond else "  FAIL ") + msg, flush = True)
    ok &= bool(cond)

print("== conversation 1", flush = True)
turns = []
r = chat(SYSTEM, turns, "Please remember this: my cat is named Quillon and my locker code is 4817."); turns.append((turns and None) or ("Please remember this: my cat is named Quillon and my locker code is 4817.", r))
r = chat(SYSTEM, turns, "Also, my favourite colour is teal and I live in Tromsø."); turns.append(("Also, my favourite colour is teal and I live in Tromsø.", r))
r = chat(SYSTEM, turns, "One more: my sister's name is Beatrix."); turns.append(("One more: my sister's name is Beatrix.", r))
stats_before = dict(mem.last)
r = chat(SYSTEM, turns, "What is my cat's name, my favourite colour, and my sister's name?")
check("Quillon" in r, "recalls the cat (turn 1)")
check("teal" in r.lower(), "recalls the colour (turn 2)")
check("Beatrix" in r, "recalls the sister (turn 3)")
check(mem.last["in_memory"] == 3 - args.context_turns and mem.last["committed"] == 1 if args.context_turns == 0 else True,
      f"turn 4: {mem.last}")
check(mem.last["short_prompt_tokens"] < mem.last["prompt_tokens"], "live prompt is shorter than the full history")

print("== regenerate (same history again): nothing new to commit", flush = True)
r2 = chat(SYSTEM, turns, "What is my cat's name, my favourite colour, and my sister's name?")
check(r2 == r, "regenerated reply identical (greedy, same snapshot, cached pages)")
check(mem.last["committed"] == 0, f"no commit on regenerate: {mem.last}")

print("== edit turn 2 (teal -> mauve, Tromsø -> Lisbon) and resend", flush = True)
edited = list(turns)
edited[1] = ("Also, my favourite colour is mauve and I live in Lisbon.", turns[1][1])
r3 = chat(SYSTEM, edited, "What is my favourite colour, and where do I live?")
check("mauve" in r3.lower(), "recalls the EDITED colour")
check("teal" not in r3.lower(), "does not leak the pre-edit colour")
check("Lisbon" in r3, "recalls the edited city")
check(mem.last["matched"] == max(0, min(1, 3 - args.context_turns)), f"rolled back to the prefix before the edit: {mem.last}")

print("== a different chat with the same system prompt: shares nothing past the system turn", flush = True)
r4 = chat(SYSTEM, [("Hi, I'm Dana and my dog is called Pretzel.", "Nice to meet you, Dana! Pretzel is a great name.")],
          "What is my dog called, and do you know anything about my cat?")
check("Pretzel" in r4, "recalls the new chat's dog")
check("Quillon" not in r4, "no cross-chat leak of the cat")

print(f"snapshots held: {len(mem.snapshots)} ({len(mem.snapshots) * mem.snapshot_bytes / 2**20:.0f} MiB)")
print("ALL PASS" if ok else "SOME CHECKS FAILED", flush = True)
sys.exit(0 if ok else 1)
