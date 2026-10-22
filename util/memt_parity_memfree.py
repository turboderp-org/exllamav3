"""
MemT Phase 1 gate (doc/memt_plan.md 3.3): with no memory state the MemT model must be the
backbone. Loads the plain EXL3 backbone, then the MemT sidecar dir built on the same quant,
and compares logits on a fixed prompt: plain vs MemT(no state) vs MemT(all-zero state).
Then a non-gating sanity: commit one turn into memory and read a follow-up from it
(state finite, readout nonzero, logits move).

    python util/memt_parity_memfree.py BACKBONE_DIR MEMT_DIR [--split-plain 21.3,24 --split-memt 22,24]

The two models must land on the same layer split for a bitwise comparison (the same layer
can round differently on the other card); MemT's memory blocks make it ~0.9 GB larger, so
give the plain model a correspondingly smaller cuda:0 budget. The script refuses to call
the gate if the splits differ.
"""

import sys, gc, argparse
import torch
from exllamav3 import Config, Model, Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("backbone_dir"); ap.add_argument("memt_dir")
ap.add_argument("--split-plain", default = None); ap.add_argument("--split-memt", default = None)
ap.add_argument("--gen-tokens", type = int, default = 48)
args = ap.parse_args()

PROMPT = "<|im_start|>user\nThe capital of France is Paris. My favourite colour is teal. What is my favourite colour?<|im_end|>\n<|im_start|>assistant\n"
NOTHINK = "<think>\n\n</think>\n\n"
FACT = "<|im_start|>user\nPlease remember this for later: my cat is named Quillon and my locker code is 4817.<|im_end|>\n<|im_start|>assistant\n" + NOTHINK
FACT_REPLY = "Got it, I'll remember that.<|im_end|>\n"
ASK = "<|im_start|>user\nWhat is my cat's name, and what is my locker code?<|im_end|>\n<|im_start|>assistant\n" + NOTHINK


def load(d, split):
    cfg = Config.from_directory(d)
    m = Model.from_config(cfg)
    m.load(progressbar = True, use_per_device = [float(v) for v in split.split(",")] if split else None)
    return cfg, m, Tokenizer.from_config(cfg)


def split_of(m):
    return [str(getattr(mod, "device", "?")) for mod in m.modules]


@torch.inference_mode()
def greedy(m, tok, ids, params, n):
    out = []
    for _ in range(n):
        t = m.forward(ids, dict(params)).float()[0, -1].argmax().item()
        out.append(t)
        ids = torch.cat([ids, torch.tensor([[t]])], dim = 1)
        if tok.decode(torch.tensor([[t]]), decode_special_tokens = True)[0] == "<|im_end|>":
            break
    return tok.decode(torch.tensor([out]), decode_special_tokens = True)[0]


def unload(m):
    m.unload()
    gc.collect()
    torch.cuda.empty_cache()


def logits(m, ids, params = None):
    return m.forward(ids, dict(params or {})).float().cpu()


backbone_dir, memt_dir = args.backbone_dir, args.memt_dir

print(f"== plain backbone: {backbone_dir}", flush = True)
_, m, tok = load(backbone_dir, args.split_plain)
split_plain = split_of(m)
ids = tok.encode(PROMPT, add_bos = False, encode_special_tokens = True)
print("prompt tokens:", ids.shape[1], flush = True)
ref = logits(m, ids)
ref2 = logits(m, ids)
print(f"control: plain vs plain (same load) bitwise {'EQUAL' if torch.equal(ref, ref2) else 'DIFFERENT'}  max|dlogit| {(ref - ref2).abs().max().item():.3e}", flush = True)
unload(m)
del m
_, m, _ = load(backbone_dir, args.split_plain)
ref3 = logits(m, ids)
print(f"control: plain vs plain (reload)    bitwise {'EQUAL' if torch.equal(ref, ref3) else 'DIFFERENT'}  max|dlogit| {(ref - ref3).abs().max().item():.3e}", flush = True)
unload(m)
del m

print(f"== MemT sidecar: {memt_dir}", flush = True)
_, m, tok2 = load(memt_dir, args.split_memt)
split_memt = split_of(m)
same_split = split_plain == split_memt
print("split plain: cuda:0 up to module", max(i for i, d in enumerate(split_plain) if d == "cuda:0"),
      "| memt: cuda:0 up to module", max(i for i, d in enumerate(split_memt) if d == "cuda:0"),
      "->", "SAME" if same_split else "DIFFERENT", flush = True)
ids2 = tok2.encode(PROMPT, add_bos = False, encode_special_tokens = True)
assert torch.equal(ids, ids2), "tokenizers disagree"
print("memory layers:", list(m.memt_layout.layers), flush = True)

nostate = logits(m, ids)
zero = logits(m, ids, {"memt_read": m.memt_zero_state("cuda")})

ok = True
for name, x in [("MemT no state", nostate), ("MemT zero state", zero)]:
    d = (x - ref).abs().max().item()
    same = torch.equal(x, ref)
    ok &= same
    lp, lq = ref.log_softmax(-1), x.log_softmax(-1)
    kl = (lp.exp() * (lp - lq)).sum(-1).max().item()
    am = (ref.argmax(-1) == x.argmax(-1)).float().mean().item()
    print(f"{name:16s} vs plain: bitwise {'EQUAL' if same else 'DIFFERENT'}  max|dlogit| {d:.3e}  max per-token KL {kl:.3e}  argmax agree {am:.3f}", flush = True)
print("GATE:", ("PASS" if ok else "FAIL") if same_split else "NOT EVALUATED (splits differ, rerun with --split-*)", flush = True)

print("== recall: commit a turn, ask in a fresh context", flush = True)
fact = tok2.encode(FACT + FACT_REPLY, add_bos = False, encode_special_tokens = True)
state = m.memt_commit(fact, m.memt_zero_state("cuda"))
for i, st in state.items():
    print(f"  layer {i:2d}: |S|_F {st.norm().item():.4f}  finite {bool(torch.isfinite(st).all())}", flush = True)
ask = tok2.encode(ASK, add_bos = False, encode_special_tokens = True)
print("  without memory:", repr(greedy(m, tok2, ask, {}, args.gen_tokens)), flush = True)
print("  with memory:   ", repr(greedy(m, tok2, ask, {"memt_read": state}, args.gen_tokens)), flush = True)
both = tok2.encode(FACT + FACT_REPLY + ASK, add_bos = False, encode_special_tokens = True)
print("  fact in context (reference):", repr(greedy(m, tok2, both, {}, args.gen_tokens)), flush = True)
unload(m)
