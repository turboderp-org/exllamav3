"""
MemT Phase 1 step 4, EXL3 side: compare the sidecar model against the release reference
saved by util/memt_release_reference.py (same token ids).

    python util/memt_release_parity.py MEMT_EXL3_DIR REF.pt

Yardstick: the no-memory KL (EXL3 backbone vs the release's bf16 backbone) is the
quantization floor. The memory path matches the release if the with-memory KL is close to
that floor, both reading our own committed state and reading the release's state (the
latter isolates the read from the commit).
"""

import sys
import torch
from exllamav3 import Config, Model

memt_dir, ref_path = sys.argv[1], sys.argv[2]
ref = torch.load(ref_path)
fact, ask = ref["fact_ids"], ref["ask_ids"]
bad = [k for k, v in ref["state"].items() if not torch.isfinite(v).all()]
assert not bad and torch.isfinite(ref["logits_nomem_ask"]).all(), f"reference has non-finite output (layers {bad})"

cfg = Config.from_directory(memt_dir)
m = Model.from_config(cfg)
m.load(progressbar = True)


def logits(ids, params = None):
    return m.forward(ids, dict(params or {})).float().cpu()


def kl(p_logits, q_logits):
    """KL(p || q) per position, p = reference."""
    lp, lq = p_logits.log_softmax(-1), q_logits.log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1)[0]


def row(name, r, x):
    k = kl(r, x)
    am = (r.argmax(-1) == x.argmax(-1)).float().mean().item()
    print(f"  {name:38s} KL mean {k.mean().item():.3e}  max {k.max().item():.3e}  last {k[-1].item():.3e}  argmax agree {am:.3f}", flush = True)


with torch.inference_mode():
    ours = m.memt_commit(fact, m.memt_zero_state("cuda"))
    print("== committed state, ours vs release (per layer)")
    for i in sorted(ours):
        a, b = ours[i].float().cpu().reshape(-1), ref["state"][i].float().reshape(-1)
        rel = ((a - b).norm() / b.norm()).item()
        cos = torch.nn.functional.cosine_similarity(a, b, dim = 0).item()
        print(f"  layer {i:2d}: |S| ours {a.norm().item():9.4f}  release {b.norm().item():9.4f}  rel err {rel:.3e}  cos {cos:.6f}", flush = True)

    ref_state = {i: s.to("cuda") for i, s in ref["state"].items()}
    print("== logits vs release (KL(release || ours), nats)")
    row("no memory, fact turn (floor)", ref["logits_nomem_fact"], logits(fact))
    row("no memory, follow-up (floor)", ref["logits_nomem_ask"], logits(ask))
    row("memory, our state", ref["logits_mem_ask"], logits(ask, {"memt_read": ours}))
    row("memory, release state", ref["logits_mem_ask"], logits(ask, {"memt_read": ref_state}))
    print("== size of the memory effect itself, for scale")
    row("release: memory vs no memory", ref["logits_mem_ask"], ref["logits_nomem_ask"])
m.unload()
