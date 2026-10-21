"""
MemT Phase 1 step 4, reference side: run the authors' transformers code (modeling_memt.py,
trust_remote_code) in bf16 on the HF checkpoint and save what the EXL3 side compares against.
bf16 Nano is ~66 GB, so it is spread over the GPUs and CPU RAM with accelerate's device_map.

    python util/memt_release_reference.py /mnt/two/weights/MemT-1.0-Nano OUT.pt [--cpu]

--cpu runs everything on the CPU (bf16, ~66 GB RAM): accelerate's GPU+CPU offload gave NaN
states and garbage logits on the offloaded layers (2026-10-21), the all-CPU path uses the
release's own PyTorch reference kernels.

--capture also records, per memory layer, the release MemTMemory's input hidden states and
outputs for the commit and the follow-up read (for util/memt_release_math_check.py, which
runs our math on exactly these activations, no quantization involved).

Saves token ids, the committed state of one turn, and follow-up logits with and without it.
Run util/memt_release_parity.py on the EXL3 sidecar afterwards (one model in VRAM at a time).
"""

import sys, time, collections
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NOTHINK = "<think>\n\n</think>\n\n"
FACT = "<|im_start|>user\nPlease remember this for later: my cat is named Quillon and my locker code is 4817.<|im_end|>\n<|im_start|>assistant\n" + NOTHINK
FACT_REPLY = "Got it, I'll remember that.<|im_end|>\n"
ASK = "<|im_start|>user\nWhat is my cat's name, and what is my locker code?<|im_end|>\n<|im_start|>assistant\n" + NOTHINK

src, out = sys.argv[1], sys.argv[2]
cpu = "--cpu" in sys.argv[3:]
capture = "--capture" in sys.argv[3:]
tok = AutoTokenizer.from_pretrained(src, trust_remote_code = True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    src, trust_remote_code = True, dtype = torch.bfloat16,
    **({"device_map": "cpu"} if cpu else {"device_map": "auto", "max_memory": {0: "21GiB", 1: "21GiB", "cpu": "100GiB"}}),
)
dm = getattr(model, "hf_device_map", None)
if dm:
    by = collections.defaultdict(list)
    for k, v in dm.items():
        by[str(v)].append(k.replace("model.language_model.", ""))
    for d, ks in by.items():
        print(f"  {d}: {len(ks)} modules, e.g. {ks[:2]} .. {ks[-2:]}", flush = True)
model.eval()
print(f"loaded in {time.time() - t0:.0f}s; devices:", sorted({str(p.device) for p in model.parameters()}), flush = True)

phase = {"name": None}
captured = collections.defaultdict(dict)
if capture:
    def make_hook(idx):
        def hook(mod, args, kwargs, output):
            if phase["name"] is None:
                return
            hidden, state, commit = args[0], args[1], args[2]
            read, updated = output
            captured[phase["name"]][idx] = {
                "hidden": hidden.detach().cpu().clone(),
                "state_in": None if state is None else state.detach().float().cpu().clone(),
                "write_in": None if kwargs.get("write_state") is None else kwargs["write_state"].detach().float().cpu().clone(),
                "commit": bool(commit),
                "read": read.detach().cpu().clone(),
                "updated": None if (updated is None or not commit) else updated.detach().float().cpu().clone(),
            }
        return hook
    for i, layer in enumerate(model.model.language_model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.memt.register_forward_hook(make_hook(i), with_kwargs = True)

dev = model.get_input_embeddings().weight.device
enc = lambda s: tok(s, return_tensors = "pt", add_special_tokens = False).input_ids.to(dev)
fact, ask = enc(FACT + FACT_REPLY), enc(ASK)
print("tokens: fact", fact.shape[1], "ask", ask.shape[1], flush = True)

with torch.no_grad():
    t0 = time.time()
    phase["name"] = "commit"
    state = model.commit_memory(fact)
    phase["name"] = None
    print(f"commit {time.time() - t0:.0f}s:", {k: round(v.norm().item(), 4) for k, v in state.items()}, flush = True)
    nomem_fact = model(fact, use_cache = False).logits.float().cpu()
    nomem = model(ask, use_cache = False).logits.float().cpu()
    phase["name"] = "read"
    mem = model(ask, use_cache = False, memory_state = state).logits.float().cpu()
    phase["name"] = None
    print("ask last-token argmax: nomem", repr(tok.decode(nomem[0, -1].argmax())), "mem", repr(tok.decode(mem[0, -1].argmax())), flush = True)

bad = [k for k, v in state.items() if not torch.isfinite(v).all()]
if bad or not torch.isfinite(nomem).all():
    print("WARNING: non-finite reference output, layers", bad, flush = True)
torch.save({
    "fact_ids": fact.cpu(), "ask_ids": ask.cpu(),
    "state": {k: v.float().cpu() for k, v in state.items()},
    "logits_nomem_fact": nomem_fact, "logits_nomem_ask": nomem, "logits_mem_ask": mem,
    **({"capture": {k: dict(v) for k, v in captured.items()}} if capture else {}),
}, out)
if capture:
    print("captured:", {k: sorted(v) for k, v in captured.items()}, flush = True)
print("saved", out, flush = True)
