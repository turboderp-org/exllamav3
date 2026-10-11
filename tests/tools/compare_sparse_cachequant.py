"""
Real-model check of sparse attention over a quantized cache. Teacher-forced: the same prompt
and continuation are run with an fp16 cache and with quantized caches, and the per-position
logits are compared against the fp16/graph run (KL, argmax agreement) over
  - the last prefill chunk (sparse prefill over the packed cache), and
  - N single-token decode steps (graph path and eager dispatch path).
Works for GLM-5.x DSA (sparse past index_topk) and Qwen3.8 QSA (past 4 * block_topk + 3).

    python tests/compare_sparse_cachequant_.py -m /mnt/str/models/glm5.3-flash/exl3/2.05bpw -p 5000 -n 48
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "eval"))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.cache import CacheLayer_quant
from ppl import get_dataset_text

p = argparse.ArgumentParser()
p.add_argument("-m", "--model_dir", required = True)
p.add_argument("-p", "--prompt_len", type = int, default = 5000)
p.add_argument("-n", "--new_tokens", type = int, default = 48)
p.add_argument("-b", "--bits", default = "8,4")
p.add_argument("-d", "--device", default = "cuda:0")
p.add_argument("--chunk", type = int, default = 2048)
p.add_argument("--no_eager", action = "store_true")
args = p.parse_args()

config = Config.from_directory(args.model_dir)
model = Model.from_config(config)
tok = Tokenizer.from_config(config)
cache_len = ((args.prompt_len + args.new_tokens + 255) // 256 + 1) * 256
caches = {"fp16": Cache(model, max_num_tokens = cache_len, max_batch_size = 1)}
for b in [int(x) for x in args.bits.split(",")]:
    caches[f"q{b}"] = Cache(model, max_num_tokens = cache_len, max_batch_size = 1,
                            layer_type = CacheLayer_quant, k_bits = b, v_bits = b)
model.load(args.device, progressbar = True)
dev = torch.device(args.device)

text = get_dataset_text({"dataset": "wiki2"})
ids = tok.encode(text, add_bos = True)[:, :args.prompt_len + args.new_tokens]
P, N = args.prompt_len, args.new_tokens
print(f" -- tokens: {ids.shape[1]} (prompt {P} + forced continuation {N})")

from exllamav3.modules import mla_attn, attn as attn_mod, dsv4 as dsv4_mod
_orig_bc_mla = mla_attn.MLAttention.bc_mla_step
_orig_bc_attn = attn_mod.Attention.bc_attn_step
_orig_bc_dsa = dsv4_mod.bc_dsa_enable

def fwd(x, params):
    y = model.forward(x, params)
    return y["logits"] if isinstance(y, dict) else y

def run(cache, eager):
    if eager:
        mla_attn.MLAttention.bc_mla_step = lambda self, *a, **k: None
        attn_mod.Attention.bc_attn_step = lambda self, *a, **k: None
        dsv4_mod.bc_dsa_enable = False
    else:
        mla_attn.MLAttention.bc_mla_step = _orig_bc_mla
        attn_mod.Attention.bc_attn_step = _orig_bc_attn
        dsv4_mod.bc_dsa_enable = _orig_bc_dsa
    t0 = time.time()
    with torch.inference_mode():
        state = cache.get_new_state() if cache.recurrent_layers else None
        def params(pos):
            pr = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, cache_len), "past_len": pos}
            if state is not None:
                pr["recurrent_states"] = [state]
            return pr
        pf = None
        for a in range(0, P, args.chunk):
            b = min(a + args.chunk, P)
            y = fwd(ids[:, a:b].to(dev), params(a))
            pf = y[0].float().cpu()          # last chunk's logits survive
        dec = []
        for i in range(N):
            y = fwd(ids[:, P + i:P + i + 1].to(dev), params(P + i))
            dec.append(y[0, -1].float().cpu())
        if state is not None:
            state.free()
    return pf, torch.stack(dec), time.time() - t0

def kl(ref, got):
    lr = torch.log_softmax(ref.double(), -1)
    lgt = torch.log_softmax(got.double(), -1)
    return (lr.exp() * (lr - lgt)).sum(-1)

results = {}
for name, cache in caches.items():
    for eager in ([False, True] if not args.no_eager else [False]):
        tag = f"{name}/{'eager' if eager else 'graph'}"
        pf, dec, dt = run(cache, eager)
        results[tag] = (pf, dec)
        print(f" -- {tag}: prefill chunk {pf.shape[0]} + {dec.shape[0]} decode steps in {dt:.1f}s")

ref_pf, ref_dec = results["fp16/graph"]
print(f"\n{'config':<12} | {'prefill KL':>10} {'max':>9} {'argmax':>7} | {'decode KL':>10} {'max':>9} {'argmax':>7}")
for tag, (pf, dec) in results.items():
    kp, kd = kl(ref_pf, pf), kl(ref_dec, dec)
    ap = (ref_pf.argmax(-1) == pf.argmax(-1)).float().mean().item() * 100
    ad = (ref_dec.argmax(-1) == dec.argmax(-1)).float().mean().item() * 100
    print(f"{tag:<12} | {kp.mean().item():>10.2e} {kp.max().item():>9.2e} {ap:>6.1f}% | "
          f"{kd.mean().item():>10.2e} {kd.max().item():>9.2e} {ad:>6.1f}%")
