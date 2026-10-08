"""
Map Triton JIT (re)compiles and AOT graph-kernel compiles during a chat-like sequence of
prompts (default: "hello", then ~100k, ~10k, ~50k needle-in-haystack turns appended to the same
conversation, mirroring examples/chat.py /nihs). Every compile is logged with the kernel name,
the constexpr values and the integer-specialization attributes (divisible-by-16 / equals-1)
Triton keyed it on, plus the compile time, attributed to the generator iterate() in which it
happened, so stalls can be traced to the argument that changed.

    python tests/probe_triton_specialization_.py -m /mnt/str/models/glm5.3-flash/exl3/2.05bpw
"""
import sys, os, time, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "examples"))
import torch

p = argparse.ArgumentParser()
p.add_argument("-m", "--model_dir", required = True)
p.add_argument("-t", "--turns", default = "0,100000,10000,50000")
p.add_argument("-d", "--device", default = "cuda:0")
p.add_argument("-cq", "--cache_quant", type = int, default = 0)
p.add_argument("--chunk", type = int, default = 2048)
p.add_argument("--new_tokens", type = int, default = 8)
p.add_argument("--raw", action = "store_true",
               help = "random token ids through model.forward (chunked prefill + decode steps) instead of the "
                      "Generator; for stubs without a tokenizer / chat template")
args = p.parse_args()

# ---- Triton JIT compile logging -----------------------------------------------------------------
import triton
from triton.runtime.jit import JITFunction
compiles = []          # (turn, iterate_idx, name, secs, constexprs, attrs)
cur = {"turn": -1, "it": -1}
_orig_do_compile = JITFunction._do_compile

def _do_compile(self, key, signature, device, constexprs, options, attrs, warmup):
    t0 = time.perf_counter()
    r = _orig_do_compile(self, key, signature, device, constexprs, options, attrs, warmup)
    dt = time.perf_counter() - t0
    names = [prm.name for prm in self.params]
    ce = {names[k[0]] if isinstance(k, tuple) else k: v for k, v in constexprs.items()}
    sp = {}
    for k, v in attrs.items():
        nm = names[k[0]] if isinstance(k, tuple) else k
        sp[nm] = ",".join(a[0].replace("tt.", "") + ("" if len(a) < 2 else f"={a[1]}") for a in v)
    compiles.append((cur["turn"], cur["it"], self.fn.__name__, dt, ce, sp))
    return r
JITFunction._do_compile = _do_compile

# AOT kernels compiled for the C++ graphs (bc_attn._compile_kernel and friends)
from exllamav3.modules.attention_fn import bc_attn
_orig_ck = bc_attn._compile_kernel
def _ck(device, fn, signature, constexprs, num_warps, num_stages, **kwargs):
    n0 = len(bc_attn._kernel_cache)
    t0 = time.perf_counter()
    r = _orig_ck(device, fn, signature, constexprs, num_warps, num_stages, **kwargs)
    if len(bc_attn._kernel_cache) != n0:
        compiles.append((cur["turn"], cur["it"], "AOT:" + fn.__name__, time.perf_counter() - t0, dict(constexprs), {}))
    return r
bc_attn._compile_kernel = _ck
for modname in ("bc_mla", "bc_dsa"):
    try:
        mod = __import__(f"exllamav3.modules.attention_fn.{modname}", fromlist = ["x"])
        if getattr(mod, "_compile_kernel", None) is _orig_ck:
            mod._compile_kernel = _ck
    except Exception:
        pass

# ---- model -------------------------------------------------------------------------------------
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler import ArgmaxSampler
from chat_util import make_haystack_prompt
import random
random.seed(0)

turns = [int(x) for x in args.turns.split(",")]
total = sum(turns) + 4096
cache_len = -(-total // 256) * 256
config = Config.from_directory(args.model_dir)
model = Model.from_config(config)
tok = None if args.raw else Tokenizer.from_config(config)   # stubs may ship no tokenizer
if args.cache_quant:
    cache = Cache(model, max_num_tokens = cache_len, max_batch_size = 1, layer_type = CacheLayer_quant,
                  k_bits = args.cache_quant, v_bits = args.cache_quant)
else:
    cache = Cache(model, max_num_tokens = cache_len, max_batch_size = 1)
model.load(args.device, progressbar = True)
gen = Generator(model, cache, tok, max_batch_size = 1, max_chunk_size = args.chunk)

if args.raw:
    # Teacher-forced replay: each turn appends n random tokens and is prefilled in chunks over
    # the accumulated context, then new_tokens single-token steps
    torch.manual_seed(0)
    vocab = config.vocab_size
    total_ids = torch.randint(0, vocab, (1, sum(max(t, 6) for t in turns) + args.new_tokens * len(turns) + 8), dtype = torch.long)
    pos = 0
    with torch.inference_mode():
        state = cache.get_new_state() if cache.recurrent_layers else None
        for ti, n in enumerate(turns):
            n = max(n, 6)
            cur["turn"] = ti
            n_before = len(compiles)
            t_turn = time.perf_counter()
            it = 0
            stalls = []
            def step(a, b, stage):
                global it
                cur["it"] = it
                params = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, cache_len), "past_len": a}
                if state is not None:
                    params["recurrent_states"] = [state]
                torch.cuda.synchronize(); t0 = time.perf_counter()
                model.forward(total_ids[:, a:b].to(args.device), params)
                torch.cuda.synchronize(); dt = time.perf_counter() - t0
                n_new = len(compiles) - n_before - sum(x[4] for x in stalls)
                stalls.append((it, stage, b, dt, n_new))
                it += 1
            end = pos + n
            for a in range(pos, end, args.chunk):
                step(a, min(a + args.chunk, end), "prefill")
            for i in range(args.new_tokens):
                step(end + i, end + i + 1, "decode")
            pos = end + args.new_tokens
            print(f"\n=== turn {ti}: +{n} tokens (context {pos}), {time.perf_counter() - t_turn:.1f}s, {len(compiles) - n_before} compiles")
            for it_, stage, prog, dt, n_new in stalls:
                if dt > 0.5 or n_new:
                    print(f"    step {it_:>3} {stage:<8} to {prog}: {dt:6.2f}s  {n_new} new compiles")
    turns = []

conv = ""
for ti, n in enumerate(turns):
    text = "hello" if n == 0 else make_haystack_prompt(n, tok)[0]
    conv += f"<|user|>\n{text}<|assistant|>\n"
    ids = tok.encode(conv, add_bos = True, encode_special_tokens = True)
    cur["turn"] = ti
    n_before = len(compiles)
    job = Job(input_ids = ids, max_new_tokens = args.new_tokens, sampler = ArgmaxSampler(), decode_special_tokens = True)
    gen.enqueue(job)
    it = 0
    stalls = []
    out = ""
    t_turn = time.perf_counter()
    while gen.num_remaining_jobs():
        cur["it"] = it
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = gen.iterate()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        stage = results[0].get("stage") if results else "?"
        prog = results[0].get("curr_progress") if results else None
        n_new = len(compiles) - n_before - sum(s[4] for s in stalls)
        stalls.append((it, stage, prog, dt, n_new))
        for r in results:
            out += r.get("text", "")
        it += 1
    print(f"\n=== turn {ti}: prompt {ids.shape[1]} tokens, {time.perf_counter() - t_turn:.1f}s, "
          f"{len(compiles) - n_before} compiles, output {out[:60]!r}")
    for it_, stage, prog, dt, n_new in stalls:
        if dt > 0.5 or n_new:
            print(f"    iterate {it_:>3} {stage:<9} progress {prog}: {dt:6.2f}s  {n_new} new compiles")
    conv += out

print("\n=== compile log (turn, iterate, kernel, seconds, specialization) ===")
for turn, it, name, dt, ce, sp in compiles:
    print(f"t{turn} it{it:<3} {dt:5.2f}s {name}")
    if ce: print(f"        constexpr: {ce}")
    if sp: print(f"        specialize: {sp}")

print("\n=== per-kernel summary: compiles per turn ===")
by = collections.defaultdict(lambda: collections.Counter())
for turn, it, name, dt, ce, sp in compiles:
    by[name][turn] += 1
for name in sorted(by, key = lambda n: -sum(by[n].values())):
    print(f"  {name:<40} " + " ".join(f"t{t}:{by[name][t]}" for t in sorted(by[name])))
