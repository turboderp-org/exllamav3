"""
Full VRAM accounting for a loaded model + generator, at three points: after load, after the
Generator is built, and after a long prompt has been prefilled and a few tokens generated
(steady state). Every live CUDA storage is found through gc and attributed to an owner:

  weights/arena      loader slab blocks (128 MiB) -- used bytes vs the tail left in each block
  weights/other      weight tensors with their own allocation (> 16 MiB, or loaded outside a
                     deferred-load bracket)
  cache              K/V (or latent / pool) cache layers
  recurrent          per-slot recurrent states (GDN/KDA/SWA rings)
  statics            g_tensor_cache entries (graph statics, workspaces), by tag
  generator          tensors owned by the Generator (recurrent-state stash, page tables ...)
  other              everything else (listed by size)

against torch's allocator counters (allocated / reserved / inactive-split) and the driver's
view of the device (CUDA context, cuBLAS/Triton runtime, non-torch allocations).

    python tests/vram_accounting_.py -m /mnt/str/models/glm5.3-flash/exl3/2.05bpw -cs 131072 -p 40000

Reports every device the model occupies (layer splits / TP), then a total.
"""
import sys, os, gc, time, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))
import torch
from exllamav3 import model_init, Generator, Job
from exllamav3.util.tensor import g_tensor_cache
from exllamav3.generator.sampler import ArgmaxSampler
from chat_util import make_haystack_prompt

parser = argparse.ArgumentParser()
model_init.add_args(parser, default_cache_size = 32768)
parser.add_argument("-p", "--prompt_tokens", type = int, default = 40000)
parser.add_argument("-n", "--new_tokens", type = int, default = 32)
parser.add_argument("--chunk", type = int, default = 2048)
parser.add_argument("--recurrent_cache_gb", type = float, default = 4.0)
args = parser.parse_args()

GiB = 1024 ** 3
ARENA_BLOCK = 128 << 20

def walk(obj, depth, out, seen, path = ""):
    """Collect tensors reachable from obj through attributes / containers. depth < 0 means
    unlimited (cycle-protected); records the attribute path of each tensor in out_paths."""
    if id(obj) in seen or (depth == 0):
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        out.append(obj); walk.paths.setdefault(skey(obj), path); return
    if isinstance(obj, (str, bytes, int, float, bool, type)) or obj is None:
        return
    if isinstance(obj, (list, tuple, set)):
        for i, x in enumerate(obj): walk(x, depth - 1, out, seen, f"{path}[{i}]")
    elif isinstance(obj, dict):
        for k, x in obj.items(): walk(x, depth - 1, out, seen, f"{path}[{k!r}]")
    elif hasattr(obj, "__dict__"):
        for k, x in vars(obj).items(): walk(x, depth - 1, out, seen, f"{path}.{k}" if path else f"{type(obj).__name__}.{k}")
walk.paths = {}

def skey(t):
    s = t.untyped_storage()
    return (s.data_ptr(), s.nbytes())

def account(tag, model, cache, gen, dev):
    torch.cuda.synchronize(dev)
    gc.collect()
    all_t = [o for o in gc.get_objects() if isinstance(o, torch.Tensor) and o.is_cuda and o.device == dev]
    # owner sets (storage keys)
    owners = {}
    def mark(ts, label):
        for t in ts:
            if isinstance(t, torch.Tensor) and t.is_cuda and t.device == dev:
                owners.setdefault(skey(t), set()).add(label)
    # arena blocks
    arena_keys = {skey(t) for t in all_t if t.dtype == torch.uint8 and t.dim() == 1 and t.numel() == ARENA_BLOCK}
    # module weights
    wt = []
    seen = {id(model.config), id(getattr(model.config, "stc", None)), id(cache), id(gen)}
    for m in model.modules:
        for sub in m:
            if hasattr(sub, "get_tensors"):
                try: wt += [t for t in sub.get_tensors() if isinstance(t, torch.Tensor)]
                except Exception: pass
            walk(sub, -1, wt, seen)
    mark(wt, "weights")
    for l in cache.layers.values():
        mark(l.get_tensors(), "cache")
    for rl in cache.recurrent_layers.values():
        ts = []
        if hasattr(rl, "get_state_tensors"):
            try: ts += list(rl.get_state_tensors())
            except Exception: pass
        walk(rl, 2, ts, set())
        mark(ts, "recurrent")
    gtc = {}
    for k, (refc, v) in g_tensor_cache.cache.items():
        if v.is_cuda and v.device == dev:
            mark([v], "statics"); gtc[skey(v)] = k.split("/")[-1]
    if gen is not None:
        ts = []; walk(gen, 3, ts, {id(model), id(cache)})
        mark(ts, "generator")
    # arena usage: bytes of weight views living inside arena blocks
    arena_used = 0
    seen_views = set()
    for t in wt:
        if t.is_cuda and t.device == dev and skey(t) in arena_keys:
            vk = (t.data_ptr(), t.numel() * t.element_size())
            if vk not in seen_views:
                seen_views.add(vk); arena_used += vk[1]
    tot = collections.Counter(); other = []
    stat_by_tag = collections.Counter(); depth_hist = collections.Counter()
    for k in {skey(t) for t in all_t}:
        nb = k[1]
        if k in arena_keys:
            tot["weights/arena"] += nb; continue
        labs = owners.get(k, set()) - {"generator"} if owners.get(k, set()) - {"generator"} else owners.get(k, set())
        if not labs:
            other.append(k); tot["other"] += nb; continue
        # Precedence: modules reference their cache layers / recurrent states, so the module walk
        # reaches those too; the explicit owners win
        for lab in ("cache", "recurrent", "statics", "generator", "weights"):
            if lab in labs: break
        if lab == "weights":
            lab = "weights/other"
            d = walk.paths.get(k, "")
            depth_hist[min(d.count(".") + d.count("["), 8)] += nb
        tot[lab] += nb
        if lab == "statics": stat_by_tag[gtc.get(k, "?")] += nb
    ms = torch.cuda.memory_stats(dev)
    alloc = ms["allocated_bytes.all.current"]; reserved = ms["reserved_bytes.all.current"]
    inactive = ms.get("inactive_split_bytes.all.current", 0)
    free, total = torch.cuda.mem_get_info(dev)
    used = total - free
    print(f"\n===== {tag} =====")
    print(f"{'weights/arena blocks':<28} {tot['weights/arena'] / GiB:8.2f} GiB  (used {arena_used / GiB:.2f}, tail {(tot['weights/arena'] - arena_used) / GiB:.2f}, {len(arena_keys)} blocks)")
    print(f"{'weights/other':<28} {tot['weights/other'] / GiB:8.2f} GiB")
    print(f"{'cache layers':<28} {tot['cache'] / GiB:8.2f} GiB")
    print(f"{'recurrent states':<28} {tot['recurrent'] / GiB:8.2f} GiB")
    print(f"{'statics (g_tensor_cache)':<28} {tot['statics'] / GiB:8.2f} GiB")
    print(f"{'generator-owned':<28} {tot['generator'] / GiB:8.2f} GiB")
    print(f"{'other / unattributed':<28} {tot['other'] / GiB:8.2f} GiB")
    print(f"{'sum of live storages':<28} {sum(tot.values()) / GiB:8.2f} GiB   torch allocated {alloc / GiB:.2f}")
    print(f"{'allocator cached (free)':<28} {(reserved - alloc - inactive) / GiB:8.2f} GiB   torch reserved {reserved / GiB:.2f}, inactive-split {inactive / GiB:.2f}, segments {ms.get("segment.all.current", 0)}")
    print(f"{'non-torch (ctx/runtime)':<28} {(used - reserved) / GiB:8.2f} GiB   device used {used / GiB:.2f} of {total / GiB:.2f} (driver)")
    if depth_hist:
        print("  direct weights by attribute depth below the block:", ", ".join(f"{d}: {v / GiB:.2f} GiB" for d, v in sorted(depth_hist.items())))
        deep = [(walk.paths.get(k, ""), k[1]) for k in owners if "weights" in owners[k] and k not in arena_keys
                and (walk.paths.get(k, "").count(".") + walk.paths.get(k, "").count("[")) >= 4]
        if deep:
            deep.sort(key = lambda x: -x[1]); print("  deepest examples:", "; ".join(f"{p} {nb / 2**20:.0f} MiB" for p, nb in deep[:3]))
    if stat_by_tag:
        print("  statics by tag:", ", ".join(f"{k} {v / 2**20:.0f} MiB" for k, v in stat_by_tag.most_common(12)))
    if other:
        big = sorted(other, key = lambda k: -k[1])[:8]
        shapes = {}
        for t in all_t:
            k = skey(t)
            if k in big and k not in shapes: shapes[k] = f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')}"
        print("  largest other:", ", ".join(f"{shapes.get(k, '?')} {k[1] / 2**20:.0f} MiB [{walk.paths.get(k, 'no path')}]" for k in big))
        # Who holds them: referrer types and attribute names for a few samples
        shown = 0
        for t in all_t:
            if skey(t) in big[:3] and shown < 3:
                shown += 1
                refs = []
                for r in gc.get_referrers(t):
                    if isinstance(r, dict):
                        keys = [k for k, v in r.items() if v is t]
                        owner = [o for o in gc.get_referrers(r) if hasattr(o, "__dict__") and vars(o) is r]
                        refs.append(f"{type(owner[0]).__name__ if owner else 'dict'}.{keys}")
                    elif isinstance(r, (list, tuple)):
                        owner = [o for o in gc.get_referrers(r) if not isinstance(o, (list, tuple, dict))]
                        refs.append(f"{type(r).__name__} in {[type(o).__name__ for o in owner][:3]}")
                    elif not isinstance(r, type):
                        refs.append(type(r).__name__)
                print(f"    sample {tuple(t.shape)} held by: {refs[:6]}")
    return collections.Counter({
        "weights": tot["weights/arena"] + tot["weights/other"] + tot["other"], "cache": tot["cache"],
        "recurrent": tot["recurrent"], "statics": tot["statics"],
        "allocated": alloc, "reserved": reserved, "used": used,
    })

devs = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
for d in devs:
    torch.cuda.reset_peak_memory_stats(d)
model, config, cache, tokenizer, *_ = model_init.init(args)
# Only devices the model actually occupies
devs = [d for d in devs if torch.cuda.memory_stats(d).get("allocated_bytes.all.current", 0) > 0]

def account_all(tag, model, cache, gen):
    totals = collections.Counter()
    for d in devs:
        totals.update(account(f"{tag} [{d}]", model, cache, gen, d))
    if len(devs) > 1:
        print(f"\n===== {tag} [all {len(devs)} devices] =====")
        for k in ("weights", "cache", "recurrent", "statics", "allocated", "reserved", "used"):
            print(f"{k:<28} {totals[k] / GiB:8.2f} GiB")

account_all("after load", model, cache, None)
gen = Generator(model = model, cache = cache, tokenizer = tokenizer, max_chunk_size = args.chunk,
                recurrent_cache_size = int(args.recurrent_cache_gb * GiB), cpu_cache_size = 0)
account_all("after generator", model, cache, gen)

prompt = make_haystack_prompt(args.prompt_tokens, tokenizer)[0]
ids = tokenizer.encode(prompt, add_bos = True, encode_special_tokens = True)
for d in devs:
    torch.cuda.reset_peak_memory_stats(d)
t0 = time.time()
job = Job(input_ids = ids, max_new_tokens = args.new_tokens, sampler = ArgmaxSampler(), decode_special_tokens = True)
gen.enqueue(job)
while gen.num_remaining_jobs():
    gen.iterate()
torch.cuda.synchronize()
print(f"\nprompt {ids.shape[1]} tokens + {args.new_tokens} generated in {time.time() - t0:.1f}s")
for d in devs:
    ms = torch.cuda.memory_stats(d)
    print(f"  [{d}] peak allocated {ms['allocated_bytes.all.peak'] / GiB:.2f} GiB, peak reserved {ms['reserved_bytes.all.peak'] / GiB:.2f} GiB")
account_all(f"steady state after {ids.shape[1]}-token prompt", model, cache, gen)
