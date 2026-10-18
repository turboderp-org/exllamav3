"""
Disk page cache tier: end-to-end test with a real model. Each phase is a separate process, because the test
examines what remains after a restart or a kill. Use a hybrid model (for example Qwen3.5/3.6/3.8 or
GLM-5.3-Flash) to include the recurrent checkpoints.

Phases:

  ref   disk tier off, every case on a fresh generator (fully cold reference)
  a     disk tier on, cold: run P then PS, flush, close
  b1    new process, disk tier on: P (restored from disk) then PS
  b2    new process, disk tier on: PS first (restore up to the deepest checkpoint inside P, replay the rest)
  kill  disk tier on, cold: start L and SIGKILL the process during the prefill (only the writes of
        Generator.write_behind can be on disk)
  b3    new process, disk tier on: L (resume from what the killed process wrote)

Environment: USE_PER_DEVICE (GPU split), KILL_AT (prefill position of the kill), CKPT_INTERVAL (checkpoint
spacing on disk, 0 keeps all), WRITE_INTERVAL (write-behind interval in seconds), DISK_CAP_MB.

Usage: python tests/tools/disk_cache_restart_e2e.py <phase> <model_dir> <disk_dir> <out_dir>
Then:  python tests/tools/disk_cache_restart_e2e.py compare - - <out_dir>
"""
import sys, os, json, time, glob, signal
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import torch

phase, model_dir, disk_dir, out_dir = sys.argv[1:5]
os.makedirs(out_dir, exist_ok = True)

if phase == "compare":
    def load(p):
        f = os.path.join(out_dir, p + ".pt")
        return torch.load(f) if os.path.exists(f) else None
    ref = load("ref")
    assert ref is not None, "no cold reference: run phase ref first"
    ok = True
    found = 0
    for p in ("a", "b1", "b2", "b3"):
        r = load(p)
        if r is None:
            continue
        found += 1
        for case in r["cases"]:
            c = r["cases"][case]
            rc = ref["cases"][case]
            x, y = c["logits0"].float(), rc["logits0"].float()
            assert torch.equal(torch.isfinite(x), torch.isfinite(y))
            m = torch.isfinite(x)
            d = (x[m] - y[m]).abs().max().item()
            same = c["tokens"] == rc["tokens"]
            first_div = next((i for i, (u, v) in enumerate(zip(c["tokens"], rc["tokens"])) if u != v), None)
            print(f"{p:6} {case:3}  time {c['time']:7.2f}s (ref {rc['time']:7.2f}s)  "
                  f"logit0 maxdiff {d:.4f}  tokens {'SAME' if same else f'DIFF@{first_div}'}  "
                  f"cached_pages {c['cached_pages']}/{c['total_pages']}")
            ok &= same
        if r.get("disk_metrics"):
            print(f"{p:6} metrics: {json.dumps(r['disk_metrics'])}")
    assert found, "no results to compare"
    print("PASS" if ok else "TOKENS DIFFER (inspect logit diffs)")
    sys.exit(0)

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler

config = Config.from_directory(model_dir)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 32768)
model.load(use_per_device = [float(x) for x in os.environ.get("USE_PER_DEVICE", "17,12").split(",")], progressbar = False)
tokenizer = Tokenizer.from_config(config)

# Deterministic corpus: library sources in sorted order
text = ""
for f in sorted(glob.glob(os.path.join(REPO, "exllamav3", "**", "*.py"), recursive = True)):
    if "disk_cache" in f:
        continue  # keep the corpus stable while this file changes
    text += open(f).read()
    if len(text) > 400000:
        break
ids = tokenizer.encode(text, add_bos = False)
assert ids.shape[-1] >= 56000
P = ids[:, :12000]
PS = ids[:, :13500]
L = ids[:, :30000]
cases = {
    "ref": [("P", P), ("PS", PS), ("L", L)],
    "a": [("P", P), ("PS", PS)],
    "b1": [("P", P), ("PS", PS)],
    "b2": [("PS", PS)],
    "kill": [("L", L)],
    "b3": [("L", L)],
}[phase]

use_disk = phase != "ref"
kill_at = int(os.environ.get("KILL_AT", "20000"))

def make_gen():
    return Generator(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
        cpu_cache_size = 2 * 1024**3,
        recurrent_cache_size = 8 * 1024**3,
        recurrent_checkpoint_interval_pp = 2048,
        disk_cache_dir = disk_dir if use_disk else None,
        disk_cache_size = int(os.environ.get("DISK_CAP_MB", "8192")) * 1024**2,
        disk_cache_checkpoint_interval = int(os.environ.get("CKPT_INTERVAL", "0")),
        disk_cache_write_interval = float(os.environ.get("WRITE_INTERVAL", "2.0")),
    )

def run(gen, input_ids, kill = False):
    job = Job(input_ids = input_ids, max_new_tokens = 32, sampler = ArgmaxSampler(), return_logits = True)
    torch.cuda.synchronize()
    t0 = time.time()
    gen.enqueue(job)
    tokens = []
    logits = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if kill and r["stage"] == "prefill" and r["curr_progress"] >= kill_at:
                dc = gen.disk_page_cache
                print(f"KILL at prefill {r['curr_progress']} after {time.time() - t0:.2f}s: "
                      f"written {dc.metrics['page_writes']} pages and {dc.metrics['state_writes']} checkpoints, "
                      f"queued {len(dc.pending_pages)}", flush = True)
                os.kill(os.getpid(), signal.SIGKILL)
            if r["stage"] == "streaming":
                if "token_ids" in r:
                    tokens += r["token_ids"].view(-1).tolist()
                if "logits" in r:
                    logits.append(r["logits"])
    t = time.time() - t0
    lg = torch.cat(logits, dim = 1)
    return {
        "tokens": tokens,
        "logits0": lg[0, 0].cpu(),
        "time": t,
        "cached_pages": job.cached_pages,
        "total_pages": job.total_pages,
    }

# Warm up kernels on a throwaway generator so the first timed case isn't paying for autotuning
warm = Generator(model = model, cache = cache, tokenizer = tokenizer)
run(warm, ids[:, 50000:54000])
del warm

res = {"cases": {}}
gen = make_gen()
if use_disk:
    print(f"disk tier: {gen.disk_page_cache.dir}  pages on disk at start: {len(gen.disk_page_cache.pages)}")
for name, inp in cases:
    if phase == "ref":
        gen = make_gen()
    res["cases"][name] = r = run(gen, inp, kill = phase == "kill")
    print(f"{phase} {name}: {r['time']:.2f}s cached {r['cached_pages']}/{r['total_pages']} "
          f"text {tokenizer.decode(torch.tensor(r['tokens']))!r:.80}")

if use_disk:
    dc = gen.disk_page_cache
    t0 = time.time()
    while gen.has_idle_work():
        gen.idle_work()
    stats = gen.get_cache_stats()
    pt_metrics = dict(gen.pagetable.metrics)
    cpu_metrics = dict(gen.cpu_page_cache.metrics)
    gen.close()
    print(f"flush+close {time.time() - t0:.2f}s")
    res["disk_metrics"] = dc.metrics
    res["pt_metrics"] = pt_metrics
    print("disk metrics", dc.metrics)
    print("pagetable metrics", pt_metrics)
    print("cpu tier metrics", cpu_metrics)
    print(f"disk tier: {stats['disk_pages']} pages, {stats['disk_states']} checkpoints, "
          f"{stats['disk_used_bytes'] / 1024**2:.0f} MB")

torch.save(res, os.path.join(out_dir, phase + ".pt"))
