"""
Reloading one Model object with CPU-offloaded experts (PRs #463 / #464): load, generate, unload(), load(),
generate. The worker object lives on the config and is reused by the second load, so its hand-off state has to
come back to the constructor's values when the last layer unregisters: stale sequence cursors queue a stream wait
on the new worker's zeroed flag block that nothing can satisfy (the first forward never returns), and a stale
worker reference on a module the second load keeps on the GPU indexes the worker's layer list past its end
(autosplit's worst-case reader, so the loads go through autosplit). The reload runs in a subprocess under a timeout so a regression fails instead
of wedging the session. Whole-layer offload (same count, fewer, none, and none first) and the per-layer split,
with a prompt long enough for the streamed prefill tier so both the slot and the staging-ring cursors are used.
"""

import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.models("mul1"), pytest.mark.slow]

SCRIPT = r"""
import sys, torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.generator.sampler import ArgmaxSampler
md, device, mode = sys.argv[1], sys.argv[2], sys.argv[3]
counts = [int(c) for c in sys.argv[4:]]
cfg = Config.from_directory(md)
tok = Tokenizer.from_config(cfg)
model = Model.from_config(cfg)
prompt = "The quick brown fox jumps over the lazy dog, and then " * 20 + "finally"
outs = []
for n in counts:
    if mode == "split": cfg.infer_params.moe_cpu_split = n
    else: cfg.infer_params.moe_cpu_offload = n
    cache = Cache(model, max_num_tokens = 2048)
    # Through the autosplit loader (whose worst-case reader is the stale-reference victim), on this device only
    use = [0.0] * torch.cuda.device_count()
    use[torch.device(device).index] = 1000.0
    model.load(use_per_device = use, progressbar = False)
    gen = Generator(model, cache, tok)
    outs.append(gen.generate(prompt = prompt, max_new_tokens = 16, add_bos = False, sampler = ArgmaxSampler()))
    gen.close()
    cache.detach_from_model(model)
    del gen, cache
    model.unload()
print("RELOAD_OK" if len(set(outs)) == 1 else "RELOAD_DIFFER " + repr(outs))
"""


@pytest.mark.parametrize("mode, counts", [
    ("offload", [4, 4]),
    ("offload", [4, 2]),
    ("offload", [4, 0]),
    ("offload", [0, 4]),
    ("split", [4, 4]),
], ids = ["offload_same", "offload_fewer", "offload_none", "offload_none_first", "split_same"])
def test_reload_same_model_object(model_id, model_dir, device, mode, counts):
    cmd = [sys.executable, "-c", SCRIPT, model_dir, str(device), mode] + [str(c) for c in counts]
    try:
        out = subprocess.run(cmd, capture_output = True, text = True, timeout = 600,
                             env = os.environ | {"PYTHONPATH": os.pathsep.join(sys.path)})
    except subprocess.TimeoutExpired as e:
        pytest.fail(f"reload hung ({mode} {counts}):\n{(e.stdout or b'')[-2000:]}\n{(e.stderr or b'')[-2000:]}")
    assert out.returncode == 0, out.stderr[-3000:]
    assert "RELOAD_OK" in out.stdout, out.stdout[-2000:]
