"""
util/device_copy.to_device: cross-device moves are probed once per device pair (both
directions from one probe), the verdict sticks, a failing pair bounces through host memory,
host<->device moves never probe, and EXLLAMA_NO_P2P_COPY overrides probing in both directions.
Needs two CUDA devices; the P2P failure is simulated by patching the probe.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3.util import device_copy as dc
from exllamav3.util.device_copy import to_device

assert torch.cuda.device_count() >= 2, "needs two CUDA devices"
d0, d1 = torch.device("cuda:0"), torch.device("cuda:1")

def reset():
    dc.reset_verdicts()
    dc.FORCED = None
    for k in dc.stats: dc.stats[k] = 0

# 1. real probe on this box: data survives, exactly one probe for the pair, both directions cached
reset()
a = torch.randn(1000, device = d0)
b = to_device(a, d1)
assert b.device == d1 and torch.equal(a.cpu(), b.cpu())
c = to_device(b, d0)
assert torch.equal(a, c)
assert dc.stats["probes"] == 1, dc.stats
assert set(dc._verdicts) == {(0, 1), (1, 0)}
real_verdict = dict(dc._verdicts)
# host<->device moves never touch the probe machinery
h = to_device(a, "cpu"); assert h.device.type == "cpu"
p = torch.empty(16, pin_memory = True); g = to_device(p, d1, non_blocking = True); assert g.device == d1
assert dc.stats["probes"] == 1
# same-device is a no-op returning the same object, and so is a None destination (TP-side
# modules have no local device; issue seen with MTP + TP)
assert to_device(a, d0) is a
assert to_device(a, None) is a and dc.stats["probes"] == 1

# 2. simulated broken fabric one way: only that direction bounces, the other stays direct
reset()
orig_probe = dc._probe_direct
dc._probe_direct = lambda src, dst: (False, True)      # src->dst corrupt, dst->src fine
x = torch.randn(4096, device = d0)
y = to_device(x, d1)
assert torch.equal(x.cpu(), y.cpu())
assert dc._verdicts == {(0, 1): True, (1, 0): False}, dc._verdicts
assert dc.stats == {"probes": 1, "bounced": 1, "direct": 0}, dc.stats
z = to_device(y, d0)
assert torch.equal(x, z)
assert dc.stats == {"probes": 1, "bounced": 1, "direct": 1}, dc.stats
# verdict sticks: more moves, no more probes
for _ in range(5): to_device(x, d1); to_device(y, d0)
assert dc.stats["probes"] == 1 and dc.stats["bounced"] == 6 and dc.stats["direct"] == 6, dc.stats
# 2b. the reverse direction first: the probe from d1's side stores both verdicts too
reset()
dc._probe_direct = lambda src, dst: (True, False)      # d1->d0 fine, d0->d1 corrupt
to_device(y, d0)
assert dc._verdicts == {(1, 0): False, (0, 1): True}, dc._verdicts
dc._probe_direct = orig_probe

# 3. the real probe agrees with the verdict recorded in 1
reset()
fwd_ok, bwd_ok = dc._probe_direct(d0, d1)
assert (not fwd_ok) == real_verdict[(0, 1)] and (not bwd_ok) == real_verdict[(1, 0)]

# 4. forced modes skip probing entirely
reset(); dc.FORCED = True
to_device(x, d1); to_device(y, d0)
assert dc.stats == {"probes": 0, "bounced": 2, "direct": 0}, dc.stats
reset(); dc.FORCED = False
to_device(x, d1); to_device(y, d0)
assert dc.stats == {"probes": 0, "bounced": 0, "direct": 2}, dc.stats

# 5. the environment variable is read at import: check the three spellings in subprocesses
import subprocess
for env, want in ((None, "None"), ("1", "True"), ("yes", "True"), ("0", "False")):
    e = dict(os.environ); e.pop("EXLLAMA_NO_P2P_COPY", None)
    if env is not None: e["EXLLAMA_NO_P2P_COPY"] = env
    out = subprocess.run([sys.executable, "-c", "from exllamav3.util import device_copy as d; print(d.FORCED)"],
                         env = e, capture_output = True, text = True, cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.stdout.strip() == want, (env, out.stdout, out.stderr[-500:])

print("ok: real verdict on this box", {"0->1 bounce": real_verdict[(0, 1)], "1->0 bounce": real_verdict[(1, 0)]})
