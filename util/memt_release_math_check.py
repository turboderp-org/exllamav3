"""
MemT Phase 1 step 4, math isolation: run our memory math (training.memt.memt_block_forward,
what MemTMemory.fuse calls) on the exact activations the release's MemTMemory saw, captured
by `util/memt_release_reference.py --capture`. No backbone, no quantization: any difference
is memory math / rounding. CPU only.

    python util/memt_release_math_check.py MEMT_EXL3_DIR REF.pt

Each layer is compared three ways against an fp64 evaluation of our math on the same inputs
("truth"): ours as served (bf16 x, fp32 math), the release's output, and ours vs release.
If the release-vs-truth error matches ours-vs-release, the gap is the release's own bf16
rounding and our port is the more exact of the two.
"""

import sys
import torch
from safetensors import safe_open
from exllamav3 import Config
from exllamav3.training.memt import MemTWeights, memt_block_forward

memt_dir, ref_path = sys.argv[1], sys.argv[2]
spec = Config.from_directory(memt_dir).memt.spec
cap = torch.load(ref_path)["capture"]

tensors = {}
with safe_open(f"{memt_dir}/memt_block.safetensors", framework = "pt") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)


def weights(i, dtype = None):
    pfx = f"model.language_model.layers.{i}.self_attn.memt."
    t = {k[len(pfx):]: v for k, v in tensors.items() if k.startswith(pfx)}
    if dtype is not None:
        t = {k: v.to(dtype) for k, v in t.items()}
    return MemTWeights.from_dict(t)


def rel(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


torch.set_grad_enabled(False)
for phase in ("read", "commit"):
    print(f"== {phase}: relative error (Frobenius)")
    print(f"  {'layer':>5s}  {'ours vs rel':>12s}  {'ours vs fp64':>12s}  {'rel vs fp64':>12s}")
    for i in sorted(cap[phase]):
        c = cap[phase][i]
        x = c["hidden"]
        st = c["state_in"]
        wr = c["write_in"]
        commit = c["commit"]
        out, nxt = memt_block_forward(x.to(torch.bfloat16), weights(i), spec, st, commit = commit, write_state = wr)
        out64, nxt64 = memt_block_forward(
            x.double(), weights(i, torch.float64), spec,
            None if st is None else st.double(), commit = commit,
            write_state = None if wr is None else wr.double())
        if phase == "read":
            a, r, t = out, c["read"], out64
        else:
            a, r, t = nxt, c["updated"], nxt64
        print(f"  {i:5d}  {rel(a, r):12.3e}  {rel(a, t):12.3e}  {rel(r, t):12.3e}", flush = True)
