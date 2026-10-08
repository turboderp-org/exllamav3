import torch, sys, os
sys.path.insert(0, "/home/nvidia/Dev/exllamav3-sm70")
os.environ["EXL3_BC_ATTN"] = "0"
os.environ["EXL3_BC_DSA"] = "0"
import exllamav3
from exllamav3 import Config, Model
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

model_dir = "/home/nvidia/Dev/model/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw"
config = Config.from_directory(model_dir)
model = Model.from_config(config)
model.load()

bsm = None
for m in model.modules:
    mlp = getattr(m, "mlp", None)
    if isinstance(mlp, BlockSparseMLP):
        bsm = mlp
        break

bsz = 3
H = bsm.hidden_size
torch.manual_seed(42)
x = torch.randn(bsz, 1, H, dtype = torch.half, device = bsm.device) * 0.1
params = {"input_ids": torch.tensor([[0], [0], [0]], device = bsm.device)}

out = bsm.forward(x, params)
se, rw = bsm.routing_fn(bsz, bsm.routing_cfg, x, params)

o = torch.zeros(bsz, H, dtype = torch.float, device = bsm.device)
for t in range(bsz):
    for pos, e_idx in enumerate(se[t].tolist()):
        w = rw[t, pos]
        xc = x.view(bsz, H)[t:t+1]
        u = bsm.ups[e_idx].forward(xc, params)
        g = bsm.gates[e_idx].forward(xc, params)
        a = u if bsm.interm_dtype == torch.half else torch.empty_like(u, dtype = bsm.interm_dtype)
        bsm.activation_fn_call(g, u, a, bsm.act_limit)
        d = bsm.downs[e_idx].forward(a, params)
        o[t] += d.view(H).float() * w
if bsm.shared_experts is not None:
    sh = bsm.shared_experts.forward(x, params)
    o += sh.view(bsz, H).float()

o_f = o.view(bsz, -1)
r_f = out.view(bsz, -1).float()
ok = True
for t in range(bsz):
    a, b = o_f[t], r_f[t]
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    rel = ((a - b).norm() / b.norm()).item()
    print(f"token {t}: corr {corr:.6f} rel-RMS {rel:.6f}")
    ok = ok and corr > 0.9999
print("PASS" if ok else "FAIL")
