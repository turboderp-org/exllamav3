# exl3_mgemm with expert-range filtering: the batched (num_tokens > 1) reduction against the
# per-token (num_tokens == 1) loop, which is the only mode that used to be allowed with
# min_index >= 0. Both arms see the same trellis/scales, the same routing and the same input
# rows, so any slot -> token mixup shows up as a gross mismatch. The shifted-range control
# confirms the comparison actually discriminates.

import torch
from exllamav3.ext import exllamav3_ext as ext

torch.manual_seed(0)
device = "cuda:0"
torch.cuda.set_device(device)

NUM_EXPERTS = 32
H = 2048
I = 1024
K = 4

# (name, first local expert, last local expert)
RANGES = [
    ("all local",        0, 32),
    ("half local",       0, 16),
    ("none local",      32, 33),
    ("offset range",     8, 24),
]


def make_experts(num, k, n):
    trellis, suh, svh = [], [], []
    for _ in range(num):
        t = torch.randint(0, 65536, (k // 16, n // 16, 256 * K // 16), dtype = torch.int32, device = device)
        trellis.append(t.to(torch.short))
        suh.append(torch.sign(torch.randn(k, device = device)).half())
        svh.append(torch.sign(torch.randn(n, device = device)).half())
    return trellis, suh, svh


def ptrs(tensors, first, last):
    return torch.tensor([t.data_ptr() for t in tensors[first:last]], dtype = torch.long, device = device)


class Proj:
    def __init__(self, num, k, n):
        self.trellis, self.suh, self.svh = make_experts(num, k, n)
        self.k, self.n = k, n

    def tables(self, first, last):
        return ptrs(self.trellis, first, last), ptrs(self.suh, first, last), ptrs(self.svh, first, last)


def mgemm(tables, A, C, A_had, indices, weights, first, last, num_tokens):
    pt, ps, pv = tables
    ext.exl3_mgemm(
        A, pt, C, ps, A_had, pv, indices, weights, K, -1, 0, 0,
        first, last, 0, num_tokens, None, None
    )


def moe(gate, up, down, first, last, x, sel, weights, num_tokens):
    """gate/up/act/down over num_tokens * top_k slots, returning the [num_tokens, H] reduction."""
    bszm = sel.numel()
    yh = torch.zeros((bszm, 1, H), dtype = torch.half, device = device)
    interm_g = torch.zeros((bszm, 1, I), dtype = torch.half, device = device)
    interm_u = torch.zeros((bszm, 1, I), dtype = torch.half, device = device)
    interm_a = torch.zeros((bszm, 1, I), dtype = torch.half, device = device)
    out_d = torch.zeros((bszm, 1, H), dtype = torch.float, device = device)
    idx = sel.reshape(1, -1)
    w = weights.reshape(1, -1)
    mgemm(gate.tables(first, last), x, interm_g, yh, idx, None, first, last, num_tokens)
    mgemm(up.tables(first, last), x, interm_u, yh, idx, None, first, last, num_tokens)
    ext.silu_mul(interm_g, interm_u, interm_a, 0.0)
    mgemm(down.tables(first, last), interm_a, out_d, interm_g, idx, w, first, last, num_tokens)
    return out_d[:num_tokens, 0, :].clone()


def batched(gate, up, down, first, last, x, sel, weights):
    """One call per projection over all slots, with each slot's own input row."""
    num_tokens, top_k = sel.shape
    rows = torch.arange(num_tokens, device = device).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)
    xg = x.index_select(0, rows).view(-1, 1, H).contiguous()
    return moe(gate, up, down, first, last, xg, sel, weights, num_tokens)


def per_token(gate, up, down, first, last, x, sel, weights):
    """Reference: one num_tokens == 1 call per token, the broadcast input the kernel already had."""
    num_tokens = sel.shape[0]
    out = torch.zeros((num_tokens, H), dtype = torch.float, device = device)
    for t in range(num_tokens):
        xt = x[t].view(1, 1, H).contiguous()
        out[t] = moe(gate, up, down, first, last, xt, sel[t:t+1], weights[t:t+1], 1)[0]
    return out


def routing(num_tokens, top_k, seed):
    g = torch.Generator().manual_seed(seed)
    sel = torch.stack([torch.randperm(NUM_EXPERTS, generator = g)[:top_k] for _ in range(num_tokens)])
    w = torch.rand((num_tokens, top_k), generator = g)
    w = w / w.sum(-1, keepdim = True)
    return sel.long().to(device), w.half().to(device)


def rel_err(a, b):
    scale = b.abs().max().item()
    err = (a - b).abs().max().item()
    return err / scale if scale > 0 else err


def main():
    gate = Proj(NUM_EXPERTS + 1, H, I)
    up = Proj(NUM_EXPERTS + 1, H, I)
    down = Proj(NUM_EXPERTS + 1, I, H)

    for name, first, last in RANGES:
        for num_tokens in [1, 2, 4, 8, 32, 128, 256]:
            for top_k in [1, 2, 6, 8]:
                sel, w = routing(num_tokens, top_k, num_tokens * 100 + top_k)
                x = (torch.randn((num_tokens, H), device = device) * 0.05).half()
                a = batched(gate, up, down, first, last, x, sel, w)
                b = per_token(gate, up, down, first, last, x, sel, w)
                live = int(((sel >= first) & (sel < last)).sum().item())
                assert torch.isfinite(a).all(), f"{name} bsz={num_tokens} k={top_k}: non-finite"
                rel = rel_err(a, b)
                assert rel < 5e-3, f"{name} bsz={num_tokens} k={top_k}: rel {rel:.2e}"
                print(f"  PASS {name:13s} bsz={num_tokens:4d} top_k={top_k} "
                      f"live={live:5d}/{num_tokens * top_k:5d}: rel {rel:.2e}")

    # Discrimination check: the same call against a shard whose range is off by one expert
    for first, last in [(0, 16), (8, 24)]:
        for num_tokens in [8, 32]:
            sel, w = routing(num_tokens, 6, num_tokens * 100 + 6)
            x = (torch.randn((num_tokens, H), device = device) * 0.05).half()
            ref = per_token(gate, up, down, first, last, x, sel, w)
            good = rel_err(batched(gate, up, down, first, last, x, sel, w), ref)
            bad = rel_err(batched(gate, up, down, first + 1, last + 1, x, sel, w), ref)
            assert bad > 100 * good, f"control [{first},{last}) bsz={num_tokens}: {bad:.2e} vs {good:.2e}"
            print(f"  PASS control  [{first:2d},{last:2d}) bsz={num_tokens:4d}: "
                  f"matched {good:.2e}, shifted {bad:.2e}")

    print("ALL PASS")


if __name__ == "__main__":
    main()
