"""
Fused decode MoE kernels (exl3_moe_coop, the bsz <= MAX_BSZN path of BlockSparseMLP): every
K 1..8 x codebook, mixed K across projections, per-expert biases, all activations (gated and
gateless), padded input/output dims, expert-range masking, shared-expert merge (gated and plain),
fp32 gate/up scratch, and bit-reproducibility, against a torch reference built from the
reconstructed weights.

    TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0" PYTHONPATH=. python -m pytest tests/test_moe_coop.py -q
"""
import os
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext

DEV = torch.device(os.environ.get("EXL3_TEST_DEVICE", "cuda:0"))
ACTS = {"silu": 0, "gelu": 1, "relu2": 2, "swiglu_oai": 3}


def rand_trellis(k, n, K, gen):
    return torch.randint(0, 65536, (k // 16, n // 16, 16 * K), dtype = torch.int32, generator = gen).to(torch.int16).to(DEV)


def rand_scale(n, gen):
    return ((torch.rand((n,), generator = gen) * 0.2 + 0.9) * torch.where(
        torch.rand((n,), generator = gen) < 0.5, -1.0, 1.0)).half().to(DEV)


def dq(trellis, K, cb):
    W = torch.empty((trellis.shape[0] * 16, trellis.shape[1] * 16), dtype = torch.half, device = DEV)
    ext.reconstruct(W, trellis, K, cb == 1, cb == 2)
    return W


class Proj:
    def __init__(self, k, n, K, cb, bias, gen, E):
        self.K, self.cb = K, cb
        self.trellis = [rand_trellis(k, n, K, gen) for _ in range(E)]
        self.suh = [rand_scale(k, gen) for _ in range(E)]
        self.svh = [rand_scale(n, gen) for _ in range(E)]
        self.bias = [(torch.randn((n,), generator = gen) * 0.1).half().to(DEV) for _ in range(E)] if bias else None
        self.W = [dq(t, K, cb) for t in self.trellis]
        tab = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype = torch.long, device = DEV)
        self.tabs = (tab(self.trellis), tab(self.suh), tab(self.svh))
        self.bias_tab = tab(self.bias) if bias else None

    def ref(self, x, e):
        # x: (m, k) half; the quantized linear as the dense path evaluates it (fp32 accumulate)
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, self.suh[e], None, 1.0)
        y = torch.empty((x.shape[0], self.W[e].shape[1]), dtype = torch.half, device = DEV)
        ext.hgemm(xh, self.W[e], y)
        ext.had_r_128(y, y, None, self.svh[e], 1.0)
        y = y.float()
        if self.bias is not None:
            y = y + self.bias[e].float()
        return y


def act_ref(act, gated, g, u, limit):
    if not gated:
        x = torch.relu(u)
    elif act == "silu":
        x = torch.nn.functional.silu(g)
    elif act == "gelu":
        x = torch.nn.functional.gelu(g, approximate = "tanh")
    elif act == "relu2":
        x = torch.relu(g) ** 2
    else:
        if limit:
            g = g.clamp(max = limit); u = u.clamp(-limit, limit)
        return (u + 1.0) * g * torch.sigmoid(1.702 * g)
    if limit:
        u = u.clamp(-limit, limit); x = x.clamp(max = limit)
    return x * u


def run_case(seed, Kg, Ku, Kd, cb, act = "silu", gated = True, bias = False, limit = 0.0,
             H = 256, Hi = 256, I = 256, Ho = 256, H_out = 256, E = 6, topk = 3, bsz = 5,
             min_e = -1, max_e = -1, shared = None, gu_f32 = False, tol = 3e-3, empty_rows = False):
    gen = torch.Generator().manual_seed(seed)
    gate = Proj(Hi, I, Kg, cb, bias, gen, E) if gated else None
    up = Proj(Hi, I, Ku, cb, bias, gen, E)
    down = Proj(I, Ho, Kd, cb, bias, gen, E)
    x = (torch.randn((bsz, H), generator = gen) * 0.05).half().to(DEV)
    n_global = E if min_e < 0 else E + 4    # global expert space with 4 foreign experts
    sel = torch.stack([torch.randperm(n_global, generator = gen)[:topk] for _ in range(bsz)])
    if empty_rows:
        # tokens whose every pick lives on another rank: rows 0 and bsz-1 get foreign experts only
        foreign = [e for e in range(n_global) if not (min_e <= e < max_e)]
        for r in (0, bsz - 1):
            sel[r] = torch.tensor([foreign[k % len(foreign)] for k in range(topk)])
    sel = sel.to(DEV)
    rw = (torch.rand((bsz, topk), generator = gen) * 0.9 + 0.1).half().to(DEV)
    if bsz * topk > 1:
        rw[0, 0] = 0.0     # a zero-weight slot must contribute nothing and not break anything
    first = 0 if min_e < 0 else min_e

    # Reference
    xp = torch.zeros((bsz, Hi), dtype = torch.half, device = DEV)
    xp[:, :H] = x
    ref = torch.zeros((bsz, H_out), dtype = torch.float, device = DEV)
    for b in range(bsz):
        for k in range(topk):
            e = int(sel[b, k]); w = float(rw[b, k])
            if min_e >= 0 and not (min_e <= e < max_e):
                continue
            el = e - first
            u = up.ref(xp[b:b + 1], el)
            g = gate.ref(xp[b:b + 1], el) if gated else u
            a = act_ref(act, gated, g, u, limit).half()
            d = down.ref(a, el)
            ref[b] += w * d[0, :H_out]
    sh_out = sh_w = None
    if shared is not None:
        sh_out = (torch.randn((bsz, H), generator = gen)).float().to(DEV)
        if shared == "gated":
            sh_w = (torch.randn((H,), generator = gen) * 0.05).half().to(DEV)
            gv = torch.sigmoid((x.float() * sh_w.float()).sum(1, keepdim = True))
            ref += gv * sh_out
        else:
            ref += sh_out

    # Kernel. Scratch sized for 4x the slots (as the module does for MAX_BSZN rows) so the
    # launcher can use split-k partial rows
    slots = bsz * topk
    smax = 4 * slots
    gu_dt = torch.float if gu_f32 else torch.half
    gu_g = torch.empty((smax, 1, I), dtype = gu_dt, device = DEV)
    gu_u = torch.empty((smax, 1, I), dtype = gu_dt, device = DEV)
    act_out = torch.empty((smax, 1, I), dtype = torch.half, device = DEV)
    d_out = torch.empty((smax, 1, Ho), dtype = torch.float, device = DEV)
    out = torch.full((8, H_out), float("nan"), dtype = torch.float, device = DEV)
    ctr = torch.zeros((smax * (I // 128) + 8 * (Ho // 128) + 2 * smax + 3,), dtype = torch.int, device = DEV)
    had_g = torch.empty((slots, Hi), dtype = torch.half, device = DEV); had_u = torch.empty_like(had_g)
    gtabs = gate.tabs if gated else up.tabs
    gK = gate.K if gated else up.K
    gbias = gate.bias_tab if gated else None

    def run():
        ext.exl3_moe_coop(
            x, sel, rw, min_e, max_e, Hi,
            *gtabs, *up.tabs, *down.tabs,
            gbias, up.bias_tab, down.bias_tab,
            gK, up.K, down.K, cb == 1, cb == 2,
            ACTS[act], limit, gated,
            had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out[:bsz],
            sh_out, sh_w,
        )
        torch.cuda.synchronize()
        return out[:bsz].clone()

    o1 = run()
    o2 = run()
    assert torch.equal(o1, o2), "not bit-reproducible"
    assert torch.isfinite(o1).all()
    scale = ref.abs().max().item()
    err = (o1 - ref).abs().max().item()
    assert err <= tol * scale, f"max err {err:.3e} vs scale {scale:.3e} (K {Kg}/{Ku}/{Kd} cb {cb} act {act})"
    return err / scale if scale > 0 else err     # an all-empty batch (no local expert) is exactly zero


@pytest.mark.parametrize("cb", [0, 1, 2])
@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6, 7, 8])
def test_bits_and_codebooks(K, cb):
    run_case(K * 10 + cb, K, K, K, cb)


def test_mixed_k():
    # gate and up share a width (the converter allocates them as one group); down may differ
    run_case(1, 3, 3, 5, 1)
    run_case(2, 8, 8, 4, 2)
    run_case(18, 2, 2, 7, 0)


@pytest.mark.parametrize("act", ["silu", "gelu", "relu2", "swiglu_oai"])
def test_activations(act):
    run_case(3, 4, 4, 4, 1, act = act, limit = 0.0)
    run_case(4, 4, 4, 4, 1, act = act, limit = 1.0)


def test_gateless_relu2():
    run_case(5, 4, 4, 4, 2, act = "relu2", gated = False)
    run_case(6, 3, 3, 6, 1, act = "relu2", gated = False, bias = True)   # gate K unused
    # Nemotron-Nano-like: gateless K6/K8 mul1 at hidden 2688 / padded intermediate 1920, expert
    # range masking with empty rows, bsz 1 and 4
    run_case(26, 6, 6, 8, 2, act = "relu2", gated = False, H = 2688, Hi = 2688, I = 1920, Ho = 2688, H_out = 2688,
             E = 6, topk = 6, bsz = 1, min_e = 4, max_e = 10, empty_rows = True)
    run_case(27, 6, 6, 6, 2, act = "relu2", gated = False, H = 2688, Hi = 2688, I = 1920, Ho = 2688, H_out = 2688,
             E = 6, topk = 6, bsz = 4, min_e = 4, max_e = 10, empty_rows = True)


def test_biases_padding_oai():
    # gpt-oss-like: biases on every projection, clamped swiglu, hidden 200 padded to 256 on both
    # the input and output side (Hi = Ho = 256, H = H_out = 200)
    run_case(7, 4, 4, 4, 2, act = "swiglu_oai", limit = 1.0, bias = True, H = 200, H_out = 200)
    run_case(8, 2, 2, 3, 1, bias = True, H = 200, H_out = 200, gated = True)


def test_expert_range_masking():
    # Local tables cover global experts [4, 10); picks outside contribute zero
    run_case(9, 4, 4, 4, 1, min_e = 4, max_e = 10)
    run_case(10, 3, 3, 3, 2, min_e = 4, max_e = 10, bias = True)


def test_empty_rows():
    # Expert-parallel sharding: tokens with no active slot on this rank must still get their row
    # written (zeros, or the shared-expert term), at bsz 1 and bsz > 1, with and without shared experts
    run_case(22, 4, 4, 4, 1, min_e = 4, max_e = 10, bsz = 1, topk = 3, empty_rows = True)
    run_case(23, 4, 4, 4, 1, min_e = 4, max_e = 10, bsz = 5, topk = 3, empty_rows = True)
    run_case(24, 3, 3, 3, 2, min_e = 4, max_e = 10, bsz = 4, topk = 2, empty_rows = True, shared = "gated", H = 200, H_out = 200)
    run_case(25, 2, 2, 5, 1, min_e = 4, max_e = 10, bsz = 1, topk = 2, empty_rows = True, shared = "plain")


@pytest.mark.parametrize("shared", ["plain", "gated"])
def test_shared_expert_merge(shared):
    run_case(11, 4, 4, 4, 1, shared = shared)


def test_fp32_gate_up_scratch():
    run_case(12, 4, 4, 4, 1, gu_f32 = True)


def test_duplicate_experts():
    # few experts, many slots: runs of up to ROWS slots share one expert's weights
    run_case(19, 4, 4, 4, 1, E = 3, topk = 3, bsz = 8)
    run_case(20, 3, 3, 3, 2, E = 2, topk = 2, bsz = 8, bias = True)
    run_case(21, 6, 6, 6, 2, E = 4, topk = 4, bsz = 8, min_e = 1, max_e = 5)


def test_batch_extremes():
    run_case(13, 4, 4, 4, 1, bsz = 1, topk = 1)
    run_case(14, 4, 4, 4, 1, bsz = 8, topk = 8, E = 12)
    run_case(15, 2, 2, 2, 2, bsz = 8, topk = 32, E = 40, H = 512, Hi = 512, I = 384, Ho = 512, H_out = 512)


def test_odd_shapes():
    # k-slices not divisible by the 16-warp split, several column groups
    run_case(16, 4, 4, 4, 1, H = 640, Hi = 640, I = 896, Ho = 640, H_out = 640)
    run_case(17, 6, 6, 6, 2, H = 384, Hi = 384, I = 1152, Ho = 384, H_out = 384)
