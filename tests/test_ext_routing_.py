"""
Reference tests for the MoE routing kernels (routing.cu): top-K selection and weight normalization for the
std-softmax and DS3/DSv4 (sigmoid / sqrt-softplus, optional selection bias) routers, both the iterative
top-K kernels and the radix-sort variants, over the whole (num_experts, K) range. Inputs are tie-free by
construction so the selected set is unambiguous. Many rows per launch so the multi-warp merge stages run
concurrently across blocks (the merge stages had shared-memory read/write races without a barrier).
"""
import sys, os
import pytest
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.ext import exllamav3_ext as ext

device = "cuda:0"
ROWS = 1024
CONFIGS = [(e, k) for e in (32, 64, 96, 128, 160, 256, 384, 512) for k in (1, 2, 4, 6, 8, 10, 16) if k <= e]


def tie_free_scores(rows, num_experts, gen):
    # distinct fp16 values per row: a permutation scaled so neighbouring values differ by more than the fp16 ulp
    perm = torch.stack([torch.randperm(num_experts, generator = gen) for _ in range(rows)])
    return (perm.float() * 0.0173 - 3.5).half().to(device)


def check(idx, w, ref_idx, ref_w, label, keys = None, tie_tol = 1e-5):
    """keys: the per-expert selection key used by the reference (None = the scores themselves, tie-free by
    construction, so the sets must match exactly). With keys given, a row whose set differs is accepted only
    if every kernel-selected key is within tie_tol of the K-th largest reference key (a rounding-order near
    tie); a race or a wrong selection puts a key far below that threshold."""
    assert idx.dtype == torch.int64 and w.dtype == torch.half
    i1, o1 = idx.sort(dim = 1); i2, o2 = ref_idx.sort(dim = 1)
    bad = (i1 != i2).any(dim = 1)
    if bad.any():
        assert keys is not None, f"{label}: selected expert set differs in {bad.sum().item()} rows"
        K = idx.shape[1]
        kth = keys.topk(K, dim = 1).values[:, -1:]
        sel_keys = keys.gather(1, idx)
        low = (sel_keys < kth - tie_tol) & bad[:, None]
        assert not low.any(), f"{label}: {low.any(dim = 1).sum().item()} rows selected experts below the top-K threshold (not a near tie)"
        assert bad.sum().item() <= max(2, idx.shape[0] // 256), f"{label}: too many near-tie rows ({bad.sum().item()})"
    ok = ~bad
    w1 = w.float().gather(1, o1)[ok]; w2 = ref_w.float().gather(1, o2)[ok]
    err = (w1 - w2).abs().max().item() if ok.any() else 0.0
    assert err < 4e-3, f"{label}: max weight error {err}"


def ref_std(scores, K, per_expert_scale):
    s = scores.float()
    top_v, top_i = s.topk(K, dim = 1)
    w = F.softmax(top_v, dim = 1)
    if per_expert_scale is not None:
        w = w * per_expert_scale.float()[top_i]
    return top_i, w.half()


def ref_ds3(scores, K, bias, scaling, act):
    s = scores.float()
    a = torch.sigmoid(s) if act == 0 else F.softplus(s).sqrt()
    key = a + bias.float() if bias is not None else s
    _, top_i = key.topk(K, dim = 1)
    sel = a.gather(1, top_i)
    w = sel * scaling / sel.sum(dim = 1, keepdim = True)
    return top_i, w.half(), key


@pytest.mark.parametrize("use_topk", [True, False])
@pytest.mark.parametrize("scaled", [False, True])
def test_routing_std(use_topk, scaled):
    gen = torch.Generator().manual_seed(1)
    for E, K in CONFIGS:
        for it in range(6):
            scores = tie_free_scores(ROWS, E, gen)
            pes = (torch.rand(E, generator = gen) + 0.5).bfloat16().to(device) if scaled else None
            idx = torch.empty((ROWS, K), dtype = torch.long, device = device)
            w = torch.empty((ROWS, K), dtype = torch.half, device = device)
            ext.routing_std_logits(scores, idx, w, pes, use_topk)
            ri, rw = ref_std(scores, K, pes)
            check(idx, w, ri, rw, f"std E={E} K={K} topk={use_topk} scaled={scaled} it={it}")


@pytest.mark.parametrize("act", [0, 1])
@pytest.mark.parametrize("with_bias", [False, True])
def test_routing_ds3_topk(act, with_bias):
    gen = torch.Generator().manual_seed(2)
    for E, K in CONFIGS:
        for it in range(6):
            scores = tie_free_scores(ROWS, E, gen)
            bias = ((torch.rand(E, generator = gen) - 0.5) * 0.2).half().to(device) if with_bias else None
            idx = torch.empty((ROWS, K), dtype = torch.long, device = device)
            w = torch.empty((ROWS, K), dtype = torch.half, device = device)
            ext.routing_ds3_nogroup_logits(scores, bias, idx, w, 2.5, True, act)
            ri, rw, keys = ref_ds3(scores, K, bias, 2.5, act)
            check(idx, w, ri, rw, f"ds3-topk E={E} K={K} act={act} bias={with_bias} it={it}", keys = keys if with_bias else None)


@pytest.mark.parametrize("act", [0, 1])
def test_routing_ds3_radix_no_bias(act):
    gen = torch.Generator().manual_seed(3)
    for E, K in CONFIGS:
        scores = tie_free_scores(ROWS, E, gen)
        idx = torch.empty((ROWS, K), dtype = torch.long, device = device)
        w = torch.empty((ROWS, K), dtype = torch.half, device = device)
        ext.routing_ds3_nogroup_logits(scores, None, idx, w, 1.0, False, act)
        ri, rw, _ = ref_ds3(scores, K, None, 1.0, act)
        check(idx, w, ri, rw, f"ds3-radix E={E} K={K} act={act}")
