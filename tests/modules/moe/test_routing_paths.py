"""
The two router paths that still built their logits with torch matmuls under tensor parallelism (the gpt-oss biased
softmax router for batched rows, and DeepSeek-V4 chunks with image rows) now compose from the deterministic ext
kernels. Each must reproduce the torch-composed reference selection on rows that are not near-ties, with matching
weights, and be reproducible run to run.
"""

import pytest
import torch
import torch.nn.functional as F

from exllamav3.modules.block_sparse_mlp_routing import routing_std_bias, _routing_sqrtsp_vl

from testlib.routing import clear_rows, make_cfg, same_sel


def test_std_bias_batched(device):
    K, E, topk = 2880, 32, 4
    torch.manual_seed(0)
    bias = (torch.randn(E, device = device) * 0.5).half()
    cfg = make_cfg(K, E, topk, device, bias = bias)
    # Rows must mostly be clear of near-ties for the selection check not to be vacuous. Pooled over the row counts:
    # whether a single row (R = 1) is clear is a coin flip of the random draw
    n_clear = n_rows = 0
    for R in (1, 7, 64, 300):
        y = torch.randn(R, K, device = device).half()
        sel, w = routing_std_bias(R, cfg, y, {})
        sel, w = sel.clone(), w.clone()
        ref_logits = torch.addmm(bias, y, cfg.gate_tensor).float()
        ref_top, ref_sel = torch.topk(ref_logits, topk, dim = -1)
        ref_w = torch.softmax(ref_top, dim = -1)
        clear = clear_rows(ref_logits, topk, 2e-2)
        assert same_sel(sel, ref_sel)[clear].all(), R
        n_clear += int(clear.sum().item())
        n_rows += R
        # weights on agreeing rows (sorted by expert id)
        agree = same_sel(sel, ref_sel)
        ws = torch.gather(w, 1, torch.sort(sel, dim = 1).indices)[agree].float()
        rws = torch.gather(ref_w, 1, torch.sort(ref_sel, dim = 1).indices)[agree]
        assert (ws - rws).abs().max().item() < 5e-3, R
        sel2, w2 = routing_std_bias(R, cfg, y, {})
        assert torch.equal(sel, sel2) and torch.equal(w, w2), R
    assert n_clear / n_rows > 0.5


@pytest.mark.parametrize("hashed", [False, True])
def test_sqrtsp_vl(device, hashed):
    K, E, topk = 2048, 128, 8
    torch.manual_seed(1)
    esb = torch.randn(E, device = device) * 0.1
    esb_vl = torch.randn(E, device = device) * 0.1
    cfg = make_cfg(K, E, topk, device, esb = esb, esb_vl = esb_vl, scaling = 2.5)
    R = 200
    y = torch.randn(R, K, device = device).half()
    vl = torch.rand(R, device = device) < 0.4
    scores = F.softplus(torch.matmul(y.float(), cfg.gate_tensor.float())).sqrt()
    # distinct experts per row, as a hash table gives
    hash_sel = torch.stack([torch.randperm(E, device = device)[:topk] for _ in range(R)]) if hashed else None
    sel, w = _routing_sqrtsp_vl(cfg, y, vl, hash_sel)
    # torch reference (the previous implementation)
    if hash_sel is None:
        b = torch.where(vl.unsqueeze(-1), esb_vl.unsqueeze(0), esb.unsqueeze(0))
        ref_sel = (scores + b).topk(topk, dim = -1).indices
        clear = clear_rows(scores + b, topk, 1e-2)
    else:
        sel_vl = (scores + esb_vl.unsqueeze(0)).topk(topk, dim = -1).indices
        ref_sel = torch.where(vl.unsqueeze(-1), sel_vl, hash_sel)
        clear = clear_rows(scores + esb_vl.unsqueeze(0), topk, 1e-2) | ~vl
    ref_w = scores.gather(1, ref_sel)
    ref_w = ref_w / ref_w.sum(dim = -1, keepdim = True) * cfg.routed_scaling_factor
    assert same_sel(sel, ref_sel)[clear].all()
    assert clear.float().mean().item() > 0.2     # sqrtsp scores are closely spaced
    agree = same_sel(sel, ref_sel)
    ws = torch.gather(w, 1, torch.sort(sel, dim = 1).indices)[agree].float()
    rws = torch.gather(ref_w, 1, torch.sort(ref_sel, dim = 1).indices)[agree]
    assert (ws - rws).abs().max().item() < 1e-2
    sel2, w2 = _routing_sqrtsp_vl(cfg, y, vl, hash_sel)
    assert torch.equal(sel, sel2) and torch.equal(w, w2)
