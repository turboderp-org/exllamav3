"""
ext.blocksparse_mlp_routing (libtorch/blocksparse_mlp.cpp), the libtorch form of standard softmax top-k routing:
logits = y @ cfg.gate_tensor, the experts with the top num_experts_per_tok logits (every expert when
params["activate_all_experts"]), weights = softmax over the selected logits. Returns (selected, weights). The order
of the selection is not part of the contract (the bsz-1 path takes torch's unsorted top-k, the others the sorted
one); the weights follow the selection.

- bsz == 1 (and not activate_all) runs into the RoutingCFG's bsz-1 statics (router_logits_bsz1 receives the logits)
  and returns those same tensors (CUDA-graph callers bake their addresses)
- other batch sizes return fresh tensors and leave the statics untouched

Reference: the logits are torch's own fp16 matmul (the function calls at::matmul), so selection is checked against
the top-k of those logits (exact: fp16 values) and against the float64 product (selected logits within the fp16
projection error of the K-th largest); weights against the float64 softmax of the selected logits, to fp16 output
rounding (2^-10 relative) plus 1e-4 absolute for the fp16 softmax arithmetic.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.routing import make_cfg


def _check(y, gate, sel, w, K):
    logits = torch.matmul(y, gate).double()
    assert sel.shape == (y.shape[0], K) and w.shape == (y.shape[0], K)
    lv = logits.gather(1, sel)
    assert (sel.sort(dim = 1).values.diff(dim = 1) != 0).all(), "duplicate experts"
    assert torch.equal(lv.sort(dim = 1, descending = True).values, logits.topk(K, dim = 1).values), \
        "not the top-K logits"
    ref64 = y.double() @ gate.double()
    kth = ref64.topk(K, dim = 1).values[:, -1:]
    assert (ref64.gather(1, sel) >= kth - 4e-3).all()
    wr = torch.softmax(lv, dim = 1)
    assert ((w.double() - wr).abs() <= 2.0 ** -10 * wr + 1e-4).all()


@pytest.mark.parametrize("E, K", [(8, 2), (64, 6), (128, 8), (256, 8)])
@pytest.mark.parametrize("bsz", [1, 3, 64])
@torch.inference_mode()
def test_topk_softmax(device, E, K, bsz):
    hdim = 1024
    cfg = make_cfg(hdim, E, K, device, seed = E + bsz)
    y = torch.randn((bsz, hdim), generator = torch.Generator().manual_seed(bsz)).half().to(device)
    for t in (cfg.router_logits_bsz1, cfg.routing_weights_bsz1, cfg.selected_experts_bsz1):
        t.fill_(-3)
    statics = [t.clone() for t in (cfg.router_logits_bsz1, cfg.routing_weights_bsz1, cfg.selected_experts_bsz1)]
    sel, w = ext.blocksparse_mlp_routing(bsz, cfg, y, {})
    _check(y, cfg.gate_tensor, sel, w, K)
    if bsz == 1:
        assert sel.data_ptr() == cfg.selected_experts_bsz1.data_ptr()
        assert w.data_ptr() == cfg.routing_weights_bsz1.data_ptr()
        assert torch.equal(cfg.router_logits_bsz1, torch.matmul(y, cfg.gate_tensor))
    else:
        for a, b in zip(statics, (cfg.router_logits_bsz1, cfg.routing_weights_bsz1, cfg.selected_experts_bsz1)):
            assert torch.equal(a, b), "bsz > 1 touched the bsz-1 statics"


@pytest.mark.parametrize("bsz", [1, 5])
@torch.inference_mode()
def test_activate_all(device, bsz):
    hdim, E, K = 512, 32, 4
    cfg = make_cfg(hdim, E, K, device, seed = bsz)
    y = torch.randn((bsz, hdim), generator = torch.Generator().manual_seed(bsz)).half().to(device)
    sel, w = ext.blocksparse_mlp_routing(bsz, cfg, y, {"activate_all_experts": True})
    _check(y, cfg.gate_tensor, sel, w, E)
    assert sel.data_ptr() != cfg.selected_experts_bsz1.data_ptr()


@pytest.mark.parametrize("bsz", [0, 1, 3])
@torch.inference_mode()
def test_empty(device, bsz):
    """No tokens: empty (0, K) results. An empty hidden dim makes every logit the empty sum (zero), so any K
    distinct experts are a valid selection, all weighted 1 / K. Selecting from no experts raises (torch's top-k)"""
    E, K = 16, 4
    cfg = make_cfg(64, E, K, device)
    cfg.gate_tensor = torch.empty((0, E), dtype = torch.half, device = device)
    sel, w = ext.blocksparse_mlp_routing(bsz, cfg, torch.empty((bsz, 0), dtype = torch.half, device = device), {})
    assert sel.shape == (bsz, K) and w.shape == (bsz, K)
    assert (sel.sort(dim = 1).values.diff(dim = 1) != 0).all()
    assert torch.allclose(w.float(), torch.full_like(w.float(), 1 / K))
    if bsz == 0:
        cfg = make_cfg(64, E, K, device)
        sel, w = ext.blocksparse_mlp_routing(0, cfg, torch.empty((0, 64), dtype = torch.half, device = device), {})
        assert sel.shape == (0, K) and w.shape == (0, K)
    cfg = make_cfg(64, 0, K, device)
    with pytest.raises(RuntimeError):
        ext.blocksparse_mlp_routing(max(bsz, 2), cfg, torch.zeros((max(bsz, 2), 64), dtype = torch.half,
                                                                  device = device), {})
