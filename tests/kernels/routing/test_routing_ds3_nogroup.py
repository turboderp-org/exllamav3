"""
ext.routing_ds3_nogroup and ext.routing_sel_norm (routing.cu), the fused router projection + selection/normalization
of the DS3-nogroup / dots (sigmoid) and DeepSeek-V4 (sqrt-softplus) MoE routers, over every projection path the
callers reach: single row with the transposed gate (fixed-order FMA GEMV, even k), single row without it or with odd
k (hgemm), multi-row with the int8 gate tables (deterministic int8 GEMM) and without (hgemm).

Contract, with s = the written scores and act = sigmoid or sqrt(softplus):
- scores (bsz, E) = fp16(hidden @ gate): reference float64 matmul; error <= half an fp16 ulp of the value
  (2^-11 |ref|) plus an accumulation term at fp16-like resolution of the dot product's Cauchy-Schwarz scale,
  2^-15 ||hidden_row|| ||gate_col|| (cuBLAS's reduced-precision split-K reductions on the hgemm path, the int8 hi/lo
  operand split's ~2^-15 resolution on the deterministic path; the FMA GEMV is fp32)
- routing_ds3_nogroup selects K distinct experts maximizing key = act(s) + bias (key = s without a bias), in
  descending key order: without a bias the selected values are exactly the top-K of s (fp16, so exact comparison,
  ties resolved either way); with a bias every selected key is within 1e-5 of the top-K (the kernel's fp32 act
  differs from float64 by a few fp32 ulp)
- weights = act(s_sel) * scaling / sum(act(s_sel)) in the selection order, to fp16 output rounding
  (2^-10 relative) plus a few fp32 ulp
- routing_sel_norm: the same weights for a given selection, order preserved
- rejections: more than 512 experts, K > 32, K > E, wrong output dtypes; routing_sel_norm K > 32
"""

import pytest
import torch
import torch.nn.functional as F

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.block_sparse_mlp_routing import _gate_t
from testlib.routing import make_cfg

CONFIGS = [(16, 4), (64, 6), (128, 8), (160, 1), (256, 8), (384, 22), (512, 32)]
PATHS = ["gemv", "hgemm1", "det", "hgemm"]
ROWS = {"gemv": 1, "hgemm1": 1, "det": 77, "hgemm": 77}


def _act(s: torch.Tensor, act: int) -> torch.Tensor:
    s = s.double()
    return torch.sigmoid(s) if act == 0 else F.softplus(s).sqrt()


def _setup(device, hdim, E, K, path, with_bias, seed):
    bias = ((torch.rand(E, generator = torch.Generator().manual_seed(seed)) - 0.5) * 0.4).half().to(device) \
        if with_bias else None
    cfg = make_cfg(hdim, E, K, device, seed = seed)
    gate_t = _gate_t(cfg)
    rows = ROWS[path]
    hidden = torch.randn((rows, hdim), generator = torch.Generator().manual_seed(seed + 1)).half().to(device)
    if path == "det" and not ext.HAS_DET_GEMM:
        pytest.skip("build without the deterministic routing GEMM")
    opt = {
        "gemv": (gate_t, None, None),
        "hgemm1": (None, None, None),
        "det": (gate_t, cfg.gate_i8, cfg.gate_sb),
        "hgemm": (gate_t, None, None),
    }[path]
    return cfg, hidden, bias, opt


def _check_scores(scores, hidden, gate):
    ref = hidden.double() @ gate.double()
    cs = hidden.double().norm(dim = 1, keepdim = True) * gate.double().norm(dim = 0, keepdim = True)
    err = (scores.double() - ref).abs()
    assert (err <= 2.0 ** -11 * ref.abs() + 2.0 ** -15 * cs).all(), f"scores: max error {err.max().item():.3e}"


def _check_weights(w, s_sel, act, scaling):
    a = _act(s_sel, act)
    ref = a * scaling / (a.sum(dim = 1, keepdim = True) + 1e-20)
    err = (w.double() - ref).abs()
    assert (err <= 2.0 ** -10 * ref.abs() + 1e-6).all(), f"weights: max error {err.max().item():.3e}"


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("act", [0, 1])
@torch.inference_mode()
def test_routing_ds3_nogroup(device, path, with_bias, act):
    hdim = 2048
    for i, (E, K) in enumerate(CONFIGS):
        cfg, hidden, bias, (gate_t, gi8, gsb) = _setup(device, hdim, E, K, path, with_bias, 10 * i + act)
        rows = hidden.shape[0]
        scores = torch.full((rows, E), 99.0, dtype = torch.half, device = device)
        sel = torch.full((rows, K), -7, dtype = torch.long, device = device)
        w = torch.full((rows, K), 99.0, dtype = torch.half, device = device)
        ext.routing_ds3_nogroup(hidden, cfg.gate_tensor, scores, bias, sel, w, 2.5, gate_t, act, gi8, gsb)
        label = f"E={E} K={K}"
        _check_scores(scores, hidden, cfg.gate_tensor)

        assert ((sel >= 0) & (sel < E)).all(), label
        assert (sel.sort(dim = 1).values.diff(dim = 1) != 0).all(), f"{label}: duplicate experts"
        s = scores.double()
        if bias is None:
            s_sel = s.gather(1, sel)
            assert torch.equal(s_sel, s.topk(K, dim = 1).values), f"{label}: not the top-K scores in descending order"
        else:
            keys = _act(s, act) + bias.double()
            k_sel = keys.gather(1, sel)
            kth = keys.topk(K, dim = 1).values[:, -1:]
            assert (k_sel >= kth - 1e-5).all(), f"{label}: selected a key below the top-K"
            assert (k_sel[:, :-1] >= k_sel[:, 1:] - 1e-5).all(), f"{label}: not in descending key order"
        _check_weights(w, scores.gather(1, sel), act, 2.5)


@pytest.mark.parametrize("hdim", [255, 2049])
@torch.inference_mode()
def test_routing_ds3_odd_hidden(device, hdim):
    """Odd k: the single-row GEMV declines and hgemm serves the row"""
    E, K = 64, 6
    cfg = make_cfg(hdim, E, K, device, seed = hdim)
    gate_t = _gate_t(cfg)
    hidden = torch.randn((1, hdim), generator = torch.Generator().manual_seed(hdim)).half().to(device)
    scores = torch.empty((1, E), dtype = torch.half, device = device)
    sel = torch.empty((1, K), dtype = torch.long, device = device)
    w = torch.empty((1, K), dtype = torch.half, device = device)
    ext.routing_ds3_nogroup(hidden, cfg.gate_tensor, scores, None, sel, w, 1.0, gate_t, 0, None, None)
    _check_scores(scores, hidden, cfg.gate_tensor)
    assert torch.equal(scores.double().gather(1, sel), scores.double().topk(K, dim = 1).values)
    _check_weights(w, scores.gather(1, sel), 0, 1.0)


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("act", [0, 1])
@torch.inference_mode()
def test_routing_sel_norm(device, path, act):
    hdim = 2048
    for i, (E, K) in enumerate(CONFIGS):
        cfg, hidden, _, (gate_t, gi8, gsb) = _setup(device, hdim, E, K, path, False, 100 + i)
        rows = hidden.shape[0]
        g = torch.Generator().manual_seed(i)
        selected = torch.stack([torch.randperm(E, generator = g)[:K] for _ in range(rows)]).to(device)
        scores = torch.full((rows, E), 99.0, dtype = torch.half, device = device)
        w = torch.full((rows, K), 99.0, dtype = torch.half, device = device)
        sel0 = selected.clone()
        ext.routing_sel_norm(hidden, cfg.gate_tensor, scores, selected, w, 1.5, gate_t, act, gi8, gsb)
        assert torch.equal(selected, sel0)
        _check_scores(scores, hidden, cfg.gate_tensor)
        _check_weights(w, scores.gather(1, selected), act, 1.5)


@torch.inference_mode()
def test_rejections(device):
    hdim = 256

    def ds3(E, K, idx_dtype = torch.long, w_dtype = torch.half):
        cfg = make_cfg(hdim, E, min(K, E), device)
        hidden = torch.randn((2, hdim), device = device).half()
        ext.routing_ds3_nogroup(hidden, cfg.gate_tensor, torch.empty((2, E), dtype = torch.half, device = device), None,
                                torch.empty((2, K), dtype = idx_dtype, device = device),
                                torch.empty((2, K), dtype = w_dtype, device = device), 1.0, None, 0, None, None)

    ds3(512, 32)
    torch.cuda.synchronize(device)
    with pytest.raises(RuntimeError, match = "Too many experts"):
        ds3(520, 8)
    with pytest.raises(RuntimeError, match = "Too many experts per token"):
        ds3(128, 33)
    with pytest.raises(RuntimeError, match = "K cannot exceed"):
        ds3(16, 17)
    with pytest.raises(RuntimeError):
        ds3(64, 8, idx_dtype = torch.int32)
    with pytest.raises(RuntimeError):
        ds3(64, 8, w_dtype = torch.float)

    cfg = make_cfg(hdim, 64, 8, device)
    hidden = torch.randn((2, hdim), device = device).half()
    with pytest.raises(RuntimeError, match = "K > 32"):
        ext.routing_sel_norm(hidden, cfg.gate_tensor, torch.empty((2, 64), dtype = torch.half, device = device),
                             torch.zeros((2, 33), dtype = torch.long, device = device),
                             torch.empty((2, 33), dtype = torch.half, device = device), 1.0, None, 0, None, None)


# Zero-size inputs of the fused router entry points (routing_ds3_nogroup, routing_sel_norm, and routing_std, which
# shares the projection paths):
# - no tokens (bsz = 0): no-op, scores and outputs untouched
# - no experts (E = 0): top-k over an empty set, raises; K = 0: the weights normalize over an empty selection, raises.
#   Both are checked before the empty-batch return, so they also fire without tokens
# - empty hidden dim: every logit is the empty sum (0) on every projection path; any K distinct experts are a valid
#   selection, all weighted scaling / K

def _fused_route(fn, hidden, gate, scores, idx, w, opt, scaling = 1.5):
    gate_t, gi8, gsb = opt
    if fn == "ds3":
        ext.routing_ds3_nogroup(hidden, gate, scores, None, idx, w, scaling, gate_t, 0, gi8, gsb)
    elif fn == "sel_norm":
        ext.routing_sel_norm(hidden, gate, scores, idx, w, scaling, gate_t, 0, gi8, gsb)
    else:
        ext.routing_std(hidden, gate, scores, idx, w, None, gate_t, None, gi8, gsb)


@pytest.mark.parametrize("fn", ["ds3", "sel_norm", "std"])
@pytest.mark.parametrize("bsz, E, K, error", [
    (0, 16, 4, None),
    (2, 0, 0, "empty expert set"),
    (0, 0, 0, "empty expert set"),
    (2, 16, 0, "K = 0"),
    (0, 16, 0, "K = 0"),
])
@pytest.mark.parametrize("path", ["gemv", "det"])
@torch.inference_mode()
def test_empty(device, fn, bsz, E, K, error, path):
    hdim = 64
    cfg = make_cfg(hdim, E, K, device)
    gate_t = _gate_t(cfg)
    opt = (gate_t, cfg.gate_i8, cfg.gate_sb) if path == "det" else (gate_t, None, None)
    hidden = torch.randn((bsz, hdim), device = device).half()
    scores = torch.full((bsz, E), 99.0, dtype = torch.half, device = device)
    idx = torch.zeros((bsz, K), dtype = torch.long, device = device) if fn == "sel_norm" else \
        torch.full((bsz, K), -7, dtype = torch.long, device = device)
    w = torch.full((bsz, K), 99.0, dtype = torch.half, device = device)
    if error:
        with pytest.raises(RuntimeError, match = error):
            _fused_route(fn, hidden, cfg.gate_tensor, scores, idx, w, opt)
    else:
        _fused_route(fn, hidden, cfg.gate_tensor, scores, idx, w, opt)
    torch.cuda.synchronize(device)
    assert (scores == 99.0).all() and (w == 99.0).all()
    assert (torch.ones(4, device = device) + 1).sum().item() == 8.0, "a later torch op failed after the empty call"


@pytest.mark.parametrize("fn", ["ds3", "sel_norm", "std"])
@pytest.mark.parametrize("path", PATHS)
@torch.inference_mode()
def test_empty_hidden(device, fn, path):
    E, K = 16, 4
    cfg = make_cfg(64, E, K, device)
    cfg.gate_tensor = torch.empty((0, E), dtype = torch.half, device = device)
    gate_t = _gate_t(cfg)
    rows = ROWS[path]
    opt = {
        "gemv": (gate_t, None, None),
        "hgemm1": (None, None, None),
        "det": (gate_t, cfg.gate_i8, cfg.gate_sb),
        "hgemm": (gate_t, None, None),
    }[path]
    hidden = torch.empty((rows, 0), dtype = torch.half, device = device)
    scores = torch.full((rows, E), 99.0, dtype = torch.half, device = device)
    idx = torch.stack([torch.randperm(E)[:K] for _ in range(rows)]).to(device) if fn == "sel_norm" else \
        torch.full((rows, K), -7, dtype = torch.long, device = device)
    w = torch.full((rows, K), 99.0, dtype = torch.half, device = device)
    _fused_route(fn, hidden, cfg.gate_tensor, scores, idx, w, opt)
    torch.cuda.synchronize(device)
    assert torch.equal(scores, torch.zeros_like(scores))
    assert ((idx >= 0) & (idx < E)).all() and (idx.sort(dim = 1).values.diff(dim = 1) != 0).all()
    expect = torch.full_like(w, (1.0 if fn == "std" else 1.5) / K)
    assert torch.allclose(w.float(), expect.float(), rtol = 2.0 ** -10, atol = 0)
