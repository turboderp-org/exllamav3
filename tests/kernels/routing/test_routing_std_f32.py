"""
fp32-logit standard router (ext.routing_std with float32 scores, ext.routing_gemv_f32 over several rows;
RoutingCFG.logits_dtype = torch.float through block_sparse_mlp_routing.routing_std).
Contract:
- scores: fixed-order fp32 FMA dot products, within 2^-20 of the float64 reference scale, for any row count
- a row's scores are the same bits whatever row count it arrives in (single-row and multi-row kernels share the
  accumulation order), so prefill, batched decode and single-token decode select identically on identical input
- selection: the float64 top-K on every row whose K-th / (K+1)-th gap exceeds fp32 rounding of the logits (far
  below fp16 logit resolution, which the half path cannot resolve)
- weights: softmax over the selected logits, to fp16 output rounding; a router bias enters before the top-k
- float32 scores without the transposed gate are rejected
"""
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.block_sparse_mlp_routing import routing_std, routing_std_bias
from tests.testlib.routing import make_cfg, clear_rows, same_sel

SHAPES = [(2560, 512, 10), (2048, 128, 8), (1040, 200, 6), (256, 30, 30)]
ROWS = [1, 2, 3, 5, 14, 65, 300]


def _f32_cfg(hdim, E, K, device, seed, bias = None):
    cfg = make_cfg(hdim, E, K, device, seed = seed, bias = bias)
    cfg.logits_dtype = torch.float
    cfg.router_logits_bsz1 = torch.empty(1, E, dtype = torch.float, device = device)
    return cfg


@torch.inference_mode()
def test_projection_rows(device):
    for hdim, E, _ in SHAPES:
        g = torch.Generator().manual_seed(hdim + E)
        gate_t = (torch.randn((E, hdim), generator = g) * hdim ** -0.5).half().to(device)
        x = torch.randn((ROWS[-1], hdim), generator = g).half().to(device)
        single = torch.empty((ROWS[-1], E), dtype = torch.float, device = device)
        for r in range(ROWS[-1]):
            ext.routing_gemv_f32(x[r], gate_t, single[r])
        ref = x.double() @ gate_t.double().T
        scale = x.double().norm(dim = 1, keepdim = True) * gate_t.double().norm(dim = 1)
        assert ((single.double() - ref).abs() <= 2 ** -20 * scale + 1e-7).all(), f"hdim={hdim} E={E}"
        for R in ROWS[1:]:
            scores = torch.full((R, E), 99.0, dtype = torch.float, device = device)
            ext.routing_gemv_f32(x[:R].contiguous(), gate_t, scores)
            assert torch.equal(scores, single[:R]), f"hdim={hdim} E={E} R={R}: scores depend on the row count"


@pytest.mark.parametrize("with_bias", [False, True], ids = ["nobias", "bias"])
@torch.inference_mode()
def test_selection_and_weights(device, with_bias):
    for i, (hdim, E, K) in enumerate(SHAPES):
        bias = ((torch.rand(E, generator = torch.Generator().manual_seed(i)) - 0.5) * 0.5).half().to(device) \
            if with_bias else None
        cfg = _f32_cfg(hdim, E, K, device, seed = 10 * i + with_bias, bias = bias)
        fn = routing_std_bias if with_bias else routing_std
        for R in ROWS:
            y = torch.randn((R, hdim), generator = torch.Generator().manual_seed(R)).half().to(device)
            sel, w = fn(R, cfg, y, {})
            sel, w = sel.clone(), w.clone()
            ref = y.double() @ cfg.gate_tensor.double()
            if bias is not None:
                ref = ref + bias.double()
            label = f"hdim={hdim} E={E} K={K} R={R}"
            if K < E:
                clear = clear_rows(ref, K, 1e-5)
                assert same_sel(sel, torch.topk(ref, K, dim = -1).indices)[clear].all(), f"{label}: selection"
                assert clear.float().mean().item() > 0.9, label
            else:
                assert same_sel(sel, torch.arange(E, device = device).expand(R, E)).all(), label
            ref_w = torch.softmax(ref.gather(1, sel), dim = -1)
            assert torch.allclose(w.double(), ref_w, rtol = 2 ** -10 + 1e-5, atol = 1e-6), f"{label}: weights"
            # Row-count independence of the whole decision: every row routed alone agrees bit for bit
            for r in range(0, R, max(1, R // 4)):
                s1, w1 = fn(1, cfg, y[r : r + 1].contiguous(), {})
                assert torch.equal(s1[0], sel[r]) and torch.equal(w1[0], w[r]), f"{label}: row {r} routed alone differs"


@torch.inference_mode()
def test_resolves_sub_fp16_margins(device):
    """Two experts whose logits differ by less than fp16 resolution at their magnitude: the fp32 path orders them
    as the float64 reference does on every row"""
    hdim, E, K = 512, 16, 1
    cfg = _f32_cfg(hdim, E, K, device, seed = 3)
    gate = cfg.gate_tensor
    gate.zero_()
    # Experts 0 and 1 share every weight except the last input, where they differ by one fp16 ulp
    gate[:, 0] = 0.25
    gate[:, 1] = 0.25
    gate[-1, 1] = torch.tensor(0.25, dtype = torch.half).item() + 2 ** -12
    y = torch.ones((7, hdim), dtype = torch.half, device = device)
    y[:, -1] = torch.tensor([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.0], dtype = torch.half, device = device)
    sel, _ = routing_std(7, cfg, y, {})
    ref = (y.double() @ gate.double()).argmax(dim = -1)
    assert torch.equal(sel[:, 0], ref)
    # The gap is below the fp16 spacing of the logits (128 +- 2^-12 * y): a half logit buffer cannot hold it
    logits = y.double() @ gate.double()
    assert ((logits[:, 0] - logits[:, 1]).abs() < 2 ** -4).all() and (logits[:, 0].half() == logits[:, 1].half()).all()


@torch.inference_mode()
def test_rejections(device):
    hdim, E, K = 256, 32, 4
    cfg = _f32_cfg(hdim, E, K, device, seed = 1)
    y = torch.zeros((2, hdim), dtype = torch.half, device = device)
    logits = torch.empty((2, E), dtype = torch.float, device = device)
    sel = torch.empty((2, K), dtype = torch.long, device = device)
    w = torch.empty((2, K), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "transposed gate"):
        ext.routing_std(y, cfg.gate_tensor, logits, sel, w, None, None, None, None, None)
    with pytest.raises(RuntimeError, match = "float16 or float32"):
        ext.routing_std(y, cfg.gate_tensor, logits.double(), sel, w, None, cfg.gate_tensor.T.contiguous(), None, None, None)
