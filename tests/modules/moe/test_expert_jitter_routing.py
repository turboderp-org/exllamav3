"""Small CUDA/ROCm smoke check for offset routing through the existing selection kernels."""
import pytest
import torch
import torch.nn.functional as F

from exllamav3.modules.block_sparse_mlp_routing import routing_jitter, routing_std_bias
from testlib.routing import make_cfg


@pytest.mark.parametrize("router", ["std", "std_bias", "dots", "sqrtsp", "sqrtsp_hash"])
@pytest.mark.parametrize("rows", [1, 6])
def test_offset_routing(device, router, rows):
    cfg = make_cfg(64, 8, 2, device)
    cfg.gate_tensor.zero_()
    cfg.gate_tensor[:8].copy_(torch.eye(8, device = device).half())
    if router == "std_bias":
        cfg.router_bias = torch.arange(8, device = device).half() * 0.05
    if router in ("dots", "sqrtsp"):
        cfg.e_score_correction_bias = torch.arange(8, device = device).half() * 0.01
    cfg.tid2eid = torch.tensor([[0, 7], [1, 6]], device = device)
    batch = 1 if rows == 1 else 2
    offsets = torch.tensor([[-0.8, 0.4, 0.1, -0.3, 0.7, -0.1, 0.2, -0.6],
                            [0.5, -0.7, 0.3, 0.1, -0.4, 0.6, -0.2, 0.8]], device = device).half()[:batch]
    y = torch.zeros(rows, 64, device = device).half()
    y[:, :8] = torch.arange(8, device = device).half() * 0.11
    params = {"expert_jitter_offsets": offsets, "expert_jitter_layout": {"layer": 0},
              "input_ids": torch.arange(batch, device = device).view(batch, 1).expand(batch, rows // batch)}
    sel, w = routing_jitter(rows, cfg, y, params, "layer", router)
    logits = (y[:, :8] + offsets.repeat_interleave(rows // batch, dim = 0)).half()
    if cfg.router_bias is not None:
        logits = logits.float() + cfg.router_bias.float()
    if router.startswith("std"):
        ref_sel = logits.topk(2, dim = -1).indices
        ref_w = logits.float().gather(1, ref_sel).softmax(dim = -1)
    else:
        scores = logits.float().sigmoid() if router == "dots" else F.softplus(logits.float()).sqrt()
        keys = scores if cfg.e_score_correction_bias is None else scores + cfg.e_score_correction_bias.float()
        ref_sel = cfg.tid2eid[params["input_ids"].reshape(-1)] if router.endswith("hash") else keys.topk(2, dim = -1).indices
        ref_w = scores.gather(1, ref_sel)
        ref_w /= ref_w.sum(dim = -1, keepdim = True)
    assert torch.equal(sel.sort(dim = -1).values, ref_sel.sort(dim = -1).values)
    torch.testing.assert_close(w.gather(1, sel.argsort(dim = -1)).float(),
                               ref_w.gather(1, ref_sel.argsort(dim = -1)), atol = 2e-3, rtol = 2e-3)


@pytest.mark.parametrize("rows", [1, 6])
@pytest.mark.parametrize("topk", [1, 2])
def test_zero_offset_preserves_biased_routing(device, rows, topk):
    cfg = make_cfg(64, 8, topk, device)
    cfg.gate_tensor.zero_()
    cfg.gate_tensor[:8].copy_(torch.eye(8, device = device).half())
    cfg.router_bias = torch.zeros(8, device = device).half()
    cfg.router_bias[0] = 2 ** -12
    cfg.router_bias[1] = -(2 ** -11)
    y = torch.zeros(rows, 64, device = device).half()
    y[:, :8] = -8
    y[:, 0] = 1
    y[:, 1] = 1 + 2 ** -10
    # The biased scores differ in fp32, but both round to 1 in fp16. A disabled
    # request in a mixed batch must retain both its selection and its weights.
    ref_sel, ref_w = routing_std_bias(rows, cfg, y, {})
    ref_sel, ref_w = ref_sel.clone(), ref_w.clone()
    batch = 1 if rows == 1 else 2
    offsets = torch.zeros(batch, 8, device = device).half()
    if batch > 1:
        offsets[1, 7] = 16  # another request enables jitter
    params = {"expert_jitter_offsets": offsets, "expert_jitter_layout": {"layer": 0}}
    sel, w = routing_jitter(rows, cfg, y, params, "layer", "std_bias")
    disabled_rows = rows // batch
    assert torch.equal(sel[:disabled_rows], ref_sel[:disabled_rows])
    assert torch.equal(w[:disabled_rows], ref_w[:disabled_rows])
