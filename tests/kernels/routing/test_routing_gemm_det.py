"""
Deterministic int8 router projection (ext.routing_gemm_det, routing_gemm.cu + det_gemm.cuh): must match an fp32
reference to half-output tolerance over row counts, expert counts (including non-multiples of the tile) and K
values, be bit-reproducible, agree bit for bit across every visible GPU (the point of the int8 scheme: fp16 tensor
cores do not across architectures), and feed ext.routing_std so that multi-row routing selects the same experts as
the reference on rows that are not near-ties. Also checks the FMA-only transcendentals the routing activations use
against float64 torch.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(not getattr(ext, "HAS_DET_GEMM", False), reason = "extension built without the deterministic GEMM")


def quant_gate(gate):
    gate_t = gate.T.contiguous()
    E, K = gate_t.shape
    g8 = torch.empty((2, E, K), dtype = torch.int8, device = gate.device)
    sb = torch.empty((E,), dtype = torch.float, device = gate.device)
    ext.det_quant_weight(gate_t, g8, sb)
    return gate_t, g8, sb


@pytest.mark.parametrize("K", [2560, 2048, 2880, 1040])
@pytest.mark.parametrize("E", [512, 128, 64, 200, 40])
@torch.inference_mode()
def test_matches_reference(device, K, E):
    torch.manual_seed(0)
    gate = (torch.randn(K, E, device = device) * (1.0 / K ** 0.5)).half()
    gate_t, g8, sb = quant_gate(gate)
    # The weight quantization itself: 14-bit fixed point per row, error well under half's
    deq = (g8[0].float() * 128 + g8[1].float()) * sb[:, None]
    assert (deq - gate_t.float()).abs().max().item() < 1e-4 * gate_t.float().abs().max().item()
    for R in (2, 3, 16, 17, 64, 65, 300, 2048):
        x = torch.randn(R, K, device = device).half()
        ref = x.float() @ gate.float()
        out = torch.empty(R, E, dtype = torch.half, device = device)
        ext.routing_gemm_det(x, g8, sb, out)
        err = (out.float() - ref).abs().max().item()
        scale = ref.abs().max().item()
        assert err < 2e-3 * scale + 1e-3, (K, E, R, err, scale)
        out2 = torch.empty_like(out)
        ext.routing_gemm_det(x, g8, sb, out2)
        assert torch.equal(out, out2), (K, E, R)


@pytest.mark.multi_gpu(2)
@torch.inference_mode()
def test_cross_device_identity(devices):
    torch.manual_seed(3)
    K, E = 2560, 512
    gate = (torch.randn(K, E) * (1.0 / K ** 0.5)).half()
    for R in (2, 64, 65, 300, 1000):
        x = torch.randn(R, K).half()
        outs = []
        for dev in devices:
            with torch.cuda.device(dev):
                _, g8, sb = quant_gate(gate.to(dev))
                out = torch.empty(R, E, dtype = torch.half, device = dev)
                ext.routing_gemm_det(x.to(dev), g8, sb, out)
                outs.append(out.cpu())
        for dev, out in zip(devices[1:], outs[1:]):
            assert torch.equal(outs[0], out), (R, torch.cuda.get_device_name(devices[0]), torch.cuda.get_device_name(dev))


@torch.inference_mode()
def test_routing_std_multirow_selection(device):
    torch.manual_seed(2)
    K, E, topk = 2560, 128, 8
    gate = (torch.randn(K, E, device = device) * (1.0 / K ** 0.5)).half()
    gate_t, g8, sb = quant_gate(gate)
    x = torch.randn(300, K, device = device).half()
    logits = torch.empty(300, E, dtype = torch.half, device = device)
    sel = torch.empty(300, topk, dtype = torch.long, device = device)
    w = torch.empty(300, topk, dtype = torch.half, device = device)
    ext.routing_std(x, gate, logits, sel, w, None, gate_t, None, g8, sb)
    ref = x.float() @ gate.float()
    ref_top = torch.topk(ref, topk, dim = -1).indices
    agree = (torch.sort(sel, dim = 1).values == torch.sort(ref_top, dim = 1).values).all(dim = 1)
    tv = torch.topk(ref, topk + 1, dim = -1).values
    clear = (tv[:, topk - 1] - tv[:, topk]) > 1e-2
    assert agree[clear].all().item()
    assert clear.float().mean().item() > 0.5


@torch.inference_mode()
def test_det_transcendentals(device):
    x = torch.cat((torch.linspace(-80, 80, 100001), torch.randn(100000) * 5)).to(device)
    y = torch.empty(3 * x.numel(), dtype = torch.float, device = device)
    ext.det_math_test(x, y)
    e, l, sp = y.view(3, -1)
    rel = lambda a, b: ((a - b).abs() / b.abs().clamp_min(1e-30)).max().item()
    assert rel(e.double(), torch.exp(x.double())) < 4e-7
    assert rel(l.double(), torch.log(x.double().abs() + 1e-30)) < 4e-7
    assert rel(sp.double(), torch.nn.functional.softplus(x.double())) < 4e-7
