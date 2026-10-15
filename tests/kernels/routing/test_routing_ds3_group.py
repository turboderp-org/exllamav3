"""
ext.routing_ds3_group and ext.routing_gemv_f32 (routing.cu): the group-limited DS3 router (DeepSeek-V3 noaux_tc,
Ling 3.x) on fp32 logits, plus the fp32-output single-row projection that feeds it.
Contract, with key = act(s) + bias (act(s) without a bias), G = E / n_group:
- group score = sum of the two largest keys of the group's G experts (a lone expert scores alone); the topk_group
  groups with the largest scores are eligible; the selection is the top-K keys among eligible experts
- every selected expert is eligible, and its key is within 1e-6 (fp32 sigmoid vs float64) of the K-th largest
  eligible key; the selection is in descending key order and holds K distinct experts
- weights = act(s_sel) * scaling / sum(act(s_sel)) in the selection order, to fp16 output rounding plus fp32 ulp
- routing_gemv_f32 scores: fp32 fixed-order FMA dot products, within 2^-20 of the float64 reference scale
- rejections: n_group not dividing E, topk_group outside [1, n_group], K beyond the eligible experts, half logits
"""
import pytest
import torch
import torch.nn.functional as F
from exllamav3.ext import exllamav3_ext as ext

CONFIGS = [(64, 8, 4, 8), (256, 8, 4, 8), (512, 8, 4, 8), (512, 16, 2, 6), (160, 10, 5, 6), (32, 2, 1, 2), (24, 24, 3, 3)]


def _act(s, act):
    s = s.double()
    return torch.sigmoid(s) if act == 0 else F.softplus(s).sqrt()


def _reference(keys, n_group, topk_group, K):
    """float64 group-limited selection: (rows, K) sorted descending keys of the eligible top-K and the eligibility mask"""
    rows, E = keys.shape
    G = E // n_group
    grouped = keys.view(rows, n_group, G)
    gscore = grouped.topk(min(2, G), dim = -1).values.sum(dim = -1)
    groups = gscore.topk(topk_group, dim = -1).indices
    mask = torch.zeros((rows, n_group), dtype = torch.bool, device = keys.device)
    mask.scatter_(1, groups, True)
    elig = mask.unsqueeze(-1).expand(rows, n_group, G).reshape(rows, E)
    masked = keys.masked_fill(~elig, float("-inf"))
    return masked.topk(K, dim = -1).values, elig


@pytest.mark.parametrize("act", [0, 1], ids = ["sigmoid", "sqrtsp"])
@pytest.mark.parametrize("with_bias", [False, True], ids = ["nobias", "bias"])
@pytest.mark.parametrize("rows", [1, 77])
@torch.inference_mode()
def test_routing_ds3_group(device, rows, with_bias, act):
    for i, (E, n_group, topk_group, K) in enumerate(CONFIGS):
        g = torch.Generator().manual_seed(100 * i + rows + act)
        scores = (torch.randn((rows, E), generator = g) * 2.0).to(device)
        bias = ((torch.rand(E, generator = g) - 0.5) * 0.4).to(device) if with_bias else None
        sel = torch.full((rows, K), -7, dtype = torch.long, device = device)
        w = torch.full((rows, K), 99.0, dtype = torch.half, device = device)
        ext.routing_ds3_group(scores, bias, sel, w, 2.5, act, n_group, topk_group)
        label = f"E={E} n_group={n_group} topk_group={topk_group} K={K}"

        assert ((sel >= 0) & (sel < E)).all(), label
        assert (sel.sort(dim = 1).values.diff(dim = 1) != 0).all(), f"{label}: duplicate experts"
        keys = _act(scores, act) + (bias.double() if bias is not None else 0.0)
        ref_keys, elig = _reference(keys, n_group, topk_group, K)
        assert elig.gather(1, sel).all(), f"{label}: selected an expert outside the eligible groups"
        k_sel = keys.gather(1, sel)
        assert (k_sel >= ref_keys[:, -1:] - 1e-6).all(), f"{label}: selected a key below the eligible top-K"
        assert (k_sel[:, :-1] >= k_sel[:, 1:] - 1e-6).all(), f"{label}: not in descending key order"
        a_sel = _act(scores, act).gather(1, sel)
        ref_w = a_sel * 2.5 / a_sel.sum(dim = 1, keepdim = True)
        assert torch.allclose(w.double(), ref_w, rtol = 2 ** -10 + 1e-5, atol = 1e-6), f"{label}: weights"


@torch.inference_mode()
def test_routing_gemv_f32(device):
    for hdim, E in ((256, 64), (2048, 512), (4096, 160)):
        g = torch.Generator().manual_seed(hdim + E)
        hidden = torch.randn((1, hdim), generator = g).half().to(device)
        gate_t = (torch.randn((E, hdim), generator = g) * hdim ** -0.5).half().to(device)
        scores = torch.full((1, E), 99.0, dtype = torch.float, device = device)
        ext.routing_gemv_f32(hidden.view(-1), gate_t, scores.view(-1))
        ref = hidden.double() @ gate_t.double().T
        scale = hidden.double().norm() * gate_t.double().norm(dim = 1)
        assert ((scores.double() - ref).abs() <= 2 ** -20 * scale + 1e-7).all(), f"hdim={hdim} E={E}"


@torch.inference_mode()
def test_rejections(device):
    scores = torch.zeros((2, 64), dtype = torch.float, device = device)
    sel = torch.zeros((2, 8), dtype = torch.long, device = device)
    w = torch.zeros((2, 8), dtype = torch.half, device = device)
    with pytest.raises(RuntimeError, match = "n_group"):
        ext.routing_ds3_group(scores, None, sel, w, 1.0, 0, 7, 3)
    with pytest.raises(RuntimeError, match = "topk_group"):
        ext.routing_ds3_group(scores, None, sel, w, 1.0, 0, 8, 9)
    with pytest.raises(RuntimeError, match = "eligible"):
        ext.routing_ds3_group(scores, None, sel, w, 1.0, 0, 16, 1)
    with pytest.raises(RuntimeError):
        ext.routing_ds3_group(scores.half(), None, sel, w, 1.0, 0, 8, 4)
