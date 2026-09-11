"""ROCm smoke tests for paths whose CUDA-only extension symbols are absent."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm binding-gate smoke tests", allow_module_level = True)

from exllamav3.modules import block_sparse_mlp as bsm
from exllamav3.modules import dsv4
from exllamav3.modules import gated_delta_net as gdn

DEVICE = "cuda"


def _routing_cfg():
    return SimpleNamespace(
        gate_tensor = torch.randn(8, 6, device = DEVICE, dtype = torch.float16),
        gate_tensor_t = None,
        router_bias = torch.randn(6, device = DEVICE, dtype = torch.float16),
        num_experts = 6,
        num_experts_per_tok = 2,
        router_logits_bsz1 = torch.empty(1, 6, device = DEVICE, dtype = torch.float16),
        routing_weights_bsz1 = torch.empty(1, 2, device = DEVICE, dtype = torch.float16),
        selected_experts_bsz1 = torch.empty(1, 2, device = DEVICE, dtype = torch.long),
        per_expert_scale = torch.linspace(0.75, 1.25, 6, device = DEVICE, dtype = torch.bfloat16),
        e_score_correction_bias = torch.linspace(-0.1, 0.1, 6, device = DEVICE),
        e_score_bias_h = None,
        routed_scaling_factor = 1.7,
        n_group = 1,
        topk_group = 1,
        tid2eid = torch.tensor([[0, 2], [1, 3], [4, 5]], device = DEVICE),
    )


def test_missing_routing_bindings_use_torch_contracts(monkeypatch):
    monkeypatch.setattr(bsm, "ext", SimpleNamespace())
    torch.manual_seed(0)
    cfg = _routing_cfg()
    y = torch.randn(3, 8, device = DEVICE, dtype = torch.float16)

    selected, weights = bsm.routing_std(3, cfg, y, {})
    logits = (y @ cfg.gate_tensor).float()
    top_v, expected_selected = torch.topk(logits, 2, dim = -1)
    expected_weights = torch.softmax(top_v, dim = -1)
    expected_weights *= cfg.per_expert_scale.float()[expected_selected]
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, expected_weights.half())

    selected, weights = bsm.routing_std_bias(3, cfg, y, {})
    logits = (y @ cfg.gate_tensor).float() + cfg.router_bias.float()
    top_v, expected_selected = torch.topk(logits, 2, dim = -1)
    expected_weights = torch.softmax(top_v, dim = -1)
    expected_weights *= cfg.per_expert_scale.float()[expected_selected]
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, expected_weights.half())

    selected, weights = bsm.routing_std(3, cfg, y, {"activate_all_experts": True})
    logits = (y @ cfg.gate_tensor).float()
    expected_selected = torch.arange(cfg.num_experts, device = DEVICE, dtype = torch.long).expand(3, -1)
    expected_weights = torch.softmax(logits, dim = -1)
    expected_weights *= cfg.per_expert_scale.float().unsqueeze(0)
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, expected_weights.half())

    selected, weights = bsm.routing_std_bias(3, cfg, y, {"activate_all_experts": True})
    logits = (y @ cfg.gate_tensor).float() + cfg.router_bias.float()
    expected_selected = torch.arange(cfg.num_experts, device = DEVICE, dtype = torch.long).expand(3, -1)
    expected_weights = torch.softmax(logits, dim = -1)
    expected_weights *= cfg.per_expert_scale.float().unsqueeze(0)
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, expected_weights.half())

    selected, weights = bsm.routing_dots(3, cfg, y, {})
    scores = torch.sigmoid((y @ cfg.gate_tensor).float())
    expected_selected = torch.topk(
        scores + cfg.e_score_correction_bias.unsqueeze(0), 2, dim = -1, sorted = False
    ).indices
    expected_weights = scores.gather(1, expected_selected)
    expected_weights *= cfg.routed_scaling_factor / expected_weights.sum(dim = -1, keepdim = True)
    assert torch.equal(selected.sort().values, expected_selected.sort().values)
    torch.testing.assert_close(
        weights.gather(1, selected.argsort(dim = 1)),
        expected_weights.half().gather(1, expected_selected.argsort(dim = 1)),
        rtol = 2e-3, atol = 2e-3,
    )

    selected, weights = bsm.routing_dots(3, cfg, y, {"activate_all_experts": True})
    scores = torch.sigmoid((y @ cfg.gate_tensor).float())
    expected_selected = torch.arange(cfg.num_experts, device = DEVICE, dtype = torch.long).expand(3, -1)
    expected_weights = scores
    expected_weights += cfg.e_score_correction_bias.unsqueeze(0)
    expected_weights *= cfg.routed_scaling_factor / expected_weights.sum(dim = -1, keepdim = True)
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, expected_weights.half())

    selected, weights = bsm.routing_sqrtsp(3, cfg, y, {})
    scores = torch.sqrt(F.softplus((y @ cfg.gate_tensor).float()))
    expected_selected = torch.topk(
        scores + cfg.e_score_correction_bias.unsqueeze(0), 2, dim = -1, sorted = False
    ).indices
    expected_weights = scores.gather(1, expected_selected)
    expected_weights *= cfg.routed_scaling_factor / expected_weights.sum(dim = -1, keepdim = True)
    assert torch.equal(selected.sort().values, expected_selected.sort().values)
    torch.testing.assert_close(
        weights.gather(1, selected.argsort(dim = 1)),
        expected_weights.half().gather(1, expected_selected.argsort(dim = 1)),
        rtol = 2e-3, atol = 2e-3,
    )

    selected, weights = bsm.routing_sqrtsp(3, cfg, y, {"activate_all_experts": True})
    scores = torch.sqrt(F.softplus((y @ cfg.gate_tensor).float()))
    expected_selected = torch.arange(cfg.num_experts, device = DEVICE, dtype = torch.long).expand(3, -1)
    expected_weights = scores * (
        cfg.routed_scaling_factor / scores.sum(dim = -1, keepdim = True)
    )
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, expected_weights.half())

    selected, weights = bsm.routing_sqrtsp_hash(3, cfg, y, {"activate_all_experts": True})
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, expected_weights.half())


def test_missing_routing_sel_norm_uses_hash_selection_and_torch_weights(monkeypatch):
    monkeypatch.setattr(bsm, "ext", SimpleNamespace())
    cfg = _routing_cfg()
    y = torch.randn(3, 8, device = DEVICE, dtype = torch.float16)
    input_ids = torch.tensor([0, 1, 2], device = DEVICE)
    selected, weights = bsm.routing_sqrtsp_hash(3, cfg, y, {"input_ids": input_ids})
    expected_selected = cfg.tid2eid[input_ids].long()
    scores = torch.sqrt(F.softplus((y @ cfg.gate_tensor).float()))
    expected_weights = scores.gather(1, expected_selected)
    expected_weights *= cfg.routed_scaling_factor / expected_weights.sum(dim = -1, keepdim = True)
    assert torch.equal(selected, expected_selected)
    torch.testing.assert_close(weights, expected_weights.half(), rtol = 2e-3, atol = 2e-3)


def test_dsv4_missing_mgemm_disables_projection_fans(monkeypatch):
    monkeypatch.setattr(dsv4, "ext", SimpleNamespace())
    attn = object.__new__(dsv4.DSV4Attention)
    attn.x_fan_ready = False
    attn._build_x_fan()
    assert attn.x_fan_ready
    assert attn.x_fan is attn.q_fan is None
    assert attn.qb_multi is attn.wob_multi is None

    attn.woa_multi_ready = False
    attn._build_woa_multi()
    assert attn.woa_multi_ready
    assert attn.wo_a_multi is None
    assert attn.woa_indices is None


def test_kda_bc_split_reflects_missing_constructor(monkeypatch):
    # Production ext.py installs _BCNone here, so the attribute exists but returns None.
    monkeypatch.setattr(gdn, "ext", SimpleNamespace(BC_GatedDeltaNetSplit = lambda *args: None))

    def projection(kind):
        return SimpleNamespace(quant_type = kind, inner = SimpleNamespace(bc = None))

    layer = object.__new__(gdn.GatedDeltaNet)
    layer.num_k_heads = 1
    layer.num_v_heads = 1
    layer.k_head_dim = 4
    layer.v_head_dim = 4
    layer.hidden_size = 8
    layer.recurrent_layers = []
    layer.qkvz_proj = layer.ba_proj = layer.z_proj = None
    layer.qkv_proj = projection("exl3")
    layer.o_proj = projection("exl3")
    layer.b_proj = projection("fp16")
    layer.f_a_proj = projection("fp16")
    layer.f_b_proj = projection("fp16")
    layer.g_a_proj = projection("fp16")
    layer.g_b_proj = projection("fp16")
    layer.conv1d_weight = torch.zeros(12, 1, 2, device = DEVICE, dtype = torch.bfloat16)
    layer.conv1d_weight_flat = torch.zeros(12, 2, device = DEVICE, dtype = torch.bfloat16)
    layer.conv1d_bias = None
    layer.conv1d_q_weight = None
    layer.dt_bias = torch.zeros(4, device = DEVICE, dtype = torch.bfloat16)
    layer.a_log = torch.zeros(1, device = DEVICE)
    layer.gate_lower_bound = None
    layer.norm = SimpleNamespace(bc = None)
    layer.beta_scale = 1.0
    layer.kda = True
    layer.bc = None
    layer.bc_split = False
    layer.bsz1_pa_args = []

    layer.load_local(torch.device(DEVICE))

    assert layer.bc is None
    assert not layer.bc_split
