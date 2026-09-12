"""Contracts for ROCm's pure-PyTorch activation and normalization fallbacks."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(_REPO_ROOT))
_FALLBACKS = _REPO_ROOT / "exllamav3" / "ext_fallbacks.py"
_spec = importlib.util.spec_from_file_location("exllamav3_ext_fallbacks_contract", _FALLBACKS)
assert _spec is not None and _spec.loader is not None
_fallback_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fallback_impl)
# These tests exercise the pure-Python fallback formulas directly; ext wiring has
# separate end-to-end coverage.
fb = _fallback_impl


@pytest.mark.parametrize(
    "fallback,activation",
    [
        (fb.silu_mul, F.silu),
        (fb.gelu_mul, lambda t: F.gelu(t, approximate = "tanh")),
        (fb.relu2_mul, lambda t: F.relu(t).square()),
        (fb.relu_mul, F.relu),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_activation_mul_clamps_activated_gate_and_up_path_before_product(fallback, activation, dtype):
    x = torch.tensor([[-6.0, -1.0, 0.5, 3.0, 12.0]], dtype = dtype)
    y = torch.tensor([[-8.0, -2.5, 1.0, 2.5, 8.0]], dtype = dtype)
    original_x, original_y = x.clone(), y.clone()
    output = torch.full(x.shape, float("nan"), dtype = torch.float16)
    limit = 2.0

    assert fallback(x, y, output, limit) is None

    expected = activation(x).clamp(max = limit) * y.clamp(min = -limit, max = limit)
    if dtype == torch.float32:
        expected = expected.clamp(min = -65504.0, max = 65504.0)
    torch.testing.assert_close(output, expected.half(), rtol = 2e-3, atol = 2e-3)
    assert torch.equal(x, original_x)
    assert torch.equal(y, original_y)


def test_silu_oai_mul_matches_clamped_swiglu_and_allows_output_alias():
    gate = torch.tensor([[-3.0, -0.5, 0.5, 4.0]], dtype = torch.float32)
    up = torch.tensor([[-4.0, -0.25, 0.75, 5.0]], dtype = torch.float32)
    expected_gate = gate.clamp(max = 2.0)
    expected_up = up.clamp(min = -2.0, max = 2.0)
    expected = (expected_up + 1.0) * expected_gate * torch.sigmoid(1.702 * expected_gate)

    output = torch.empty_like(gate, dtype = torch.float16)
    fb.silu_oai_mul(gate, up, output, 2.0)
    torch.testing.assert_close(output, expected.half())

    aliased = gate.half()
    fb.silu_oai_mul(aliased, up.half(), aliased, 2.0)
    torch.testing.assert_close(aliased, expected.half(), rtol = 2e-3, atol = 2e-3)


def test_xielu_matches_native_formula_and_signature():
    x = torch.tensor([[-3.0, -1e-8, 0.0, 0.5, 2.0]], dtype = torch.float32)
    output = torch.empty_like(x, dtype = torch.float16)
    alpha_p = torch.tensor(-0.4, dtype = torch.bfloat16)
    alpha_n = torch.tensor(0.7, dtype = torch.float16)
    fb.xielu(x, output, alpha_p, alpha_n)

    p = F.softplus(alpha_p.float())
    n = F.softplus(alpha_n.float()) + 0.5
    eps = -9.9838e-7
    expected = torch.where(
        x > 0,
        p * x.square() + 0.5 * x,
        (torch.expm1(torch.minimum(x, torch.tensor(eps))) - x) * n + 0.5 * x,
    )
    torch.testing.assert_close(output, expected.half())


def test_add_sigmoid_gate_accumulates_broadcast_gate():
    x = torch.arange(24, dtype = torch.float32).reshape(2, 3, 4) / 10
    gate = torch.tensor([[[-2.0], [0.0], [2.0]], [[1.0], [-1.0], [0.5]]])
    output = torch.randn_like(x)
    expected = output + x * torch.sigmoid(gate)
    fb.add_sigmoid_gate(x, gate, output)
    torch.testing.assert_close(output, expected)


def test_add_sigmoid_gate_proj_uses_y_times_dim_by_one_weight():
    x = torch.randn(2, 3, 4, dtype = torch.float32)
    y = torch.randn(2, 3, 4, dtype = torch.float16)
    w = torch.randn(4, 1, dtype = torch.float16)
    output = torch.randn_like(x)
    expected = output + x * torch.sigmoid(y.float() @ w.float())
    fb.add_sigmoid_gate_proj(x, y, output, w)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("shape", [(1, 2, 3, 4), (2, 1, 5, 8), (2, 3, 1, 2)])
@pytest.mark.parametrize("kind", ["sigmoid", "softplus"])
def test_broadcast_gates_expand_only_trailing_feature_dimension(shape, kind):
    output = torch.randn(shape, dtype = torch.float16)
    gate = torch.randn(shape[:-1], dtype = torch.float16)
    expected = output.float() * (
        torch.sigmoid(gate.float()) if kind == "sigmoid" else F.softplus(gate.float())
    ).unsqueeze(-1)
    if kind == "sigmoid":
        fb.mul_sigmoid_broadcast_(output, gate)
    else:
        fb.mul_softplus_broadcast_(output, gate)
    torch.testing.assert_close(output, expected.half(), rtol = 2e-3, atol = 2e-3)


def test_deinterleave_qg_preserves_heads_with_arbitrary_leading_dimensions():
    leading, heads, head_dim = (2, 3, 4), 3, 8
    source = torch.arange(
        torch.tensor(leading).prod().item() * heads * 2 * head_dim,
        dtype = torch.float16,
    ).reshape(*leading, heads * 2 * head_dim)
    chunks = source.view(*leading, heads, 2 * head_dim)
    q = torch.empty(*leading, heads, head_dim, dtype = torch.float16)
    g = torch.empty(*leading, heads * head_dim, dtype = torch.float16)
    fb.deinterleave_qg(source, q, g, head_dim)
    torch.testing.assert_close(q, chunks[..., :head_dim])
    torch.testing.assert_close(g, chunks[..., head_dim:].reshape_as(g))


def _rms(x, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.square().mean(dim = -1, keepdim = True) + eps)


def test_rms_norm_span_heads_flattens_last_two_dimensions():
    x = torch.randn(2, 3, 4, dtype = torch.float16)
    w = torch.randn(12, dtype = torch.float16)
    output = torch.empty_like(x, dtype = torch.float32)
    fb.rms_norm(x, w, output, 1e-5, 0.25, 1.5, True, False)
    expected = _rms(x.flatten(-2), 1e-5) * 1.5 * (w.float() + 0.25)
    torch.testing.assert_close(output.flatten(-2), expected)


def test_rms_norm_cycles_grouped_weights_by_flat_row_and_preserves_residual():
    x = torch.randn(2, 3, 4, dtype = torch.float16)
    w = torch.randn(3, 4, dtype = torch.bfloat16)
    output = torch.randn_like(x, dtype = torch.float32)
    original = output.clone()
    fb.rms_norm(x, w, output, 1e-6, -0.125, 0.75, False, True, 3)
    flat = _rms(x.reshape(-1, 4), 1e-6) * 0.75
    rows = torch.arange(flat.shape[0]) % 3
    expected = original.reshape(-1, 4) + flat * (w.float()[rows] - 0.125)
    torch.testing.assert_close(output.reshape(-1, 4), expected)


@pytest.mark.parametrize("gate_first", [False, True])
@pytest.mark.parametrize("gate_act", [0, 1])
def test_gated_rms_norm_matches_grouped_gate_order_bias_and_activation(gate_first, gate_act):
    x = torch.randn(2, 3, 4, dtype = torch.bfloat16)
    gate = torch.randn_like(x, dtype = torch.float32)
    w = torch.randn(3, 4, dtype = torch.float32)
    output = torch.empty_like(gate)
    fb.gated_rms_norm(x, w, output, gate, 1e-5, 0.2, 3, gate_first, gate_act)

    activation = torch.sigmoid(gate) if gate_act == 1 else F.silu(gate)
    norm_input = x.float() * activation if gate_first else x.float()
    normalized = _rms(norm_input.reshape(-1, 4), 1e-5)
    rows = torch.arange(normalized.shape[0]) % 3
    expected = normalized * (w[rows] + 0.2)
    if not gate_first:
        expected *= activation.reshape(-1, 4)
    torch.testing.assert_close(output.reshape(-1, 4), expected)


def test_softcap_writes_output_and_supports_in_place_operation():
    x = torch.tensor([[-8.0, -1.0, 0.0, 2.0, 9.0]], dtype = torch.float32)
    expected = 3.0 * torch.tanh(x / 3.0)
    output = torch.empty_like(x)
    assert fb.softcap(x, output, 3.0) is None
    torch.testing.assert_close(output, expected)
    fb.softcap(x, x, 3.0)
    torch.testing.assert_close(x, expected)


def test_paged_dequant_bounds_decoded_rows_to_block_window(monkeypatch):
    page_size, pages, dim, bits = 256, 8, 64, 4
    groups = dim // 32
    packed = torch.zeros((pages, page_size, groups * bits), dtype = torch.int32)
    scales = torch.ones((pages, page_size, groups), dtype = torch.float16)
    output_k = torch.full((pages, page_size, 1, dim), float("nan"), dtype = torch.float16)
    output_v = torch.full_like(output_k, float("nan"))
    lengths = torch.tensor([1500], dtype = torch.int32)
    table = torch.arange(pages, dtype = torch.int32).unsqueeze(0)
    decoded_rows = 0

    def fake_dequant(_packed, _scales, output, _compand_a):
        nonlocal decoded_rows
        decoded_rows += output.shape[0]
        output.zero_()

    monkeypatch.setattr(_fallback_impl, "dequant_cache_cont", fake_dequant)
    _fallback_impl.dequant_cache_paged(
        packed, scales, output_k, packed, scales, output_v,
        lengths, table, page_size, 10, 0.0,
    )

    # dim=64 maps one token to one native launch block chunk, so the ten-token
    # window begins at the containing 256-token block: [1280, 1500).
    assert decoded_rows == 2 * (1500 - 1280)
    assert torch.isnan(output_k.reshape(-1, dim)[:1280]).all()
    assert torch.isfinite(output_k.reshape(-1, dim)[1280:1500]).all()
