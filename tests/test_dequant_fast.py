"""
Correctness tests for the fast dequant path (audit A1).

``EXL3LoRAHadFunction`` reconstructs only the INNER trellis weight and applies
the Hadamard/sign transforms to the activations:

    y = had(had(x * suh) @ W_inner) * svh  +  scale*x@A@B  -  offs*x@a0@b0  + bias

which must equal the legacy path's ``x @ W_eff (+ ...)`` where

    W_eff = diag(suh) @ H @ W_inner @ H @ diag(svh)      (H block-diagonal, 128)

-- i.e. exactly ``LinearEXL3.get_weight_tensor()``. Tiers:

  1. test_gradcheck_f64
        Rigorous finite-difference gradcheck of the hand-written backward,
        float64 on CPU, torch only. LoRA + bias + pissa offset all active,
        plus the frozen (r=0) reference view.

  2. test_matches_legacy_reference
        Forward AND all gradients must match ``reference_forward`` (plain
        autograd) run on the composed W_eff -- pins the transform math, the
        adjoint, and the pissa-offset sign against the legacy path. Exact
        (float64, same H both sides).

  3. test_mixed_dtype_paths
        The production dtype mix (bf16 activations / fp16 inner weight /
        fp32 master adapters) runs, stays finite, returns fp32 adapter grads.

  4. test_backward_cache
        The backbone recompute->backward weight cache: passthrough outside the
        phase, store/hit/evict inside, no stale reuse across phases. Imports
        the backbone seam lazily -- SKIPs if the package can't import (no ext).

  5. test_real_exl3_layer  (opt-in: EXL3_TEST_MODEL or --model, GPU + ext)
        On real quantized linears: the fast forward vs x @ get_weight_tensor()
        with the RUNTIME H -- catches any disagreement between these tests'
        synthetic Sylvester H and the shipped hadamard tables.

Run directly::

    python tests/test_dequant_fast.py
    python tests/test_dequant_fast.py --model /path/to/exl3/model   # + tier 5
"""

from __future__ import annotations
import os
import sys
import argparse
import functools
import importlib.util
import torch
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# Load qlora_linear directly from its file (torch-only; avoids importing the
# exllamav3 package -- and its CUDA extension build -- for the CPU tiers).
_spec = importlib.util.spec_from_file_location(
    "exllamav3_qlora_linear",
    os.path.join(_ROOT, "exllamav3", "training", "qlora_linear.py"),
)
qlora_linear = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qlora_linear)

EXL3LoRAHadFunction = qlora_linear.EXL3LoRAHadFunction
reference_forward = qlora_linear.reference_forward

try:
    from util import run_timed
except ImportError:  # pytest collection from repo root
    from tests.util import run_timed

HAD_DIM = 128


def sylvester_hadamard(n: int, dtype=torch.float64) -> torch.Tensor:
    """Normalized Hadamard via Sylvester's construction (orthogonal:
    H @ H.T = I). Stands in for backbone.hadamard_128; the Function must be
    correct for ANY orthogonal H, and tier 5 covers the shipped one."""
    h = torch.tensor([[1.0]], dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    assert h.shape[0] == n
    return h / (n ** 0.5)


def compose_w_eff(w_inner, suh, svh, had):
    """The legacy effective weight: diag(suh) @ H @ W_inner @ H @ diag(svh),
    H block-diagonal at HAD_DIM over each feature dim (what get_weight_tensor
    builds via preapply_had_l / preapply_had_r and the sign vectors)."""
    n_in, n_out = w_inner.shape
    w = (had @ w_inner.view(n_in // HAD_DIM, HAD_DIM, n_out)).view(n_in, n_out)
    w = w * suh.unsqueeze(1)
    w = (w.view(n_in, n_out // HAD_DIM, HAD_DIM) @ had).view(n_in, n_out)
    w = w * svh.unsqueeze(0)
    return w


def make_case(n_in=256, n_out=384, r=4, batch=3, seq=5, dtype=torch.float64,
              with_offs=True, with_bias=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    w_inner = torch.randn(n_in, n_out, generator=g, dtype=dtype) * 0.05
    # Real suh/svh are random sign flips; use signs to stay faithful.
    suh = torch.where(torch.rand(n_in, generator=g) < 0.5, -1.0, 1.0).to(dtype)
    svh = torch.where(torch.rand(n_out, generator=g) < 0.5, -1.0, 1.0).to(dtype)
    had = sylvester_hadamard(HAD_DIM, dtype)
    x = torch.randn(batch, seq, n_in, generator=g, dtype=dtype) * 0.3
    a = torch.randn(n_in, r, generator=g, dtype=dtype) * 0.1
    b = torch.randn(r, n_out, generator=g, dtype=dtype) * 0.1
    bias = torch.randn(n_out, generator=g, dtype=dtype) * 0.1 if with_bias else None
    offs_a = (torch.randn(n_in, r, generator=g, dtype=dtype) * 0.1
              if with_offs else None)
    offs_b = (torch.randn(r, n_out, generator=g, dtype=dtype) * 0.1
              if with_offs else None)
    return w_inner, suh, svh, had, x, a, b, bias, offs_a, offs_b


def test_gradcheck_f64():
    """Finite-difference gradcheck of the hand-written backward (f64)."""
    w_inner, suh, svh, had, x, a, b, bias, offs_a, offs_b = make_case(
        batch=2, seq=3)
    x = x.requires_grad_(True)
    a = a.requires_grad_(True)
    b = b.requires_grad_(True)
    bias = bias.requires_grad_(True)

    def fn(x_, a_, b_, bias_):
        return EXL3LoRAHadFunction.apply(
            x_, a_, b_, bias_, 0.7, lambda: w_inner, suh, svh, had,
            offs_a, offs_b, 0.7)

    assert torch.autograd.gradcheck(fn, (x, a, b, bias), eps=1e-6, atol=1e-8)

    # r=0 / no-offset / no-bias variant (the frozen reference view).
    x2 = x.detach().clone().requires_grad_(True)

    def fn_frozen(x_):
        return EXL3LoRAHadFunction.apply(
            x_, None, None, None, 1.0, lambda: w_inner, suh, svh, had,
            None, None, 1.0)

    assert torch.autograd.gradcheck(fn_frozen, (x2,), eps=1e-6, atol=1e-8)
    print("  gradcheck (lora+bias+offset, and frozen): OK")


def test_matches_legacy_reference():
    """Fast path == plain autograd on the composed W_eff, forward and grads."""
    for with_offs in (False, True):
        w_inner, suh, svh, had, x, a, b, bias, offs_a, offs_b = make_case(
            with_offs=with_offs)
        scale = 0.6
        w_eff = compose_w_eff(w_inner, suh, svh, had)

        # Reference: legacy math (x @ W_eff with the offset folded into the
        # weight, exactly what DiffLinear._weight_closure serves).
        w_ref = w_eff - scale * (offs_a @ offs_b) if with_offs else w_eff
        x_r = x.detach().clone().requires_grad_(True)
        a_r = a.detach().clone().requires_grad_(True)
        b_r = b.detach().clone().requires_grad_(True)
        y_ref = reference_forward(x_r, w_ref, a_r, b_r, scale, bias)
        y_ref.pow(2).sum().backward()

        x_f = x.detach().clone().requires_grad_(True)
        a_f = a.detach().clone().requires_grad_(True)
        b_f = b.detach().clone().requires_grad_(True)
        y_fast = EXL3LoRAHadFunction.apply(
            x_f, a_f, b_f, bias, scale, lambda: w_inner, suh, svh, had,
            offs_a, offs_b, scale)
        y_fast.pow(2).sum().backward()

        for name, t_ref, t_fast in (
            ("y", y_ref, y_fast),
            ("grad_x", x_r.grad, x_f.grad),
            ("grad_a", a_r.grad, a_f.grad),
            ("grad_b", b_r.grad, b_f.grad),
        ):
            d = (t_ref - t_fast).abs().max().item()
            assert d < 1e-10, f"{name} mismatch (offs={with_offs}): {d:.3e}"
    print("  fast path == legacy reference (fwd + all grads, ±pissa offset): OK")


def test_mixed_dtype_paths():
    """fp32 master adapters + fp16 inner weight + bf16 activations: the
    production dtype mix must run, stay finite, and land near the f64 truth."""
    w_inner, suh, svh, had, x, a, b, bias, offs_a, offs_b = make_case()
    w16, suh16, svh16, had16 = (t.to(torch.float16)
                                for t in (w_inner, suh, svh, had))
    xbf = x.to(torch.bfloat16).requires_grad_(True)
    a32 = a.to(torch.float32).requires_grad_(True)
    b32 = b.to(torch.float32).requires_grad_(True)
    y = EXL3LoRAHadFunction.apply(
        xbf, a32, b32, bias.to(torch.bfloat16), 0.5, lambda: w16,
        suh16, svh16, had16,
        offs_a.to(torch.bfloat16), offs_b.to(torch.bfloat16), 0.5)
    assert y.dtype == torch.bfloat16
    y.float().pow(2).sum().backward()
    ref = reference_forward(
        x.float(),
        (compose_w_eff(w_inner, suh, svh, had) - 0.5 * (offs_a @ offs_b)).float(),
        a.float(), b.float(), 0.5, bias.float())
    rel = (y.float() - ref).norm().item() / ref.norm().item()
    assert rel < 5e-2, f"bf16/fp16 path far from f64 reference: rel {rel:.3e}"
    for gr, nm in ((xbf.grad, "x"), (a32.grad, "a"), (b32.grad, "b")):
        assert gr is not None and torch.isfinite(gr.float()).all(), nm
    assert a32.grad.dtype == torch.float32 and b32.grad.dtype == torch.float32
    print(f"  mixed-dtype (bf16 x / fp16 W / fp32 adapters): OK (rel {rel:.1e})")


def test_backward_cache():
    """Store/hit/evict semantics of the recompute->backward weight cache."""
    try:
        from exllamav3.training import backbone
    except Exception as e:  # pragma: no cover -- ext-less container
        print(f"  SKIP backward-cache test (backbone import failed: {e})")
        return

    calls = {"n": 0}

    def raw():
        calls["n"] += 1
        return torch.full((2, 2), float(calls["n"]))

    fn = backbone._cached_weight(("k",), raw)

    # Outside the phase: every call reconstructs.
    fn(); fn()
    assert calls["n"] == 2

    # Inside: first call stores, second hits (same tensor object) and evicts,
    # third is a fresh miss (stored again).
    with backbone.backward_dequant_cache():
        w1 = fn()
        assert calls["n"] == 3
        w2 = fn()
        assert calls["n"] == 3 and w2 is w1
        fn()
        assert calls["n"] == 4

    # The unpaired store above must NOT leak into a new phase.
    with backbone.backward_dequant_cache():
        fn()
        assert calls["n"] == 5, "stale weight leaked across phases"

    # enable=False is a passthrough.
    with backbone.backward_dequant_cache(enable=False):
        fn(); fn()
    assert calls["n"] == 7
    print("  backward cache (passthrough/store/hit/evict/no-leak): OK")


def _collect_linears(module, out):
    from exllamav3.modules import Linear
    if isinstance(module, Linear):
        out.append(module)
    for child in getattr(module, "modules", []):
        _collect_linears(child, out)



@pytest.fixture
def model_dir() -> str:
    """Opt-in real-model tier: EXL3_TEST_MODEL names a quantized EXL3 model directory. Shadows the
    registry-backed conftest fixture, which needs a model() marker these standalone tests don't carry."""
    path = os.environ.get("EXL3_TEST_MODEL")
    if not path:
        pytest.skip("opt-in: set EXL3_TEST_MODEL to an EXL3 model directory")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return path


def test_real_exl3_layer(model_dir: str):
    """Tier 5 (GPU, opt-in): fast forward vs x @ get_weight_tensor() with the
    RUNTIME Hadamard on real quantized linears -- pins the synthetic-H tests
    to the shipped tables, through the same backbone seam the trainer uses."""
    from exllamav3 import Config, Model
    from exllamav3.training import backbone

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    model.load(device="cuda:0", progressbar=False)

    _, blocks, _, _ = backbone.split_decoder(model)
    lins = []
    _collect_linears(blocks[0], lins)
    _collect_linears(blocks[len(blocks) // 2], lins)

    checked = 0
    for lin in lins:
        parts = backbone.frozen_trellis_parts(lin)
        if parts is None:
            continue
        inner_fn, suh, svh = parts
        had = backbone.hadamard_128(suh.device, suh.dtype)
        w_eff = lin.inner.get_weight_tensor()          # [in, out] half
        x = (torch.randn(64, lin.in_features, device=suh.device,
                         dtype=torch.float16) * 0.3)
        y_fast = EXL3LoRAHadFunction.apply(
            x, None, None, None, 1.0, inner_fn, suh, svh, had,
            None, None, 1.0)
        y_ref = x @ w_eff
        rel = ((y_fast - y_ref).norm() / y_ref.norm()).item()
        assert rel < 5e-3, f"{lin.key}: fast vs get_weight_tensor rel {rel:.2e}"

        # S36: the bf16-emitting reconstruct must match fp16-reconstruct +
        # .to(bf16) bit-for-bit (same fp16 dequant math, one RN rounding at
        # the store), and the bf16 fast path must stay in the bf16 error
        # class vs an fp32 ground truth.
        w16 = lin.inner.get_inner_weight_tensor()
        wbf = lin.inner.get_inner_weight_tensor(out_dtype=torch.bfloat16)
        assert (wbf == w16.to(torch.bfloat16)).all(), \
            f"{lin.key}: bf16 reconstruct not bit-exact vs fp16 + cast"
        pbf = backbone.frozen_trellis_parts(lin, torch.bfloat16)
        ifn_bf, suh_bf, svh_bf = pbf
        assert suh_bf.dtype == svh_bf.dtype == torch.bfloat16
        had_bf = backbone.hadamard_128(suh_bf.device, suh_bf.dtype)
        y_bf = EXL3LoRAHadFunction.apply(
            x.to(torch.bfloat16), None, None, None, 1.0,
            ifn_bf, suh_bf, svh_bf, had_bf, None, None, 1.0)
        y_true = x.float() @ w_eff.float()
        rel_bf = ((y_bf.float() - y_true).norm() / y_true.norm()).item()
        assert rel_bf < 1e-2, f"{lin.key}: bf16 fast path rel {rel_bf:.2e}"

        checked += 1
        print(f"  {lin.key}: fast vs W_eff rel_err {rel:.2e} "
              f"(bf16 path {rel_bf:.2e}, emission bit-exact)  OK")
    assert checked > 0, "no trellis linears found to check"
    print(f"  real-layer parity on {checked} linears: OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("EXL3_TEST_MODEL") or None,
                    help="EXL3 model dir for the tier-5 real-layer test (GPU)")
    args = ap.parse_args()

    tests = [
        test_gradcheck_f64,
        test_matches_legacy_reference,
        test_mixed_dtype_paths,
        test_backward_cache,
    ]
    if args.model:
        tests.append(functools.partial(test_real_exl3_layer, args.model))
    else:
        print("  (skipping tier-5 real-layer test; pass --model or set "
              "EXL3_TEST_MODEL)")
    run_timed(tests, label="test_dequant_fast")
    print("ALL PASS")


if __name__ == "__main__":
    main()
