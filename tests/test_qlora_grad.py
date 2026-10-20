"""
Gradient-correctness proof of concept for QLoRA-over-EXL3.

This is the "is it even real?" spike from the feasibility writeup. It
answers one question: can we get *correct* gradients for the LoRA adapters
(and for the input, so loss can flow to earlier layers) through a frozen,
dequantized EXL3 weight?

Three tiers, in increasing hardware requirements:

  1. test_gradcheck_cpu_f64
        Rigorous finite-difference gradcheck of the hand-written backward
        in EXL3LoRAFunction, in float64 on CPU. Requires only torch. This
        is the real proof.

  2. test_matches_reference
        The memory-efficient custom Function must produce gradients
        identical to the plain-autograd reference_forward. Analytic vs
        analytic, so exact.

  3. test_real_exl3_layer  (opt-in, needs GPU + a model + built extension)
        Wrap one real EXL3 linear from a converted model, confirm that
        x @ get_weight_tensor() matches the layer's own kernel forward, and
        that the custom backward agrees with autograd in float32.

Run directly::

    python tests/test_qlora_grad.py
    python tests/test_qlora_grad.py --model /path/to/exl3/model   # tier 3

or under pytest (tier 3 is skipped unless EXL3_TEST_MODEL is set).
"""

from __future__ import annotations
import os
import sys
import argparse
import importlib.util
import torch
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# Load qlora_linear directly from its file. It only depends on torch (no
# intra-package imports), so tiers 1-2 run on any machine -- importing the
# exllamav3 package would otherwise trigger a CUDA extension build, which is
# only actually needed for the tier-3 real-model test.
_spec = importlib.util.spec_from_file_location(
    "exllamav3_qlora_linear",
    os.path.join(_ROOT, "exllamav3", "training", "qlora_linear.py"),
)
_qll = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qll)

EXL3LoRAFunction = _qll.EXL3LoRAFunction
reference_forward = _qll.reference_forward
qlora_linear_forward = _qll.qlora_linear_forward
QLoRALinear = _qll.QLoRALinear


def _mock_weight_fn(weight: torch.Tensor):
    """A constant weight provider standing in for EXL3 reconstruction."""
    return lambda: weight


def test_gradcheck_cpu_f64():
    """Finite-difference gradcheck of the custom backward (the real proof)."""
    torch.manual_seed(0)
    in_f, out_f, r, n = 24, 16, 4, 8
    dt = torch.float64

    weight = torch.randn(in_f, out_f, dtype=dt)            # frozen base
    x = torch.randn(n, in_f, dtype=dt, requires_grad=True)
    a = torch.randn(in_f, r, dtype=dt, requires_grad=True)
    b = torch.randn(r, out_f, dtype=dt, requires_grad=True)
    bias = torch.randn(out_f, dtype=dt, requires_grad=True)
    scale = 0.5

    ok = torch.autograd.gradcheck(
        lambda x_, a_, b_, bias_: EXL3LoRAFunction.apply(
            x_, a_, b_, bias_, scale, _mock_weight_fn(weight)
        ),
        (x, a, b, bias),
        eps=1e-6, atol=1e-6, rtol=1e-4,
    )
    assert ok
    print("[tier 1] gradcheck (cpu/f64) PASSED")


def test_gradcheck_no_bias():
    """Same, with bias=None, to exercise the optional-grad paths."""
    torch.manual_seed(1)
    in_f, out_f, r, n = 20, 12, 3, 5
    dt = torch.float64
    weight = torch.randn(in_f, out_f, dtype=dt)
    x = torch.randn(n, in_f, dtype=dt, requires_grad=True)
    a = torch.randn(in_f, r, dtype=dt, requires_grad=True)
    b = torch.randn(r, out_f, dtype=dt, requires_grad=True)
    scale = 1.7
    ok = torch.autograd.gradcheck(
        lambda x_, a_, b_: EXL3LoRAFunction.apply(
            x_, a_, b_, None, scale, _mock_weight_fn(weight)
        ),
        (x, a, b),
        eps=1e-6, atol=1e-6, rtol=1e-4,
    )
    assert ok
    print("[tier 1] gradcheck no-bias (cpu/f64) PASSED")


def test_matches_reference():
    """Custom memory-efficient Function == plain-autograd reference."""
    torch.manual_seed(2)
    in_f, out_f, r, n = 32, 48, 8, 16
    dt = torch.float64
    weight = torch.randn(in_f, out_f, dtype=dt)
    scale = 0.3

    def make_leaves():
        x = torch.randn(n, in_f, dtype=dt, requires_grad=True)
        a = torch.randn(in_f, r, dtype=dt, requires_grad=True)
        b = torch.randn(r, out_f, dtype=dt, requires_grad=True)
        bias = torch.randn(out_f, dtype=dt, requires_grad=True)
        return x, a, b, bias

    # Reference (autograd does everything)
    x1, a1, b1, bias1 = make_leaves()
    y1 = reference_forward(x1, weight, a1, b1, scale, bias1)
    y1.pow(2).sum().backward()

    # Custom Function (hand-written backward, recomputes weight)
    x2, a2, b2, bias2 = (t.detach().clone().requires_grad_(True)
                         for t in (x1, a1, b1, bias1))
    y2 = qlora_linear_forward(x2, _mock_weight_fn(weight), a2, b2, scale, bias2)
    y2.pow(2).sum().backward()

    assert torch.allclose(y1, y2, atol=1e-9)
    for name, g1, g2 in [
        ("grad_x", x1.grad, x2.grad),
        ("grad_a", a1.grad, a2.grad),
        ("grad_b", b1.grad, b2.grad),
        ("grad_bias", bias1.grad, bias2.grad),
    ]:
        assert torch.allclose(g1, g2, atol=1e-8), f"{name} mismatch"
    print("[tier 2] custom Function matches autograd reference PASSED")


def test_module_b_zero_init_is_noop():
    """A freshly-initialised QLoRALinear must equal the base projection."""
    torch.manual_seed(3)
    in_f, out_f = 16, 16
    weight = torch.randn(in_f, out_f, dtype=torch.float64)
    mod = QLoRALinear(
        weight_fn=_mock_weight_fn(weight),
        in_features=in_f, out_features=out_f,
        r=4, alpha=8.0, dtype=torch.float64,
    )
    x = torch.randn(5, in_f, dtype=torch.float64)
    y = mod(x)
    assert torch.allclose(y, x @ weight, atol=1e-9), "B=0 init should be a no-op"
    print("[tier 2] QLoRALinear zero-init no-op PASSED")



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
    """
    Tier 3: real EXL3 weight, on GPU. Opt-in.

    Loads a converted model, picks the first quantized Linear, and:
      (a) checks x @ get_weight_tensor() matches the layer's kernel forward
          (forward parity -- confirms our weight orientation is right), and
      (b) checks the custom backward agrees with autograd in float32.
    """
    assert torch.cuda.is_available(), "tier 3 needs CUDA"
    from exllamav3 import Config, Model
    from exllamav3.modules import Linear
    from exllamav3.modules.quant import LinearEXL3

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    model.load()

    # Find a loaded EXL3 linear.
    target = None
    for m in model:
        if isinstance(m, Linear) and isinstance(getattr(m, "inner", None), LinearEXL3):
            target = m
            break
    assert target is not None, "No EXL3 Linear found in model"
    dev = target.device
    in_f, out_f = target.in_features, target.out_features
    print(f" -- using layer {target.key}  [{in_f} -> {out_f}] on {dev}")

    # (a) Forward parity: dequant-matmul vs the layer's own kernel forward.
    x = torch.randn(64, in_f, dtype=torch.float16, device=dev)
    with torch.inference_mode():
        y_kernel = target.inner.forward(x, params={"reconstruct": True})
        w = target.inner.get_weight_tensor()           # [in, out], half
        y_dequant = x @ w
    rel = (y_kernel.float() - y_dequant.float()).norm() / y_kernel.float().norm().clamp_min(1e-6)
    print(f" -- forward parity relative error: {rel.item():.3e}")
    assert rel < 1e-2, "Dequant-matmul does not match kernel forward"

    # (b) Backward agreement in float32.
    wf = lambda: target.inner.get_weight_tensor().float()
    r = 8
    xf = torch.randn(32, in_f, device=dev, dtype=torch.float32, requires_grad=True)
    a = torch.randn(in_f, r, device=dev, dtype=torch.float32, requires_grad=True) * 0.01
    b = torch.randn(r, out_f, device=dev, dtype=torch.float32, requires_grad=True) * 0.01
    a, b = a.detach().requires_grad_(True), b.detach().requires_grad_(True)
    scale = 2.0

    y_ref = reference_forward(xf, wf(), a, b, scale)
    y_ref.pow(2).sum().backward()
    gx_ref, ga_ref, gb_ref = xf.grad.clone(), a.grad.clone(), b.grad.clone()

    xf.grad = a.grad = b.grad = None
    y_cus = qlora_linear_forward(xf, wf, a, b, scale)
    y_cus.pow(2).sum().backward()

    for name, gr, gc in [("grad_x", gx_ref, xf.grad), ("grad_a", ga_ref, a.grad), ("grad_b", gb_ref, b.grad)]:
        rel = (gr - gc).norm() / gr.norm().clamp_min(1e-6)
        print(f" -- {name} relative error (custom vs autograd): {rel.item():.3e}")
        assert rel < 1e-4, f"{name} mismatch on real layer"
    print("[tier 3] real EXL3 layer PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("EXL3_TEST_MODEL"),
                    help="Path to a converted EXL3 model dir (enables tier 3)")
    args = ap.parse_args()

    from util import run_timed
    from functools import partial
    tests = [
        test_gradcheck_cpu_f64,
        test_gradcheck_no_bias,
        test_matches_reference,
        test_module_b_zero_init_is_noop,
    ]
    if args.model:
        tests.append(partial(test_real_exl3_layer, args.model))
    else:
        print("[tier 3] skipped (pass --model or set EXL3_TEST_MODEL to run on GPU)")
    run_timed(tests, label="qlora-grad")

    print("\nAll runnable QLoRA gradient PoC checks passed.")


if __name__ == "__main__":
    main()
