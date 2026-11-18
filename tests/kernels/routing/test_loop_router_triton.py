"""Compare the fused router with the original FP32 torch implementation."""
import os
import importlib.util
import pathlib
import sys
import types

import pytest
import torch

# Load these isolated leaf modules without building the optional native extension.
_root = pathlib.Path(__file__).resolve().parents[3] / "exllamav3" / "modules"
_package = types.ModuleType("_loop_router_test")
_package.__path__ = [str(_root)]
sys.modules[_package.__name__] = _package
_module = types.ModuleType("_loop_router_test.module")
_module.Module = object
sys.modules[_module.__name__] = _module


def _load(name):
    spec = importlib.util.spec_from_file_location("_loop_router_test." + name, _root / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


route = _load("loop_router").route
route_triton = _load("loop_router_triton").route_triton


def check_router(device, streams, fixed, dtype, rows, large):
    torch.manual_seed(37 + rows + streams)
    hidden = 17
    outputs = 2 * streams + (0 if fixed else streams * streams)
    x = torch.randn(rows, streams, hidden, device = device).to(dtype)
    weight = torch.randn(outputs, streams * hidden, device = device) * (2.0 if large else 0.1)
    bias = torch.randn(outputs, device = device) * (8.0 if large else 0.1)
    expected = route(x, weight, bias, streams, 20, fixed, 1.3)
    actual = route_triton(x, weight, bias, streams, 20, fixed, 1.3)
    for name, a, e in zip(("branch", "post", "mix"), actual, expected):
        diff = (a.float() - e.float()).abs()
        rel = diff / e.float().abs().clamp_min(1.e-30)
        print(f"{device} s={streams} fixed={fixed} dtype={dtype} rows={rows} large={large} {name}: max abs={diff.max().item():.3g} rel={rel.max().item():.3g}")
        # Low-precision branch values are rounded after the matmul and scaling.
        # Triton 3.9 interpreter BF16 casts differ from hardware rounding.
        atol = (0.08 if device.type == "cpu" else 0.032) if name == "branch" and dtype == torch.bfloat16 else 0.004 if name == "branch" and dtype == torch.float16 else 3.e-5
        torch.testing.assert_close(a, e, rtol = 5.e-5, atol = atol)


@pytest.mark.cuda_only
@pytest.mark.parametrize("streams", [2, 4])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,large", [(1, False), (3, True), (256, False)])
def test_gpu(device, streams, fixed, dtype, rows, large, monkeypatch):
    monkeypatch.setenv("EXL3_LOOP_ROUTER_TRITON", "0")
    check_router(device, streams, fixed, dtype, rows, large)


@pytest.mark.nogpu
@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") != "1", reason = "requires Triton interpreter")
@pytest.mark.parametrize("streams", [2, 4])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,large", [(1, False), (3, True), (256, False)])
def test_interpreter(streams, fixed, dtype, rows, large):
    check_router(torch.device("cpu"), streams, fixed, dtype, rows, large)


@pytest.mark.cuda_only
def test_cuda_dispatch_and_fallback(device, monkeypatch):
    x = torch.randn(3, 2, 33, device = device, dtype = torch.float16)
    weight = torch.randn(8, 66, device = device) * 0.1
    bias = torch.randn(8, device = device)
    monkeypatch.setenv("EXL3_LOOP_ROUTER_TRITON", "0")
    expected = route(x, weight, bias, 2, 20, False, 1.0)
    monkeypatch.setenv("EXL3_LOOP_ROUTER_TRITON", "1")
    actual = route(x, weight, bias, 2, 20, False, 1.0)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol = 5.e-5, atol = 0.003 if a.dtype == torch.float16 else 3.e-5)
