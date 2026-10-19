"""CPU coverage of EXL3 slice tiling without importing the CUDA extension.

Execute the production method and Hadamard helpers with a dense stand-in for
the CUDA dequantizer. This tests transform math, slice offsets and workspace
bounds; it does not validate the CUDA dequantizer itself.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import unittest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(path, names, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef)
                    and n.name == class_name).body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def load_constants(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "constants", "exec"), namespace)


def check_weight_slice_transform_and_workspace(start, width):
    torch.manual_seed(123)
    k, n = 256, start + width + 128
    inner = torch.randn(k, n).half() * 0.02
    su, sv = torch.randn(k).half(), torch.randn(n).half()
    h = torch.ones(1, 1)
    while h.shape[0] < 128:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    ns = dict(torch=torch, math=math, had_k=128, had_n=128,
              get_hadamard_dt=lambda dim, device, dtype, scale: h.to(device=device, dtype=dtype) * scale)
    load_constants("exllamav3/modules/quant/exl3_lib/quantize.py", {"_HAD_MAX_GEMM_ROWS"}, ns)
    load_definitions("exllamav3/modules/quant/exl3_lib/quantize.py",
                     {"preapply_had_l", "preapply_had_r"}, ns)
    left, right = ns["preapply_had_l"], ns["preapply_had_r"]
    expected = left(inner, 128)
    expected *= su[:, None]
    expected = right(expected, 128)
    expected *= sv[None, :]

    calls, transform_widths = [], []

    def reconstruct(out, trellis, K, mcg, mul1, offset):
        assert out.is_contiguous()
        calls.append((offset, out.shape[1]))
        out.copy_(inner[:, offset:offset + out.shape[1]])

    def track(fn):
        def wrapped(x, dim):
            transform_widths.append(x.shape[1])
            return fn(x, dim)
        return wrapped

    # Read the real constants too, so the workspace assertion detects changes.
    load_constants("exllamav3/modules/quant/exl3.py",
                   {"MAX_WEIGHT_TRANSFORM_SLICE_N", "RECONSTRUCT_SLICE_GRANULARITY_N"}, ns)
    ns.update(ext=SimpleNamespace(reconstruct_slice=reconstruct),
              preapply_had_l=track(left), preapply_had_r=track(right))
    load_definitions("exllamav3/modules/quant/exl3.py", {"get_weight_tensor_slice"}, ns, "LinearEXL3")
    layer = SimpleNamespace(in_features=k, out_features=n, su=None, sv=None,
                            suh=su, svh=sv, trellis=inner, K=3, mcg=False, mul1=False)
    actual = ns["get_weight_tensor_slice"](layer, start, width)
    torch.testing.assert_close(actual, expected[:, start:start + width], atol=1e-4, rtol=1e-3)
    assert actual.is_contiguous() and actual.dtype == torch.half
    assert max(transform_widths) <= 2048
    assert calls == [(start + off, min(2048, width - off)) for off in range(0, width, 2048)]
    with unittest.TestCase().assertRaises(AssertionError):
        ns["get_weight_tensor_slice"](layer, 1, 128)
    with unittest.TestCase().assertRaises(AssertionError):
        ns["get_weight_tensor_slice"](layer, n, 128)


class WeightSliceWorkspaceTests(unittest.TestCase):
    def test_transform_and_workspace(self):
        for start, width in [(0, 128), (128, 2048), (128, 4480)]:
            with self.subTest(start=start, width=width):
                check_weight_slice_transform_and_workspace(start, width)


if __name__ == "__main__":
    unittest.main()
