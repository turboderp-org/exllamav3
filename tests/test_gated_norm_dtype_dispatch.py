"""KDA norm dtype dispatch contracts; native CUDA is explicitly opt-in."""
from __future__ import annotations
import ast
from itertools import product
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'exllamav3/modules/gated_rmsnorm.py'


def forward_with_adapter(native):
    cls = next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'GatedRMSNorm')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    method.decorator_list = []
    tree = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[]))
    scope = {'torch': torch, 'ext': SimpleNamespace(gated_rms_norm=native)}
    exec(compile(tree, str(SOURCE), 'exec'), scope)
    return scope['forward']


def independent(x, weight, gate, dtype):
    # Independent high-precision equation, rounded once to the requested output.
    # No production norm/helper participates in the expected values.
    a, w, g = x.double(), weight.double(), gate.double()
    return (a / (a.square().mean(-1, keepdim=True) + 1e-6).sqrt()
            * w / (1 + (-g).exp())).to(dtype)


def fixture(xdtype=torch.bfloat16, wdtype=torch.float16, gdtype=torch.float32, outdtype=torch.float16):
    x = torch.linspace(-1.7, 2.3, 256).view(2, 128).to(xdtype)
    w = torch.linspace(0.7, 1.3, 128).to(wdtype)
    g = torch.linspace(-2.0, 1.8, 256).view(2, 128).to(gdtype)
    obj = SimpleNamespace(gate_activation='sigmoid', out_dtype=outdtype, weight=w,
                          rms_norm_eps=1e-6, constant_bias=0.0, groups=1, gate_first=False)
    return obj, x, g


class DtypeDispatch(unittest.TestCase):
    def check_fallback(self, obj, x, gate, override=None):
        native = Mock(side_effect=AssertionError('unsupported native dispatch'))
        before = (obj.weight, obj.weight.data_ptr(), obj.weight.dtype, obj.weight.clone())
        actual = forward_with_adapter(native)(obj, x, {}, out_dtype=override, gate=gate)
        dtype = override or obj.out_dtype or x.dtype
        # FP64 oracle and FP32 implementation can straddle an output rounding
        # boundary: permit one relative output ULP (or the FP32 arithmetic bound).
        torch.testing.assert_close(actual, independent(x, obj.weight, gate, dtype),
                                   rtol=max(1e-5, torch.finfo(dtype).eps), atol=2e-6)
        native.assert_not_called()
        self.assertIs(obj.weight, before[0])
        self.assertEqual(obj.weight.data_ptr(), before[1])
        self.assertEqual(obj.weight.dtype, before[2])
        self.assertTrue(torch.equal(obj.weight, before[3]))
        self.assertEqual(actual.dtype, dtype)

    def test_half_weight_uses_fallback_without_rebinding(self):
        self.check_fallback(*fixture())

    def test_half_gate_uses_fallback(self):
        self.check_fallback(*fixture(wdtype=torch.bfloat16, gdtype=torch.float16))

    def test_bfloat_output_and_default_output_use_fallback(self):
        for dtype in (torch.bfloat16, None):
            with self.subTest(dtype=dtype):
                self.check_fallback(*fixture(wdtype=torch.bfloat16, outdtype=dtype))
        self.check_fallback(*fixture(wdtype=torch.float32), override=torch.bfloat16)

    def test_float_input_uses_fallback(self):
        self.check_fallback(*fixture(xdtype=torch.float32, wdtype=torch.float32, outdtype=torch.float32))

    def test_noncontiguous_arguments_use_fallback(self):
        for field in ('x', 'weight', 'gate'):
            with self.subTest(field=field):
                obj, x, g = fixture(wdtype=torch.bfloat16)
                if field == 'x':
                    x = x.repeat_interleave(2, dim=-1)[..., ::2]
                elif field == 'weight':
                    obj.weight = obj.weight.repeat_interleave(2)[::2]
                else:
                    g = g.repeat_interleave(2, dim=-1)[..., ::2]
                self.check_fallback(obj, x, g)

    def test_existing_native_supported_combinations_still_dispatch(self):
        for w, g, out in product((torch.bfloat16, torch.float32), (torch.bfloat16, torch.float32), (torch.float16, torch.float32)):
            with self.subTest(weight=w, gate=g, output=out):
                obj, x, gate = fixture(wdtype=w, gdtype=g, outdtype=out)
                native = Mock(side_effect=lambda x, w, y, *args: y.zero_())
                actual = forward_with_adapter(native)(obj, x, {}, gate=gate)
                native.assert_called_once()
                self.assertEqual(actual.dtype, out)
                self.assertIs(native.call_args.args[1], obj.weight)
                self.assertEqual(native.call_args.args[-1], 1)

    def test_silu_dispatch_is_unchanged(self):
        obj, x, gate = fixture(wdtype=torch.bfloat16)
        obj.gate_activation = 'silu'
        native = Mock(side_effect=lambda x, w, y, *args: y.zero_())
        forward_with_adapter(native)(obj, x, {}, gate=gate)
        native.assert_called_once()
        self.assertEqual(native.call_args.args[-1], 0)


@unittest.skipUnless(os.environ.get('EXL3_TEST_GATED_NORM_GPU') == '1', 'explicit native CUDA opt-in required')
class NativeDtypes(unittest.TestCase):
    def test_native_and_fallback_54_dtype_combinations(self):
        from exllamav3.modules import GatedRMSNorm
        count = 0
        for inp, weight, gate, out in product((torch.bfloat16, torch.float32),
                (torch.float16, torch.bfloat16, torch.float32),
                (torch.float16, torch.bfloat16, torch.float32),
                (torch.float16, torch.bfloat16, torch.float32)):
            with self.subTest(input=inp, weight=weight, gate=gate, output=out):
                obj, x, g = fixture(inp, weight, gate, out)
                norm = GatedRMSNorm(None, 'dtype-contract', 1e-6, out_dtype=out, gate_activation='sigmoid')
                norm.weight = torch.nn.Parameter(obj.weight.cuda(), requires_grad=False)
                original_pointer = norm.weight.data_ptr()
                actual = norm.forward(x.cuda(), {}, gate=g.cuda()).cpu()
                torch.testing.assert_close(actual, independent(x, obj.weight, g, out),
                                           rtol=max(1e-5, torch.finfo(out).eps), atol=2e-6)
                self.assertEqual(norm.weight.data_ptr(), original_pointer)
                self.assertEqual(norm.weight.dtype, weight)
                self.assertTrue(torch.equal(norm.weight.cpu(), obj.weight))
                count += 1
        self.assertEqual(count, 54)
        torch.cuda.synchronize()
        print('NATIVE_KDA_NORM_DTYPE_CASES', count)


if __name__ == '__main__':
    unittest.main()
