"""CPU-only import/availability regressions for the optional DSA BC path.

Load the real module with isolated dependency stubs so these tests do not need
Torch, Triton, the native extension, or model weights.
"""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "exllamav3/modules/attention_fn/bc_dsa.py"
PACKAGE = "_bc_dsa_import_test"
KERNEL_NAMES = (
    "_dsa_attn_split_kernel",
    "_dsa_attn_combine_kernel",
    "_dsa_indexer_fewq_kernel",
)


def load_bc_dsa(has_triton, enabled = None, debug = "0", missing_kernel = None):
    modules = {}
    for name in (PACKAGE, f"{PACKAGE}.modules", f"{PACKAGE}.modules.attention_fn"):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module

    def dependency(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        modules[name] = module
        return module

    dependency("torch")
    dependency(f"{PACKAGE}.ext", exllamav3_ext = object())
    dependency(f"{PACKAGE}.constants", PAGE_SIZE = 256)
    dependency(f"{PACKAGE}.util.tensor", g_tensor_cache = object())
    dependency(f"{PACKAGE}.modules.attention_fn.bc_attn", _compile_kernel = Mock())
    kernels = {name: object() for name in KERNEL_NAMES if name != missing_kernel} if has_triton else {}
    dependency(f"{PACKAGE}.modules.attention_fn.dsa_triton", has_triton = has_triton, **kernels)

    env = dict(os.environ)
    env.pop("EXL3_BC_DSA", None)
    if enabled is not None:
        env["EXL3_BC_DSA"] = enabled
    env["EXL3_BC_DSA_DEBUG"] = debug

    name = f"{PACKAGE}.modules.attention_fn.bc_dsa"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    modules[name] = module
    with patch.dict(sys.modules, modules), patch.dict(os.environ, env, clear = True):
        spec.loader.exec_module(module)
    return module, kernels


class TestBCDsaImport(unittest.TestCase):

    def test_import_without_triton_disables_bc(self):
        for enabled in (None, "0", "1"):
            with self.subTest(enabled = enabled):
                module, _ = load_bc_dsa(False, enabled = enabled)
                self.assertFalse(module.bc_dsa_enable)
                for name in KERNEL_NAMES:
                    self.assertFalse(hasattr(module, name))

    def test_builders_decline_without_triton_even_in_debug_mode(self):
        for debug in ("0", "1"):
            with self.subTest(debug = debug):
                module, _ = load_bc_dsa(False, debug = debug)
                for builder_name, class_name, count in (
                    ("build_bc_dsa", "BCDsa", 4),
                    ("build_bc_dsa_batch", "BCDsaBatch", 3),
                ):
                    with patch.object(module, class_name) as constructor:
                        result = getattr(module, builder_name)(*([None] * count))
                        self.assertIsNone(result)
                        constructor.assert_not_called()

    def test_available_triton_preserves_kernels_and_environment_switch(self):
        for enabled, expected in ((None, True), ("0", False), ("1", True)):
            with self.subTest(enabled = enabled):
                module, kernels = load_bc_dsa(True, enabled = enabled)
                self.assertEqual(module.bc_dsa_enable, expected)
                for name, kernel in kernels.items():
                    self.assertIs(getattr(module, name), kernel)

    def test_available_triton_does_not_hide_missing_kernel_errors(self):
        for name in KERNEL_NAMES:
            with self.subTest(kernel = name), self.assertRaises(ImportError):
                load_bc_dsa(True, missing_kernel = name)

    def test_available_triton_preserves_builder_results_and_error_policy(self):
        for debug in ("0", "1"):
            module, _ = load_bc_dsa(True, debug = debug)
            for builder_name, class_name, count in (
                ("build_bc_dsa", "BCDsa", 4),
                ("build_bc_dsa_batch", "BCDsaBatch", 3),
            ):
                with self.subTest(debug = debug, builder = builder_name):
                    args = tuple(object() for _ in range(count))
                    builder = getattr(module, builder_name)
                    sentinel = object()
                    with patch.object(module, class_name, return_value = sentinel) as constructor:
                        self.assertIs(builder(*args), sentinel)
                        constructor.assert_called_once_with(*args)
                    with patch.object(module, class_name, side_effect = RuntimeError("build failed")):
                        if debug == "1":
                            with self.assertRaisesRegex(RuntimeError, "build failed"):
                                builder(*args)
                        else:
                            self.assertIsNone(builder(*args))


if __name__ == "__main__":
    unittest.main()
