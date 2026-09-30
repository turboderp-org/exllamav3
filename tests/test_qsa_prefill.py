"""CPU-only dispatch contracts; these do not validate the CUDA kernels."""

from contextlib import nullcontext
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.modules.attention_fn import qsa_prefill, qsa_triton, triton_paged


def descriptor(shape, dtype=torch.int32):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype,
                           is_cuda=True, device=torch.device("cuda:0"),
                           is_contiguous=lambda: True,
                           numel=lambda: shape[0])


def fixture(rows=512, length=4096):
    pages = (length + 255) // 256
    q = descriptor((rows, 24, 256), torch.float16)
    kv = descriptor((pages, 256, 128))
    scales = descriptor((pages, 256))
    args = (q, kv, kv, descriptor((rows, 2080)), 0.0625,
            descriptor((rows, pages)), 256, (scales, scales, 8, 8), 2)
    context = {"length": length, "block_table": descriptor((pages,))}
    return args, context


class DispatchChecks(unittest.TestCase):
    def dispatch(self, args, context):
        return qsa_triton.qsa_sparse_attend_rows(*args, prefill_context=context)

    def test_disabled_keeps_native_path(self):
        args, context = fixture()
        with patch.object(qsa_triton, "_qsa_prefill_enabled", False), \
             patch.object(qsa_triton, "_qsa_sparse_attend_rows_native", return_value="native") as native, \
             patch.object(qsa_prefill, "try_attend", side_effect=AssertionError("disabled")):
            self.assertEqual(self.dispatch(args, context), "native")
            native.assert_called_once_with(*args)

    def test_shared_pool_probe_falls_back_before_cuda_or_allocation(self):
        args, context = fixture(length=1048576)
        with patch.object(qsa_triton, "_qsa_prefill_enabled", True), \
             patch.object(qsa_triton, "_qsa_sparse_attend_rows_native", return_value="native"), \
             patch("torch.cuda.get_device_capability", side_effect=AssertionError("must not query CUDA")), \
             patch("torch.empty", side_effect=AssertionError("must not allocate")), \
             patch.object(qsa_prefill, "stage_qkv", side_effect=AssertionError("must not stage")):
            self.assertEqual(self.dispatch(args, context), "native")

    def test_full_supported_window_remains_optimized(self):
        args, context = fixture(length=262144)
        with patch.object(qsa_triton, "_qsa_prefill_enabled", True), \
             patch.object(qsa_triton, "_qsa_sparse_attend_rows_native", side_effect=AssertionError("not native")), \
             patch("torch.cuda.get_device_capability", return_value=(12, 1)), \
             patch("torch.cuda.device", return_value=nullcontext()), \
             patch.object(qsa_prefill, "_attend", return_value="optimized") as optimized:
            self.assertEqual(self.dispatch(args, context), "optimized")
            optimized.assert_called_once_with(*args, context)

    def test_decode_and_missing_context_stay_native(self):
        for rows, length in ((1, 4096), (255, 4096), (512, 511)):
            args, context = fixture(rows=rows, length=length)
            self.assertIsNone(qsa_prefill.try_attend(*args, context=context))
        args, _ = fixture()
        self.assertIsNone(qsa_prefill.try_attend(*args))

    def test_unvalidated_layout_and_device_fall_back(self):
        args, context = fixture()
        with patch("torch.cuda.get_device_capability", return_value=(8, 0)), \
             patch.object(qsa_prefill, "_attend", side_effect=AssertionError("unsupported GPU")):
            self.assertIsNone(qsa_prefill.try_attend(*args, context=context))
        args[0].dtype = torch.bfloat16
        self.assertIsNone(qsa_prefill.try_attend(*args, context=context))
        args, context = fixture()
        context["block_table"] = descriptor((1,))
        self.assertIsNone(qsa_prefill.try_attend(*args, context=context))
        args, context = fixture()
        args = (*args[:7], (*args[7][:2], 4, 4), args[8])
        self.assertIsNone(qsa_prefill.try_attend(*args, context=context))

    def test_dense_prefix_uses_call_local_staging_override(self):
        q = torch.empty((256, 24, 256), dtype=torch.float16)
        table = torch.zeros((256, 1), dtype=torch.int32)
        context = {"length": 256, "block_table": table[0]}
        previous = triton_paged._qc_staging
        with patch.object(triton_paged, "paged_attn_triton_prefill", return_value=q.unsqueeze(0)) as dense:
            result = qsa_prefill._attend(q, None, None, None, .0625, table, 256,
                                         (None, None, 8, 8), 2, context)
            self.assertEqual(result.shape, q.shape)
            self.assertIs(dense.call_args.kwargs["qc_staging"], False)
            self.assertEqual(triton_paged._qc_staging, previous)

    def test_short_sparse_remainder_keeps_native_path(self):
        args, context = fixture(rows=128)
        with patch.object(qsa_triton, "_qsa_sparse_attend_rows_native", return_value="native") as native:
            self.assertEqual(qsa_prefill._sparse_attend(*args, context), "native")
            native.assert_called_once_with(*args)


if __name__ == "__main__":
    unittest.main()
