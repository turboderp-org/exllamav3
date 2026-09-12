"""ROCm coverage for portable compiled rewind/cache bindings."""
from __future__ import annotations

import pytest
import torch

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm rewind binding tests", allow_module_level = True)

from exllamav3.ext import exllamav3_ext as ext


def test_portable_rewind_and_cache_bindings_exist_and_state_rewind_copies():
    for name in (
        "ConvRewindJob", "StateRewindJob", "batched_conv_rewind",
        "batched_state_rewind", "dspark_write_rows",
    ):
        assert hasattr(ext, name), f"missing ROCm binding: {name}"

    state = torch.arange(4 * 8, device = "cuda", dtype = torch.float32).reshape(4, 8)
    expected = state[3].clone()
    job = ext.StateRewindJob(state[3].data_ptr(), state[0].data_ptr(), state[0].numel())
    ext.batched_state_rewind([job], 0)
    torch.cuda.synchronize()

    torch.testing.assert_close(state[0], expected)
