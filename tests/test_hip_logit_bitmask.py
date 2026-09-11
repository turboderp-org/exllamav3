"""ROCm contracts for the packed eager logit-mask fallback."""
from __future__ import annotations

import pytest
import torch

if not (torch.version.hip and torch.cuda.is_available()):
    pytest.skip("ROCm packed-logit-bitmask tests", allow_module_level = True)

from exllamav3.ext import exllamav3_ext as ext

DEVICE = "cuda"


def _packed(rows, words, enabled):
    values = [[0] * words for _ in range(rows)]
    for row, index in enabled:
        values[row][index >> 5] |= 1 << (index & 31)
    # torch.int32 accepts signed words; reinterpret values containing bit 31.
    for row in values:
        for word, value in enumerate(row):
            if value >= 1 << 31:
                row[word] = value - (1 << 32)
    return torch.tensor(values, dtype = torch.int32, device = DEVICE)


def _reference(logits, bitmask):
    rows, dim = logits.shape
    keep = torch.zeros((rows, dim), dtype = torch.bool, device = DEVICE)
    for row in range(rows):
        bits = bitmask[0 if bitmask.shape[0] == 1 else row]
        for index in range(min(dim, bitmask.shape[1] * 32)):
            keep[row, index] = bool((int(bits[index >> 5].item()) & 0xffffffff) >> (index & 31) & 1)
    return torch.where(keep, logits, torch.full_like(logits, float("-inf")))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_apply_logit_bitmask_shared_mask_respects_lsb_order_and_padding(dtype):
    logits = torch.arange(2 * 37, device = DEVICE, dtype = dtype).reshape(2, 37)
    bitmask = _packed(1, 1, [(0, 0), (0, 3), (0, 31)])
    output = torch.full_like(logits, 123)

    assert ext.apply_logit_bitmask(logits, output, bitmask) is None

    torch.testing.assert_close(output, _reference(logits, bitmask))
    torch.testing.assert_close(logits, torch.arange(2 * 37, device = DEVICE, dtype = dtype).reshape(2, 37))
    assert torch.isneginf(output[:, 32:]).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_apply_logit_bitmask_uses_independent_mask_rows(dtype):
    logits = torch.arange(2 * 40, device = DEVICE, dtype = dtype).reshape(2, 40)
    bitmask = _packed(2, 2, [(0, 1), (0, 31), (1, 2), (1, 32), (1, 39)])
    output = torch.empty_like(logits)

    ext.apply_logit_bitmask(logits, output, bitmask)

    torch.testing.assert_close(output, _reference(logits, bitmask))
    assert torch.isneginf(output[0, 2])
    assert torch.isneginf(output[1, 1])


def test_apply_logit_bitmask_rejects_bad_dtype_and_shape():
    logits = torch.zeros((2, 8), device = DEVICE, dtype = torch.float16)
    output = torch.empty_like(logits)
    bitmask = torch.zeros((1, 1), device = DEVICE, dtype = torch.int32)

    with pytest.raises((TypeError, ValueError, RuntimeError), match = "dtype mismatch"):
        ext.apply_logit_bitmask(logits, output.float(), bitmask)
    with pytest.raises((TypeError, ValueError, RuntimeError), match = "bad bitmask batch dim"):
        ext.apply_logit_bitmask(logits, output, bitmask.expand(3, -1).contiguous())
    with pytest.raises((TypeError, ValueError, RuntimeError), match = "bitmask"):
        ext.apply_logit_bitmask(logits, output, bitmask.long())
