"""
Odd-length edge cases of the fp16 elementwise/sampling kernels: softcap and Gumbel noise process half2 pairs and
must handle an odd element count; argmax_sample's paired read must respect an odd max_logit exactly. Softcap is
checked against torch tanh in fp32.

Empty inputs of argmax_sample / gumbel_sample: no rows is a no-op; an empty vocabulary (no columns, or a max_logit
bound below 1) has no maximum and raises, leaving ids untouched. Neither leaves a CUDA error pending.
"""

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext


@torch.inference_mode()
def test_softcap_fp16_odd_numel(device):
    for n in (1, 3, 2047, 2049, 4097):
        x = (torch.randn(1, n, device = device) * 40).half(); y = torch.empty_like(x)
        ext.softcap(x, y, 30.0)
        ref = (torch.tanh(x.float() / 30.0) * 30.0).half()
        torch.testing.assert_close(y, ref, rtol = 2e-3, atol = 2e-2)


@torch.inference_mode()
def test_gumbel_fp16_odd_size_touches_last_element(device):
    for n in (3, 2047, 2049, 4097):
        x = torch.zeros(1, n, device = device, dtype = torch.half); y = torch.empty_like(x)
        ext.gumbel_noise_f16(x, y, 12345)
        assert (y != 0).all(), f"n={n}: {(y == 0).sum().item()} elements received no noise (last: {y[0, -1].item()})"
        assert torch.isfinite(y.float()).all()


@torch.inference_mode()
def test_argmax_odd_max_logit(device):
    n = 64
    for max_logit in (5, 33, 63):
        x = torch.full((1, n), -10.0, device = device, dtype = torch.half)
        x[0, max_logit - 1] = 5.0      # last valid logit is the max
        x[0, max_logit] = 50.0         # first excluded logit is larger still and must be ignored
        ids = torch.empty((1, 1), dtype = torch.long, device = device)
        ext.argmax_sample(x, ids, max_logit)
        assert ids.item() == max_logit - 1, f"max_logit={max_logit}: argmax picked {ids.item()}"


def _device_still_works(device):
    # A launch error left pending by the call would surface in the next unrelated launch
    y = torch.ones(4, device = device) * 2
    torch.cuda.synchronize(device)
    assert y.sum().item() == 8


@pytest.mark.parametrize("fn", ["argmax_sample", "gumbel_sample"])
@pytest.mark.parametrize("bsz, vocab, max_logit, raises", [
    (0, 64, 0, False),          # no rows: nothing to sample
    (0, 64, 7, False),
    (2, 0, 0, True),            # no columns: argmax of an empty row
    (0, 0, 0, True),
    (2, 64, -1, True),          # bound excludes every column
])
@torch.inference_mode()
def test_empty_sample(device, fn, bsz, vocab, max_logit, raises):
    logits = torch.randn(bsz, vocab, device = device).half()
    ids = torch.full((bsz, 1), -7, dtype = torch.long, device = device)
    args = (logits, ids, max_logit) + ((123,) if fn == "gumbel_sample" else ())
    if raises:
        with pytest.raises(RuntimeError, match = f"{fn}: .*empty vocabulary"):
            getattr(ext, fn)(*args)
        assert (ids == -7).all()
    else:
        getattr(ext, fn)(*args)
    _device_still_works(device)
