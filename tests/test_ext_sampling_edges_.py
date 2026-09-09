"""
Odd-length edge cases of the fp16 elementwise/sampling kernels: softcap and Gumbel noise process half2 pairs and
must handle an odd element count; argmax_sample's paired read must respect an odd max_logit exactly.
"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.ext import exllamav3_ext as ext

device = "cuda:0"


def test_softcap_fp16_odd_numel():
    for n in (1, 3, 2047, 2049, 4097):
        x = (torch.randn(1, n, device = device) * 40).half(); y = torch.empty_like(x)
        ext.softcap(x, y, 30.0)
        ref = (torch.tanh(x.float() / 30.0) * 30.0).half()
        torch.testing.assert_close(y, ref, rtol = 2e-3, atol = 2e-2)


def test_gumbel_fp16_odd_size_touches_last_element():
    for n in (3, 2047, 2049, 4097):
        x = torch.zeros(1, n, device = device, dtype = torch.half); y = torch.empty_like(x)
        ext.gumbel_noise_f16(x, y, 12345)
        assert (y != 0).all(), f"n={n}: {(y == 0).sum().item()} elements received no noise (last: {y[0, -1].item()})"
        assert torch.isfinite(y.float()).all()


def test_argmax_odd_max_logit():
    n = 64
    for max_logit in (5, 33, 63):
        x = torch.full((1, n), -10.0, device = device, dtype = torch.half)
        x[0, max_logit - 1] = 5.0      # last valid logit is the max
        x[0, max_logit] = 50.0         # first excluded logit is larger still and must be ignored
        ids = torch.empty((1, 1), dtype = torch.long, device = device)
        ext.argmax_sample(x, ids, max_logit)
        assert ids.item() == max_logit - 1, f"max_logit={max_logit}: argmax picked {ids.item()}"
