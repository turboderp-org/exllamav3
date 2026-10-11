"""
Sliding-window decode in the Triton paged flash-decoding kernel at long context: the kv splits cover only the
window, so for a window far shorter than the sequence the result must match the fp32 reference
testlib.attention.ref_attn over the window, for any split count (including more splits than window tiles), causal
and bidirectional (DFlash drafters), q_len 1..8.
"""
import pytest
import torch

from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode
from testlib.attention import gather_pages, rand_paged_cache, ref_attn
from testlib.compare import assert_rel_close


@pytest.mark.parametrize("past,window", [(40000, 2048), (5000, 2048), (1500, 2048), (9000, 256)])
@pytest.mark.parametrize("q_len", [1, 8])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("num_splits", [None, 1, 3, 64])
def test_decode_window_long(device, past, window, q_len, causal, num_splits):
    torch.manual_seed(past + window + q_len)
    B, KVH, H, D = 1, 8, 32, 128
    T = past + q_len
    kc, vc, bt, sl = rand_paged_cache(B, T + 8, KVH, D, past, device)
    k = torch.randn((B, q_len, KVH, D), dtype = torch.half, device = device)
    v = torch.randn_like(k)
    q = torch.randn((B, q_len, H, D), dtype = torch.half, device = device)
    out = paged_attn_triton_decode(q, k, v, kc, vc, bt, sl, causal = causal,
                                   window_size = (window, 0 if causal else -1), num_splits = num_splits)
    ref = ref_attn(q, gather_pages(kc, bt, T), gather_pages(vc, bt, T), causal = causal, window = window)
    assert_rel_close(out, ref, 8e-3)
