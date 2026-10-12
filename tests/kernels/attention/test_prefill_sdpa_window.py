"""
The sm_75 prefill path of paged_attn_triton_prefill (util/backend.py: EXL3_SDPA_PREFILL / EXL3_FA75, on by
default only on Turing): the chunk attends over a dense fp16 window, the staged dequantization of a quantized
cache or the fp16 cache's own pages, through SDPA or fa75, instead of the Triton kernel. Forced on here and
compared with the Triton kernel on the same cache: quantized and fp16 caches, GQA and MHA, head_dim 128 (SDPA)
and 256 (fa75 and SDPA), several sequences of uneven length, chunk lengths below and above the usual staging
threshold. The reference is the default path on the same device, so this runs on every CUDA device.
"""

import pytest
import torch

from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_prefill
from exllamav3.util import backend
from testlib.attention import rand_packed_cache

pytestmark = pytest.mark.cuda_only

PAGE = 256


def _run(q, kc, vc, qc, bt, sl, q_len, n_kv, k = None, v = None):
    return paged_attn_triton_prefill(
        q = q, k = k, v = v, k_cache = kc, v_cache = vc, block_table = bt, cache_seqlens = sl, causal = True,
        qc = qc, max_kv_len = bt.shape[1] * PAGE, pre_appended_len = 0 if k is not None else q_len,
        n_kv_heads_override = n_kv if qc is not None else None,
    )


@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("bits", [0, 4], ids = ["fp16_cache", "q4_cache"])
@pytest.mark.parametrize("n_q, n_kv", [(8, 2), (4, 4)], ids = ["gqa", "mha"])
@pytest.mark.parametrize("q_len", [16, 300])
@pytest.mark.parametrize("fa75", [0, 1])
def test_window_path_matches_triton(head_dim, bits, n_q, n_kv, q_len, fa75, device, monkeypatch):
    if fa75 and head_dim != 256:
        pytest.skip("fa75 only applies to head_dim 256")
    torch.manual_seed(head_dim + bits + n_q + q_len)
    bsz, ctx = 3, 2048
    num_pages = ctx // PAGE
    bt = torch.arange(bsz * num_pages, dtype = torch.int32, device = device).view(bsz, num_pages)
    # Past lengths that leave the appended chunk straddling page boundaries, one sequence nearly empty
    sl = torch.tensor([1000, 37, ctx - q_len - 5], dtype = torch.int32, device = device)
    assert int(sl.max()) + q_len <= ctx
    q = torch.randn((bsz, q_len, n_q, head_dim), dtype = torch.float16, device = device)

    if bits:
        kc, vc, qc = rand_packed_cache(bsz * num_pages, PAGE, n_kv, head_dim, bits, device)
        kv = (None, None)
    else:
        kc = torch.randn((bsz * num_pages, PAGE, n_kv, head_dim), dtype = torch.float16, device = device)
        vc = torch.randn_like(kc); qc = None
        kv = (torch.randn((bsz, q_len, n_kv, head_dim), dtype = torch.float16, device = device),
              torch.randn((bsz, q_len, n_kv, head_dim), dtype = torch.float16, device = device))

    # Reference: the default path (Triton kernel; quantized caches stage at 256+ rows, direct below)
    monkeypatch.setenv("EXL3_SDPA_PREFILL", "0")
    assert not backend.sdpa_prefill(device)
    ref = _run(q, kc.clone(), vc.clone(), qc, bt, sl, q_len, n_kv, *kv)

    monkeypatch.setenv("EXL3_SDPA_PREFILL", "1")
    monkeypatch.setenv("EXL3_FA75", str(fa75))
    assert backend.sdpa_prefill(device) and backend.qc_prefill_two_pass_min_q(device) == 9
    out = _run(q, kc.clone(), vc.clone(), qc, bt, sl, q_len, n_kv, *kv)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, ref, rtol = 0, atol = 2e-3)
