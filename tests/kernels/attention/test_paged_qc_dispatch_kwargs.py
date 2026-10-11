"""
The quantized-cache dispatch wrappers (fn_triton_paged_attn_decode_qc / fn_triton_paged_attn_prefill_qc) forward
AttnArgs fields as keyword arguments to the paged Triton kernels. A field the wrapper passes but the kernel does
not accept is a TypeError on the first decode step of every quantized-cache model, which no GPU-free import check
catches. Bind each wrapper's call against the real kernel signature with the kernels stubbed out, so the test needs
neither a GPU nor a Triton compile. Regression: v1.6.1 passed staging_pages (a prefill-only staging scratch size)
to the decode kernel too.
"""
import inspect

import pytest
import torch

from exllamav3.modules.attention_fn import triton_paged
from exllamav3.modules.attention_fn.common import AttnArgs


def _qc_args(q_len: int) -> AttnArgs:
    bsz, n_q, n_kv, dim, pages, page_size = 1, 8, 2, 128, 4, 256
    q = torch.zeros((bsz, q_len, n_q, dim), dtype = torch.float16)
    qk = torch.zeros((pages, page_size, n_kv, dim // 8), dtype = torch.int32)
    sk = torch.zeros((pages, page_size, n_kv, 1), dtype = torch.float16)
    return AttnArgs(
        bsz = bsz, q_len = q_len, num_q_heads = n_q, dim = dim, kv_len = 1024, num_kv_heads = n_kv,
        q = q, k = None, v = None, k_cache = None, v_cache = None,
        causal = True, sm_scale = dim ** -0.5, cu_seqlens = None, max_seqlen = None,
        window_size = None, softcap = 0.0,
        block_table = torch.arange(pages, dtype = torch.int32).unsqueeze(0),
        cache_seqlens = torch.tensor([512], dtype = torch.int32),
        q_cache = (qk, sk, qk, sk, 4, 4),
        max_kv_len = 1024,
        staging_pages = pages,
    )


@pytest.mark.nogpu
@pytest.mark.parametrize("wrapper,kernel,q_len", [
    ("fn_triton_paged_attn_decode_qc", "paged_attn_triton_decode", 1),
    ("fn_triton_paged_attn_decode_qc", "paged_attn_triton_decode", 16),
    ("fn_triton_paged_attn_prefill_qc", "paged_attn_triton_prefill", 17),
    ("fn_triton_paged_attn_prefill_qc", "paged_attn_triton_prefill", 4096),
])
def test_qc_wrapper_kwargs_bind_to_kernel(monkeypatch, wrapper, kernel, q_len):
    real_sig = inspect.signature(getattr(triton_paged, kernel))
    calls = []

    def stub(*a, **kw):
        # Raises TypeError exactly where the real kernel would, without running it
        real_sig.bind(*a, **kw)
        calls.append(kw)
        return torch.zeros(1)

    monkeypatch.setattr(triton_paged, kernel, stub)
    out = getattr(triton_paged, wrapper)(_qc_args(q_len))
    assert out is not None and len(calls) == 1
    # The staging scratch size is a prefill-only concern; the decode kernel dequantizes online
    assert ("staging_pages" in calls[0]) == (kernel == "paged_attn_triton_prefill")
