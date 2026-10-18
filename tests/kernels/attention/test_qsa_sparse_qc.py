"""
qsa_sparse_attend_rows over packed-quantized K/V pages (the QSA sparse gather with CacheLayer_qsa_quant) against
the fp16 kernel fed the SAME values dequantized by the CUDA kernel (testlib.attention.quant_roundtrip): tests the
online-dequant / rotation path, not the quantizer.
"""
import pytest
import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.attention_fn.qsa_triton import qsa_sparse_attend_rows
from testlib.attention import quant_roundtrip
from testlib.compare import assert_rel_close


@pytest.mark.parametrize("k_bits,v_bits", [(8, 8), (4, 4), (6, 3), (2, 8)])
@pytest.mark.parametrize("R,H,kvh,hd,topk", [(1, 24, 2, 256, 2051), (5, 8, 2, 128, 300), (3, 16, 4, 64, 96)])
def test_qsa_sparse_qc(device, R, H, kvh, hd, topk, k_bits, v_bits):
    torch.manual_seed(R * 7 + k_bits)
    pages = 12
    rows = pages * PAGE_SIZE
    k = torch.randn((rows, kvh, hd), dtype = torch.half, device = device)
    v = torch.randn((rows, kvh, hd), dtype = torch.half, device = device)
    q = torch.randn((R, H, hd), dtype = torch.half, device = device)
    K_pad = -(-topk // 32) * 32
    indices = torch.full((R, K_pad), -1, dtype = torch.int32, device = device)
    n_pos = pages * PAGE_SIZE
    for r in range(R):
        n = min(topk, n_pos - 1)
        indices[r, :n] = torch.randperm(n_pos - 1, device = device)[:n].int()
    bt = torch.stack([torch.randperm(pages, device = device, dtype = torch.int32) for _ in range(R)])
    scale = hd ** -0.5

    qk, sk, k_deq = quant_roundtrip(k.view(rows, kvh * hd), k_bits)
    qv, sv, v_deq = quant_roundtrip(v.view(rows, kvh * hd), v_bits)
    ref = qsa_sparse_attend_rows(
        q, k_deq.view(rows, kvh, hd), v_deq.view(rows, kvh, hd), indices, scale,
        block_table = bt, page_size = PAGE_SIZE,
    ).clone()
    got = qsa_sparse_attend_rows(
        q, qk.view(pages, PAGE_SIZE, -1), qv.view(pages, PAGE_SIZE, -1), indices, scale,
        block_table = bt, page_size = PAGE_SIZE,
        qc = (sk, sv, k_bits, v_bits), n_kv_heads = kvh,
    )
    assert_rel_close(got, ref, 8e-3)
