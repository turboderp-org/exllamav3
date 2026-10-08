"""
Non-power-of-two head dims in the Triton paged attention kernels (zero-padded tiles): decode, prefill over a cache
(causal, sliding window), cache-less bidirectional/causal, GQA, and the quantized-cache paths (head_dim a multiple
of 32), against the fp32 reference testlib.attention.ref_attn. Power-of-two dims are included as controls.
"""
import pytest
import torch
import triton

from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.attention_fn.triton_paged import (
    _paged_kv_update_kernel, paged_attn_triton_decode, paged_attn_triton_prefill,
)
from testlib.attention import gather_pages, quant_roundtrip, rand_paged_cache, ref_attn
from testlib.compare import assert_rel_close


def make_case(B, kvh, hd, past, q_len, seed, device):
    """Paged cache holding `past` positions plus new k / v / q rows; 4x GQA below 8 kv heads"""
    torch.manual_seed(seed)
    kc, vc, bt, sl = rand_paged_cache(B, past + q_len + 3, kvh, hd, past, device)
    k = torch.randn((B, q_len, kvh, hd), dtype = torch.half, device = device)
    v = torch.randn_like(k)
    q = torch.randn((B, q_len, kvh * 4 if kvh < 8 else kvh, hd), dtype = torch.half, device = device)
    return kc, vc, bt, sl, k, v, q


@pytest.mark.parametrize("hd", [72, 80, 96, 112, 160, 128])
@pytest.mark.parametrize("q_len,past,kvh", [(1, 700, 2), (8, 300, 4), (1, 40, 1)])
def test_decode_hdpad(device, hd, q_len, past, kvh):
    B = 2
    kc, vc, bt, sl, k, v, q = make_case(B, kvh, hd, past, q_len, hd * 7 + q_len, device)
    out = paged_attn_triton_decode(q, k, v, kc, vc, bt, sl, causal = True)
    T = past + q_len
    ref = ref_attn(q, gather_pages(kc, bt, T), gather_pages(vc, bt, T), causal = True)
    assert_rel_close(out, ref, 8e-3)


@pytest.mark.parametrize("hd", [72, 80, 96, 112, 160, 128])
@pytest.mark.parametrize("q_len,past,window", [(300, 500, None), (513, 0, None), (200, 900, 64)])
def test_prefill_hdpad(device, hd, q_len, past, window):
    B, kvh = 2, 2
    kc, vc, bt, sl, k, v, q = make_case(B, kvh, hd, past, q_len, hd * 3 + q_len, device)
    out = paged_attn_triton_prefill(q, k, v, kc, vc, bt, sl, causal = True,
                                    window_size = (window, 0) if window else None)
    T = past + q_len
    ref = ref_attn(q, gather_pages(kc, bt, T), gather_pages(vc, bt, T), causal = True, window = window)
    assert_rel_close(out, ref, 8e-3)


@pytest.mark.parametrize("hd", [72, 80, 96, 112, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_nocache_hdpad(device, hd, causal):
    torch.manual_seed(hd)
    B, S, H, KVH = 2, 1000, 8, 4
    q = torch.randn((B, S, H, hd), dtype = torch.half, device = device)
    k = torch.randn((B, S, KVH, hd), dtype = torch.half, device = device)
    v = torch.randn_like(k)
    out = paged_attn_triton_prefill(q, None, None, None, None, None, None, causal = causal, k_new = k, v_new = v)
    ref = ref_attn(q, k, v, causal = causal)
    assert_rel_close(out, ref, 8e-3)


@pytest.mark.parametrize("hd,bits", [(96, 8), (96, 4), (160, 6), (128, 4)])
@pytest.mark.parametrize("q_len,past", [(1, 700), (8, 300), (300, 500)])
def test_qc_hdpad(device, hd, bits, q_len, past):
    """Packed quantized cache with a non-power-of-two head dim (multiple of 32): the kernels read the SAME values
    the CUDA dequantizer produces"""
    B, kvh = 2, 2
    kc, vc, bt, sl, k, v, q = make_case(B, kvh, hd, past, q_len, hd + bits + q_len, device)
    T = past + q_len
    # Write the new rows into the fp16 cache first so the packed cache holds everything
    with torch.cuda.device(q.device):
        _paged_kv_update_kernel[(B * q_len, kvh, 1)](
            k, v, kc, vc, bt, sl, bt.shape[1], q_len, kvh, PAGE_SIZE, hd, triton.next_power_of_2(hd),
            num_warps = 2, num_stages = 3)
    pages = kc.shape[0]
    qk, sk, kdeq = quant_roundtrip(kc.view(pages, PAGE_SIZE, kvh * hd), bits)
    qv, sv, vdeq = quant_roundtrip(vc.view(pages, PAGE_SIZE, kvh * hd), bits)
    fn = paged_attn_triton_decode if q_len <= 16 else paged_attn_triton_prefill
    out = fn(q, None, None, qk, qv, bt, sl, causal = True, qc = (sk, sv, bits, bits),
             pre_appended_len = q_len, n_kv_heads_override = kvh)
    kdeq = kdeq.view(pages, PAGE_SIZE, kvh, hd)
    vdeq = vdeq.view(pages, PAGE_SIZE, kvh, hd)
    ref = ref_attn(q, gather_pages(kdeq, bt, T), gather_pages(vdeq, bt, T), causal = True)
    assert_rel_close(out, ref, 1.2e-2)
