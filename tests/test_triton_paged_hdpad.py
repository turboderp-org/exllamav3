"""
Non-power-of-two head dims in the Triton paged attention kernels (zero-padded tiles): decode,
prefill over a cache (causal, sliding window), cache-less bidirectional/causal, GQA, and the
quantized-cache paths (head_dim a multiple of 32), against an fp32 torch reference. Power-of-two
dims are included as controls.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.modules.attention_fn.triton_paged import (
    paged_attn_triton_decode, paged_attn_triton_prefill, has_triton,
)
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.constants import PAGE_SIZE

device = "cuda:0"
pytestmark = pytest.mark.skipif(not has_triton, reason = "requires Triton")


def ref_attn(q, k, v, causal, past, window = None):
    """q (B, Q, H, D) attends to k/v (B, T, KVH, D); rows are the last Q positions of T."""
    B, Q, H, D = q.shape
    T, KVH = k.shape[1], k.shape[2]
    g = H // KVH
    kk = k.repeat_interleave(g, dim = 2).float(); vv = v.repeat_interleave(g, dim = 2).float()
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kk) * D ** -0.5
    qpos = (T - Q + torch.arange(Q, device = q.device)).view(Q, 1)
    kpos = torch.arange(T, device = q.device).view(1, T)
    mask = torch.ones((Q, T), dtype = torch.bool, device = q.device)
    if causal:
        mask &= kpos <= qpos
    if window is not None:
        mask &= kpos >= qpos - window
    s = s.masked_fill(~mask.view(1, 1, Q, T), -float("inf"))
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), vv)


def make_cache(B, T_alloc, KVH, D, past, q_len, seed):
    torch.manual_seed(seed)
    pages = -(-T_alloc // PAGE_SIZE)
    kc = torch.randn((B * pages, PAGE_SIZE, KVH, D), dtype = torch.half, device = device)
    vc = torch.randn_like(kc)
    perm = torch.randperm(B * pages, device = device, dtype = torch.int32)
    bt = perm.view(B, pages)
    sl = torch.full((B,), past, dtype = torch.int32, device = device)
    k = torch.randn((B, q_len, KVH, D), dtype = torch.half, device = device); v = torch.randn_like(k)
    q = torch.randn((B, q_len, KVH * 4 if KVH < 8 else KVH, D), dtype = torch.half, device = device)
    return kc, vc, bt, sl, k, v, q


def gather(kc, bt, T):
    """Logical (B, T, KVH, D) view of the first T positions of each sequence."""
    B, pages = bt.shape
    flat = kc[bt.long().view(-1)].view(B, pages * PAGE_SIZE, kc.shape[2], kc.shape[3])
    return flat[:, :T]


@pytest.mark.parametrize("hd", [72, 80, 96, 112, 160, 128])
@pytest.mark.parametrize("q_len,past,kvh", [(1, 700, 2), (8, 300, 4), (1, 40, 1)])
def test_decode_hdpad(hd, q_len, past, kvh):
    B = 2
    kc, vc, bt, sl, k, v, q = make_cache(B, past + q_len + 3, kvh, hd, past, q_len, hd * 7 + q_len)
    out = paged_attn_triton_decode(q, k, v, kc, vc, bt, sl, causal = True)
    T = past + q_len
    ref = ref_attn(q, gather(kc, bt, T), gather(vc, bt, T), True, past)
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 8e-3, f"rel err {err:.3e}"


@pytest.mark.parametrize("hd", [72, 80, 96, 112, 160, 128])
@pytest.mark.parametrize("q_len,past,window", [(300, 500, None), (513, 0, None), (200, 900, 64)])
def test_prefill_hdpad(hd, q_len, past, window):
    B, kvh = 2, 2
    kc, vc, bt, sl, k, v, q = make_cache(B, past + q_len + 3, kvh, hd, past, q_len, hd * 3 + q_len)
    out = paged_attn_triton_prefill(q, k, v, kc, vc, bt, sl, causal = True,
                                    window_size = (window, 0) if window else None)
    T = past + q_len
    ref = ref_attn(q, gather(kc, bt, T), gather(vc, bt, T), True, past, window)
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 8e-3, f"rel err {err:.3e}"


@pytest.mark.parametrize("hd", [72, 80, 96, 112, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_nocache_hdpad(hd, causal):
    torch.manual_seed(hd)
    B, S, H, KVH = 2, 1000, 8, 4
    q = torch.randn((B, S, H, hd), dtype = torch.half, device = device)
    k = torch.randn((B, S, KVH, hd), dtype = torch.half, device = device); v = torch.randn_like(k)
    out = paged_attn_triton_prefill(q, None, None, None, None, None, None, causal = causal, k_new = k, v_new = v)
    ref = ref_attn(q, k, v, causal, 0)
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 8e-3, f"rel err {err:.3e}"


def _quant_cache(kc, bits):
    pages, ps, kvh, hd = kc.shape
    rows = pages * ps
    pq = torch.empty((rows, kvh * hd // 32 * bits), dtype = torch.int32, device = device)
    sc = torch.empty((rows, kvh * hd // 32), dtype = torch.half, device = device)
    ext.quant_cache_cont(kc.reshape(rows, kvh * hd).contiguous(), pq, sc, 0.0)
    deq = torch.empty((rows, kvh * hd), dtype = torch.half, device = device)
    ext.dequant_cache_cont(pq, sc, deq, 0.0)
    return pq.view(pages, ps, -1), sc.view(pages, ps, -1), deq.view(pages, ps, kvh, hd)


@pytest.mark.parametrize("hd,bits", [(96, 8), (96, 4), (160, 6), (128, 4)])
@pytest.mark.parametrize("q_len,past", [(1, 700), (8, 300), (300, 500)])
def test_qc_hdpad(hd, bits, q_len, past):
    """Packed quantized cache with a non-power-of-two head dim (multiple of 32): the kernels
    read the SAME values the CUDA dequantizer produces."""
    B, kvh = 2, 2
    kc, vc, bt, sl, k, v, q = make_cache(B, past + q_len + 3, kvh, hd, past, q_len, hd + bits + q_len)
    T = past + q_len
    # write the new rows into the fp16 cache first so the packed cache holds everything
    from exllamav3.modules.attention_fn.triton_paged import _paged_kv_update_kernel
    import triton
    with torch.cuda.device(q.device):
        _paged_kv_update_kernel[(B * q_len, kvh, 1)](
            k, v, kc, vc, bt, sl, bt.shape[1], q_len, kvh, PAGE_SIZE, hd, triton.next_power_of_2(hd),
            num_warps = 2, num_stages = 3)
    qk, sk, kdeq = _quant_cache(kc, bits); qv, sv, vdeq = _quant_cache(vc, bits)
    fn = paged_attn_triton_decode if q_len <= 16 else paged_attn_triton_prefill
    out = fn(q, None, None, qk, qv, bt, sl, causal = True, qc = (sk, sv, bits, bits),
             pre_appended_len = q_len, n_kv_heads_override = kvh)
    ref = ref_attn(q, gather(kdeq, bt, T), gather(vdeq, bt, T), True, past)
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1.2e-2, f"rel err {err:.3e}"
