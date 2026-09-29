"""
Sliding-window decode in the Triton paged flash-decoding kernel at long context: the kv splits cover only the
window, so for a window far shorter than the sequence the result must match an fp32 reference over the window, for
any split count (including more splits than window tiles), causal and bidirectional (DFlash drafters), q_len 1..8.

    python -m pytest tests/test_triton_decode_window.py -v
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.modules.attention_fn.triton_paged import paged_attn_triton_decode
from exllamav3.constants import PAGE_SIZE

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

device = "cuda:0"


def ref_attn(q, k, v, causal, window):
    B, Q, H, D = q.shape
    T, KVH = k.shape[1], k.shape[2]
    g = H // KVH
    kk = k.repeat_interleave(g, dim = 2).float(); vv = v.repeat_interleave(g, dim = 2).float()
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kk) * D ** -0.5
    qpos = (T - Q + torch.arange(Q, device = q.device)).view(Q, 1)
    kpos = torch.arange(T, device = q.device).view(1, T)
    mask = kpos >= qpos - window
    if causal:
        mask &= kpos <= qpos
    s = s.masked_fill(~mask.view(1, 1, Q, T), -float("inf"))
    return torch.einsum("bhqk,bkhd->bqhd", torch.softmax(s, -1), vv)


@pytest.mark.parametrize("past,window", [(40000, 2048), (5000, 2048), (1500, 2048), (9000, 256)])
@pytest.mark.parametrize("q_len", [1, 8])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("num_splits", [None, 1, 3, 64])
def test_decode_window_long(past, window, q_len, causal, num_splits):
    torch.manual_seed(past + window + q_len)
    B, KVH, H, D = 1, 8, 32, 128
    T = past + q_len
    pages = -(-(T + 8) // PAGE_SIZE)
    kc = torch.randn((B * pages, PAGE_SIZE, KVH, D), dtype = torch.half, device = device)
    vc = torch.randn_like(kc)
    bt = torch.randperm(B * pages, device = device, dtype = torch.int32).view(B, pages)
    sl = torch.full((B,), past, dtype = torch.int32, device = device)
    k = torch.randn((B, q_len, KVH, D), dtype = torch.half, device = device); v = torch.randn_like(k)
    q = torch.randn((B, q_len, H, D), dtype = torch.half, device = device)
    out = paged_attn_triton_decode(q, k, v, kc, vc, bt, sl, causal = causal, window_size = (window, 0 if causal else -1),
                                   num_splits = num_splits)
    flat_k = kc[bt.long().view(-1)].view(B, pages * PAGE_SIZE, KVH, D)[:, :T]
    flat_v = vc[bt.long().view(-1)].view(B, pages * PAGE_SIZE, KVH, D)[:, :T]
    ref = ref_attn(q, flat_k, flat_v, causal, window)
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 8e-3, f"rel err {err:.3e}"
