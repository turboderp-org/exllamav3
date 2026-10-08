"""
Attention references and synthetic caches for the attention kernel tests.

Layouts (shared with the kernels):
    q            (bsz, q_len, n_heads, head_dim) fp16
    k, v         (bsz, seq_len, n_kv_heads, head_dim) fp16
    paged cache  (num_pages, PAGE_SIZE, n_kv_heads, head_dim) fp16, addressed through a (bsz, pages) int32 block
                 table; sequence b's position t lives at page block_table[b, t // PAGE_SIZE], row t % PAGE_SIZE
    packed cache (num_pages, page_size, token_dim // 32 * bits) int32 words + (num_pages, page_size, token_dim // 32)
                 fp16 group scales, token_dim = n_kv_heads * head_dim (the quantized cache format of
                 quant_cache_cont / dequant_cache_cont)
"""

import torch

from exllamav3.constants import PAGE_SIZE


def ref_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
             window: int | None = None) -> torch.Tensor:
    """fp32 reference: q (B, Q, H, D) attends to k / v (B, T, KVH, D) with GQA (H a multiple of KVH). The query
    rows are the last Q positions of T (bottom-right aligned causal mask); a window admits keys at positions
    >= q_pos - window. Returns (B, Q, H, D) fp32"""
    B, Q, H, D = q.shape
    T, KVH = k.shape[1], k.shape[2]
    g = H // KVH
    kk = k.repeat_interleave(g, dim = 2).float()
    vv = v.repeat_interleave(g, dim = 2).float()
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


def rand_paged_cache(bsz: int, max_len: int, n_kv: int, head_dim: int, past: int, device) -> tuple:
    """Random fp16 paged K/V cache with room for max_len positions per sequence, pages assigned through a random
    permutation. Returns (k_cache, v_cache, block_table, cache_seqlens), every sequence holding `past` positions"""
    pages = -(-max_len // PAGE_SIZE)
    kc = torch.randn((bsz * pages, PAGE_SIZE, n_kv, head_dim), dtype = torch.half, device = device)
    vc = torch.randn_like(kc)
    bt = torch.randperm(bsz * pages, device = device, dtype = torch.int32).view(bsz, pages)
    sl = torch.full((bsz,), past, dtype = torch.int32, device = device)
    return kc, vc, bt, sl


def rand_packed_cache(num_pages: int, page_size: int, n_kv: int, head_dim: int, bits: int, device) -> tuple:
    """Random packed-quantized K/V pages (any bit pattern is a valid payload) and group scales in 0.1..1.1.
    Returns (k_words, v_words, qc) with qc = (k_scales, v_scales, bits, bits) as the paged kernels take it"""
    token_dim = n_kv * head_dim
    shape = (num_pages, page_size, token_dim // 32 * bits)      # bits int32 words per 32-element group
    kc = torch.randint(-2**31, 2**31, shape, dtype = torch.int64, device = device).to(torch.int32)
    vc = torch.randint(-2**31, 2**31, shape, dtype = torch.int64, device = device).to(torch.int32)
    ks = torch.rand((num_pages, page_size, token_dim // 32), dtype = torch.float16, device = device) + 0.1
    vs = torch.rand((num_pages, page_size, token_dim // 32), dtype = torch.float16, device = device) + 0.1
    return kc, vc, (ks, vs, bits, bits)


def gather_pages(cache: torch.Tensor, block_table: torch.Tensor, length: int) -> torch.Tensor:
    """Logical (B, length, ...) view of the first `length` positions of each sequence of a paged cache"""
    B, pages = block_table.shape
    flat = cache[block_table.long().view(-1)].view(B, pages * cache.shape[1], *cache.shape[2:])
    return flat[:, :length]


def quant_roundtrip(t: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize t (..., width) fp16 to the packed cache format (width a multiple of 32) and dequantize it with the
    CUDA kernels. Returns (words (..., width // 32 * bits) int32, scales (..., width // 32) fp16, dequantized like
    t): kernels reading the packed cache must see exactly the dequantized values"""
    from exllamav3.ext import exllamav3_ext as ext
    lead, width = t.shape[:-1], t.shape[-1]
    flat = t.reshape(-1, width).contiguous()
    rows = flat.shape[0]
    pq = torch.empty((rows, width // 32 * bits), dtype = torch.int32, device = t.device)
    sc = torch.empty((rows, width // 32), dtype = torch.half, device = t.device)
    ext.quant_cache_cont(flat, pq, sc, 0.0)
    deq = torch.empty_like(flat)
    ext.dequant_cache_cont(pq, sc, deq, 0.0)
    return pq.view(*lead, -1), sc.view(*lead, -1), deq.view(t.shape)
