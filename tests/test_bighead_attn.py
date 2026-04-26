import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
test_correctness.py
-------------------
Compares bighead_attn against torch.nn.functional.scaled_dot_product_attention
across a range of shapes, head configurations, and causal settings.

Run after building the extension:
    python setup.py build_ext --inplace
    python test_correctness.py
"""

import torch
import torch.nn.functional as F
from exllamav3.ext import exllamav3_ext as ext
from torch.nn.attention.bias import causal_lower_right

device = "cuda:3"


# ── Reference ────────────────────────────────────────────────────────────────

def sdpa_reference(q, k, v, causal: bool) -> torch.Tensor:
    """
    Inputs are in our kernel's layout:  [bsz, seq_len, n_heads, dim]
    SDPA wants:                         [bsz, n_heads, seq_len, dim]

    For GQA (n_kv_heads < n_q_heads) we repeat K/V before passing to SDPA so
    the reference is always correct regardless of PyTorch version.

    For causal masking with q_len < kv_len, PyTorch's is_causal=True uses a
    bottom-right-aligned causal mask, which is exactly what the kernel does:
    query at q_pos can attend to KV positions 0 .. (kv_len - q_len + q_pos).
    """
    bsz, q_len, n_q_heads, dim = q.shape
    bsz, kv_len, n_kv_heads, dim = k.shape
    G = n_q_heads // n_kv_heads

    # [bsz, n_heads, seq, dim]
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2).repeat_interleave(G, dim = 1)  # expand KV heads
    v_t = v.transpose(1, 2).repeat_interleave(G, dim = 1)

    # fp32 for reference accuracy
    out_t = F.scaled_dot_product_attention(
        q_t.float(), k_t.float(), v_t.float(),
        # is_causal = causal,
        attn_mask = causal_lower_right(q_len, kv_len)
    )
    # back to [bsz, q_len, n_q_heads, dim], fp16
    return out_t.to(torch.float16).transpose(1, 2)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_case(
        bsz, q_len, kv_len,
        n_q_heads, n_kv_heads, dim,
        kv_chunk_size, causal,
        atol = 1e-2, rtol = 1e-2,
):
    tag = (f"bsz={bsz} q={q_len} kv={kv_len} "
           f"Hq={n_q_heads} Hkv={n_kv_heads} D={dim} "
           f"chunk={kv_chunk_size} causal={causal}")

    torch.manual_seed(42)
    q = torch.randn(bsz, q_len, n_q_heads, dim, dtype = torch.float16, device = device)
    k = torch.randn(bsz, kv_len, n_kv_heads, dim, dtype = torch.float16, device = device)
    v = torch.randn(bsz, kv_len, n_kv_heads, dim, dtype = torch.float16, device = device)
    o = torch.empty_like(q)

    rbytes = q.numel() * 2 + k.numel() * 2 + v.numel() * 2
    wbytes = o.numel() * 2
    tbytes = rbytes + wbytes
    tag += f"  read: {rbytes:,}  write: {wbytes:,}  total: {tbytes:,}"

    ext.bighead_attn(q, k, v, o, kv_chunk_size, causal, 0.0)
    ref = sdpa_reference(q, k, v, causal)

    if torch.allclose(o, ref, atol = atol, rtol = rtol):
        print(f"  UNPAGED PASS  {tag}")
    else:
        abs_err = (o.float() - ref.float()).abs()
        print(f"  UNPAGED FAIL  {tag}")
        print(f"          max_abs={abs_err.max():.4f}  "
              f"mean_abs={abs_err.mean():.4f}  "
              f"fraction_outside_tol="
              f"{((abs_err > atol + rtol * ref.float().abs()).float().mean()):.4f}")

    n_pages = ((kv_len + 255) // 256)
    k_cache = torch.zeros(bsz * n_pages, 256, n_kv_heads, dim, device = device, dtype = torch.half)
    v_cache = torch.zeros(bsz * n_pages, 256, n_kv_heads, dim, device = device, dtype = torch.half)
    block_table = torch.arange(bsz * n_pages, dtype = torch.int32, device = device).view(bsz, n_pages)
    cache_seqlens = torch.zeros((bsz,), dtype = torch.int32, device = device)

    ext.bighead_attn_paged(q, k, v, k_cache, v_cache, block_table, cache_seqlens, o, kv_chunk_size, causal, 0.0)

    if torch.allclose(o, ref, atol = atol, rtol = rtol):
        print(f" PAGED PASS    {tag}")
    else:
        abs_err = (o.float() - ref.float()).abs()
        print(f" PAGED FAIL    {tag}")
        print(f"               max_abs={abs_err.max():.4f}  "
              f"mean_abs={abs_err.mean():.4f}  "
              f"fraction_outside_tol="
              f"{((abs_err > atol + rtol * ref.float().abs()).float().mean()):.4f}")


# ── Test matrix ───────────────────────────────────────────────────────────────

def main():
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}\n")

    # (bsz, q_len, kv_len, n_q_heads, n_kv_heads, dim, kv_chunk_size, causal)
    cases = [
        # Basic decode (single token)
        (1, 1, 512, 8, 1, 64, 128, False),
        (1, 1, 512, 8, 1, 64, 128, True),
        (1, 1, 2048, 8, 2, 128, 256, False),
        (1, 1, 4096, 16, 2, 256, 512, True),
        (1, 1, 4096, 16, 2, 512, 512, True),  # chunk_size == kv_len / 8
        # Speculative decode (small q_len > 1)
        (1, 4, 512, 8, 1, 64, 128, True),
        (1, 4, 2048, 16, 2, 128, 256, True),
        (1, 8, 4096, 16, 2, 256, 512, True),
        # Batch > 1
        (2, 1, 1024, 8, 2, 64, 256, False),
        (3, 1, 1024, 8, 2, 64, 256, False),
        (4, 1, 1024, 8, 2, 64, 256, False),
        (4, 1, 2048, 16, 2, 128, 512, True),
        (7, 1, 2048, 16, 2, 128, 512, True),
        # GQA fan-out variety
        (1, 1, 1024, 4, 1, 128, 256, True),  # G=4
        (1, 1, 1024, 8, 1, 128, 256, True),  # G=8  (G_MAX)
        # d=512 (the motivating case)
        (1, 1, 32768, 16, 2, 512, 128, True),
        (1, 4, 32768, 16, 2, 512, 128, True),
        # About as large as we can test with SDPA
        (1, 1, 1024*128, 16, 2, 512, 64, True),
        (1, 1, 1024*128, 16, 2, 512, 128, True),
        (1, 1, 1024*128, 16, 2, 512, 256, True),
        (1, 1, 1024*128, 16, 2, 512, 512, True),
        (1, 8, 1024*128, 16, 2, 512, 128, True),
        # Chunk boundary edge cases
        (1, 1, 500, 4, 1, 64, 128, True),  # kv_len not multiple of chunk_size
        (1, 1, 129, 4, 1, 64, 64, True),  # exactly 3 chunks but last is 1 elem
    ]

    for case in cases:
        run_case(*case)


if __name__ == "__main__":
    main()