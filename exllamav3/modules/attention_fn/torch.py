import torch
from .common import AttnArgs, AttnFn, get_non_causal_span_arglist
import torch.nn.functional as F


def _attn_cpu_native_single(
    q, k, v,
    sm_scale, causal, window_left, window_right, softcap, sinks, cu_seqlens, q_start,
):
    from ...ext import exllamav3_ext as ext
    fn = getattr(ext, "attn_cpu", None)
    if fn is None:
        raise RuntimeError("attn_cpu native kernel unavailable")
    # Speed: fp16/bf16 QK dots would pay per-dot cvtph/shift conversion (K rows are
    # reloaded L times). One upfront upcast to fp32 makes the hot loops pure f32
    # (like the torch fallback's fp16->fp32 path); cost is O(N*D) vs O(N^2*D).
    if q.dtype != torch.float32:
        qc = q.to(torch.float32, copy = True)
        kc = k.to(torch.float32, copy = True)
        vc = v.to(torch.float32, copy = True)
        qc = qc.contiguous() if not qc.is_contiguous() else qc
        kc = kc.contiguous() if not kc.is_contiguous() else kc
        vc = vc.contiguous() if not vc.is_contiguous() else vc
    else:
        qc = q.contiguous() if q.is_contiguous() else q.contiguous()
        kc = k.contiguous() if k.is_contiguous() else k.contiguous()
        vc = v.contiguous() if v.is_contiguous() else v.contiguous()
    o = torch.empty(q.shape, dtype = q.dtype, device = q.device)
    sinks_c = None
    if sinks is not None:
        sinks_c = sinks.float().contiguous() if sinks.dtype != torch.float32 or not sinks.is_contiguous() else sinks
    cu_c = None
    if cu_seqlens is not None:
        cu_c = cu_seqlens.contiguous() if cu_seqlens.is_contiguous() else cu_seqlens.contiguous()
        if cu_c.device.type != "cpu":
            cu_c = cu_c.cpu()
    fn(qc, kc, vc, o, float(sm_scale), bool(causal),
       int(window_left), int(window_right), float(softcap or 0.0),
       sinks_c, cu_c, int(q_start))
    return o


def _attn_cpu_nocache(args: AttnArgs) -> torch.Tensor | None:
    # Full CPU ceiling: dense + varlen (bsz1 or shared bsz>1) + SWA + softcap + sinks +
    # non-causal spans (prefix attention per span). Cached attention remains unsupported.
    if args.has_kv_cache():
        return None
    if args.is_varlen() and args.non_causal_spans:
        return None
    try:
        if args.non_causal_spans:
            outs = []
            for span in args.non_causal_spans:
                a, b, nc = span[:3]
                pre = span[3] if len(span) > 3 else 0
                l = b - a
                if args.window_size is not None and args.window_size > 0 and nc:
                    wl_span = max(args.window_size, l + pre)
                    wr_span = l - 1
                else:
                    wl_span, wr_span = args.get_window_size()
                q_span = args.q[:, a:b]
                k_pre = args.k[:, :b]
                v_pre = args.v[:, :b]
                outs.append(_attn_cpu_native_single(
                    q_span, k_pre, v_pre,
                    args.sm_scale, not nc, wl_span, wr_span,
                    args.softcap, args.sinks, None, a,
                ))
            return torch.cat(outs, dim = 1)
        wl, wr = args.get_window_size()
        return _attn_cpu_native_single(
            args.q, args.k, args.v,
            args.sm_scale, args.causal, wl, wr,
            args.softcap, args.sinks, args.cu_seqlens, 0,
        )
    except Exception:
        return None


def _causal_lower_right(*args):
    # torch.nn.attention.bias pulls in torch._dynamo (~0.5 s at import) and is only needed on this
    # fallback path, so import on first use
    from torch.nn.attention.bias import causal_lower_right
    return causal_lower_right(*args)

has_warned_sdpa_fallback = False
def _warn_sdpa_fallback():
    global has_warned_sdpa_fallback
    if has_warned_sdpa_fallback:
        return
    col_default = "\u001b[0m"
    col_red = "\u001b[31;1m"
    print(
        f"{col_red} !! Warning, using SDPA fallback for large head size. VRAM usage will be high "
        f"and inference on long sequences will be slow. Consider installing `triton` / `triton-windows`"
        f"or `xformers` to improve performance.{col_default}"
    )
    has_warned_sdpa_fallback = True


def fn_torch_sdpa_fallback_nocache(args: AttnArgs) -> torch.Tensor | None:
    if args.has_kv_cache():
        return None

    # CPU routing for speed. NOTE: the fused native kernel (attn_cpu) is SLOWER
    # than torch CPU SDPA on simple shapes, so simple stays on torch:
    # Ryzen 9700X / 8 threads, fp16, median ms —
    #   dense-1k: torch 6.37 vs AVX 19.79 vs scalar 53.59;
    #   dense-4k: torch 93.64 vs AVX 680.22 vs scalar 1393.09;
    #   varlen 32x64: torch-batched 0.78 vs AVX 1.98 vs scalar 5.28.
    # Torch lowers to oneDNN/MKL blocked GEMM (~500-700 GFLOPS); per-dot streaming
    # in attn_cpu reaches only ~100-160 GFLOPS (AVX still 2-3x scalar).
    # Ceiling shapes (SWA/windowed, bsz>1-varlen, softcap, sinks, spans) have no
    # torch fast path (manual masks materialize N^2 or OOM); native is primary
    # there (restricted KV ranges, no N^2 temp). Example: SWA L=4096,w=128 native
    # 8.7ms vs dense-full torch 25.2ms. CUDA for reference (RTX 3090, torch SDPA
    # fp16): 1k 0.10ms / 4k 1.28ms, ~150-500x the AVX times.
    if args.q.is_cpu:
        needs_native = (
            args.is_swa() or
            args.is_varlen() and args.q.shape[0] != 1 or
            args.softcap != 0.0 or
            args.sinks is not None or
            args.non_causal_spans
        )
        if needs_native:
            return _attn_cpu_nocache(args)
    elif (
        args.is_swa() or
        args.is_varlen() and args.q.shape[0] != 1 or
        args.softcap != 0.0 or
        args.sinks is not None or
        args.non_causal_spans
    ):
        return None

    if args.dim > 256:
        _warn_sdpa_fallback()

    q_t = args.q.transpose(1, 2)
    k_t = args.k.transpose(1, 2)
    v_t = args.v.transpose(1, 2)

    # CPU execution: fp16 has no vectorized flash-attention kernel (~2x slower);
    # fp32 and bf16 benchmark identically here, so upcast to fp32 and restore dtype
    out_dtype = q_t.dtype
    if q_t.device.type == "cpu" and q_t.dtype == torch.float16:
        q_t = q_t.to(torch.float32)
        k_t = k_t.to(torch.float32)
        v_t = v_t.to(torch.float32)

    if args.is_varlen():
        # Packed variable-length input: one segment per cu_seqlens interval. Each
        # segment starts at position 0, so is_causal is applied per segment.
        # max_seqlen is unused here; lengths are derived from the boundaries.
        bounds = args.cu_seqlens.tolist()
        lens = [e - s for s, e in zip(bounds[:-1], bounds[1:])]

        # Equal-length segments (e.g. one video clip: a frame grid per segment) reshape
        # into a single dense batched SDPA call instead of a Python loop of narrow calls.
        # is_causal applies within each L x L block either way; unequal-length packs
        # (multi-image) keep the loop below.
        if len(lens) > 1 and sum(lens) == q_t.shape[2] and all(l == lens[0] for l in lens):
            L = lens[0]
            H = q_t.shape[1]
            Hkv = k_t.shape[1]
            S = len(lens)
            q_b = q_t.reshape(1, H, S, L, -1).squeeze(0).transpose(0, 1)
            k_b = k_t.reshape(1, Hkv, S, L, -1).squeeze(0).transpose(0, 1)
            v_b = v_t.reshape(1, Hkv, S, L, -1).squeeze(0).transpose(0, 1)
            o_b = F.scaled_dot_product_attention(
                q_b,
                k_b,
                v_b,
                is_causal = args.causal,
                enable_gqa = args.is_gqa(),
                scale = args.sm_scale,
            )
            return o_b.transpose(0, 1).reshape(1, H, S * L, -1).to(out_dtype).transpose(1, 2)

        o = [
            F.scaled_dot_product_attention(
                q_t[:, :, s:e],
                k_t[:, :, s:e],
                v_t[:, :, s:e],
                is_causal = args.causal,
                enable_gqa = args.is_gqa(),
                scale = args.sm_scale,
            )
            for s, e in zip(bounds[:-1], bounds[1:])
        ]
        return torch.cat(o, dim = 2).to(out_dtype).transpose(1, 2)

    return F.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        is_causal = args.causal,
        enable_gqa = args.is_gqa(),
        scale = args.sm_scale,
    ).to(out_dtype).transpose(1, 2)


def fn_torch_sdpa_fallback_cache(args: AttnArgs) -> torch.Tensor | None:
    if (
        args.is_varlen() or
        not args.has_kv_cache() or
        args.dim < 512 or
        args.softcap != 0.0 or
        args.sinks is not None or
        args.is_swa()
    ):
        return None

    if args.dim > 256:
        _warn_sdpa_fallback()

    if not args.non_causal_spans:
        return _torch_bighead_fallback(
            q = args.q,
            k = args.k,
            v = args.v,
            k_cache = args.k_cache,
            v_cache = args.v_cache,
            block_table = args.block_table,
            cache_seqlens = args.cache_seqlens,
            causal = args.causal,
            softmax_scale = args.sm_scale,
            window_size = args.get_window_size(),
            softcap = args.softcap
        )
    else:
        arglist = get_non_causal_span_arglist(args)
        o = [_torch_bighead_fallback(**a) for a in arglist]
        return torch.cat(o, dim = 1)


def _torch_bighead_fallback(
    q, k, v,
    k_cache, v_cache,
    block_table, cache_seqlens,
    causal = False, softmax_scale = None,
    window_size = None,
    softcap = None,
    chunk_size = 512,
):
    batch, seqlen_q, nheads, headdim = q.shape
    _, seqlen_new, nheads_k, _ = k.shape
    block_size = k_cache.shape[1]

    _warn_sdpa_fallback()
    outputs = []
    for b in range(batch):
        seq_len = cache_seqlens[b].item()
        total_len = seq_len + seqlen_new

        # Gather this sequence's blocks into a contiguous page-aligned buffer
        num_blocks_needed = (total_len + block_size - 1) // block_size
        phys_blocks = block_table[b, :num_blocks_needed]
        k_buf = k_cache[phys_blocks].reshape(-1, nheads_k, headdim)
        v_buf = v_cache[phys_blocks].reshape(-1, nheads_k, headdim)

        # In-place copy new tokens into the buffer
        k_buf[seq_len:total_len] = k[b]
        v_buf[seq_len:total_len] = v[b]

        # Transpose kv once for all chunks: (1, nheads_k, buf_len, headdim)
        k_sdpa_full = k_buf.transpose(0, 1).unsqueeze(0)
        v_sdpa_full = v_buf.transpose(0, 1).unsqueeze(0)

        # Process q in chunks
        chunk_outputs = []
        for chunk_start in range(0, seqlen_q, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seqlen_q)
            q_chunk = q[b, chunk_start:chunk_end]  # (chunk_len, nheads, headdim)
            q_sdpa = q_chunk.transpose(0, 1).unsqueeze(0)
            chunk_len = chunk_end - chunk_start

            if causal:
                # This chunk's last query sits at absolute position: (total_len - seqlen_q) + chunk_end - 1
                # So it only needs kv up to that position (inclusive)
                kv_end = total_len - seqlen_q + chunk_end
                k_sdpa = k_sdpa_full[:, :, :kv_end]
                v_sdpa = v_sdpa_full[:, :, :kv_end]
                attn_mask = _causal_lower_right(chunk_len, kv_end)
            else:
                k_sdpa = k_sdpa_full[:, :, :total_len]
                v_sdpa = v_sdpa_full[:, :, :total_len]
                attn_mask = None

            o = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa,
                attn_mask = attn_mask,
                scale = softmax_scale,
                enable_gqa = True,
            )
            chunk_outputs.append(o.squeeze(0).transpose(0, 1))

        outputs.append(torch.cat(chunk_outputs, dim = 0))

        # Write back only the new tokens to the paged cache
        first_block = seq_len // block_size
        last_block = (total_len - 1) // block_size

        for block_idx in range(first_block, last_block + 1):
            phys = phys_blocks[block_idx]
            blk_start = block_idx * block_size
            blk_end = blk_start + block_size

            write_start = max(blk_start, seq_len)
            write_end = min(blk_end, total_len)

            off_start = write_start - blk_start
            off_end = write_end - blk_start

            src_start = write_start - seq_len
            src_end = write_end - seq_len

            k_cache[phys, off_start:off_end] = k[b, src_start:src_end]
            v_cache[phys, off_start:off_end] = v[b, src_start:src_end]

    return torch.stack(outputs)
