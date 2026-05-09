import torch
from .common import AttnArgs, AttnFn, get_non_causal_span_arglist
from torch.nn.attention.bias import causal_lower_right
import torch.nn.functional as F

has_warned_sdpa_fallback = False
def _warn_sdpa_fallback():
    global has_warned_sdpa_fallback
    if has_warned_sdpa_fallback:
        return
    col_default = "\u001b[0m"
    col_red = "\u001b[31;1m"
    print(
        f"{col_red} !! Warning, using SDPA fallback for large head size. VRAM usage will be high "
        f"and inference on long sequences will be slow. Consider installing `xformers` to improve "
        f"performance.{col_default}"
    )
    has_warned_sdpa_fallback = True


def fn_torch_sdpa_fallback_nocache(args: AttnArgs) -> torch.Tensor | None:
    if (
        args.has_kv_cache() or
        args.is_swa() or
        args.is_varlen() or
        args.softcap != 0.0 or
        args.non_causal_spans
    ):
        return None

    if args.dim > 256:
        _warn_sdpa_fallback()

    return F.scaled_dot_product_attention(
        args.q.transpose(1, 2),
        args.k.transpose(1, 2),
        args.v.transpose(1, 2),
        is_causal = args.causal,
        enable_gqa = args.is_gqa(),
        scale = args.sm_scale,
    ).transpose(1, 2)


def fn_torch_sdpa_fallback_cache(args: AttnArgs) -> torch.Tensor | None:
    if (
        args.is_varlen() or
        not args.has_kv_cache() or
        args.dim < 512 or
        args.softcap != 0.0 or
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
                attn_mask = causal_lower_right(chunk_len, kv_end)
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
