import torch
from .common import AttnArgs, AttnFn, get_non_causal_span_arglist

try:
    import xformers.ops as xops
    from xformers.ops.fmha import ck, cutlass, flash
    from xformers.ops.fmha.attn_bias import (
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        BlockDiagonalPaddedKeysMask,
        LowerTriangularFromBottomRightMask,
        LowerTriangularMask,
    )
    # Monkey-patch xformers for sm_120 support, what could go wrong
    from xformers.ops.fmha import cutlass as xf_cutlass
    xf_cutlass.FwOp.CUDA_MAXIMUM_COMPUTE_CAPABILITY = (12, 0)
    has_xformers = True
except ModuleNotFoundError:
    has_xformers = False


# Hack required for xformers currently since GQA is broken
# https://github.com/facebookresearch/xformers/issues/1392
def _stable_xformers_gqa_via_4d(q5, k5, v5, attn_bias = None, scale = None):
    B, Mq, G, H, K = q5.shape
    out = torch.empty(B, Mq, G * H, K, device = q5.device, dtype = q5.dtype)
    for g in range(G):
        qg = q5[:, :, g]
        kg = k5[:, :, g, :1, :].expand(-1, -1, H, -1)
        vg = v5[:, :, g, :1, :].expand(-1, -1, H, -1)
        og = xops.memory_efficient_attention(
            qg, kg, vg,
            attn_bias = attn_bias,
            scale = scale,
            op = (cutlass.FwOp, None),
        )
        out[:, :, g * H:(g + 1) * H, :] = og
    return out


def fn_xformers_cutlass_fallback_nocache(args: AttnArgs) -> torch.Tensor | None:
    if (
        not has_xformers or
        args.is_varlen() or
        args.has_kv_cache() or
        args.dim < 512 or
        args.softcap != 0.0 or
        args.non_causal_spans or
        args.is_swa()
    ):
        return None

    if args.is_gqa():
        ngroups = args.num_q_heads // args.num_kv_heads
        q_ = args.q.view(args.bsz, args.q_len, args.num_kv_heads, ngroups, args.dim)
        k_ = args.k.unsqueeze(3).expand(-1, -1, -1, ngroups, -1)
        v_ = args.v.unsqueeze(3).expand(-1, -1, -1, ngroups, -1)

        o = _stable_xformers_gqa_via_4d(
            q_, k_, v_,
            attn_bias = LowerTriangularMask() if args.causal else None,
            scale = args.sm_scale,
        )
        o = o.reshape(args.bsz, args.q_len, args.num_q_heads, args.dim)

    else:
        o = xops.memory_efficient_attention(
            args.q, args.k, args.v,
            attn_bias = LowerTriangularMask() if args.causal else None,
            scale = args.sm_scale,
            op = (cutlass.FwOp, None),
        )

    return o


def fn_xformers_cutlass_fallback_cache(args: AttnArgs) -> torch.Tensor | None:
    if (
        not has_xformers or
        args.is_varlen() or
        not args.has_kv_cache() or
        args.dim < 512 or
        args.softcap != 0.0 or
        args.is_swa()
    ):
        return None

    if not args.non_causal_spans:
        return _xformers_bighead_fallback(
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
        o = [_xformers_bighead_fallback(**a) for a in arglist]
        return torch.cat(o, dim = 1)


def _xformers_bighead_fallback(
    q, k, v,
    k_cache, v_cache,
    block_table, cache_seqlens,
    causal = False,
    softmax_scale = None,
    window_size = None,
    softcap = None,
    chunk_size = 1024,
):
    batch, seqlen_q, nheads, headdim = q.shape
    _, seqlen_new, nheads_k, _ = k.shape
    block_size = k_cache.shape[1]
    ngroups = nheads // nheads_k

    outputs = []
    for b in range(batch):
        seq_len = cache_seqlens[b].item()
        total_len = seq_len + seqlen_new

        # Gather pages into contiguous buffer
        num_blocks_needed = (total_len + block_size - 1) // block_size
        phys_blocks = block_table[b, :num_blocks_needed]
        k_buf = k_cache[phys_blocks].view(-1, nheads_k, headdim)
        v_buf = v_cache[phys_blocks].view(-1, nheads_k, headdim)

        # Write new tokens into buffer
        k_buf[seq_len:total_len] = k[b]
        v_buf[seq_len:total_len] = v[b]

        # Prepare for xformers: (1, seq, nheads_k, ngroups, headdim) for GQA
        k_xf = k_buf[:total_len].unsqueeze(0)  # (1, total_len, nheads_k, headdim)
        v_xf = v_buf[:total_len].unsqueeze(0)

        attn_bias = LowerTriangularFromBottomRightMask() if causal else None

        if ngroups > 1:
            q_xf = q[b].unsqueeze(0).view(1, seqlen_q, nheads_k, ngroups, headdim)
            k_xf = k_xf.unsqueeze(3).expand(-1, -1, -1, ngroups, -1)
            v_xf = v_xf.unsqueeze(3).expand(-1, -1, -1, ngroups, -1)
            o = _stable_xformers_gqa_via_4d(
                q_xf, k_xf, v_xf,
                attn_bias = attn_bias,
                scale = softmax_scale,
            )
            o = o.reshape(1, seqlen_q, nheads, headdim)
        else:
            q_xf = q[b].unsqueeze(0)
            o = xops.memory_efficient_attention(
                q_xf, k_xf, v_xf,
                attn_bias = attn_bias,
                scale = softmax_scale,
                op = (cutlass.FwOp, None),
            )

        outputs.append(o.squeeze(0))

        # Write back affected pages
        first_block = seq_len // block_size
        last_block = (total_len - 1) // block_size
        for block_idx in range(first_block, last_block + 1):
            phys = phys_blocks[block_idx]
            blk_start = block_idx * block_size
            write_start = max(blk_start, seq_len)
            write_end = min(blk_start + block_size, total_len)
            off_start = write_start - blk_start
            off_end = write_end - blk_start
            src_start = write_start - seq_len
            src_end = write_end - seq_len
            k_cache[phys, off_start:off_end] = k[b, src_start:src_end]
            v_cache[phys, off_start:off_end] = v[b, src_start:src_end]

    return torch.stack(outputs)