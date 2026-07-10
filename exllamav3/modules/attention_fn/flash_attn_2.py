import torch
from .common import AttnArgs, AttnFn, get_non_causal_span_arglist

# flash-attn 2 is an optional dependency; when absent (or broken), these backends decline every
# call and the built-in Triton kernels serve instead
try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache, flash_attn_varlen_func
    has_fa2 = True
except (ModuleNotFoundError, ImportError):
    has_fa2 = False

def fn_flash_attn_with_kvcache(args: AttnArgs) -> torch.Tensor | None:
    if (
        not has_fa2 or
        args.sinks is not None or
        args.is_varlen() or
        not args.has_kv_cache() or
        args.dim > 256
    ):
        return None

    if not args.non_causal_spans:
        return flash_attn_with_kvcache(
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
        o = [flash_attn_with_kvcache(**a) for a in arglist]
        return torch.cat(o, dim = 1)


def fn_flash_attn_func(args: AttnArgs) -> torch.Tensor | None:
    if (
        not has_fa2 or
        args.sinks is not None or
        args.is_varlen() or
        args.has_kv_cache() or
        args.dim > 256 or
        args.non_causal_spans
    ):
        return None

    return flash_attn_func(
        q = args.q,
        k = args.k,
        v = args.v,
        causal = args.causal,
        softmax_scale = args.sm_scale,
        window_size = args.get_window_size(),
        softcap = args.softcap,
    )


def fn_flash_attn_varlen_func(args: AttnArgs) -> torch.Tensor | None:
    if (
        not has_fa2 or
        args.sinks is not None or
        not args.is_varlen() or
        args.bsz > 1 or
        args.has_kv_cache() or
        args.dim > 256 or
        args.non_causal_spans
    ):
        return None

    return flash_attn_varlen_func(
        q = args.q.squeeze(0),
        k = args.k.squeeze(0),
        v = args.v.squeeze(0),
        cu_seqlens_q = args.cu_seqlens,
        cu_seqlens_k = args.cu_seqlens,
        max_seqlen_q = args.max_seqlen,
        max_seqlen_k = args.max_seqlen,
        causal = args.causal,
        softmax_scale = args.sm_scale,
        window_size = args.get_window_size(),
        softcap = args.softcap,
    )
