import torch
from ...cache import CacheLayer, Cache
from .common import AttnArgs, AttnFn
from .flash_attn_2 import fn_flash_attn_with_kvcache, fn_flash_attn_func, fn_flash_attn_varlen_func
from .bighead_scalar import fn_bighead_scalar_attn
from .torch import fn_torch_sdpa_fallback_cache, fn_torch_sdpa_fallback_nocache
from .xformers import fn_xformers_cutlass_fallback_cache, fn_xformers_cutlass_fallback_nocache
from .triton_paged import fn_triton_paged_attn, fn_triton_paged_attn_longq

# Candidate attn functions in order of preference
attn_fns: list[AttnFn] = [
    fn_flash_attn_with_kvcache,
    fn_flash_attn_func,
    fn_flash_attn_varlen_func,
    fn_bighead_scalar_attn,
    fn_triton_paged_attn,
    fn_triton_paged_attn_longq,
    fn_xformers_cutlass_fallback_cache,
    fn_xformers_cutlass_fallback_nocache,
    fn_torch_sdpa_fallback_cache,
    fn_torch_sdpa_fallback_nocache
]

def _tensor_desc(t: torch.Tensor | None) -> str:
    if t is None:
        return "None"
    return f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device} contiguous={t.is_contiguous()}"


def _print_no_attn_match_report(args: AttnArgs):
    tried = ", ".join(fn.__name__ for fn in attn_fns)
    print(
        "No matching attention function found.\n"
        f"  shape: bsz={args.bsz} q_len={args.q_len} kv_len={args.kv_len} "
        f"q_heads={args.num_q_heads} kv_heads={args.num_kv_heads} dim={args.dim}\n"
        f"  flags: cache={args.has_kv_cache()} varlen={args.is_varlen()} gqa={args.is_gqa()} "
        f"causal={args.causal} window_size={args.window_size} softcap={args.softcap} "
        f"non_causal_spans={args.non_causal_spans is not None}\n"
        f"  scale: sm_scale={args.sm_scale} max_seqlen={args.max_seqlen}\n"
        f"  q: {_tensor_desc(args.q)}\n"
        f"  k: {_tensor_desc(args.k)}\n"
        f"  v: {_tensor_desc(args.v)}\n"
        f"  k_cache: {_tensor_desc(args.k_cache)}\n"
        f"  v_cache: {_tensor_desc(args.v_cache)}\n"
        f"  block_table: {_tensor_desc(args.block_table)}\n"
        f"  cache_seqlens: {_tensor_desc(args.cache_seqlens)}\n"
        f"  cu_seqlens: {_tensor_desc(args.cu_seqlens)}\n"
        f"  tried: {tried}"
    )


def attn_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: CacheLayer | Cache | None = None,
    cache_idx: int | None = None,
    cache_instance: int | None = None,
    causal: bool = True,
    sm_scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    window_size: int | None = None,
    softcap: float = 0.0,
    block_table: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    non_causal_spans: list | None = None,
):
    bsz, q_len, num_q_heads, dim = q.shape
    _, kv_len, num_kv_heads, _ = k.shape

    # Get cache tensors
    if cache is not None:
        assert block_table is not None
        assert cache_seqlens is not None
        if isinstance(cache, CacheLayer):
            k_cache, v_cache = cache.get_kv(cache_seqlens, block_table, window_size)
        elif isinstance(cache, Cache):
            k_cache, v_cache = cache.get_layer(cache_idx, cache_seqlens, block_table, window_size, cache_instance)
    else:
        k_cache, v_cache = None, None

    # Defaults
    if sm_scale is None:
        sm_scale = dim ** (-0.5)

    # Dispatch
    args = AttnArgs(
        bsz, q_len, num_q_heads, dim,
        kv_len, num_kv_heads,
        q, k, v,
        k_cache, v_cache,
        causal,
        sm_scale,
        cu_seqlens, max_seqlen,
        window_size,
        softcap,
        block_table, cache_seqlens,
        non_causal_spans,
    )
    args.sanity_check()

    for fn in attn_fns:
        o = fn(args)
        if o is not None:
            break
    else:
        _print_no_attn_match_report(args)
        raise ValueError("No matching attention function")

    # Update cache
    if cache is not None:
        if isinstance(cache, CacheLayer):
            cache.update_kv(cache_seqlens, block_table, k_cache, v_cache, q_len)
        elif isinstance(cache, Cache):
            cache.update_layer(cache_idx, cache_seqlens, block_table, k_cache, v_cache, q_len, cache_instance)

    return o
