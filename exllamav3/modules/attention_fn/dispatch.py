import torch
from ...cache import CacheLayer, Cache, CacheLayer_quant
from .common import AttnArgs, AttnFn
from .flash_attn_2 import fn_flash_attn_with_kvcache, fn_flash_attn_func, fn_flash_attn_varlen_func, has_fa2
from .bighead_scalar import fn_bighead_scalar_attn
from .torch import fn_torch_sdpa_fallback_cache, fn_torch_sdpa_fallback_nocache
from .xformers import fn_xformers_cutlass_fallback_cache, fn_xformers_cutlass_fallback_nocache
from .triton_paged import (
    fn_triton_paged_attn,
    fn_triton_paged_attn_longq,
    fn_triton_paged_attn_decode,
    fn_triton_paged_attn_prefill,
    fn_triton_varlen_attn,
    fn_triton_paged_attn_decode_qc,
    fn_triton_paged_attn_prefill_qc,
    fn_triton_attn_nocache,
    has_triton,
    _is_power_of_2,
)

# Candidate attn functions in order of preference. The Triton decode/prefill kernels lead by
# default (measured faster than FA2 on Ampere/Ada/consumer Blackwell); set EXL3_PREFER_FA2=1 to
# restore flash-attn priority for A/B testing (read once at import)
import os
_prefer_fa2 = os.environ.get("EXL3_PREFER_FA2", "0") != "0"
if _prefer_fa2 and not has_fa2:
    print(" !! EXL3_PREFER_FA2 is set but flash-attn is not available; using built-in kernels")
    _prefer_fa2 = False

_fns_triton_fast: list[AttnFn] = [
    fn_triton_paged_attn_decode,
    fn_triton_paged_attn_prefill,
    fn_triton_varlen_attn,
]

# Quant-direct calls carry the packed cache in q_cache and leave k_cache/v_cache as None, which makes them
# indistinguishable from cache-less attention to any backend that only checks has_kv_cache(). Such a backend
# would silently attend over just the new K/V rows and ignore the cached context, so quant-direct calls only
# ever dispatch over the qc-aware functions
_fns_qc: list[AttnFn] = [
    fn_triton_paged_attn_decode_qc,
    fn_triton_paged_attn_prefill_qc,
]

# Quantized caches feed the attention kernels directly (online dequant, no full-size fp16
# temporaries). EXL3_QC_ATTN=0 restores the dequantize-then-attend path for A/B testing
_qc_attn = os.environ.get("EXL3_QC_ATTN", "1") != "0"

_fns_fa2: list[AttnFn] = [
    fn_flash_attn_with_kvcache,
    fn_flash_attn_func,
    fn_flash_attn_varlen_func,
]

attn_fns: list[AttnFn] = (
    (_fns_fa2 + _fns_triton_fast) if _prefer_fa2 else (_fns_triton_fast + _fns_fa2)
) + [
    fn_triton_attn_nocache,
    fn_triton_paged_attn,
    fn_triton_paged_attn_longq,
    fn_bighead_scalar_attn,
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
        f"non_causal_spans={args.non_causal_spans is not None} sinks={args.sinks is not None}\n"
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
    sinks: torch.Tensor | None = None,
    dispatch_cache: dict | None = None,
):
    """
    Select and run the first compatible attention implementation for the supplied tensors.

    The dispatcher builds an AttnArgs description covering regular, varlen and paged-cache attention modes, obtains
    K/V cache tensors when a Cache or CacheLayer is provided, and tries registered attention backends in preference
    order. After a cached attention call, any updated K/V tensors are written back through the same cache interface.
    """
    bsz, q_len, num_q_heads, dim = q.shape
    _, kv_len, num_kv_heads, _ = k.shape

    # Get cache tensors. Quantized layers pass their packed tensors straight to the attention
    # kernels when possible: new K/V are quantized into the cache up front and never
    # materialized as full fp16 cache-sized temporaries
    q_cache = None
    if cache is not None:
        assert block_table is not None
        assert cache_seqlens is not None
        layer = cache if isinstance(cache, CacheLayer) else cache.layers[cache_idx, cache_instance or 0]
        if (
            _qc_attn and has_triton and
            isinstance(layer, CacheLayer_quant) and
            layer.compand_a == 0.0 and
            q.dtype == torch.float16 and
            dim <= 512 and _is_power_of_2(dim) and
            cu_seqlens is None
        ):
            layer.update_kv_direct(cache_seqlens, block_table, k, v, q_len)
            q_cache = layer.get_qkv()
            k_cache, v_cache = None, None
        else:
            k_cache, v_cache = layer.get_kv(cache_seqlens, block_table, window_size if window_size is not None else -1)
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
        q_cache,
        sinks,
    )
    # Quant-direct calls select among the qc-aware backends only; a separate hint slot keeps a function that
    # won a cache-less or fp16-cache call from being retried on quant-direct arguments (it cannot see q_cache
    # and would accept them as cache-less)
    candidates = _fns_qc if q_cache is not None else attn_fns
    hint_key = "fn_qc" if q_cache is not None else "fn"

    # Retry the backend that matched last time for this caller before scanning the full list.
    # Candidate functions return None on incompatible arguments, so a stale hint self-corrects
    fn = dispatch_cache.get(hint_key) if dispatch_cache is not None else None
    o = fn(args) if fn is not None else None

    if o is None:
        args.sanity_check()
        for fn in candidates:
            o = fn(args)
            if o is not None:
                break
        else:
            _print_no_attn_match_report(args)
            raise ValueError("No matching attention function")
        if dispatch_cache is not None:
            dispatch_cache[hint_key] = fn

    # Update cache (quant-direct mode already wrote the new K/V before the attention call)
    if cache is not None and q_cache is None:
        if isinstance(cache, CacheLayer):
            cache.update_kv(cache_seqlens, block_table, k_cache, v_cache, q_len)
        elif isinstance(cache, Cache):
            cache.update_layer(cache_idx, cache_seqlens, block_table, k_cache, v_cache, q_len, cache_instance)

    return o
