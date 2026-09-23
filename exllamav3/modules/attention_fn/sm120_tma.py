import math
import os
from functools import lru_cache

import torch

from ...ext import exllamav3_ext as ext
from .common import AttnArgs


has_sm120_tma = hasattr(ext, "sm120_tma_attn_paged") and hasattr(ext, "sm120_tma_attn_supported")
_enabled = os.environ.get("EXL3_SM120_TMA_ATTN", "1") != "0"
_min_q = int(os.environ.get("EXL3_SM120_TMA_MIN_Q", "33"))
_split_env = int(os.environ.get("EXL3_SM120_TMA_SPLIT_K", "0"))
_q_group_env = int(os.environ.get("EXL3_SM120_TMA_Q_GROUP", "0"))


@lru_cache(maxsize = None)
def _supported(device: int) -> bool:
    return ext.sm120_tma_attn_supported(device)


@lru_cache(maxsize = None)
def _sm_count(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _pick_split(args: AttnArgs, q_group_mode: int) -> int:
    if args.dim == 128:
        return 1
    if _split_env in (1, 3):
        return _split_env

    ratio = args.num_q_heads // args.num_kv_heads
    group = q_group_mode or (8 if ratio % 8 == 0 else 2)
    ncols1 = 64 // group
    q_tiles = math.ceil(args.q_len / ncols1)
    gqa_tiles = math.ceil(ratio / group)
    programs = q_tiles * gqa_tiles * args.num_kv_heads * args.bsz
    # Prefer the host-known past length over the block table capacity, which covers the whole job
    if args.max_kv_len is not None:
        kv_len = args.max_kv_len + args.q_len
    else:
        kv_len = args.block_table.shape[1] * 256
    if kv_len < 8192 or programs == 0:
        return 1

    sms = _sm_count(args.q.device.index)
    waves = math.ceil(programs / sms)
    waves3 = math.ceil(3 * programs / sms)
    if waves3 / (3.0 * waves) > 0.95:
        return 1

    partial_bytes = 3 * args.bsz * args.q_len * args.num_q_heads * args.dim * 4
    return 3 if partial_bytes <= 256 * 1024 * 1024 else 1


def fn_sm120_tma_attn_prefill(args: AttnArgs) -> torch.Tensor | None:
    if (
        not has_sm120_tma or
        not _enabled or
        args.q.device.type != "cuda" or
        not _supported(args.q.device.index) or
        args.is_varlen() or
        not args.has_kv_cache() or
        args.q_cache is not None or
        args.q_len < _min_q or
        args.dim not in (128, 256, 512) or
        args.q.dtype != torch.float16 or
        args.k.dtype != torch.float16 or
        args.v.dtype != torch.float16 or
        args.k_cache.dtype != torch.float16 or
        args.v_cache.dtype != torch.float16 or
        args.k.shape[1] != args.q_len or
        args.non_causal_spans or
        args.sinks is not None or
        args.is_swa() or
        not args.q.is_contiguous() or
        not args.k.is_contiguous() or
        not args.v.is_contiguous() or
        not args.k_cache.is_contiguous() or
        not args.v_cache.is_contiguous() or
        not args.block_table.is_contiguous() or
        not args.cache_seqlens.is_contiguous()
    ):
        return None

    ratio = args.num_q_heads // args.num_kv_heads
    q_group_mode = _q_group_env
    if args.dim in (128, 512):
        q_group_mode = 8
    elif q_group_mode == 0:
        q_group_mode = 8 if ratio % 8 == 0 else 2
    elif q_group_mode not in (2, 8):
        q_group_mode = 8 if ratio % 8 == 0 else 2
    split_k = _pick_split(args, q_group_mode)

    return ext.sm120_tma_attn_paged(
        args.q,
        args.k,
        args.v,
        args.k_cache,
        args.v_cache,
        args.block_table,
        args.cache_seqlens,
        args.causal,
        args.sm_scale,
        args.softcap,
        split_k,
        q_group_mode,
    )
