from __future__ import annotations
from typing_extensions import override
import torch
from ..constants import PAGE_SIZE
from ..model import Config
from .cache import CacheLayer
from typing import TYPE_CHECKING
from exllamav3.ext import exllamav3_ext as ext
if TYPE_CHECKING:
    from ..modules import Attention
import numpy as np

class CacheLayer_lloyd_max(CacheLayer):
    """
    KV-cache layer using Lloyd-Max optimal quantization.

    Identical interface to CacheLayer_quant; the only difference is that
    quant/dequant calls go to the Lloyd-Max CUDA kernels (quant_lm_cache_paged /
    dequant_lm_cache_paged) which use precomputed Gaussian-optimal codebooks
    instead of uniform rounding.  The on-disk / in-memory tensor layout is
    identical to CacheLayer_quant for the same bit-width.

    When asymmetric=True, K is quantized using per-sub-group scale + zero-point
    (uniform grid on [min, max]) while V continues to use the symmetric
    Lloyd-Max path.  This exploits the statistical structure of K vectors
    (non-zero mean, heterogeneous channel variance) for improved SQNR.
    Requires sub_scale_size=8 (sub-block scales); asymmetric with sub_scale_size=32
    is not supported.
    """

    def __init__(
        self,
        config: Config | None,
        attention: Attention,
        cache_id: int,
        max_num_tokens: int,
        k_bits: int,
        v_bits: int,
        sub_scale_size: int = 32,
        asymmetric: bool = False,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)

        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}."
        assert (2 <= k_bits <= 8) and (2 <= v_bits <= 8), "quantized cache must be from 2 to 8 bits"
        assert sub_scale_size in (8, 32), "sub_scale_size must be 8 (sub-block) or 32 (standard)"
        assert not (asymmetric and sub_scale_size != 8), \
            "asymmetric=True requires sub_scale_size=8"

        self.shape = (
            (max_num_tokens // PAGE_SIZE, PAGE_SIZE, attention.num_kv_heads, attention.head_dim)
            if attention else None
        )

        self.k_bits = k_bits
        self.v_bits = v_bits
        self.sub_scale_size = sub_scale_size
        self.asymmetric = asymmetric
        self.token_dim = attention.num_kv_heads * attention.head_dim
        self.qshape_k = ((max_num_tokens // PAGE_SIZE, PAGE_SIZE, self.token_dim // 32 * k_bits) if attention else None)
        self.qshape_v = ((max_num_tokens // PAGE_SIZE, PAGE_SIZE, self.token_dim // 32 * v_bits) if attention else None)
        self.qshape_s = ((max_num_tokens // PAGE_SIZE, PAGE_SIZE, self.token_dim // sub_scale_size) if attention else None)

        self.qk = None
        self.qv = None
        self.sk = None
        self.sv = None
        # Zero-point tensors for asymmetric K quantization (same shape as sk)
        self.zk = None
        self.device = None


    @override
    def alloc(self, device: torch.device):
        self.device = device
        self.qk = torch.zeros(self.qshape_k, dtype = torch.int, device = device) if self.shape else None
        self.qv = torch.zeros(self.qshape_v, dtype = torch.int, device = device) if self.shape else None
        self.sk = torch.zeros(self.qshape_s, dtype = torch.half, device = device) if self.shape else None
        self.sv = torch.zeros(self.qshape_s, dtype = torch.half, device = device) if self.shape else None
        if self.asymmetric:
            self.zk = torch.zeros(self.qshape_s, dtype = torch.half, device = device) if self.shape else None


    @override
    def free(self):
        self.device = None
        self.qk = None
        self.qv = None
        self.sk = None
        self.sv = None
        self.zk = None


    @override
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor):
        k = torch.empty(self.shape, dtype = torch.half, device = self.device)
        v = torch.empty(self.shape, dtype = torch.half, device = self.device)
        if self.asymmetric:
            ext.dequant_lm_cache_paged_asym(
                self.qk, self.sk, self.zk, k,
                self.qv, self.sv, v,
                cache_seqlens, block_table, PAGE_SIZE
            )
        elif self.sub_scale_size == 8:
            ext.dequant_lm_cache_paged_sub(self.qk, self.sk, k, self.qv, self.sv, v, cache_seqlens, block_table, PAGE_SIZE)
        else:
            ext.dequant_lm_cache_paged(self.qk, self.sk, k, self.qv, self.sv, v, cache_seqlens, block_table, PAGE_SIZE)
        return k, v


    @override
    def get_kv_alloc_placeholder(self):
        k = torch.empty(self.shape, dtype = torch.half, device = self.device)
        v = torch.empty(self.shape, dtype = torch.half, device = self.device)
        return k, v


    @override
    def update_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        if self.asymmetric:
            ext.quant_lm_cache_paged_asym(
                k, self.qk, self.sk, self.zk,
                v, self.qv, self.sv,
                cache_seqlens, block_table,
                PAGE_SIZE,
                length
            )
        elif self.sub_scale_size == 8:
            ext.quant_lm_cache_paged_sub(
                k, self.qk, self.sk,
                v, self.qv, self.sv,
                cache_seqlens, block_table,
                PAGE_SIZE,
                length
            )
        else:
            ext.quant_lm_cache_paged(
                k, self.qk, self.sk,
                v, self.qv, self.sv,
                cache_seqlens, block_table,
                PAGE_SIZE,
                length
            )


    @override
    def copy_page(self, source: CacheLayer_lloyd_max, from_page: int, to_page: int, num_tokens: int):
        assert self.qshape_k == source.qshape_k
        assert self.qshape_v == source.qshape_v
        self.qk[to_page, :num_tokens, :].copy_(source.qk[from_page, :num_tokens, :], non_blocking = True)
        self.qv[to_page, :num_tokens, :].copy_(source.qv[from_page, :num_tokens, :], non_blocking = True)
        self.sk[to_page, :num_tokens, :].copy_(source.sk[from_page, :num_tokens, :], non_blocking = True)
        self.sv[to_page, :num_tokens, :].copy_(source.sv[from_page, :num_tokens, :], non_blocking = True)
        if self.asymmetric and source.zk is not None:
            self.zk[to_page, :num_tokens, :].copy_(source.zk[from_page, :num_tokens, :], non_blocking = True)


    @override
    def get_tensors(self):
        tensors = [self.qk, self.qv, self.sk, self.sv]
        if self.asymmetric:
            tensors.append(self.zk)
        return tensors


    @override
    def storage_size(self):
        base = (
            np.prod(self.qshape_k) * torch.int.itemsize +
            np.prod(self.qshape_v) * torch.int.itemsize +
            2 * np.prod(self.qshape_s) * torch.half.itemsize
        )
        if self.asymmetric:
            # Additional zero-point tensor for K (same shape as sk)
            base += np.prod(self.qshape_s) * torch.half.itemsize
        return base


    @override
    def overhead_size(self):
        return 2 * np.prod(self.shape[2:]) * torch.half.itemsize


    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_lloyd_max,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "k_bits": self.k_bits,
                "v_bits": self.v_bits,
                "sub_scale_size": self.sub_scale_size,
                "asymmetric": self.asymmetric,
            }
        }
