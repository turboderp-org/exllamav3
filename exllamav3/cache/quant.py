from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..constants import PAGE_SIZE
from ..model import Model, Config
from .cache import CacheLayer
from typing import TYPE_CHECKING
from exllamav3.ext import exllamav3_ext as ext
if TYPE_CHECKING:
    from ..modules import Attention
import numpy as np

class CacheLayer_quant(CacheLayer):

    def __init__(
        self,
        config: Config | None,
        attention: Attention,
        cache_id: int,
        max_num_tokens: int,
        k_bits: int,
        v_bits: int,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)

        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}."
        assert (2 <= k_bits <= 8) and (2 <= v_bits <= 8), "quantized cache must be from 2 to 8 bits"

        self.shape = (
            (max_num_tokens // PAGE_SIZE, PAGE_SIZE, attention.num_kv_heads, attention.head_dim)
            if attention else None
        )

        self.k_bits = k_bits
        self.v_bits = v_bits
        self.token_dim = attention.num_kv_heads * attention.head_dim
        self.qshape_k = ((max_num_tokens // PAGE_SIZE, PAGE_SIZE, self.token_dim // 32 * k_bits) if attention else None)
        self.qshape_v = ((max_num_tokens // PAGE_SIZE, PAGE_SIZE, self.token_dim // 32 * v_bits) if attention else None)
        self.qshape_s = ((max_num_tokens // PAGE_SIZE, PAGE_SIZE, self.token_dim // 32) if attention else None)

        self.qk = None
        self.qv = None
        self.sk = None
        self.sv = None
        self.device = None


    @override
    def alloc(self, device: torch.device):
        self.device = device
        self.qk = torch.zeros(self.qshape_k, dtype = torch.int, device = device) if self.shape else None
        self.qv = torch.zeros(self.qshape_v, dtype = torch.int, device = device) if self.shape else None
        self.sk = torch.zeros(self.qshape_s, dtype = torch.half, device = device) if self.shape else None
        self.sv = torch.zeros(self.qshape_s, dtype = torch.half, device = device) if self.shape else None


    @override
    def free(self):
        self.device = None
        self.qk = None
        self.qv = None
        self.sk = None
        self.sv = None


    @override
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor):
        k = torch.empty(self.shape, dtype = torch.half, device = self.device)
        v = torch.empty(self.shape, dtype = torch.half, device = self.device)
        ext.dequant_cache_paged(self.qk, self.sk, k, self.qv, self.sv, v, cache_seqlens, block_table, PAGE_SIZE)
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
        ext.quant_cache_paged(
            k, self.qk, self.sk,
            v, self.qv, self.sv,
            cache_seqlens, block_table,
            PAGE_SIZE,
            length
        )


    @override
    def copy_page(self, source: CacheLayer_quant, from_page: int, to_page: int, num_tokens: int):
        assert self.qshape_k == source.qshape_k
        assert self.qshape_v == source.qshape_v
        self.qk[to_page, :num_tokens, :].copy_(source.qk[from_page, :num_tokens, :], non_blocking = True)
        self.qv[to_page, :num_tokens, :].copy_(source.qv[from_page, :num_tokens, :], non_blocking = True)
        self.sk[to_page, :num_tokens, :].copy_(source.sk[from_page, :num_tokens, :], non_blocking = True)
        self.sv[to_page, :num_tokens, :].copy_(source.sv[from_page, :num_tokens, :], non_blocking = True)


    @override
    def get_tensors(self):
        return [self.qk, self.qv, self.sk, self.sv]


    @override
    def storage_size(self):
        return (
            np.prod(self.qshape_k) * torch.int.itemsize +
            np.prod(self.qshape_v) * torch.int.itemsize +
            2 * np.prod(self.qshape_s) * torch.half.itemsize
        )


    @override
    def overhead_size(self):
        return 2 * np.prod(self.shape[2:]) * torch.half.itemsize


    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_quant,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "k_bits": self.k_bits,
                "v_bits": self.v_bits,
            }
        }