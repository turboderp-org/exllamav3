from __future__ import annotations
from typing_extensions import override
import numpy as np
import torch
from .cache import CacheLayer
from .fp16 import CacheLayer_fp16
from .quant import CacheLayer_quant
from ..constants import PAGE_SIZE


class QSAPlanes:
    """
    Mixin adding the QSA indexer's side planes to a K/V cache layer: per-token RAW indexer keys
    (unnormed, unroped) and per-4-token-block POOLED keys (fp32 mean -> norm -> rope at block
    start, written once a block completes). PAGE_SIZE is a multiple of the compress ratio, so
    blocks never straddle pages and page sharing / copy-on-write / defrag carry the planes along
    with the KV they describe. The planes stay fp16 whatever the K/V storage is: they are the
    input to every block score, and 128 + 32 values per token is small next to the K/V.
    """

    def _init_planes(self, attention, max_num_tokens: int):
        idx = attention.qsa_indexer
        assert idx is not None
        self.index_head_dim = idx.head_dim
        self.compress_ratio = idx.compress_ratio
        assert PAGE_SIZE % self.compress_ratio == 0
        num_pages = max_num_tokens // PAGE_SIZE
        self.raw_k_shape = (num_pages, PAGE_SIZE, self.index_head_dim)
        self.pooled_shape = (num_pages, PAGE_SIZE // self.compress_ratio, self.index_head_dim)
        self.raw_k = None
        self.pooled = None

    @override
    def alloc(self, device: torch.device):
        super().alloc(device)
        self.raw_k = torch.zeros(self.raw_k_shape, dtype = torch.half, device = device)
        self.pooled = torch.zeros(self.pooled_shape, dtype = torch.half, device = device)

    @override
    def free(self):
        super().free()
        self.raw_k = None
        self.pooled = None

    @override
    def copy_page(self, source, from_page: int, to_page: int, num_tokens: int):
        super().copy_page(source, from_page, to_page, num_tokens)
        self.raw_k[to_page, :num_tokens].copy_(source.raw_k[from_page, :num_tokens], non_blocking = True)
        nb = (num_tokens + self.compress_ratio - 1) // self.compress_ratio
        self.pooled[to_page, :nb].copy_(source.pooled[from_page, :nb], non_blocking = True)

    @override
    def get_tensors(self):
        return super().get_tensors() + [self.raw_k, self.pooled]

    @override
    def storage_size(self):
        return super().storage_size() + \
            (np.prod(self.raw_k_shape) + np.prod(self.pooled_shape)) * torch.half.itemsize


class CacheLayer_qsa(QSAPlanes, CacheLayer_fp16):
    """fp16 KV cache layer with the QSA indexer planes."""

    def __init__(
        self,
        config,
        attention,
        cache_id: int,
        max_num_tokens: int,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)
        self._init_planes(attention, max_num_tokens)

    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_qsa,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens
            }
        }


class CacheLayer_qsa_quant(QSAPlanes, CacheLayer_quant):
    """Quantized KV cache layer (CacheLayer_quant packing, read online by the dense and the
    gathered sparse attention kernels) with the fp16 QSA indexer planes."""

    def __init__(
        self,
        config,
        attention,
        cache_id: int,
        max_num_tokens: int,
        k_bits: int,
        v_bits: int,
        compand_a: float = 0.0,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens, k_bits, v_bits, compand_a)
        self._init_planes(attention, max_num_tokens)

    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_qsa_quant,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "k_bits": self.k_bits,
                "v_bits": self.v_bits,
            }
        }
