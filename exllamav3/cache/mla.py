from __future__ import annotations
from typing_extensions import override
import torch
from ..constants import PAGE_SIZE
from .cache import CacheLayer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..modules import MLAttention
    from ..model import Config
import numpy as np


class CacheLayer_MLA_fp16(CacheLayer):
    """
    Cache layer for multi-head latent attention.

    MLA is MQA over a compressed latent, so instead of per-head K and V this stores one latent
    vector and one shared RoPE key per token:

        k -> ckv  (pages, PAGE_SIZE, 1, kv_lora_rank)
        v -> kpe  (pages, PAGE_SIZE, 1, qk_rope_head_dim)

    The two tensors occupy the regular k/v slots, so paging, page copying, storage accounting and
    TP export all work unchanged; only the widths differ, and the latent doubles as both K and V
    inside the attention kernel.

    For a 128-head model this is 1152 bytes per token per layer against 81920 for the same model's
    expanded K/V, which is the entire reason to run attention in absorbed form.
    """

    def __init__(
        self,
        config: Config | None,
        attention: MLAttention,
        cache_id: int,
        max_num_tokens: int,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)

        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}."

        if attention:
            pages = max_num_tokens // PAGE_SIZE
            self.kv_lora_rank = attention.kv_lora_rank
            self.qk_rope_head_dim = attention.qk_rope_head_dim
            self.shape_c = (pages, PAGE_SIZE, 1, self.kv_lora_rank)
            self.shape_r = (pages, PAGE_SIZE, 1, self.qk_rope_head_dim)
        else:
            self.kv_lora_rank = None
            self.qk_rope_head_dim = None
            self.shape_c = None
            self.shape_r = None

        self.k = None    # latent, (pages, PAGE_SIZE, 1, kv_lora_rank)
        self.v = None    # rope key, (pages, PAGE_SIZE, 1, qk_rope_head_dim)
        self.device = None


    @override
    def alloc(self, device: torch.device):
        self.device = device
        self.k = torch.zeros(self.shape_c, dtype = torch.half, device = device) if self.shape_c else None
        self.v = torch.zeros(self.shape_r, dtype = torch.half, device = device) if self.shape_r else None


    @override
    def free(self):
        self.device = None
        self.k = None
        self.v = None


    @override
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor, sliding_window: int = -1) -> tuple:
        return self.k, self.v


    @override
    def update_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        # fp16 storage: attention already wrote through update_kv_direct
        pass


    @override
    def update_kv_direct(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        """Append new latent (k) and rope-key (v) rows, shaped (bsz, length, kv_lora_rank) and
        (bsz, length, qk_rope_head_dim)."""
        from ..modules.attention_fn.mla_triton import mla_kv_append
        mla_kv_append(
            k.reshape(k.shape[0], length, self.kv_lora_rank),
            v.reshape(v.shape[0], length, self.qk_rope_head_dim),
            self.k, self.v, block_table, cache_seqlens,
        )


    @override
    def copy_page(self, source: CacheLayer_MLA_fp16, from_page: int, to_page: int, num_tokens: int):
        assert self.shape_c == source.shape_c and self.shape_r == source.shape_r
        self.k[to_page, :num_tokens, :, :].copy_(source.k[from_page, :num_tokens, :, :], non_blocking = True)
        self.v[to_page, :num_tokens, :, :].copy_(source.v[from_page, :num_tokens, :, :], non_blocking = True)


    @override
    def get_tensors(self):
        return [self.k, self.v]


    @override
    def storage_size(self):
        return (np.prod(self.shape_c) + np.prod(self.shape_r)) * torch.half.itemsize


    @override
    def overhead_size(self):
        return 0


    @override
    def tp_export(self, plan):
        # The latent cache is MQA and cannot be split across ranks; each rank holds a full copy
        # and writes it independently from its replicated kv_a projection
        return {
            "cls": CacheLayer_MLA_fp16,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens
            }
        }


class CacheLayer_MLA_quant(CacheLayer):
    """
    Quantized MLA cache layer: the compressed latent is stored in the same packed format as the
    MHA quantized cache (32-value groups, absmax midpoint grid, power-of-two bit planes, values
    in the H32-rotated domain), written by the same CUDA quantizer and read online by the shared
    Triton plane loaders inside the MLA attention kernels -- no fp16 temporaries at any point.

    The shared RoPE key stays fp16 unconditionally: it is 64 values per token against the
    latent's 512, it is the input to every head's positional score, and keeping it exact removes
    the one part of the cache whose error is coherent across heads. Accordingly `k_bits` sets the
    latent width and `v_bits` is accepted for interface compatibility but unused (there is no
    separate V -- the latent is both).

    Per token at 512/64: fp16 1152 B; Q8 672 B (1.7x); Q6 544 B (2.1x); Q4 416 B (2.8x).
    """

    def __init__(
        self,
        config: Config | None,
        attention: MLAttention,
        cache_id: int,
        max_num_tokens: int,
        k_bits: int,
        v_bits: int | None = None,
        compand_a: float = 0.0,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)

        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}."
        assert 2 <= k_bits <= 8, "quantized MLA cache must be from 2 to 8 bits"
        assert compand_a == 0.0, \
            "compander is not supported by the online-dequant loaders (same as the MHA qc path)"

        self.bits = k_bits
        self.k_bits = k_bits
        self.v_bits = v_bits

        if attention:
            pages = max_num_tokens // PAGE_SIZE
            self.kv_lora_rank = attention.kv_lora_rank
            self.qk_rope_head_dim = attention.qk_rope_head_dim
            assert self.kv_lora_rank % 32 == 0
            groups = self.kv_lora_rank // 32
            self.qshape = (pages, PAGE_SIZE, groups * k_bits)
            self.sshape = (pages, PAGE_SIZE, groups)
            self.shape_r = (pages, PAGE_SIZE, 1, self.qk_rope_head_dim)
        else:
            self.qshape = None
            self.sshape = None
            self.shape_r = None

        self.qk = None   # packed latent, int32
        self.sk = None   # fp16 group scales
        self.v = None    # rope key, fp16
        self.device = None
        self._scratch = {}


    @override
    def alloc(self, device: torch.device):
        self.device = device
        self.qk = torch.zeros(self.qshape, dtype = torch.int, device = device) if self.qshape else None
        self.sk = torch.zeros(self.sshape, dtype = torch.half, device = device) if self.sshape else None
        self.v = torch.zeros(self.shape_r, dtype = torch.half, device = device) if self.shape_r else None


    @override
    def free(self):
        self.device = None
        self.qk = None
        self.sk = None
        self.v = None
        self._scratch = {}


    def get_qc(self):
        """Packed latent + scales + fp16 rope pages for the online-dequant attention kernels."""
        return self.qk, self.sk, self.v, self.bits


    @override
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor, sliding_window: int = -1) -> tuple:
        raise NotImplementedError(
            "CacheLayer_MLA_quant has no dequantize-to-fp16 path; the attention kernels read the "
            "packed cache directly via get_qc()"
        )


    @override
    def update_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        raise NotImplementedError("use update_kv_direct")


    @override
    def update_kv_direct(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        """Quantize new latent rows (k) into the packed pages and copy their rope keys (v)."""
        from ..modules.attention_fn.mla_triton import mla_kv_quant_append
        mla_kv_quant_append(
            k.reshape(k.shape[0], length, self.kv_lora_rank),
            v.reshape(v.shape[0], length, self.qk_rope_head_dim),
            self.qk,
            self.sk,
            self.v,
            block_table,
            cache_seqlens,
            self.bits,
        )


    @override
    def copy_page(self, source: CacheLayer_MLA_quant, from_page: int, to_page: int, num_tokens: int):
        assert self.qshape == source.qshape and self.shape_r == source.shape_r
        self.qk[to_page, :num_tokens, :].copy_(source.qk[from_page, :num_tokens, :], non_blocking = True)
        self.sk[to_page, :num_tokens, :].copy_(source.sk[from_page, :num_tokens, :], non_blocking = True)
        self.v[to_page, :num_tokens, :, :].copy_(source.v[from_page, :num_tokens, :, :], non_blocking = True)


    @override
    def get_tensors(self):
        return [self.qk, self.sk, self.v]


    @override
    def storage_size(self):
        return (
            np.prod(self.qshape) * torch.int.itemsize +
            np.prod(self.sshape) * torch.half.itemsize +
            np.prod(self.shape_r) * torch.half.itemsize
        )


    @override
    def overhead_size(self):
        # Contiguous quantization temporaries for one row (scaled by chunk length at runtime)
        return int(np.prod(self.qshape[2:])) * torch.int.itemsize + \
            int(np.prod(self.sshape[2:])) * torch.half.itemsize


    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_MLA_quant,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "k_bits": self.k_bits,
                "v_bits": self.v_bits,
            }
        }
