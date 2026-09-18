from __future__ import annotations
from typing_extensions import override
import torch
from ..constants import PAGE_SIZE
from .cache import CacheLayer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..modules import Attention
    from ..model import Model, Config
import numpy as np

# NVFP4 element format: E2M1 (1 sign, 2 exp, 1 mantissa), 16 levels
#   code: [s | 2 exp bits | 1 mantissa bit]; positive magnitudes by idx (code & 7):
#   0:0  1:0.5  2:1  3:1.5  4:2  5:3  6:4  7:6
# Block scale: one E4M3 per 16 elements = amax/6 (clamped to E4M3 range). Two 4-bit
# codes per uint8 byte, low nibble = even element index (matches the RadixArk NVFP4
# checkpoint layout verified empirically during the M0.4 dequant work).

_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)   # nearest-level boundaries


def nvfp4_quantize(x: torch.Tensor):
    """fp16/bf16 [..., G] with G % 16 == 0 -> (packed uint8 [..., G//2], scales e4m3 [..., G//16])"""
    x = x.half()
    g = x.shape[-1]
    xg = x.reshape(*x.shape[:-1], g // 16, 16)
    amax = xg.abs().amax(dim = -1, keepdim = True)
    scale = (amax / 6.0).clamp(max = 448.0).to(torch.float8_e4m3fn)
    sc = scale.half()                                     # decoded scale
    t = (xg / torch.where(sc == 0, torch.ones_like(sc), sc)).abs().clamp(max = 6.0)
    idx = torch.zeros_like(t, dtype = torch.uint8)
    for b in _E2M1_BOUNDS:
        idx += (t > b).to(torch.uint8)
    codes = idx | ((xg < 0).to(torch.uint8) << 3)
    codes = codes.reshape(*x.shape[:-1], g)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), scale.squeeze(-1).contiguous()


def nvfp4_dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of nvfp4_quantize -> fp16 [..., G]"""
    codes_lo = (packed & 0xF).to(torch.uint8)
    codes_hi = (packed >> 4).to(torch.uint8)
    codes = torch.stack((codes_lo, codes_hi), dim = -1).flatten(-2)
    idx = codes & 7
    sgn = (codes >> 3).half()
    t = idx.half()
    mag = torch.zeros_like(t)
    for i, v in enumerate((0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)):
        mag += torch.where(idx == i + 1, torch.full_like(t, v), torch.zeros_like(t))
    g = codes.shape[-1]
    sc = scales.half().repeat_interleave(16, dim = -1)   # [.., G/16] -> [.., G]
    val = mag * torch.where(sgn > 0, torch.full_like(mag, -1.0), torch.ones_like(mag))
    return (val * sc).contiguous()


class CacheLayer_nvfp4(CacheLayer):
    """KV cache stored in the NVFP4 format: E2M1 elements + E4M3 block scales (16 elems/
    block), 4.5 bits per element. Reads take the Triton online-dequant path; the generic
    get_kv returns dequantized fp16 tensors for non-Triton backends and diagnostics."""

    def __init__(
        self,
        config: Config | None,
        attention: Attention,
        cache_id: int,
        max_num_tokens: int,
        **kwargs,   # accept k_bits/v_bits/compand_a for Cache(**layer_kwargs) compatibility
    ):
        super().__init__(config, attention, cache_id, max_num_tokens)

        assert max_num_tokens % PAGE_SIZE == 0, \
            f"max_num_tokens must be a multiple of {PAGE_SIZE}."
        assert attention is None or attention.head_dim % 32 == 0, \
            "NVFP4 cache requires head_dim % 32 == 0 (byte and block alignment)"

        if attention:
            pages = max_num_tokens // PAGE_SIZE
            h, d = attention.num_kv_heads, attention.head_dim
            self.shape4 = (pages, PAGE_SIZE, h, d // 2)
            self.shape_s = (pages, PAGE_SIZE, h, d // 16)
        else:
            self.shape4 = None
            self.shape_s = None
        self.shape = None
        self.k = None
        self.v = None
        self.ks = None
        self.vs = None
        self.device = None


    @override
    def alloc(self, device: torch.device):
        self.device = device
        self.k = torch.zeros(self.shape4, dtype = torch.uint8, device = device) if self.shape4 else None
        self.v = torch.zeros(self.shape4, dtype = torch.uint8, device = device) if self.shape4 else None
        self.ks = torch.zeros(self.shape_s, dtype = torch.float8_e4m3fn, device = device) if self.shape_s else None
        self.vs = torch.zeros(self.shape_s, dtype = torch.float8_e4m3fn, device = device) if self.shape_s else None


    @override
    def free(self):
        self.device = None
        self.k = self.v = self.ks = self.vs = None


    @override
    def get_qkv(self):
        return None    # not a packed-int quant layer


    def get_paged(self) -> tuple:
        """(k4, k_scales, v4, v_scales) for the Triton online-dequant path."""
        return self.k, self.ks, self.v, self.vs


    @override
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor, sliding_window: int = -1) -> tuple:
        # Raw tensors are uint8; only meaningful to callers that know the format. The
        # attention dispatch dequantizes when handing pages to generic backends.
        return self.k, self.v


    def dequant_full(self) -> tuple:
        k = nvfp4_dequantize(self.k.int(), self.ks.half()) if self.k is not None else None
        v = nvfp4_dequantize(self.v.int(), self.vs.half()) if self.v is not None else None
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
        # Torch quantize + scatter (fallback path and pre-write callers). The Triton fast
        # path quantizes and appends in-kernel instead
        bsz = k.shape[0]
        for b in range(bsz):
            pos = int(cache_seqlens[b])
            rows = torch.arange(pos, pos + length, device = k.device)
            page_idx = rows // PAGE_SIZE
            page_off = rows - page_idx * PAGE_SIZE
            phys = block_table[b].index_select(0, page_idx)
            kq, ksc = nvfp4_quantize(k[b, : length])
            vq, vsc = nvfp4_quantize(v[b, : length])
            self.k[phys, page_off] = kq
            self.ks[phys, page_off] = ksc
            self.v[phys, page_off] = vq
            self.vs[phys, page_off] = vsc


    @override
    def update_kv_direct(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        self.update_kv(cache_seqlens, block_table, k, v, length)


    @override
    def copy_page(self, source: CacheLayer_nvfp4, from_page: int, to_page: int, num_tokens: int):
        assert self.shape4 == source.shape4
        self.k[to_page, :num_tokens].copy_(source.k[from_page, :num_tokens], non_blocking = True)
        self.v[to_page, :num_tokens].copy_(source.v[from_page, :num_tokens], non_blocking = True)
        self.ks[to_page, :num_tokens].copy_(source.ks[from_page, :num_tokens], non_blocking = True)
        self.vs[to_page, :num_tokens].copy_(source.vs[from_page, :num_tokens], non_blocking = True)


    @override
    def get_tensors(self):
        return [self.k, self.v, self.ks, self.vs]


    @override
    def storage_size(self):
        if not self.shape4:
            return 0
        # K and V have identical packed layouts (bytes + one fp8 scale per 16 elems)
        return (np.prod(self.shape4) + np.prod(self.shape_s)) * 2


    @override
    def overhead_size(self):
        return 0


    @override
    def tp_export(self, plan):
        return {
            "cls": CacheLayer_nvfp4,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens
            }
        }
