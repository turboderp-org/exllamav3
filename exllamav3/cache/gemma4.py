from __future__ import annotations

from typing import TYPE_CHECKING, Type

import torch
from typing_extensions import override

from ..constants import PAGE_SIZE
from ..ext import exllamav3_ext as ext
from ..model import Config
from .fp16 import CacheLayer_fp16
from .quant import CacheLayer_quant

if TYPE_CHECKING:
    from ..modules import Attention


def _normalize_role_max_tokens(role_max_tokens: int | None, default_max_tokens: int | None) -> int | None:
    if role_max_tokens is None:
        return default_max_tokens
    if default_max_tokens is None:
        return role_max_tokens
    role_max_tokens = min(role_max_tokens, default_max_tokens)
    role_max_tokens = max(PAGE_SIZE, role_max_tokens)
    role_max_tokens = (role_max_tokens // PAGE_SIZE) * PAGE_SIZE
    return role_max_tokens


def _default_swa_max_tokens(attention: "Attention", default_max_tokens: int | None) -> int | None:
    if default_max_tokens is None:
        return None
    sliding_window = getattr(attention, "sliding_window", -1)
    if sliding_window is None or sliding_window < 0:
        return default_max_tokens
    # Mirror llama.cpp's intent: keep SWA sized to the sliding window plus a
    # modest runtime headroom instead of inheriting the full cache budget.
    return _normalize_role_max_tokens(sliding_window + PAGE_SIZE, default_max_tokens)


class Gemma4QuantCacheLayer(CacheLayer_quant):

    def __init__(
        self,
        config: Config | None,
        attention: Attention,
        cache_id: int,
        max_num_tokens: int,
        k_bits: int,
        v_bits: int,
    ):
        super().__init__(config, attention, cache_id, max_num_tokens, k_bits, v_bits)
        self.shadow_k_pages: dict[int, torch.Tensor] | None = None
        self.shadow_v_pages: dict[int, torch.Tensor] | None = None

    @override
    def alloc(self, device: torch.device):
        super().alloc(device)
        self.shadow_k_pages = {}
        self.shadow_v_pages = {}

    @override
    def free(self):
        super().free()
        self.shadow_k_pages = None
        self.shadow_v_pages = None

    def _ensure_shadow_page(self, page_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.device is not None
        assert self.shadow_k_pages is not None and self.shadow_v_pages is not None
        sk = self.shadow_k_pages.get(page_idx)
        sv = self.shadow_v_pages.get(page_idx)
        if sk is None or sv is None:
            shape = self.shape[1:]
            sk = torch.zeros(shape, dtype = torch.half, device = self.device)
            sv = torch.zeros(shape, dtype = torch.half, device = self.device)
            self.shadow_k_pages[page_idx] = sk
            self.shadow_v_pages[page_idx] = sv
        return sk, sv

    def write_shadow_pages(
        self,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        bsz, seqlen, _, _ = k.shape
        positions = (
            cache_seqlens.to(dtype = torch.long).unsqueeze(1) +
            torch.arange(seqlen, device = block_table.device, dtype = torch.long).unsqueeze(0)
        )
        page_idx = block_table.gather(1, torch.div(positions, PAGE_SIZE, rounding_mode = "floor")).reshape(-1)
        page_pos = positions.remainder(PAGE_SIZE).reshape(-1)
        flat_k = k.reshape(-1, self.attention.num_kv_heads, self.attention.head_dim)
        flat_v = v.reshape(-1, self.attention.num_kv_heads, self.attention.head_dim)

        unique_pages, inverse = torch.unique(page_idx, sorted = False, return_inverse = True)
        for local_idx, page in enumerate(unique_pages.tolist()):
            mask = inverse == local_idx
            sk, sv = self._ensure_shadow_page(int(page))
            local_pos = page_pos[mask].to(dtype = torch.long)
            sk[local_pos] = flat_k[mask]
            sv[local_pos] = flat_v[mask]

    def gather_shadow_pages(
        self,
        block_table: torch.Tensor,
        total_lens: torch.Tensor,
        gathered_k: torch.Tensor | None = None,
        gathered_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.device is not None
        assert self.shadow_k_pages is not None and self.shadow_v_pages is not None
        bsz = block_table.shape[0]
        max_total = int(total_lens.max())
        target_shape = (bsz, max_total, self.attention.num_kv_heads, self.attention.head_dim)
        target_shape_heads = (bsz, self.attention.num_kv_heads, max_total, self.attention.head_dim)
        heads_first = gathered_k is not None and gathered_k.shape == target_shape_heads
        if gathered_k is None:
            gathered_k = torch.empty(target_shape, dtype = torch.half, device = self.device)
        elif gathered_k.shape not in (target_shape, target_shape_heads):
            gathered_k = torch.empty(target_shape, dtype = torch.half, device = self.device)
            heads_first = False
        if gathered_v is None:
            gathered_v = torch.empty(target_shape, dtype = torch.half, device = self.device)
        elif gathered_v.shape not in (target_shape, target_shape_heads):
            gathered_v = torch.empty(target_shape, dtype = torch.half, device = self.device)
            heads_first = False
        for b in range(bsz):
            total = int(total_lens[b])
            if total == 0:
                continue
            num_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
            pages = block_table[b, :num_pages].to(dtype = torch.long).tolist()
            shadow_k = torch.stack([self._ensure_shadow_page(page_idx)[0] for page_idx in pages], dim = 0)
            shadow_v = torch.stack([self._ensure_shadow_page(page_idx)[1] for page_idx in pages], dim = 0)
            flat_k = shadow_k.reshape(-1, self.attention.num_kv_heads, self.attention.head_dim)
            flat_v = shadow_v.reshape(-1, self.attention.num_kv_heads, self.attention.head_dim)
            if heads_first:
                gathered_k[b, :, :total].copy_(flat_k[:total].transpose(0, 1), non_blocking = True)
                gathered_v[b, :, :total].copy_(flat_v[:total].transpose(0, 1), non_blocking = True)
            else:
                gathered_k[b, :total].copy_(flat_k[:total], non_blocking = True)
                gathered_v[b, :total].copy_(flat_v[:total], non_blocking = True)
        return gathered_k, gathered_v

    @override
    def copy_page(self, source: "Gemma4QuantCacheLayer", from_page: int, to_page: int, num_tokens: int):
        super().copy_page(source, from_page, to_page, num_tokens)
        if (
            source.shadow_k_pages is None or
            source.shadow_v_pages is None or
            self.shadow_k_pages is None or
            self.shadow_v_pages is None
        ):
            return
        src_k = source.shadow_k_pages.get(from_page)
        src_v = source.shadow_v_pages.get(from_page)
        if src_k is None or src_v is None:
            return
        dst_k, dst_v = self._ensure_shadow_page(to_page)
        dst_k[:num_tokens].copy_(src_k[:num_tokens], non_blocking = True)
        dst_v[:num_tokens].copy_(src_v[:num_tokens], non_blocking = True)

    @override
    def tp_export(self, plan):
        return {
            "cls": self.__class__,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
                "k_bits": self.k_bits,
                "v_bits": self.v_bits,
            }
        }


class Gemma4SingleQuantCacheLayer(Gemma4QuantCacheLayer):

    def _build_compact_update_index_map(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        length: int,
        local_block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, blocks_per_seq = block_table.shape
        positions = (
            cache_seqlens.to(dtype = torch.long).unsqueeze(1) +
            torch.arange(length, device = block_table.device, dtype = torch.long).unsqueeze(0)
        )
        page_idx = torch.div(positions, PAGE_SIZE, rounding_mode = "floor")
        page_pos = positions.remainder(PAGE_SIZE)

        if local_block_table is None:
            local_pages = bsz * blocks_per_seq
            local_block_table = torch.arange(
                local_pages,
                dtype = block_table.dtype,
                device = block_table.device,
            ).view(bsz, blocks_per_seq)

        local_page = local_block_table.gather(1, page_idx)
        target_page = block_table.gather(1, page_idx)
        return local_block_table, local_page, target_page, page_pos

    def _stage_compact_update_inputs(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.device is not None

        bsz, blocks_per_seq = block_table.shape
        local_pages = bsz * blocks_per_seq
        local_block_table, local_page, _, page_pos = self._build_compact_update_index_map(
            cache_seqlens,
            block_table,
            length,
        )

        staged_k = torch.empty(
            (local_pages, PAGE_SIZE, self.attention.num_kv_heads, self.attention.head_dim),
            dtype = torch.half,
            device = self.device,
        )
        staged_v = torch.empty_like(staged_k)

        flat_local_page = local_page.reshape(-1).to(dtype = torch.long)
        flat_page_pos = page_pos.reshape(-1).to(dtype = torch.long)
        staged_k[flat_local_page, flat_page_pos] = k.reshape(-1, self.attention.num_kv_heads, self.attention.head_dim)
        staged_v[flat_local_page, flat_page_pos] = v.reshape(-1, self.attention.num_kv_heads, self.attention.head_dim)

        return local_block_table, staged_k, staged_v

    def _scatter_compact_quantized_updates(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        local_block_table: torch.Tensor,
        local_qk: torch.Tensor,
        local_qv: torch.Tensor,
        local_sk: torch.Tensor,
        local_sv: torch.Tensor,
        length: int,
    ) -> None:
        _, local_page, target_page, page_pos = self._build_compact_update_index_map(
            cache_seqlens,
            block_table,
            length,
            local_block_table = local_block_table,
        )

        flat_local_page = local_page.reshape(-1).to(dtype = torch.long)
        flat_target_page = target_page.reshape(-1).to(dtype = torch.long)
        flat_page_pos = page_pos.reshape(-1).to(dtype = torch.long)

        self.qk[flat_target_page, flat_page_pos] = local_qk[flat_local_page, flat_page_pos]
        self.qv[flat_target_page, flat_page_pos] = local_qv[flat_local_page, flat_page_pos]
        self.sk[flat_target_page, flat_page_pos] = local_sk[flat_local_page, flat_page_pos]
        self.sv[flat_target_page, flat_page_pos] = local_sv[flat_local_page, flat_page_pos]

    def get_kv_compact(
        self,
        total_lens: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k_delta: torch.Tensor | None = None,
        v_delta: torch.Tensor | None = None,
        delta_len: int = 0,
        gathered_k: torch.Tensor | None = None,
        gathered_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = block_table.shape[0]
        max_total = int(total_lens.max())
        target_shape = (bsz, max_total, self.attention.num_kv_heads, self.attention.head_dim)
        target_shape_heads = (bsz, self.attention.num_kv_heads, max_total, self.attention.head_dim)
        heads_first = gathered_k is not None and gathered_k.shape == target_shape_heads
        if gathered_k is None:
            gathered_k = torch.empty(target_shape, dtype = torch.half, device = self.device)
        elif gathered_k.shape not in (target_shape, target_shape_heads):
            gathered_k = torch.empty(target_shape, dtype = torch.half, device = self.device)
            heads_first = False
        if gathered_v is None:
            gathered_v = torch.empty(target_shape, dtype = torch.half, device = self.device)
        elif gathered_v.shape not in (target_shape, target_shape_heads):
            gathered_v = torch.empty(target_shape, dtype = torch.half, device = self.device)
            heads_first = False
        if (
            k_delta is not None and
            v_delta is not None and
            delta_len > 0
        ):
            if heads_first and hasattr(ext, "dequant_cache_paged_gather_delta_heads"):
                ext.dequant_cache_paged_gather_delta_heads(
                    self.qk, self.sk, k_delta.contiguous(), gathered_k,
                    self.qv, self.sv, v_delta.contiguous(), gathered_v,
                    cache_seqlens, block_table,
                    PAGE_SIZE,
                    max_total,
                    delta_len,
                )
                return gathered_k, gathered_v
            if hasattr(ext, "dequant_cache_paged_gather_delta"):
                ext.dequant_cache_paged_gather_delta(
                    self.qk, self.sk, k_delta.contiguous(), gathered_k,
                    self.qv, self.sv, v_delta.contiguous(), gathered_v,
                    cache_seqlens, block_table,
                    PAGE_SIZE,
                    max_total,
                    delta_len,
                )
                return gathered_k, gathered_v

        if heads_first and hasattr(ext, "dequant_cache_paged_gather_heads"):
            ext.dequant_cache_paged_gather_heads(
                self.qk, self.sk, gathered_k,
                self.qv, self.sv, gathered_v,
                cache_seqlens, block_table,
                PAGE_SIZE,
                max_total,
            )
        elif hasattr(ext, "dequant_cache_paged_gather"):
            ext.dequant_cache_paged_gather(
                self.qk, self.sk, gathered_k,
                self.qv, self.sv, gathered_v,
                cache_seqlens, block_table,
                PAGE_SIZE,
                max_total,
            )
        else:
            cache_k, cache_v = self.get_kv(cache_seqlens, block_table)
            if heads_first:
                temp_k = torch.empty(target_shape, dtype = torch.half, device = self.device)
                temp_v = torch.empty(target_shape, dtype = torch.half, device = self.device)
                for b in range(bsz):
                    total = int(cache_seqlens[b])
                    if total == 0:
                        continue
                    num_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
                    pages = block_table[b, :num_pages].to(dtype = torch.long)
                    flat_k = cache_k.index_select(0, pages).reshape(-1, self.attention.num_kv_heads, self.attention.head_dim)
                    flat_v = cache_v.index_select(0, pages).reshape(-1, self.attention.num_kv_heads, self.attention.head_dim)
                    temp_k[b, :total].copy_(flat_k[:total], non_blocking = True)
                    temp_v[b, :total].copy_(flat_v[:total], non_blocking = True)
                gathered_k.copy_(temp_k.transpose(1, 2), non_blocking = True)
                gathered_v.copy_(temp_v.transpose(1, 2), non_blocking = True)
            else:
                for b in range(bsz):
                    total = int(cache_seqlens[b])
                    if total == 0:
                        continue
                    num_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
                    pages = block_table[b, :num_pages].to(dtype = torch.long)
                    flat_k = cache_k.index_select(0, pages).reshape(-1, self.attention.num_kv_heads, self.attention.head_dim)
                    flat_v = cache_v.index_select(0, pages).reshape(-1, self.attention.num_kv_heads, self.attention.head_dim)
                    gathered_k[b, :total].copy_(flat_k[:total], non_blocking = True)
                    gathered_v[b, :total].copy_(flat_v[:total], non_blocking = True)

        if k_delta is not None and v_delta is not None and delta_len > 0:
            for b in range(bsz):
                start = int(cache_seqlens[b])
                if heads_first:
                    gathered_k[b, :, start : start + delta_len].copy_(k_delta[b].transpose(0, 1), non_blocking = True)
                    gathered_v[b, :, start : start + delta_len].copy_(v_delta[b].transpose(0, 1), non_blocking = True)
                else:
                    gathered_k[b, start : start + delta_len].copy_(k_delta[b], non_blocking = True)
                    gathered_v[b, start : start + delta_len].copy_(v_delta[b], non_blocking = True)
        return gathered_k, gathered_v

    def update_kv_compact(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int,
    ) -> None:
        if k.dim() == 3:
            bsz = k.shape[0]
            k = k.contiguous().view(bsz, length, self.attention.num_kv_heads, self.attention.head_dim)
            v = v.contiguous().view(bsz, length, self.attention.num_kv_heads, self.attention.head_dim)

        if hasattr(ext, "quant_cache_paged_delta"):
            ext.quant_cache_paged_delta(
                k.contiguous(), self.qk, self.sk,
                v.contiguous(), self.qv, self.sv,
                cache_seqlens, block_table,
                PAGE_SIZE,
                length,
            )
            return

        # quant_cache_paged expects inputs addressed in cache/page space, like the generic
        # flash-attn path. The compact Gemma4 path only has the new delta K/V span, so stage
        # it into a temporary local page layout, quantize there, then scatter the updated token
        # slots back into the real cache pages.
        local_block_table, staged_k, staged_v = self._stage_compact_update_inputs(
            cache_seqlens,
            block_table,
            k,
            v,
            length,
        )

        local_pages = staged_k.shape[0]
        local_qk = torch.empty(
            (local_pages, PAGE_SIZE, self.token_dim // 32 * self.k_bits),
            dtype = torch.int,
            device = self.device,
        )
        local_qv = torch.empty(
            (local_pages, PAGE_SIZE, self.token_dim // 32 * self.v_bits),
            dtype = torch.int,
            device = self.device,
        )
        local_sk = torch.empty(
            (local_pages, PAGE_SIZE, self.token_dim // 32),
            dtype = torch.half,
            device = self.device,
        )
        local_sv = torch.empty_like(local_sk)

        ext.quant_cache_paged(
            staged_k, local_qk, local_sk,
            staged_v, local_qv, local_sv,
            cache_seqlens, local_block_table,
            PAGE_SIZE,
            length,
        )

        self._scatter_compact_quantized_updates(
            cache_seqlens,
            block_table,
            local_block_table,
            local_qk,
            local_qv,
            local_sk,
            local_sv,
            length,
        )


class Gemma4FullQuantCacheLayer(Gemma4SingleQuantCacheLayer):
    cache_role = "full"


class Gemma4SWAQuantCacheLayer(Gemma4SingleQuantCacheLayer):
    cache_role = "swa"


class Gemma4FullCacheLayer(CacheLayer_fp16):
    cache_role = "full"

    @override
    def tp_export(self, plan):
        return {
            "cls": self.__class__,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
            }
        }


class Gemma4SWACacheLayer(CacheLayer_fp16):
    cache_role = "swa"

    @override
    def tp_export(self, plan):
        return {
            "cls": self.__class__,
            "args": {
                "cache_id": self.cache_id,
                "max_num_tokens": self.max_num_tokens,
            }
        }


def select_gemma4_cache_layer(
    default_layer_type: Type[CacheLayer_fp16 | CacheLayer_quant],
    attention: Attention,
    layer_types: list[str],
    cache_kwargs: dict | None = None,
):
    layer_type = layer_types[attention.layer_idx]
    cache_kwargs = cache_kwargs or {}
    layer_cache_kwargs = dict(cache_kwargs)
    layer_max_num_tokens = cache_kwargs.get("max_num_tokens")
    selected = None

    if layer_type == "full_attention":
        full_max_num_tokens = cache_kwargs.get("full_max_num_tokens")
        layer_max_num_tokens = _normalize_role_max_tokens(full_max_num_tokens, layer_max_num_tokens)
    else:
        swa_max_num_tokens = cache_kwargs.get("swa_max_num_tokens")
        if swa_max_num_tokens is None:
            swa_max_num_tokens = _default_swa_max_tokens(attention, layer_max_num_tokens)
        layer_max_num_tokens = _normalize_role_max_tokens(swa_max_num_tokens, layer_max_num_tokens)

    layer_cache_kwargs.pop("swa_max_num_tokens", None)
    layer_cache_kwargs.pop("full_max_num_tokens", None)
    layer_cache_kwargs.pop("max_num_tokens", None)

    if issubclass(default_layer_type, CacheLayer_quant):
        selected = Gemma4FullQuantCacheLayer if layer_type == "full_attention" else Gemma4SWAQuantCacheLayer

    elif issubclass(default_layer_type, CacheLayer_fp16):
        selected = Gemma4FullCacheLayer if layer_type == "full_attention" else Gemma4SWACacheLayer

    if selected is None:
        return default_layer_type

    if layer_max_num_tokens is None and layer_cache_kwargs == cache_kwargs:
        return selected

    result = {
        "layer_type": selected,
        "cache_kwargs": layer_cache_kwargs,
    }
    if layer_max_num_tokens is not None:
        result["max_num_tokens"] = layer_max_num_tokens
    return result
