from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

from ..cache.quant import CacheLayer_quant
from ..ext import exllamav3_ext as ext
from ..model.config import Config
from ..model.model_tp_alloc import TPAllocation
from ..util.rope import RoPE
from ..util.tensor import get_for_device
from ..constants import PAGE_SIZE
from ..util.tensor import to2
from . import Attention, GatedMLP, Linear, Module, RMSNorm, TransformerBlock


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim = -1)


def _apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    return (x * cos) + (_rotate_half(x) * sin)


def _apply_multidimensional_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    ndim = 2
    channels_per_dim = 2 * (x.shape[-1] // (2 * ndim))
    if channels_per_dim <= 0:
        raise ValueError(f"Invalid multidimensional RoPE channel count: {x.shape[-1]}")

    split_sizes = [channels_per_dim] * ndim
    remainder = x.shape[-1] - channels_per_dim * ndim
    if remainder > 0:
        split_sizes.append(remainder)

    x_parts = torch.split(x, split_sizes, dim = -1)
    cos_parts = torch.split(cos, channels_per_dim, dim = -1)
    sin_parts = torch.split(sin, channels_per_dim, dim = -1)
    out_parts = [
        _apply_rotary_pos_emb(x_part, cos_part, sin_part)
        for x_part, cos_part, sin_part in zip(x_parts[:ndim], cos_parts, sin_parts)
    ]
    if remainder > 0:
        out_parts.append(x_parts[-1])
    return torch.cat(out_parts, dim = -1)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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
        for b in range(bsz):
            start = int(cache_seqlens[b])
            for t in range(seqlen):
                pos = start + t
                page_idx = int(block_table[b, pos // PAGE_SIZE])
                page_pos = pos % PAGE_SIZE
                sk, sv = self._ensure_shadow_page(page_idx)
                sk[page_pos].copy_(k[b, t], non_blocking = True)
                sv[page_pos].copy_(v[b, t], non_blocking = True)


    def gather_shadow_pages(
        self,
        block_table: torch.Tensor,
        total_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.device is not None
        assert self.shadow_k_pages is not None and self.shadow_v_pages is not None
        bsz = block_table.shape[0]
        max_total = int(total_lens.max())
        gathered_k = torch.zeros(
            (bsz, max_total, self.attention.num_kv_heads, self.attention.head_dim),
            dtype = torch.half,
            device = self.device,
        )
        gathered_v = torch.zeros_like(gathered_k)
        for b in range(bsz):
            total = int(total_lens[b])
            if total == 0:
                continue
            num_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
            cursor = 0
            for local_page in range(num_pages):
                page_idx = int(block_table[b, local_page])
                sk, sv = self._ensure_shadow_page(page_idx)
                count = min(PAGE_SIZE, total - cursor)
                gathered_k[b, cursor : cursor + count].copy_(sk[:count], non_blocking = True)
                gathered_v[b, cursor : cursor + count].copy_(sv[:count], non_blocking = True)
                cursor += count
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


class Gemma4SingleQuantCacheLayer(Gemma4QuantCacheLayer):

    @override
    def get_kv(self, cache_seqlens: torch.Tensor, block_table: torch.Tensor):
        k = torch.empty(self.shape, dtype = torch.half, device = self.device)
        v = torch.empty(self.shape, dtype = torch.half, device = self.device)
        ext.dequant_cache_paged_single(self.qk, self.sk, k, cache_seqlens, block_table, PAGE_SIZE)
        ext.dequant_cache_paged_single(self.qv, self.sv, v, cache_seqlens, block_table, PAGE_SIZE)
        return k, v


    def get_kv_compact(
        self,
        total_lens: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = block_table.shape[0]
        max_total = int(total_lens.max())
        compact_shape = (bsz, max_total, self.token_dim)
        k = torch.empty(compact_shape, dtype = torch.half, device = self.device)
        v = torch.empty_like(k)
        ext.dequant_cache_paged_compact_single(self.qk, self.sk, k, cache_seqlens, block_table, PAGE_SIZE)
        ext.dequant_cache_paged_compact_single(self.qv, self.sv, v, cache_seqlens, block_table, PAGE_SIZE)
        view_shape = (bsz, max_total, self.attention.num_kv_heads, self.attention.head_dim)
        return k.view(view_shape), v.view(view_shape)


    @override
    def update_kv(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int
    ):
        ext.quant_cache_paged_single(k, self.qk, self.sk, cache_seqlens, block_table, PAGE_SIZE, length)
        ext.quant_cache_paged_single(v, self.qv, self.sv, cache_seqlens, block_table, PAGE_SIZE, length)


    def update_kv_compact(
        self,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        length: int,
    ) -> None:
        bsz = k.shape[0]
        k_compact = k.contiguous().view(bsz, length, self.token_dim)
        v_compact = v.contiguous().view(bsz, length, self.token_dim)
        ext.quant_cache_paged_compact_single(
            k_compact,
            self.qk,
            self.sk,
            cache_seqlens,
            block_table,
            PAGE_SIZE,
            length,
        )
        ext.quant_cache_paged_compact_single(
            v_compact,
            self.qv,
            self.sv,
            cache_seqlens,
            block_table,
            PAGE_SIZE,
            length,
        )


class Gemma4VisionRoPE:

    def __init__(
        self,
        device: torch.device,
        head_dim: int,
        rope_theta: float,
    ):
        spatial_dim = head_dim // 2
        self.device = device
        self.inv_freq = 1.0 / (
            rope_theta ** (
                torch.arange(0, spatial_dim, 2, dtype = torch.int64, device = device).float() / spatial_dim
            )
        )


    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        all_cos = []
        all_sin = []
        for i in range(2):
            dim_position_ids = position_ids[:, :, i][:, None, :].float()
            freqs = (inv_freq_expanded @ dim_position_ids).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim = -1)
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())
        cos = torch.cat(all_cos, dim = -1).to(dtype = x.dtype, device = x.device)
        sin = torch.cat(all_sin, dim = -1).to(dtype = x.dtype, device = x.device)
        return cos, sin


class Gemma4Attention(Attention):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        use_k_as_v: bool,
        v_norm: RMSNorm | None,
        force_quantized_fallback: bool = False,
        **kwargs,
    ):
        key_v = kwargs.get("key_v")
        super().__init__(
            config=config,
            key=key,
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            key_v=kwargs["key_k"] if use_k_as_v else key_v,
            **{k: v for k, v in kwargs.items() if k != "key_v"},
        )

        self.use_k_as_v = use_k_as_v
        self.force_quantized_fallback = force_quantized_fallback
        if use_k_as_v:
            self.modules.remove(self.v_proj)
            self.v_proj = None

        self.v_norm = v_norm
        self.register_submodule(self.v_norm)


    def _get_vision_group_ids(self, params: dict) -> torch.Tensor | None:
        group_ids = get_for_device(params, "vision_group_ids", self.device, None)
        if group_ids is None:
            return None
        if group_ids.numel() == 0 or not (group_ids >= 0).any():
            return None
        return group_ids


    def _build_mm_mask(
        self,
        bsz: int,
        seqlen: int,
        total_lens: torch.Tensor,
        cache_seqlens: torch.Tensor | None,
        vision_group_ids: torch.Tensor | None,
        q_dtype: torch.dtype,
        device: torch.device,
        causal: bool,
    ) -> torch.Tensor:
        max_total = int(total_lens.max())
        mask = torch.full(
            (bsz, 1, seqlen, max_total),
            torch.finfo(q_dtype).min,
            dtype = q_dtype,
            device = device,
        )
        for b in range(bsz):
            total = int(total_lens[b])
            if total == 0:
                continue
            past = int(cache_seqlens[b]) if cache_seqlens is not None else 0
            full_groups = None
            if vision_group_ids is not None:
                full_groups = torch.full((total,), -1, dtype = torch.int32, device = device)
                full_groups[past : past + seqlen] = vision_group_ids[b]
            for qi in range(seqlen):
                q_abs = past + qi
                if not causal:
                    start = 0 if self.sliding_window < 0 else max(0, q_abs - self.sliding_window)
                    end = total if self.sliding_window < 0 else min(total, q_abs + self.sliding_window + 1)
                    mask[b, 0, qi, start:end] = 0
                else:
                    end = min(total, q_abs + 1)
                    start = 0 if self.sliding_window < 0 else max(0, q_abs - self.sliding_window)
                    if end > start:
                        mask[b, 0, qi, start:end] = 0
                if full_groups is not None:
                    q_group = int(full_groups[q_abs])
                    if q_group >= 0:
                        same_group = full_groups[:total] == q_group
                        mask[b, 0, qi, same_group] = 0
        return mask


    def _mm_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.sm_scale if self.sm_scale is not None else self.head_dim ** -0.5
        if self.logit_softcapping:
            if self.gqa and k.shape[1] != q.shape[1]:
                repeat = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(repeat, dim = 1)
                v = v.repeat_interleave(repeat, dim = 1)
            scores = torch.matmul(q.float(), k.transpose(-1, -2).float()) * scale
            scores = torch.tanh(scores / self.logit_softcapping) * self.logit_softcapping
            scores = scores + mask.float()
            probs = torch.softmax(scores, dim = -1, dtype = torch.float32)
            return torch.matmul(probs, v.float()).to(q.dtype)

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask = mask,
            is_causal = False,
            enable_gqa = self.gqa,
            scale = scale,
        )


    def optimizer_targets(self):
        q = self.q_proj.optimizer_targets()
        k = self.k_proj.optimizer_targets()
        o = self.o_proj.optimizer_targets()
        return [[q, k, o]]


    def load_local(self, device, **kwargs):

        if self.num_kv_heads == 0:
            return

        for cl in self.cache_layers:
            cl.alloc(device)

        if self.rope_settings:
            self.rope = RoPE(
                device,
                self.rope_settings,
            )

        if self.q_norm and isinstance(self.q_norm, RMSNorm) and not self.q_norm.span_heads:
            self.q_norm_tensor = self.q_norm.weight.data
            self.k_norm_tensor = self.k_norm.weight.data


    def project_qkv(self, x: torch.Tensor, params: dict) -> tuple:
        bsz, q_len, _ = x.shape
        q = self.q_proj.forward(x, params)

        if self.interleaved_gate:
            q, g = torch.chunk(q.view(bsz, q_len, -1, self.head_dim * 2), 2, dim = -1)
            g = g.reshape(bsz, q_len, -1)
        elif self.g_proj:
            g = self.g_proj.forward(x, params)
        else:
            g = None

        k = self.k_proj.forward(x, params)
        v = k if self.v_proj is None else self.v_proj.forward(x, params)

        if self.v_norm is not None:
            v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim)
            v = self.v_norm.forward(v, params, out_dtype = torch.half)
            v = v.view(bsz, q_len, self.num_kv_heads * self.head_dim)

        return q, k, v, g


    def _write_cache_pages(
        self,
        cache_tensor: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        bsz, seqlen, _, _ = values.shape
        for b in range(bsz):
            start = int(cache_seqlens[b])
            for t in range(seqlen):
                pos = start + t
                page = int(block_table[b, pos // PAGE_SIZE])
                page_pos = pos % PAGE_SIZE
                cache_tensor[page, page_pos].copy_(values[b, t], non_blocking = True)


    def _gather_cache_pages(
        self,
        cache_tensor: torch.Tensor,
        block_table: torch.Tensor,
        total_lens: torch.Tensor,
    ) -> torch.Tensor:
        bsz = block_table.shape[0]
        max_total = int(total_lens.max())
        gathered = torch.zeros(
            (bsz, max_total, self.num_kv_heads, self.head_dim),
            dtype = cache_tensor.dtype,
            device = cache_tensor.device,
        )
        for b in range(bsz):
            total = int(total_lens[b])
            if total == 0:
                continue
            num_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
            pages = block_table[b, :num_pages].long()
            flat = cache_tensor.index_select(0, pages).reshape(-1, self.num_kv_heads, self.head_dim)
            gathered[b, :total].copy_(flat[:total], non_blocking = True)
        return gathered


    def decode_flash_attn_fallback(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        cache = params.get("cache")
        block_table = get_for_device(params, "block_table", self.device)
        cache_seqlens = get_for_device(params, "cache_seqlens", self.device)
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        inv_freq = get_for_device(params, "inv_freq", self.device, None)
        causal = params.get("causal", True)
        has_mm_embeddings = bool(params.get("indexed_embeddings"))
        vision_group_ids = self._get_vision_group_ids(params)
        if self.sliding_window < 0:
            vision_group_ids = None

        q, k, v, g = self.project_qkv(x, params)
        q = q.view(bsz, seqlen, self.num_q_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        if self.q_norm:
            if self.tp_span_heads_norm:
                q, k = self.apply_qk_norms_tp(q, k, params)
            elif not self.rope or self.q_norm_tensor is None:
                q = self.q_norm.forward(q, params, out_dtype = torch.half)
                k = self.k_norm.forward(k, params, out_dtype = torch.half)

        if self.rope:
            q, k = self.rope.apply(
                q, k,
                position,
                positions,
                position_ids,
                True,
                self.q_norm_tensor if not self.tp_span_heads_norm else None,
                self.k_norm_tensor if not self.tp_span_heads_norm else None,
                self.norm_eps,
                self.norm_constant_bias,
                inv_freq,
                self.post_rope_norm
            )

        total_lens = cache_seqlens + seqlen
        cache_layer = cache.layers[self.layer_idx]
        use_shadow_cache = isinstance(cache_layer, Gemma4QuantCacheLayer) and (
            self.sliding_window < 0 or has_mm_embeddings
        )
        use_compact_cache = (
            self.force_quantized_fallback and
            isinstance(cache_layer, Gemma4SingleQuantCacheLayer) and
            not use_shadow_cache
        )

        if use_shadow_cache:
            cache_layer.write_shadow_pages(block_table, cache_seqlens, k, v)
            all_k, all_v = cache_layer.gather_shadow_pages(block_table, total_lens)
            all_k = all_k.transpose(1, 2)
            all_v = all_v.transpose(1, 2)
            if isinstance(cache_layer, Gemma4SingleQuantCacheLayer):
                cache_layer.update_kv_compact(cache_seqlens, block_table, k, v, seqlen)
            else:
                cache.update_layer(self.layer_idx, cache_seqlens, block_table, k, v, seqlen)
        elif use_compact_cache:
            compact_k, compact_v = cache_layer.get_kv_compact(total_lens, cache_seqlens, block_table)
            for b in range(bsz):
                start = int(cache_seqlens[b])
                compact_k[b, start : start + seqlen].copy_(k[b], non_blocking = True)
                compact_v[b, start : start + seqlen].copy_(v[b], non_blocking = True)
            all_k = compact_k.transpose(1, 2)
            all_v = compact_v.transpose(1, 2)
            cache_layer.update_kv_compact(cache_seqlens, block_table, k, v, seqlen)
        else:
            cache_k, cache_v = cache.get_layer(self.layer_idx, cache_seqlens, block_table)
            self._write_cache_pages(cache_k, block_table, cache_seqlens, k)
            self._write_cache_pages(cache_v, block_table, cache_seqlens, v)
            all_k = self._gather_cache_pages(cache_k, block_table, total_lens).transpose(1, 2)
            all_v = self._gather_cache_pages(cache_v, block_table, total_lens).transpose(1, 2)

        q = q.transpose(1, 2)
        mask = self._build_mm_mask(
            bsz,
            seqlen,
            total_lens,
            cache_seqlens,
            vision_group_ids,
            q.dtype,
            q.device,
            causal,
        )
        o = self._mm_attention(q, all_k, all_v, mask)

        if not use_shadow_cache and not use_compact_cache and hasattr(cache_layer, "qk"):
            cache.update_layer(self.layer_idx, cache_seqlens, block_table, k, v, seqlen)

        if self.headwise_gate:
            o *= g.sigmoid().unsqueeze(-1)
        o = o.transpose(1, 2).contiguous().reshape((bsz, seqlen, self.num_q_heads * self.head_dim))
        if self.interleaved_gate:
            o *= g.sigmoid()

        return self.project_o(o, bsz, seqlen, params)


    def decode_sdpa_nc(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        causal = params.get("causal", True)
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        inv_freq = get_for_device(params, "inv_freq", self.device, None)
        vision_group_ids = self._get_vision_group_ids(params)
        if self.sliding_window < 0:
            vision_group_ids = None

        q, k, v, g = self.project_qkv(x, params)
        q = q.view(bsz, seqlen, self.num_q_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        if self.q_norm:
            if self.tp_span_heads_norm:
                q, k = self.apply_qk_norms_tp(q, k, params)
            elif not self.rope or self.q_norm_tensor is None:
                q = self.q_norm.forward(q, params, out_dtype = torch.half)
                k = self.k_norm.forward(k, params, out_dtype = torch.half)

        if self.rope:
            q, k = self.rope.apply(
                q, k,
                position,
                positions,
                position_ids,
                True,
                self.q_norm_tensor if not self.tp_span_heads_norm else None,
                self.k_norm_tensor if not self.tp_span_heads_norm else None,
                self.norm_eps,
                self.norm_constant_bias,
                inv_freq,
                self.post_rope_norm
            )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if vision_group_ids is not None:
            total_lens = torch.full((bsz,), seqlen, dtype = torch.int32, device = q.device)
            mask = self._build_mm_mask(
                bsz,
                seqlen,
                total_lens,
                None,
                vision_group_ids,
                q.dtype,
                q.device,
                causal,
            )
            o = self._mm_attention(q, k, v, mask)
        else:
            o = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal = causal,
                enable_gqa = self.gqa,
                scale = self.sm_scale,
            )

        if self.headwise_gate:
            o *= g.sigmoid().unsqueeze(-1)
        o = o.transpose(1, 2).contiguous().reshape((bsz, seqlen, self.num_q_heads * self.head_dim))
        if self.interleaved_gate:
            o *= g.sigmoid()

        return self.project_o(o, bsz, seqlen, params)


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        if self.num_kv_heads == 0:
            x = torch.zeros_like(x, dtype = self.out_dtype)
            return to2(x, out_dtype, self.out_dtype)

        bsz, seqlen, _ = x.shape
        attn_mode = params.get("attn_mode", "flash_attn_nc")
        has_mm_embeddings = bool(params.get("indexed_embeddings"))
        vision_group_ids = self._get_vision_group_ids(params)
        cache = params.get("cache")
        if self.sliding_window < 0:
            vision_group_ids = None
        force_quantized_fallback = (
            self.force_quantized_fallback and
            attn_mode == "flash_attn" and
            cache is not None and
            isinstance(cache.layers[self.layer_idx], Gemma4QuantCacheLayer)
        )

        if self.head_dim > 256 or vision_group_ids is not None or has_mm_embeddings or force_quantized_fallback:
            match attn_mode:
                case "flash_attn_nc":
                    x = self.decode_sdpa_nc(x, bsz, seqlen, params)
                case "flash_attn":
                    x = self.decode_flash_attn_fallback(x, bsz, seqlen, params)
                case "sdpa_nc":
                    x = self.decode_sdpa_nc(x, bsz, seqlen, params)
                case _:
                    raise ValueError(f"Unknown attn_mode: {attn_mode}")
            return to2(x, out_dtype, self.out_dtype)

        return super().forward(x, params, out_dtype)


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        storage = 0
        storage += self.q_proj.storage_size()
        storage += self.k_proj.storage_size()
        storage += self.o_proj.storage_size()
        for cl in self.cache_layers:
            storage += cl.storage_size()
        overhead_d = 0
        overhead_d += self.hidden_size * (self.out_dtype or torch.half).itemsize
        overhead_s = 0
        for cl in self.cache_layers:
            overhead_s += cl.overhead_size()
        overhead_s += 2 * self.num_q_heads * self.head_dim * torch.half.itemsize
        overhead_s += 2 * self.num_kv_heads * self.head_dim * torch.half.itemsize
        recons = max(
            self.q_proj.recons_size(),
            self.k_proj.recons_size(),
            self.o_proj.recons_size(),
        )
        channel_width = 1
        channels_to_split = self.num_kv_heads
        while channel_width * self.head_dim < 128:
            assert channels_to_split % 2 == 0, \
                "Model's K/V heads cannot divide into 128-channel tensors"
            channel_width *= 2
            channels_to_split //= 2
        assert (channel_width * self.head_dim) % 128 == 0, \
            "Model's K/V heads cannot divide into 128-channel tensors"
        return [
            TPAllocation(
                key = self.key,
                channel_width = channel_width,
                channel_unit = "heads",
                storage_per_device = 0,
                storage_to_split = storage,
                overhead_per_device = overhead_d,
                overhead_to_split = overhead_s,
                recons_temp = recons,
                channels_to_split = channels_to_split,
                limit_key = "attn"
            )
        ]


class Gemma4TransformerBlock(TransformerBlock):

    def __init__(
        self,
        config: Config,
        key: str,
        **kwargs,
    ):
        super().__init__(config=config, key=key, **kwargs)
        self.layer_scalar_key = f"{key}.layer_scalar"
        self.layer_scalar = None
        self.layer_scalar_numel = 1


    def optimizer_targets(self):
        return super().optimizer_targets()


    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        layer_scalar = self.config.stc.get_tensor(
            self.layer_scalar_key,
            device,
            allow_bf16 = True,
            no_defer = True,
        )
        self.layer_scalar = nn.Parameter(layer_scalar, requires_grad = False)
        self.layer_scalar_numel = layer_scalar.numel()


    def unload(self):
        super().unload()
        self.layer_scalar = None


    def get_tensors(self):
        if self.layer_scalar is None:
            return {}
        return {
            self.layer_scalar_key: self.layer_scalar.data.contiguous(),
        }


    def weights_numel(self):
        return super().weights_numel() + self.layer_scalar_numel


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        x = super().forward(x, params, out_dtype = None)
        if self.layer_scalar is not None:
            x = x * self.layer_scalar.to(dtype = x.dtype)
        return to2(x, out_dtype, self.out_dtype)


class Gemma4Router(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        rms_norm_eps: float,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scalar_root_size = hidden_size ** -0.5
        self.scale_key = f"{key}.scale"
        self.per_expert_scale_key = f"{key}.per_expert_scale"
        self.scale = None
        self.per_expert_scale = None
        self.extra_numel = 0

        self.norm = RMSNorm(
            config = config,
            key = f"{key}.norm",
            rms_norm_eps = rms_norm_eps,
            out_dtype = torch.float,
            unweighted = True,
        )
        self.proj = Linear(
            config = config,
            key = f"{key}.proj",
            in_features = hidden_size,
            out_features = num_experts,
            qmap = None,
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.register_submodule(self.norm)
        self.register_submodule(self.proj)


    @override
    def optimizer_targets(self):
        return []


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.scale = self.config.stc.get_tensor(
            self.scale_key,
            device,
            allow_bf16 = True,
            no_defer = True,
        )
        self.per_expert_scale = self.config.stc.get_tensor(
            self.per_expert_scale_key,
            device,
            allow_bf16 = True,
            no_defer = True,
        )
        self.extra_numel = self.scale.numel() + self.per_expert_scale.numel()


    @override
    def unload(self):
        super().unload()
        self.scale = None
        self.per_expert_scale = None
        self.extra_numel = 0


    @override
    def get_tensors(self):
        if self.scale is None or self.per_expert_scale is None:
            return {}
        return {
            self.scale_key: self.scale.data.contiguous(),
            self.per_expert_scale_key: self.per_expert_scale.data.contiguous(),
        }


    @override
    def weights_numel(self):
        return super().weights_numel() + self.extra_numel


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.norm.forward_torch(x, params, out_dtype = torch.float)
        y = y * self.scale.to(dtype = y.dtype)
        y *= self.scalar_root_size

        logits = self.proj.forward(y.half(), params, out_dtype = torch.float).float()
        probs = torch.softmax(logits, dim = -1)

        if params.get("activate_all_experts"):
            selected = (
                torch.arange(self.num_experts, dtype = torch.long, device = x.device)
                .repeat((x.shape[0], 1))
            )
            weights = probs * self.per_expert_scale.to(dtype = probs.dtype).unsqueeze(0)
            return selected, weights

        top_k_weights, top_k_index = torch.topk(
            probs,
            k = self.num_experts_per_tok,
            dim = -1,
        )
        top_k_weights /= top_k_weights.sum(dim = -1, keepdim = True)
        top_k_weights *= self.per_expert_scale[top_k_index].to(dtype = top_k_weights.dtype)
        return top_k_index, top_k_weights


class Gemma4Experts(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        qmap: str,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        self.gates = []
        self.ups = []
        self.downs = []

        fkey_gate_up = f"{key}.experts.gate_up_proj"
        fkey_down = f"{key}.experts.down_proj"

        for idx in range(num_experts):
            gate = Linear(
                config = config,
                key = f"{key}.experts.{idx}.gate_proj",
                fkey = fkey_gate_up,
                fidx = idx,
                frange = (0, intermediate_size),
                frange_dim = 1,
                in_features = hidden_size,
                out_features = intermediate_size,
                qmap = qmap + ".input",
                out_dtype = torch.half,
                transposed_load = True,
                transpose_fused_weights = True,
                ftranspose_after_load = False,
                qgroup = key + ".experts.gud",
            )
            up = Linear(
                config = config,
                key = f"{key}.experts.{idx}.up_proj",
                fkey = fkey_gate_up,
                fidx = idx,
                frange = (intermediate_size, intermediate_size * 2),
                frange_dim = 1,
                in_features = hidden_size,
                out_features = intermediate_size,
                qmap = qmap + ".input",
                out_dtype = torch.half,
                transposed_load = True,
                transpose_fused_weights = True,
                ftranspose_after_load = False,
                qgroup = key + ".experts.gud",
            )
            down = Linear(
                config = config,
                key = f"{key}.experts.{idx}.down_proj",
                fkey = fkey_down,
                fidx = idx,
                in_features = intermediate_size,
                out_features = hidden_size,
                qmap = qmap + f".{idx}.down",
                out_dtype = torch.float,
                allow_input_padding = True,
                transposed_load = True,
                transpose_fused_weights = True,
                ftranspose_after_load = False,
                qgroup = key + ".experts.gud",
            )

            self.gates.append(gate)
            self.ups.append(up)
            self.downs.append(down)
            self.register_submodule(gate)
            self.register_submodule(up)
            self.register_submodule(down)


    @override
    def optimizer_targets(self):
        g, u, d = [], [], []
        for m in self.gates:
            g += m.optimizer_targets()
        for m in self.ups:
            u += m.optimizer_targets()
        for m in self.downs:
            d += m.optimizer_targets()
        return [[g + u, d]]


    @override
    def forward(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        y = x.view(-1, self.hidden_size)
        final_hidden_states = torch.zeros_like(y, dtype = torch.float)

        num_tokens, top_k = selected_experts.shape
        flat_experts = selected_experts.reshape(-1)
        flat_weights = routing_weights.reshape(-1).to(dtype = torch.float)
        flat_tokens = torch.arange(num_tokens, device = y.device).repeat_interleave(top_k)

        order = flat_experts.argsort()
        expert_sorted = flat_experts[order]
        token_sorted = flat_tokens[order]
        weight_sorted = flat_weights[order]

        expert_count = torch.bincount(expert_sorted, minlength = self.num_experts)
        expert_ptr = torch.empty(self.num_experts + 1, dtype = torch.long, device = y.device)
        expert_ptr[0] = 0
        expert_ptr[1:] = expert_count.cumsum(0)

        for expert_idx in range(self.num_experts):
            start = int(expert_ptr[expert_idx])
            end = int(expert_ptr[expert_idx + 1])
            if start == end:
                continue

            top_x = token_sorted[start:end]
            current_state = y.index_select(0, top_x)
            gate = self.gates[expert_idx].forward(current_state, params)
            up = self.ups[expert_idx].forward(current_state, params)
            current_hidden_states = F.gelu(gate, approximate = "tanh") * up
            current_hidden_states = self.downs[expert_idx].forward(current_hidden_states, params)
            current_hidden_states *= weight_sorted[start:end].unsqueeze(1).to(dtype = current_hidden_states.dtype)
            final_hidden_states.index_add_(0, top_x, current_hidden_states)

        return to2(final_hidden_states.view(x.shape), out_dtype, torch.float)


class Gemma4MoEFeedForward(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        rms_norm_eps: float,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size

        self.dense_mlp = GatedMLP(
            config = config,
            key = f"{key}.mlp",
            hidden_size = hidden_size,
            intermediate_size = intermediate_size,
            key_up = "up_proj",
            key_gate = "gate_proj",
            key_down = "down_proj",
            qmap = "block.mlp",
            activation_fn = "gelu",
            interm_dtype = torch.half,
            out_dtype = torch.float,
        )
        self.dense_post_norm = RMSNorm(
            config = config,
            key = f"{key}.post_feedforward_layernorm_1",
            rms_norm_eps = rms_norm_eps,
            out_dtype = torch.float,
        )
        self.routed_pre_norm = RMSNorm(
            config = config,
            key = f"{key}.pre_feedforward_layernorm_2",
            rms_norm_eps = rms_norm_eps,
        )
        self.routed_post_norm = RMSNorm(
            config = config,
            key = f"{key}.post_feedforward_layernorm_2",
            rms_norm_eps = rms_norm_eps,
            out_dtype = torch.float,
        )
        self.router = Gemma4Router(
            config = config,
            key = f"{key}.router",
            hidden_size = hidden_size,
            num_experts = num_experts,
            num_experts_per_tok = num_experts_per_tok,
            rms_norm_eps = rms_norm_eps,
        )
        self.experts = Gemma4Experts(
            config = config,
            key = key,
            hidden_size = hidden_size,
            intermediate_size = moe_intermediate_size,
            num_experts = num_experts,
            qmap = "block.mlp",
        )

        self.register_submodule(self.dense_mlp)
        self.register_submodule(self.dense_post_norm)
        self.register_submodule(self.routed_pre_norm)
        self.register_submodule(self.routed_post_norm)
        self.register_submodule(self.router)
        self.register_submodule(self.experts)


    @override
    def optimizer_targets(self):
        return [self.dense_mlp.optimizer_targets(), self.experts.optimizer_targets()]


    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        dense = self.dense_mlp.forward(x, params)
        dense = self.dense_post_norm.forward(dense, params)

        selected_experts, routing_weights = self.router.forward(
            residual.view(-1, self.hidden_size),
            params,
        )
        routed_input = self.routed_pre_norm.forward(residual, params, out_dtype = torch.half)
        routed = self.experts.forward(routed_input, selected_experts, routing_weights, params)
        routed = self.routed_post_norm.forward(routed, params)

        y = dense + routed
        return to2(y, out_dtype, torch.float)


class Gemma4MoETransformerBlock(Gemma4TransformerBlock):

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if self.attn:
            y = self.attn_norm.forward(x, params, out_dtype = torch.half) if self.attn_norm else x.half()
            y = self.attn.forward(y, params)
            if params.get("prefill"):
                return x
            if self.attn_post_norm:
                y = self.attn_post_norm.forward(y, params)
            x = x + y

        if self.mlp:
            residual = x
            y = self.mlp_norm.forward(x, params, out_dtype = torch.half) if self.mlp_norm else x.half()
            y = self.mlp.forward(y, residual, params)
            if self.mlp_post_norm:
                y = self.mlp_post_norm.forward(y, params)
            x = residual + y

        if self.layer_scalar is not None:
            x = x * self.layer_scalar.to(dtype = x.dtype)

        return to2(x, out_dtype, self.out_dtype)


class Gemma4VisionPatchEmbedder(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
        patch_dim: int,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.position_embedding_size = config.vision.position_embedding_size
        self.position_embedding_key = f"{key}.position_embedding_table"
        self.position_embedding_table = None
        self.position_embedding_numel = 0

        self.input_proj = Linear(
            config = config,
            key = f"{key}.input_proj",
            in_features = patch_dim,
            out_features = hidden_size,
            qmap = None,
            out_dtype = torch.half,
            pad_to = 1,
        )
        self.register_submodule(self.input_proj)


    @override
    def optimizer_targets(self):
        return []


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.position_embedding_table = self.config.stc.get_tensor(
            self.position_embedding_key,
            device,
            float2half = True,
            allow_bf16 = True,
            no_defer = True,
        )
        self.position_embedding_numel = self.position_embedding_table.numel()


    @override
    def unload(self):
        super().unload()
        self.position_embedding_table = None
        self.position_embedding_numel = 0


    @override
    def get_tensors(self):
        if self.position_embedding_table is None:
            return {}
        return {
            self.position_embedding_key: self.position_embedding_table.contiguous(),
        }


    @override
    def weights_numel(self):
        return super().weights_numel() + self.position_embedding_numel


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        position_ids = get_for_device(params, "image_position_ids", self.device)
        padding_positions = (position_ids == -1).all(dim = -1)
        clamped_positions = position_ids.clamp(min = 0)
        x = 2.0 * (x - 0.5)
        hidden_states = self.input_proj.forward(x, params, out_dtype = torch.half)

        pos_x = clamped_positions[..., 0].reshape(-1)
        pos_y = clamped_positions[..., 1].reshape(-1)
        table = self.position_embedding_table
        pos_emb = table[0].index_select(0, pos_x) + table[1].index_select(0, pos_y)
        pos_emb = pos_emb.view(position_ids.shape[0], position_ids.shape[1], self.hidden_size).to(hidden_states.dtype)
        pos_emb = torch.where(padding_positions.unsqueeze(-1), torch.zeros_like(pos_emb), pos_emb)
        hidden_states = hidden_states + pos_emb
        return to2(hidden_states, out_dtype, torch.half)


class Gemma4VisionAttention(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        hidden_size: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        super().__init__(config, key, None)
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.gqa = (num_q_heads != num_kv_heads)
        self.rope_theta = rope_theta
        self.rope = None

        self.q_proj = Linear(config, f"{key}.q_proj.linear", hidden_size, num_q_heads * head_dim, qmap = None)
        self.k_proj = Linear(config, f"{key}.k_proj.linear", hidden_size, num_kv_heads * head_dim, qmap = None)
        self.v_proj = Linear(config, f"{key}.v_proj.linear", hidden_size, num_kv_heads * head_dim, qmap = None)
        self.o_proj = Linear(config, f"{key}.o_proj.linear", num_q_heads * head_dim, hidden_size, qmap = None)
        self.q_norm = RMSNorm(config, f"{key}.q_norm", rms_norm_eps = rms_norm_eps)
        self.k_norm = RMSNorm(config, f"{key}.k_norm", rms_norm_eps = rms_norm_eps)
        self.v_norm = RMSNorm(config, f"{key}.v_norm", rms_norm_eps = rms_norm_eps, unweighted = True)

        self.register_submodule(self.q_proj)
        self.register_submodule(self.k_proj)
        self.register_submodule(self.v_proj)
        self.register_submodule(self.o_proj)
        self.register_submodule(self.q_norm)
        self.register_submodule(self.k_norm)
        self.register_submodule(self.v_norm)


    @override
    def optimizer_targets(self):
        q = self.q_proj.optimizer_targets()
        k = self.k_proj.optimizer_targets()
        v = self.v_proj.optimizer_targets()
        o = self.o_proj.optimizer_targets()
        return [[q, k + v, o]]


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.rope = Gemma4VisionRoPE(device, self.head_dim, self.rope_theta)


    @override
    def unload(self):
        super().unload()
        self.rope = None


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        position_ids = get_for_device(params, "image_position_ids", self.device)
        padding_positions = (position_ids == -1).all(dim = -1)
        bsz, seqlen, _ = x.shape

        q = self.q_proj.forward(x, params).view(bsz, seqlen, self.num_q_heads, self.head_dim)
        k = self.k_proj.forward(x, params).view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.v_proj.forward(x, params).view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        q = self.q_norm.forward(q, params, out_dtype = torch.half)
        k = self.k_norm.forward(k, params, out_dtype = torch.half)
        v = self.v_norm.forward(v, params, out_dtype = torch.half)

        cos, sin = self.rope.forward(q, position_ids)
        q = _apply_multidimensional_rope(q, cos, sin).transpose(1, 2)
        k = _apply_multidimensional_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)

        if self.gqa:
            repeat = self.num_q_heads // self.num_kv_heads
            k = _repeat_kv(k, repeat)
            v = _repeat_kv(v, repeat)

        attn_mask = torch.zeros((bsz, 1, 1, seqlen), dtype = q.dtype, device = q.device)
        attn_mask.masked_fill_(padding_positions.unsqueeze(1).unsqueeze(1), torch.finfo(q.dtype).min)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * 1.0
        attn_weights = attn_weights + attn_mask
        attn_weights = torch.softmax(attn_weights, dim = -1, dtype = torch.float32).to(q.dtype)
        y = torch.matmul(attn_weights, v)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.num_q_heads * self.head_dim)
        y = self.o_proj.forward(y, params)
        return to2(y, out_dtype, torch.half)


class Gemma4VisionPooler(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        hidden_size: int,
    ):
        super().__init__(config, key, None)
        self.root_hidden_size = hidden_size ** 0.5


    @override
    def optimizer_targets(self):
        return []


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        position_ids = get_for_device(params, "image_position_ids", self.device)
        output_length = int(params["image_output_length"])
        if output_length > x.shape[1]:
            raise ValueError(f"Cannot pool {x.shape[1]} patches to {output_length} soft tokens.")

        padding_positions = (position_ids == -1).all(dim = -1)
        x = x.masked_fill(padding_positions.unsqueeze(-1), 0.0)

        if x.shape[1] != output_length:
            input_seq_len = x.shape[1]
            k = int((input_seq_len // output_length) ** 0.5)
            k_squared = k ** 2
            if k_squared * output_length != input_seq_len:
                raise ValueError(f"Cannot pool {x.shape} to {output_length}: {k=}^2 mismatch")
            clamped_positions = position_ids.clamp(min = 0)
            max_x = clamped_positions[..., 0].max(dim = -1, keepdim = True)[0] + 1
            kernel_idxs = torch.div(clamped_positions, k, rounding_mode = "floor")
            kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
            weights = F.one_hot(kernel_idxs.long(), output_length).float() / k_squared
            x = weights.transpose(1, 2) @ x.float()
            params["image_pooler_mask"] = torch.logical_not((weights == 0).all(dim = 1))
        else:
            x = x.float()
            params["image_pooler_mask"] = ~padding_positions

        x = x * self.root_hidden_size
        return to2(x, out_dtype, torch.float)


class Gemma4VisionStandardize(Module):

    def __init__(
        self,
        config: Config,
        key: str,
    ):
        super().__init__(config, key, None)
        self.bias_key = f"{key}.std_bias"
        self.scale_key = f"{key}.std_scale"
        self.std_bias = None
        self.std_scale = None
        self.extra_numel = 0


    @override
    def optimizer_targets(self):
        return []


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.std_bias = self.config.stc.get_tensor(self.bias_key, device, float2half = True, allow_bf16 = True, no_defer = True)
        self.std_scale = self.config.stc.get_tensor(self.scale_key, device, float2half = True, allow_bf16 = True, no_defer = True)
        self.extra_numel = self.std_bias.numel() + self.std_scale.numel()


    @override
    def unload(self):
        super().unload()
        self.std_bias = None
        self.std_scale = None
        self.extra_numel = 0


    @override
    def get_tensors(self):
        if self.std_bias is None or self.std_scale is None:
            return {}
        return {
            self.bias_key: self.std_bias.contiguous(),
            self.scale_key: self.std_scale.contiguous(),
        }


    @override
    def weights_numel(self):
        return self.extra_numel


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        y = (x.float() - self.std_bias.float()) * self.std_scale.float()
        return to2(y, out_dtype, torch.float)


class Gemma4VisionProjector(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        in_features: int,
        out_features: int,
        rms_norm_eps: float,
    ):
        super().__init__(config, key, None)
        self.in_features = in_features
        self.out_features = out_features
        self.rms_norm_eps = rms_norm_eps
        self.weight = None
        self._numel = 0


    @override
    def optimizer_targets(self):
        return []


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.weight = self.config.stc.get_tensor(
            f"{self.key}.weight",
            device,
            transpose = True,
            allow_bf16 = True,
            no_defer = True,
        )
        self._numel = self.weight.numel()


    @override
    def unload(self):
        super().unload()
        self.weight = None
        self._numel = 0


    @override
    def get_tensors(self):
        if self.weight is None:
            return {}
        return {
            f"{self.key}.weight": self.weight.T.contiguous(),
        }


    @override
    def weights_numel(self):
        return self._numel


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        y = torch.matmul(x.float(), self.weight.float())
        y = y * torch.rsqrt(y.pow(2).mean(dim = -1, keepdim = True) + self.rms_norm_eps)
        return to2(y, out_dtype, torch.float)
