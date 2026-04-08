from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

from ..cache.gemma4 import Gemma4QuantCacheLayer, Gemma4SingleQuantCacheLayer
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
        super_key_v = kwargs.get("key_k") if use_k_as_v and kwargs.get("key_k") is not None else key_v
        if use_k_as_v and kwargs.get("key_k") is None and kwargs.get("v_proj") is None:
            kwargs["v_proj"] = kwargs.get("k_proj")
        super().__init__(
            config=config,
            key=key,
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            key_v=super_key_v,
            **{k: v for k, v in kwargs.items() if k != "key_v"},
        )

        self.use_k_as_v = use_k_as_v
        self.force_quantized_fallback = force_quantized_fallback
        self.disable_exl3_mgemm = (
            config is not None and
            getattr(config, "num_hidden_layers", None) == 30 and
            bool(getattr(config, "enable_moe_block", False)) and
            getattr(config, "num_kv_heads", None) == 8 and
            getattr(config, "num_global_kv_heads", None) == 2
        )
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

    @override
    def get_block_table(self, params: dict) -> torch.Tensor:
        key = "block_table_full" if self.sliding_window < 0 else "block_table_swa"
        block_table = get_for_device(params, key, self.device, None)
        if block_table is None:
            block_table = get_for_device(params, "block_table", self.device)
        return block_table

    @override
    def get_cache_seqlens(self, params: dict) -> torch.Tensor:
        key = "cache_seqlens_full" if self.sliding_window < 0 else "cache_seqlens_swa"
        cache_seqlens = get_for_device(params, key, self.device, None)
        if cache_seqlens is None:
            cache_seqlens = get_for_device(params, "cache_seqlens", self.device)
        return cache_seqlens

    @override
    def project_qkv(self, x: torch.Tensor, params: dict) -> tuple:
        if not self.disable_exl3_mgemm:
            return super().project_qkv(x, params)

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
        return q, k, v, g


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
        mask = torch.zeros((bsz, 1, seqlen, max_total), dtype = torch.bool, device = device)
        if max_total == 0:
            return mask

        total_lens = total_lens.to(device = device, dtype = torch.long)
        past = (
            cache_seqlens.to(device = device, dtype = torch.long)
            if cache_seqlens is not None else
            torch.zeros((bsz,), dtype = torch.long, device = device)
        )

        q_offsets = torch.arange(seqlen, device = device, dtype = torch.long).view(1, seqlen, 1)
        q_abs = past.view(bsz, 1, 1) + q_offsets
        kv_idx = torch.arange(max_total, device = device, dtype = torch.long).view(1, 1, max_total)
        valid_kv = kv_idx < total_lens.view(bsz, 1, 1)

        if self.sliding_window < 0:
            visible = valid_kv.expand(-1, seqlen, -1) if not causal else (kv_idx <= q_abs) & valid_kv
        else:
            start = torch.clamp_min(q_abs - self.sliding_window, 0)
            if causal:
                visible = (kv_idx >= start) & (kv_idx <= q_abs) & valid_kv
            else:
                end = q_abs + self.sliding_window
                visible = (kv_idx >= start) & (kv_idx <= end) & valid_kv

        if vision_group_ids is not None:
            vision_group_ids = vision_group_ids.to(device = device, dtype = torch.int32)
            full_groups = torch.full((bsz, max_total), -1, dtype = torch.int32, device = device)
            group_pos = past.view(bsz, 1) + torch.arange(seqlen, device = device, dtype = torch.long).view(1, seqlen)
            row_idx = torch.arange(bsz, device = device, dtype = torch.long).view(bsz, 1).expand(-1, seqlen)
            valid_group_pos = group_pos < total_lens.view(bsz, 1)
            full_groups[row_idx[valid_group_pos], group_pos[valid_group_pos]] = vision_group_ids[valid_group_pos]
            q_groups = vision_group_ids.view(bsz, seqlen, 1)
            same_group = (q_groups >= 0) & (full_groups.unsqueeze(1) == q_groups)
            visible = (visible | same_group) & valid_kv

        return visible.unsqueeze(1)

    def _get_mm_mask_cached(
        self,
        params: dict,
        bsz: int,
        seqlen: int,
        total_lens: torch.Tensor,
        cache_seqlens: torch.Tensor | None,
        vision_group_ids: torch.Tensor | None,
        q_dtype: torch.dtype,
        device: torch.device,
        causal: bool,
    ) -> torch.Tensor:
        cache = params.setdefault("_gemma4_mm_mask_cache", {})
        cache_key = (
            device.type,
            device.index if device.index is not None else -1,
            self.sliding_window < 0,
            causal,
            bsz,
            seqlen,
            tuple(total_lens.shape),
            int(total_lens.data_ptr()),
            0 if cache_seqlens is None else int(cache_seqlens.data_ptr()),
            0 if vision_group_ids is None else int(vision_group_ids.data_ptr()),
        )
        if cache_key not in cache:
            cache[cache_key] = self._build_mm_mask(
                bsz,
                seqlen,
                total_lens,
                cache_seqlens,
                vision_group_ids,
                q_dtype,
                device,
                causal,
            )
        return cache[cache_key]

    def _get_mm_visible_positions_cached(
        self,
        params: dict,
        mask: torch.Tensor,
        total_lens: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = params.setdefault("_gemma4_mm_visible_kv_cache", {})
        cache_key = (
            device.type,
            device.index if device.index is not None else -1,
            self.sliding_window < 0,
            tuple(mask.shape),
            int(mask.data_ptr()),
            int(total_lens.data_ptr()),
        )
        if cache_key not in cache:
            bsz = mask.shape[0]
            seqlen = mask.shape[2]
            visible_any = mask.squeeze(1).any(dim = 1)
            counts = visible_any.sum(dim = 1, dtype = torch.int32)
            max_selected = int(counts.max().item()) if counts.numel() > 0 else 0
            if max_selected == 0:
                selected_positions = torch.zeros((bsz, 0), dtype = torch.int32, device = device)
                reduced_mask = torch.zeros((bsz, 1, seqlen, 0), dtype = torch.bool, device = device)
                cache[cache_key] = (selected_positions, counts, reduced_mask)
                return cache[cache_key]

            max_total = visible_any.shape[1]
            positions = torch.arange(max_total, dtype = torch.int32, device = device).unsqueeze(0).expand(bsz, -1)
            sentinel = torch.full_like(positions, max_total)
            packed_positions = torch.where(visible_any, positions, sentinel)
            packed_positions = torch.sort(packed_positions, dim = 1, stable = True).values[:, :max_selected]

            valid_selected = (
                torch.arange(max_selected, device = device, dtype = torch.int32).unsqueeze(0) <
                counts.unsqueeze(1)
            )
            gather_positions = packed_positions.clamp_max(max_total - 1).to(dtype = torch.long)
            selected_positions = torch.where(
                valid_selected,
                packed_positions,
                torch.zeros_like(packed_positions),
            )
            reduced_mask = torch.take_along_dim(
                mask.squeeze(1),
                gather_positions.unsqueeze(1).expand(-1, seqlen, -1),
                dim = 2,
            ) & valid_selected.unsqueeze(1)
            reduced_mask = reduced_mask.unsqueeze(1)
            cache[cache_key] = (selected_positions, counts, reduced_mask)
        return cache[cache_key]

    def _validate_local_cache_span(
        self,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor | None,
        seqlen: int,
    ) -> None:
        if cache_seqlens is None:
            return
        capacity = block_table.shape[1] * PAGE_SIZE
        required = cache_seqlens + seqlen
        if not (required <= capacity).all():
            actual = int(required.max())
            raise ValueError(
                f"Gemma4 local cache view overflow in layer {self.layer_idx}: "
                f"need {actual} tokens, but local block table only covers {capacity}. "
                "Increase swa_cache_size or reduce the effective local write span."
            )


    def _get_cache_layer(self, cache):
        if self.has_split_cache:
            return self.tp_cache_lookup[cache]
        return cache.layers[self.layer_idx]


    def _mm_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.sm_scale if self.sm_scale is not None else self.head_dim ** -0.5
        if self.logit_softcapping:
            qf = q.float()
            kf = k.float()
            vf = v.float()
            if self.gqa and k.shape[1] != q.shape[1]:
                repeat = q.shape[1] // k.shape[1]
                qf = qf.reshape(q.shape[0], k.shape[1], repeat, q.shape[2], q.shape[3])
                scores = torch.matmul(qf, kf.unsqueeze(2).transpose(-1, -2)) * scale
                scores = torch.tanh(scores / self.logit_softcapping) * self.logit_softcapping
                scores.masked_fill_(mask.unsqueeze(2).logical_not(), torch.finfo(scores.dtype).min)
                probs = torch.softmax(scores, dim = -1, dtype = torch.float32)
                out = torch.matmul(probs, vf.unsqueeze(2))
                return out.reshape(q.shape[0], q.shape[1], q.shape[2], q.shape[3]).to(q.dtype)

            scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
            scores = torch.tanh(scores / self.logit_softcapping) * self.logit_softcapping
            scores.masked_fill_(mask.logical_not(), torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim = -1, dtype = torch.float32)
            return torch.matmul(probs, vf).to(q.dtype)

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
        gathered: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz = block_table.shape[0]
        max_total = int(total_lens.max())
        target_shape = (bsz, max_total, self.num_kv_heads, self.head_dim)
        target_shape_heads = (bsz, self.num_kv_heads, max_total, self.head_dim)
        heads_first = gathered is not None and gathered.shape == target_shape_heads
        if gathered is None:
            gathered = torch.empty(target_shape, dtype = cache_tensor.dtype, device = cache_tensor.device)
        elif gathered.shape not in (target_shape, target_shape_heads):
            gathered = torch.empty(target_shape, dtype = cache_tensor.dtype, device = cache_tensor.device)
            heads_first = False
        for b in range(bsz):
            total = int(total_lens[b])
            if total == 0:
                continue
            num_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
            pages = block_table[b, :num_pages].long()
            flat = cache_tensor.index_select(0, pages).reshape(-1, self.num_kv_heads, self.head_dim)
            if heads_first:
                gathered[b, :, :total].copy_(flat[:total].transpose(0, 1), non_blocking = True)
            else:
                gathered[b, :total].copy_(flat[:total], non_blocking = True)
        return gathered

    def _gather_selected_cache_pages(
        self,
        cache_tensor: torch.Tensor,
        block_table: torch.Tensor,
        selected_positions: torch.Tensor,
        selected_counts: torch.Tensor,
        gathered: torch.Tensor,
    ) -> torch.Tensor:
        bsz = block_table.shape[0]
        target_shape = (bsz, selected_positions.shape[1], self.num_kv_heads, self.head_dim)
        target_shape_heads = (bsz, self.num_kv_heads, selected_positions.shape[1], self.head_dim)
        heads_first = gathered.shape == target_shape_heads
        assert gathered.shape in (target_shape, target_shape_heads)
        for b in range(bsz):
            total = int(selected_counts[b].item())
            if total == 0:
                continue
            idx = selected_positions[b, :total].to(dtype = torch.long)
            pages = block_table[b].gather(0, torch.div(idx, PAGE_SIZE, rounding_mode = "floor"))
            page_pos = idx.remainder(PAGE_SIZE)
            flat = cache_tensor[pages, page_pos]
            if heads_first:
                gathered[b, :, :total].copy_(flat.transpose(0, 1), non_blocking = True)
            else:
                gathered[b, :total].copy_(flat, non_blocking = True)
        return gathered

    def _get_kv_workspace(
        self,
        params: dict,
        bsz: int,
        max_total: int,
        dtype: torch.dtype,
        device: torch.device,
        heads_first: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = params.setdefault("_gemma4_kv_workspace", {})
        key = (
            device.type,
            device.index if device.index is not None else -1,
            bsz,
            max_total,
            self.num_kv_heads,
            self.head_dim,
            dtype,
            heads_first,
        )
        pair = cache.get(key)
        target_shape = (
            (bsz, self.num_kv_heads, max_total, self.head_dim)
            if heads_first else
            (bsz, max_total, self.num_kv_heads, self.head_dim)
        )
        if pair is None or pair[0].shape != target_shape:
            pair = (
                torch.empty(target_shape, dtype = dtype, device = device),
                torch.empty(target_shape, dtype = dtype, device = device),
            )
            cache[key] = pair
        return pair

    def _get_kv_flat_workspace(
        self,
        params: dict,
        max_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache = params.setdefault("_gemma4_kv_flat_workspace", {})
        token_dim = self.num_kv_heads * self.head_dim
        key = (
            device.type,
            device.index if device.index is not None else -1,
            max_tokens,
            token_dim,
            dtype,
        )
        pair = cache.get(key)
        target_shape = (max_tokens, token_dim)
        if pair is None or pair[0].shape != target_shape:
            pair = (
                torch.empty(target_shape, dtype = dtype, device = device),
                torch.empty(target_shape, dtype = dtype, device = device),
            )
            cache[key] = pair
        return pair

    def _gather_selected_compact_kv(
        self,
        cache_layer: Gemma4SingleQuantCacheLayer,
        params: dict,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        selected_positions: torch.Tensor,
        selected_counts: torch.Tensor,
        k_delta: torch.Tensor,
        v_delta: torch.Tensor,
        gathered_k: torch.Tensor,
        gathered_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = block_table.shape[0]
        max_selected = selected_positions.shape[1]
        if hasattr(ext, "dequant_cache_paged_select_delta_heads"):
            ext.dequant_cache_paged_select_delta_heads(
                cache_layer.qk, cache_layer.sk, k_delta.contiguous(), gathered_k,
                cache_layer.qv, cache_layer.sv, v_delta.contiguous(), gathered_v,
                cache_seqlens, block_table,
                selected_positions.contiguous(), selected_counts.contiguous(),
                PAGE_SIZE,
                max_selected,
                k_delta.shape[1],
            )
            return gathered_k, gathered_v
        flat_k, flat_v = self._get_kv_flat_workspace(params, max_selected, torch.half, gathered_k.device)
        cache_k_tmp, cache_v_tmp = self._get_kv_flat_workspace(
            params,
            max_selected,
            torch.half,
            gathered_k.device,
        )

        for b in range(bsz):
            total = int(selected_counts[b].item())
            if total == 0:
                continue

            idx = selected_positions[b, :total]
            cache_len = int(cache_seqlens[b].item())
            cache_mask = idx < cache_len
            delta_mask = ~cache_mask
            flat_k_b = flat_k[:total]
            flat_v_b = flat_v[:total]

            if cache_mask.any():
                cache_idx = idx[cache_mask]
                pages = block_table[b].gather(0, torch.div(cache_idx, PAGE_SIZE, rounding_mode = "floor"))
                page_pos = cache_idx.remainder(PAGE_SIZE)
                qk_tokens = cache_layer.qk[pages, page_pos]
                qv_tokens = cache_layer.qv[pages, page_pos]
                sk_tokens = cache_layer.sk[pages, page_pos]
                sv_tokens = cache_layer.sv[pages, page_pos]
                cache_total = int(cache_idx.numel())
                ext.dequant_cache_cont(qk_tokens, sk_tokens, cache_k_tmp[:cache_total])
                ext.dequant_cache_cont(qv_tokens, sv_tokens, cache_v_tmp[:cache_total])
                flat_k_b[cache_mask] = cache_k_tmp[:cache_total]
                flat_v_b[cache_mask] = cache_v_tmp[:cache_total]

            if delta_mask.any():
                delta_idx = idx[delta_mask] - cache_len
                flat_k_b[delta_mask] = k_delta[b, delta_idx].reshape(-1, self.num_kv_heads * self.head_dim)
                flat_v_b[delta_mask] = v_delta[b, delta_idx].reshape(-1, self.num_kv_heads * self.head_dim)

            gathered_k[b, :, :total].copy_(
                flat_k_b.view(total, self.num_kv_heads, self.head_dim).transpose(0, 1),
                non_blocking = True,
            )
            gathered_v[b, :, :total].copy_(
                flat_v_b.view(total, self.num_kv_heads, self.head_dim).transpose(0, 1),
                non_blocking = True,
            )

        return gathered_k, gathered_v


    def decode_flash_attn_fallback(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        cache = params.get("cache")
        block_table = self.get_block_table(params)
        cache_seqlens = self.get_cache_seqlens(params)
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
        q = q.transpose(1, 2)
        mask = self._get_mm_mask_cached(
            params,
            bsz,
            seqlen,
            total_lens,
            cache_seqlens,
            vision_group_ids,
            q.dtype,
            q.device,
            causal,
        )
        selected_positions = None
        selected_counts = None
        reduced_mask = None
        if vision_group_ids is not None:
            selected_positions, selected_counts, reduced_mask = self._get_mm_visible_positions_cached(
                params,
                mask,
                total_lens,
                q.device,
            )
        cache_layer = self._get_cache_layer(cache)
        # Full/global Gemma4 layers can use the compact quantized path when backed
        # by the single-quant cache. MM still needs the downstream bsz1 graph fast
        # path disabled; text-only can stay on the normal graph path.
        allow_full_compact_cache = (
            self.sliding_window < 0 and
            isinstance(cache_layer, Gemma4SingleQuantCacheLayer)
        )
        use_shadow_cache = (
            isinstance(cache_layer, Gemma4QuantCacheLayer) and
            self.sliding_window < 0 and
            not allow_full_compact_cache
        )
        use_compact_cache = (
            isinstance(cache_layer, Gemma4SingleQuantCacheLayer) and
            not use_shadow_cache and
            (
                allow_full_compact_cache or
                self.force_quantized_fallback or
                has_mm_embeddings or
                vision_group_ids is not None
            )
        )
        self._validate_local_cache_span(block_table, cache_seqlens, seqlen)
        use_reduced_mm_gather = (
            vision_group_ids is not None and
            selected_positions is not None and
            selected_positions.shape[1] > 0 and
            not torch.equal(selected_counts.to(dtype = total_lens.dtype), total_lens)
        )
        gather_len = int(selected_counts.max().item()) if use_reduced_mm_gather else int(total_lens.max())
        workspace_k, workspace_v = self._get_kv_workspace(
            params,
            bsz,
            gather_len,
            torch.half,
            q.device,
            heads_first = True,
        )

        if use_shadow_cache:
            cache_layer.write_shadow_pages(block_table, cache_seqlens, k, v)
            all_k, all_v = cache_layer.gather_shadow_pages(
                block_table,
                total_lens,
                gathered_k = workspace_k,
                gathered_v = workspace_v,
            )
            attn_mask = mask
            if isinstance(cache_layer, Gemma4SingleQuantCacheLayer):
                cache_layer.update_kv_compact(cache_seqlens, block_table, k, v, seqlen)
            else:
                cache.update_layer(self.layer_idx, cache_seqlens, block_table, k, v, seqlen)
        elif use_compact_cache:
            if use_reduced_mm_gather:
                compact_k, compact_v = self._gather_selected_compact_kv(
                    cache_layer,
                    params,
                    block_table,
                    cache_seqlens,
                    selected_positions,
                    selected_counts,
                    k,
                    v,
                    workspace_k,
                    workspace_v,
                )
                attn_mask = reduced_mask
            else:
                compact_k, compact_v = cache_layer.get_kv_compact(
                    total_lens,
                    cache_seqlens,
                    block_table,
                    k_delta = k,
                    v_delta = v,
                    delta_len = seqlen,
                    gathered_k = workspace_k,
                    gathered_v = workspace_v,
                )
                attn_mask = mask
            all_k = compact_k
            all_v = compact_v
            cache_layer.update_kv_compact(cache_seqlens, block_table, k, v, seqlen)
        else:
            cache_k, cache_v = cache.get_layer(self.layer_idx, cache_seqlens, block_table)
            self._write_cache_pages(cache_k, block_table, cache_seqlens, k)
            self._write_cache_pages(cache_v, block_table, cache_seqlens, v)
            if use_reduced_mm_gather:
                all_k = self._gather_selected_cache_pages(
                    cache_k,
                    block_table,
                    selected_positions,
                    selected_counts,
                    workspace_k,
                )
                all_v = self._gather_selected_cache_pages(
                    cache_v,
                    block_table,
                    selected_positions,
                    selected_counts,
                    workspace_v,
                )
                attn_mask = reduced_mask
            else:
                all_k = self._gather_cache_pages(cache_k, block_table, total_lens, gathered = workspace_k)
                all_v = self._gather_cache_pages(cache_v, block_table, total_lens, gathered = workspace_v)
                attn_mask = mask

        o = self._mm_attention(q, all_k, all_v, attn_mask)

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
            mask = self._get_mm_mask_cached(
                params,
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
        needs_custom_mm_mask = vision_group_ids is not None
        force_quantized_fallback = (
            self.force_quantized_fallback and
            attn_mode == "flash_attn" and
            cache is not None and
            isinstance(self._get_cache_layer(cache), Gemma4QuantCacheLayer)
        )

        if self.head_dim > 256 or needs_custom_mm_mask or force_quantized_fallback:
            match attn_mode:
                case "flash_attn_nc":
                    x = self.decode_sdpa_nc(x, bsz, seqlen, params)
                case "flash_attn":
                    x = self.decode_flash_attn_fallback(x, bsz, seqlen, params)
                case "sdpa_nc":
                    x = self.decode_sdpa_nc(x, bsz, seqlen, params)
                case _:
                    raise ValueError(f"Unknown attn_mode: {attn_mode}")
            if self.tp_reduce:
                params["backend"].all_reduce(x)
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


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        q_norm_span_heads = (
            self.q_norm is not None and
            isinstance(self.q_norm, RMSNorm) and
            self.q_norm.span_heads
        )

        return {
            "cls": Gemma4Attention,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "hidden_size": self.hidden_size,
                "head_dim": self.head_dim,
                "rope_settings": self.rope_settings,
                "sm_scale": self.sm_scale,
                "out_dtype": self.out_dtype,
                "sliding_window": self.sliding_window,
                "logit_softcapping": self.logit_softcapping,
                "post_rope_norm": self.post_rope_norm,
                "tp_split_norm": self.tp_split_norm,
                "use_k_as_v": self.use_k_as_v,
                "force_quantized_fallback": self.force_quantized_fallback,
            },
            "num_kv_heads": self.num_kv_heads,
            **{name: _export(getattr(self, name, None)) for name in (
                "q_norm",
                "k_norm",
                "v_norm",
                "q_proj",
                "k_proj",
                "v_proj",
                "kv_proj",
                "o_proj",
                "g_proj",
            )},
            "device": self.device,
            "cache_layers": [
                cl.tp_export(plan) for cl in self.cache_layers
            ],
            "n_gqa": self.num_q_heads // self.num_kv_heads,
            "q_norm_span_heads": q_norm_span_heads,
            "q_global_dim": self.num_q_heads * self.head_dim if self.q_norm else 0,
            "k_global_dim": self.num_kv_heads * self.head_dim if self.k_norm else 0,
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        key = exported["kwargs"]["key"]
        head_dim = exported["kwargs"]["head_dim"]
        n_gqa = exported["n_gqa"]
        device = local_context["device"]
        tp_split_norm = exported["kwargs"]["tp_split_norm"]
        first, last, unit = plan[key]
        assert unit == "heads"
        num_kv_heads = last - first
        num_q_heads = num_kv_heads * n_gqa

        q_split = (True, first * head_dim * n_gqa, last * head_dim * n_gqa) \
            if num_kv_heads else None
        qh_split = (True, first * n_gqa, last * n_gqa) \
            if num_kv_heads else None
        kv_split = (True, first * head_dim, last * head_dim) \
            if num_kv_heads else None
        o_split = (False, first * head_dim * n_gqa, last * head_dim * n_gqa) \
            if num_kv_heads else None
        q_norm_span_heads = exported.get("q_norm_span_heads", False)
        if q_norm_span_heads:
            norm_q_split = (first * head_dim * n_gqa, last * head_dim * n_gqa) \
                if num_kv_heads else None
            norm_k_split = (first * head_dim, last * head_dim) \
                if num_kv_heads else None
        else:
            norm_q_split = (first * n_gqa, last * n_gqa) \
                if num_kv_heads else None
            norm_k_split = (first, last) \
                if num_kv_heads else None

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        def _import_split(name, split):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import_split(local_context, exported[name], plan, split) \
                if split and exported.get(name) else None

        k_proj = _import_split("k_proj", kv_split)
        v_proj = _import_split("v_proj", kv_split)
        if exported["kwargs"]["use_k_as_v"]:
            v_proj = k_proj

        module = Gemma4Attention(
            config = None,
            **exported["kwargs"],
            num_q_heads = num_q_heads,
            num_kv_heads = num_kv_heads,
            q_norm = _import_split("q_norm", norm_q_split) if tp_split_norm else _import("q_norm"),
            k_norm = _import_split("k_norm", norm_k_split) if tp_split_norm else _import("k_norm"),
            v_norm = _import("v_norm"),
            q_proj = _import_split("q_proj", q_split),
            k_proj = k_proj,
            v_proj = v_proj,
            kv_proj = _import_split("kv_proj", kv_split),
            o_proj = _import_split("o_proj", o_split),
            g_proj = _import_split("g_proj", qh_split),
        )

        if num_kv_heads:
            cache_layers = exported["cache_layers"]
            if len(cache_layers):
                module.has_split_cache = True
                for cl in exported["cache_layers"]:
                    cli = cl["cls"](None, module, **cl["args"])
                    module.cache_layers.append(cli)
                    module.tp_cache_lookup[cl["args"]["cache_id"]] = cli

        module.device = device
        module.q_global_dim = exported.get("q_global_dim", 0)
        module.k_global_dim = exported.get("k_global_dim", 0)
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True
        module.load_local(device)
        return module


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


    def _should_disable_bsz1_graph(self, params: dict) -> bool:
        if self.mlp is None or not isinstance(self.attn, Gemma4Attention):
            return False
        if self.attn.sliding_window >= 0:
            return False
        has_mm_embeddings = bool(params.get("indexed_embeddings"))
        has_vision_groups = self.attn._get_vision_group_ids(params) is not None
        if not (has_mm_embeddings or has_vision_groups):
            return False
        return bool(params.get("_force_disable_bsz1_graph"))


    def _set_bsz1_graph_flag(self, params: dict) -> tuple[bool, bool | None]:
        disable_bsz1_graph = self._should_disable_bsz1_graph(params)
        if not disable_bsz1_graph:
            return False, None
        previous = params.get("_disable_bsz1_graph")
        params["_disable_bsz1_graph"] = True
        return True, previous


    def _restore_bsz1_graph_flag(self, params: dict, had_flag: bool, previous: bool | None) -> None:
        if not had_flag:
            return
        if previous is None:
            params.pop("_disable_bsz1_graph", None)
        else:
            params["_disable_bsz1_graph"] = previous


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


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": Gemma4TransformerBlock,
            "kwargs": {
                "key": self.key,
                "out_dtype": self.out_dtype,
            },
            **{name: _export(getattr(self, name, None)) for name in (
                "attn_norm",
                "attn",
                "attn_post_norm",
                "mlp_norm",
                "mlp",
                "mlp_post_norm",
            )},
            "layer_scalar": producer.send(self.layer_scalar),
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        device = local_context["device"]

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        module = Gemma4TransformerBlock(
            config = None,
            **exported["kwargs"],
            attn_norm = _import("attn_norm"),
            attn = _import("attn"),
            attn_post_norm = _import("attn_post_norm"),
            mlp_norm = _import("mlp_norm"),
            mlp = _import("mlp"),
            mlp_post_norm = _import("mlp_post_norm"),
        )
        module.device = device
        layer_scalar = consumer.recv(exported["layer_scalar"], cuda = True)
        module.layer_scalar = nn.Parameter(layer_scalar, requires_grad = False) if layer_scalar is not None else None
        module.layer_scalar_numel = 0 if layer_scalar is None else layer_scalar.numel()
        torch.cuda.synchronize()
        return module


class Gemma4GatedMLP(GatedMLP):

    def __init__(self, config: Config | None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.disable_exl3_mgemm = (
            config is not None and
            getattr(config, "num_hidden_layers", None) == 30 and
            bool(getattr(config, "enable_moe_block", False)) and
            getattr(config, "num_kv_heads", None) == 8 and
            getattr(config, "num_global_kv_heads", None) == 2
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not self.disable_exl3_mgemm and not params.get("_disable_bsz1_graph"):
            return super().forward(x, params, out_dtype = out_dtype)

        bsz, q_len, _ = x.shape

        if self.num_slices == 0:
            d = torch.zeros_like(x, dtype = self.out_dtype)
            if self.tp_reduce:
                params["backend"].all_reduce(d, False)
            return to2(d, out_dtype, self.out_dtype)

        qs = params.get("q_mlp_slice")
        r = [qs] if qs is not None else range(0, self.num_slices)
        d = None

        for s in r:
            # Keep the 26B Gemma4 path on the conservative per-linear route so
            # the model does not depend on the shared EXL3 mixed-shape mGEMM
            # bugfixes that are being split into a separate global PR.
            if self.disable_exl3_mgemm or self.multi_gu[s] is None or bsz * q_len > 32:
                g = self.gates[s].forward(x, params)
                u = self.ups[s].forward(x, params)
                a = torch.empty_like(u, dtype = torch.half) if self.interm_dtype != torch.half else u
                self.activation_fn_call(g, u, a, self.act_limit)
                d_ = self.downs[s].forward(a, params)
            else:
                xg = x.view(1, bsz * q_len, self.hidden_size)
                guh = torch.empty((2, bsz * q_len, self.hidden_size), dtype = self.interm_dtype, device = x.device)
                gu = torch.empty((2, bsz * q_len, self.multi_gu[s].out_features), dtype = self.interm_dtype, device = x.device)
                ext.exl3_mgemm(
                    xg,
                    self.multi_gu[s].ptrs_trellis,
                    gu,
                    self.multi_gu[s].ptrs_suh,
                    guh,
                    self.multi_gu[s].ptrs_svh,
                    None,
                    None,
                    self.multi_gu[s].K,
                    -1,
                    self.multi_gu[s].mcg,
                    self.multi_gu[s].mul1,
                    -1,
                    -1,
                    0,
                )
                g = gu[0].view(bsz, q_len, self.multi_gu[s].out_features)
                u = gu[1].view(bsz, q_len, self.multi_gu[s].out_features)
                a = torch.empty_like(u, dtype = torch.half) if self.interm_dtype != torch.half else u
                self.activation_fn_call(g, u, a, self.act_limit)
                d_ = self.downs[s].forward(a, params)

            if d is None:
                d = d_
            else:
                d += d_

        if self.tp_reduce:
            params["backend"].all_reduce(d)

        return to2(d, out_dtype, self.out_dtype)


class Gemma4Router(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        rms_norm_eps: float,
        norm: RMSNorm | None = None,
        proj: Linear | None = None,
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

        self.norm = norm or RMSNorm(
            config = config,
            key = f"{key}.norm",
            rms_norm_eps = rms_norm_eps,
            out_dtype = torch.float,
            unweighted = True,
        )
        self.proj = proj or Linear(
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


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": Gemma4Router,
            "kwargs": {
                "key": self.key,
                "hidden_size": self.hidden_size,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "rms_norm_eps": self.norm.rms_norm_eps,
            },
            "norm": _export(self.norm),
            "proj": _export(self.proj),
            "scale": producer.send(self.scale),
            "per_expert_scale": producer.send(self.per_expert_scale),
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        device = local_context["device"]

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        module = Gemma4Router(
            config = None,
            **exported["kwargs"],
            norm = _import("norm"),
            proj = _import("proj"),
        )
        module.device = device
        module.scale = consumer.recv(exported["scale"], cuda = True)
        module.per_expert_scale = consumer.recv(exported["per_expert_scale"], cuda = True)
        module.extra_numel = 0
        if module.scale is not None:
            module.extra_numel += module.scale.numel()
        if module.per_expert_scale is not None:
            module.extra_numel += module.per_expert_scale.numel()
        torch.cuda.synchronize()
        return module


class Gemma4Experts(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        qmap: str,
        num_local_experts: int | None = None,
        gates: list[Linear | Module] | None = None,
        ups: list[Linear | Module] | None = None,
        downs: list[Linear | Module] | None = None,
        routing_first: int | None = None,
        routing_last: int | None = None,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts if num_local_experts is not None else num_experts
        self.routing_first = routing_first
        self.routing_last = routing_last
        self.tp_reduce = False

        self.gates = []
        self.ups = []
        self.downs = []

        if gates is not None:
            assert ups is not None and len(ups) == len(gates)
            assert downs is not None and len(downs) == len(gates)
            self.gates = gates
            self.ups = ups
            self.downs = downs
            for gate, up, down in zip(gates, ups, downs):
                self.register_submodule(gate)
                self.register_submodule(up)
                self.register_submodule(down)
            return

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

        if self.routing_first is None or self.num_local_experts == self.num_experts:
            flat_expert_local = flat_experts
            num_bins = self.num_local_experts
        else:
            sentinel = self.num_local_experts
            flat_expert_local = flat_experts - self.routing_first
            valid = (flat_expert_local >= 0) & (flat_expert_local < self.num_local_experts)
            flat_expert_local = torch.where(
                valid,
                flat_expert_local,
                torch.full_like(flat_expert_local, sentinel),
            )
            num_bins = self.num_local_experts + 1

        order = flat_expert_local.argsort()
        expert_sorted = flat_expert_local[order]
        token_sorted = flat_tokens[order]
        weight_sorted = flat_weights[order]

        expert_count = torch.bincount(expert_sorted, minlength = num_bins)
        expert_ptr = torch.empty(num_bins + 1, dtype = torch.long, device = y.device)
        expert_ptr[0] = 0
        expert_ptr[1:] = expert_count.cumsum(0)

        for expert_idx in range(self.num_local_experts):
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

        final_hidden_states = final_hidden_states.view(x.shape)
        if self.tp_reduce:
            params["backend"].all_reduce(final_hidden_states, self.num_local_experts > 0)
        return to2(final_hidden_states, out_dtype, torch.float)


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
        dense_mlp: GatedMLP | None = None,
        dense_post_norm: RMSNorm | None = None,
        routed_pre_norm: RMSNorm | None = None,
        routed_post_norm: RMSNorm | None = None,
        router: Gemma4Router | None = None,
        experts: Gemma4Experts | None = None,
        routing_device: int | None = None,
    ):
        super().__init__(config, key, None)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routing_device = routing_device

        self.dense_mlp = dense_mlp or Gemma4GatedMLP(
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
        self.dense_post_norm = dense_post_norm or RMSNorm(
            config = config,
            key = f"{key}.post_feedforward_layernorm_1",
            rms_norm_eps = rms_norm_eps,
            out_dtype = torch.float,
        )
        self.routed_pre_norm = routed_pre_norm or RMSNorm(
            config = config,
            key = f"{key}.pre_feedforward_layernorm_2",
            rms_norm_eps = rms_norm_eps,
        )
        self.routed_post_norm = routed_post_norm or RMSNorm(
            config = config,
            key = f"{key}.post_feedforward_layernorm_2",
            rms_norm_eps = rms_norm_eps,
            out_dtype = torch.float,
        )
        self.router = router or Gemma4Router(
            config = config,
            key = f"{key}.router",
            hidden_size = hidden_size,
            num_experts = num_experts,
            num_experts_per_tok = num_experts_per_tok,
            rms_norm_eps = rms_norm_eps,
        )
        self.experts = experts or Gemma4Experts(
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


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        tpa_list = []
        tpa_list += self.dense_mlp.make_tp_allocation(options)

        stc = self.config.stc
        def _norm_storage(module: RMSNorm) -> int:
            if module.unweighted:
                return 0
            return sum(stc.get_tensor_sizes(f"{module.key}.weight"))

        router_storage = (
            _norm_storage(self.dense_post_norm) +
            _norm_storage(self.routed_pre_norm) +
            _norm_storage(self.routed_post_norm) +
            _norm_storage(self.router.norm) +
            self.router.proj.storage_size() +
            sum(stc.get_tensor_sizes(self.router.scale_key)) +
            sum(stc.get_tensor_sizes(self.router.per_expert_scale_key))
        )

        experts_storage = 0
        for g in self.experts.gates:
            experts_storage += g.storage_size()
        for u in self.experts.ups:
            experts_storage += u.storage_size()
        for d in self.experts.downs:
            experts_storage += d.storage_size()
        recons = max(
            self.experts.gates[0].recons_size(),
            self.experts.ups[0].recons_size(),
            self.experts.downs[0].recons_size(),
        )
        tpa_list.append(
            TPAllocation(
                key = self.key,
                channel_width = 1,
                channel_unit = "experts",
                storage_per_device = router_storage,
                storage_to_split = experts_storage,
                overhead_per_device = self.hidden_size * torch.float.itemsize,
                overhead_to_split = 2 * self.moe_intermediate_size * torch.half.itemsize,
                recons_temp = recons,
                channels_to_split = self.num_experts,
                limit_key = "moe",
            )
        )
        return tpa_list


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": Gemma4MoEFeedForward,
            "kwargs": {
                "key": self.key,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "moe_intermediate_size": self.moe_intermediate_size,
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "rms_norm_eps": self.routed_pre_norm.rms_norm_eps,
            },
            "dense_mlp": _export(self.dense_mlp),
            "dense_post_norm": _export(self.dense_post_norm),
            "routed_pre_norm": _export(self.routed_pre_norm),
            "routed_post_norm": _export(self.routed_post_norm),
            "router": _export(self.router),
            "experts": {
                "key": self.experts.key,
                "hidden_size": self.experts.hidden_size,
                "intermediate_size": self.experts.intermediate_size,
                "num_experts": self.experts.num_experts,
                "gates": [_export(self.experts.gates[i]) for i in range(self.experts.num_experts)],
                "ups": [_export(self.experts.ups[i]) for i in range(self.experts.num_experts)],
                "downs": [_export(self.experts.downs[i]) for i in range(self.experts.num_experts)],
            },
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        key = exported["kwargs"]["key"]
        device = local_context["device"]
        output_device = local_context["output_device"]
        first, last, unit = plan[key]
        assert unit == "experts"

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        def _import_i(bucket, i):
            nonlocal exported, plan
            entry = exported["experts"][bucket][i]
            return entry["cls"].tp_import(local_context, entry, plan) if entry is not None else None

        num_local_experts = max(0, last - first)
        experts = Gemma4Experts(
            config = None,
            key = exported["experts"]["key"],
            hidden_size = exported["experts"]["hidden_size"],
            intermediate_size = exported["experts"]["intermediate_size"],
            num_experts = exported["experts"]["num_experts"],
            qmap = "block.mlp",
            num_local_experts = num_local_experts,
            gates = [_import_i("gates", i) for i in range(first, last)],
            ups = [_import_i("ups", i) for i in range(first, last)],
            downs = [_import_i("downs", i) for i in range(first, last)],
            routing_first = first,
            routing_last = last,
        )
        experts.device = device
        experts.tp_reduce = True

        module = Gemma4MoEFeedForward(
            config = None,
            **exported["kwargs"],
            dense_mlp = _import("dense_mlp"),
            dense_post_norm = _import("dense_post_norm"),
            routed_pre_norm = _import("routed_pre_norm"),
            routed_post_norm = _import("routed_post_norm"),
            router = _import("router"),
            experts = experts,
            routing_device = output_device,
        )
        module.device = device
        return module


    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        dense = self.dense_mlp.forward(x, params)
        dense = self.dense_post_norm.forward(dense, params)

        flat_residual = residual.view(-1, self.hidden_size)
        if self.routing_device is not None:
            backend = params["backend"]
            num_tokens = flat_residual.shape[0]
            if backend.device == self.routing_device:
                selected_experts, routing_weights = self.router.forward(flat_residual, params)
            else:
                selected_experts = torch.empty(
                    (num_tokens, self.num_experts_per_tok),
                    dtype = torch.long,
                    device = flat_residual.device,
                )
                routing_weights = torch.empty(
                    (num_tokens, self.num_experts_per_tok),
                    dtype = torch.float,
                    device = flat_residual.device,
                )
            backend.broadcast(selected_experts, src_device = self.routing_device)
            backend.broadcast(routing_weights, src_device = self.routing_device)
        else:
            selected_experts, routing_weights = self.router.forward(flat_residual, params)
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
            had_flag, previous = self._set_bsz1_graph_flag(params)
            try:
                y = self.mlp.forward(y, residual, params)
            finally:
                self._restore_bsz1_graph_flag(params, had_flag, previous)
            if self.mlp_post_norm:
                y = self.mlp_post_norm.forward(y, params)
            x = residual + y

        if self.layer_scalar is not None:
            x = x * self.layer_scalar.to(dtype = x.dtype)

        return to2(x, out_dtype, self.out_dtype)


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": Gemma4MoETransformerBlock,
            "kwargs": {
                "key": self.key,
                "out_dtype": self.out_dtype,
            },
            **{name: _export(getattr(self, name, None)) for name in (
                "attn_norm",
                "attn",
                "attn_post_norm",
                "mlp_norm",
                "mlp",
                "mlp_post_norm",
            )},
            "layer_scalar": producer.send(self.layer_scalar),
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        device = local_context["device"]

        def _import(name):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
                if exported.get(name) else None

        module = Gemma4MoETransformerBlock(
            config = None,
            **exported["kwargs"],
            attn_norm = _import("attn_norm"),
            attn = _import("attn"),
            attn_post_norm = _import("attn_post_norm"),
            mlp_norm = _import("mlp_norm"),
            mlp = _import("mlp"),
            mlp_post_norm = _import("mlp_post_norm"),
        )
        module.device = device
        layer_scalar = consumer.recv(exported["layer_scalar"], cuda = True)
        module.layer_scalar = nn.Parameter(layer_scalar, requires_grad = False) if layer_scalar is not None else None
        module.layer_scalar_numel = 0 if layer_scalar is None else layer_scalar.numel()
        torch.cuda.synchronize()
        return module


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
