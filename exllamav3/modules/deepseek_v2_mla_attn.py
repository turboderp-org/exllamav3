from __future__ import annotations
from collections.abc import Sequence
from typing_extensions import override

import torch
import torch.nn.functional as F
import flashinfer

from ..constants import PAGE_SIZE
from ..model.config import Config
from ..util.rope import RopeSettings, RoPE
from ..util.tensor import get_for_device, to2
from . import Module, Linear, RMSNorm
from .attn import (
    get_flashinfer_workspace,
    make_paged_kv_metadata,
)


class DeepseekV2MLAAttention(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        num_q_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_settings: RopeSettings | None,
        q_lora_rank: int | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
    ):
        super().__init__(config, key, None)

        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = 1
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = kv_lora_rank + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.rope_settings = rope_settings
        self.rope = None
        self.out_dtype = out_dtype
        self.sm_scale = self.qk_head_dim ** -0.5
        qmap_input = qmap + ".input" if qmap else None
        qmap_kva = qmap + ".kva" if qmap else None
        qmap_o = qmap + ".o" if qmap else None

        if self.q_lora_rank is None:
            self.q_proj = Linear(
                config,
                f"{key}.q_proj",
                hidden_size,
                num_q_heads * self.qk_head_dim,
                qmap = qmap_input,
                qbits_mod_key = "q",
            )
            self.q_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
            self.register_submodule(self.q_proj)
        else:
            self.q_a_proj = Linear(
                config,
                f"{key}.q_a_proj",
                hidden_size,
                self.q_lora_rank,
                qmap = qmap_input,
                qbits_mod_key = "q",
            )
            self.q_a_layernorm = RMSNorm(
                config = config,
                key = f"{key}.q_a_layernorm",
                rms_norm_eps = 1e-6,
            )
            self.q_b_proj = Linear(
                config,
                f"{key}.q_b_proj",
                self.q_lora_rank,
                num_q_heads * self.qk_head_dim,
                qmap = qmap_input,
                qbits_mod_key = "q",
            )
            self.q_proj = self.q_b_proj
            self.register_submodule(self.q_a_proj)
            self.register_submodule(self.q_a_layernorm)
            self.register_submodule(self.q_b_proj)

        self.kv_a_proj_with_mqa = Linear(
            config,
            f"{key}.kv_a_proj_with_mqa",
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            qmap = qmap_input,
            qbits_mod_key = "k",
        )
        self.kv_a_layernorm = RMSNorm(
            config = config,
            key = f"{key}.kv_a_layernorm",
            rms_norm_eps = 1e-6,
        )
        self.kv_b_proj = Linear(
            config,
            f"{key}.kv_b_proj",
            kv_lora_rank,
            num_q_heads * (qk_nope_head_dim + v_head_dim),
            qmap = qmap_kva,
            qbits_mod_key = "v",
        )
        self.o_proj = Linear(
            config,
            f"{key}.o_proj",
            num_q_heads * v_head_dim,
            hidden_size,
            qmap = qmap_o,
            out_dtype = out_dtype,
            qbits_mod_key = "o",
        )
        self.k_proj = self.kv_a_proj_with_mqa
        self.v_proj = self.kv_b_proj

        self.register_submodule(self.kv_a_proj_with_mqa)
        self.register_submodule(self.kv_a_layernorm)
        self.register_submodule(self.kv_b_proj)
        self.register_submodule(self.o_proj)

        self.caps.update({
            "kv_cache": True
        })

        self.cache_layers = []
        self.tp_cache_lookup = {}
        self.has_split_cache = False

        self.flashinfer_mla_wrappers: dict[str, flashinfer.BatchMLAPagedAttentionWrapper] = {}

        self.mla_w_uk_t = None
        self.mla_w_o_abs = None
        self.mla_o_bias = None
        self.mla_absorption_ready = False


    @override
    def optimizer_targets(self):
        q = self.q_proj.optimizer_targets() if self.q_lora_rank is None else (
            self.q_a_proj.optimizer_targets() + self.q_b_proj.optimizer_targets()
        )
        k = self.kv_a_proj_with_mqa.optimizer_targets()
        v = self.kv_b_proj.optimizer_targets()
        o = self.o_proj.optimizer_targets()
        return [[q, k + v, o]]


    def _build_absorption_matrices(self):
        if self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64:
            raise NotImplementedError(
                f"Current flashinfer MLA cache append supports kv_lora_rank=512 and qk_rope_head_dim=64, "
                f"got {self.kv_lora_rank}/{self.qk_rope_head_dim}"
            )

        kv_weight = self.kv_b_proj.inner.get_weight_tensor().float()
        o_weight = self.o_proj.inner.get_weight_tensor().float()

        kv_head_width = self.qk_nope_head_dim + self.v_head_dim
        kv_width = self.num_q_heads * kv_head_width
        v_width = self.num_q_heads * self.v_head_dim
        assert kv_weight.shape[1] == kv_width, \
            f"Unexpected kv_b_proj shape: {tuple(kv_weight.shape)}"
        assert o_weight.shape[0] == v_width and o_weight.shape[1] == self.hidden_size, \
            f"Unexpected o_proj shape: {tuple(o_weight.shape)}"

        w_kv = kv_weight.view(self.kv_lora_rank, self.num_q_heads, kv_head_width)
        w_uk = w_kv[..., :self.qk_nope_head_dim]
        w_uv = w_kv[..., self.qk_nope_head_dim:]

        w_uk = w_uk.permute(1, 2, 0).contiguous()
        w_uv = w_uv.permute(1, 0, 2).contiguous()
        w_o = o_weight.view(self.num_q_heads, self.v_head_dim, self.hidden_size).contiguous()

        w_o_abs = torch.matmul(w_uv, w_o).contiguous()
        self.mla_w_uk_t = w_uk.half()
        self.mla_w_o_abs = w_o_abs.half()

        bias = self.o_proj.inner.get_bias_tensor()
        self.mla_o_bias = bias.to(torch.half, copy = True) if bias is not None else None


    def _ensure_absorption_ready(self):
        if not self.mla_absorption_ready:
            self._build_absorption_matrices()
            self.mla_absorption_ready = True


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)

        for cl in self.cache_layers:
            cl.alloc(device)

        if self.rope_settings:
            self.rope = RoPE(device, self.rope_settings)
        self.mla_absorption_ready = False


    @override
    def unload(self):
        for cl in self.cache_layers:
            cl.free()

        self.rope = None
        self.flashinfer_mla_wrappers = {}
        self.mla_w_uk_t = None
        self.mla_w_o_abs = None
        self.mla_o_bias = None
        self.mla_absorption_ready = False
        super().unload()


    def _project_q_ckv(self, x: torch.Tensor, params: dict):
        bsz, seqlen, _ = x.shape

        if self.q_lora_rank is None:
            q = self.q_proj.forward(x, params)
        else:
            q_a = self.q_a_proj.forward(x, params)
            q_a = self.q_a_layernorm.forward(q_a, params, out_dtype = torch.half)
            q = self.q_b_proj.forward(q_a, params)

        q = q.view(bsz, seqlen, self.num_q_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim = -1)

        ckv_kpe = self.kv_a_proj_with_mqa.forward(x, params)
        ckv_kpe = ckv_kpe[..., : self.kv_lora_rank + self.qk_rope_head_dim]
        ckv, k_pe = torch.split(ckv_kpe, [self.kv_lora_rank, self.qk_rope_head_dim], dim = -1)
        ckv = self.kv_a_layernorm.forward(ckv, params, out_dtype = torch.half)
        k_pe = k_pe.view(bsz, seqlen, 1, self.qk_rope_head_dim)

        return q_nope, q_pe, ckv, k_pe


    def _apply_rope(
        self,
        q_pe: torch.Tensor,
        k_pe: torch.Tensor,
        params: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope is None:
            return q_pe, k_pe
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        inv_freq = get_for_device(params, "inv_freq", self.device, None)
        q_pe, k_pe = self.rope.apply(
            q_pe,
            k_pe,
            position,
            positions,
            position_ids,
            True,
            inv_freq = inv_freq,
        )
        return q_pe, k_pe


    def _project_query_absorbed(self, q_nope: torch.Tensor) -> torch.Tensor:
        self._ensure_absorption_ready()
        q_h = q_nope.permute(2, 0, 1, 3).reshape(self.num_q_heads, -1, self.qk_nope_head_dim).contiguous()
        q_abs = torch.bmm(q_h, self.mla_w_uk_t)
        q_abs = q_abs.permute(1, 0, 2).reshape(
            q_nope.shape[0],
            q_nope.shape[1],
            self.num_q_heads,
            self.kv_lora_rank,
        )
        return q_abs


    def _project_output_absorbed(self, o_ckv: torch.Tensor) -> torch.Tensor:
        self._ensure_absorption_ready()
        bsz, seqlen, _, _ = o_ckv.shape
        o_h = o_ckv.permute(2, 0, 1, 3).reshape(self.num_q_heads, -1, self.kv_lora_rank).contiguous()
        y_h = torch.bmm(o_h, self.mla_w_o_abs)
        y = y_h.sum(dim = 0).reshape(bsz, seqlen, self.hidden_size).contiguous()
        if self.mla_o_bias is not None:
            y += self.mla_o_bias
        return y


    def get_flashinfer_mla_wrapper(self, backend: str = "auto") -> flashinfer.BatchMLAPagedAttentionWrapper:
        wrapper = self.flashinfer_mla_wrappers.get(backend)
        if wrapper is None:
            workspace = get_flashinfer_workspace(self.device)
            wrapper = flashinfer.BatchMLAPagedAttentionWrapper(
                workspace,
                backend = backend,
            )
            self.flashinfer_mla_wrappers[backend] = wrapper
        return wrapper


    def _get_mla_backend_order(self, requested_backend: str | Sequence[str]) -> list[str]:
        if isinstance(requested_backend, str):
            candidates = [requested_backend]
        else:
            candidates = [b for b in requested_backend if isinstance(b, str)]

        backends: list[str] = []
        for backend in candidates:
            if backend not in backends:
                backends.append(backend)
        return backends


    def _plan_mla_wrapper(
        self,
        wrapper: flashinfer.BatchMLAPagedAttentionWrapper,
        params: dict,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        seqlen: int,
        causal: bool,
    ):
        planned = params.get("flashinfer_mla_planned_wrappers")
        if planned is None:
            planned = {}
            params["flashinfer_mla_planned_wrappers"] = planned

        wrapper_id = id(wrapper)
        plan_marker = (
            kv_indptr.data_ptr(),
            kv_indices.data_ptr(),
            kv_lens.data_ptr(),
            self.num_q_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            q_dtype,
            kv_dtype,
            seqlen,
            causal,
        )
        if planned.get(wrapper_id) == plan_marker:
            return

        wrapper.plan(
            qo_indptr = qo_indptr,
            kv_indptr = kv_indptr,
            kv_indices = kv_indices,
            kv_len_arr = kv_lens,
            num_heads = self.num_q_heads,
            head_dim_ckv = self.kv_lora_rank,
            head_dim_kpe = self.qk_rope_head_dim,
            page_size = PAGE_SIZE,
            causal = causal,
            sm_scale = self.sm_scale,
            q_data_type = q_dtype,
            kv_data_type = kv_dtype,
        )
        planned[wrapper_id] = plan_marker


    def _run_mla_torch_fallback(
        self,
        q_nope_abs: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_lens: torch.Tensor,
        seqlen: int,
        causal: bool,
    ) -> torch.Tensor:
        bsz = q_nope_abs.shape[0]
        o_ckv = torch.empty_like(q_nope_abs)
        q_cat = torch.cat((q_nope_abs, q_pe), dim = -1)

        for b in range(bsz):
            kv_len = int(kv_lens[b].item())
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            pages = kv_indices[start:end].to(dtype = torch.long)

            ckv_seq = ckv_cache.index_select(0, pages).reshape(-1, self.kv_lora_rank)[:kv_len]
            kpe_seq = kpe_cache.index_select(0, pages).reshape(-1, self.qk_rope_head_dim)[:kv_len]

            q_b = q_cat[b].permute(1, 0, 2).unsqueeze(0).contiguous()
            k_b = torch.cat((
                ckv_seq.unsqueeze(1).expand(kv_len, self.num_q_heads, self.kv_lora_rank),
                kpe_seq.unsqueeze(1).expand(kv_len, self.num_q_heads, self.qk_rope_head_dim),
            ), dim = -1).permute(1, 0, 2).unsqueeze(0).contiguous()
            v_b = ckv_seq.unsqueeze(1).expand(kv_len, self.num_q_heads, self.kv_lora_rank)
            v_b = v_b.permute(1, 0, 2).unsqueeze(0).contiguous()

            if causal:
                prefix_len = max(kv_len - seqlen, 0)
                q_idx = torch.arange(seqlen, device = q_b.device, dtype = torch.int32).unsqueeze(1)
                k_idx = torch.arange(kv_len, device = q_b.device, dtype = torch.int32).unsqueeze(0)
                allowed = k_idx <= (prefix_len + q_idx)
                attn_mask = torch.zeros((1, 1, seqlen, kv_len), dtype = q_b.dtype, device = q_b.device)
                attn_mask = attn_mask.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), float("-inf"))
                o_b = F.scaled_dot_product_attention(
                    q_b,
                    k_b,
                    v_b,
                    attn_mask = attn_mask,
                    is_causal = False,
                    enable_gqa = False,
                    scale = self.sm_scale,
                )
            else:
                o_b = F.scaled_dot_product_attention(
                    q_b,
                    k_b,
                    v_b,
                    is_causal = False,
                    enable_gqa = False,
                    scale = self.sm_scale,
                )

            o_ckv[b] = o_b.squeeze(0).permute(1, 0, 2)

        return o_ckv


    def _append_paged_mla_cache(
        self,
        ckv: torch.Tensor,
        k_pe: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        batch_ids: torch.Tensor | None,
        token_positions: torch.Tensor | None,
        kv_indptr: torch.Tensor | None,
        kv_indices: torch.Tensor | None,
        kv_last_page_len: torch.Tensor | None,
    ):
        bsz, q_len, _, _ = k_pe.shape

        assert cache_k.shape[2] == 1 and cache_v.shape[2] == 1, \
            "DeepSeek MLA cache expects num_kv_heads = 1"

        if batch_ids is None:
            batch_ids = (
                torch.arange(bsz, dtype = torch.int32, device = ckv.device)
                .view(-1, 1)
                .expand(-1, q_len)
                .reshape(-1)
                .contiguous()
            )
        if token_positions is None:
            token_offsets = torch.arange(q_len, dtype = torch.int32, device = ckv.device)
            token_positions = (
                cache_seqlens.to(dtype = torch.int32).view(-1, 1)
                + token_offsets.view(1, -1)
            ).reshape(-1).contiguous()

        if kv_indptr is None or kv_indices is None or kv_last_page_len is None:
            kv_lens = cache_seqlens.to(dtype = torch.int32) + q_len
            kv_indptr, kv_indices, kv_last_page_len = make_paged_kv_metadata(block_table, kv_lens)

        ckv_cache = cache_k.squeeze(2)[..., :self.kv_lora_rank]
        kpe_cache = cache_v.squeeze(2)[..., :self.qk_rope_head_dim]

        flashinfer.append_paged_mla_kv_cache(
            append_ckv = ckv.view(-1, self.kv_lora_rank).contiguous(),
            append_kpe = k_pe.view(-1, self.qk_rope_head_dim).contiguous(),
            batch_indices = batch_ids,
            positions = token_positions,
            ckv_cache = ckv_cache,
            kpe_cache = kpe_cache,
            kv_indices = kv_indices,
            kv_indptr = kv_indptr,
            kv_last_page_len = kv_last_page_len,
        )

        return kv_indptr, kv_indices, kv_last_page_len, ckv_cache, kpe_cache


    def decode_flashinfer_nc(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        causal = params.get("causal", True)
        q_nope, q_pe, ckv, k_pe = self._project_q_ckv(x, params)
        q_pe, k_pe = self._apply_rope(q_pe, k_pe, params)

        kv = self.kv_b_proj.forward(ckv, params)
        kv = kv.view(bsz, seqlen, self.num_q_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim = -1)
        k_pe = k_pe.expand(-1, -1, self.num_q_heads, -1)

        q = torch.cat((q_nope, q_pe), dim = -1).transpose(1, 2)
        k = torch.cat((k_nope, k_pe), dim = -1).transpose(1, 2)
        v = v.transpose(1, 2)

        o = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal = causal,
            enable_gqa = False,
            scale = self.sm_scale,
        )
        o = o.transpose(1, 2).reshape(bsz, seqlen, self.num_q_heads * self.v_head_dim)
        o = self.o_proj.forward(o, params)
        return o


    def decode_flashinfer(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        cache = params.get("cache")
        if cache is None:
            return self.decode_flashinfer_nc(x, bsz, seqlen, params)
        block_table = get_for_device(params, "block_table", self.device)
        cache_seqlens = get_for_device(params, "cache_seqlens", self.device)
        assert block_table is not None and cache_seqlens is not None, \
            "flashinfer MLA mode requires block_table and cache_seqlens"
        batch_ids = get_for_device(params, "flashinfer_batch_ids", self.device, None)
        token_positions = get_for_device(params, "flashinfer_token_positions", self.device, None)
        kv_indptr = get_for_device(params, "flashinfer_kv_indptr", self.device, None)
        kv_indices = get_for_device(params, "flashinfer_kv_indices", self.device, None)
        kv_last_page_len = get_for_device(params, "flashinfer_kv_last_page_len", self.device, None)
        qo_indptr = get_for_device(params, "flashinfer_qo_indptr", self.device, None)
        causal = params.get("causal", True)

        q_nope, q_pe, ckv, k_pe = self._project_q_ckv(x, params)
        q_pe, k_pe = self._apply_rope(q_pe, k_pe, params)
        q_nope_abs = self._project_query_absorbed(q_nope)

        if self.has_split_cache:
            cache_k, cache_v = self.tp_cache_lookup[cache].get_kv(cache_seqlens, block_table)
        else:
            cache_k, cache_v = cache.get_layer(self.layer_idx, cache_seqlens, block_table)

        kv_indptr, kv_indices, kv_last_page_len, ckv_cache, kpe_cache = self._append_paged_mla_cache(
            ckv = ckv,
            k_pe = k_pe,
            cache_k = cache_k,
            cache_v = cache_v,
            cache_seqlens = cache_seqlens,
            block_table = block_table,
            batch_ids = batch_ids,
            token_positions = token_positions,
            kv_indptr = kv_indptr,
            kv_indices = kv_indices,
            kv_last_page_len = kv_last_page_len,
        )
        if qo_indptr is None:
            qo_indptr = torch.arange(
                0,
                (bsz + 1) * seqlen,
                seqlen,
                dtype = torch.int32,
                device = self.device,
            )

        q_nope_flat = q_nope_abs.view(-1, self.num_q_heads, self.kv_lora_rank).contiguous()
        q_pe_flat = q_pe.view(-1, self.num_q_heads, self.qk_rope_head_dim).contiguous()
        kv_lens = get_for_device(params, "flashinfer_kv_lens", self.device, None)
        if kv_lens is None:
            kv_lens = cache_seqlens.to(dtype = torch.int32) + seqlen

        requested_backend = params.get("flashinfer_mla_backend", "auto")
        backend_order = self._get_mla_backend_order(requested_backend)

        o_ckv = None
        last_error: Exception | None = None
        for backend in backend_order:
            try:
                wrapper = self.get_flashinfer_mla_wrapper(backend)
                self._plan_mla_wrapper(
                    wrapper = wrapper,
                    params = params,
                    qo_indptr = qo_indptr,
                    kv_indptr = kv_indptr,
                    kv_indices = kv_indices,
                    kv_lens = kv_lens,
                    q_dtype = q_nope_flat.dtype,
                    kv_dtype = ckv_cache.dtype,
                    seqlen = seqlen,
                    causal = causal,
                )
                run_kwargs = {}
                if backend == "cutlass":
                    run_kwargs["kv_len"] = kv_lens
                    run_kwargs["page_table"] = block_table
                o_ckv = wrapper.run(
                    q_nope = q_nope_flat,
                    q_pe = q_pe_flat,
                    ckv_cache = ckv_cache,
                    kpe_cache = kpe_cache,
                    **run_kwargs,
                ).view(bsz, seqlen, self.num_q_heads, self.kv_lora_rank)
                break
            except (RuntimeError, TypeError, ValueError, AssertionError) as exc:
                last_error = exc
                self.flashinfer_mla_wrappers.pop(backend, None)

        if o_ckv is None:
            if params.get("flashinfer_mla_enable_torch_fallback", False):
                o_ckv = self._run_mla_torch_fallback(
                    q_nope_abs = q_nope_abs,
                    q_pe = q_pe,
                    ckv_cache = ckv_cache,
                    kpe_cache = kpe_cache,
                    kv_indptr = kv_indptr,
                    kv_indices = kv_indices,
                    kv_lens = kv_lens,
                    seqlen = seqlen,
                    causal = causal,
                )
            else:
                raise RuntimeError(
                    f"MLA backend execution failed for {self.key}, requested={requested_backend}, "
                    f"candidates={backend_order}"
                ) from last_error

        if self.has_split_cache:
            self.tp_cache_lookup[cache].update_kv(cache_seqlens, block_table, cache_k, cache_v, seqlen)
        else:
            cache.update_layer(self.layer_idx, cache_seqlens, block_table, cache_k, cache_v, seqlen)

        return self._project_output_absorbed(o_ckv)


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        attn_mode = params.get("attn_mode", "flashinfer_nc")
        match attn_mode:
            case "sdpa_nc":
                x = self.decode_flashinfer_nc(x, bsz, seqlen, params)
            case "flash_attn" | "flashinfer":
                x = self.decode_flashinfer(x, bsz, seqlen, params)
            case "flash_attn_nc" | "flashinfer_nc":
                x = self.decode_flashinfer_nc(x, bsz, seqlen, params)
            case _:
                raise ValueError(f"Unknown attn_mode: {attn_mode}")

        return to2(x, out_dtype, self.out_dtype)
