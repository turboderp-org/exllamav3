import torch

from ...ext import exllamav3_ext as ext
from ...constants import PAGE_SIZE
from ...util.tensor import g_tensor_cache
from .bc_attn import bc_attn_enable, _trace_build, _compile_kernel, _get_sm_count, _is_pow2, \
    MAX_BSZ, MAX_QLEN

"""
Graph-captured decode attention for MLA (BC_MLAttention): the whole decode block -- q
projections (direct or LoRA), the latent projection, kv_a norm + rope-key/rope-query staging,
partial RoPE, W_UK absorption, cache append and the absorbed flash-decoding kernels, W_UV
unfold and o_proj -- runs as one C++ call, captured as one CUDA graph per (bsz, q_len) slot
after a warmup run and replayed with only the input/output/seqlens/block-table/positions
pointers patched. The Triton kernels are the same ones the dispatch path JITs, compiled ahead
of time with the slot shapes baked as constexprs; the block-table width and split configuration
are runtime kernel arguments patched per call, so context growth never recaptures.

Follows bc_attn.py; shares its enable flag (EXL3_BC_ATTN=0 disables both). Unsupported
module/cache configurations fall back to the dispatch path by design (build_bc_mla returns
None); unexpected failures while building raise.
"""


class BCMLA:
    """Python-side owner of one ext.BC_MLAttention (per attention module and cache layer):
    collects the projection/norm/rope/cache/flat handles at construction and compiles +
    registers the per-slot kernels lazily."""

    def __init__(self, module, layer):
        from ...cache.mla import CacheLayer_MLA_quant

        self.module = module
        # module.device may be a plain string or int; normalize (torch.device is idempotent)
        self.device = torch.device(module.device)
        m = module

        self.num_q_heads = m.num_q_heads
        self.hidden_size = m.hidden_size
        self.kv_lora_rank = m.kv_lora_rank
        self.qk_rope_head_dim = m.qk_rope_head_dim
        self.qk_nope_head_dim = m.qk_nope_head_dim
        self.qk_head_dim = m.qk_head_dim
        self.v_head_dim = m.v_head_dim
        self.q_lora_rank = m.q_lora_rank or 0
        self.sm_scale = m.sm_scale
        self.o_dtype = m.o_proj.inner.default_out_dtype

        # Padded projection widths: the statics carry the projections' actual N, the staging and
        # absorb kernels read at the true offsets (Q_STRIDE/CKV_STRIDE)
        self.w_q = m.q_proj.out_features
        self.w_kv = m.kv_a_proj_with_mqa.out_features

        self.quant = isinstance(layer, CacheLayer_MLA_quant)
        if self.quant:
            self.cache_ckv, self.cache_scales, self.cache_kpe, self.k_bits = layer.get_qc()
            from .triton_paged import _get_h32
            h32 = _get_h32(self.device)
        else:
            self.cache_ckv, self.cache_kpe = layer.k, layer.v
            self.cache_scales, self.k_bits = None, 0
            h32 = g_tensor_cache.get(self.device, (1,), torch.half, "bcm_dummy")
        self.h32 = h32

        # Hadamard scratch for the EXL3 GEMMs, sized for the widest input among them
        w = max(self.hidden_size, self.q_lora_rank, self.num_q_heads * self.v_head_dim)
        xh = g_tensor_cache.get(self.device, (MAX_BSZ * MAX_QLEN * w,), torch.half, "bcm_xh")

        # The staging kernel reads the norm weight as fp16; checkpoints usually store it bf16
        # (fp16-exact at weight magnitudes). The ext object keeps the converted copy alive
        kv_norm_w = m.kv_a_layernorm.weight.data.half().contiguous()

        rope = m.rope
        self.bc = ext.BC_MLAttention(
            num_q_heads = self.num_q_heads,
            hidden_size = self.hidden_size,
            page_size = PAGE_SIZE,
            kv_lora_rank = self.kv_lora_rank,
            qk_rope_head_dim = self.qk_rope_head_dim,
            qk_nope_head_dim = self.qk_nope_head_dim,
            v_head_dim = self.v_head_dim,
            q_lora_rank = self.q_lora_rank,
            q_proj = m.q_proj.inner.bc,
            q_a_proj = m.q_a_proj.inner.bc if m.q_a_proj is not None else None,
            q_a_norm_w = m.q_a_layernorm.weight.data if m.q_a_proj is not None else None,
            kv_a_proj = m.kv_a_proj_with_mqa.inner.bc,
            o_proj = m.o_proj.inner.bc,
            kv_norm_w = kv_norm_w,
            norm_eps = m.norm_eps,
            inv_freq = rope.inv_freq,
            rope_style = int(rope.rope_settings.rope_style),
            attn_factor = rope.attn_factor,
            rotate_dims = rope.rope_settings.rotate_dims,
            w_uk_flat = m.w_uk_flat,
            w_uv_flat = m.w_uv_flat,
            quant_cache = self.quant,
            cache_ckv = self.cache_ckv,
            cache_kpe = self.cache_kpe,
            cache_scales = self.cache_scales,
            xh = xh,
            h32 = h32,
        )
        self.configured = set()

    def _configure(self, bsz: int, q_len: int):
        import triton
        from .mla_triton import (
            _mla_stage_kernel,
            _mla_absorb_kernel,
            _mla_kv_update_kernel,
            _mla_kv_quant_scatter_kernel,
            _mla_decode_split_kernel,
            _mla_decode_combine_kernel,
            _mla_unfold_kernel,
        )

        dev = self.device
        m = self.module
        H = self.num_q_heads
        D_c, D_r = self.kv_lora_rank, self.qk_rope_head_dim
        D_nope, D_v, QK = self.qk_nope_head_dim, self.v_head_dim, self.qk_head_dim
        R = bsz * q_len

        k_stage = _compile_kernel(dev, _mla_stage_kernel,
            {n: "*fp16" for n in ("q_full", "ckv_kpe", "kv_norm_w", "q_pe", "ckv", "kpe")}
            | {n: "constexpr" for n in (
                "eps", "n_q_heads", "QK_DIM", "Q_STRIDE", "CKV_STRIDE", "D_nope", "D_c", "D_r")},
            dict(eps = float(m.norm_eps), n_q_heads = H, QK_DIM = QK, Q_STRIDE = self.w_q,
                 CKV_STRIDE = self.w_kv, D_nope = D_nope, D_c = D_c, D_r = D_r),
            4, 2)

        absorb_bm = min(64, triton.next_power_of_2(max(R, 16)))
        k_absorb = _compile_kernel(dev, _mla_absorb_kernel,
            {"q": "*fp16", "w_uk_flat": "*fp16", "out": "*fp16", "R": "i32"}
            | {n: "constexpr" for n in (
                "n_q_heads", "QK_DIM", "Q_STRIDE", "D_nope", "NOPE_PAD", "D_c", "BLOCK_M", "BLOCK_N")},
            dict(n_q_heads = H, QK_DIM = QK, Q_STRIDE = self.w_q, D_nope = D_nope,
                 NOPE_PAD = triton.next_power_of_2(D_nope), D_c = D_c,
                 BLOCK_M = absorb_bm, BLOCK_N = 128),
            4, 2)

        if self.quant:
            groups = D_c // 32
            w_tot = groups * self.k_bits
            k_append = _compile_kernel(dev, _mla_kv_quant_scatter_kernel,
                {"tmp_q": "*i32", "tmp_s": "*fp16", "kpe_new": "*fp16", "qk": "*i32",
                 "sk": "*fp16", "kpe_cache": "*fp16", "block_table": "*i32",
                 "cache_seqlens": "*i32", "num_pages_per_seq": "i32", "append_len": "i32"}
                | {n: "constexpr" for n in ("page_size", "W_TOT", "W_PAD", "N_G", "D_r")},
                dict(page_size = PAGE_SIZE, W_TOT = w_tot,
                     W_PAD = triton.next_power_of_2(w_tot), N_G = groups, D_r = D_r),
                2, 2)
        else:
            k_append = _compile_kernel(dev, _mla_kv_update_kernel,
                {"ckv_new": "*fp16", "kpe_new": "*fp16", "ckv_cache": "*fp16",
                 "kpe_cache": "*fp16", "block_table": "*i32", "cache_seqlens": "*i32",
                 "num_pages_per_seq": "i32", "append_len": "i32"}
                | {n: "constexpr" for n in ("page_size", "D_c", "D_r")},
                dict(page_size = PAGE_SIZE, D_c = D_c, D_r = D_r),
                4, 2)

        # Same tuning as the dispatch wrapper (mla_attn_triton_decode)
        block_m = triton.next_power_of_2(q_len)
        block_h = max(16 // block_m, 1)
        block_h = min(block_h, max(1, triton.next_power_of_2(H)))
        block_rows = block_m * block_h
        block_n = 64 if self.quant else (32 if D_c <= 512 else 16)
        n_warps = 8 if self.quant else 4
        n_stages = 1 if self.quant else 3
        h_blocks = triton.cdiv(H, block_h)
        programs = bsz * h_blocks

        # The live split count and length are runtime arguments derived from the block-table
        # bound per call (patched into the graph); the grid is sized to the cap. FINAL = False:
        # the combine pass always runs, so the split count can vary freely at replay
        target = 2 * _get_sm_count(dev)
        splits_cap = max(1, min(target // programs, 128))

        cache_t = "*i32" if self.quant else "*fp16"
        k_split = _compile_kernel(dev, _mla_decode_split_kernel,
            {"q_lat": "*fp16", "q_pe": "*fp16", "ckv_cache": cache_t, "kpe_cache": "*fp16",
             "ckv_scales": "*fp16", "h32": "*fp16", "block_table": "*i32",
             "cache_seqlens": "*i32", "out": "*fp16", "partial_o": "*fp32",
             "partial_ml": "*fp32", "split_len": "i32", "num_pages_per_seq": "i32",
             "num_splits": "i32"}
            | {n: "constexpr" for n in (
                "QC", "QC_TRANS", "Q_PE_TM", "bsz", "q_len", "pre_appended_len", "n_q_heads",
                "page_size", "D_c", "D_r", "scale", "CAUSAL", "FINAL",
                "BLOCK_M", "BLOCK_H", "BLOCK_ROWS", "BLOCK_N")},
            dict(QC = self.k_bits, QC_TRANS = True, Q_PE_TM = True, bsz = bsz, q_len = q_len,
                 pre_appended_len = q_len, n_q_heads = H, page_size = PAGE_SIZE, D_c = D_c,
                 D_r = D_r, scale = float(self.sm_scale), CAUSAL = True, FINAL = False,
                 BLOCK_M = block_m, BLOCK_H = block_h, BLOCK_ROWS = block_rows,
                 BLOCK_N = block_n),
            n_warps, n_stages)

        k_combine = _compile_kernel(dev, _mla_decode_combine_kernel,
            {"partial_o": "*fp32", "partial_ml": "*fp32", "out": "*fp16", "h32": "*fp16",
             "num_splits": "i32"}
            | {n: "constexpr" for n in (
                "QC", "bsz", "q_len", "n_q_heads", "D_c", "BLOCK_M", "BLOCK_H", "BLOCK_ROWS")},
            dict(QC = self.k_bits, bsz = bsz, q_len = q_len, n_q_heads = H, D_c = D_c,
                 BLOCK_M = block_m, BLOCK_H = block_h, BLOCK_ROWS = block_rows),
            4, 1)

        unfold_bm = min(64, triton.next_power_of_2(max(R, 16)))
        k_unfold = _compile_kernel(dev, _mla_unfold_kernel,
            {"o_lat": "*fp16", "w_uv_flat": "*fp16", "out": "*fp16", "R": "i32"}
            | {n: "constexpr" for n in ("n_q_heads", "D_c", "D_v", "BLOCK_M", "BLOCK_K")},
            dict(n_q_heads = H, D_c = D_c, D_v = D_v, BLOCK_M = unfold_bm, BLOCK_K = 128),
            4, 2)

        # Static intermediates, shared between layers with the same shapes on the same device
        q_full = g_tensor_cache.get(dev, (R, self.w_q), torch.half, "bcm_qfull")
        q_a = g_tensor_cache.get(dev, (R, self.q_lora_rank), torch.half, "bcm_qa") \
            if self.q_lora_rank else None
        ckv_kpe = g_tensor_cache.get(dev, (R, self.w_kv), torch.half, "bcm_ckvkpe")
        ckv = g_tensor_cache.get(dev, (R, D_c), torch.half, "bcm_ckv")
        kpe = g_tensor_cache.get(dev, (R, D_r), torch.half, "bcm_kpe")
        q_pe = g_tensor_cache.get(dev, (R, H, D_r), torch.half, "bcm_qpe")
        q_lat = g_tensor_cache.get(dev, (H, R, D_c), torch.half, "bcm_qlat")
        o_lat = g_tensor_cache.get(dev, (H, R, D_c), torch.half, "bcm_olat")
        o = g_tensor_cache.get(dev, (R, H * D_v), torch.half, "bcm_o")
        partial_o = g_tensor_cache.get(dev, (programs * splits_cap * block_rows * D_c,), torch.float, "bcm_po")
        partial_ml = g_tensor_cache.get(dev, (programs * splits_cap * block_rows * 2,), torch.float, "bcm_ml")
        if self.quant:
            qtmp = g_tensor_cache.get(dev, (R, D_c // 32 * self.k_bits), torch.int, "bcm_qtmp")
            stmp = g_tensor_cache.get(dev, (R, D_c // 32), torch.half, "bcm_stmp")
        else:
            qtmp = stmp = None

        self.bc.configure_slot(
            bsz, q_len,
            q_full, q_a, ckv_kpe, ckv, kpe, q_pe, q_lat, o_lat, o, partial_o, partial_ml,
            qtmp, stmp,
            k_stage, k_absorb, k_append, k_split, k_combine, k_unfold,
            block_n, splits_cap, programs,
            triton.cdiv(R, absorb_bm), D_c // 128, triton.cdiv(R, unfold_bm),
        )

    def step(
        self,
        x: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        position: int,
        positions: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, q_len, _ = x.shape
        if (bsz, q_len) not in self.configured:
            self._configure(bsz, q_len)
            self.configured.add((bsz, q_len))
        y = torch.empty((bsz, q_len, self.hidden_size), dtype = self.o_dtype, device = x.device)
        self.bc.run(bsz, q_len, x, y, cache_seqlens, block_table, position, positions, position_ids)
        return y


def _proj_ok(p, in_features = None, out_features = None):
    """Quantized projection usable inside the graph: EXL3 with a bound BC class, no bias, and
    (where required) exact unpadded widths."""
    return (
        p is not None and p.quant_type == "exl3" and p.inner.bc is not None and
        p.inner.bias is None and
        (in_features is None or p.in_features == in_features) and
        (out_features is None or p.out_features_unpadded == out_features == p.out_features)
    )


def build_bc_mla(module, layer):
    """Build a BCMLA for the module/cache-layer pair, or return None when the configuration is
    not supported (caller falls back to the dispatch path)."""
    from ...cache.mla import CacheLayer_MLA_fp16, CacheLayer_MLA_quant
    from ...util.rope import RopeStyle

    m = module
    H, D_c, D_r, D_v = m.num_q_heads, m.kv_lora_rank, m.qk_rope_head_dim, m.v_head_dim
    dev = torch.device(m.device)
    if not (
        bc_attn_enable and
        m.rope is not None and m.rope.rope_settings.rope_style != RopeStyle.NONE and
        m.rope.llama_4_scaling_beta == 0.0 and
        # The staging/attention kernels index with tl.arange over these widths
        _is_pow2(D_c) and _is_pow2(D_r) and _is_pow2(D_v) and D_c % 128 == 0 and
        # Projections read x directly (no padded-input staging) and write the statics; the q and
        # kv widths may pad (the kernels read at true offsets), the rest must be exact
        _proj_ok(m.q_proj, in_features = m.q_lora_rank or m.hidden_size) and
        (m.q_a_proj is None or (
            _proj_ok(m.q_a_proj, in_features = m.hidden_size, out_features = m.q_lora_rank) and
            m.q_a_layernorm.weight is not None
        )) and
        _proj_ok(m.kv_a_proj_with_mqa, in_features = m.hidden_size) and
        _proj_ok(m.o_proj, in_features = H * D_v, out_features = m.hidden_size) and
        m.kv_a_layernorm.weight is not None and
        m.w_uk_flat is not None and
        not m.has_split_cache and
        isinstance(layer, (CacheLayer_MLA_fp16, CacheLayer_MLA_quant)) and
        (not isinstance(layer, CacheLayer_MLA_quant) or (
            layer.qk is not None and layer.qk.device == dev
        )) and
        (not isinstance(layer, CacheLayer_MLA_fp16) or (
            layer.k is not None and layer.k.device == dev
        ))
    ):
        _trace_build(m, None, "mla")
        return None
    bcm = BCMLA(m, layer)
    _trace_build(m, bcm, "mla")
    return bcm
