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

        # NoPE models (GLM5.3: qk_rope_head_dim 0) have no rope instance; the rope stages are
        # compiled out of the graph and the ext object gets inert placeholders
        rope = m.rope
        if rope is None or self.qk_rope_head_dim == 0:
            rope_inv_freq = torch.zeros((1,), dtype = torch.float, device = self.device)
            rope_style, attn_factor, rotate_dims = 0, 1.0, 0
        else:
            rope_inv_freq = rope.inv_freq
            rope_style = int(rope.rope_settings.rope_style)
            attn_factor = rope.attn_factor
            rotate_dims = rope.rope_settings.rotate_dims
        self.rope_inv_freq = rope_inv_freq
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
            inv_freq = rope_inv_freq,
            rope_style = rope_style,
            attn_factor = attn_factor,
            rotate_dims = rotate_dims,
            # The module zeroes the beta on its RoPE instance (the rope stage must not scale
            # k_pe, and only sees the q_pe slice anyway); the graph applies the full-query
            # scale as its own stage on q_full
            l4_scaling_beta = m.l4_beta,
            l4_scaling_original = m.l4_original,
            w_uk_flat = m.w_uk_flat,
            w_uv_flat = m.w_uv_flat,
            quant_cache = self.quant,
            cache_ckv = self.cache_ckv,
            cache_kpe = self.cache_kpe,
            cache_scales = self.cache_scales,
            xh = xh,
            h32 = h32,
        )

        # DSA lightning indexer (GLM-5.2). Full layers project/norm/rope/append keys every
        # step and score + select in the sparse regime; shared layers gather through an
        # externally produced index list (patched pointer). See mla_attention.h
        self.indexer_mode = m.indexer_mode
        self.index_topk = m.index_topk
        # kpool (GLM5.3): the key plane packs [k || gate_scores] rows and selection runs over
        # pooled keys in a second plane; only full-indexer layers do kpool work
        self.index_kpool = m.index_kpool if m.indexer_mode == "full" else 0
        if m.indexer_mode == "full":
            self.cache_kidx = layer.get_idx()
            self.idx_norm_eps = m.idx_k_norm.layernorm_eps
            if self.index_kpool:
                self.cache_kpool = layer.get_pool()
                # the checkpoint gate weight is (D_i, hidden); hgemm consumes (in, out)
                self.idx_gate_w = m.idx_kpool_gate.data.T.contiguous()
            else:
                self.cache_kpool = self.idx_gate_w = None
            self.bc.set_indexer(
                mode = 1,
                wq_b = m.idx_wq_b.inner.bc,
                wk_w = m.idx_wk.inner.weight,
                k_norm_w = m.idx_k_norm.weight.data.half().contiguous(),
                k_norm_b = m.idx_k_norm.bias.data.half().contiguous(),
                weights_w = m.idx_weights.inner.weight,
                kidx = self.cache_kidx.view(-1, m.idx_plane_dim),
                n_heads = m.index_n_heads,
                head_dim = m.index_head_dim,
                topk = m.index_topk,
                kpool = self.index_kpool,
                kpool_tail = m.index_kpool_tail,
                gate_w = self.idx_gate_w,
                kpool_ape = m.idx_kpool_ape if self.index_kpool else None,
                kpool_plane = self.cache_kpool.view(-1, m.index_head_dim)
                    if self.index_kpool else None,
            )
        elif m.indexer_mode == "shared":
            self.cache_kidx = None
            self.bc.set_indexer(
                mode = 2, wq_b = None, wk_w = None, k_norm_w = None, k_norm_b = None,
                weights_w = None, kidx = None,
                n_heads = m.index_n_heads, head_dim = m.index_head_dim, topk = m.index_topk,
            )
        else:
            self.cache_kidx = None

        self.slot_indices = {}
        self.configured = set()

    def _configure(self, bsz: int, q_len: int, regime: int):
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

        # Static intermediates, shared between layers on the same device. Bucketed flat
        # backings sliced to shape: every (bsz, q_len) slot draws from the same per-tag
        # allocations instead of keeping a fresh set per distinct R, so late-configured
        # slots (odd prefill-tail lengths, new draft depths) do not grow the footprint
        # beyond the largest slot's
        def sbuf(tag, *shape, dtype = torch.half):
            n = 1
            for s in shape: n *= s
            return g_tensor_cache.get_bucketed(dev, n, dtype, tag).view(*shape)
        q_full = sbuf("bcm_qfull", R, self.w_q)
        q_a = sbuf("bcm_qa", R, self.q_lora_rank) if self.q_lora_rank else None
        ckv_kpe = sbuf("bcm_ckvkpe", R, self.w_kv)
        ckv = sbuf("bcm_ckv", R, D_c)
        kpe = sbuf("bcm_kpe", R, D_r)
        q_pe = sbuf("bcm_qpe", R, H, D_r)
        q_lat = sbuf("bcm_qlat", H, R, D_c)
        o_lat = sbuf("bcm_olat", H, R, D_c)
        o = sbuf("bcm_o", R, H * D_v)
        partial_o = sbuf("bcm_po", programs * splits_cap * block_rows * D_c, dtype = torch.float)
        partial_ml = sbuf("bcm_ml", programs * splits_cap * block_rows * 2, dtype = torch.float)
        if self.quant:
            qtmp = sbuf("bcm_qtmp", R, D_c // 32 * self.k_bits, dtype = torch.int)
            stmp = sbuf("bcm_stmp", R, D_c // 32)
        else:
            qtmp = stmp = None

        self.bc.configure_slot(
            bsz, q_len, regime,
            q_full, q_a, ckv_kpe, ckv, kpe, q_pe, q_lat, o_lat, o, partial_o, partial_ml,
            qtmp, stmp,
            k_stage, k_absorb, k_append, k_split, k_combine, k_unfold,
            block_n, splits_cap, programs,
            triton.cdiv(R, absorb_bm), D_c // 128, triton.cdiv(R, unfold_bm),
        )

        if self.indexer_mode is not None:
            self._configure_dsa(bsz, q_len, regime)

    def _configure_dsa(self, bsz: int, q_len: int, regime: int):
        from .mla_triton import (
            _mla_idx_norm_kernel,
            _mla_plane_update_kernel,
        )
        from .dsa_triton import (
            _dsa_indexer_fewq_kernel,
            _dsa_attn_split_kernel,
            _dsa_attn_combine_kernel,
        )

        dev = self.device
        m = self.module
        H = self.num_q_heads
        D_c, D_r = self.kv_lora_rank, self.qk_rope_head_dim
        D = D_c + D_r
        R = bsz * q_len
        R_pad = max(R, 8)   # cuBLASLt picks a ~13x slower kernel below M = 8 (bc_dsa pattern)
        full = self.indexer_mode == "full"
        BLOCK_H = 16
        N_SPLITS = 16
        hb = -(-H // BLOCK_H)
        # kpool: selection picks topk/P pools, expansion emits raw indices plus up to P-1
        # tail tokens per query row
        P = self.index_kpool
        if P:
            sel = self.index_topk // P
            kp = -(-(sel * P + (P - 1 if m.index_kpool_tail else 0)) // 32) * 32
            kp_pool = -(-sel // 32) * 32
        else:
            sel = kp_pool = 0
            kp = -(-self.index_topk // 32) * 32

        def sbuf(tag, *shape, dtype = torch.half):
            n = 1
            for s in shape: n *= s
            return g_tensor_cache.get_bucketed(dev, n, dtype, tag).view(*shape)

        k_idx_norm = k_plane_append = k_fewq = None
        x_st = kidx = kidx_n = qidx = wts = scores = None
        fewq_gy = 0
        if full:
            Hi, Di = m.index_n_heads, m.index_head_dim
            x_st = sbuf("bcm_xst", R_pad, self.hidden_size)
            x_st.zero_()
            kidx = sbuf("bcm_kidxr", R_pad, Di)
            kidx_n = sbuf("bcm_kidxn", R, Di)
            k_idx_norm = _compile_kernel(dev, _mla_idx_norm_kernel,
                {"k_raw": "*fp16", "w": "*fp16", "b": "*fp16", "k_out": "*fp16", "R": "i32"}
                | {n: "constexpr" for n in ("eps", "D")},
                dict(eps = float(self.idx_norm_eps), D = Di), 2, 1)
            plane_sig = {"rows_new": "*fp16", "plane_cache": "*fp16", "block_table": "*i32",
                 "cache_seqlens": "*i32", "num_pages_per_seq": "i32", "append_len": "i32"} \
                | {n: "constexpr" for n in ("page_size", "D", "DST_D", "DST_OFF")}
            k_plane_append = _compile_kernel(dev, _mla_plane_update_kernel, plane_sig,
                dict(page_size = PAGE_SIZE, D = Di,
                     DST_D = 2 * Di if P else 0, DST_OFF = 0), 2, 2)

        gidx = pool_idx = None
        k_gate_append = k_pool_update = k_pool_expand = None
        if full and P:
            from .dsa_triton import _dsa_pool_update_kernel
            gidx = sbuf("bcm_gidx", R_pad, Di)
            gidx.zero_()
            k_gate_append = _compile_kernel(dev, _mla_plane_update_kernel, plane_sig,
                dict(page_size = PAGE_SIZE, D = Di, DST_D = 2 * Di, DST_OFF = Di), 2, 2)
            k_pool_update = _compile_kernel(dev, _dsa_pool_update_kernel,
                {"plane": "*fp16", "pool_plane": "*fp16", "ape": "*fp32",
                 "block_table": "*i32", "cache_seqlens": "*i32",
                 "num_pages_per_row": "i32", "append_len": "i32"}
                | {n: "constexpr" for n in ("page_size", "P", "D", "MAXPOOLS")},
                dict(page_size = PAGE_SIZE, P = P, D = Di, MAXPOOLS = q_len // P + 1), 2, 1)

        indices = dsa_arr = ws_ml = ws_acc = None
        k_dsa_split = k_dsa_combine = None
        if regime == 1:
            if full:
                Hi, Di = m.index_n_heads, m.index_head_dim
                qidx = sbuf("bcm_qidx", R, Hi * Di)
                wts = sbuf("bcm_wts", R_pad, Hi)
                wts.zero_()
                # Scoring covers the full plane capacity every step (bounds written -inf in
                # kernel), so no stale scores survive from longer contexts
                if P:
                    # Pool capacity padded up to the tile width: tiles past T retire and the
                    # store mask covers the pad, so the -inf fill survives in the tail
                    cap = self.cache_kpool.view(-1, Di).shape[0]
                    s_max = -(-cap // 128) * 128
                else:
                    s_max = self.cache_kidx.view(-1, Di).shape[0]
                    assert s_max % 128 == 0
                scores = sbuf("bcm_scores", R, s_max)
                # The scoring kernel writes only [0, T); the warmup top-k scans the full
                # static width, so the tail must hold -inf from this one-time fill (the
                # captured graph patches the top-k scan width to T afterwards)
                scores.fill_(-float("inf"))
                # Batched slots score in MULTIROW mode: T / q_pos0 / bound_max are per-job
                # device pointers into the seq-state array, the block table one row per job
                mr = 1 if bsz > 1 else 0
                bnd_t = "*i32:16" if mr else "i32"
                sig = {
                    "q_idx": "*fp16:16", "w": "*fp16:16", "k_idx": "*fp16:16",
                    "scores": "*fp16:16", "T": bnd_t, "R": "i32", "q_pos0": bnd_t,
                    "bound_max": bnd_t, "block_table": "*i32:16", "num_pages_per_row": "i32",
                } | {n: "constexpr" for n in (
                    "H_i", "H_pad", "D_i", "S_stride", "compress_rate", "scale", "BLOCK_N",
                    "SEQ", "MULTIROW", "EPP", "DEBUG_BOUNDS", "DEBUG_PAGES")}
                consts = dict(
                    H_i = Hi, H_pad = max(16, 1 << (Hi - 1).bit_length()), D_i = Di,
                    S_stride = s_max, compress_rate = P if P else 1,
                    scale = Di ** -0.5 * Hi ** -0.5, BLOCK_N = 128,
                    SEQ = q_len, MULTIROW = mr, EPP = PAGE_SIZE // P if P else PAGE_SIZE,
                    DEBUG_BOUNDS = 0, DEBUG_PAGES = 0,
                )
                k_fewq = _compile_kernel(dev, _dsa_indexer_fewq_kernel, sig, consts, 8, 2)
                fewq_gy = s_max // 128

            # The indices static is deliberately SHARED between layers (one tag): layers run
            # in order on one stream, so a full layer's selection is in place when its shared
            # consumers gather through it, exactly like the eager params flow
            indices = sbuf("bcm_dsa_idx", R, kp, dtype = torch.int32)
            if full and P:
                from .dsa_triton import _dsa_pool_expand_kernel
                pool_idx = sbuf("bcm_pool_idx", R, kp_pool, dtype = torch.int32)
                k_pool_expand = _compile_kernel(dev, _dsa_pool_expand_kernel,
                    {"pool_idx": "*i32", "out": "*i32", "q_pos0": "i32"}
                    | {n: "constexpr" for n in (
                        "P", "SEL", "K_pad", "KP_pool", "TAIL", "SEQ", "MULTIROW", "BLOCK")},
                    dict(P = P, SEL = sel, K_pad = kp, KP_pool = kp_pool,
                         TAIL = 1 if m.index_kpool_tail else 0, SEQ = q_len,
                         MULTIROW = 0, BLOCK = 256), 4, 1)
            if bsz > 1:
                dsa_arr = g_tensor_cache.get(dev, (2, MAX_BSZ), torch.int32, "bcm_dsa_arr")
            # ws_acc rows are D_c wide: OUT_LATENT never accumulates the rope half
            ws_ml = sbuf("bcm_dsa_wsml", R * hb * N_SPLITS * BLOCK_H * 2, dtype = torch.float)
            ws_acc = sbuf("bcm_dsa_wsacc", R * hb * N_SPLITS * BLOCK_H * D_c, dtype = torch.float)
            self.slot_indices[(bsz, q_len)] = indices

            sig_s = {
                "q": "*fp16:16", "ring": "*fp16:16", "kv_chunk": "*fp16:16",
                "pool_c": "*fp16:16", "pool_r": "*fp16:16", "block_table": "*i32:16",
                "indices": "*i32:16", "ws_ml": "*fp32:16", "ws_acc": "*fp32:16",
                "k_len": "i32", "win_len": "i32", "pool_len": "i32",
                "num_pages_per_row": "i32", "q_pos0": "i32", "win_floor": "i32",
                "ring_beg": "i32", "slot_ids": "i32", "ring_stride": "i32",
            } | {n: "constexpr" for n in (
                "H", "page_size", "D_c", "D_c_pad", "D_r", "K_pad", "compress_rate", "scale",
                "HAS_WINDOW", "DENSE_POOL", "BLOCK_H", "BLOCK_N", "BLOCK_W", "SEQ",
                "MULTIROW", "DEBUG_BOUNDS", "DEBUG_PAGES", "Q_SPLIT", "OUT_LATENT")}
            consts_s = dict(
                H = H, page_size = PAGE_SIZE, D_c = D_c,
                D_c_pad = 1 << (D_c - 1).bit_length(), D_r = D_r, K_pad = kp,
                compress_rate = 1, scale = float(self.sm_scale),
                HAS_WINDOW = False, DENSE_POOL = False,
                BLOCK_H = BLOCK_H, BLOCK_N = 32, BLOCK_W = 16,
                # MULTIROW under Q_SPLIT only routes each job to its own block-table row;
                # the per-job state args stay scalar (causality lives in the selection)
                SEQ = q_len, MULTIROW = 1 if bsz > 1 else 0,
                DEBUG_BOUNDS = 0, DEBUG_PAGES = 0,
                Q_SPLIT = 1, OUT_LATENT = 1,
            )
            k_dsa_split = _compile_kernel(dev, _dsa_attn_split_kernel, sig_s, consts_s, 4, 2)

            sig_c = {
                "ws_ml": "*fp32:16", "ws_acc": "*fp32:16", "sinks": "*fp32:16",
                "derot_inv_freq": "*fp32:16",
                "out": "*fp16:16", "q_pos0": "i32", "R": "i32", "n_splits": "i32",
            } | {n: "constexpr" for n in (
                "H", "D_c", "D_r", "HAS_SINKS", "DEROTATE", "HPG", "BLOCK_H", "BLOCK_D",
                "SEQ", "MULTIROW", "OUT_LATENT")}
            consts_c = dict(
                H = H, D_c = D_c, D_r = D_r, HAS_SINKS = False, DEROTATE = False,
                HPG = 0, BLOCK_H = BLOCK_H, BLOCK_D = 128,
                SEQ = 1, MULTIROW = 0, OUT_LATENT = 1,
            )
            k_dsa_combine = _compile_kernel(dev, _dsa_attn_combine_kernel, sig_c, consts_c, 4, 2)

        self.bc.configure_slot_dsa(
            bsz, q_len, regime, x_st, kidx, kidx_n, qidx, wts, scores, indices,
            dsa_arr, ws_ml, ws_acc, k_idx_norm, k_plane_append, k_fewq,
            k_dsa_split, k_dsa_combine, hb, N_SPLITS, fewq_gy,
            gidx, pool_idx, k_gate_append, k_pool_update, k_pool_expand,
        )

    def step(
        self,
        x: torch.Tensor,
        params: dict,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        position: int,
        positions: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        bsz, q_len, _ = x.shape

        # kpool slots are single-job only: the batched scoring path derives token-unit
        # bounds on device, and the pooled scan counts pools
        if self.index_kpool and bsz > 1:
            return None

        if self.indexer_mode is None:
            regime, t_total, ext_indices = 0, 0, None
        else:
            host_seqlens = params.get("_mla_host_seqlens")
            assert host_seqlens is not None
            t_total = max(host_seqlens) + q_len
            regime = 1 if t_total > self.index_topk else 0
            ext_indices = None
            if regime:
                # Sparse restriction: fp16 latent cache (the gather kernels read fp16 rows)
                if self.quant:
                    return None
                # Single job: the scalar position drives the scoring bounds (the generator
                # usually passes positions as a tensor and leaves the scalar at 0). Batched
                # slots derive their per-job bounds on device from cache_seqlens instead
                if bsz == 1:
                    position = host_seqlens[0]
                if self.indexer_mode == "shared":
                    ext_indices = params.get("dsa_topk_indices")
                    if ext_indices is None:
                        return None
                    if ext_indices.device != x.device:
                        ext_indices = ext_indices.to(x.device)

        if (bsz, q_len, regime) not in self.configured:
            self._configure(bsz, q_len, regime)
            self.configured.add((bsz, q_len, regime))
        y = torch.empty((bsz, q_len, self.hidden_size), dtype = self.o_dtype, device = x.device)
        self.bc.run(bsz, q_len, x, y, cache_seqlens, block_table, position, positions,
                    position_ids, regime, t_total, ext_indices)
        if regime and self.indexer_mode == "full":
            params["dsa_topk_indices"] = self.slot_indices[(bsz, q_len)]
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
        # NoPE models (D_r 0) compile the rope stages out; otherwise a rope instance with a
        # supported style is required
        (D_r == 0 or (
            m.rope is not None and m.rope.rope_settings.rope_style != RopeStyle.NONE and
            _is_pow2(D_r)
        )) and
        # The staging/attention kernels index with tl.arange over these widths
        _is_pow2(D_c) and _is_pow2(D_v) and D_c % 128 == 0 and
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
        # DSA indexer layers (GLM-5.2): full layers need the quantized wq_b, the fp16 key/
        # weight heads, the biased key norm and the paged key plane on this cache layer
        (m.indexer_mode is None or (
            _is_pow2(m.index_head_dim) and m.qk_rope_head_dim <= m.index_head_dim and
            # kpool (GLM5.3): pools never straddle pages, selection counts whole pools, and
            # full layers need the gate/APE tensors and the pooled plane
            (not m.index_kpool or (
                PAGE_SIZE % m.index_kpool == 0 and
                m.index_topk % m.index_kpool == 0 and
                (m.indexer_mode == "shared" or (
                    m.idx_kpool_ape is not None and m.idx_kpool_gate is not None and
                    layer.get_pool() is not None
                ))
            )) and
            (m.indexer_mode == "shared" or (
                m.q_lora_rank and _proj_ok(m.idx_wq_b, in_features = m.q_lora_rank) and
                getattr(m.idx_wk.inner, "weight", None) is not None and
                m.idx_wk.inner.weight.dtype == torch.half and
                getattr(m.idx_weights.inner, "weight", None) is not None and
                m.idx_weights.inner.weight.dtype == torch.half and
                m.idx_k_norm.weight is not None and m.idx_k_norm.bias is not None and
                layer.get_idx() is not None
            ))
        )) and
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
