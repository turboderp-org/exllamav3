import os

import torch

from ...ext import exllamav3_ext as ext
from ...constants import PAGE_SIZE
from ...util.tensor import g_tensor_cache

"""
Graph-captured decode attention (BC_Attention): the whole attention block for a decode step --
q/k/v projections, fused head norm + RoPE, cache append and the flash-decoding kernels, then
o_proj -- runs as one C++ call, captured as one CUDA graph per (bsz, q_len) slot after a warmup
run and replayed with only the input/output/seqlens/block-table/positions pointers patched.

The attention kernels are the same Triton kernels the dispatch path JITs, compiled ahead of time
(triton.compile -> cubin) with the slot shapes and split configuration baked as constexprs, and
launched from C++ through the TritonKernel ext class. The block-table width and split length are
runtime kernel arguments frozen at capture, so when the generator's block table grows the slot
is recaptured without recompiling (a new split count does recompile; Triton's disk cache makes
that cheap after the first run).

Instances are keyed per cache layer, since the cache tensors are baked into the captured
graphs. Static intermediates come from g_tensor_cache and are shared between layers of the same
shape on the same device.

QSA (Qwen3.8-Flash-Next): modules with a qsa_indexer arm the sparse-selection stages
(set_qsa), following the DSA-on-MLA pattern in bc_mla.py. Armed instances maintain the
indexer's raw/pooled key planes in-graph on every step (dense slots included, so the planes
stay complete below the sparse threshold) and regime-1 slots (single-job, q_len 1..16,
context past the threshold) run scoring / top-k / expansion / gathered GQA attention instead 
of the dense flash-decoding kernels. Scoring and expansion reuse the DSA kernels (QSA's 
uniform-head-weight relu score and forced tail block match them exactly); the stage, pool-update
and gathered split kernels live in qsa_triton.py. A module with an indexer that cannot be armed
declines the BC path outright: an unarmed graph would let the planes go stale under dense decode
and poison later sparse selection.

Enabled by default; EXL3_BC_ATTN=0 disables the path. Unsupported module/cache configurations
fall back to the dispatch path by design (build_bc_attn returns None); unexpected failures
while building the path raise.
"""

bc_attn_enable = os.environ.get("EXL3_BC_ATTN", "1") != "0"

# EXL3_BC_ATTN_TRACE=1: print build/decline per module/layer (activation check for A/B tests)
_bc_trace = os.environ.get("EXL3_BC_ATTN_TRACE", "0") != "0"

def _trace_build(m, result, kind):
    if _bc_trace:
        print(f" -- BC-{kind}: {'built' if result is not None else 'DECLINED'}"
              f" layer {getattr(m, 'layer_idx', '?')} device {m.device}")

MAX_BSZ = 8
MAX_QLEN = 16
MAX_R = MAX_BSZ * MAX_QLEN

_kernel_cache = {}
_sm_count = {}


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _compile_kernel(device: torch.device, fn, signature: dict, constexprs: dict,
                    num_warps: int, num_stages: int):
    key = (device.index, fn.__name__, tuple(sorted(constexprs.items())), num_warps, num_stages,
           tuple(sorted(signature.items())))
    k = _kernel_cache.get(key)
    if k is None:
        import triton
        from triton.compiler import ASTSource
        # A ":16" suffix on a signature entry declares the argument 16-byte divisible
        # (pointer alignment / scalar divisibility), matching the JIT's specialization --
        # without it the AOT kernel loses vectorized global loads (~1.5x on
        # bandwidth-heavy kernels). Every launch MUST then honor the alignment
        attrs = {}
        sig = {}
        for name, ty in signature.items():
            if isinstance(ty, str) and ty.endswith(":16"):
                sig[name] = ty[:-3]
                attrs[(fn.arg_names.index(name),)] = [["tt.divisibility", 16]]
            else:
                sig[name] = ty
        with torch.cuda.device(device):
            src = ASTSource(fn = fn, signature = sig, constexprs = constexprs, attrs = attrs)
            ck = triton.compile(src, options = {"num_warps": num_warps, "num_stages": num_stages})
            k = ext.TritonKernel(ck.asm["cubin"], ck.metadata.name, ck.metadata.num_warps, ck.metadata.shared)
        _kernel_cache[key] = k
    return k


def _get_sm_count(device: torch.device | int) -> int:
    # TP shards store their device as a plain index
    idx = device.index if hasattr(device, "index") else device
    if idx not in _sm_count:
        _sm_count[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _sm_count[idx]


class BCAttn:
    """Python-side owner of one ext.BC_Attention (per attention module and cache layer):
    collects the projection/norm/rope/cache handles at construction and compiles + registers the
    per-slot attention kernels lazily."""

    @staticmethod
    def _has_bc(proj):
        return proj is not None and proj.quant_type == "exl3" and proj.inner.bc is not None

    @staticmethod
    def _fp16_gate_weight(proj):
        """Weight of an unquantized full-gate projection the graph can run through cublas
        (input staged through a static, so the captured node needs no patching), or None."""
        if proj is None or proj.quant_type != "fp16":
            return None
        w = proj.inner.weight
        if w.dtype != torch.half or not w.is_contiguous() or proj.inner.bias is not None:
            return None
        return w

    def __init__(self, module, cache_k, cache_v, k_scales = None, v_scales = None,
                 k_bits = 0, v_bits = 0, qsa_layer = None):
        self.module = module
        # TP shards store their device as a plain index; normalize for .index consumers
        self.device = torch.device(module.device) if isinstance(module.device, int) else module.device
        self.head_dim = module.head_dim
        self.num_q_heads = module.num_q_heads
        self.num_kv_heads = module.num_kv_heads
        self.hidden_size = module.hidden_size
        self.sm_scale = module.sm_scale
        self.window_size = module.sliding_window
        self.softcap = module.logit_softcapping

        self.quant = k_bits > 0
        self.cache_k, self.cache_v = cache_k, cache_v
        self.k_scales, self.v_scales = k_scales, v_scales
        self.k_bits, self.v_bits = k_bits, v_bits
        if self.quant:
            from .triton_paged import _get_h32
            h32 = _get_h32(self.device)
        else:
            h32 = g_tensor_cache.get(self.device, (1,), torch.half, "bca_dummy")
        self.max_pages = self.cache_k.shape[0]

        # Padded hidden dim (gpt-oss): the quantized projections' K is the 128-aligned width;
        # the graph stages x through a zero-padded static and trims the o_proj output
        self.hidden_padded = module.q_proj.in_features
        self.sinks = getattr(module, "sinks", None)

        rope = module.rope
        mkv = module.multi_kv
        mqg = module.multi_qg
        w = max(self.hidden_padded, self.num_q_heads * self.head_dim)
        xh = g_tensor_cache.get(self.device, (2 * MAX_R * w,), torch.half, "bca_xh")

        self.o_dtype = module.o_proj.inner.default_out_dtype
        self.use_k_as_v = getattr(module, "use_k_as_v", False)

        # 0 = none, 1 = headwise (one gate per head, always an unquantized fp16 projection, run
        # as a captured cublas gemm over the staged input; sigmoid or softplus by gate_softplus),
        # 2 = full (o *= sigmoid(g)), 3 = interleaved q/g projection
        if getattr(module, "interleaved_gate", False):
            self.gate_mode = 3
        elif module.g_proj is not None:
            self.gate_mode = 1 if getattr(module, "headwise_gate", False) else 2
        else:
            self.gate_mode = 0
        self.gate_softplus = getattr(module, "gate_softplus", False)

        # Unquantized gate weight: the headwise gate is always fp16 (too narrow to quantize);
        # a full gate is fp16 only in older quants that stored g_proj unquantized. Either runs
        # as a captured cublas gemm over the statically staged input instead of an exl3 kernel
        self.g_weight = None
        if self.gate_mode == 1:
            self.g_weight = self._fp16_gate_weight(module.g_proj)
            assert self.g_weight is not None, "BC_Attention: headwise gate requires an fp16 g_proj"
        elif self.gate_mode == 2 and mqg is None and not self._has_bc(module.g_proj):
            self.g_weight = self._fp16_gate_weight(module.g_proj)

        self.bc = ext.BC_Attention(
            num_q_heads = self.num_q_heads,
            num_kv_heads = self.num_kv_heads,
            head_dim = self.head_dim,
            hidden_size = self.hidden_size,
            hidden_size_padded = self.hidden_padded,
            page_size = PAGE_SIZE,
            q_proj = module.q_proj.inner.bc,
            k_proj = module.k_proj.inner.bc if self._has_bc(module.k_proj) else None,
            v_proj = module.v_proj.inner.bc if (not self.use_k_as_v and self._has_bc(module.v_proj)) else None,
            kv_ptrs_trellis = mkv.ptrs_trellis if mkv is not None else None,
            kv_ptrs_suh = mkv.ptrs_suh if mkv is not None else None,
            kv_ptrs_svh = mkv.ptrs_svh if mkv is not None else None,
            kv_K = mkv.K if mkv is not None else 0,
            kv_mcg = bool(mkv.mcg) if mkv is not None else False,
            kv_mul1 = bool(mkv.mul1) if mkv is not None else False,
            o_proj = module.o_proj.inner.bc,
            use_k_as_v = self.use_k_as_v,
            gate_mode = self.gate_mode,
            gate_softplus = self.gate_softplus,
            g_proj = module.g_proj.inner.bc if (self.gate_mode == 2 and self._has_bc(module.g_proj)) else None,
            g_weight = self.g_weight,
            qg_ptrs_trellis = mqg.ptrs_trellis if mqg is not None else None,
            qg_ptrs_suh = mqg.ptrs_suh if mqg is not None else None,
            qg_ptrs_svh = mqg.ptrs_svh if mqg is not None else None,
            qg_K = mqg.K if mqg is not None else 0,
            qg_mcg = bool(mqg.mcg) if mqg is not None else False,
            qg_mul1 = bool(mqg.mul1) if mqg is not None else False,
            q_norm = module.q_norm_tensor,
            k_norm = module.k_norm_tensor,
            norm_eps = module.norm_eps,
            norm_constant_bias = module.norm_constant_bias,
            v_norm = module.v_norm is not None,
            v_norm_w = module.v_norm.weight.data if (module.v_norm is not None and module.v_norm.weight is not None) else None,
            v_norm_eps = module.v_norm.rms_norm_eps if module.v_norm is not None else 1e-6,
            v_norm_constant_bias = module.v_norm.constant_bias if module.v_norm is not None else 0.0,
            v_norm_constant_scale = module.v_norm.constant_scale if module.v_norm is not None else 1.0,
            # NoPE (rope is None): the rope stage is omitted from the graph
            inv_freq = rope.inv_freq if rope is not None else None,
            rope_style = int(rope.rope_settings.rope_style) if rope is not None else 0,
            attn_factor = rope.attn_factor if rope is not None else 1.0,
            l4_scaling_beta = rope.llama_4_scaling_beta if rope is not None else 0.0,
            l4_scaling_original = rope.llama_4_scaling_original if rope is not None else 0,
            rotate_dims = rope.rope_settings.rotate_dims if rope is not None else 0,
            quant_cache = self.quant,
            cache_k = self.cache_k,
            cache_v = self.cache_v,
            cache_k_scales = self.k_scales,
            cache_v_scales = self.v_scales,
            xh = xh,
            h32 = h32,
            sinks = self.sinks,
        )

        # QSA indexer (Qwen3.8-Flash-Next): arm the sparse-selection stages. The armed instance
        # maintains the raw/pooled key planes in-graph on every step (both regimes), so the
        # caller must not also run the eager plane upkeep
        self.qsa = qsa_layer is not None
        if self.qsa:
            idx = module.qsa_indexer
            self.qsa_idx = idx
            self.qsa_layer = qsa_layer
            self.qsa_threshold = idx.sparse_threshold()
            self.bc.set_qsa(
                qk_proj = idx.index_qk_proj.inner.bc,
                q_norm_w = idx.q_layernorm.weight.data.half().contiguous(),
                k_norm_w = idx.k_layernorm.weight.data.half().contiguous(),
                norm_eps = idx.q_layernorm.rms_norm_eps,
                n_heads = idx.n_heads,
                head_dim = idx.head_dim,
                topk = idx.block_topk,
                compress_ratio = idx.compress_ratio,
                raw_plane = qsa_layer.raw_k.view(-1, idx.head_dim),
                pool_plane = qsa_layer.pooled.view(-1, idx.head_dim),
            )
        self.slot_widths = {}

    def _configure(self, bsz: int, q_len: int, causal: bool, regime: int):
        import triton
        from .triton_paged import (
            _paged_attn_decode_split_kernel,
            _paged_attn_decode_combine_kernel,
            _paged_kv_update_kernel,
            _normalize_window,
        )

        dev = self.device
        hd = self.head_dim
        qh, kvh = self.num_q_heads, self.num_kv_heads
        group_size = qh // kvh

        block_n = max(16, 8192 // hd)
        block_m = triton.next_power_of_2(q_len)
        block_h = max(16 // block_m, 1)
        block_rows = block_m * block_h
        h_blocks = triton.cdiv(group_size, block_h)
        programs = bsz * kvh * h_blocks

        # The live split count and split length are runtime kernel arguments derived from the
        # block-table bound per call (patched into the graph); the grid is sized to the cap
        target = 2 * _get_sm_count(dev)
        splits_cap = max(1, min(target // programs, 128))
        window_left, window_right = _normalize_window(self.window_size)

        cache_t = "*i32" if self.quant else "*fp16"
        sig = {
            "q": "*fp16", "k_cache": cache_t, "v_cache": cache_t,
            "block_table": "*i32", "cache_seqlens": "*i32", "out": "*fp16",
            "partial_o": "*fp32", "partial_ml": "*fp32",
            "k_scales": "*fp16", "v_scales": "*fp16", "h32": "*fp16",
            "split_len": "i32", "num_pages_per_seq": "i32", "num_splits": "i32",
            "sinks": "*fp32",
        } | {n: "constexpr" for n in (
            "QCK", "QCV", "q_len", "kv_append_len", "n_q_heads", "n_kv_heads",
            "page_size", "head_dim", "scale", "CAUSAL", "WINDOW_LEFT", "WINDOW_RIGHT",
            "SOFTCAP", "FINAL", "HAS_SINKS", "BLOCK_M", "BLOCK_H", "BLOCK_ROWS", "BLOCK_N")}
        consts = dict(
            QCK = self.k_bits, QCV = self.v_bits,
            q_len = q_len, kv_append_len = q_len, n_q_heads = qh, n_kv_heads = kvh,
            page_size = PAGE_SIZE, head_dim = hd, scale = float(self.sm_scale),
            CAUSAL = bool(causal), WINDOW_LEFT = window_left, WINDOW_RIGHT = window_right,
            SOFTCAP = float(self.softcap or 0.0), FINAL = False, HAS_SINKS = False,
            BLOCK_M = block_m, BLOCK_H = block_h, BLOCK_ROWS = block_rows, BLOCK_N = block_n,
        )
        k_split = _compile_kernel(dev, _paged_attn_decode_split_kernel, sig, consts, 4, 2)

        sig_c = {
            "partial_o": "*fp32", "partial_ml": "*fp32", "out": "*fp16", "h32": "*fp16",
            "num_splits": "i32", "sinks": "*fp32",
        } | {n: "constexpr" for n in (
            "QCV", "HAS_SINKS", "q_len", "n_q_heads", "n_kv_heads", "head_dim",
            "BLOCK_M", "BLOCK_H", "BLOCK_ROWS")}
        consts_c = dict(
            QCV = self.v_bits, HAS_SINKS = self.sinks is not None, q_len = q_len,
            n_q_heads = qh, n_kv_heads = kvh, head_dim = hd,
            BLOCK_M = block_m, BLOCK_H = block_h, BLOCK_ROWS = block_rows,
        )
        k_combine = _compile_kernel(dev, _paged_attn_decode_combine_kernel, sig_c, consts_c, 4, 1)

        k_update = None
        if not self.quant:
            sig_u = {
                "k": "*fp16", "v": "*fp16", "k_cache": "*fp16", "v_cache": "*fp16",
                "block_table": "*i32", "cache_seqlens": "*i32", "num_pages_per_seq": "i32",
            } | {n: "constexpr" for n in (
                "kv_append_len", "n_kv_heads", "page_size", "head_dim", "BLOCK_D")}
            consts_u = dict(
                kv_append_len = q_len, n_kv_heads = kvh, page_size = PAGE_SIZE,
                head_dim = hd, BLOCK_D = triton.next_power_of_2(hd),
            )
            k_update = _compile_kernel(dev, _paged_kv_update_kernel, sig_u, consts_u, 2, 3)

        # Static intermediates, shared between layers with the same shapes on the same device
        R = bsz * q_len
        gate_a = gate_b = None
        if self.gate_mode == 2:
            # Full gate: one (2, R, n) buffer; q is its first slice so the fused q+g mgemm can
            # write both halves in one pass
            gate_a = g_tensor_cache.get(dev, (2, R, qh * hd), torch.half, "bca_qg")
            q = gate_a[0].view(bsz, q_len, qh, hd)
        else:
            q = g_tensor_cache.get(dev, (bsz, q_len, qh, hd), torch.half, "bca_q")
            if self.gate_mode == 1:
                # Headwise gate: one fp16 value per head
                gate_a = g_tensor_cache.get(dev, (R, qh), torch.half, "bca_gh")
            elif self.gate_mode == 3:
                gate_a = g_tensor_cache.get(dev, (R, 2 * qh * hd), torch.half, "bca_qgi")
                gate_b = g_tensor_cache.get(dev, (R, qh * hd), torch.half, "bca_g")
        kv = g_tensor_cache.get(dev, (2, R, kvh * hd), torch.half, "bca_kv")
        o = g_tensor_cache.get(dev, (bsz, q_len, qh, hd), torch.half, "bca_o")
        # Regime-1 slots never launch the dense split/combine; their partials are sized by the
        # sparse kernels in _configure_qsa (same bucketed tags, so the footprint is the max)
        pn_o = programs * splits_cap * block_rows * hd
        pn_ml = programs * splits_cap * block_rows * 2
        if regime == 1:
            pn_o, pn_ml = self._qsa_partial_sizes(bsz * q_len)
        partial_o = g_tensor_cache.get_bucketed(dev, pn_o, torch.float, "bca_po")
        partial_ml = g_tensor_cache.get_bucketed(dev, pn_ml, torch.float, "bca_ml")

        # Padded hidden dim: zero-padded input staging and padded o_proj output. The pad columns
        # of xp are zeroed here and never written afterwards (the graph copies only the exact
        # hidden width into it). An fp16 gate also stages the input (static operand for the
        # captured cublas node), without the padded output
        xp, yp = None, None
        if self.hidden_padded != self.hidden_size or self.g_weight is not None:
            xp = g_tensor_cache.get(dev, (R, self.hidden_padded), torch.half, "bca_xp")
            xp.zero_()
        if self.hidden_padded != self.hidden_size:
            yp = g_tensor_cache.get(dev, (R, self.hidden_padded), self.o_dtype or torch.half, "bca_yp")

        self.bc.configure_slot(
            bsz, q_len, regime,
            q, kv, o, partial_o, partial_ml,
            gate_a, gate_b,
            k_split, k_combine, k_update,
            block_n, splits_cap,
            xp, yp,
        )

        if self.qsa:
            self._configure_qsa(bsz, q_len, regime)

    # ---- QSA -----------------------------------------------------------------------------------

    def _qsa_sparse_geometry(self, rows: int):
        """Static selection/gather widths and the sparse split configuration; rows = bsz *
        q_len query rows (the gather kernel's batch axis)."""
        import triton
        idx = self.qsa_idx
        cr = idx.compress_ratio
        sel = idx.block_topk
        k_pad = -(-(sel * cr + cr - 1) // 32) * 32
        kp_pool = -(-sel // 32) * 32
        group = self.num_q_heads // self.num_kv_heads
        block_h = 16
        h_blocks = triton.cdiv(group, block_h)
        programs = rows * self.num_kv_heads * h_blocks
        block_n = 32
        target = 2 * _get_sm_count(self.device)
        splits = max(1, min(target // programs, -(-k_pad // (4 * block_n)), 128))
        per_split = -(-k_pad // splits)
        split_len = -(-per_split // block_n) * block_n
        return k_pad, kp_pool, block_h, h_blocks, programs, block_n, splits, split_len

    def _qsa_partial_sizes(self, rows: int):
        k_pad, kp_pool, block_h, h_blocks, programs, block_n, splits, split_len = \
            self._qsa_sparse_geometry(rows)
        return programs * splits * block_h * self.head_dim, programs * splits * block_h * 2

    def _configure_qsa(self, bsz: int, q_len: int, regime: int):
        import triton
        from .mla_triton import _mla_plane_update_kernel
        from .dsa_triton import _dsa_indexer_fewq_kernel, _dsa_pool_expand_kernel
        from .qsa_triton import (
            _qsa_stage_kernel,
            _qsa_pool_update_kernel,
            _qsa_sparse_split_kernel,
        )
        from .triton_paged import _paged_attn_decode_combine_kernel

        dev = self.device
        idx = self.qsa_idx
        Hi, Di, cr = idx.n_heads, idx.head_dim, idx.compress_ratio
        R = bsz * q_len
        rope = self.module.rope
        # The rotary width (sin/cos table width): the C++ side narrows the query rope view to
        # it and the pool kernel rotates its leading segment
        rotate_dims = 2 * rope.inv_freq.numel()

        def sbuf(tag, *shape, dtype = torch.half):
            n = 1
            for sh in shape: n *= sh
            return g_tensor_cache.get_bucketed(dev, n, dtype, tag).view(*shape)

        qk = sbuf("bca_qsa_qk", R, (Hi + 1) * Di)
        q = sbuf("bca_qsa_q", R, Hi, Di)
        kraw = sbuf("bca_qsa_kraw", R, Di)

        k_stage = _compile_kernel(dev, _qsa_stage_kernel,
            {"qk": "*fp16", "q_norm_w": "*fp16", "q_out": "*fp16", "k_out": "*fp16", "R": "i32"}
            | {n: "constexpr" for n in ("eps", "H_i", "D")},
            dict(eps = float(idx.q_layernorm.rms_norm_eps), H_i = Hi, D = Di), 2, 1)

        k_raw_append = _compile_kernel(dev, _mla_plane_update_kernel,
            {"rows_new": "*fp16", "plane_cache": "*fp16", "block_table": "*i32",
             "cache_seqlens": "*i32", "num_pages_per_seq": "i32", "append_len": "i32"}
            | {n: "constexpr" for n in ("page_size", "D", "DST_D", "DST_OFF")},
            dict(page_size = PAGE_SIZE, D = Di, DST_D = 0, DST_OFF = 0), 2, 2)

        k_pool_update = _compile_kernel(dev, _qsa_pool_update_kernel,
            {"raw_plane": "*fp16", "pool_plane": "*fp16", "k_norm_w": "*fp16",
             "inv_freq": "*fp32", "block_table": "*i32", "cache_seqlens": "*i32",
             "num_pages_per_row": "i32", "append_len": "i32"}
            | {n: "constexpr" for n in (
                "page_size", "P", "D", "ROPE_R", "attn_factor", "eps", "MAXPOOLS")},
            dict(page_size = PAGE_SIZE, P = cr, D = Di, ROPE_R = rotate_dims,
                 attn_factor = float(rope.attn_factor),
                 eps = float(idx.k_layernorm.rms_norm_eps), MAXPOOLS = q_len // cr + 1), 2, 1)

        wts = scores = pool_idx = indices = None
        k_fewq = k_expand = k_sp_split = k_sp_combine = None
        fewq_gy = splits = split_len = programs = 0
        if regime == 1:
            k_pad, kp_pool, block_h, h_blocks, programs, block_n, splits, split_len = \
                self._qsa_sparse_geometry(R)
            wts = sbuf("bca_qsa_wts", R, Hi)
            wts.fill_(1.0)
            cap = self.qsa_layer.pooled.shape[0] * self.qsa_layer.pooled.shape[1]
            s_max = -(-cap // 128) * 128
            scores = sbuf("bca_qsa_scores", R, s_max)
            # The scoring kernel writes only [0, T); the warmup top-k scans the full static
            # width, so the tail must hold -inf from this one-time fill (the captured graph
            # patches the top-k scan width to T afterwards)
            scores.fill_(-float("inf"))
            pool_idx = sbuf("bca_qsa_pool_idx", R, kp_pool, dtype = torch.int32)
            indices = sbuf("bca_qsa_indices", R, k_pad, dtype = torch.int32)

            k_fewq = _compile_kernel(dev, _dsa_indexer_fewq_kernel,
                {"q_idx": "*fp16:16", "w": "*fp16:16", "k_idx": "*fp16:16",
                 "scores": "*fp16:16", "T": "i32", "R": "i32", "q_pos0": "i32",
                 "bound_max": "i32", "block_table": "*i32:16", "num_pages_per_row": "i32"}
                | {n: "constexpr" for n in (
                    "H_i", "H_pad", "D_i", "S_stride", "compress_rate", "scale", "BLOCK_N",
                    "SEQ", "MULTIROW", "EPP", "DEBUG_BOUNDS", "DEBUG_PAGES")},
                dict(H_i = Hi, H_pad = max(16, 1 << (Hi - 1).bit_length()), D_i = Di,
                     S_stride = s_max, compress_rate = cr, scale = float(idx.scale),
                     BLOCK_N = 128, SEQ = q_len, MULTIROW = 0, EPP = PAGE_SIZE // cr,
                     DEBUG_BOUNDS = 0, DEBUG_PAGES = 0), 8, 2)
            fewq_gy = s_max // 128

            k_expand = _compile_kernel(dev, _dsa_pool_expand_kernel,
                {"pool_idx": "*i32", "out": "*i32", "q_pos0": "i32"}
                | {n: "constexpr" for n in (
                    "P", "SEL", "K_pad", "KP_pool", "TAIL", "SEQ", "MULTIROW", "BLOCK")},
                dict(P = cr, SEL = idx.block_topk, K_pad = k_pad, KP_pool = kp_pool,
                     TAIL = 1, SEQ = q_len, MULTIROW = 0, BLOCK = 256), 4, 1)

            k_sp_split = _compile_kernel(dev, _qsa_sparse_split_kernel,
                {"q": "*fp16", "k_cache": "*fp16", "v_cache": "*fp16", "block_table": "*i32",
                 "indices": "*i32", "partial_o": "*fp32", "partial_ml": "*fp32",
                 "k_len": "i32", "num_pages_per_seq": "i32", "num_splits": "i32",
                 "split_len": "i32"}
                | {n: "constexpr" for n in (
                    "n_q_heads", "n_kv_heads", "page_size", "head_dim", "K_pad", "scale",
                    "BLOCK_H", "BLOCK_N", "PAGED")},
                dict(n_q_heads = self.num_q_heads, n_kv_heads = self.num_kv_heads,
                     page_size = PAGE_SIZE, head_dim = self.head_dim, K_pad = k_pad,
                     scale = float(self.sm_scale), BLOCK_H = block_h, BLOCK_N = block_n,
                     PAGED = 1),
                4, 2)

            k_sp_combine = _compile_kernel(dev, _paged_attn_decode_combine_kernel,
                {"partial_o": "*fp32", "partial_ml": "*fp32", "out": "*fp16", "h32": "*fp16",
                 "num_splits": "i32", "sinks": "*fp32"}
                | {n: "constexpr" for n in (
                    "QCV", "HAS_SINKS", "q_len", "n_q_heads", "n_kv_heads", "head_dim",
                    "BLOCK_M", "BLOCK_H", "BLOCK_ROWS")},
                # q_len 1: the sparse gather treats every query row as a batch (programs =
                # R * kv_heads * h_blocks), so the combine's output row is the batch index
                # alone -- compiling the true q_len here would scatter row r to row r * q_len
                dict(QCV = 0, HAS_SINKS = False, q_len = 1,
                     n_q_heads = self.num_q_heads, n_kv_heads = self.num_kv_heads,
                     head_dim = self.head_dim, BLOCK_M = 1, BLOCK_H = block_h,
                     BLOCK_ROWS = block_h), 4, 1)

        self.bc.configure_slot_qsa(
            bsz, q_len, regime,
            qk, q, kraw,
            k_stage, k_raw_append, k_pool_update,
            rotate_dims,
            wts, scores, pool_idx, indices,
            k_fewq, k_expand, k_sp_split, k_sp_combine,
            fewq_gy, splits, split_len, programs,
        )

    def step(
        self,
        x: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        position: int,
        positions: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        inv_freq: torch.Tensor | None,
        causal: bool = True,
        host_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        bsz, q_len, _ = x.shape

        # QSA regime: dense below the sparse threshold (top-k cannot exclude anything there, so
        # dense attention is exact), gathered sparse above it. Sparse slots are decode-only;
        # other shapes fall back to the eager path, which maintains the planes identically
        regime, t_total = 0, 0
        if self.qsa:
            assert host_seqlens is not None, "BC_Attention: QSA step requires host seqlens"
            t_total = int(host_seqlens.max().item()) + q_len
            if t_total > self.qsa_threshold:
                # sparse slots are single-job; every query row of the chunk gets its own
                # selection (MTP-verify shapes included)
                if bsz > 1 or not causal:
                    return None
                regime = 1
                # The generator usually passes positions as a tensor and leaves the scalar at
                # 0; the scoring/expansion bounds consume the scalar
                position = int(host_seqlens[0].item())

        # The captured graph freezes the inv_freq table geometry (table flag, stride, partial
        # head dim) and the causality of the attention kernels, so either changing means
        # reconfigure (in practice constant per model). Everything else that varies per call --
        # position source, block-table pointer/width, split configuration -- is a runtime
        # argument patched into the graph
        skey = (tuple(inv_freq.shape) if inv_freq is not None else None, causal)
        if self.slot_widths.get((bsz, q_len, regime), ...) != skey:
            self._configure(bsz, q_len, causal, regime)
            self.slot_widths[(bsz, q_len, regime)] = skey
        y = torch.empty((bsz, q_len, self.hidden_size), dtype = self.o_dtype, device = x.device)
        self.bc.run(bsz, q_len, x, y, cache_seqlens, block_table, position, positions,
                    position_ids, inv_freq, regime, t_total)
        return y


def _qsa_module_eligible(m):
    """QSA-indexer requirements (Qwen3.8-Flash-Next). A module WITH an indexer must be fully
    armable or the BC path declines outright: an unarmed graph would let the key planes go
    stale under dense decode and poison later sparse selection."""
    idx = getattr(m, "qsa_indexer", None)
    if idx is None:
        return True
    p = idx.index_qk_proj
    # The rotary width is the sin/cos table width, 2 * len(inv_freq) (rope_settings.rotate_dims
    # is the rotation-section count, not a width)
    rd = 2 * m.rope.inv_freq.numel() if (m.rope is not None and m.rope.inv_freq is not None) else 0
    return (
        m.rope is not None and
        p is not None and p.quant_type == "exl3" and p.inner.bc is not None and
        p.inner.bias is None and
        getattr(p, "out_features_unpadded", p.out_features) == p.out_features ==
            (idx.n_heads + 1) * idx.head_dim and
        idx.q_layernorm.weight is not None and idx.k_layernorm.weight is not None and
        idx.q_layernorm.constant_bias == 1.0 and idx.k_layernorm.constant_bias == 1.0 and
        _is_pow2(idx.head_dim) and
        # The pool-update kernel splits rows into (rope lo, rope hi, pass) segments, each a
        # power-of-two tl.arange
        rd % 2 == 0 and 2 <= rd <= idx.head_dim and _is_pow2(rd // 2) and
        (idx.head_dim == rd or _is_pow2(idx.head_dim - rd)) and
        m.rope.rope_settings.rotate_dims == 1 and
        PAGE_SIZE % idx.compress_ratio == 0 and
        # The gathered sparse kernels support none of these
        (m.sliding_window is None or m.sliding_window < 0) and not m.logit_softcapping and
        getattr(m, "sinks", None) is None
    )


def _module_eligible(m):
    """Module-level requirements shared by the global-attention and SWA builders."""
    return (
        bc_attn_enable and
        _qsa_module_eligible(m) and
        # NoPE is supported (the rope stage is skipped), but the head norms run inside the rope
        # kernel, so a norm-only module without rope has nowhere to apply them
        (m.rope is not None or m.q_norm is None) and
        # Gates: interleaved, full and headwise are all graphed; the headwise (always-fp16)
        # projection runs as a captured cublas gemm over the statically staged input, so it
        # needs a weight that passes the fp16-gate checks
        (not m.headwise_gate or BCAttn._fp16_gate_weight(m.g_proj) is not None) and
        (not getattr(m, "interleaved_gate", False) or m.head_dim % 8 == 0) and
        (not m.full_gate or m.g_proj is None or m.multi_qg is not None or
            (m.g_proj.quant_type == "exl3" and m.g_proj.inner.bc is not None) or
            BCAttn._fp16_gate_weight(m.g_proj) is not None) and
        (m.v_norm is None or (type(m.v_norm).__name__ == "RMSNorm" and not m.v_norm.span_heads)) and
        # TP shards are eligible: the shard owns its split cache layers directly (the opaque cache
        # handle is resolved before bc_attn_step) and the output all-reduce runs after the captured
        # block returns. Span-heads norms stay declined (cross-rank norm inside the block)
        not getattr(m, "tp_span_heads_norm", False) and
        (m.q_norm is None or m.q_norm_tensor is not None) and
        _is_pow2(m.head_dim) and m.head_dim <= 512 and
        m.num_q_heads % m.num_kv_heads == 0 and
        # Padded dims: the projection inputs stage through a zero-padded static and the o_proj
        # output is trimmed, but the q/k/v/gate outputs and the o_proj input must be the exact
        # head dims (the attention statics are sized to them)
        all(p is None or getattr(p, "out_features_unpadded", None) in (None, p.out_features)
            for p in (m.q_proj, getattr(m, "k_proj", None), getattr(m, "v_proj", None),
                      getattr(m, "g_proj", None))) and
        (not hasattr(m.o_proj, "in_features_unpadded") or
            m.o_proj.in_features == m.o_proj.in_features_unpadded) and
        m.q_proj is not None and m.q_proj.quant_type == "exl3" and m.q_proj.inner.bc is not None and
        m.o_proj is not None and m.o_proj.quant_type == "exl3" and m.o_proj.inner.bc is not None and
        (m.multi_kv is not None or (
            m.k_proj is not None and m.k_proj.quant_type == "exl3" and m.k_proj.inner.bc is not None and
            (getattr(m, "use_k_as_v", False) or (
                m.v_proj is not None and m.v_proj.quant_type == "exl3" and m.v_proj.inner.bc is not None
            ))
        ))
    )


def build_bc_swa(module, layer_state):
    """Build a BCAttn over a sliding-window recurrent state: the per-slot fp16 K/V span pool
    viewed as a paged cache (state size is a multiple of the page size, so the geometry is fixed
    and each slot captures exactly once). Returns None when unsupported."""
    m = module
    k_states, v_states = layer_state.get_state_tensors()
    if not (
        _module_eligible(m) and
        getattr(m, "qsa_indexer", None) is None and
        k_states is not None and
        k_states.device == torch.device(m.device) and
        m.kv_state_size % PAGE_SIZE == 0
    ):
        _trace_build(m, None, "swa")
        return None
    k_pages = k_states.view(-1, PAGE_SIZE, m.num_kv_heads, m.head_dim)
    v_pages = v_states.view(-1, PAGE_SIZE, m.num_kv_heads, m.head_dim)
    bca = BCAttn(m, k_pages, v_pages)
    _trace_build(m, bca, "swa")
    return bca


def build_bc_attn(module, layer):
    """Build a BCAttn for the module/cache-layer pair, or return None when the configuration
    is not supported (caller falls back to the dispatch path)."""
    from ...cache import CacheLayer_quant, CacheLayer_fp16
    from ...cache.qsa import CacheLayer_qsa

    m = module
    qsa_idx = getattr(m, "qsa_indexer", None)
    if not (
        _module_eligible(m) and
        isinstance(layer, (CacheLayer_quant, CacheLayer_fp16)) and
        (not isinstance(layer, CacheLayer_quant) or (
            layer.compand_a == 0.0 and layer.qk is not None and
            layer.qk.device == torch.device(m.device)
        )) and
        (not isinstance(layer, CacheLayer_fp16) or (
            layer.k is not None and layer.k.device == torch.device(m.device)
        )) and
        # A QSA module needs the side planes on this layer (and the fp16 cache they imply)
        (qsa_idx is None or (
            isinstance(layer, CacheLayer_qsa) and layer.raw_k is not None and
            layer.raw_k.device == torch.device(m.device)
        ))
    ):
        _trace_build(m, None, "attn")
        return None
    if isinstance(layer, CacheLayer_quant):
        bca = BCAttn(m, layer.qk, layer.qv, layer.sk, layer.sv, layer.k_bits, layer.v_bits)
    else:
        bca = BCAttn(m, layer.k, layer.v,
                     qsa_layer = layer if qsa_idx is not None else None)
    _trace_build(m, bca, "attn")
    return bca
