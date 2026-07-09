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

EXL3_BC_ATTN=0 disables the path.
"""

bc_attn_enable = os.environ.get("EXL3_BC_ATTN", "0") != "0"
bc_attn_debug = os.environ.get("EXL3_BC_ATTN_DEBUG", "0") != "0"

MAX_BSZ = 4
MAX_QLEN = 16
MAX_R = MAX_BSZ * MAX_QLEN

_kernel_cache = {}
_sm_count = {}


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _compile_kernel(device: torch.device, fn, signature: dict, constexprs: dict,
                    num_warps: int, num_stages: int):
    key = (device.index, fn.__name__, tuple(sorted(constexprs.items())), num_warps, num_stages)
    k = _kernel_cache.get(key)
    if k is None:
        import triton
        from triton.compiler import ASTSource
        with torch.cuda.device(device):
            src = ASTSource(fn = fn, signature = signature, constexprs = constexprs)
            ck = triton.compile(src, options = {"num_warps": num_warps, "num_stages": num_stages})
            k = ext.TritonKernel(ck.asm["cubin"], ck.metadata.name, ck.metadata.num_warps, ck.metadata.shared)
        _kernel_cache[key] = k
    return k


def _get_sm_count(device: torch.device) -> int:
    if device.index not in _sm_count:
        _sm_count[device.index] = torch.cuda.get_device_properties(device).multi_processor_count
    return _sm_count[device.index]


class BCAttn:
    """Python-side owner of one ext.BC_Attention (per attention module and cache layer):
    collects the projection/norm/rope/cache handles at construction and compiles + registers the
    per-slot attention kernels lazily."""

    @staticmethod
    def _has_bc(proj):
        return proj is not None and proj.quant_type == "exl3" and proj.inner.bc is not None

    def __init__(self, module, layer):
        from ...cache import CacheLayer_quant

        self.module = module
        self.device = module.device
        self.head_dim = module.head_dim
        self.num_q_heads = module.num_q_heads
        self.num_kv_heads = module.num_kv_heads
        self.hidden_size = module.hidden_size
        self.sm_scale = module.sm_scale
        self.window_size = module.sliding_window
        self.softcap = module.logit_softcapping

        self.quant = isinstance(layer, CacheLayer_quant)
        if self.quant:
            self.cache_k, self.cache_v = layer.qk, layer.qv
            self.k_scales, self.v_scales = layer.sk, layer.sv
            self.k_bits, self.v_bits = layer.k_bits, layer.v_bits
            from .triton_paged import _get_h32
            h32 = _get_h32(self.device)
        else:
            self.cache_k, self.cache_v = layer.k, layer.v
            self.k_scales = self.v_scales = None
            self.k_bits = self.v_bits = 0
            h32 = g_tensor_cache.get(self.device, (1,), torch.half, "bca_dummy")
        self.max_pages = self.cache_k.shape[0]

        rope = module.rope
        mkv = module.multi_kv
        mqg = module.multi_qg
        w = max(self.hidden_size, self.num_q_heads * self.head_dim)
        xh = g_tensor_cache.get(self.device, (2 * MAX_R * w,), torch.half, "bca_xh")

        self.o_dtype = module.o_proj.inner.default_out_dtype
        self.use_k_as_v = module.use_k_as_v

        # 0 = none, 2 = full (o *= sigmoid(g)), 3 = interleaved q/g projection. Headwise gates
        # (fp16 projection through cublas, no patchable sites) are rejected by build_bc_attn
        if module.interleaved_gate:
            self.gate_mode = 3
        elif module.g_proj is not None:
            self.gate_mode = 2
        else:
            self.gate_mode = 0

        self.bc = ext.BC_Attention(
            num_q_heads = self.num_q_heads,
            num_kv_heads = self.num_kv_heads,
            head_dim = self.head_dim,
            hidden_size = self.hidden_size,
            page_size = PAGE_SIZE,
            q_proj = module.q_proj.inner.bc,
            k_proj = module.k_proj.inner.bc if self._has_bc(module.k_proj) else None,
            v_proj = module.v_proj.inner.bc if (not module.use_k_as_v and self._has_bc(module.v_proj)) else None,
            kv_ptrs_trellis = mkv.ptrs_trellis if mkv is not None else None,
            kv_ptrs_suh = mkv.ptrs_suh if mkv is not None else None,
            kv_ptrs_svh = mkv.ptrs_svh if mkv is not None else None,
            kv_K = mkv.K if mkv is not None else 0,
            kv_mcg = bool(mkv.mcg) if mkv is not None else False,
            kv_mul1 = bool(mkv.mul1) if mkv is not None else False,
            o_proj = module.o_proj.inner.bc,
            use_k_as_v = self.use_k_as_v,
            gate_mode = self.gate_mode,
            g_proj = module.g_proj.inner.bc if (self.gate_mode == 2 and self._has_bc(module.g_proj)) else None,
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
            inv_freq = rope.inv_freq,
            rope_style = int(rope.rope_settings.rope_style),
            attn_factor = rope.attn_factor,
            l4_scaling_beta = rope.llama_4_scaling_beta,
            l4_scaling_original = rope.llama_4_scaling_original,
            post_rope_norm = module.post_rope_norm,
            rotate_dims = rope.rope_settings.rotate_dims,
            quant_cache = self.quant,
            cache_k = self.cache_k,
            cache_v = self.cache_v,
            cache_k_scales = self.k_scales,
            cache_v_scales = self.v_scales,
            xh = xh,
            h32 = h32,
        )
        self.slot_widths = {}

    def _configure(self, bsz: int, q_len: int, bt_width: int):
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

        max_k_len = bt_width * PAGE_SIZE + q_len
        target = 2 * _get_sm_count(dev)
        num_splits = max(1, min(target // programs, triton.cdiv(max_k_len, 4 * block_n), 128))
        split_len = triton.cdiv(triton.cdiv(max_k_len, num_splits), block_n) * block_n
        window_left, window_right = _normalize_window(self.window_size)

        cache_t = "*i32" if self.quant else "*fp16"
        sig = {
            "q": "*fp16", "k_cache": cache_t, "v_cache": cache_t,
            "block_table": "*i32", "cache_seqlens": "*i32", "out": "*fp16",
            "partial_o": "*fp32", "partial_ml": "*fp32",
            "k_scales": "*fp16", "v_scales": "*fp16", "h32": "*fp16",
            "split_len": "i32", "num_pages_per_seq": "i32",
        } | {n: "constexpr" for n in (
            "num_splits", "QCK", "QCV", "q_len", "kv_append_len", "n_q_heads", "n_kv_heads",
            "page_size", "head_dim", "scale", "CAUSAL", "WINDOW_LEFT", "WINDOW_RIGHT",
            "SOFTCAP", "FINAL", "BLOCK_M", "BLOCK_H", "BLOCK_ROWS", "BLOCK_N")}
        consts = dict(
            num_splits = num_splits, QCK = self.k_bits, QCV = self.v_bits,
            q_len = q_len, kv_append_len = q_len, n_q_heads = qh, n_kv_heads = kvh,
            page_size = PAGE_SIZE, head_dim = hd, scale = float(self.sm_scale),
            CAUSAL = True, WINDOW_LEFT = window_left, WINDOW_RIGHT = window_right,
            SOFTCAP = float(self.softcap or 0.0), FINAL = num_splits == 1,
            BLOCK_M = block_m, BLOCK_H = block_h, BLOCK_ROWS = block_rows, BLOCK_N = block_n,
        )
        k_split = _compile_kernel(dev, _paged_attn_decode_split_kernel, sig, consts, 4, 2)

        k_combine = None
        if num_splits > 1:
            sig_c = {
                "partial_o": "*fp32", "partial_ml": "*fp32", "out": "*fp16", "h32": "*fp16",
            } | {n: "constexpr" for n in (
                "num_splits", "QCV", "q_len", "n_q_heads", "n_kv_heads", "head_dim",
                "BLOCK_M", "BLOCK_H", "BLOCK_ROWS")}
            consts_c = dict(
                num_splits = num_splits, QCV = self.v_bits, q_len = q_len,
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
            if self.gate_mode == 3:
                gate_a = g_tensor_cache.get(dev, (R, 2 * qh * hd), torch.half, "bca_qgi")
                gate_b = g_tensor_cache.get(dev, (R, qh * hd), torch.half, "bca_g")
        kv = g_tensor_cache.get(dev, (2, R, kvh * hd), torch.half, "bca_kv")
        o = g_tensor_cache.get(dev, (bsz, q_len, qh, hd), torch.half, "bca_o")
        if num_splits > 1:
            partial_o = g_tensor_cache.get(dev, (programs * num_splits * block_rows * hd,), torch.float, "bca_po")
            partial_ml = g_tensor_cache.get(dev, (programs * num_splits * block_rows * 2,), torch.float, "bca_ml")
        else:
            partial_o = g_tensor_cache.get(dev, (1,), torch.float, "bca_po1")
            partial_ml = g_tensor_cache.get(dev, (1,), torch.float, "bca_ml1")

        self.bc.configure_slot(
            bsz, q_len, bt_width,
            q, kv, o, partial_o, partial_ml,
            gate_a, gate_b,
            k_split, k_combine, k_update,
            num_splits, split_len,
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
    ) -> torch.Tensor:
        bsz, q_len, _ = x.shape
        bt_width = block_table.shape[1]
        # The captured graph freezes the block-table width and the inv_freq table geometry
        # (table flag, stride, partial head dim); either changing means reconfigure. The
        # position source (scalar/positions/position_ids) is a runtime branch, patched per call
        skey = (bt_width, tuple(inv_freq.shape) if inv_freq is not None else None)
        if self.slot_widths.get((bsz, q_len)) != skey:
            self._configure(bsz, q_len, bt_width)
            self.slot_widths[(bsz, q_len)] = skey
        y = torch.empty((bsz, q_len, self.hidden_size), dtype = self.o_dtype, device = x.device)
        self.bc.run(bsz, q_len, x, y, cache_seqlens, block_table, position, positions, position_ids, inv_freq)
        return y


def build_bc_attn(module, layer):
    """Build a BCAttn for the module/cache-layer pair, or return None when the configuration
    is not supported (caller falls back to the dispatch path)."""
    from ...cache import CacheLayer_quant, CacheLayer_fp16

    m = module
    try:
        if not (
            bc_attn_enable and
            isinstance(layer, (CacheLayer_quant, CacheLayer_fp16)) and
            (not isinstance(layer, CacheLayer_quant) or (
                layer.compand_a == 0.0 and layer.qk is not None and
                layer.qk.device == torch.device(m.device)
            )) and
            (not isinstance(layer, CacheLayer_fp16) or (
                layer.k is not None and layer.k.device == torch.device(m.device)
            )) and
            m.rope is not None and
            m.rope_settings is not None and
            # Gates: interleaved and full are graphed; headwise (fp16 gate projection through
            # cublas, no patchable sites) is not
            not m.headwise_gate and
            not m.ve_gate and
            (not m.interleaved_gate or m.head_dim % 8 == 0) and
            (not m.full_gate or m.g_proj is None or m.multi_qg is not None or
                (m.g_proj.quant_type == "exl3" and m.g_proj.inner.bc is not None)) and
            (m.v_norm is None or (type(m.v_norm).__name__ == "RMSNorm" and not m.v_norm.span_heads)) and
            not m.tp_reduce and not m.has_split_cache and not m.tp_span_heads_norm and
            (m.q_norm is None or m.q_norm_tensor is not None) and
            _is_pow2(m.head_dim) and m.head_dim <= 512 and
            m.num_q_heads % m.num_kv_heads == 0 and
            m.q_proj is not None and m.q_proj.quant_type == "exl3" and m.q_proj.inner.bc is not None and
            m.o_proj is not None and m.o_proj.quant_type == "exl3" and m.o_proj.inner.bc is not None and
            (m.multi_kv is not None or (
                m.k_proj is not None and m.k_proj.quant_type == "exl3" and m.k_proj.inner.bc is not None and
                (m.use_k_as_v or (
                    m.v_proj is not None and m.v_proj.quant_type == "exl3" and m.v_proj.inner.bc is not None
                ))
            ))
        ):
            return None
        return BCAttn(m, layer)
    except Exception:
        if bc_attn_debug:
            raise
        return None
