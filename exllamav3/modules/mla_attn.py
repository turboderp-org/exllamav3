from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config
from ..util.rope import RopeSettings, RoPE
from ..util.tensor import get_for_device, to2
from . import Module, Linear, RMSNorm
from ..model.model_tp_alloc import TPAllocation
from .attention_fn.mla_triton import (
    mla_attn_triton_decode,
    mla_attn_triton_prefill,
    mla_attn_triton_prefill_mha,
    mla_absorb,
    mla_unfold,
    has_triton,
)
from .attention_fn.bc_attn import MAX_BSZ as _bc_max_bsz
import os

# Prefill strategy: "mha" (default) up-projects past tiles from the compressed cache and attends
# in MHA form. ~2.8x fewer FLOPs than running the absorbed form over the whole context. "absorbed"
# restores the single-kernel absorbed prefill for A/B testing
_prefill_mode = os.environ.get("EXL3_MLA_PREFILL", "mha")

# Query lengths at or below this use the flash-decoding kernel (kv split across programs);
# above it, the long-query kernel (q split across programs) wins
MAX_DECODE_QLEN = 16


class MLAttention(Module):
    """
    Multi-head latent attention (DeepSeek-V2/V3, Kimi-Linear).

    Attention runs in absorbed form end to end. The cache holds only the compressed latent and the
    shared RoPE key: 576 values per token, against 81920 for the equivalent expanded K/V of a
    128-head model. Per-head K and V are never materialized, in prefill or decode. What makes
    that possible is folding the kv_b up-projection into the query and the output instead:

        scores = (q_nope @ W_UK) . c_kv  +  q_pe . k_pe
        o      = (softmax(scores) @ c_kv) @ W_UV

    Both folds are batched GEMMs over the head axis, so the queries and the attention output are
    kept head-major throughout and the two kv_lora_rank-wide tensors are never permuted.

    W_UK and W_UV stay unquantized. They are pure weight streaming (measured at 76-81% of memory
    peak), the absorb is a bmm rather than a GEMM so the exl3 kernels do not apply, and folding
    W_UK into q_b_proj instead would triple that projection (strictly worse than the 33 MB per
    layer this costs on a 128-head model.)
    """

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
        sm_scale: float | None = None,
        rms_norm_eps: float = 1e-6,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        key_q: str = "q_proj",
        key_q_a: str = "q_a_proj",
        key_q_b: str = "q_b_proj",
        key_q_a_norm: str = "q_a_layernorm",
        key_kv_a: str = "kv_a_proj_with_mqa",
        key_kv_a_norm: str = "kv_a_layernorm",
        key_kv_b: str = "kv_b_proj",
        key_o: str = "o_proj",
        qbits_key: str = "bits",
        select_hq_bits: int = 0,
    ):
        super().__init__(config, key, None)

        self.q_priority = 2 + select_hq_bits
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = 1
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.rope_settings = rope_settings
        self.rope = None
        self.out_dtype = out_dtype
        self.key_kv_b = key_kv_b
        self.norm_eps = rms_norm_eps

        # The softmax scale follows the *unabsorbed* head dim: absorption does not change the
        # scores, only how they are computed. Architectures with YaRN mscale pass their own
        self.sm_scale = sm_scale if sm_scale is not None else self.qk_head_dim ** -0.5

        # head_dim is what the rest of the stack reads for reporting and allocation; the latent
        # width is what actually lands in the cache
        self.head_dim = self.qk_head_dim

        qmap_in = qmap + ".input" if qmap is not None else None
        qmap_o = qmap + ".o" if qmap is not None else None

        # Query path: either a direct projection or a LoRA-style pair with a norm between
        if q_lora_rank is None:
            self.q_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
            self.q_proj = Linear(
                config, f"{key}.{key_q}", hidden_size, num_q_heads * self.qk_head_dim,
                qmap = qmap_in, out_dtype = torch.half, trim_padded_out = True,
                select_hq_bits = select_hq_bits, qbits_key = qbits_key,
            )
            self.register_submodule(self.q_proj)
        else:
            self.q_a_proj = Linear(
                config, f"{key}.{key_q_a}", hidden_size, q_lora_rank,
                qmap = qmap_in, out_dtype = torch.half, trim_padded_out = True,
                select_hq_bits = select_hq_bits, qbits_key = qbits_key,
            )
            self.q_a_layernorm = RMSNorm(
                config, f"{key}.{key_q_a_norm}", rms_norm_eps, out_dtype = torch.half,
            )
            self.q_b_proj = Linear(
                config, f"{key}.{key_q_b}", q_lora_rank, num_q_heads * self.qk_head_dim,
                qmap = qmap + ".q_a" if qmap is not None else None,
                out_dtype = torch.half, trim_padded_out = True,
                select_hq_bits = select_hq_bits, qbits_key = qbits_key,
            )
            self.q_proj = self.q_b_proj
            self.register_submodule(self.q_a_proj)
            self.register_submodule(self.q_a_layernorm)
            self.register_submodule(self.q_b_proj)

        # Latent path. The output of this projection goes straight into the cache, so it is the
        # one place where quantization error compounds over the whole context
        self.kv_a_proj_with_mqa = Linear(
            config, f"{key}.{key_kv_a}", hidden_size, kv_lora_rank + qk_rope_head_dim,
            qmap = qmap_in, out_dtype = torch.half, trim_padded_out = True,
            select_hq_bits = select_hq_bits, qbits_key = qbits_key,
        )
        self.kv_a_layernorm = RMSNorm(
            config, f"{key}.{key_kv_a_norm}", rms_norm_eps, out_dtype = torch.half,
        )
        self.register_submodule(self.kv_a_proj_with_mqa)
        self.register_submodule(self.kv_a_layernorm)

        self.o_proj = Linear(
            config, f"{key}.{key_o}", num_q_heads * v_head_dim, hidden_size,
            qmap = qmap_o, out_dtype = out_dtype, trim_padded_out = True,
            select_hq_bits = select_hq_bits, qbits_key = qbits_key,
        )
        self.register_submodule(self.o_proj)

        self.caps.update({
            "kv_cache": True
        })

        self.cache_layers = []
        self.tp_cache_lookup = {}
        self.has_split_cache = False
        self.dispatch_cache = {}

        # kv_b_proj, stored ONLY in the flattened (kv_lora_rank, H * dim) form: the prefill
        # up-projection GEMMs consume it directly, and the decode absorb/unfold run as Triton
        # kernels that read per-head column blocks out of the same layout - one resident copy
        # serves every path, and no cuBLAS batched GEMM is involved anywhere in the module
        self.w_uk_flat = None   # (kv_lora_rank, H * qk_nope_head_dim)
        self.w_uv_flat = None   # (kv_lora_rank, H * v_head_dim)
        self._scratch = {}


    def cache_layer_type(self, default, kwargs: dict):
        """MLA stores a latent instead of per-head K/V, so it overrides the cache layer the Cache
        was constructed with. A quantized cache request maps to the packed-latent layer: k_bits
        sets the latent width, the shared rope key stays fp16 (v_bits is accepted but unused -
        there is no separate V)."""
        from ..cache import CacheLayer_fp16, CacheLayer_MLA_fp16, CacheLayer_quant, CacheLayer_MLA_quant
        if issubclass(default, CacheLayer_quant):
            return CacheLayer_MLA_quant, kwargs
        if issubclass(default, CacheLayer_fp16):
            return CacheLayer_MLA_fp16, {}
        raise NotImplementedError(
            f"{default.__name__} is not supported for MLA layers; use CacheLayer_fp16 or "
            f"CacheLayer_quant"
        )


    @override
    def optimizer_targets(self):
        q = (self.q_a_proj.optimizer_targets() if self.q_a_proj else []) + \
            self.q_proj.optimizer_targets()
        kv = self.kv_a_proj_with_mqa.optimizer_targets()
        o = self.o_proj.optimizer_targets()
        return [[q, kv, o]]


    def load_local(self, device, **kwargs):
        for cl in self.cache_layers:
            cl.alloc(device)

        if self.rope_settings:
            self.rope = RoPE(device, self.rope_settings)

        # kv_b_proj maps the latent to per-head K-nope and V. Attention never applies it as that
        # GEMM; the halves fold into the query/output (decode) or up-project past tiles (prefill)
        w = self.config.stc.get_tensor(f"{self.key}.{self.key_kv_b}.weight", device, no_defer = True)
        assert w.shape == (self.num_q_heads * (self.qk_nope_head_dim + self.v_head_dim),
                           self.kv_lora_rank), \
            f"{self.key}.{self.key_kv_b}: unexpected shape {tuple(w.shape)}"
        H, nope, v, D_c = self.num_q_heads, self.qk_nope_head_dim, self.v_head_dim, self.kv_lora_rank
        w = w.view(H, nope + v, D_c)
        self.w_uk_flat = torch.empty((D_c, H * nope), dtype = torch.half, device = device)
        self.w_uk_flat.view(D_c, H, nope).copy_(w[:, :nope, :].permute(2, 0, 1))
        self.w_uv_flat = torch.empty((D_c, H * v), dtype = torch.half, device = device)
        self.w_uv_flat.view(D_c, H, v).copy_(w[:, nope:, :].permute(2, 0, 1))


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)
        self.load_local(device, **kwargs)


    @override
    def get_tensors(self):
        # kv_b stays unquantized, so it is carried into the converted model; reconstruct the
        # checkpoint layout from the flats (bf16 -> fp16 is value-exact at weight magnitudes)
        t = {}
        if self.w_uk_flat is not None:
            H, nope, v = self.num_q_heads, self.qk_nope_head_dim, self.v_head_dim
            D_c = self.kv_lora_rank
            uk = self.w_uk_flat.view(D_c, H, nope).permute(1, 2, 0)
            uv = self.w_uv_flat.view(D_c, H, v).permute(1, 2, 0)
            t[f"{self.key}.{self.key_kv_b}.weight"] = \
                torch.cat([uk, uv], dim = 1).reshape(H * (nope + v), D_c).contiguous()
        return t


    @override
    def unload(self):
        super().unload()
        for cl in self.cache_layers:
            cl.free()
        self.rope = None
        self.w_uk_flat = None
        self.w_uv_flat = None
        self._scratch = {}
        self.dispatch_cache = {}


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        attn_mode = params.get("attn_mode", "flash_attn_nc")
        match attn_mode:
            case "flash_attn":
                x = self.decode_flash_attn(x, bsz, seqlen, params)
            case "flash_attn_nc":
                x = self.decode_flash_attn_nc(x, bsz, seqlen, params)
            case _:
                raise ValueError(f"Unknown attn_mode: {attn_mode}")
        return to2(x, out_dtype, self.out_dtype)


    def project_q(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        if self.q_a_proj is None:
            q = self.q_proj.forward(x, params)
        else:
            q = self.q_a_proj.forward(x, params)
            q = self.q_a_layernorm.forward(q, params, out_dtype = torch.half)
            q = self.q_b_proj.forward(q, params)
        return q


    def _attend(self, x, bsz, seqlen, params, ckv_cache, kpe_cache, block_table, cache_seqlens,
                append, qc = None, host_seqlens = None):
        """Projections, absorption, attention and o_proj, shared by the cached and cache-less
        paths. `append` writes the new latent/rope rows into the supplied page tensors."""
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        inv_freq = get_for_device(params, "inv_freq", self.device, None)
        causal = params.get("causal", True)

        H = self.num_q_heads
        R = bsz * seqlen

        from .attention_fn.mla_triton import _dbg_sync

        # Queries
        _dbg_sync("attend-entry (upstream modules)", x.device)
        q = self.project_q(x, params).view(R, H, self.qk_head_dim)
        _dbg_sync("project_q", x.device)
        q_nope = q[:, :, :self.qk_nope_head_dim]
        q_pe = q[:, :, self.qk_nope_head_dim:].reshape(bsz, seqlen, H, self.qk_rope_head_dim)

        # Latent K/V. The normalized latent is what gets cached, matching the reference order
        # (kv_a_layernorm is applied before kv_b_proj would be)
        ckv_kpe = self.kv_a_proj_with_mqa.forward(x, params)
        _dbg_sync("kv_a_proj", x.device)
        ckv = self.kv_a_layernorm.forward(
            ckv_kpe[..., :self.kv_lora_rank].contiguous(), params, out_dtype = torch.half
        )
        k_pe = ckv_kpe[..., self.kv_lora_rank:].reshape(bsz, seqlen, 1, self.qk_rope_head_dim).contiguous()
        _dbg_sync("kv_a_layernorm+slice", x.device)

        if self.rope is not None:
            q_pe, k_pe = self.rope.apply(
                q_pe, k_pe, position, positions, position_ids, True,
                None, None, self.norm_eps, 0.0, inv_freq, False,
            )
            _dbg_sync("rope", x.device)

        use_mha = seqlen > MAX_DECODE_QLEN and _prefill_mode == "mha" and causal
        if not use_mha:
            # Absorb W_UK into the queries, per head, straight from the flat layout. This runs as
            # a Triton kernel rather than a cuBLAS batched GEMM: the strided-batched fp16 form
            # made cuBLAS pick an SM120 nvjet TMA kernel that intermittently MMU-faults (caught
            # with cuda-gdb after presenting as flaky illegal-memory-access crashes whose
            # incidence tracked allocation layout), and the kernel reads any strides for free
            q_lat = mla_absorb(q.view(R, H, self.qk_head_dim), self.w_uk_flat, H, self.qk_nope_head_dim)
            q_pe_hm = q_pe.reshape(R, H, self.qk_rope_head_dim).permute(1, 0, 2).contiguous()

        append(ckv, k_pe)

        if use_mha:
            # MHA-form prefill: everything (past and current chunk) is read back from the cache
            # and attended over per-head up-projections. RoPE produced q_pe as a copy (the strided
            # slice cannot reshape into a view), so fold it back into q's pe columns for the
            # kernel's packed [nope | pe] per-head rows
            q = q.view(R, H, self.qk_head_dim)
            q[:, :, self.qk_nope_head_dim:] = q_pe.reshape(R, H, self.qk_rope_head_dim)
            o = mla_attn_triton_prefill_mha(
                q,
                self.w_uk_flat, self.w_uv_flat,
                ckv_cache, kpe_cache, block_table, host_seqlens,
                bsz, seqlen, self.v_head_dim, self.qk_nope_head_dim, self.sm_scale,
                pre_appended_len = seqlen,
                qc = qc,
            )
            o = o.reshape(bsz, seqlen, H * self.v_head_dim)
            return self.o_proj.forward(o, params)

        kernel = mla_attn_triton_decode if seqlen <= MAX_DECODE_QLEN else mla_attn_triton_prefill
        extra = {}
        o_lat = kernel(
            q_lat, q_pe_hm, ckv_cache, kpe_cache, block_table, cache_seqlens,
            bsz = bsz, q_len = seqlen,
            causal = causal, softmax_scale = self.sm_scale,
            pre_appended_len = seqlen,
            qc = qc,
            **extra,
        )

        from .attention_fn.mla_triton import _debug_sync
        if _debug_sync:
            # NaN/Inf in the attention output would reach the MoE router of the next block, and a
            # top-k over non-finite logits can select garbage expert ids -> wild pointer loads
            if not torch.isfinite(o_lat).all():
                bad = (~torch.isfinite(o_lat)).sum().item()
                raise RuntimeError(
                    f"MLA debug: non-finite attention output, layer {self.layer_idx}, "
                    f"{bad}/{o_lat.numel()} elements, bsz={bsz} seqlen={seqlen} dev={o_lat.device}")

        # Unfold W_UV per head from the flat layout; the kernel emits token-major output, so it
        # feeds o_proj without a permute
        o = mla_unfold(o_lat, self.w_uv_flat, self.v_head_dim)
        o = o.reshape(bsz, seqlen, H * self.v_head_dim)
        return self.o_proj.forward(o, params)


    def decode_flash_attn(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        cache = params.get("cache")
        if self.has_split_cache:
            cache = self.tp_cache_lookup[cache]
        block_table = get_for_device(params, "block_table", self.device)
        cache_seqlens = get_for_device(params, "cache_seqlens", self.device)
        assert params.get("non_causal_spans") is None, \
            "MLAttention does not support non-causal spans"

        layer = cache if not hasattr(cache, "layers") else \
            cache.layers[self.layer_idx, params.get("layer_instance") or 0]

        # Graph-captured C++ path for the whole decode block (projections through o_proj as one
        # replayed CUDA graph). Falls back to the dispatch path for unsupported configurations
        if (
            seqlen <= MAX_DECODE_QLEN and bsz <= _bc_max_bsz and
            params.get("causal", True) and params.get("inv_freq") is None
        ):
            y = self.bc_mla_step(x, params, layer, block_table, cache_seqlens)
            if y is not None:
                return y

        from ..cache import CacheLayer_MLA_quant
        if isinstance(layer, CacheLayer_MLA_quant):
            # Packed latent feeds the kernels directly (online dequant, rotated domain); the rope
            # key pages are fp16 either way
            ckv_cache, sk, kpe_cache, bits = layer.get_qc()
            qc = (sk, bits)
        else:
            ckv_cache, kpe_cache = layer.get_kv(cache_seqlens, block_table)
            qc = None

        # Host-side lengths for the tiled prefill (one device sync per forward, shared across
        # layers via the params dict; prefill is not a graphed path)
        if seqlen > MAX_DECODE_QLEN:
            host_seqlens = params.get("_mla_host_seqlens")
            if host_seqlens is None:
                host_seqlens = params["_mla_host_seqlens"] = cache_seqlens.cpu().tolist()
        else:
            host_seqlens = None

        return self._attend(
            x, bsz, seqlen, params, ckv_cache, kpe_cache, block_table, cache_seqlens,
            append = lambda ckv, k_pe:
                layer.update_kv_direct(cache_seqlens, block_table, ckv, k_pe, seqlen),
            qc = qc,
            host_seqlens = host_seqlens,
        )


    def bc_mla_step(self, x, params, layer, block_table, cache_seqlens):
        """Graph-captured decode block, or None when the module/cache-layer pair is not
        supported (caller falls back to the dispatch path)."""
        key = ("bcm", id(layer))
        bcm = self.dispatch_cache.get(key)
        if bcm is None:
            from .attention_fn.bc_mla import build_bc_mla
            bcm = self.dispatch_cache[key] = (build_bc_mla(self, layer) or False)
        if bcm is False:
            return None
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        return bcm.step(
            x.contiguous(), cache_seqlens, block_table, position, positions, position_ids
        )


    def decode_flash_attn_nc(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        """Cache-less attention over the current chunk only, used by the quantization calibration
        pass. The chunk's own latent/rope rows go into a scratch page pool with an identity block
        table, so this runs the same kernels as the cached path rather than a separate variant."""
        from ..constants import PAGE_SIZE
        from .attention_fn.mla_triton import mla_kv_append

        assert params.get("cache") is None, "Cache provided for attn_mode: flash_attn_nc"

        pages = (seqlen + PAGE_SIZE - 1) // PAGE_SIZE
        dev = x.device
        # Only rows below seqlen are ever read (the kernel derives its bound from cache_seqlens +
        # pre_appended_len), so the page tail does not need initializing
        ckv_cache = torch.empty((bsz * pages, PAGE_SIZE, 1, self.kv_lora_rank),
                                dtype = torch.half, device = dev)
        kpe_cache = torch.empty((bsz * pages, PAGE_SIZE, 1, self.qk_rope_head_dim),
                                dtype = torch.half, device = dev)
        block_table = torch.arange(bsz * pages, dtype = torch.int32, device = dev).view(bsz, pages)
        cache_seqlens = torch.zeros((bsz,), dtype = torch.int32, device = dev)

        return self._attend(
            x, bsz, seqlen, params, ckv_cache, kpe_cache, block_table, cache_seqlens,
            append = lambda ckv, k_pe:
                mla_kv_append(
                    ckv.reshape(bsz, seqlen, self.kv_lora_rank),
                    k_pe.reshape(bsz, seqlen, self.qk_rope_head_dim),
                    ckv_cache, kpe_cache, block_table, cache_seqlens,
                ),
            host_seqlens = [0] * bsz,
        )


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        raise NotImplementedError()


    def tp_export(self, plan, producer):
        raise NotImplementedError("Tensor-parallel inference is not implemented for MLA layers")
