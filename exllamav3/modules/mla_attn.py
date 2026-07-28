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
    has_triton,
)

# Query lengths at or below this use the flash-decoding kernel (kv split across programs);
# above it, the long-query kernel (q split across programs) wins
MAX_DECODE_QLEN = 16


class MLAttention(Module):
    """
    Multi-head latent attention (DeepSeek-V2/V3, Kimi-Linear).

    Attention runs in absorbed form end to end. The cache holds only the compressed latent and the
    shared RoPE key -- 576 values per token, against 81920 for the equivalent expanded K/V of a
    128-head model -- and per-head K and V are never materialized, in prefill or decode. What makes
    that possible is folding the kv_b up-projection into the query and the output instead:

        scores = (q_nope @ W_UK) . c_kv  +  q_pe . k_pe
        o      = (softmax(scores) @ c_kv) @ W_UV

    Both folds are batched GEMMs over the head axis, so the queries and the attention output are
    kept head-major throughout and the two kv_lora_rank-wide tensors are never permuted.

    W_UK and W_UV stay unquantized. They are pure weight streaming (measured at 76-81% of memory
    peak), the absorb is a bmm rather than a GEMM so the exl3 kernels do not apply, and folding
    W_UK into q_b_proj instead would triple that projection -- strictly worse than the 33 MB per
    layer this costs on a 128-head model.
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

        # Absorption matrices, split out of kv_b_proj at load
        self.w_uk = None    # (H, qk_nope_head_dim, kv_lora_rank)
        self.w_uv = None    # (H, kv_lora_rank, v_head_dim)
        self.kv_b = None
        self._scratch = {}


    def cache_layer_type(self, default, kwargs: dict):
        """MLA stores a latent instead of per-head K/V, so it overrides the cache layer the Cache
        was constructed with. A quantized cache request maps to the packed-latent layer: k_bits
        sets the latent width, the shared rope key stays fp16 (v_bits is accepted but unused --
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

        # kv_b_proj maps the latent to per-head K-nope and V. Absorbed attention never applies it
        # directly; it folds the two halves into the query and the output instead
        w = self.config.stc.get_tensor(f"{self.key}.{self.key_kv_b}.weight", device, no_defer = True)
        assert w.shape == (self.num_q_heads * (self.qk_nope_head_dim + self.v_head_dim),
                           self.kv_lora_rank), \
            f"{self.key}.{self.key_kv_b}: unexpected shape {tuple(w.shape)}"
        w = w.view(self.num_q_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        # score = q_nope . (W_kb c) = (q_nope @ W_kb) . c
        self.w_uk = w[:, :self.qk_nope_head_dim, :].contiguous().half()
        # o = W_vb (sum_k p_k c_k) = o_lat @ W_vb^T
        self.w_uv = w[:, self.qk_nope_head_dim:, :].transpose(1, 2).contiguous().half()
        self.kv_b = w.contiguous()


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)
        self.load_local(device, **kwargs)


    @override
    def get_tensors(self):
        # Unquantized, so it has to be carried into the converted model verbatim
        t = {}
        if self.kv_b is not None:
            t[f"{self.key}.{self.key_kv_b}.weight"] = self.kv_b.view(
                self.num_q_heads * (self.qk_nope_head_dim + self.v_head_dim), self.kv_lora_rank
            ).contiguous()
        return t


    @override
    def unload(self):
        super().unload()
        for cl in self.cache_layers:
            cl.free()
        self.rope = None
        self.w_uk = None
        self.w_uv = None
        self.kv_b = None
        self._scratch = {}


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
                append, qc = None):
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

        # Absorb W_UK into the queries: (H, R, nope) x (H, nope, kv_lora) -> (H, R, kv_lora).
        #
        # The A operand is materialized contiguous rather than passed as the strided view: on
        # SM120, cuBLAS routes the strided-batched fp16 GEMM to an nvjet TMA kernel
        # (nvjet_sm120_hsh_mma_*_tmaAB) that intermittently MMU-faults on the strided view.
        # Caught with cuda-gdb after presenting as flaky illegal-memory-access crashes whose
        # incidence depended on cache quantization width and device split (allocation layout
        # deciding whether the wild access lands in mapped pool memory). The copy is small
        # (R x H x nope fp16) and the unfold bmm below already has a contiguous A
        q_lat = torch.bmm(q_nope.permute(1, 0, 2).contiguous(), self.w_uk)
        q_pe_hm = q_pe.reshape(R, H, self.qk_rope_head_dim).permute(1, 0, 2).contiguous()
        _dbg_sync("absorb-bmm", x.device)

        append(ckv, k_pe)

        kernel = mla_attn_triton_decode if seqlen <= MAX_DECODE_QLEN else mla_attn_triton_prefill
        extra = dict(scratch = self._scratch) if seqlen <= MAX_DECODE_QLEN else {}
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

        # Unfold W_UV: (H, R, kv_lora) x (H, kv_lora, v_head) -> (H, R, v_head)
        o = torch.bmm(o_lat, self.w_uv)
        o = o.permute(1, 0, 2).reshape(bsz, seqlen, H * self.v_head_dim)
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

        from ..cache import CacheLayer_MLA_quant
        if isinstance(layer, CacheLayer_MLA_quant):
            # Packed latent feeds the kernels directly (online dequant, rotated domain); the rope
            # key pages are fp16 either way
            ckv_cache, sk, kpe_cache, bits = layer.get_qc()
            qc = (sk, bits)
        else:
            ckv_cache, kpe_cache = layer.get_kv(cache_seqlens, block_table)
            qc = None

        return self._attend(
            x, bsz, seqlen, params, ckv_cache, kpe_cache, block_table, cache_seqlens,
            append = lambda ckv, k_pe:
                layer.update_kv_direct(cache_seqlens, block_table, ckv, k_pe, seqlen),
            qc = qc,
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
        )


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        raise NotImplementedError()


    def tp_export(self, plan, producer):
        raise NotImplementedError("Tensor-parallel inference is not implemented for MLA layers")
