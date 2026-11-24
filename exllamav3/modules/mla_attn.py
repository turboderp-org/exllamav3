from __future__ import annotations
from typing_extensions import override
import torch
from ..util.device_copy import to_device
from ..model.config import Config
from ..util.rope import RopeSettings, RoPE
from ..util.tensor import get_for_device, to2
from . import Module, Linear, RMSNorm
from .layernorm import LayerNorm
from ..model.model_tp_alloc import TPAllocation
from .attention_fn.mla_triton import (
    mla_attn_triton_decode,
    mla_attn_triton_prefill,
    mla_attn_triton_prefill_mha,
    mla_absorb,
    mla_unfold,
)
from .attention_fn.bc_attn import MAX_BSZ as _bc_max_bsz
from .attention_fn.smem import NoFittingConfig
import os

# Prefill strategy: "mha" (default) up-projects past tiles from the compressed cache and attends
# in MHA form. ~2.8x fewer FLOPs than running the absorbed form over the whole context. "absorbed"
# restores the single-kernel absorbed prefill for A/B testing
_prefill_mode = os.environ.get("EXL3_MLA_PREFILL", "mha")

# EXL3_BC_MLA=0 disables the graph-captured decode block (dispatch path only), for A/B testing
_bc_mla_enable = os.environ.get("EXL3_BC_MLA", "1") != "0"

# Query lengths at or below this use the flash-decoding kernel (kv split across programs);
# above it, the long-query kernel (q split across programs) wins
MAX_DECODE_QLEN = 16

# Width of one indexer-scoring tile (keys per kernel call) in the eager top-k path. Bounds
# the score transient at (256 rows, tile) regardless of context length; must be a multiple
# of the page size (256)
_score_tile = int(os.environ.get("EXL3_DSA_SCORE_TILE", 32768))
assert _score_tile % 256 == 0 and _score_tile > 0



def _host_seqlens(params: dict, cache_seqlens: torch.Tensor) -> list:
    """Host copy of cache_seqlens, once per forward (shared across layers via the params
    dict). The generator supplies cache_seqlens as a host (pinned staging) tensor, in which
    case this is a free read; only harnesses that pass device tensors pay a sync."""
    host = params.get("_mla_host_seqlens")
    if host is None:
        src = params.get("cache_seqlens")
        if isinstance(src, torch.Tensor) and src.device.type == "cpu":
            host = src.tolist()
        else:
            host = cache_seqlens.cpu().tolist()
        params["_mla_host_seqlens"] = host
    return host

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
        key_head_gate: str | None = None,
        key_out_gate: str | None = None,
        key_sinks: str | None = None,
        window: int = 0,
        qbits_key: str = "bits",
        select_hq_bits: int = 0,
        indexer_mode: str | None = None,
        index_n_heads: int = 0,
        index_head_dim: int = 0,
        index_topk: int = 0,
        index_norm_eps: float = 1e-6,
        index_kpool: int = 0,
        index_kpool_tail: bool = True,
        index_kpool_mode: str = "gate",
        index_keys: dict | None = None,
        key_indexer: str = "indexer",
        cache_layer_idx: int | None = None,
        table_rope_settings: RopeSettings | None = None,
        submodules: dict | None = None,
        tp_affinity: str | None = None,
    ):
        super().__init__(config, key, None)
        # Tensor-parallel placement group (see TPAllocation.affinity_key): DSA "shared" layers must
        # share the device of the "full" layer whose selection they reuse
        self.tp_affinity = tp_affinity

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
        # A caller's per-token frequency table (params["inv_freq"], M-RoPE) is built with one set of rope
        # settings; a layer rotating with another base rescales the table by the ratio of the two
        self.table_rope_settings = table_rope_settings
        self.table_scale = None
        # Llama-4 position scale (Mistral-Small-4): set in load_local from the RoPE instance
        self.l4_beta = 0.0
        self.l4_original = 1
        self.out_dtype = out_dtype
        self.key_kv_b = key_kv_b
        self.key_indexer = key_indexer
        self.norm_eps = rms_norm_eps

        # The softmax scale follows the *unabsorbed* head dim: absorption does not change the
        # scores, only how they are computed. Architectures with YaRN mscale pass their own
        self.sm_scale = sm_scale if sm_scale is not None else self.qk_head_dim ** -0.5

        # head_dim is what the rest of the stack reads for reporting and allocation; the latent
        # width is what actually lands in the cache
        self.head_dim = self.qk_head_dim

        # DSA lightning indexer (GLM-5.2 / DeepSeek-V3.2-on-MLA). "full" layers score and select
        # index_topk tokens per query and publish the selection; "shared" layers reuse the
        # nearest preceding full layer's selection. "external" layers take every forward's selection from
        # the caller (params["mla_selection"]: (rows, K_pad) int32 with K_pad = index_topk rounded up to
        # 32, -1 for unused entries, and the columns in use). None = plain dense MLA
        assert indexer_mode in (None, "full", "shared", "external")
        self.indexer_mode = indexer_mode
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.index_norm_eps = index_norm_eps
        # GLM5.3-style k-pool compression: keys are pooled index_kpool at a time (softmax over
        # cached per-token gate scores + a learned in-pool APE), scoring and selection run over
        # pools, and selections expand back to raw token indices (plus the incomplete tail
        # pool's tokens, appended raw, when index_kpool_tail). The cached plane then holds
        # [k || gate_scores] per token
        self.index_kpool = index_kpool
        self.index_kpool_tail = index_kpool_tail
        # Hy4 pooling ("proj"): the cached plane holds the normalized token key alone, and a pool's key
        # is the softmax-weighted sum of a value projection of its members' keys (gate projection + APE
        # as the logits, all in fp32), normalized and rotated at the pool's first position. Queries
        # rotate their trailing rope dims, and a query always takes its own pool raw, scoring only the
        # index_topk / index_kpool - 1 best pools that closed before it
        assert index_kpool_mode in ("gate", "proj")
        self.index_kpool_mode = index_kpool_mode
        self.index_kpool_own = index_kpool_mode == "proj"
        self.index_keys = {
            "wq_b": "wq_b", "wk": "wk", "k_norm": "k_norm", "weights": "weights_proj",
            "ape": "index_kpool_compress_ape", "gate": "index_kpool_compress_gate",
            "pool_value": "key_value_projection.weight", "pool_gate": "key_gate_projection.weight",
            "pool_norm": "key_norm.weight",
            **(index_keys or {}),
        }
        self.idx_kpool_ape = None
        self.idx_kpool_gate = None
        self.idx_kpool_value = None
        self.idx_kpool_norm = None
        # Cache layers allocate a per-token indexer-key plane for layers that score selections
        self.idx_plane_dim = (
            None if indexer_mode != "full" else
            index_head_dim * 2 if index_kpool and index_kpool_mode == "gate" else index_head_dim
        )

        # Learned per-head sink logit in the softmax denominator ({key}.{key_sinks}, fp32), and a causal
        # sliding window: a query attends to the last `window` positions, its own included. Either one
        # sends attention through the gathered kernel (dsa_attn), which carries both
        self.key_sinks = key_sinks
        self.sinks = None
        self.window = window
        self.out_gate = None

        self.q_a_proj = None
        self.q_a_layernorm = None
        self.q_b_proj = None
        self.q_proj = None
        self.kv_a_proj_with_mqa = None
        self.kv_a_layernorm = None
        self.o_proj = None
        self.head_gate = None
        self.idx_wq_b = None
        self.idx_wk = None
        self.idx_k_norm = None
        self.idx_weights = None

        # A layer of a drafter that shares its target's K/V reads and writes the target's cache layer at
        # this index (through the target's Cache, which the caller passes) and owns none
        self.cache_layer_idx = cache_layer_idx

        self.cache_layers = []
        self.tp_cache_lookup = {}
        self.has_split_cache = False
        self.tp_reduce = False
        self.dispatch_cache = {}

        # kv_b_proj, stored ONLY in the flattened (kv_lora_rank, H * dim) form: the prefill
        # up-projection GEMMs consume it directly, and the decode absorb/unfold run as Triton
        # kernels that read per-head column blocks out of the same layout - one resident copy
        # serves every path, and no cuBLAS batched GEMM is involved anywhere in the module
        self.w_uk_flat = None   # (kv_lora_rank, H * qk_nope_head_dim)
        self.w_uv_flat = None   # (kv_lora_rank, H * v_head_dim)
        self._scratch = {}

        # Tensor-parallel rank that does not hold this layer (MLA runs whole on one device): a
        # stub that only takes part in the trailing all-reduce. It deliberately has no kv_cache
        # cap, so the rank's cache bookkeeping never looks for a cache layer here
        if num_q_heads == 0:
            return

        # In a TP worker the submodules arrive prebuilt (imported from the parent process)
        # instead of being constructed against the tensor collection
        def _sub(name, factory):
            m = submodules.get(name) if submodules is not None else factory()
            self.register_submodule(m)
            return m

        qmap_in = qmap + ".input" if qmap is not None else None
        qmap_o = qmap + ".o" if qmap is not None else None

        # Query path: either a direct projection or a LoRA-style pair with a norm between
        if q_lora_rank is None:
            self.q_proj = _sub("q_proj", lambda: Linear(
                config, f"{key}.{key_q}", hidden_size, num_q_heads * self.qk_head_dim,
                qmap = qmap_in, out_dtype = torch.half, trim_padded_out = True,
                select_hq_bits = select_hq_bits, qbits_key = qbits_key,
            ))
        else:
            self.q_a_proj = _sub("q_a_proj", lambda: Linear(
                config, f"{key}.{key_q_a}", hidden_size, q_lora_rank,
                qmap = qmap_in, out_dtype = torch.half, trim_padded_out = True,
                select_hq_bits = select_hq_bits, qbits_key = qbits_key,
            ))
            self.q_a_layernorm = _sub("q_a_layernorm", lambda: RMSNorm(
                config, f"{key}.{key_q_a_norm}", rms_norm_eps, out_dtype = torch.half,
            ))
            self.q_b_proj = _sub("q_b_proj", lambda: Linear(
                config, f"{key}.{key_q_b}", q_lora_rank, num_q_heads * self.qk_head_dim,
                qmap = qmap + ".q_a" if qmap is not None else None,
                out_dtype = torch.half, trim_padded_out = True,
                select_hq_bits = select_hq_bits, qbits_key = qbits_key,
            ))
            self.q_proj = self.q_b_proj

        # Latent path. The output of this projection goes straight into the cache, so it is the
        # one place where quantization error compounds over the whole context
        self.kv_a_proj_with_mqa = _sub("kv_a_proj_with_mqa", lambda: Linear(
            config, f"{key}.{key_kv_a}", hidden_size, kv_lora_rank + qk_rope_head_dim,
            qmap = qmap_in, out_dtype = torch.half, trim_padded_out = True,
            select_hq_bits = select_hq_bits, qbits_key = qbits_key,
        ))
        self.kv_a_layernorm = _sub("kv_a_layernorm", lambda: RMSNorm(
            config, f"{key}.{key_kv_a_norm}", rms_norm_eps, out_dtype = torch.half,
        ))

        self.o_proj = _sub("o_proj", lambda: Linear(
            config, f"{key}.{key_o}", num_q_heads * v_head_dim, hidden_size,
            qmap = qmap_o, out_dtype = out_dtype, trim_padded_out = True,
            select_hq_bits = select_hq_bits, qbits_key = qbits_key,
        ))

        # Ling 3.x: a per-head sigmoid gate on the attention output, from the block input (kept
        # unquantized: the reference checkpoint excludes it from fp8)
        if key_head_gate is not None:
            self.head_gate = _sub("head_gate", lambda: Linear(
                config, f"{key}.{key_head_gate}", hidden_size, num_q_heads,
                qmap = None, out_dtype = torch.float, pad_to = 1,
            ))

        # Hy4: an elementwise sigmoid gate on the attention output, from the block input. It is as
        # large as o_proj and reads the same input as the q/kv projections, so it quantizes with them
        if key_out_gate is not None:
            self.out_gate = _sub("out_gate", lambda: Linear(
                config, f"{key}.{key_out_gate}", hidden_size, num_q_heads * v_head_dim,
                qmap = qmap_in, out_dtype = torch.half, trim_padded_out = True,
                select_hq_bits = select_hq_bits, qbits_key = qbits_key,
            ))

        ik = self.index_keys
        if indexer_mode == "full":
            assert q_lora_rank is not None, "DSA indexer queries project from the q_a latent"
            self.idx_wq_b = _sub("idx_wq_b", lambda: Linear(
                config, f"{key}.{key_indexer}.{ik['wq_b']}", q_lora_rank, index_n_heads * index_head_dim,
                qmap = qmap + ".q_a" if qmap is not None else None,
                out_dtype = torch.half, trim_padded_out = True,
                select_hq_bits = select_hq_bits, qbits_key = qbits_key,
            ))
            # Key head and per-head scoring weights are router-like: tiny, and selection noise
            # is coherent across every layer sharing it, so they stay unquantized
            self.idx_wk = _sub("idx_wk", lambda: Linear(
                config, f"{key}.{key_indexer}.{ik['wk']}", hidden_size, index_head_dim,
                qmap = None, out_dtype = torch.half, pad_to = 1,
            ))
            self.idx_k_norm = _sub("idx_k_norm", lambda: (RMSNorm if index_kpool_mode == "proj" else LayerNorm)(
                config, f"{key}.{key_indexer}.{ik['k_norm']}", index_norm_eps, out_dtype = torch.half,
            ))
            self.idx_weights = _sub("idx_weights", lambda: Linear(
                config, f"{key}.{key_indexer}.{ik['weights']}", hidden_size, index_n_heads,
                qmap = None, out_dtype = torch.half, pad_to = 1,
            ))

        if cache_layer_idx is None:
            self.caps.update({
                "kv_cache": True
            })



    def _mha_form(self, q, q_pe, ckv_cache, kpe_cache, block_table, host_seqlens, bsz, seqlen, qc, params):
        """MHA-form attention: everything (past and current chunk) is read back from the cache
        and attended over per-head up-projections. RoPE produced q_pe as a copy (the strided
        slice cannot reshape into a view), so fold it back into q's pe columns for the kernel's
        packed [nope | pe] per-head rows."""
        H = self.num_q_heads
        R = bsz * seqlen
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
        return self._out(o, params)

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
            self.q_proj.optimizer_targets() + \
            (self.idx_wq_b.optimizer_targets() if self.idx_wq_b else [])
        kv = self.kv_a_proj_with_mqa.optimizer_targets() + \
            (self.out_gate.optimizer_targets() if self.out_gate else [])
        o = self.o_proj.optimizer_targets()
        return [[q, kv, o]]


    def load_local(self, device, **kwargs):
        if self.num_q_heads == 0:
            return

        for cl in self.cache_layers:
            cl.alloc(device)

        # A TP worker receives the k-pool parameters and the kv_b flats prebuilt from the parent
        # process (it has no tensor collection); only a loading module reads them itself
        if self.index_kpool and self.indexer_mode == "full" and self.idx_kpool_ape is None:
            stc = self.config.stc
            def raw(name, dtype):
                return stc.get_tensor(
                    f"{self.key}.{self.key_indexer}.{self.index_keys[name]}", device,
                    no_defer = True, allow_bf16 = True, arena = False).to(dtype)
            self.idx_kpool_ape = raw("ape", torch.float)
            if self.index_kpool_mode == "proj":
                self.idx_kpool_gate = raw("pool_gate", torch.float)
                self.idx_kpool_value = raw("pool_value", torch.float)
                self.idx_kpool_norm = raw("pool_norm", torch.float)
            else:
                self.idx_kpool_gate = raw("gate", torch.half)

        if self.key_sinks is not None and self.sinks is None:
            self.sinks = self.config.stc.get_tensor(
                f"{self.key}.{self.key_sinks}", device, no_defer = True, allow_bf16 = True, arena = False
            ).float().contiguous()

        if self.rope_settings:
            self.rope = RoPE(device, self.rope_settings)
            if self.table_rope_settings is not None:
                self.table_scale = (self.rope.inv_freq / RoPE(device, self.table_rope_settings).inv_freq).float()
            # The Llama-4 scale multiplies the FULL query post-rope (HF semantics, q only). Only
            # the q_pe/k_pe slices pass through the rope kernel here, which would miss q_nope, so
            # take the beta out of the kernel and scale the whole query in _attend instead
            self.l4_beta = self.rope.llama_4_scaling_beta
            self.l4_original = self.rope.llama_4_scaling_original
            self.rope.llama_4_scaling_beta = 0.0

        # kv_b_proj maps the latent to per-head K-nope and V. Attention never applies it as that
        # GEMM; the halves fold into the query/output (decode) or up-project past tiles (prefill)
        if self.w_uk_flat is not None:
            return
        # Source only: the halves are copied into w_uk_flat / w_uv_flat below and w is dropped,
        # so it stays out of the loader's slab blocks
        w = self.config.stc.get_tensor(f"{self.key}.{self.key_kv_b}.weight", device, no_defer = True, arena = False)
        if w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # fp8 checkpoint: this tensor is read raw rather than through a Linear, so apply the
            # inverse weight scale here (scalar per-tensor or block grid)
            si = self.config.stc.get_tensor(
                f"{self.key}.{self.key_kv_b}.weight_scale_inv", device, optional = True, no_defer = True, arena = False
            )
            wf = w.float()
            if si is not None:
                si = si.float()
                if si.dim() == 2:
                    r, c = wf.shape
                    sr, sc = si.shape
                    wf = (wf.view(sr, r // sr, sc, c // sc) * si.view(sr, 1, sc, 1)).view(r, c)
                else:
                    wf = wf * si
            w = wf.half()
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
        # K-pool compression parameters are raw (unquantized) module tensors, carried as-is
        if self.idx_kpool_ape is not None:
            pre = f"{self.key}.{self.key_indexer}."
            t[pre + self.index_keys["ape"]] = self.idx_kpool_ape.contiguous()
            if self.index_kpool_mode == "proj":
                t[pre + self.index_keys["pool_gate"]] = self.idx_kpool_gate.contiguous()
                t[pre + self.index_keys["pool_value"]] = self.idx_kpool_value.contiguous()
                t[pre + self.index_keys["pool_norm"]] = self.idx_kpool_norm.contiguous()
            else:
                t[pre + self.index_keys["gate"]] = self.idx_kpool_gate.contiguous()
        if self.sinks is not None:
            t[f"{self.key}.{self.key_sinks}"] = self.sinks.contiguous()
        return t


    @override
    def unload(self):
        super().unload()
        for cl in self.cache_layers:
            cl.free()
        self.rope = None
        self.table_scale = None
        self.w_uk_flat = None
        self.w_uv_flat = None
        self.idx_kpool_ape = None
        self.idx_kpool_gate = None
        self.idx_kpool_value = None
        self.idx_kpool_norm = None
        self.sinks = None
        self._scratch = {}
        self.dispatch_cache = {}


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        # TP rank without this layer: contribute nothing, stay in step with the collective
        if self.num_q_heads == 0:
            x = torch.zeros_like(x, dtype = self.out_dtype)
            if self.tp_reduce:
                self.tp_collect(params["backend"], x, False)
            return to2(x, out_dtype, self.out_dtype)

        bsz, seqlen, _ = x.shape
        if self.head_gate is not None:
            assert not self.tp_reduce, "Tensor parallelism does not split the MLA head gate yet"
            params["mla_head_gate"] = torch.sigmoid(self.head_gate.forward(x, params).float())
        if self.out_gate is not None:
            assert not self.tp_reduce, "Tensor parallelism does not split the MLA output gate yet"
        attn_mode = params.get("attn_mode", "flash_attn_nc")
        match attn_mode:
            case "flash_attn":
                x = self.decode_flash_attn(x, bsz, seqlen, params)
            case "flash_attn_nc":
                x = self.decode_flash_attn_nc(x, bsz, seqlen, params)
            case _:
                raise ValueError(f"Unknown attn_mode: {attn_mode}")
        if self.tp_reduce:
            self.tp_collect(params["backend"], x)
        return to2(x, out_dtype, self.out_dtype)


    def _out(self, o: torch.Tensor, params: dict) -> torch.Tensor:
        gate = params.pop("mla_head_gate", None)
        if gate is not None:
            shape = o.shape
            o = (o.view(*shape[:-1], self.num_q_heads, self.v_head_dim).float() * gate.unsqueeze(-1)).to(o.dtype).view(shape)
        gate = params.pop("mla_out_gate", None)
        if gate is not None:
            o = o * gate.sigmoid_().view(o.shape)
        return self.o_proj.forward(o, params)


    def project_q(self, x: torch.Tensor, params: dict, return_resid: bool = False):
        if self.q_a_proj is None:
            q = self.q_proj.forward(x, params)
            return (q, None) if return_resid else q
        q_resid = self.q_a_proj.forward(x, params)
        q_resid = self.q_a_layernorm.forward(q_resid, params, out_dtype = torch.half)
        q = self.q_b_proj.forward(q_resid, params)
        return (q, q_resid) if return_resid else q


    def _indexer_rope_(self, x4, position, positions, position_ids, inv_freq, trailing = False):
        """In-place partial rope on the leading (or trailing) qk_rope_head_dim dims of a (bsz,
        seqlen, heads, D) tensor, via a strided slice view: the kernel reads through the
        head stride, so no slice copy or writeback is made (the eager mirror of the BC
        path's rope_gr on its kidx4/qidx4 narrow views)."""
        from ..ext import exllamav3_ext as ext
        if self.rope is None or self.qk_rope_head_dim == 0:
            return
        rope = self.rope
        v = x4[..., -self.qk_rope_head_dim :] if trailing else x4[..., : self.qk_rope_head_dim]
        ext.rope(
            v, v, None, None,
            rope.inv_freq if inv_freq is None else inv_freq,
            position,
            positions.contiguous() if positions is not None else None,
            position_ids.contiguous() if position_ids is not None else None,
            int(rope.rope_settings.rope_style),
            rope.attn_factor,
            None, None, self.norm_eps, 0.0,
            rope.llama_4_scaling_beta, rope.llama_4_scaling_original,
            rope.rope_settings.rotate_dims, 0,
        )

    def _indexer_keys(self, x, params, position, positions, position_ids, inv_freq):
        """Roped indexer keys for the current chunk, (bsz, seqlen, index_head_dim). Only the
        first qk_rope_head_dim dims rotate (interleaved pairing, same table as the main
        attention); k_norm is a biased LayerNorm applied before the rotation."""
        k = self.idx_wk.forward(x, params)
        k = self.idx_k_norm.forward(k, params, out_dtype = torch.half).contiguous()
        if self.index_kpool_mode == "proj":
            # The plane holds the normalized key; pooling projects and rotates it (_kpool_proj)
            return k
        bsz, seqlen, D_i = k.shape
        k = k.view(bsz, seqlen, 1, D_i)
        self._indexer_rope_(k, position, positions, position_ids, inv_freq)
        k = k.view(bsz, seqlen, D_i)
        if self.index_kpool:
            # Pack the per-token pool-gate scores next to the key: pooled keys are rebuilt
            # from this plane (a pool's members' gate scores feed the in-pool softmax)
            gate = torch.nn.functional.linear(x.to(torch.half), self.idx_kpool_gate)
            k = torch.cat((k, gate), dim = -1)
        return k


    def _indexer_topk(self, x, params, q_resid, bsz, seqlen, host_seqlens,
                      position, positions, position_ids, inv_freq,
                      k_idx_chunk = None, idx_pool = None, block_table = None):
        """Lightning-indexer scoring + top-k selection, per batch row. Keys come either from
        the current chunk (cache-less path, k_idx_chunk) or the paged indexer plane (idx_pool +
        block_table). Returns -1-padded int32 indices, (bsz * seqlen, K_pad); selection is per
        query token, shared by all attention heads."""
        from .attention_fn.dsa_triton import dsa_indexer_scores
        from ..ext import exllamav3_ext as ext

        H_i, D_i = self.index_n_heads, self.index_head_dim
        q_idx = self.idx_wq_b.forward(q_resid, params).view(bsz, seqlen, H_i, D_i).contiguous()
        self._indexer_rope_(q_idx, position, positions, position_ids, inv_freq)
        # Raw head weights; the scoring kernel folds in the D_i**-0.5 and H_i**-0.5 scales and
        # runs the relu-weighted reduction in fp32, matching the reference
        w = self.idx_weights.forward(x, params)

        t_max = max(host_seqlens) + seqlen
        k_pad = -(-min(self.index_topk, t_max) // 32) * 32
        indices = torch.empty((bsz * seqlen, k_pad), dtype = torch.int32, device = x.device)
        # Row slabs bound the transient score matrix in one direction, tiles over the visible
        # context bound it in the other: nothing here scales with T, so the reference forward
        # at load time is a true worst case for any later context length. Selection is
        # per-row and per-tile top-k merge is exact (any global top-k member is in its own
        # tile's top-k), so neither slabbing nor tiling changes membership
        slab = 256
        t_tile = _score_tile
        if idx_pool is not None:
            # Tiles slice the block table whole pages at a time
            epp = idx_pool.shape[1]
            t_tile = max(epp, t_tile // epp * epp)
        from ..util.tensor import g_tensor_cache
        for b in range(bsz):
            for r0 in range(0, seqlen, slab):
                r1 = min(r0 + slab, seqlen)
                rows = r1 - r0
                pos0 = host_seqlens[b] + r0
                t_slab = host_seqlens[b] + r1
                k_sel = min(self.index_topk, t_slab)
                out_slab = indices[b * seqlen + r0 : b * seqlen + r1]

                def tile_scores(t0, t1, scores_out = None):
                    if k_idx_chunk is not None:
                        return dsa_indexer_scores(
                            q_idx[b, r0:r1], w[b, r0:r1], k_idx_chunk[b][t0:t1],
                            pos0 - t0, 1, t1 - t0, scores = scores_out,
                        )
                    epp = idx_pool.shape[1]
                    bt = block_table[b]
                    if t0:
                        bt = bt[t0 // epp : -(-t1 // epp)]
                    return dsa_indexer_scores(
                        q_idx[b, r0:r1], w[b, r0:r1], idx_pool.view(-1, D_i),
                        pos0 - t0, 1, t1 - t0, scores = scores_out,
                        block_table = bt, epp = epp,
                    )

                dev = x.device
                # One fixed score-row stride (the tile width) for every slab and tile: the
                # stride is a kernel constexpr, so letting it follow the visible length
                # recompiled the scoring kernel on every new 128-key boundary
                s_stride = -(-t_tile // 128) * 128
                s_backing = g_tensor_cache.get(dev, (slab * s_stride,), torch.half, "dsa_stile")

                if t_slab <= t_tile:
                    sc = tile_scores(0, t_slab, s_backing[: rows * s_stride].view(rows, s_stride))
                    ext.dsa_topk(sc, out_slab, k_sel, None, 0)
                    continue

                # Tiled path: fixed-size score/index backings, running (score, index) top-k
                # candidate set merged tile by tile
                i_backing = g_tensor_cache.get(dev, (slab * k_pad,), torch.int32, "dsa_itile")
                run_scr = torch.full((rows, k_sel), -float("inf"), dtype = torch.half, device = dev)
                run_idx = torch.full((rows, k_sel), -1, dtype = torch.int32, device = dev)
                for t0 in range(0, t_slab, t_tile):
                    t1 = min(t0 + t_tile, t_slab)
                    sc = tile_scores(t0, t1, s_backing[: rows * s_stride].view(rows, s_stride))
                    k_t = min(k_sel, t1 - t0)
                    kp_t = -(-k_t // 32) * 32
                    ti = i_backing[: rows * kp_t].view(rows, kp_t)
                    ext.dsa_topk(sc, ti, k_t, None, 0)
                    t_scr = sc.gather(1, ti.clamp_min(0).long())
                    t_scr = t_scr.masked_fill(ti < 0, -float("inf"))
                    t_idx = torch.where(ti >= 0, ti + t0, ti)
                    cand_scr = torch.cat((run_scr, t_scr), dim = 1)
                    cand_idx = torch.cat((run_idx, t_idx), dim = 1)
                    run_scr, sel = cand_scr.topk(k_sel, dim = 1)
                    run_idx = cand_idx.gather(1, sel)
                out_slab.fill_(-1)
                out_slab[:, :k_sel] = torch.where(
                    run_scr > -float("inf"), run_idx, run_idx.new_full((), -1))
        return indices


    def _update_pool_plane(self, idx_layer, block_table, host_seqlens, seqlen, inv_freq = None, shift = None):
        """
        Incrementally extend the pooled-key plane with the pools completed by this chunk's
        append. A pool's key is the (gate_scores + APE)-softmax-weighted mean of its
        index_kpool members' keys; members are read back from the packed per-token plane
        (already appended), so pools straddling chunk boundaries resolve correctly. Complete
        pools are immutable, so each is written exactly once.

        A Hy4 pool key carries the rotation of its first token: read from the caller's per-token
        frequency table (inv_freq, by slot) when there is one, otherwise the slot moved by the chunk's
        own offset between positions and slots (shift, per row). The latter holds for the pools a chunk
        of plain text completes, as long as the pool does not reach back into an image
        """
        P, D_i = self.index_kpool, self.index_head_dim
        plane = idx_layer.get_idx()
        epp = plane.shape[1]
        proj = self.index_kpool_mode == "proj"
        flat = plane.view(-1, D_i if proj else 2 * D_i)
        ape = self.idx_kpool_ape
        for b in range(block_table.shape[0] if block_table.dim() == 2 else 1):
            pos0 = host_seqlens[b]
            first_pool = pos0 // P
            last_pool = (pos0 + seqlen) // P
            n_new = last_pool - first_pool
            if n_new <= 0:
                continue
            bt = block_table[b]
            tok = torch.arange(first_pool * P, last_pool * P, device = plane.device)
            rows = bt[tok // epp].long() * epp + tok % epp
            if proj:
                table = inv_freq[b : b + 1] if inv_freq is not None and inv_freq.dim() == 3 else inv_freq
                pool_keys = self._kpool_proj(
                    flat[rows].view(n_new, P, D_i), first_pool, table,
                    shift[b] if shift is not None and table is None else None,
                )
            else:
                members = flat[rows].view(n_new, P, 2 * D_i).float()
                keys, gates = members[..., :D_i], members[..., D_i:]
                probs = torch.softmax(gates + ape.unsqueeze(0), dim = 1)
                pool_keys = (probs * keys).sum(dim = 1).to(torch.half)
            idx_layer.update_pool_direct(
                torch.tensor([first_pool], dtype = torch.int32, device = plane.device),
                bt.unsqueeze(0) if bt.dim() == 1 else bt,
                pool_keys.unsqueeze(0),
            )


    def _kpool_proj(self, members: torch.Tensor, first_pool: int, inv_freq = None, shift = None) -> torch.Tensor:
        """Hy4 pooled keys: members (n, P, D_i) are the normalized token keys of n complete pools
        starting at pool first_pool. The pooling runs in fp32 (a softmax over P slots; fp16 rounding
        reorders near-ties and changes which pools a query recalls), then the pooled key is
        normalized and rotated at its pool's first position: its slot, looked up in a per-token
        frequency table (inv_freq) or moved by shift. Returns (n, D_i) fp16"""
        import torch.nn.functional as F
        n, P, D_i = members.shape
        m = members.float()
        val = F.linear(m, self.idx_kpool_value)
        gate = F.linear(m, self.idx_kpool_gate) + self.idx_kpool_ape
        pooled = (val * torch.softmax(gate, dim = 1)).sum(dim = 1)
        pooled = pooled * torch.rsqrt(pooled.square().mean(dim = -1, keepdim = True) + self.index_norm_eps)
        pooled = (pooled * self.idx_kpool_norm).half().contiguous()
        if n:
            pos = ((first_pool + torch.arange(n, device = members.device)) * P).int().view(1, n)
            if shift is not None:
                pos = pos + shift
            self._indexer_rope_(pooled.view(1, n, 1, D_i), 0, None, pos.contiguous(), inv_freq, trailing = True)
        return pooled


    def _indexer_topk_kpool(self, x, params, q_resid, bsz, seqlen, host_seqlens,
                            k_idx_chunk = None, idx_pool = None, pool_plane = None,
                            block_table = None, position = 0, positions = None, position_ids = None,
                            inv_freq = None):
        """
        K-pool indexer selection (GLM5.3): scoring and top-k run over pooled keys
        (index_kpool consecutive tokens per entry), selections expand back to raw indices,
        and the query's incomplete tail pool is appended as raw tokens. Paged path scores
        the incrementally maintained pooled plane through the tiled scorer (compress_rate
        gives the per-query causal pool bound); the cache-less path (k_idx_chunk, packed
        [k || gate] rows) builds pools eagerly. Returns -1-padded int32 indices.
        """
        import torch.nn.functional as F
        from .attention_fn.dsa_triton import dsa_indexer_scores
        from ..ext import exllamav3_ext as ext
        from ..util.tensor import g_tensor_cache

        H_i, D_i, P = self.index_n_heads, self.index_head_dim, self.index_kpool
        q_idx = self.idx_wq_b.forward(q_resid, params).view(bsz, seqlen, H_i, D_i).contiguous()
        w = self.idx_weights.forward(x, params)
        # own: a query scores the pools that closed before its own pool (pool < pos // P) and takes its
        # own pool raw, complete or not; otherwise it scores every complete pool up to its position
        # (pool < (pos + 1) // P) and the incomplete tail pool comes along raw
        own = self.index_kpool_own
        if own:
            self._indexer_rope_(q_idx, position, positions, position_ids, inv_freq, trailing = True)
        sel_pools = self.index_topk // P - (1 if own else 0)
        tail_w = P if own else (P - 1 if self.index_kpool_tail else 0)

        t_max = max(host_seqlens) + seqlen
        sel_pools_max = min(sel_pools, (t_max - 1) // P if own else t_max // P)
        out_w = max(sel_pools_max * P + tail_w, 1)
        k_pad = -(-out_w // 32) * 32
        indices = torch.full((bsz * seqlen, k_pad), -1, dtype = torch.int32, device = x.device)

        def append_tail(b, base_cols):
            if not tail_w or P <= 1:
                return
            q_pos = host_seqlens[b] + torch.arange(seqlen, device = x.device)
            if own:
                tail_count = q_pos.remainder(P) + 1
                tail_start = q_pos + 1 - tail_count
            else:
                vis = q_pos + 1
                tail_count = vis.remainder(P)
                tail_start = vis - tail_count
            toff = torch.arange(tail_w, device = x.device)
            tail = tail_start.unsqueeze(1) + toff.unsqueeze(0)
            tail = torch.where(toff.unsqueeze(0) < tail_count.unsqueeze(1),
                               tail, tail.new_full((), -1))
            rows = indices[b * seqlen : (b + 1) * seqlen]
            rows[:, base_cols : base_cols + tail_w] = tail.int()

        if pool_plane is not None:
            # Paged pooled plane: same slab/tile structure as _indexer_topk, entries are
            # pools. dsa_indexer_scores' causal bound (q_pos0 + r + 1) // compress_rate is
            # exactly the visible-complete-pool count
            wf = w  # raw head weights, scales folded in-kernel
            epp = pool_plane.shape[1]
            slab = 256
            t_tile = max(epp, _score_tile // epp * epp)
            for b in range(bsz):
                pools_b = (host_seqlens[b] + seqlen) // P
                base_cols = 0
                for r0 in range(0, seqlen, slab):
                    r1 = min(r0 + slab, seqlen)
                    rows = r1 - r0
                    # The scorer bounds row r at (pos0 + r + 1) // P pools; own-pool mode stops one position short
                    pos0 = host_seqlens[b] + r0 - (1 if own else 0)
                    pools_slab = (pos0 + rows) // P
                    k_sel = min(sel_pools, pools_slab)
                    if k_sel <= 0:
                        continue
                    out_slab = indices[b * seqlen + r0 : b * seqlen + r1]

                    def tile_scores(t0, t1, scores_out = None):
                        bt = block_table[b]
                        if t0:
                            bt = bt[t0 // epp : -(-t1 // epp)]
                        return dsa_indexer_scores(
                            q_idx[b, r0:r1], wf[b, r0:r1], pool_plane.view(-1, D_i),
                            pos0 - t0 * P, P, t1 - t0, scores = scores_out,
                            block_table = bt, epp = epp,
                        )

                    dev = x.device
                    # Fixed score-row stride (see _indexer_topk): a constexpr in the kernel
                    s_stride = -(-t_tile // 128) * 128
                    s_backing = g_tensor_cache.get(dev, (slab * s_stride,), torch.half, "dsa_stile")
                    kp_sel = -(-k_sel // 32) * 32

                    if pools_slab <= t_tile:
                        sc = tile_scores(0, pools_slab,
                                         s_backing[: rows * s_stride].view(rows, s_stride))
                        pool_idx = torch.empty((rows, kp_sel), dtype = torch.int32, device = dev)
                        ext.dsa_topk(sc, pool_idx, k_sel, None, 0)
                        pool_idx = pool_idx[:, :k_sel]
                    else:
                        i_backing = g_tensor_cache.get(dev, (slab * kp_sel,), torch.int32, "dsa_itile")
                        run_scr = torch.full((rows, k_sel), -float("inf"), dtype = torch.half, device = dev)
                        run_idx = torch.full((rows, k_sel), -1, dtype = torch.int32, device = dev)
                        for t0 in range(0, pools_slab, t_tile):
                            t1 = min(t0 + t_tile, pools_slab)
                            sc = tile_scores(t0, t1, s_backing[: rows * s_stride].view(rows, s_stride))
                            k_t = min(k_sel, t1 - t0)
                            kp_t = -(-k_t // 32) * 32
                            ti = i_backing[: rows * kp_t].view(rows, kp_t)
                            ext.dsa_topk(sc, ti, k_t, None, 0)
                            t_scr = sc.gather(1, ti.clamp_min(0).long())
                            t_scr = t_scr.masked_fill(ti < 0, -float("inf"))
                            t_idx = torch.where(ti >= 0, ti + t0, ti)
                            cand_scr = torch.cat((run_scr, t_scr), dim = 1)
                            cand_idx = torch.cat((run_idx, t_idx), dim = 1)
                            run_scr, sel = cand_scr.topk(k_sel, dim = 1)
                            run_idx = cand_idx.gather(1, sel)
                        pool_idx = torch.where(
                            run_scr > -float("inf"), run_idx, run_idx.new_full((), -1))

                    # Expand pools to raw token indices
                    exp = (pool_idx.unsqueeze(-1) * P +
                           torch.arange(P, dtype = torch.int32, device = dev)).flatten(1)
                    exp = torch.where(
                        pool_idx.unsqueeze(-1).expand(-1, -1, P).flatten(1) >= 0,
                        exp, exp.new_full((), -1))
                    out_slab[:, : exp.shape[1]] = exp
                    base_cols = max(base_cols, exp.shape[1])
                append_tail(b, base_cols)
            return indices

        # Cache-less eager path (parity/compare): pools built from the packed chunk rows
        for b in range(bsz):
            t_total = host_seqlens[b] + seqlen
            plane = k_idx_chunk[b][:t_total]

            n_pools = t_total // P
            base_cols = 0
            if n_pools:
                if self.index_kpool_mode == "proj":
                    table = inv_freq[b : b + 1] if inv_freq is not None and inv_freq.dim() == 3 else inv_freq
                    pool_keys = self._kpool_proj(plane[: n_pools * P].view(n_pools, P, D_i), 0, table).float()
                else:
                    keys, gates = plane[:, :D_i].float(), plane[:, D_i:].float()
                    gk = keys[: n_pools * P].view(n_pools, P, D_i)
                    gg = gates[: n_pools * P].view(n_pools, P, D_i)
                    probs = torch.softmax(gg + self.idx_kpool_ape.unsqueeze(0), dim = 1)
                    pool_keys = (probs * gk).sum(dim = 1)

                sc = torch.einsum("shd,pd->shp", q_idx[b].float(), pool_keys)
                sc = F.relu(sc * (D_i ** -0.5))
                sc = torch.einsum("sh,shp->sp", w[b].float() * (H_i ** -0.5), sc)

                q_pos = host_seqlens[b] + torch.arange(seqlen, device = x.device)
                pool_end = torch.arange(n_pools, device = x.device) * P + (P - 1)
                if own:
                    hidden = pool_end.unsqueeze(0) >= (q_pos - q_pos.remainder(P)).unsqueeze(1)
                else:
                    hidden = pool_end.unsqueeze(0) > q_pos.unsqueeze(1)
                sc = sc.masked_fill(hidden, -float("inf"))

                select_k = min(sel_pools, n_pools)
                top_sc, top_pool = sc.topk(select_k, dim = -1)
                exp = (top_pool.unsqueeze(-1) * P +
                       torch.arange(P, device = x.device)).view(seqlen, -1)
                exp = torch.where(
                    (top_sc > -float("inf")).unsqueeze(-1).expand(-1, -1, P).reshape(seqlen, -1),
                    exp, exp.new_full((), -1))
                indices[b * seqlen : (b + 1) * seqlen, : exp.shape[1]] = exp.int()
                base_cols = exp.shape[1]
            append_tail(b, base_cols)

        return indices

    def _l4_scale(self, bsz, seqlen, position, positions, position_ids, device) -> torch.Tensor:
        """Llama-4 query scale, 1 + beta * ln(1 + pos // original_max), as (R, 1, 1) fp16.
        Position resolution mirrors the rope kernel: position_ids > positions > position."""
        if position_ids is not None:
            pos = position_ids.view(bsz, -1)[:, :seqlen].float()
        elif positions is not None:
            pos = positions.view(bsz, 1).float() + torch.arange(seqlen, device = device, dtype = torch.float)
        else:
            pos = (position + torch.arange(seqlen, device = device, dtype = torch.float)).expand(bsz, seqlen)
        scale = 1.0 + self.l4_beta * torch.log1p(torch.floor(pos / self.l4_original))
        return scale.to(torch.half).reshape(bsz * seqlen, 1, 1)


    def _attend(self, x, bsz, seqlen, params, ckv_cache, kpe_cache, block_table, cache_seqlens,
                append, qc = None, host_seqlens = None, idx_layer = None):
        """Projections, absorption, attention and o_proj, shared by the cached and cache-less
        paths. `append` writes the new latent/rope rows into the supplied page tensors.
        `idx_layer` is the cache layer holding the paged indexer-key plane (full-indexer layers
        on the cached path only)."""
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        inv_freq = get_for_device(params, "inv_freq", self.device, None)
        if inv_freq is not None and self.table_scale is not None:
            # One rescaled table per forward for the layers of this base on this device
            key = ("inv_freq_scaled", self.device, self.rope_settings.rope_theta)
            scaled = params.get(key)
            if scaled is None:
                scaled = params[key] = (inv_freq * self.table_scale).contiguous()
            inv_freq = scaled
        causal = params.get("causal", True)

        H = self.num_q_heads
        R = bsz * seqlen

        from .attention_fn.mla_triton import _dbg_sync

        # Sparse DSA applies once the visible context exceeds the selection budget; below that,
        # top-k selection is all-inclusive and the dense path is bit-equivalent
        if self.indexer_mode == "external":
            assert causal and "mla_selection" in params, "MLA layer with an external selection was given none"
            sparse = True
        elif self.indexer_mode is not None:
            assert causal, "DSA indexer layers are causal-only"
            assert host_seqlens is not None
            sparse = max(host_seqlens) + seqlen > self.index_topk
        else:
            sparse = False
        # Sinks and windows exist in the gathered kernel only: a dense or windowed layer hands it the
        # positions each query sees as an explicit selection
        gathered = (self.sinks is not None or self.window > 0) and not sparse
        if gathered:
            assert causal, "MLA sinks and windows are causal-only"
            if host_seqlens is None:
                host_seqlens = _host_seqlens(params, cache_seqlens)

        # Output gate logits, from the block input (the graph path runs its own)
        if self.out_gate is not None:
            params["mla_out_gate"] = self.out_gate.forward(x, params)

        # Queries
        _dbg_sync("attend-entry (upstream modules)", x.device)
        q, q_resid = self.project_q(x, params, return_resid = True)
        q = q.view(R, H, self.qk_head_dim)
        if self.l4_beta:
            # Whole query (nope + pe), before the split so the absorbed and MHA paths both
            # inherit it; a per-token scalar commutes with the rotation
            q *= self._l4_scale(bsz, seqlen, position, positions, position_ids, x.device)
        _dbg_sync("project_q", x.device)
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
                None, None, self.norm_eps, 0.0, inv_freq,
            )
            _dbg_sync("rope", x.device)

        use_mha = seqlen > MAX_DECODE_QLEN and _prefill_mode == "mha" and causal and not sparse and not gathered
        if not use_mha:
            # Absorb W_UK into the queries, per head, straight from the flat layout. This runs as
            # a Triton kernel rather than a cuBLAS batched GEMM: the strided-batched fp16 form
            # made cuBLAS pick an SM120 nvjet TMA kernel that intermittently MMU-faults (caught
            # with cuda-gdb after presenting as flaky illegal-memory-access crashes whose
            # incidence tracked allocation layout), and the kernel reads any strides for free
            q_lat = mla_absorb(q.view(R, H, self.qk_head_dim), self.w_uk_flat, H, self.qk_nope_head_dim)
            q_pe_hm = q_pe.reshape(R, H, self.qk_rope_head_dim).permute(1, 0, 2).contiguous()

        append(ckv, k_pe)

        # A selection the graph path already made for this step (it declined after the module
        # had updated the indexer planes) is taken as it is
        pre = params.pop("mla_selection", None)
        k_len = 0
        if pre is not None:
            indices, k_len = pre
        else:
            k_idx = self._indexer_update(
                x, params, idx_layer, cache_seqlens, block_table, host_seqlens, seqlen,
                position, positions, position_ids, inv_freq,
            )
            if sparse or gathered:
                indices, k_len = self._selection(
                    x, params, q_resid, bsz, seqlen, host_seqlens, sparse, k_idx, idx_layer, block_table,
                    position, positions, position_ids, inv_freq, cache_seqlens,
                )
        if sparse or gathered:
            # NoPE's zero-width slice is still a view of the entire projected query.
            # Give the unused rope argument its own empty storage before dropping q.
            if self.qk_rope_head_dim == 0:
                q_pe = torch.empty_like(q_pe)
            # The sparse kernel reads q_lat and token-major q_pe; the unused query
            # projection can now return to the allocator before attention output/staging.
            del q, q_pe_hm
            o_lat = self._attend_sparse(
                q_lat, q_pe, bsz, seqlen, params, ckv_cache, kpe_cache, block_table, indices, qc,
                pool_len = max(host_seqlens) + seqlen, k_len = k_len,
            )
            del q_lat, q_pe
            o = mla_unfold(o_lat, self.w_uv_flat, self.v_head_dim)
            del o_lat
            o = o.reshape(bsz, seqlen, H * self.v_head_dim)
            return self._out(o, params)

        if use_mha:
            return self._mha_form(q, q_pe, ckv_cache, kpe_cache, block_table, host_seqlens, bsz, seqlen, qc, params)

        kernel = mla_attn_triton_decode if seqlen <= MAX_DECODE_QLEN else mla_attn_triton_prefill
        extra = {}
        try:
            o_lat = kernel(
                q_lat, q_pe_hm, ckv_cache, kpe_cache, block_table, cache_seqlens,
                bsz = bsz, q_len = seqlen,
                causal = causal, softmax_scale = self.sm_scale,
                pre_appended_len = seqlen,
                qc = qc,
                **extra,
            )
        except NoFittingConfig:
            # No latent-form tile fits this device's shared memory at this query length (the
            # latent width sets the floor; Turing at 512-wide latents). The MHA form attends
            # over per-head up-projections from the same cache at any query length, and the
            # un-absorbed q is still live here. Dense causal only, as for use_mha
            if not (causal and not sparse and self.w_uk_flat is not None):
                raise
            if host_seqlens is None:
                host_seqlens = _host_seqlens(params, cache_seqlens)
            del q_lat, q_pe_hm
            return self._mha_form(q, q_pe, ckv_cache, kpe_cache, block_table, host_seqlens, bsz, seqlen, qc, params)

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
        # feeds o_proj without a permute. The queries are dead by now
        del q, q_lat, q_pe, q_pe_hm
        o = mla_unfold(o_lat, self.w_uv_flat, self.v_head_dim)
        del o_lat
        o = o.reshape(bsz, seqlen, H * self.v_head_dim)
        return self._out(o, params)


    def _indexer_update(self, x, params, idx_layer, cache_seqlens, block_table, host_seqlens, seqlen,
                        position, positions, position_ids, inv_freq):
        """Full-indexer layers compute this chunk's indexer keys unconditionally: on the cached path
        they must land in the paged plane even while the context is still dense, so the selection has
        complete history once it activates. Returns the chunk's keys (None for other layers)"""
        if self.indexer_mode != "full":
            return None
        k_idx = self._indexer_keys(x, params, position, positions, position_ids, inv_freq)
        if idx_layer is not None:
            idx_layer.update_idx_direct(cache_seqlens, block_table, k_idx, seqlen)
            if self.index_kpool:
                # Where the rows' positions leave their cache slots (text after an image), by how much
                if positions is not None:
                    shift = positions.view(-1).int() - cache_seqlens.view(-1).int()
                else:
                    shift = position - cache_seqlens.view(-1).int()
                self._update_pool_plane(idx_layer, block_table, host_seqlens, seqlen, inv_freq, shift)
        return k_idx


    def _selection(self, x, params, q_resid, bsz, seqlen, host_seqlens, sparse, k_idx, idx_layer, block_table,
                   position, positions, position_ids, inv_freq, cache_seqlens = None) -> tuple:
        """The positions each query row attends over, for the gathered kernels: the indexer's picks in
        the sparse regime, otherwise the window or visible prefix of a layer that always gathers
        (_visible_indices). Returns ((bsz * seqlen, K_pad) int32, -1 for unused entries, columns in use
        or 0 for all). q_resid may be a callable, evaluated only when the indexer scores.

        A drafter that reuses this layer's selections (params["export_mla_selection"], layer indices)
        gets them with the exported states, as (bsz, rows, K_pad): every row of a decode step, the last
        row of a longer chunk. Only selections made here are exported, which covers every forward of a
        layer with sinks or a window; other layers attend without one below the selection budget, and
        select inside the graph path above it"""
        sel = self._select(
            x, params, q_resid, bsz, seqlen, host_seqlens, sparse, k_idx, idx_layer, block_table,
            position, positions, position_ids, inv_freq, cache_seqlens,
        )
        if self.layer_idx in params.get("export_mla_selection", ()):
            states = params.get("export_states")
            if states is None:
                states = params["export_states"] = []
            rows = sel[0].view(bsz, seqlen, -1)
            states.append(rows if seqlen <= MAX_DECODE_QLEN else rows[:, -1:].clone())
        return sel


    def _select(self, x, params, q_resid, bsz, seqlen, host_seqlens, sparse, k_idx, idx_layer, block_table,
                position, positions, position_ids, inv_freq, cache_seqlens = None) -> tuple:
        if not sparse:
            return self._visible_indices(bsz, seqlen, host_seqlens, x.device, params, cache_seqlens)
        if self.indexer_mode != "full":
            indices = params.get("dsa_topk_indices")
            assert indices is not None, \
                "shared-indexer DSA layer found no top-k selection in params"
            return to_device(indices, x.device), 0
        if callable(q_resid):
            q_resid = q_resid()
        if self.index_kpool:
            indices = self._indexer_topk_kpool(
                x, params, q_resid, bsz, seqlen, host_seqlens,
                k_idx_chunk = k_idx if idx_layer is None else None,
                pool_plane = idx_layer.get_pool() if idx_layer is not None else None,
                block_table = block_table,
                position = position, positions = positions, position_ids = position_ids,
                inv_freq = inv_freq,
            )
        else:
            indices = self._indexer_topk(
                x, params, q_resid, bsz, seqlen, host_seqlens,
                position, positions, position_ids, inv_freq,
                k_idx_chunk = k_idx if idx_layer is None else None,
                idx_pool = idx_layer.get_idx() if idx_layer is not None else None,
                block_table = block_table,
            )
        params["dsa_topk_indices"] = indices
        return indices, 0


    def _visible_indices(self, bsz, seqlen, host_seqlens, device, params = None, cache_seqlens = None) -> tuple:
        """Explicit selection for a dense or windowed layer: row r of job b sees positions
        [max(0, p - window + 1), p] at p = host_seqlens[b] + r (the whole prefix without a window).
        Returns ((bsz * seqlen, K_pad) int32, -1 padded, and the number of columns in use). The column
        stride is a kernel constexpr, so it is the window or the indexer's selection budget, whatever
        the context holds so far; only a layer with neither grows it, in powers of two. The list is the
        same for every layer of its kind on a device, so one forward builds it once (kept in params).
        cache_seqlens is the device copy of host_seqlens when the caller has one: building the list
        from it keeps the host from waiting on the device (a host list would be copied in synchronously)"""
        t_max = max(host_seqlens) + seqlen
        if self.window:
            width, bound = min(self.window, t_max), self.window
        elif self.indexer_mode is not None:
            width, bound = t_max, self.index_topk
        else:
            width, bound = t_max, 1 << max(t_max - 1, 31).bit_length()
        k_pad = -(-bound // 32) * 32
        key = ("mla_visible", device, width, k_pad, bsz, seqlen)
        if params is not None:
            cached = params.get(key)
            if cached is not None and cached[0] == host_seqlens:
                return cached[1], width
        if cache_seqlens is not None and cache_seqlens.device == device and cache_seqlens.numel() == bsz:
            base = cache_seqlens.view(bsz, 1).int()
        elif not any(host_seqlens):
            base = torch.zeros((bsz, 1), dtype = torch.int32, device = device)
        else:
            base = torch.tensor(host_seqlens, dtype = torch.int32, device = device).view(bsz, 1)
        p = (base + torch.arange(seqlen, dtype = torch.int32, device = device).view(1, seqlen)).view(-1, 1)
        idx = p - (width - 1) + torch.arange(k_pad, dtype = torch.int32, device = device).view(1, k_pad)
        idx = torch.where((idx >= 0) & (idx <= p), idx, idx.new_full((), -1))
        if params is not None:
            params[key] = (list(host_seqlens), idx)
        return idx, width


    def _attend_sparse(self, q_lat, q_pe, bsz, seqlen, params, ckv_cache, kpe_cache,
                       block_table, indices, qc, pool_len = 0, k_len = 0):
        """Gathered attention over the top-k selected latent rows, returning the head-major latent
        output (H, R, D_c) for the caller's unfold (V3.2-on-MLA form of
        dsa_attn: no window, no sinks, V is the latent). The chunk's own rows are already in
        the paged pool (fp16 or packed-quantized; the packed form is dequantized online by the
        gather kernel) and the indexer's causal bound keeps the selection causal, so the
        kernel needs no mask of its own. The kernel reads the head-major absorbed queries and
        the token-major rope queries directly and emits the head-major latent output the
        unfold consumes: no packed query copy, no output slice/transpose, and the rope half
        of the weighted sum is never accumulated."""
        from .attention_fn.dsa_triton import dsa_attn

        H = self.num_q_heads
        R = bsz * seqlen
        D_r = self.qk_rope_head_dim

        # A single-row block table is shared by every query row inside dsa_attn (stride-0
        # lookup), so bsz 1 never materializes the (R, pages) expansion which would other-
        # wise be the one sparse-path transient that grows with context (pages)
        bt = block_table if bsz == 1 or seqlen == 1 \
            else block_table.repeat_interleave(seqlen, dim = 0)
        o_lat = dsa_attn(
            q_lat, ckv_cache, kpe_cache, bt,
            indices = indices, k_len = k_len or indices.shape[1],
            scale = self.sm_scale, page_size = ckv_cache.shape[1],
            q_pe = q_pe.reshape(R, H, D_r), out_latent = True,
            qc = qc,   # packed latent pages read online (scales, bits), or staged for prefill
            pool_len = pool_len,   # entries the selection can reference (context, not pool)
            sinks = self.sinks,
        )
        return o_lat


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

        layer_idx = self.layer_idx if self.cache_layer_idx is None else self.cache_layer_idx
        layer = cache if not hasattr(cache, "layers") else \
            cache.layers[layer_idx, params.get("layer_instance") or 0]

        # Graph-captured C++ path for the whole decode block (projections through o_proj as one
        # replayed CUDA graph). Falls back to the dispatch path for unsupported configurations,
        # including per-step declines (missing shared selection)
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

        # Host-side lengths for the tiled prefill and the sparse-DSA decision, shared across
        # layers via the params dict; free when the generator passes host cache_seqlens
        if seqlen > MAX_DECODE_QLEN or self.indexer_mode is not None or self.sinks is not None or self.window:
            host_seqlens = _host_seqlens(params, cache_seqlens)
        else:
            host_seqlens = None

        return self._attend(
            x, bsz, seqlen, params, ckv_cache, kpe_cache, block_table, cache_seqlens,
            append = lambda ckv, k_pe:
                layer.update_kv_direct(cache_seqlens, block_table, ckv, k_pe, seqlen),
            qc = qc,
            host_seqlens = host_seqlens,
            idx_layer = layer if self.idx_plane_dim else None,
        )


    def bc_mla_step(self, x, params, layer, block_table, cache_seqlens):
        """Graph-captured decode block, or None when the module/cache-layer pair is not
        supported (caller falls back to the dispatch path)."""
        key = ("bcm", id(layer))
        bcm = self.dispatch_cache.get(key)
        if bcm is None:
            from .attention_fn.bc_mla import build_bc_mla
            bcm = self.dispatch_cache[key] = ((build_bc_mla(self, layer) if _bc_mla_enable else None) or False)
        if bcm is False:
            return None
        if self.indexer_mode is not None or bcm.gathered:
            # Host lengths for the dense/sparse regime decision; same key the dispatch path
            # uses, free when the generator passes host cache_seqlens
            host_seqlens = _host_seqlens(params, cache_seqlens)
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        x = x.contiguous()
        if bcm.gathered and self.indexer_mode != "external":
            # Sinks and windows live on the gathered kernels: the selection (window, visible prefix
            # or the indexer's picks, with the indexer planes updated first) is made here and the
            # graph runs the rest of the block through it
            bsz, seqlen, _ = x.shape
            idx_layer = layer if self.idx_plane_dim else None
            sparse = self.indexer_mode is not None and max(host_seqlens) + seqlen > self.index_topk
            k_idx = self._indexer_update(
                x, params, idx_layer, cache_seqlens, block_table, host_seqlens, seqlen,
                position, positions, position_ids, None,
            )
            params["mla_selection"] = self._selection(
                x, params,
                lambda: self.q_a_layernorm.forward(self.q_a_proj.forward(x, params), params, out_dtype = torch.half),
                bsz, seqlen, host_seqlens, sparse, k_idx, idx_layer, block_table,
                position, positions, position_ids, None, cache_seqlens,
            )
        y = bcm.step(x, params, cache_seqlens, block_table, position, positions, position_ids)
        if y is not None:
            params.pop("mla_selection", None)
        return y


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


    def _autosplit_layer(self, params):
        """The cache layer the autosplit hooks act on, with its quantization flag and the
        chunk length, or None when the module has nothing to prepare or measure"""

        if os.environ.get("EXL3_AUTOSPLIT_WORSTCASE", "1") == "0":
            return None
        cache = params.get("cache")
        if cache is None or self.device is None:
            return None
        from ..cache import CacheLayer_MLA_quant, CacheLayer_MLA_fp16
        layer = cache if not hasattr(cache, "layers") else \
            cache.layers[self.layer_idx, params.get("layer_instance") or 0]
        quant = isinstance(layer, CacheLayer_MLA_quant)
        if not quant and not isinstance(layer, CacheLayer_MLA_fp16):
            return None
        return layer, quant, params["batch_shape"][1]

    def autosplit_prepare(self, params):
        """Decode slot statics for the whole (bsz, q_len) family, allocated before the loader's
        measuring window (they stay resident; nothing is graph-captured at load time)"""
        found = self._autosplit_layer(params)
        if found is None:
            return
        layer, quant, chunk = found

        # Decode statics: every buffer the (bsz <= MAX_BSZ, q_len <= 16) slot family can
        # request, both regimes. Backings are bucketed and shared across slots and layers,
        # so configuring the largest and smallest shapes bounds the whole family
        key = ("bcm", id(layer))
        bcm = self.dispatch_cache.get(key)
        if bcm is None:
            from .attention_fn.bc_mla import build_bc_mla
            bcm = build_bc_mla(self, layer)
        if bcm:
            from .attention_fn.bc_attn import MAX_BSZ
            regimes = (1,) if bcm.gathered else (0, 1) if self.indexer_mode is not None else (0,)
            for b, q in ((1, 1), (MAX_BSZ, MAX_DECODE_QLEN)):
                for rg in regimes:
                    bcm._configure(b, q, rg)


    def autosplit_extra_measure(self, params):
        """Sparse DSA prefill at maximum context, which the (1, chunk)-at-context-0 measuring
        forward does not reach: exercised here so an OoM lands where the loader advances to
        the next device rather than after deployment. Outputs are discarded"""
        found = self._autosplit_layer(params)
        if found is None:
            return
        layer, quant, chunk = found
        # Sparse prefill at maximum context. Synthetic state: every block-table entry aliases
        # page 0, zeroed so the math stays finite
        if self.indexer_mode is None:
            return
        from ..constants import PAGE_SIZE
        num_pages = (layer.qk if quant else layer.k).shape[0]
        t_syn = num_pages * PAGE_SIZE - chunk
        if t_syn + chunk <= self.index_topk:
            return   # cache too small to ever reach the sparse regime
        if quant:
            layer.qk[0].zero_()
            layer.sk[0].zero_()
        else:
            layer.k[0].zero_()
        layer.v[0].zero_()
        if self.idx_plane_dim:
            layer.get_idx()[0].zero_()
            if self.index_kpool:
                layer.get_pool()[0].zero_()
        p2 = {k2: v2 for k2, v2 in params.items() if k2 not in
              ("dev_cache", "_mla_host_seqlens", "positions", "position_ids")}
        p2["cache_seqlens"] = torch.tensor([t_syn], dtype = torch.int32)
        p2["block_table"] = torch.zeros((1, num_pages), dtype = torch.int32)
        p2["position"] = t_syn
        # Selections thread from full to shared layers exactly as in a real forward, via
        # the loader's shared params dict
        ind = params.get("_as_dsa_indices")
        if ind is not None:
            p2["dsa_topk_indices"] = ind
        x = torch.zeros((1, chunk, self.hidden_size), dtype = torch.half, device = self.device)
        self.forward(x, p2)
        if self.indexer_mode == "full":
            params["_as_dsa_indices"] = p2.get("dsa_topk_indices")


    # Tensor-parallel: the layer runs whole on one device. The latent K/V cache is MQA (nothing
    # to split by head), the absorbed decode kernels read all heads of a query row, and the
    # DSA/k-pool indexers are headless, so a head split would replicate the whole KV side for
    # little gain. The allocator places the layer as a single channel (max_devices = 1); the
    # other ranks get a head-less stub whose forward only joins the trailing all-reduce with
    # no contribution, so the residual stream stays in step on every rank
    _tp_submodules = (
        "q_proj", "q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa",
        "kv_a_layernorm", "o_proj", "idx_wq_b", "idx_wk", "idx_k_norm", "idx_weights",
    )
    _tp_tensors = ("w_uk_flat", "w_uv_flat", "idx_kpool_ape", "idx_kpool_gate")

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        stc = self.config.stc
        linears = [m for m in (getattr(self, n) for n in self._tp_submodules) if isinstance(m, Linear)]
        storage = sum(m.storage_size() for m in linears)
        storage += sum(stc.get_tensor_sizes(f"{self.key}.{self.key_kv_b}"))
        for n in ("q_a_layernorm", "kv_a_layernorm", "idx_k_norm"):
            if getattr(self, n) is not None:
                storage += sum(stc.get_tensor_sizes(getattr(self, n).key))
        if self.index_kpool:
            for n in ("index_kpool_compress_ape", "index_kpool_compress_gate"):
                storage += sum(stc.get_tensor_sizes(f"{self.key}.{self.key_indexer}.{n}"))
        for cl in self.cache_layers:
            storage += cl.storage_size()
        overhead_d = self.hidden_size * (self.out_dtype or torch.half).itemsize
        overhead_s = 0
        for cl in self.cache_layers:
            overhead_s += cl.overhead_size()
        # Per-token transients: q (nope + pe) and its absorbed latent form, the latent attention
        # output and the unfolded o rows, plus the new latent/rope row itself
        overhead_s += self.num_q_heads * (self.qk_head_dim + self.kv_lora_rank + self.qk_rope_head_dim) \
            * torch.half.itemsize
        overhead_s += self.num_q_heads * (self.kv_lora_rank + self.v_head_dim) * torch.half.itemsize
        overhead_s += (self.kv_lora_rank + self.qk_rope_head_dim) * torch.half.itemsize
        recons = max((m.recons_size() for m in linears), default = 0)
        tpa = TPAllocation(
            key = self.key,
            channel_width = self.num_q_heads,
            channel_unit = "heads",
            storage_per_device = 0,
            storage_to_split = storage,
            overhead_per_device = overhead_d,
            overhead_to_split = overhead_s,
            recons_temp = recons,
            channels_to_split = 1,
            limit_key = "attn",
            max_devices = 1,
            affinity_key = self.tp_affinity,
        )
        return [tpa]


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        # q_proj aliases q_b_proj on the LoRA query path; export each module once
        names = [n for n in self._tp_submodules if not (n == "q_proj" and self.q_lora_rank is not None)]

        return {
            "cls": MLAttention,
            "kwargs": {
                "key": self.key,
                "layer_idx": self.layer_idx,
                "hidden_size": self.hidden_size,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "v_head_dim": self.v_head_dim,
                "rope_settings": self.rope_settings,
                "q_lora_rank": self.q_lora_rank,
                "sm_scale": self.sm_scale,
                "rms_norm_eps": self.norm_eps,
                "out_dtype": self.out_dtype,
                "key_kv_b": self.key_kv_b,
                "indexer_mode": self.indexer_mode,
                "index_n_heads": self.index_n_heads,
                "index_head_dim": self.index_head_dim,
                "index_topk": self.index_topk,
                "index_norm_eps": self.index_norm_eps,
                "index_kpool": self.index_kpool,
                "index_kpool_tail": self.index_kpool_tail,
                "key_indexer": self.key_indexer,
            },
            "num_q_heads": self.num_q_heads,
            **{name: _export(getattr(self, name)) for name in names},
            **{name: (producer.send(t) if (t := getattr(self, name)) is not None else None)
               for name in self._tp_tensors},
            "cache_layers": [cl.tp_export(plan) for cl in self.cache_layers],
            "device": self.device,
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        kw = exported["kwargs"]
        key = kw["key"]
        device = local_context["device"]
        consumer = local_context["consumer"]
        first, last, unit = plan[key]
        assert unit == "heads"
        num_q_heads = last - first
        assert num_q_heads in (0, exported["num_q_heads"]), \
            "MLA layers run whole on one device (allocation max_devices = 1)"

        def _import(name):
            nonlocal exported, plan
            e = exported.get(name)
            return e["cls"].tp_import(local_context, e, plan) if (e and num_q_heads) else None

        # Submodule keys are not in the plan, so every import is the whole (unsplit) module
        submodules = {name: _import(name) for name in MLAttention._tp_submodules}
        module = MLAttention(
            config = None,
            **kw,
            num_q_heads = num_q_heads,
            submodules = submodules,
        )
        module.device = device

        if num_q_heads:
            for name in MLAttention._tp_tensors:
                e = exported.get(name)
                setattr(module, name, consumer.recv(e, cuda = True) if e is not None else None)
            cache_layers = exported["cache_layers"]
            if len(cache_layers):
                module.has_split_cache = True
                for cl in cache_layers:
                    cli = cl["cls"](None, module, **cl["args"])
                    module.cache_layers.append(cli)
                    module.tp_cache_lookup[cl["args"]["cache_id"]] = cli

        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True
            module.tp_owner = module.tp_single_owner(local_context, key)

        module.load_local(device)
        torch.cuda.synchronize()
        return module
