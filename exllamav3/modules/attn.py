from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from ..model.config import Config
from ..util.rope import RopeSettings, RoPE
from ..util.tensor import get_for_device, to2
from . import Module, Linear, RMSNorm, LayerNorm
from ..constants import PAGE_SIZE
from ..cache import Cache
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from ..util import profile_opt
from .multilinear import MultiLinear
from ..ext import exllamav3_ext as ext
from ..model.model_tp_alloc import TPAllocation
import torch.distributed as dist

"""
SDPA:
    
    attn_mode: "sdpa_nc"
    position (optional, default = 0): int *OR*
    positions: shape (bsz) *OR*
    position_ids: shape (bsz, seq_len)    
    - no cache
    - no chunking
    - batch shape is determined by shape of input_ids
    - no logit softcap support (Gemma)
                    
Flash Attention:
                
    attn_mode: "flash_attn"
    batch_shape: tuple of (bsz, max_seq_len)
    cache: Cache with capacity of at least bsz*max_seq_len tokens
    past_len: int, *OR*
    cache_seqlens: shape (bsz) 
    position: int (overrides past_len for position emb)
    positions: shape (bsz) (overrides cache_seqlens for position emb) *OR*
    position_ids: shape (bsz, seq_len) (overrides cache_seqlens for position emb)
    - max_seq_len must be divisible by 256
    
    attn_mode: "flash_attn"
    block_table: list of page indices, shape (bsz, pages_per_seq)
    cache: Paged cache
    cache_seqlens: shape (bsz)
    positions: shape (bsz) (overrides cache_seqlens for position emb) *OR*
    position_ids: shape (bsz, seq_len) (overrides cache_seqlens for position emb)

    attn_mode: "flash_attn_nc"
    position (optional, default = 0): int *OR*
    positions: shape (bsz) *OR*
    position_ids: shape (bsz, seq_len)    
    - no cache
    - no chunking
    - batch shape is determined by shape of input_ids
"""

def prepare_sdpa_nc(input_ids: torch.Tensor, params: dict) -> torch.Tensor:
    assert "cache" not in params, \
        f"Cache provided for attn_mode: sdpa_nc"
    return input_ids


def prepare_flash_attn_nc(input_ids: torch.Tensor, params: dict) -> torch.Tensor:
    assert "cache" not in params, \
        f"Cache provided for attn_mode: sdpa_nc"
    return input_ids


def prepare_flash_attn(input_ids: torch.Tensor, params: dict) -> torch.Tensor:
    bsz, seq_len = input_ids.shape

    if "batch_shape" in params:
        cache = params["cache"]
        cache_bsz, cache_max_seq_len = params["batch_shape"]
        past_len = params.get("past_len")
        cache_seqlens = params.get("cache_seqlens") if past_len is None else None
        position = params.get("position") if past_len is None else None
        positions = params.get("positions") if past_len is None else None
        position_ids = params.get("position_ids") if past_len is None else None
        assert cache_bsz >= bsz, "batch size too large for cache"
        assert cache_max_seq_len % PAGE_SIZE == 0, f"cache seq len must be a multiple of {PAGE_SIZE}"
        # assert (past_len is not None) ^ (cache_seqlens is not None), "Need either past_len or cache_seqlens"
        assert bsz * cache_max_seq_len <= cache.max_num_tokens, "Cache too small for batch shape"
        cache_bsz = min(bsz, cache_bsz)
        num_pages = cache_bsz * cache_max_seq_len // PAGE_SIZE
        block_table = torch.arange(num_pages, dtype = torch.int32).view(cache_bsz, cache_max_seq_len // PAGE_SIZE)
        if past_len is not None:
            cache_seqlens = torch.tensor([past_len], dtype = torch.int32).repeat(bsz)
            if position is None: position = past_len
        else:
            if positions is None and position_ids is None: positions = cache_seqlens
        if position is None: position = 0
        params["block_table"] = block_table
        params["cache_seqlens"] = cache_seqlens
        params["position"] = position
        params["positions"] = positions
        params["position_ids"] = position_ids

    elif "block_table" in params:
        positions = params.get("positions")
        position_ids = params.get("position_ids")
        cache_seqlens = params.get("cache_seqlens")
        if positions is None and position_ids is None: positions = cache_seqlens
        params["cache_seqlens"] = cache_seqlens
        params["positions"] = positions
        params["position_ids"] = position_ids

    return input_ids


def prepare_for_attn(input_ids: torch.Tensor, params: dict) -> torch.Tensor:
    """
    Add attn parameters to state
    """
    attn_mode = params.get("attn_mode", "flash_attn_nc")
    match attn_mode:
        case "sdpa_nc":
            return prepare_sdpa_nc(input_ids, params)
        case "flash_attn":
            return prepare_flash_attn(input_ids, params)
        case "flash_attn_nc":
            return prepare_flash_attn_nc(input_ids, params)
        case _:
            raise ValueError(f"Unknown attn_mode: {attn_mode}")


class Attention(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        layer_idx: int,
        hidden_size: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        rope_settings: RopeSettings | None,
        sm_scale: float | None = None,
        key_q: str | None = None,
        key_k: str | None = None,
        key_v: str | None = None,
        key_o: str | None = None,
        key_fused_qkv: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        sliding_window: int  = -1,
        logit_softcapping: float = 0.0,
        q_norm: RMSNorm | LayerNorm | None = None,
        k_norm: RMSNorm | LayerNorm | None = None,
        q_proj: Linear | Module | None = None,
        k_proj: Linear | Module | None = None,
        v_proj: Linear | Module | None = None,
        kv_proj: Linear | Module | None = None,
        o_proj: Linear | Module | None = None,
        interleaved_gate: bool = False,
    ):
        super().__init__(config, key, None)

        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.gqa = (num_q_heads != num_kv_heads)
        self.sm_scale = sm_scale
        self.rope_settings = rope_settings
        self.rope = None
        self.out_dtype = out_dtype
        self.sliding_window = sliding_window
        self.logit_softcapping = logit_softcapping
        self.interleaved_gate = interleaved_gate

        if self.num_kv_heads == 0:
            return

        if key_fused_qkv:
            assert not interleaved_gate, "Attn: interleaved_gate not implemented for fused QKV tensor"
            fkey = f"{key}.{key_fused_qkv}"
            frange_q = (0, num_q_heads * head_dim)
            frange_k = (frange_q[1], frange_q[1] + num_kv_heads * head_dim)
            frange_v = (frange_k[1], frange_k[1] + num_kv_heads * head_dim)
        else:
            fkey, frange_q, frange_k, frange_v = None, None, None, None

        if key_q or frange_q:
            f = 2 if interleaved_gate else 1
            self.q_proj = Linear(config, f"{key}.{key_q}", hidden_size, num_q_heads * head_dim * f, qmap = qmap + ".input", fkey = fkey, frange = frange_q, qbits_mod_key = "q")
            self.register_submodule(self.q_proj)
        else:
            assert q_proj
            self.q_proj = q_proj
            self.register_submodule(self.q_proj)

        if key_k or frange_k:
            assert key_v or frange_v
            self.k_proj = Linear(config, f"{key}.{key_k}", hidden_size, num_kv_heads * head_dim, qmap =  qmap + ".input", fkey = fkey, frange = frange_k, qbits_mod_key = "k")
            self.v_proj = Linear(config, f"{key}.{key_v}", hidden_size, num_kv_heads * head_dim, qmap =  qmap + ".input", fkey = fkey, frange = frange_v, qbits_mod_key = "v")
            self.register_submodule(self.k_proj)
            self.register_submodule(self.v_proj)
        else:
            if kv_proj:
                self.kv_proj = kv_proj
                self.register_submodule(self.kv_proj)
            else:
                assert k_proj and v_proj
                self.k_proj = k_proj
                self.v_proj = v_proj
                self.register_submodule(self.k_proj)
                self.register_submodule(self.v_proj)

        if key_o:
            self.o_proj = Linear(config, f"{key}.{key_o}", num_q_heads * head_dim, hidden_size, qmap =  qmap + ".o", out_dtype = out_dtype, qbits_mod_key = "o")
            self.register_submodule(self.o_proj)
        else:
            assert o_proj
            self.o_proj = o_proj
            self.register_submodule(self.o_proj)

        if q_norm:
            assert k_norm, "Must have both Q and K norms, or neither"
            self.q_norm = q_norm
            self.k_norm = k_norm
            self.register_submodule(self.q_norm)
            self.register_submodule(self.k_norm)
            if isinstance(q_norm, RMSNorm):
                self.norm_eps = q_norm.rms_norm_eps
                self.norm_constant_bias = q_norm.constant_bias
                assert self.norm_eps == k_norm.rms_norm_eps
            else:
                self.norm_eps = q_norm.layernorm_eps
                self.norm_constant_bias = 0.0
        else:
            self.q_norm = None
            self.k_norm = None
            self.norm_eps = 1e-6
            self.norm_constant_bias = 0.0

        self.caps.update({
            "kv_cache": True
        })

        self.cache_layers = []
        self.tp_cache_lookup = {}
        self.multi_kv = None
        self.tp_reduce = False

        self.q_norm_tensor = None
        self.k_norm_tensor = None

        self.has_split_cache = False


    @override
    def optimizer_targets(self):
        q = self.q_proj.optimizer_targets()
        k = self.k_proj.optimizer_targets()
        v = self.v_proj.optimizer_targets()
        o = self.o_proj.optimizer_targets()
        return [[q, k + v, o]]


    def load_local(self, device, **kwargs):

        if self.num_kv_heads == 0:
            return

        # Cache
        for cl in self.cache_layers:
            cl.alloc(device)

        if self.rope_settings:
            self.rope = RoPE(
                device,
                self.rope_settings,
            )

        # Test if K and V proj can be fused
        if (
            device != torch.device("cpu") and
            self.k_proj.quant_type == "exl3" and
            self.v_proj.quant_type == "exl3" and
            self.k_proj.out_features == self.v_proj.out_features and
            self.k_proj.inner.K == self.v_proj.inner.K and
            self.k_proj.inner.bias is None and
            self.v_proj.inner.bias is None
        ):
            self.multi_kv = MultiLinear(self. device, [self.k_proj, self.v_proj])

        # Head norm
        if self.q_norm and isinstance(self.q_norm, RMSNorm) and not self.q_norm.span_heads:
            self.q_norm_tensor = self.q_norm.weight.data
            self.k_norm_tensor = self.k_norm.weight.data


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device)
        self.load_local(device, **kwargs)


    @override
    def unload(self):
        super().unload()

        for cl in self.cache_layers:
            cl.free()

        self.rope = None

        if self.multi_kv is not None:
            self.multi_kv.unload()
            self.multi_kv = None

        self.q_norm_tensor = None
        self.k_norm_tensor = None


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        if self.num_kv_heads == 0:
            x = torch.zeros_like(x, dtype = self.out_dtype)
            if self.tp_reduce:
                params["backend"].all_reduce(x, False)
        else:
            bsz, seqlen, _ = x.shape
            attn_mode = params.get("attn_mode", "flash_attn_nc")
            match attn_mode:
                case "sdpa_nc":
                    x = self.decode_sdpa_nc(x, bsz, seqlen, params)
                case "flash_attn":
                    x = self.decode_flash_attn(x, bsz, seqlen, params)
                case "flash_attn_nc":
                    x = self.decode_flash_attn_nc(x, bsz, seqlen, params)
                case _:
                    raise ValueError(f"Unknown attn_mode: {attn_mode}")
            if self.tp_reduce:
                params["backend"].all_reduce(x)

        return to2(x, out_dtype, self.out_dtype)


    def project_qkv(self, x: torch.Tensor, params: dict) -> tuple:
        bsz, q_len, dim = x.shape
        q = self.q_proj.forward(x, params)

        if self.interleaved_gate:
            q, g = torch.chunk(q.view(bsz, q_len, -1, self.head_dim * 2), 2, dim = -1)
            g = g.reshape(bsz, q_len, -1)
        else:
            g = None

        if self.multi_kv is None or bsz * q_len > 32:
            k = self.k_proj.forward(x, params)
            v = self.v_proj.forward(x, params)

        else:
            x = x.view(1, bsz * q_len, dim)
            kvh = torch.empty((2, bsz * q_len, dim), dtype = torch.half, device = x.device)
            kv = torch.empty((2, bsz * q_len, self.num_kv_heads * self.head_dim), dtype = torch.half, device = x.device)
            ext.exl3_mgemm(
                x,
                self.multi_kv.ptrs_trellis,
                kv,
                self.multi_kv.ptrs_suh,
                kvh,
                self.multi_kv.ptrs_svh,
                None,
                None,
                self.multi_kv.K,
                -1,
                self.multi_kv.mcg,
                self.multi_kv.mul1,
                -1,
                -1,
                0
            )
            k = kv[0].view(bsz, q_len, self.num_kv_heads * self.head_dim)
            v = kv[1].view(bsz, q_len, self.num_kv_heads * self.head_dim)

        return q, k, v, g


    def project_o(self, o: torch.Tensor, bsz: int, seqlen: int, params: dict) -> torch.Tensor:
        o = o.reshape(bsz, seqlen, self.num_q_heads * self.head_dim)
        x = self.o_proj.forward(o, params)
        return x


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

        q, k, v, g = self.project_qkv(x, params)
        q = q.view(bsz, seqlen, self.num_q_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        assert self.sliding_window < 0, \
            "Torch SDPA does not support sliding window attention (SWA)"
        assert self.logit_softcapping == 0.0, \
            "Torch SDPA does not support logit softcapping"

        if self.q_norm and (not self.rope or self.q_norm_tensor is None):
            q = self.q_norm.forward(q, params, out_dtype = torch.half)
            k = self.k_norm.forward(k, params, out_dtype = torch.half)

        if self.rope:
            q, k = self.rope.apply(
                q, k,
                position,
                positions,
                position_ids,
                True,
                self.q_norm_tensor,
                self.k_norm_tensor,
                self.norm_eps,
                self.norm_constant_bias,
                inv_freq
            )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal = causal, enable_gqa = self.gqa)
        o = o.transpose(1, 2)
        if self.interleaved_gate: o *= g.sigmoid()

        o = self.project_o(o, bsz, seqlen, params)
        return o


    def decode_flash_attn_nc(
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

        q, k, v, g = self.project_qkv(x, params)
        q = q.view(bsz, seqlen, self.num_q_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        if self.q_norm and (not self.rope or self.q_norm_tensor is None):
            q = self.q_norm.forward(q, params, out_dtype = torch.half)
            k = self.k_norm.forward(k, params, out_dtype = torch.half)

        if self.rope:
            q, k = self.rope.apply(
                q, k,
                position,
                positions,
                position_ids,
                True,
                self.q_norm_tensor,
                self.k_norm_tensor,
                self.norm_eps,
                self.norm_constant_bias,
                inv_freq,
            )

        o = flash_attn_func(
            q = q,
            k = k,
            v = v,
            causal = causal,
            softmax_scale = self.sm_scale,
            window_size = (self.sliding_window, self.sliding_window),
            softcap = self.logit_softcapping
        )
        o = o.view((bsz, seqlen, self.num_q_heads * self.head_dim))
        if self.interleaved_gate: o *= g.sigmoid()

        o = self.project_o(o, bsz, seqlen, params)
        return o


    def decode_flash_attn(
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

        q, k, v, g = self.project_qkv(x, params)
        q = q.view(bsz, seqlen, self.num_q_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # TODO: Add LayerNorm option to fused norm/RoPE kernel
        if self.q_norm and (not self.rope or self.q_norm_tensor is None):
            q = self.q_norm.forward(q, params, out_dtype = torch.half)
            k = self.k_norm.forward(k, params, out_dtype = torch.half)

        if self.rope:
            q, k = self.rope.apply(
                q, k,
                position,
                positions,
                position_ids,
                True,
                self.q_norm_tensor,
                self.k_norm_tensor,
                self.norm_eps,
                self.norm_constant_bias,
                inv_freq
            )

        if self.has_split_cache:
            cache_k, cache_v = self.tp_cache_lookup[cache].get_kv(cache_seqlens, block_table)
        else:
            cache_k, cache_v = cache.get_layer(self.layer_idx, cache_seqlens, block_table)

        o = flash_attn_with_kvcache(
            q = q,
            k = k,
            v = v,
            k_cache = cache_k,
            v_cache = cache_v,
            block_table = block_table,
            cache_seqlens = cache_seqlens,
            causal = causal,
            softmax_scale = self.sm_scale,
            window_size = (self.sliding_window, self.sliding_window),
            softcap = self.logit_softcapping
        )

        if self.has_split_cache:
            self.tp_cache_lookup[cache].update_kv(cache_seqlens, block_table, cache_k, cache_v, seqlen)
        else:
            cache.update_layer(self.layer_idx, cache_seqlens, block_table, cache_k, cache_v, seqlen)

        o = o.view((bsz, seqlen, self.num_q_heads * self.head_dim))
        if self.interleaved_gate: o *= g.sigmoid()

        o = self.project_o(o, bsz, seqlen, params)
        return o


    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        storage = 0
        storage += self.q_proj.storage_size()
        storage += self.k_proj.storage_size()
        storage += self.v_proj.storage_size()
        storage += self.o_proj.storage_size()
        for cl in self.cache_layers:
            storage += cl.storage_size()
        overhead_d = 0
        overhead_d += self.hidden_size * (self.out_dtype or torch.half).itemsize
        overhead_s = 0
        for cl in self.cache_layers:
            overhead_s += cl.overhead_size()
        overhead_s += 2 * self.num_q_heads * self.head_dim * torch.half.itemsize  # q, o
        overhead_s += 2 * self.num_kv_heads * self.head_dim * torch.half.itemsize  # k, v
        recons = max(
            self.q_proj.recons_size(),
            self.k_proj.recons_size(),
            self.v_proj.recons_size(),
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
        # TODO: Account for flash-attn temp VRAM usage
        tpa = TPAllocation(
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
        return [tpa]


    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."

        def _export(child):
            nonlocal producer
            return child.tp_export(plan, producer) if child is not None else None

        return {
            "cls": Attention,
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
            },
            "num_kv_heads": self.num_kv_heads,
            **{name: _export(getattr(self, name, None)) for name in (
                "q_norm",
                "k_norm",
                "q_proj",
                "k_proj",
                "v_proj",
                "kv_proj",
                "o_proj",
            )},
            "device": self.device,
            "cache_layers": [
                cl.tp_export(plan) for cl in self.cache_layers
            ],
            "n_gqa": self.num_q_heads // self.num_kv_heads
        }


    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        key = exported["kwargs"]["key"]
        head_dim = exported["kwargs"]["head_dim"]
        n_gqa = exported["n_gqa"]
        device = local_context["device"]
        first, last, unit = plan[key]
        assert unit == "heads"
        num_kv_heads = last - first
        num_q_heads = num_kv_heads * n_gqa

        q_split = (True, first * head_dim * n_gqa, last * head_dim * n_gqa) \
            if num_kv_heads else None
        kv_split = (True, first * head_dim, last * head_dim) \
            if num_kv_heads else None
        o_split = (False, first * head_dim * n_gqa, last * head_dim * n_gqa) \
            if num_kv_heads else None
        norm_q_split = (first * n_gqa, last * n_gqa) \
            if num_kv_heads else None
        norm_k_split = (first, last) \
            if num_kv_heads else None

        # def _import(name):
        #     nonlocal exported, plan
        #     return exported[name]["cls"].tp_import(local_context, exported[name], plan) \
        #         if exported.get(name) else None

        def _import_split(name, split):
            nonlocal exported, plan
            return exported[name]["cls"].tp_import_split(local_context, exported[name], plan, split) \
                if split and exported.get(name) else None

        module = Attention(
            config = None,
            **exported["kwargs"],
            num_q_heads = num_q_heads,
            num_kv_heads = num_kv_heads,
            q_norm = _import_split("q_norm", norm_q_split),
            k_norm = _import_split("k_norm", norm_k_split),
            q_proj = _import_split("q_proj", q_split),
            k_proj = _import_split("k_proj", kv_split),
            v_proj = _import_split("v_proj", kv_split),
            kv_proj = _import_split("kv_proj", kv_split),
            o_proj = _import_split("o_proj", o_split),
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
        if not kwargs.get("skip_reduction"):
            module.tp_reduce = True
        module.load_local(device)
        torch.cuda.synchronize()
        return module