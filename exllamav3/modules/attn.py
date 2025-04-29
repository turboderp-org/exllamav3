from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..models import Config
from ..util.rope import RopeSettings, RoPE
from ..util.tensor import get_for_device, to2
from . import Module, Linear, RMSNorm
from ..device import get_device_context, release_device_context
from ..constants import PAGE_SIZE
from ..cache import Cache
from flash_attn import flash_attn_func, flash_attn_with_kvcache

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
        cache_seqlens = params.get("cache_seqlens")
        position = params.get("position")
        positions = params.get("positions")
        position_ids = params.get("position_ids")
        assert cache_bsz >= bsz, "batch size too large for cache"
        assert cache_max_seq_len % PAGE_SIZE == 0, f"cache seq len must be a multiple of {PAGE_SIZE}"
        assert (past_len is not None) ^ (cache_seqlens is not None), "Need either past_len or cache_seqlens"
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
        config: Config,
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
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        super().__init__(config, key, None)

        self.device_context = None
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

        if key_fused_qkv:
            fkey = f"{key}.{key_fused_qkv}"
            frange_q = (0, num_q_heads * head_dim)
            frange_k = (frange_q[1], frange_q[1] + num_kv_heads * head_dim)
            frange_v = (frange_k[1], frange_k[1] + num_kv_heads * head_dim)
        else:
            fkey, frange_q, frange_k, frange_v = None, None, None, None

        self.q_proj = Linear(config, f"{key}.{key_q}", hidden_size, num_q_heads * head_dim, qmap = qmap + ".input", fkey = fkey, frange = frange_q)
        self.k_proj = Linear(config, f"{key}.{key_k}", hidden_size, num_kv_heads * head_dim, qmap =  qmap + ".input", fkey = fkey, frange = frange_k)
        self.v_proj = Linear(config, f"{key}.{key_v}", hidden_size, num_kv_heads * head_dim, qmap =  qmap + ".input", fkey = fkey, frange = frange_v)
        self.o_proj = Linear(config, f"{key}.{key_o}", num_q_heads * head_dim, hidden_size, qmap =  qmap + ".o", out_dtype = out_dtype)

        self.register_submodule(self.q_proj)
        self.register_submodule(self.k_proj)
        self.register_submodule(self.v_proj)
        self.register_submodule(self.o_proj)

        if q_norm:
            assert k_norm, "Must have both Q and K norms, or neither"
            self.q_norm = q_norm
            self.k_norm = k_norm
            self.register_submodule(self.q_norm)
            self.register_submodule(self.k_norm)
        else:
            self.q_norm = None
            self.k_norm = None

        self.caps.update({
            "kv_cache": True
        })

        self.cache_layers = []


    @override
    def load(self, device: torch.Device, **kwargs):
        self.device_context = get_device_context(self.config, device)
        super().load(device)

        for cl in self.cache_layers:
            cl.alloc(device)

        if self.rope_settings:
            self.rope = RoPE(
                device,
                self.rope_settings,
            )

        # self.join_qkv_fwd = (
        #     (self.q_proj.quant_type, self.k_proj.quant_type, self.v_proj.quant_type)
        #     == ("exl3", "exl3", "exl3")
        # )


    @override
    def unload(self):
        self.device_context = release_device_context(self.config, self.device)
        super().unload()

        for cl in self.cache_layers:
            cl.free()

        self.rope = None


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
            case "sdpa_nc":
                x = self.decode_sdpa_nc(x, bsz, seqlen, params)
            case "flash_attn":
                x = self.decode_flash_attn(x, bsz, seqlen, params)
            case "flash_attn_nc":
                x = self.decode_flash_attn_nc(x, bsz, seqlen, params)
            case _:
                raise ValueError(f"Unknown attn_mode: {attn_mode}")

        return to2(x, out_dtype, self.out_dtype)


    def project_qkv(self, x: torch.Tensor, params: dict) -> tuple:
        q = self.q_proj.forward(x, params)
        k = self.k_proj.forward(x, params)
        v = self.v_proj.forward(x, params)
        return q, k, v


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

        q, k, v = self.project_qkv(x, params)
        q = q.view(bsz, seqlen, self.num_q_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        assert self.sliding_window < 0, \
            "Torch SDPA does not support sliding window attention (SWA)"
        assert self.logit_softcapping == 0.0, \
            "Torch SDPA does not support logit softcapping"

        if self.q_norm:
            q = self.q_norm.forward(q, params, out_dtype = torch.half)
            k = self.k_norm.forward(k, params, out_dtype = torch.half)

        if self.rope:
            q, k = self.rope.apply(q, k, position, positions, position_ids)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal = causal, enable_gqa = self.gqa)
        o = o.transpose(1, 2)

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

        q, k, v = self.project_qkv(x, params)
        q = q.view(bsz, seqlen, self.num_q_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        if self.q_norm:
            q = self.q_norm.forward(q, params, out_dtype = torch.half)
            k = self.k_norm.forward(k, params, out_dtype = torch.half)

        if self.rope:
            q, k = self.rope.apply(q, k, position, positions, position_ids, in_place = True)

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
        causal = params.get("causal", True)

        q, k, v = self.project_qkv(x, params)
        q = q.view(bsz, seqlen, self.num_q_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        if self.q_norm:
            q = self.q_norm.forward(q, params, out_dtype = torch.half)
            k = self.k_norm.forward(k, params, out_dtype = torch.half)

        if self.rope:
            q, k = self.rope.apply(q, k, position, positions, position_ids, in_place = True)

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
        cache.update_layer(self.layer_idx, cache_seqlens, block_table, cache_k, cache_v, seqlen)
        o = o.view((bsz, seqlen, self.num_q_heads * self.head_dim))

        # TODO: Store updated cache layer

        o = self.project_o(o, bsz, seqlen, params)
        return o
