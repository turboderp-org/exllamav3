from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache
from .. import LayerNorm
from ...model.config import Config
from ...modules import Module, Linear, RMSNorm
from ...util.rope import RopeSettings, RoPE
from ...util.tensor import get_for_device, to2

class DFlashInputLayer(Module):

    def __init__(
        self,
        config: Config,
        key: str,
        key_norm: str,
        hidden_size: int,
        target_state_size: int,
        mask_token_id: int,
        rms_norm_eps: float,
        native_draft_len: int,
        out_dtype: torch.dtype | None = torch.float,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "DFlashInputLayer"
        self.qmap = qmap
        self.key = key
        self.hidden_size = hidden_size
        self.target_state_size = target_state_size
        self.out_dtype = out_dtype
        self.native_draft_len = native_draft_len

        self.proj = Linear(
            config = config,
            key = f"{key}",
            in_features = self.target_state_size,
            out_features = self.hidden_size,
            qmap = (qmap + ".input") if qmap else None,
            out_dtype = out_dtype,
            pad_to = 1
        )

        self.norm = RMSNorm(
            config = config,
            key = f"{key_norm}",
            rms_norm_eps = rms_norm_eps,
            out_dtype = out_dtype,
        )

        self.register_submodule(self.proj)
        self.register_submodule(self.norm)

        self.mask_token_id = mask_token_id

        # Populated by attach_to()
        self.embedding = None
        self.caps.update({"x_cpu": True})


    def optimizer_targets(self):
        raise NotImplementedError()


    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)


    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        return x


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ):
        bsz, seqlen = x.shape
        noise_mask = torch.full((bsz, self.native_draft_len - 1), self.mask_token_id, dtype = torch.long)
        x = torch.cat((x, noise_mask), dim = -1)
        x = self.embedding().forward(x, params)
        return x


class DFlashAttention(Module):

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
        key_q: str | None = None,
        key_k: str | None = None,
        key_v: str | None = None,
        key_o: str | None = None,
        key_fused_qkv: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        sliding_window: int = -1,
        q_norm: RMSNorm | LayerNorm | None = None,
        k_norm: RMSNorm | LayerNorm | None = None,
        q_proj: Linear | Module | None = None,
        k_proj: Linear | Module | None = None,
        v_proj: Linear | Module | None = None,
        o_proj: Linear | Module | None = None,
        select_hq_bits: int = 0,
    ):
        super().__init__(config, key, None)

        self.q_priority = 2 + select_hq_bits
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.gqa = (num_q_heads != num_kv_heads)
        self.rope_settings = rope_settings
        self.rope = None
        self.out_dtype = out_dtype
        self.sliding_window = sliding_window
        self.sliding_window_0 = min(sliding_window, 0)

        # Create q, k, v projections
        if key_fused_qkv:
            fkey = f"{key}.{key_fused_qkv}"
            frange_q = (0, num_q_heads * head_dim)
            frange_k = (frange_q[1], frange_q[1] + num_kv_heads * head_dim)
            frange_v = (frange_k[1], frange_k[1] + num_kv_heads * head_dim)
        else:
            fkey, frange_q, frange_k, frange_v = None, None, None, None

        if key_q:
            self.q_proj = Linear(
                config,
                f"{key}.{key_q}",
                hidden_size,
                num_q_heads * head_dim,
                qmap = qmap + ".input" if qmap is not None else None,
                fkey = fkey,
                frange = frange_q,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".qkv",
            )
            self.register_submodule(self.q_proj)
        else:
            assert q_proj
            self.q_proj = q_proj
            self.register_submodule(self.q_proj)

        if key_k:
            assert key_v or frange_v
            self.k_proj = Linear(
                config,
                f"{key}.{key_k}",
                hidden_size,
                num_kv_heads * head_dim,
                qmap =  qmap + ".input" if qmap is not None else None,
                fkey = fkey,
                frange = frange_k,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".qkv",
            )
            self.v_proj = Linear(
                config,
                f"{key}.{key_v}",
                hidden_size,
                num_kv_heads * head_dim,
                qmap =  qmap + ".input" if qmap is not None else None,
                fkey = fkey,
                frange = frange_v,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".qkv",
            )
            self.register_submodule(self.k_proj)
            self.register_submodule(self.v_proj)
        else:
            assert k_proj and v_proj
            self.k_proj = k_proj
            self.v_proj = v_proj
            self.register_submodule(self.k_proj)
            self.register_submodule(self.v_proj)

        # Create o proj
        if key_o:
            self.o_proj = Linear(
                config,
                f"{key}.{key_o}",
                num_q_heads * head_dim,
                hidden_size,
                qmap = qmap + ".o" if qmap is not None else None,
                out_dtype = out_dtype,
                select_hq_bits = select_hq_bits,
                qgroup = key + ".o" if qmap is not None else None,
            )
            self.register_submodule(self.o_proj)
        else:
            assert o_proj
            self.o_proj = o_proj
            self.register_submodule(self.o_proj)

        # Register q/k norms
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

        self.q_norm_tensor = None
        self.k_norm_tensor = None


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

        # Head norm
        if self.q_norm and isinstance(self.q_norm, RMSNorm) and not self.q_norm.span_heads:
            self.q_norm_tensor = self.q_norm.weight.data
            self.k_norm_tensor = self.k_norm.weight.data


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)
        self.load_local(device, **kwargs)


    @override
    def unload(self):
        super().unload()

        for cl in self.cache_layers:
            cl.free()

        self.rope = None

        self.q_norm_tensor = None
        self.k_norm_tensor = None


    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        cache = params.get("cache")
        block_table = get_for_device(params, "block_table", self.device)
        cache_seqlens = get_for_device(params, "cache_seqlens", self.device)
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        inv_freq = get_for_device(params, "inv_freq", self.device, None)

        bsz, seqlen, dim = x.shape
        cache_k, cache_v = cache.get_layer(self.layer_idx, cache_seqlens, block_table, -1)

        q = self.q_proj.forward(x, params)
        k = self.k_proj.forward(x, params)
        v = self.v_proj.forward(x, params)
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
                False,
            )

        o = flash_attn_with_kvcache(
            q = q,
            k = k,
            v = v,
            k_cache = cache_k,
            v_cache = cache_v,
            block_table = block_table,
            cache_seqlens = cache_seqlens,
            window_size = (self.sliding_window, self.sliding_window_0),
            causal = False,
        )

        o = o.reshape(bsz, seqlen, self.num_q_heads * self.head_dim)
        o = self.o_proj.forward(o, params)

        cache.update_layer(self.layer_idx, cache_seqlens, block_table, cache_k, cache_v, seqlen)
        return o
