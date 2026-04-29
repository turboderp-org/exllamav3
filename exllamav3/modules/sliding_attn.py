from __future__ import annotations
from typing_extensions import override
import torch
from ..cache import CacheableState
from ..model.config import Config
from ..util.rope import RopeSettings, RoPE
from ..util.tensor import get_for_device, to2, g_tensor_cache
from . import Module, Linear, RMSNorm, LayerNorm
from ..constants import PAGE_SIZE
from flash_attn import flash_attn_func, flash_attn_with_kvcache, flash_attn_varlen_func
from .multilinear import MultiLinear
from ..ext import exllamav3_ext as ext
from ..util import profile_opt

"""
Dedicated attention layer for SWA layers. Maintains recurrent state instead of KV cache
"""

class SWA_RecurrentState(CacheableState):
    def __init__(
        self,
        position: int | None = 0,
        positions: list[int] | None = None,
        k_state: torch.Tensor | None = None,
        v_state: torch.Tensor | None = None,
        batched = False,
        batch: list[SWA_RecurrentState] | None = None,
        window_beg: int = 0,
        attn_window_size: int = None,
        kept_window_size: int = None,
    ):
        super().__init__()
        self.position = position
        self.positions = positions
        self.k_state = k_state
        self.v_state = v_state
        self.batched = batched
        self.batch = batch
        self.window_beg = window_beg
        self.attn_window_size = attn_window_size
        self.kept_window_size = kept_window_size

    @override
    def stash(self):
        # TODO: Option to preallocate and pin space for stashed states
        assert isinstance(self.k_state, torch.Tensor)
        return SWA_RecurrentState(
            self.position,
            self.positions,
            self.k_state.cpu(),
            self.v_state.cpu(),
            window_beg = self.window_beg,
            attn_window_size = self.attn_window_size,
            kept_window_size = self.kept_window_size,
        )

    @override
    def unstash(self, device, trim_position):
        trim = self.position - trim_position
        assert 0 <= trim <= self.get_cachable_interval()
        k = self.k_state
        v = self.v_state
        assert self.positions is None
        return SWA_RecurrentState(
            trim_position,
            self.positions,
            k.to(device, non_blocking = True),
            v.to(device, non_blocking = True),
            window_beg = self.window_beg,
            attn_window_size = self.attn_window_size,
            kept_window_size = self.kept_window_size,
        )

    @override
    def get_size(self):
        if self.k_state is None:
            return 0
        return (
            self.k_state.element_size() * self.k_state.numel() +
            self.v_state.element_size() * self.v_state.numel()
        )

    @override
    def get_cachable_interval(self):
        # As long as the start of the context remains, we can reconstruct back to position 0
        if self.position <= self.kept_window_size:
            return self.position
        # Otherwise we can reconstruct up to the overprovisioned interval
        return self.kept_window_size - self.attn_window_size

    def collect_batch(self, batch: list[SWA_RecurrentState]):
        assert len(batch) > 1
        return SWA_RecurrentState(
            None,
            None,
            None,
            None,
            batched = True,
            batch = batch,
            attn_window_size = self.attn_window_size,
            kept_window_size = self.kept_window_size,
        )

    def distribute_batch(self, batch: list[SWA_RecurrentState]):
        assert self.batched
        # .forward() has already updated each batch item separately

    @override
    def reset(self):
        self.k_state = None
        self.v_state = None
        self.window_beg = 0
        self.position = 0

    @override
    def force_position(self, position: int):
        self.position = position
        self.window_beg = (position + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE - self.attn_window_size
        self.window_beg = max(self.window_beg, 0)

    @override
    def clone(self):
        return SWA_RecurrentState(
            self.position,
            self.positions,
            self.k_state.clone() if self.k_state is not None else None,
            self.v_state.clone() if self.v_state is not None else None,
            self.batched,
            self.batch,
            self.window_beg,
            self.attn_window_size,
            self.kept_window_size,
        )

    @override
    def rewind(self, count: int):
        assert not self.batched
        assert count <= self.position - self.window_beg
        self.position -= count


class SlidingAttention(Module):

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
        key_g: str | None = None,
        qmap: str | None = None,
        out_dtype: torch.dtype | None = None,
        sliding_window: int = -1,
        sliding_window_overp: int = 2 * PAGE_SIZE,
        logit_softcapping: float = 0.0,
        q_norm: RMSNorm | LayerNorm | None = None,
        k_norm: RMSNorm | LayerNorm | None = None,
        v_norm: RMSNorm | LayerNorm | None = None,
        q_proj: Linear | Module | None = None,
        k_proj: Linear | Module | None = None,
        v_proj: Linear | Module | None = None,
        o_proj: Linear | Module | None = None,
        g_proj: Linear | Module | None = None,
        post_rope_norm: bool = False,
        full_gate: bool = False,
        select_hq_bits: int = 0,
    ):
        super().__init__(config, key, None)
        assert sliding_window > 0

        self.q_priority = 2 + select_hq_bits
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.gqa = (num_q_heads != num_kv_heads)
        self.sm_scale = sm_scale or self.head_dim ** (-0.5)
        self.rope_settings = rope_settings
        self.rope = None
        self.out_dtype = out_dtype
        self.sliding_window = sliding_window
        self.sliding_window_overp = sliding_window_overp
        self.kv_state_size = sliding_window + sliding_window_overp
        self.logit_softcapping = logit_softcapping
        self.post_rope_norm = post_rope_norm
        self.full_gate = full_gate
        self.stage_k = None
        self.stage_v = None

        if post_rope_norm:
            assert q_norm is None and k_norm is None, \
                "Post-RoPE norm only supported without weights"

        if self.num_kv_heads == 0:
            return

        # Create q, k, v projections
        fkey, frange_q, frange_k, frange_v = None, None, None, None

        if key_q or frange_q:
            f = 1
            self.q_proj = Linear(
                config,
                f"{key}.{key_q}",
                hidden_size,
                num_q_heads * head_dim * f,
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

        if key_k or frange_k:
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
                qmap =  qmap + ".o" if qmap is not None else None,
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

        # Register v norm
        if v_norm:
            self.v_norm = v_norm
            self.register_submodule(self.v_norm)
        else:
            self.v_norm = None

        # Register headwise gate
        if key_g:
            gate_features = num_q_heads * head_dim if full_gate else num_q_heads
            _qmap = ".input" if full_gate else None
            self.g_proj = Linear(
                config,
                f"{key}.{key_g}",
                hidden_size,
                gate_features,
                qmap = _qmap,
                out_dtype = torch.half,
                pad_to = 1,
                select_hq_bits = select_hq_bits,
            )
            self.headwise_gate = not full_gate
            self.register_submodule(self.g_proj)
        else:
            if g_proj:
                self.g_proj = g_proj
                self.headwise_gate = not full_gate
                self.register_submodule(self.g_proj)
            else:
                self.g_proj = None
                self.headwise_gate = False

        self.caps.update({
            "recurrent_cache": True,
            "sliding_window_overp": sliding_window_overp
        })

        self.multi_kv = None
        self.multi_qg = None
        self.tp_reduce = False

        self.q_norm_tensor = None
        self.k_norm_tensor = None

        self.has_split_cache = False

        self.prealloc_qgh_1 = None
        self.prealloc_qg_1 = None
        self.prealloc_kvh_1 = None
        self.prealloc_kv_1 = None

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
            self.prealloc_kvh_1 = g_tensor_cache.get(device, (2, 1, self.hidden_size), torch.half, "kvh_1")
            self.prealloc_kv_1 = g_tensor_cache.get(device, (2, 1, self.num_kv_heads * self.head_dim), torch.half, "kv_1")

        # Test if Q and G proj can be fused
        if (
            self.g_proj is not None and
            device != torch.device("cpu") and
            self.q_proj.quant_type == "exl3" and
            self.g_proj.quant_type == "exl3" and
            self.q_proj.out_features == self.g_proj.out_features and
            self.q_proj.inner.K == self.g_proj.inner.K and
            self.q_proj.inner.bias is None and
            self.g_proj.inner.bias is None
        ):
            self.multi_qg = MultiLinear(self. device, [self.q_proj, self.g_proj])
            self.prealloc_qgh_1 = g_tensor_cache.get(device, (2, 1, self.hidden_size), torch.half, "qgh_1")
            self.prealloc_qg_1 = g_tensor_cache.get(device, (2, 1, self.num_q_heads * self.head_dim), torch.half, "qg_1")

        # Head norm
        if self.q_norm and isinstance(self.q_norm, RMSNorm) and not self.q_norm.span_heads:
            self.q_norm_tensor = self.q_norm.weight.data
            self.k_norm_tensor = self.k_norm.weight.data


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)
        max_chunk_size = kwargs.get("max_chunk_size", 2048)
        self.stage_k = g_tensor_cache.get(device, (1, max_chunk_size + self.kv_state_size, self.num_kv_heads, self.head_dim), torch.half, "swa_k")
        self.stage_v = g_tensor_cache.get(device, (1, max_chunk_size + self.kv_state_size, self.num_kv_heads, self.head_dim), torch.half, "swa_v")
        self.load_local(device, **kwargs)


    @override
    def unload(self):
        super().unload()

        self.rope = None

        if self.multi_kv is not None:
            self.multi_kv.unload()
            self.multi_kv = None

        if self.multi_qg is not None:
            self.multi_qg.unload()
            self.multi_qg = None

        self.q_norm_tensor = None
        self.k_norm_tensor = None
        self.stage_k = None
        self.stage_v = None

        self.prealloc_qgh_1 = None
        self.prealloc_qg_1 = None
        self.prealloc_kvh_1 = None
        self.prealloc_kv_1 = None


    def new_recurrent_state(self):
        return SWA_RecurrentState(
            attn_window_size = self.sliding_window,
            kept_window_size = self.kv_state_size,
        )


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

        if self.multi_qg is None or bsz * q_len > 32:
            q = self.q_proj.forward(x, params)
            if self.g_proj:
                g = self.g_proj.forward(x, params)
            else:
                g = None
        else:
            x = x.view(1, bsz * q_len, dim)
            if bsz * q_len == 1:
                qgh = self.prealloc_qgh_1
                qg = self.prealloc_qg_1
            else:
                qgh = torch.empty((2, bsz * q_len, dim), dtype = torch.half, device = x.device)
                qg = torch.empty((2, bsz * q_len, self.num_q_heads * self.head_dim), dtype = torch.half, device = x.device)
            ext.exl3_mgemm(
                x,
                self.multi_qg.ptrs_trellis,
                qg,
                self.multi_qg.ptrs_suh,
                qgh,
                self.multi_qg.ptrs_svh,
                None,
                None,
                self.multi_qg.K,
                -1,
                self.multi_qg.mcg,
                self.multi_qg.mul1,
                -1,
                -1,
                0
            )
            q = qg[0].view(bsz, q_len, self.num_q_heads * self.head_dim)
            g = qg[1].view(bsz, q_len, self.num_q_heads * self.head_dim)

        if self.multi_kv is None or bsz * q_len > 32:
            k = self.k_proj.forward(x, params)
            v = self.v_proj.forward(x, params)

        else:
            x = x.view(1, bsz * q_len, dim)
            if bsz * q_len == 1:
                kvh = self.prealloc_kvh_1
                kv = self.prealloc_kv_1
            else:
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

        q = q.view(bsz, q_len, self.num_q_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        if self.v_norm is not None:
            v = self.v_norm.forward(v, params, out_dtype = torch.half)

        return q, k, v, g


    def project_o(self, o: torch.Tensor, bsz: int, seqlen: int, params: dict) -> torch.Tensor:
        o = o.reshape(bsz, seqlen, self.num_q_heads * self.head_dim)
        x = self.o_proj.forward(o, params)
        return x


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

        if self.q_norm:
            if not self.rope or self.q_norm_tensor is None:
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
                self.post_rope_norm
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

        if self.headwise_gate: o *= g.sigmoid().unsqueeze(-1)
        o = o.view((bsz, seqlen, self.num_q_heads * self.head_dim))
        if self.full_gate: o *= g.sigmoid()
        o = self.project_o(o, bsz, seqlen, params)
        return o


    def decode_flash_attn(
        self,
        x: torch.Tensor,
        bsz: int,
        seqlen: int,
        params: dict,
    ):
        position = params.get("position", 0)
        positions = get_for_device(params, "positions", self.device, None)
        position_ids = get_for_device(params, "position_ids", self.device, None)
        inv_freq = get_for_device(params, "inv_freq", self.device, None)
        causal = params.get("causal", True)
        non_causal_spans = params.get("non_causal_spans")

        q, k, v, g = self.project_qkv(x, params)

        if self.q_norm:
            if not self.rope or self.q_norm_tensor is None:
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
                self.post_rope_norm
            )

        # Get or initialize recurrent state, may be list of tensors if some batch items haven't filled the window yet
        # TODO: Currently, batch is a list of split caches processed at bsz 1, since caches are large and constructing
        #       temporary batch tensors causes excessive fragmentation and unpredictable VRAM usage
        rsb = params.get("recurrent_states")
        assert rsb is not None
        rsb = rsb[self.layer_idx, params.get("layer_instance", 0)]
        if rsb.batched:
            rsb = rsb.batch
        else:
            rsb = [rsb]
        assert len(rsb) == bsz

        ro = []
        for i, rs in enumerate(rsb):
            q_ = q[i:i+1]
            k_ = k[i:i+1]
            v_ = v[i:i+1]

            # Create empty state if unitialized
            if rs.k_state is None:
                rs.k_state = torch.empty((1, self.kv_state_size, self.num_kv_heads, self.head_dim), dtype = torch.half, device = x.device)
                rs.v_state = torch.empty((1, self.kv_state_size, self.num_kv_heads, self.head_dim), dtype = torch.half, device = x.device)

            k_state = rs.k_state
            v_state = rs.v_state

            # Physical cache state
            cache_beg = rs.window_beg
            cache_pos = rs.position - cache_beg
            cache_wpos = max(cache_pos - self.sliding_window, 0) // PAGE_SIZE * PAGE_SIZE
            cache_hot = cache_pos - cache_wpos

            def pad(z):
                return (z + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
            seqlen_pad = pad(seqlen)

            # Case 1: new KV fits in cache without shifting
            if cache_pos + seqlen <= self.kv_state_size:
                shift = 0
                wshift = 0
                cache_seqlens = cache_pos
                cache_seqlens_nc = cache_pos
                temp_cache = False

            # Case 2: new KV fits in cache if we shift
            elif seqlen_pad <= self.sliding_window_overp - PAGE_SIZE:
                shift = seqlen_pad
                wshift = seqlen_pad
                k_state[:, :-shift, :, :].copy_(k_state[:, shift:, :, :].clone())
                v_state[:, :-shift, :, :].copy_(v_state[:, shift:, :, :].clone())
                cache_seqlens = cache_pos - shift
                cache_seqlens_nc = cache_pos - shift
                temp_cache = False

            # Case 3: temporary cache+kv fits in preallocated buffer
            elif cache_hot + seqlen_pad <= self.stage_k.shape[1]:
                if cache_hot:
                    self.stage_k[:, :cache_hot].copy_(k_state[:, cache_wpos : cache_pos])
                    self.stage_v[:, :cache_hot].copy_(v_state[:, cache_wpos : cache_pos])
                self.stage_k[:, cache_hot : cache_hot + seqlen].copy_(k_)
                self.stage_v[:, cache_hot : cache_hot + seqlen].copy_(v_)
                k_state = self.stage_k
                v_state = self.stage_v
                k_ = None
                v_ = None
                cache_seqlens = cache_hot + seqlen
                cache_seqlens_nc = cache_hot
                shift = pad(cache_seqlens) - self.kv_state_size
                wshift = shift + cache_wpos
                temp_cache = True
                # assert shift > 0

            # Case 4: create new temp tensors
            else:
                k_state = torch.cat((k_state[:, cache_wpos : cache_pos], k_), dim = 1)
                v_state = torch.cat((v_state[:, cache_wpos : cache_pos], v_), dim = 1)
                k_ = None
                v_ = None
                cache_seqlens = cache_hot + seqlen
                cache_seqlens_nc = cache_hot
                shift = pad(cache_seqlens) - self.kv_state_size
                wshift = shift + cache_wpos
                temp_cache = True
                assert shift > 0

            if not non_causal_spans:
                o = flash_attn_with_kvcache(
                    q = q_,
                    k = k_,
                    v = v_,
                    k_cache = k_state,
                    v_cache = v_state,
                    causal = causal,
                    cache_seqlens = cache_seqlens,
                    softmax_scale = self.sm_scale,
                    window_size = (self.sliding_window, 0),
                    softcap = self.logit_softcapping
                )
            else:
                o = []
                for a, b, c in non_causal_spans:
                    l = b - a
                    if k_ is None:
                        cache_seqlens_nc += l
                    o_ = flash_attn_with_kvcache(
                        q = q_[:, a : b],
                        k = k_[:, a : b] if k_ is not None else None,
                        v = v_[:, a : b] if v_ is not None else None,
                        k_cache = k_state,
                        v_cache = v_state,
                        causal = not c,
                        cache_seqlens = cache_seqlens_nc,
                        softmax_scale = self.sm_scale,
                        window_size = (
                            (max(self.sliding_window, l), l - 1)
                            if c else
                            (self.sliding_window, 0)
                        ),
                        softcap = self.logit_softcapping
                    )
                    if k_ is not None:
                        cache_seqlens_nc += l
                    o.append(o_)
                o = torch.cat(o, dim = 1)
            ro.append(o)

            # If KV not updated inplace, copy tail of extended temp tensors back into state
            # Copy inplace to keep tensors contiguous and to avoid fragmentation
            if temp_cache:
                rs.k_state.copy_(k_state[:, shift : shift + self.kv_state_size, :, :])
                rs.v_state.copy_(v_state[:, shift : shift + self.kv_state_size, :, :])

            # Advance state
            rs.position += seqlen
            rs.window_beg += wshift

            assert rs.k_state.shape[1] == self.kv_state_size

        # Concat attn outputs for batch
        o = torch.cat(ro, dim = 0) if len(ro) > 1 else o
        assert o.shape[1] == seqlen

        if self.headwise_gate: o *= g.sigmoid().unsqueeze(-1)
        o = o.view((bsz, seqlen, self.num_q_heads * self.head_dim))
        if self.full_gate: o *= g.sigmoid()
        o = self.project_o(o, bsz, seqlen, params)
        return o
