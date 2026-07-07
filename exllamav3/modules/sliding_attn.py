from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config
from ..util.rope import RopeSettings, RoPE
from ..util.tensor import get_for_device, to2, g_tensor_cache
from . import Module, Linear, RMSNorm, LayerNorm
from ..constants import PAGE_SIZE
from flash_attn import flash_attn_func, flash_attn_with_kvcache, flash_attn_varlen_func
from .multilinear import MultiLinear
from ..ext import exllamav3_ext as ext
from ..cache import Cache
from ..util import profile_opt

MAX_PRE_SEQ = 8192

class SWAState:

    def __init__(
        self,
        cache: Cache,
        slot: int,
        position: int,
        clear: bool = True,
        stashed: dict = None,
        test_state: bool = False,
    ):
        assert test_state or position == 0 or stashed is not None, \
            "State must be new, restored from checkpoint or marked as a test state."

        self.slot = slot
        self.position = position
        self.cache = cache
        self.last_history = 0
        self.window_beg = position // PAGE_SIZE * PAGE_SIZE
        self.wshift = 0

        if stashed is not None:
            self.unstash(stashed)

        self.checkpoint_size = sum(
            l.get_checkpoint_size()
            for l in self.cache.get_all_recurrent_layers().values()
        )


    def free(self):
        self.cache.release_state(self)


    def rewind(self, num_tokens: int):
        assert num_tokens <= self.position - self.window_beg
        self.position -= num_tokens
        self.last_history = 0


    def stash(self):
        stashed = {
            "position": self.position,
            "checkpoint_size": self.checkpoint_size,
            "window_beg": self.window_beg,
        }
        for k, l in self.cache.get_all_recurrent_layers().items():
            stashed[k] = l.stash(self.slot, self.position)
        return stashed


    def unstash(self, stashed: dict):
        assert self.position == stashed["position"]
        self.window_beg = stashed["window_beg"]
        for k, l in self.cache.get_all_recurrent_layers().items():
            l.unstash(self.slot, stashed[k], self.position)


    def post_advance(self):
        self.window_beg += self.wshift


    def reset(self):
        self.position = 0
        self.window_beg = 0
        self.wshift = 0

class SWALayerState:

    def __init__(
        self,
        module: SlidingAttention,
        max_batch_size: int,
        max_history: int,
        cache_id: int,
    ):
        assert module.kv_state_size % 256 == 0
        self.module = module
        self.k_state = torch.empty(
            (max_batch_size, module.kv_state_size, module.num_kv_heads, module.head_dim),
            dtype = torch.half,
            device = "meta"
        )
        self.v_state = torch.empty(
            (max_batch_size, module.kv_state_size, module.num_kv_heads, module.head_dim),
            dtype = torch.half,
            device = "meta"
        )
        self.device = None
        self.max_history = max_history
        self.max_batch_size = max_batch_size
        self.cache_id = cache_id


    def get_checkpoint_size(self):
        return (
            self.module.sliding_window * self.module.num_kv_heads * self.module.head_dim * 2 +
            self.module.sliding_window * self.module.num_kv_heads * self.module.head_dim * 2
        )


    def alloc(self, device):
        self.k_state = torch.empty_like(self.k_state, device = device)
        self.v_state = torch.empty_like(self.v_state, device = device)
        self.device = device


    def free(self):
        self.k_state = torch.empty_like(self.k_state, device = "meta")
        self.v_state = torch.empty_like(self.v_state, device = "meta")
        self.device = None


    def clear(self, idx: int):
        pass


    def get_state_tensors(self):
        return (
            self.k_state,
            self.v_state,
        )


    def rewind(self, slot: int, last_history: int, num_tokens: int):
        pass


    def stash(self, slot, position):
        b = min(self.module.kv_state_size, position)
        a = max(0, b - self.module.sliding_window)
        return (
            self.k_state[slot, a:b].cpu(),
            self.v_state[slot, a:b].cpu()
        )


    def unstash(self, slot, stashed, position):
        b = min(self.module.kv_state_size, position)
        a = max(0, b - self.module.sliding_window)
        k, v = stashed
        self.k_state[slot, a:b].copy_(k)
        self.v_state[slot, a:b].copy_(v)


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
        self.layer_state_cls = SWALayerState

        self.multi_kv = None
        self.multi_qg = None

        self.q_norm_tensor = None
        self.k_norm_tensor = None

        self.has_split_cache = False

        self.prealloc_qgh_1 = None
        self.prealloc_qg_1 = None
        self.prealloc_kvh_1 = None
        self.prealloc_kv_1 = None
        self.pre_seqs = None

        self.recurrent_layers = []
        self.tp_recurrent_lookup = {}
        self.tp_reduce = False


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

        # Recurrent states
        for rl in self.recurrent_layers:
            rl.alloc(device)

        if self.rope_settings:
            self.rope = RoPE(
                device,
                self.rope_settings,
            )

        # Test if K and V proj can be fused
        if (
            not self.config.infer_params.no_reconstruct and
            device != torch.device("cpu") and
            self.k_proj.quant_type == "exl3" and
            self.v_proj.quant_type == "exl3" and
            self.k_proj.out_features == self.v_proj.out_features and
            self.k_proj.inner.K == self.v_proj.inner.K and
            self.k_proj.inner.bias is None and
            self.v_proj.inner.bias is None and
            self.config.infer_params.use_mgemm(self.k_proj.inner.K, self.k_proj.out_features)
        ):
            self.multi_kv = MultiLinear(self. device, [self.k_proj, self.v_proj])
            self.prealloc_kvh_1 = g_tensor_cache.get(device, (2, 1, self.hidden_size), torch.half, "kvh_1")
            self.prealloc_kv_1 = g_tensor_cache.get(device, (2, 1, self.num_kv_heads * self.head_dim), torch.half, "kv_1")

        # Test if Q and G proj can be fused
        if (
            not self.config.infer_params.no_reconstruct and
            self.g_proj is not None and
            device != torch.device("cpu") and
            self.q_proj.quant_type == "exl3" and
            self.g_proj.quant_type == "exl3" and
            self.q_proj.out_features == self.g_proj.out_features and
            self.q_proj.inner.K == self.g_proj.inner.K and
            self.q_proj.inner.bias is None and
            self.g_proj.inner.bias is None and
            self.config.infer_params.use_mgemm(self.q_proj.inner.K, self.q_proj.out_features)
        ):
            self.multi_qg = MultiLinear(self. device, [self.q_proj, self.g_proj])
            self.prealloc_qgh_1 = g_tensor_cache.get(device, (2, 1, self.hidden_size), torch.half, "qgh_1")
            self.prealloc_qg_1 = g_tensor_cache.get(device, (2, 1, self.num_q_heads * self.head_dim), torch.half, "qg_1")

        # Head norm
        if self.q_norm and isinstance(self.q_norm, RMSNorm) and not self.q_norm.span_heads:
            self.q_norm_tensor = self.q_norm.weight.data
            self.k_norm_tensor = self.k_norm.weight.data

        # Create seqlens for flash_attn_with_kvcache to avoid extra torch.full on launch
        self.pre_seqs = torch.arange(MAX_PRE_SEQ, dtype = torch.int, device = device)


    @override
    def load(self, device: torch.Device, **kwargs):
        super().load(device, **kwargs)
        max_chunk_size = kwargs.get("max_chunk_size", 2048)
        self.stage_k = g_tensor_cache.get(device, (1, max_chunk_size + self.kv_state_size, self.num_kv_heads, self.head_dim), torch.half, "swa_stage_k")
        self.stage_v = g_tensor_cache.get(device, (1, max_chunk_size + self.kv_state_size, self.num_kv_heads, self.head_dim), torch.half, "swa_stage_v")
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
        self.pre_seqs = None


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

        # Get recurrent state tensors
        rsg = params.get("recurrent_states")
        assert rsg is not None, "SlidingAttention.forward() called in flash_attn mode with no recurrent states"
        layer_instance = (self.layer_idx, params.get("layer_instance", 0))
        rsl = rsg[0].cache.get_recurrent_layer(layer_instance)
        k_states, v_states = rsl.get_state_tensors()

        # TODO: Currently iterates over batch items, but could use paged attn to batch items together, at least
        #       when no noncausal spans
        ro = []
        for i, rs in enumerate(rsg):
            q_ = q[i:i+1]
            k_ = k[i:i+1]
            v_ = v[i:i+1]

            k_state = k_states[rs.slot].unsqueeze(0)
            v_state = v_states[rs.slot].unsqueeze(0)

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
                rs.wshift = 0
                cache_seqlens = cache_pos
                cache_seqlens_nc = cache_pos
                temp_cache = False

            # Case 2: new KV fits in cache if we shift
            elif seqlen_pad <= self.sliding_window_overp - PAGE_SIZE:
                shift = seqlen_pad
                rs.wshift = seqlen_pad
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
                c_k_state = k_state
                c_v_state = v_state
                k_state = self.stage_k
                v_state = self.stage_v
                k_ = None
                v_ = None
                cache_seqlens = cache_hot + seqlen
                cache_seqlens_nc = cache_hot
                shift = pad(cache_seqlens) - self.kv_state_size
                rs.wshift = shift + cache_wpos
                temp_cache = True
                # assert shift > 0

            # Case 4: create new temp tensors
            else:
                c_k_state = k_state
                c_v_state = v_state
                k_state = torch.cat((k_state[:, cache_wpos : cache_pos], k_), dim = 1)
                v_state = torch.cat((v_state[:, cache_wpos : cache_pos], v_), dim = 1)
                k_ = None
                v_ = None
                cache_seqlens = cache_hot + seqlen
                cache_seqlens_nc = cache_hot
                shift = pad(cache_seqlens) - self.kv_state_size
                rs.wshift = shift + cache_wpos
                temp_cache = True
                assert shift > 0

            if not non_causal_spans:
                if cache_seqlens < MAX_PRE_SEQ:
                    cache_seqlens = self.pre_seqs[cache_seqlens : cache_seqlens + 1]

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
                c_k_state.copy_(k_state[:, shift : shift + self.kv_state_size, :, :])
                c_v_state.copy_(v_state[:, shift : shift + self.kv_state_size, :, :])
                k_state = c_k_state
                v_state = c_v_state

            assert k_state.shape[1] == self.kv_state_size

        # Concat attn outputs for batch
        o = torch.cat(ro, dim = 0) if len(ro) > 1 else o
        assert o.shape[1] == seqlen

        if self.headwise_gate: ext.mul_sigmoid_broadcast_(o, g)
        o = o.view((bsz, seqlen, self.num_q_heads * self.head_dim))
        if self.full_gate: ext.mul_sigmoid_(o, g)

        o = self.project_o(o, bsz, seqlen, params)
        return o
