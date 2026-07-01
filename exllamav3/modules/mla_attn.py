from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from .module import Module
from .linear import Linear
from .attention_fn import attn_dispatch
from ..util.tensor import get_for_device

# Multi-head latent attention (DeepSeek-V2/V3 family). Low-rank q (optional) and
# kv projections with decoupled rope; QK head dim may differ from V head dim.
# With a paged cache, K/V are up-projected and cached at qk_head_dim (V padded)
# so the standard cache layout and flash-attn path apply. Without a cache, two
# full-sequence batch-1 paths: naive (up-project K/V, dtype scores) and absorbed
# (score in latent space, fp32). The absorbed path reads kv_b's raw fp16 weight,
# which is why kv_b stays unquantized.
#
# Model-specific behavior hooks in by subclassing: hook_q_lora / hook_kv_latent /
# hook_attn_out (identity here) and extra_kv (extra always-visible latent/rope
# positions, e.g. learned sinks; None here).


def _rms_std(x, w, eps):
    dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim = -1, keepdim = True) + eps)
    return (y * w.float()).to(dtype)


def _rms_kvc(x, w, eps):
    # variant: weight multiplied in dtype after the fp32 normalize
    dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim = -1, keepdim = True) + eps)
    return y.to(dtype) * w.to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat((-x2, x1), dim = -1)


def _apply_rotary(x, cos, sin):
    dtype = x.dtype
    x = x.float()
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return ((x * cos.float()) + (_rotate_half(x) * sin.float())).to(dtype)


def _apply_rotary_dtype(x, cos, sin):
    cos = cos.to(device = x.device, dtype = x.dtype)
    sin = sin.to(device = x.device, dtype = x.dtype)
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


def _interleave_pairs(x):
    # DeepSeek rope layout: [x0 x1 x2 x3 ...] -> [x0 x2 ... | x1 x3 ...]
    s = x.shape
    return x.view(*s[:-1], s[-1] // 2, 2).transpose(-1, -2).reshape(s)


class MLAAttention(Module):

    def __init__(
        self,
        config,
        key,
        layer_idx,
        hidden_size,
        num_heads,
        q_lora_rank,          # None -> direct q_proj
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        rope_theta,
        rms_norm_eps,
        qmap = None,
        sliding_window = None,
        absorbed = False,
        rope_interleave_pairs = False,
        kv_norm_dtype_mul = False,
        key_q_a = "q_a_proj",
        key_q_a_norm = "q_a_layernorm",
        key_q_b = "q_b_proj",
        key_q = "q_proj",
        key_kv_a = "kv_a_proj_with_mqa",
        key_kv_a_norm = "kv_a_layernorm",
        key_kv_b = "kv_b_proj",
        key_o = "o_proj",
        sm_scale = None,
    ):
        super().__init__(config, key, None)
        self.module_name = "MLAAttention"
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.absorbed = absorbed
        self.rope_interleave_pairs = rope_interleave_pairs
        self.kv_norm_dtype_mul = kv_norm_dtype_mul
        self.scaling = sm_scale or self.qk_head_dim ** -0.5
        self.key_q_a_norm = key_q_a_norm
        self.key_kv_a_norm = key_kv_a_norm

        # Paged cache stores up-projected K/V at qk_head_dim, V zero-padded from
        # v_head_dim, so the standard cache layout and flash-attn path apply
        self.num_kv_heads = num_heads
        self.head_dim = self.qk_head_dim
        self.cache_layers = []
        self.caps.update({"kv_cache": True})

        if q_lora_rank:
            self.q_a_proj = Linear(config, f"{key}.{key_q_a}", hidden_size, q_lora_rank, qmap = None)
            self.q_b_proj = Linear(config, f"{key}.{key_q_b}", q_lora_rank,
                                   num_heads * self.qk_head_dim, qmap = qmap + ".qb" if qmap else None)
            self.q_proj = None
            self.register_submodule(self.q_a_proj)
            self.register_submodule(self.q_b_proj)
        else:
            self.q_a_proj = None
            self.q_b_proj = None
            self.q_proj = Linear(config, f"{key}.{key_q}", hidden_size,
                                 num_heads * self.qk_head_dim, qmap = qmap + ".qb" if qmap else None)
            self.register_submodule(self.q_proj)
        self.kv_a_proj = Linear(config, f"{key}.{key_kv_a}", hidden_size,
                                kv_lora_rank + qk_rope_head_dim, qmap = None)
        self.kv_b_proj = Linear(config, f"{key}.{key_kv_b}", kv_lora_rank,
                                num_heads * (qk_nope_head_dim + v_head_dim), qmap = None)
        self.o_proj = Linear(config, f"{key}.{key_o}", num_heads * v_head_dim, hidden_size,
                             qmap = qmap + ".o" if qmap else None)
        for m in (self.kv_a_proj, self.kv_b_proj, self.o_proj):
            self.register_submodule(m)

        self.q_a_norm_w = None
        self.kv_a_norm_w = None
        self._cos = None
        self._sin = None

    # Subclass hooks, identity/None here
    def hook_q_lora(self, q_lora):
        return q_lora

    def hook_kv_latent(self, k_latent):
        return k_latent

    def hook_attn_out(self, attn):
        return attn

    def extra_kv(self):
        return None

    def load_extra(self, device, get):
        pass

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        for cl in self.cache_layers:
            cl.alloc(device)
        # no_defer: deferred placeholders only materialize after load, and these
        # tensors are consumed/transformed before that
        get = lambda k, **kw: self.config.stc.get_tensor(f"{self.key}.{k}", device,
                                                         float2half = True, no_defer = True, **kw)
        if self.q_lora_rank:
            self.q_a_norm_w = get(f"{self.key_q_a_norm}.weight")
        self.kv_a_norm_w = get(f"{self.key_kv_a_norm}.weight")
        self.load_extra(device, get)

    @override
    def unload(self):
        super().unload()
        for cl in self.cache_layers:
            cl.free()
        self.q_a_norm_w = self.kv_a_norm_w = None
        self._cos = self._sin = None

    def optimizer_targets(self):
        q = (self.q_b_proj or self.q_proj).optimizer_targets()
        return [[q, self.o_proj.optimizer_targets()]]

    def get_tensors(self):
        # raw (non-Linear) tensors, re-emitted by the convert compile step
        if self.device is None:
            return {}
        t = {}
        if self.q_a_norm_w is not None:
            t[f"{self.key}.{self.key_q_a_norm}.weight"] = self.q_a_norm_w
        t[f"{self.key}.{self.key_kv_a_norm}.weight"] = self.kv_a_norm_w
        t.update(self.get_tensors_extra())
        return t

    def get_tensors_extra(self):
        return {}

    def _cos_sin(self, seq_len, device, dtype):
        if self._cos is None or self._cos.shape[0] < seq_len or self._cos.device != device:
            half_dim = self.qk_rope_head_dim // 2
            inv_freq = 1.0 / (self.rope_theta ** (
                torch.arange(0, half_dim, device = device, dtype = torch.float32) / half_dim))
            positions = torch.arange(seq_len, device = device, dtype = torch.float32)
            freqs = torch.outer(positions, inv_freq)
            emb = torch.cat((freqs, freqs), dim = -1)
            self._cos = emb.cos().to(dtype)
            self._sin = emb.sin().to(dtype)
        return self._cos[:seq_len], self._sin[:seq_len]

    def _kv_b_weight(self):
        w = self.kv_b_proj.inner.weight  # stored [in, out]
        w = w.transpose(0, 1).view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        return w

    def _mask(self, seq_len, device):
        q_pos = torch.arange(seq_len, device = device).unsqueeze(1)
        k_pos = torch.arange(seq_len, device = device).unsqueeze(0)
        mask = k_pos <= q_pos
        if self.sliding_window is not None:
            mask = mask & (k_pos >= q_pos - self.sliding_window + 1)
        return mask

    def forward(self, x, params, out_dtype = None):
        if params.get("cache") is not None:
            return self._fwd_paged(x, params)
        # No cache: [seq, hidden] or [1, seq, hidden], batch size 1
        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        seq_len = x.shape[0]
        if self.q_lora_rank:
            q_lora = self.q_a_proj.forward(x, params)[..., : self.q_lora_rank]
            q_lora = self.hook_q_lora(q_lora)
            q_lora = _rms_std(q_lora, self.q_a_norm_w, self.rms_norm_eps)
            q = self.q_b_proj.forward(q_lora, params)[..., : self.num_heads * self.qk_head_dim]
        else:
            q = self.q_proj.forward(x, params)[..., : self.num_heads * self.qk_head_dim]
        q = q.view(seq_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim = -1)

        kv = self.kv_a_proj.forward(x, params)[..., : self.kv_lora_rank + self.qk_rope_head_dim]
        k_latent, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim = -1)
        k_latent = self.hook_kv_latent(k_latent)

        if self.rope_interleave_pairs:
            q_pe = _interleave_pairs(q_pe)
            k_pe = _interleave_pairs(k_pe)

        if self.absorbed:
            attn = self._fwd_absorbed(x, q_nope, q_pe, k_latent, k_pe)
        else:
            attn = self._fwd_naive(x, q_nope, q_pe, k_latent, k_pe)
        attn = self.hook_attn_out(attn)
        out = self.o_proj.forward(attn, params)[..., : x.shape[-1]]
        return out.reshape(orig_shape)

    def _fwd_paged(self, x, params):
        # [bsz, q_len, hidden] against the paged cache. K/V are up-projected and
        # cached at qk_head_dim (V zero-padded), giving the standard cache layout
        # and attention path. Rope positions come from cache_seqlens.
        assert self.extra_kv() is None, "extra_kv not supported on the paged path"
        bsz, q_len, hidden = x.shape
        nope, rope_d, v_dim = self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim

        if self.q_lora_rank:
            q_lora = self.q_a_proj.forward(x, params)[..., : self.q_lora_rank]
            q_lora = self.hook_q_lora(q_lora)
            q_lora = _rms_std(q_lora, self.q_a_norm_w, self.rms_norm_eps)
            q = self.q_b_proj.forward(q_lora, params)[..., : self.num_heads * self.qk_head_dim]
        else:
            q = self.q_proj.forward(x, params)[..., : self.num_heads * self.qk_head_dim]
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [nope, rope_d], dim = -1)

        kv = self.kv_a_proj.forward(x, params)[..., : self.kv_lora_rank + rope_d]
        k_latent, k_pe = torch.split(kv, [self.kv_lora_rank, rope_d], dim = -1)
        k_latent = self.hook_kv_latent(k_latent)
        norm = _rms_kvc if self.kv_norm_dtype_mul else _rms_std
        k_lat_n = norm(k_latent, self.kv_a_norm_w, self.rms_norm_eps)
        kv_up = self.kv_b_proj.forward(k_lat_n, {})[..., : self.num_heads * (nope + v_dim)]
        kv_up = kv_up.view(bsz, q_len, self.num_heads, nope + v_dim)
        k_nope, v = torch.split(kv_up, [nope, v_dim], dim = -1)

        if self.rope_interleave_pairs:
            q_pe = _interleave_pairs(q_pe)
            k_pe = _interleave_pairs(k_pe)

        cache_seqlens = get_for_device(params, "cache_seqlens", x.device)
        block_table = get_for_device(params, "block_table", x.device)
        max_pos = int(cache_seqlens.max().item()) + q_len
        cos, sin = self._cos_sin(max_pos, x.device, q_pe.dtype)
        pos = cache_seqlens.long().unsqueeze(1) + torch.arange(q_len, device = x.device).unsqueeze(0)
        cos_q = cos[pos].unsqueeze(2)
        sin_q = sin[pos].unsqueeze(2)
        q_pe = ((q_pe.float() * cos_q.float()) + (_rotate_half(q_pe.float()) * sin_q.float())).to(q.dtype)
        cos_k = cos[pos].to(k_pe.dtype)
        sin_k = sin[pos].to(k_pe.dtype)
        k_pe = (k_pe * cos_k) + (_rotate_half(k_pe) * sin_k)

        q = torch.cat((q_nope, q_pe), dim = -1)
        k = torch.cat((k_nope, k_pe.unsqueeze(2).expand(-1, -1, self.num_heads, -1)), dim = -1)
        v = F.pad(v, (0, self.qk_head_dim - v_dim))

        o = attn_dispatch(
            q = q.contiguous(),
            k = k.contiguous(),
            v = v.contiguous(),
            cache = params.get("cache"),
            cache_idx = self.layer_idx,
            cache_instance = params.get("layer_instance"),
            block_table = block_table,
            cache_seqlens = cache_seqlens,
            causal = True,
            sm_scale = self.scaling,
            window_size = self.sliding_window,
        )
        o = o[..., : v_dim].reshape(bsz, q_len, self.num_heads * v_dim)
        o = self.hook_attn_out(o)
        return self.o_proj.forward(o, params)[..., : hidden]

    def _fwd_naive(self, x, q_nope, q_pe, k_latent, k_pe):
        seq_len = x.shape[0]
        nope, v_dim = self.qk_nope_head_dim, self.v_head_dim
        norm = _rms_kvc if self.kv_norm_dtype_mul else _rms_std
        k_lat_n = norm(k_latent, self.kv_a_norm_w, self.rms_norm_eps)
        kv_up = self.kv_b_proj.forward(k_lat_n, {})[..., : self.num_heads * (nope + v_dim)]
        kv_up = kv_up.view(seq_len, self.num_heads, nope + v_dim)
        k_nope, v = torch.split(kv_up, [nope, v_dim], dim = -1)

        cos, sin = self._cos_sin(seq_len, x.device, q_pe.dtype)
        q_pe = _apply_rotary(q_pe, cos, sin)
        k_pe = _apply_rotary_dtype(k_pe, cos, sin).unsqueeze(1).expand(-1, self.num_heads, -1)

        n_extra = 0
        extra = self.extra_kv()
        if extra is not None:
            x_lat, x_kpe = extra
            x_kv = self.kv_b_proj.forward(x_lat, {})[..., : self.num_heads * (nope + v_dim)]
            x_kv = x_kv.view(-1, self.num_heads, nope + v_dim)
            x_k_nope, x_v = torch.split(x_kv, [nope, v_dim], dim = -1)
            x_kpe = x_kpe.view(-1, 1, self.qk_rope_head_dim).expand(-1, self.num_heads, -1)
            k_nope = torch.cat([x_k_nope.to(k_nope.dtype), k_nope], dim = 0)
            k_pe = torch.cat([x_kpe.to(k_pe.dtype), k_pe], dim = 0)
            v = torch.cat([x_v.to(v.dtype), v], dim = 0)
            n_extra = k_nope.shape[0] - seq_len

        scores = torch.einsum("thd,shd->hts", q_nope, k_nope)
        scores = scores + torch.einsum("thr,shr->hts", q_pe, k_pe)
        scores = scores * self.scaling
        mask = self._mask(seq_len, x.device)
        if n_extra > 0:
            mask = torch.cat([torch.ones(seq_len, n_extra, dtype = torch.bool, device = x.device), mask], dim = -1)
        mask = mask.unsqueeze(0)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.float(), dim = -1).to(dtype = q_nope.dtype)
        probs = torch.where(mask, probs, torch.zeros_like(probs))
        return torch.einsum("hts,shv->thv", probs, v).reshape(seq_len, -1)

    def _fwd_absorbed(self, x, q_nope, q_pe, k_latent, k_pe):
        seq_len = x.shape[0]
        cos, sin = self._cos_sin(seq_len, x.device, q_pe.dtype)
        q_pe = _apply_rotary(q_pe, cos, sin)
        k_pe = _apply_rotary(k_pe, cos, sin)
        w = self._kv_b_weight()
        w_uk_t = w[:, : self.qk_nope_head_dim, :]
        w_uv = w[:, self.qk_nope_head_dim :, :].transpose(1, 2)
        q_lat = torch.einsum("thd,hdr->thr", q_nope.float(), w_uk_t.float()).to(q_nope.dtype)
        k_lat = _rms_std(k_latent, self.kv_a_norm_w, self.rms_norm_eps)

        n_extra = 0
        extra = self.extra_kv()
        if extra is not None:
            x_lat, x_kpe = extra
            k_lat = torch.cat([x_lat.to(q_lat.dtype), k_lat], dim = 0)
            k_pe = torch.cat([x_kpe.to(q_pe.dtype), k_pe], dim = 0)
            n_extra = k_lat.shape[0] - seq_len

        scores = torch.einsum("qhr,kr->qhk", q_lat.float(), k_lat.float())
        scores = scores + torch.einsum("qhr,kr->qhk", q_pe.float(), k_pe.float())
        scores = scores * self.scaling
        mask = self._mask(seq_len, x.device)
        if n_extra > 0:
            mask = torch.cat([torch.ones(seq_len, n_extra, dtype = torch.bool, device = x.device), mask], dim = -1)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        probs = torch.softmax(scores, dim = -1)
        probs = torch.where(mask.unsqueeze(1), probs, torch.zeros_like(probs))
        attn_lat = torch.einsum("qhk,kr->qhr", probs, k_lat.float()).to(q_lat.dtype)
        attn = torch.einsum("thr,hrv->thv", attn_lat.float(), w_uv.float())
        return attn.to(x.dtype).reshape(seq_len, -1)
