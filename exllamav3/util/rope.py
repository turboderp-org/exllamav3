from dataclasses import dataclass

import torch
import math
from enum import IntEnum
from ..ext import exllamav3_ext as ext

# Reference:
# https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/modeling_rope_utils.py

class RopeStyle(IntEnum):
    NONE = 0
    GPTJ = 1
    NEOX = 2

@dataclass
class RopeSettings:
    head_dim: int = 128
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    partial_rotary_factor: float = 1.0
    max_position_embeddings: int | None = None
    original_max_position_embeddings: int | None = None
    rope_style: RopeStyle = RopeStyle.NEOX
    override_max_position_embeddings: int | None = None


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim = -1)


def _rotate_half_gptj(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope_embed_qk(q, k, sin, cos):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    return q, k


def _apply_rope_embed_q(q, sin, cos):
    q = q.transpose(1, 2)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos) + (_rotate_half(q) * sin)
    q = q.transpose(1, 2)
    return q


class RoPE:

    # TODO: Alpha and linear scaling overrides (?)

    def __init__(
        self,
        device: torch.device | str,
        rope_settings: RopeSettings,
    ):
        self.device = device
        self.rope_settings = rope_settings

        self.cached_sin = None
        self.cached_cos = None
        self.cached_sincos_max = 0

        t = None
        rs = self.rope_settings
        if rs.rope_scaling is not None:
            t = rs.rope_scaling.get("rope_type", rs.rope_scaling.get("type"))
        match t:
            case None:
                self.inv_freq, self.attn_factor = self._rope_params_default()
            case "llama3":
                self.inv_freq, self.attn_factor = self._rope_params_llama3()
            case "linear":
                self.inv_freq, self.attn_factor = self._rope_params_linear()
            case "yarn":
                self.inv_freq, self.attn_factor = self._rope_params_yarn()
            case "longrope" | "su":
                self.inv_freq, self.attn_factor = self._rope_params_longrope()
            case _:
                raise ValueError(f"Unknown rope_type: {t}")


    def _rope_params_default(self):
        rs = self.rope_settings
        base = rs.rope_theta
        dim = int(rs.head_dim * rs.partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype = torch.int64, device = self.device).float() / dim))
        return inv_freq, 1.0


    def _rope_params_llama3(self):
        rs = self.rope_settings
        inv_freq, attention_factor = self._rope_params_default()
        factor = rs.rope_scaling.get("factor", 8.0)
        low_freq_factor = rs.rope_scaling.get("low_freq_factor", 1.0)
        high_freq_factor = rs.rope_scaling.get("high_freq_factor", 4.0)
        old_context_len = rs.rope_scaling.get("original_max_position_embeddings", 8192)
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        wavelen = 2 * math.pi / inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = (wavelen >= high_freq_wavelen) * (wavelen <= low_freq_wavelen)
        inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        return inv_freq, 1.0


    def _rope_params_linear(self):
        rs = self.rope_settings
        inv_freq, attention_factor = self._rope_params_default()
        factor = rs.rope_scaling.get("factor", 1.0)
        inv_freq /= factor
        return inv_freq, attention_factor


    def _rope_params_yarn(self):
        rs = self.rope_settings
        assert rs.max_position_embeddings is not None, \
            "YaRN scaling requires explicit max_position_embeddings"
        base = rs.rope_theta
        dim = int(rs.head_dim * rs.partial_rotary_factor)
        factor = rs.rope_scaling.get("factor")
        attn_factor = rs.rope_scaling.get("attention_factor", 0.1 * math.log(factor) + 1.0)
        beta_fast = rs.rope_scaling.get("beta_fast", 32)
        beta_slow = rs.rope_scaling.get("beta_slow", 1)
        def find_correction_dim(num_rotations):
            return (dim * math.log(rs.max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))
        def find_correction_range(low_rot, high_rot):
            _low = math.floor(find_correction_dim(low_rot))
            _high = math.ceil(find_correction_dim(high_rot))
            return max(_low, 0), min(_high, dim - 1)
        def linear_ramp_factor(_min, _max, _dim):
            if _min == _max:
                _max += 0.001
            linear_func = (torch.arange(_dim, dtype = torch.float32, device = self.device) - _min) / (_max - _min)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func
        pos_freqs = base ** (torch.arange(0, dim, 2, device = self.device).float() / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)
        low, high = find_correction_range(beta_fast, beta_slow)
        inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).float()
        inv_freq = inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
        inv_freq += inv_freq_extrapolation * inv_freq_extrapolation_factor
        return inv_freq, attn_factor


    def _rope_params_longrope(self):
        rs = self.rope_settings
        base = rs.rope_theta
        dim = int(rs.head_dim * rs.partial_rotary_factor)
        a = rs.max_position_embeddings
        a_override = rs.override_max_position_embeddings or a
        b = rs.rope_scaling.get("original_max_position_embeddings", rs.original_max_position_embeddings)
        if a_override > b:
            factors = rs.rope_scaling.get("long_factor")
            ext_factors = torch.tensor(factors, dtype = torch.float32, device = self.device)
        else:
            factors = rs.rope_scaling.get("short_factor")
            ext_factors = torch.tensor(factors, dtype = torch.float32, device = self.device)
        if a > b:
            scaling = math.sqrt(1 + math.log(a / b) / math.log(b))
        else:
            scaling = 1.0
        inv_freq = 1.0 / (ext_factors * base ** (torch.arange(0, dim, 2, device = self.device).float() / dim))
        return inv_freq, scaling


    def compute_sincos(self, position_ids: torch.Tensor):
        rs = self.rope_settings
        freqs = torch.einsum("i,j->ij", position_ids, self.inv_freq)
        sin = freqs.sin()
        cos = freqs.cos()
        if self.attn_factor != 1.0:
            sin *= self.attn_factor
            cos *= self.attn_factor
        match rs.rope_style:
            case RopeStyle.NEOX:
                sin = torch.cat((sin, sin), dim = -1)
                cos = torch.cat((cos, cos), dim = -1)
            case RopeStyle.GPTJ:
                sin = torch.repeat_interleave(sin, 2, dim = -1)
                cos = torch.repeat_interleave(cos, 2, dim = -1)
        return sin.half(), cos.half()


    def expand_cache(self, pos_id_end: int):
        interval = 2048
        if pos_id_end >= self.cached_sincos_max:
            pmax = self.cached_sincos_max
            nmax = (pos_id_end // interval + 1) * interval
            nsin, ncos = self.compute_sincos(torch.arange(pmax, nmax, device = self.device))
            self.cached_sin = torch.cat((self.cached_sin, nsin), dim = 0) if pmax > 0 else nsin
            self.cached_cos = torch.cat((self.cached_cos, ncos), dim = 0) if pmax > 0 else ncos
            self.cached_sincos_max = nmax


    def apply_torch(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        pos: int = 0,
        positions: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        in_place = False,
    ):
        # TODO: GPTJ, partial rotary factor
        if in_place:
            q_ = q
            k_ = k

        if len(q.shape) == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0) if k is not None else k
            squeeze = True
        else:
            squeeze = False
        bsz, qlen, numheads_q, dim = q.shape

        if positions is not None:
            position_ids = torch.arange(qlen, device = self.device).unsqueeze(0).repeat(bsz, 1)
            position_ids += positions.unsqueeze(1)

        if position_ids is not None:
            if len(position_ids.shape) == 1:
                position_ids = position_ids.unsqueeze(0)
            else:
                assert position_ids.shape[0] == bsz
            self.expand_cache(position_ids.max().item())
            sin = self.cached_sin[position_ids]
            cos = self.cached_cos[position_ids]

        else:
            self.expand_cache(pos + qlen)
            sin = self.cached_sin[pos : pos + qlen].unsqueeze(0)
            cos = self.cached_cos[pos : pos + qlen].unsqueeze(0)

        if k is not None:
            q, k = _apply_rope_embed_qk(q, k, sin, cos)
        else:
            q = _apply_rope_embed_q(q, sin, cos)

        if squeeze:
            q = q.squeeze(0)
            k = k.squeeze(0) if k is not None else k

        if in_place:
            q_.copy_(q)
            k_.copy_(k)
            return q_, k_
        else:
            return q, k


    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        position: int = 0,
        positions: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        in_place = False,
    ):
        q = q.contiguous()
        if k is not None: k = k.contiguous()
        if positions is not None: positions = positions.contiguous()
        if position_ids is not None: position_ids = position_ids.contiguous()

        if len(q.shape) == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        if not in_place:
            out_q = torch.empty_like(q)
            out_k = torch.empty_like(k) if k is not None else None
        else:
            out_q = q
            out_k = k

        ext.rope(
            q,
            out_q,
            k,
            out_k,
            self.inv_freq,
            position,
            positions,
            position_ids,
            self.rope_settings.rope_style,
            self.attn_factor
        )
            
        if squeeze:
            out_q = out_q.squeeze(0)
            out_k = out_k.squeeze(0) if out_k is not None else None

        return out_q, out_k
