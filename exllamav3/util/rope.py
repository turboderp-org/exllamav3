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
    rotary_dim: int | None = None
    partial_rotary_factor: float = 1.0
    max_position_embeddings: int | None = None
    original_max_position_embeddings: int | None = None
    rope_style: RopeStyle = RopeStyle.NEOX
    override_max_position_embeddings: int | None = None
    llama_4_scaling_beta: float = 0.0
    override_type: str | None = None
    rotate_dims: int = 1
    # DeepSeek-style YaRN: attn_factor = get_mscale(f, mscale) / get_mscale(f, mscale_all_dim),
    # paired with the architecture folding mscale_all_dim into sm_scale. Off by default because
    # non-DeepSeek configs (Mistral) carry the same keys as inert defaults
    yarn_mscale_ratio: bool = False

    def print(self):
        print(f" -- RoPE settings")
        print(f"    head_dim: {self.head_dim}")
        print(f"    rope_scaling: {self.rope_scaling}")
        print(f"    rope_theta: {self.rope_theta}")
        print(f"    rotary_dim: {self.rotary_dim}")
        print(f"    partial_rotary_factor: {self.partial_rotary_factor}")
        print(f"    max_position_embeddings: {self.max_position_embeddings}")
        print(f"    original_max_position_embeddings: {self.original_max_position_embeddings}")
        print(f"    rope_style: {self.rope_style.name}")
        print(f"    llama_4_scaling_beta: {self.llama_4_scaling_beta}")


def yarn_inv_freq(
    dim: int,
    base: float,
    device,
    rope_scaling: dict | None = None,
    factor: float | None = None,
    original_max_position_embeddings: int | None = None,
) -> torch.Tensor:
    """
    Inverse frequency table for one rope family: plain 1 / base^(2i/dim) when rope_scaling
    is absent or not yarn, else the HF _compute_yarn_parameters interpolation ramp.
    Frequency table ONLY -- the yarn attention factor is the caller's concern:
    RoPE._rope_params_yarn resolves and applies it (legacy HF semantics, including the
    max_position-derived factor override it passes in explicitly), while DeepSeek-V4 pins
    it to 1.0 by never applying one and calls this with the raw config dict per rope
    family (main theta unscaled, compress theta yarn-ramped).
    factor / original_max_position_embeddings override the dict values when given.
    """
    pos_freqs = base ** (torch.arange(0, dim, 2, device = device).float() / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    is_yarn = rope_scaling is not None and \
        rope_scaling.get("type", rope_scaling.get("rope_type", "default")) == "yarn"
    if not is_yarn and factor is None:
        return inv_freq_extrapolation
    sc = rope_scaling or {}
    if factor is None:
        factor = float(sc["factor"])
    if original_max_position_embeddings is None:
        original_max_position_embeddings = int(sc["original_max_position_embeddings"])
    beta_fast = float(sc.get("beta_fast", 32))
    beta_slow = float(sc.get("beta_slow", 1))
    truncate = sc.get("truncate", True)

    def find_correction_dim(num_rotations):
        return (dim * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))) \
            / (2 * math.log(base))

    low = find_correction_dim(beta_fast)
    high = find_correction_dim(beta_slow)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim // 2, dtype = torch.float32, device = device) - low) / (high - low)
    inv_freq_extrapolation_factor = 1 - torch.clamp(linear_func, 0, 1).float()
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    inv_freq = inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
    inv_freq += inv_freq_extrapolation * inv_freq_extrapolation_factor
    return inv_freq


def _rotate_half_neox(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim = -1)

def _apply_rope_embed_q_neox(q, sin, cos):
    q = q.transpose(1, 2)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos) + (_rotate_half_neox(q) * sin)
    q = q.transpose(1, 2)
    return q

def _apply_rope_embed_qk_neox(q, k, sin, cos):
    return (
        _apply_rope_embed_q_neox(q, sin, cos),
        _apply_rope_embed_q_neox(k, sin, cos)
    )


def _rotate_half_gptj(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def _apply_rope_embed_q_gptj(q, sin, cos):
    q = q.transpose(1, 2)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos) + (_rotate_half_gptj(q) * sin)
    q = q.transpose(1, 2)
    return q

def _apply_rope_embed_qk_gptj(q, k, sin, cos):
    return (
        _apply_rope_embed_q_gptj(q, sin, cos),
        _apply_rope_embed_q_gptj(k, sin, cos)
    )


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

        self.mrope_interleaved = None
        self.mrope_section = None

        self.llama_4_scaling_beta = 0.0
        self.llama_4_scaling_original = 1  # Unused when beta=0

        t = rope_settings.override_type
        rs = self.rope_settings
        if not t:
            if rs.rope_scaling is not None:
                t = rs.rope_scaling.get("rope_type", rs.rope_scaling.get("type"))
        # MRoPE sectioning is orthogonal to the scaling type: a multimodal model keeps its
        # mrope_section next to whatever rope_type it uses (Qwen3.5 with YaRN context extension
        # as recommended by its model card, for one), so read it before dispatching on the type
        if rs.rope_scaling:
            self.mrope_interleaved = rs.rope_scaling.get("mrope_interleaved")  # Ignored in HF impl., always True
            self.mrope_section = rs.rope_scaling.get("mrope_section")
        match t:
            case None:
                self._rope_params_default()
            case "default" | "mrope":
                self._rope_params_default()
            case "proportional":
                self._rope_params_proportional()
            case "llama3":
                self._rope_params_llama3()
            case "linear":
                self._rope_params_linear()
            case "yarn":
                self._rope_params_yarn()
            case "longrope" | "su":
                self._rope_params_longrope()
            case _:
                raise ValueError(f"Unknown rope_type: {t}")


    def _rope_params_default(self):
        rs = self.rope_settings
        base = rs.rope_theta
        dim = rs.rotary_dim or int(rs.head_dim * rs.partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype = torch.int64, device = self.device).float() / dim))
        self.inv_freq, self.attn_factor = inv_freq, 1.0


    def _rope_params_llama3(self):
        rs = self.rope_settings
        self._rope_params_default()
        factor = rs.rope_scaling.get("factor", 8.0)
        low_freq_factor = rs.rope_scaling.get("low_freq_factor", 1.0)
        high_freq_factor = rs.rope_scaling.get("high_freq_factor", 4.0)
        old_context_len = rs.rope_scaling.get("original_max_position_embeddings", 8192)
        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        wavelen = 2 * math.pi / self.inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, self.inv_freq / factor, self.inv_freq)
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = (wavelen >= high_freq_wavelen) * (wavelen <= low_freq_wavelen)
        inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        self.inv_freq, self.attn_factor = inv_freq, 1.0


    def _rope_params_linear(self):
        rs = self.rope_settings
        self._rope_params_default()
        factor = rs.rope_scaling.get("factor", 1.0)
        self.inv_freq /= factor


    def _rope_params_proportional(self):
        rs = self.rope_settings
        head_dim = rs.head_dim
        base = rs.rope_theta
        factor = rs.rope_scaling.get("factor", 1.0) if rs.rope_scaling else 1.0
        rope_proportion = rs.partial_rotary_factor

        rope_angles = int(rope_proportion * head_dim // 2)
        inv_freq_rotated = 1.0 / (
            base ** (
                torch.arange(0, 2 * rope_angles, 2, dtype = torch.int64, device = self.device).float() / head_dim
            )
        )
        nope_angles = head_dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat(
                (
                    inv_freq_rotated,
                    torch.zeros(nope_angles, dtype = torch.float32, device = self.device),
                ),
                dim = 0,
            )
        else:
            inv_freq = inv_freq_rotated
        self.inv_freq, self.attn_factor = inv_freq / factor, 1.0


    def _rope_params_yarn(self):
        rs = self.rope_settings
        max_position_embeddings = rs.override_max_position_embeddings or rs.max_position_embeddings
        assert max_position_embeddings is not None, \
            "YaRN scaling requires explicit max_position_embeddings"
        base = rs.rope_theta
        dim = rs.rotary_dim or int(rs.head_dim * rs.partial_rotary_factor)
        original_max_position_embeddings = rs.rope_scaling.get("original_max_position_embeddings")
        has_original_max_position_embeddings = original_max_position_embeddings is not None
        if not has_original_max_position_embeddings:
            original_max_position_embeddings = max_position_embeddings

        try:
            original_max_position_embeddings_int = int(original_max_position_embeddings)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                "YaRN original_max_position_embeddings must be an integer, "
                f"got {original_max_position_embeddings!r}"
            ) from None
        if original_max_position_embeddings != original_max_position_embeddings_int:
            raise ValueError(
                "YaRN original_max_position_embeddings must be an integer, "
                f"got {original_max_position_embeddings!r}"
            )
        original_max_position_embeddings = original_max_position_embeddings_int

        if has_original_max_position_embeddings:
            factor = max_position_embeddings / original_max_position_embeddings
        else:
            factor = rs.rope_scaling.get("factor")

        attn_factor = rs.rope_scaling.get("attention_factor")
        if attn_factor is None:
            def get_mscale(scale, mscale = 1.0):
                if scale <= 1:
                    return 1.0
                return 0.1 * mscale * math.log(scale) + 1.0
            mscale = rs.rope_scaling.get("mscale")
            mscale_all_dim = rs.rope_scaling.get("mscale_all_dim")
            if rs.yarn_mscale_ratio and mscale and mscale_all_dim:
                # DeepSeek-family semantics, opted into by the architecture: sin/cos get the
                # mscale/mscale_all_dim ratio while the arch folds mscale_all_dim into sm_scale.
                # Only meaningful together with that sm_scale adjustment, so it cannot be inferred
                # from the presence of the config keys alone: Mistral yarn configs carry
                # mscale = mscale_all_dim = 1.0 as inert defaults, and taking the ratio there
                # (as HF transformers does) silently drops the YaRN attention factor
                attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
            elif rs.rope_scaling.get("llama_4_scaling_beta"):
                # Position-dependent attention scaling supersedes the static YaRN factor
                # (Ministral-3: ppl 6.81 with 1.0 vs 7.38 with the paper formula)
                attn_factor = 1.0
            else:
                attn_factor = get_mscale(factor)
        self.llama_4_scaling_beta = rs.rope_scaling.get("llama_4_scaling_beta", 0.0)
        self.llama_4_scaling_original = original_max_position_embeddings
        self.inv_freq = yarn_inv_freq(
            dim, base, self.device,
            rope_scaling = rs.rope_scaling,
            factor = factor,
            original_max_position_embeddings = original_max_position_embeddings,
        )
        self.attn_factor = attn_factor


    def _rope_params_longrope(self):
        rs = self.rope_settings
        base = rs.rope_theta
        dim = rs.rotary_dim or int(rs.head_dim * rs.partial_rotary_factor)
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
        self.inv_freq, self.attn_factor = inv_freq, scaling


    def compute_sincos(self, position_ids: torch.Tensor):
        rs = self.rope_settings
        freqs = torch.einsum("i,j->ij", position_ids.float(), self.inv_freq)
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
        return sin, cos


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
        inv_freq: torch.Tensor | None = None,
    ):
        # Host reference for every CPU vision-tower call shape (the native rope_cpu
        # kernel's oracle): float32 math, single round to the caller's dtype.
        # Frequency sources: a table (inv_freq.dim() > 1, rows are precomputed
        # angles: Step3.7 GPT-J grid, Mistral/Qwen NEOX tables) or a vector (1-D
        # override or self.inv_freq, angles are pos * freq). Position resolution
        # mirrors rope.cu get_pos: position_ids (2-D shared, or 3-D per-rdim for
        # rotate_dims == 2) > positions[batch] + token > pos + token.
        in_dtype = q.dtype
        style = self.rope_settings.rope_style
        rotate_dims = self.rope_settings.rotate_dims
        attn_factor = self.attn_factor

        if positions is not None and position_ids is not None:
            raise ValueError("RoPE: positions and position_ids are mutually exclusive")

        if in_place:
            q_ = q
            k_ = k

        qf = q.float()
        kf = k.float() if k is not None else None

        if len(qf.shape) == 3:
            qf = qf.unsqueeze(0)
            kf = kf.unsqueeze(0) if kf is not None else kf
            squeeze = True
        else:
            squeeze = False
        bsz, qlen, _, dim = qf.shape

        if position_ids is not None and position_ids.dim() == 3:
            # Gemma4 partial rotary: per-section positions [bsz, qlen, rotate_dims]
            assert position_ids.size(0) == bsz and position_ids.size(1) == qlen and \
                position_ids.size(2) == rotate_dims, \
                f"RoPE: 3-D position_ids must be [bsz, qlen, rotate_dims], got {tuple(position_ids.shape)}"
            pos_idx = position_ids.long()
        elif position_ids is not None:
            pid = position_ids.long()
            if pid.dim() == 1:
                pid = pid.unsqueeze(0)
            assert pid.dim() == 2 and pid.size(0) == bsz and pid.size(1) == qlen, \
                f"RoPE: 2-D position_ids must be [bsz, qlen], got {tuple(position_ids.shape)}"
            pos_idx = pid.unsqueeze(-1).expand(bsz, qlen, rotate_dims)
        elif positions is not None:
            base = positions.long().view(bsz, 1, 1) + \
                torch.arange(qlen, dtype = torch.long, device = q.device).view(1, qlen, 1)
            pos_idx = base.expand(bsz, qlen, rotate_dims)
        else:
            base = int(pos) + torch.arange(qlen, dtype = torch.long, device = q.device).view(1, qlen, 1)
            pos_idx = base.expand(bsz, qlen, rotate_dims)

        ivf = self.inv_freq if inv_freq is None else inv_freq
        ivf_f = ivf.float()
        if ivf_f.dim() > 1:
            # Table of precomputed angles, row [.., P/2]. 3-D tables carry the batch
            # dim first (Step3.7 patches: [batch, seq, P/2]); 2-D tables are indexed
            # by position only (Mistral gathered table).
            if ivf_f.dim() == 3:
                b_idx = torch.arange(bsz, device = ivf_f.device).view(bsz, 1, 1) \
                    .expand(bsz, qlen, rotate_dims)
                rows = ivf_f[b_idx, pos_idx.to(ivf_f.device)]
            else:
                rows = ivf_f[pos_idx.to(ivf_f.device)]
            P = ivf_f.size(-1) * 2
            sin_t = rows.sin() * attn_factor  # [bsz, qlen, R, P/2]
            cos_t = rows.cos() * attn_factor
        else:
            P = ivf_f.numel() * 2
            ang = pos_idx.float().unsqueeze(-1) * ivf_f.view(1, 1, 1, -1)
            sin_t = ang.sin() * attn_factor  # [bsz, qlen, R, P/2]
            cos_t = ang.cos() * attn_factor

        if rotate_dims == 1:
            assert P <= dim, f"RoPE: rotary width {P} exceeds head_dim {dim}"
        else:
            assert dim == P * rotate_dims, \
                f"RoPE: head_dim {dim} != partial {P} * rotate_dims {rotate_dims}"

        def _rot_sections(x):
            parts = []
            for rdim in range(rotate_dims):
                s = sin_t[:, :, rdim, :].unsqueeze(2)
                c = cos_t[:, :, rdim, :].unsqueeze(2)
                sec = x[..., rdim * P:(rdim + 1) * P]
                if style == RopeStyle.NEOX:
                    x1, x2 = sec.split(P // 2, dim = -1)
                    parts.append(torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim = -1))
                else:
                    xe = sec[..., 0::2]
                    xo = sec[..., 1::2]
                    parts.append(torch.stack((xe * c - xo * s, xo * c + xe * s), dim = -1).flatten(-2))
            out = parts[0] if rotate_dims == 1 else torch.cat(parts, dim = -1)
            if out.shape[-1] < dim:
                out = torch.cat((out, x[..., out.shape[-1]:]), dim = -1)
            return out

        qf = _rot_sections(qf)
        if kf is not None:
            kf = _rot_sections(kf)

        q_out = qf.to(in_dtype)
        k_out = kf.to(in_dtype) if kf is not None else None

        if squeeze:
            q_out = q_out.squeeze(0)
            k_out = k_out.squeeze(0) if k_out is not None else None

        if in_place:
            q_.copy_(q_out)
            if k_ is not None:
                k_.copy_(k_out)
            return q_, k_
        else:
            return q_out, k_out


    def apply_cpu(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        position: int = 0,
        positions: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        in_place = False,
        inv_freq: torch.Tensor | None = None,
        q_norm: torch.Tensor | None = None,
        k_norm: torch.Tensor | None = None,
    ):
        # CPU mirror of apply's CUDA path: normalize to 4-D, try the fused native
        # kernel, fall back to apply_torch. Never raises for shape reasons; the
        # reference raises on truly inconsistent geometry, like rope_gr's checks.
        # G4 scope, kept loud: fused q/k norms and Llama-4 query scaling have no
        # CPU path yet. Silently dropping them would corrupt Gemma4/Ministral
        # towers (attn.py fuses norms into rope when q_norm_tensor exists).
        if q_norm is not None or k_norm is not None:
            raise ValueError("CPU RoPE with fused q/k norms is not supported yet (G4)")
        if self.llama_4_scaling_beta != 0.0:
            raise ValueError("CPU RoPE with Llama-4 scaling is not supported")
        q_orig, k_orig = q, k
        q4 = q.contiguous()
        k4 = k.contiguous() if k is not None else None
        pos4 = positions.contiguous() if positions is not None else None
        pid4 = position_ids.contiguous() if position_ids is not None else None
        ivf = self.inv_freq if inv_freq is None else inv_freq
        if ivf.dim() == 0:
            raise ValueError("RoPE: inv_freq must be a vector or angle table")
        if ivf.device != q.device:
            # Tiny table/vector (at most thousands of floats); a tower running on
            # CPU may still hold it on CUDA (Gemma4 partial rotary passes no
            # inv_freq override). Both native and torch paths need it on q's device.
            ivf = ivf.to(q.device)
        ivf = ivf.contiguous()

        if len(q4.shape) == 3:
            q4 = q4.unsqueeze(0)
            k4 = k4.unsqueeze(0) if k4 is not None else k4
            squeeze = True
        else:
            squeeze = False

        if not in_place:
            out_q = torch.empty_like(q4)
            out_k = torch.empty_like(k4) if k4 is not None else None
        else:
            out_q = q4
            out_k = k4

        if self._rope_cpu_ext(q4, out_q, k4, out_k, ivf, position, pos4, pid4):
            if squeeze:
                out_q = out_q.squeeze(0)
                out_k = out_k.squeeze(0) if out_k is not None else None
            return out_q, out_k

        return self.apply_torch(
            q_orig, k_orig,
            pos = position,
            positions = positions,
            position_ids = position_ids,
            in_place = in_place,
            inv_freq = ivf,
        )


    def _rope_cpu_ext(
        self,
        q: torch.Tensor,
        out_q: torch.Tensor,
        k: torch.Tensor | None,
        out_k: torch.Tensor | None,
        inv_freq: torch.Tensor,
        position,
        positions: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> bool:
        """Native fused rope_cpu; False when unsupported (caller falls back to torch)."""
        fn = getattr(ext, "rope_cpu", None)
        if fn is None:
            return False
        try:
            position = int(position)
        except (TypeError, ValueError):
            return False
        style = self.rope_settings.rope_style
        if style not in (RopeStyle.GPTJ, RopeStyle.NEOX):
            return False
        for t in (q, out_q):
            if not (t.is_cpu and t.is_contiguous() and t.is_floating_point()):
                return False
            if t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                return False
        if q.dim() != 4 or q.stride(3) != 1 or out_q.stride() != q.stride():
            return False
        if k is not None:
            for t in (k, out_k):
                if not (t.is_cpu and t.is_contiguous() and t.is_floating_point()):
                    return False
                if t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    return False
            if k.dim() != 4 or k.stride(3) != 1 or out_k.stride() != k.stride():
                return False
            if k.size(0) != q.size(0) or k.size(1) != q.size(1) or k.size(3) != q.size(3):
                return False
        if not (inv_freq.is_cpu and inv_freq.is_contiguous() and
                inv_freq.dtype == torch.float32):
            return False
        if (positions is not None) and (position_ids is not None):
            return False
        for p in (positions, position_ids):
            if p is not None and (not p.is_cpu or not p.is_contiguous() or
                    p.dtype not in (torch.int32, torch.int64)):
                return False
        rotate_dims = self.rope_settings.rotate_dims
        if not (1 <= rotate_dims <= 4):
            return False
        P = inv_freq.size(-1) * 2 if inv_freq.dim() > 1 else inv_freq.numel() * 2
        hd = q.size(3)
        if rotate_dims == 1:
            if P > hd:
                return False
        elif hd != P * rotate_dims:
            return False
        if position_ids is not None:
            rd = position_ids.dim()
            if rd not in (2, 3):
                return False
            if position_ids.size(0) != q.size(0) or position_ids.size(1) != q.size(1):
                return False
            if rd == 3 and position_ids.size(2) != rotate_dims:
                return False
        if positions is not None:
            if positions.dim() != 1 or positions.size(0) != q.size(0):
                return False
        try:
            fn(q, out_q, k, out_k, inv_freq, position, positions, position_ids,
               int(style), float(self.attn_factor), int(rotate_dims))
        except Exception:
            return False
        return True


    def get_mrope_freqs(
        self,
        input_ids: torch.Tensor,
        embeddings: list,  #[MMEmbedding],
        max_length: int,
    ):
        # Create 3D position IDs
        ids = input_ids.squeeze(0).contiguous()
        mrope_pos_ids = torch.zeros((3, max_length), dtype = torch.long).contiguous()
        spans = []
        grids = []
        merge_size = None
        for embedding in embeddings:
            spans.append((embedding.first_index, embedding.last_index))
            grids.append(embedding.grid_thw)
            if merge_size:
                assert merge_size == embedding.mrope_merge_size, "mrope_merge_size varies across MMEmbeddings"
            else:
                merge_size = embedding.mrope_merge_size
        if merge_size is None: merge_size = 1
        next_pos_idx = ext.gen_mrope_pos_ids(mrope_pos_ids, ids, merge_size, spans, grids)

        # Interleave frequencies
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, 1, -1, 1)
        position_ids_expanded = mrope_pos_ids[:, None, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start = 1):  # H, W
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]

        return freqs_t.contiguous(), next_pos_idx


    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        position: int = 0,
        positions: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        in_place = False,
        q_norm: torch.Tensor | None = None,
        k_norm: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
        norm_constant_bias: float = 0.0,
        inv_freq: torch.Tensor | None = None,
    ):
        q = q.contiguous()
        if k is not None: k = k.contiguous()
        if positions is not None: positions = positions.contiguous()
        if position_ids is not None: position_ids = position_ids.contiguous()

        # CPU tensors (vision towers): the extension kernel is CUDA-only, so run the
        # fused host kernel when available, else the torch reference above. All three
        # vision call shapes are supported: external-angle tables (Step3.7 GPT-J,
        # Mistral/Qwen NEOX gathers) and partial rotary (Gemma4 rotate_dims=2).
        if q.is_cpu:
            return self.apply_cpu(
                q, k,
                position = position,
                positions = positions,
                position_ids = position_ids,
                in_place = in_place,
                inv_freq = inv_freq,
                q_norm = q_norm,
                k_norm = k_norm,
            )
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
            q, out_q,
            k, out_k,
            self.inv_freq if inv_freq is None else inv_freq,
            position,
            positions,
            position_ids,
            self.rope_settings.rope_style,
            self.attn_factor,
            q_norm,
            k_norm,
            norm_eps,
            norm_constant_bias,
            self.llama_4_scaling_beta,
            self.llama_4_scaling_original,
            self.rope_settings.rotate_dims,
            0,
        )
            
        if squeeze:
            out_q = out_q.squeeze(0)
            out_k = out_k.squeeze(0) if out_k is not None else None

        return out_q, out_k
