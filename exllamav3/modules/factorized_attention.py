from __future__ import annotations

import torch
from typing_extensions import override

from ..model.config import Config
from ..util.tensor import get_for_device
from .attention_fn import attn_dispatch
from .attention_fn.factorized_triton import factorized_attention_decode
from .linear import Linear
from .recurrent import RecurrentMixer
from .rmsnorm import zero_centered_rms_norm


def shift_kv(current, last, weight, positions):
    """Interpolate with the previous raw head, starting a new stream at position zero."""
    previous = torch.cat((last.unsqueeze(1), current[:, :-1]), dim = 1)
    weight = weight.sigmoid().masked_fill(positions[..., None] == 0, 0).unsqueeze(-1)
    return weight * previous + (1 - weight) * current


class FactorizedDifferentialAttention(RecurrentMixer):
    """Factorized K/V with a recurrent token shift and one noise head per KV group."""
    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rank: int,
        sliding_window: int,
        softcap: float,
        norm_eps: float,
        qmap: str | None = None,
        projection_keys: dict[str, str] | None = None,
        tensor_keys: dict[str, str] | None = None,
    ):
        if num_kv_heads <= 0 or num_q_heads <= num_kv_heads or num_q_heads % num_kv_heads:
            raise ValueError(f"{key}: expected an equal number of query heads per KV group")
        if min(hidden_size, head_dim, rank, sliding_window, norm_eps) <= 0 or softcap < 0:
            raise ValueError(f"{key}: invalid factorized attention dimensions or scaling")
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rank = rank
        self.output_size = (num_q_heads - num_kv_heads) * head_dim
        self.sliding_window = sliding_window
        self.softcap = softcap
        self.norm_eps = norm_eps
        state_shapes = ((self.num_kv_heads, self.head_dim),) * 2
        super().__init__(config, key, layer_idx, state_shapes, {
            "softmax_scaler": (self.num_q_heads,),
            "q_norm.norm.weight": (self.head_dim,), "k_norm.norm.weight": (self.head_dim,),
        }, tensor_keys)
        self.cache_layers = []
        self.dispatch_cache = {}
        self.caps["kv_cache"] = True

        widths = {
            "c_q": self.num_q_heads * self.head_dim,
            "W_A_k": self.num_kv_heads * self.rank, "W_A_v": self.num_kv_heads * self.rank,
            "W_B_k": self.rank * self.head_dim, "W_B_v": self.rank * self.head_dim,
            "shift_proj_k": self.num_kv_heads, "shift_proj_v": self.num_kv_heads,
            "lambda_proj": self.num_kv_heads,
        }
        projection_keys = projection_keys or {}
        self.projections = {
            name: Linear(
                config, f"{key}.{projection_keys.get(name, name)}", in_features = hidden_size, out_features = width,
                qmap = qmap, trim_padded_out = True,
            )
            for name, width in widths.items()
        }
        for linear in self.projections.values():
            self.register_submodule(linear)

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        for layer in self.cache_layers:
            layer.alloc(device)

    @override
    def unload(self):
        super().unload()
        for layer in self.cache_layers:
            layer.free()
        self.dispatch_cache.clear()

    @override
    def optimizer_targets(self):
        return [linear.optimizer_targets() for linear in self.projections.values()]

    @override
    def forward(self, x, params, out_dtype = None):
        projection_input = x.half()

        def project(name):
            return self.projections[name].forward(projection_input, params, out_dtype = torch.float)

        def tpa(suffix):
            left = project(f"W_A_{suffix}").unflatten(-1, (self.num_kv_heads, self.rank))
            right = project(f"W_B_{suffix}").unflatten(-1, (self.rank, self.head_dim))
            return (left @ right) / self.rank

        cache = params.get("cache")
        lengths = params.get("cache_seqlens") if cache is not None else None
        positions = params.get("position_ids")
        if positions is None:
            starts = params.get("positions")
            if starts is None:
                starts = lengths if lengths is not None else params.get("position", 0)
            if isinstance(starts, torch.Tensor):
                starts = starts.to(x.device, non_blocking = True).reshape(-1, 1)
            positions = starts + torch.arange(x.shape[1], device = x.device).unsqueeze(0)
        else:
            positions = get_for_device(params, "position_ids", x.device)

        fused = (
            x.is_cuda and x.shape[1] == 1 and params.get("recurrent_slots") is not None
            and self.head_dim == 128 and self.rank == 4
        )
        if fused:
            layer, slots = self.state_slots(x, params)
            q, k, v = factorized_attention_decode(
                project("c_q"), project("W_A_k"), project("W_B_k"), project("W_A_v"), project("W_B_v"),
                project("shift_proj_k"), project("shift_proj_v"), self.tensors, slots,
                positions.expand(x.shape[0], 1).contiguous(), layer.tensors,
                self.num_q_heads, self.num_kv_heads, self.head_dim, self.rank,
                self.sliding_window, self.norm_eps,
            )
        else:
            layer, slots, (prev_k, prev_v) = self.load_state(x, params)
            raw_k, raw_v = tpa("k"), tpa("v")
            k = shift_kv(raw_k, prev_k, project("shift_proj_k"), positions)
            v = shift_kv(raw_v, prev_v, project("shift_proj_v"), positions)
            q = project("c_q").unflatten(-1, (self.num_q_heads, self.head_dim))
            q = zero_centered_rms_norm(q, self.tensors["q_norm.norm.weight"], self.norm_eps)
            k = zero_centered_rms_norm(k, self.tensors["k_norm.norm.weight"], self.norm_eps)
            pos_scale = (positions.float() + 1).clamp_max(self.sliding_window).log()
            q = q * pos_scale[..., None, None] * self.tensors["softmax_scaler"][None, None, :, None]
        attn = attn_dispatch(
            q = q.half(), k = k.half(), v = v.half(),
            cache = cache, cache_idx = self.layer_idx, cache_instance = params.get("layer_instance"),
            block_table = get_for_device(params, "block_table", x.device) if cache is not None else None,
            cache_seqlens = get_for_device(params, "cache_seqlens", x.device) if lengths is not None else None,
            max_kv_len = int(lengths.max()) if lengths is not None and lengths.device.type == "cpu" else None,
            window_size = self.sliding_window, softcap = self.softcap,
            sm_scale = 1 / self.head_dim, dispatch_cache = self.dispatch_cache,
        )
        if not fused:
            self.save_state(layer, slots, (raw_k[:, -1], raw_v[:, -1]))
        heads_per_group = self.num_q_heads // self.num_kv_heads
        groups = attn.float().unflatten(-2, (self.num_kv_heads, heads_per_group))
        noise_weight = project("lambda_proj").sigmoid()[..., None, None]
        signal = groups[..., :-1, :] - noise_weight * groups[..., -1:, :]
        return signal.flatten(-3).to(out_dtype or torch.float32)
