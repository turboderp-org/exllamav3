from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from typing_extensions import override

from ..model.config import Config
from ..util.backend import gfx_arch
from .linear import Linear
from .mamba3_triton import apply_block_state, bias_rotate, block_weights, mamba3_decode, mimo_readout, scan_block_states
from .multilinear import MultiLinear
from .recurrent import RecurrentMixer
from .rmsnorm import zero_centered_rms_norm


def rotate_half(x, angle):
    """Rotate a prefix of each half of the head by the supplied angles."""
    half = x.shape[-1] // 2
    count = angle.shape[-1]
    left, right = x[..., :count], x[..., half:half + count]
    cosine, sine = angle.cos(), angle.sin()
    return torch.cat((
        left * cosine - right * sine, x[..., count:half],
        left * sine + right * cosine, x[..., half + count:],
    ), dim = -1)


def rotate_qk(q, k, q_bias, k_bias, angles):
    rank, size = q.shape[-2:]
    if (
        q.is_cuda and gfx_arch(q.device) == "gfx1100" and rank * size <= 4096
        and q.dtype == k.dtype == torch.float32 and q.stride()[-2:] == k.stride()[-2:] == (size, 1)
        and q_bias.is_contiguous() and k_bias.is_contiguous()
    ):
        cosine, sine = angles.cos().contiguous(), angles.sin().contiguous()
        return bias_rotate(q, q_bias, cosine, sine), bias_rotate(k, k_bias, cosine, sine)

    return (
        rotate_half(q[:, None] + q_bias[None, :, None], angles.unsqueeze(-2)),
        rotate_half(k[:, None] + k_bias[None, :, None], angles.unsqueeze(-2)),
    )


def mamba3_block(query, key, value, log_decay, dt, trap, state, previous_key, previous_value, block_size = None):
    """Evaluate equal-length causal blocks of the FP32 trapezoidal recurrence.

    Q, K and V are (batch, heads, time, rank, channels). Only the incoming
    matrix state depends on the preceding block. Time must be divisible
    by block_size when supplied. Batch the other products
    over blocks, keeping the quadratic score matrix local to each block.
    """
    if query.shape[2] == 1:
        state = state + ((1 - trap) * dt).unsqueeze(-1) * (previous_value.mT @ previous_key)
        state = log_decay.exp().unsqueeze(-1) * state + (trap * dt).unsqueeze(-1) * (
            value.squeeze(2).mT @ key.squeeze(2)
        )
        return (query.squeeze(2) @ state.mT).unsqueeze(2), state

    block_size = block_size or query.shape[2]

    def blocks(x):
        return x.unflatten(2, (-1, block_size))

    # The first trapezoid in each block uses the preceding block's final sample.
    last_key = torch.cat((previous_key.unsqueeze(2), blocks(key)[:, :, :-1, -1]), dim = 2)
    last_value = torch.cat((previous_value.unsqueeze(2), blocks(value)[:, :, :-1, -1]), dim = 2)
    current_weight = blocks(trap * dt)
    previous_weight = blocks((1 - trap) * dt)
    boundary = previous_weight[..., :1, None] * (last_value.mT @ last_key)
    cumulative = blocks(log_decay).cumsum(dim = -1)
    carried = cumulative.exp()
    weights = block_weights(cumulative, current_weight, previous_weight)

    rank = query.shape[-2]
    q = blocks(query).flatten(3, 4)
    k = blocks(key).flatten(3, 4)
    v = blocks(value)
    scores = (q @ k.mT).unflatten(-2, (block_size, rank)).unflatten(-1, (block_size, rank))
    scores = scores * weights.unsqueeze(-1).unsqueeze(-3)
    local = scores.flatten(-4, -3).flatten(-2) @ v.flatten(3, 4)
    end_value = v * weights[..., -1, :, None, None]
    end_update = end_value.flatten(3, 4).mT @ k

    incoming, state = scan_block_states(state, boundary, end_update, carried)
    output = apply_block_state(q, incoming, local, carried, rank)
    return output.unflatten(-2, (block_size, rank)).flatten(2, 3), state


class Mamba3(RecurrentMixer):
    """Mamba-3 MIMO mixer with input-dependent decay and trapezoidal updates.

    The checkpoint projections supply Z, V, dt, decay, trap and shared B, C,
    rotary angles. The recurrence, gating and committed state use FP32.
    """
    block_size = 32
    max_parallel_blocks = 8

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        rank: int,
        norm_eps: float,
        qmap: str | None = None,
        key_in: str = "in_proj",
        key_in_dyn: str = "in_proj_dyn",
        tensor_keys: dict[str, str] | None = None,
    ):
        if min(hidden_size, num_heads, head_dim, state_size, rank, norm_eps) <= 0 or state_size % 4:
            raise ValueError(f"{key}: invalid Mamba-3 head or rotary geometry")
        self.head_dim = head_dim
        self.heads = num_heads
        self.rank = rank
        self.state_size = state_size
        self.norm_eps = norm_eps
        self.angle_size = self.state_size // 4
        self.output_size = self.heads * self.head_dim
        h, r, p, n = self.heads, self.rank, self.head_dim, self.state_size
        state_shapes = ((h, self.angle_size), (h, p, n), (h, r, n), (h, p))
        super().__init__(config, key, layer_idx, state_shapes, {
            "B_bias": (h, r, n), "C_bias": (h, r, n),
            "D": (h,), "dt_bias": (h,),
            "in_proj_mimo_x": (h, r, p), "in_proj_mimo_z": (h, r, p), "out_proj_mimo": (h, r, p),
            "B_norm.norm.weight": (n,), "C_norm.norm.weight": (n,),
        }, tensor_keys)
        self.in_proj = Linear(
            config, f"{key}.{key_in}", in_features = hidden_size, out_features = h * (2 * p + 3),
            qmap = qmap, trim_padded_out = True,
        )
        self.in_proj_dyn = Linear(
            config, f"{key}.{key_in_dyn}", in_features = hidden_size, out_features = 2 * r * n + self.angle_size,
            qmap = qmap, trim_padded_out = True,
        )
        self.register_submodule(self.in_proj)
        self.register_submodule(self.in_proj_dyn)

        self.decode_projections = None

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        if device is not None and torch.device(device).type == "cuda" and gfx_arch(torch.device(device)) == "gfx1100":
            self.decode_projections = MultiLinear.for_decode([self.in_proj, self.in_proj_dyn], device)

    @override
    def unload(self):
        self.decode_projections = None
        super().unload()

    @override
    def optimizer_targets(self):
        return [self.in_proj.optimizer_targets(), self.in_proj_dyn.optimizer_targets()]

    @override
    def forward(self, x, params, out_dtype = None):
        width, rank, size = self.head_dim, self.rank, self.state_size
        projection_input = x.half()
        bundle = self.decode_projections
        if bundle is not None and bundle.can_project_one(projection_input, params):
            rows = bundle.project_one(projection_input, torch.float)
            projected = rows[0:1, :, :self.in_proj.out_features_unpadded]
            dynamic = rows[1:2, :, :self.in_proj_dyn.out_features_unpadded]
        else:
            projected = self.in_proj.forward(projection_input, params, out_dtype = torch.float)
            dynamic = self.in_proj_dyn.forward(projection_input, params, out_dtype = torch.float)
        if x.is_cuda and x.shape[1] == 1 and (width, size, rank) == (64, 128, 4):
            layer, slots = self.state_slots(x, params)
            if layer is not None:
                output = mamba3_decode(
                    projected, dynamic, self.tensors, slots, layer.tensors,
                    self.heads, width, size, rank, self.norm_eps,
                )
                return output.to(out_dtype or torch.float32)

        layer, slots, (angle, state, prev_k, prev_v) = self.load_state(x, params)
        z, v, dt, decay, trap = projected.unflatten(-1, (self.heads, 2 * width + 3)).split(
            (width, width, 1, 1, 1), dim = -1,
        )
        kin, qin, raw_angle = dynamic.split(
            (rank * size, rank * size, self.angle_size), dim = -1,
        )
        dt = F.softplus(dt.squeeze(-1).transpose(1, 2) + self.tensors["dt_bias"][None, :, None])
        decay = -F.softplus(decay.squeeze(-1).transpose(1, 2)).clamp_min(1e-4) * dt
        trap = trap.squeeze(-1).transpose(1, 2).sigmoid()
        angle_steps = raw_angle.tanh().unsqueeze(1) * dt.unsqueeze(-1) * math.pi
        qin = zero_centered_rms_norm(qin.unflatten(-1, (rank, size)), self.tensors["C_norm.norm.weight"], self.norm_eps)
        kin = zero_centered_rms_norm(kin.unflatten(-1, (rank, size)), self.tensors["B_norm.norm.weight"], self.norm_eps)
        v = v.transpose(1, 2)
        z = z.transpose(1, 2)
        value_proj = self.tensors["in_proj_mimo_x"]
        gate_proj = self.tensors["in_proj_mimo_z"]
        output_proj = self.tensors["out_proj_mimo"]

        outputs = []
        # Bound temporary storage while amortizing launches across full blocks.
        start = 0
        while start < x.shape[1]:
            remaining = x.shape[1] - start
            length = min(remaining, self.block_size)
            count = min(remaining // length, max(1, self.max_parallel_blocks // x.shape[0]))
            part = slice(start, start + count * length)
            start = part.stop
            increments = angle_steps[:, :, part].unflatten(2, (count, length)).cumsum(dim = 3)
            # Wrap the angle after each block, including within a batched group.
            starts = []
            for delta in increments[:, :, :, -1].unbind(2):
                starts.append(angle)
                angle = (angle + delta).remainder(2 * math.pi)
            angles = (torch.stack(starts, dim = 2).unsqueeze(3) + increments).flatten(2, 3)
            q, k = rotate_qk(qin[:, part], kin[:, part], self.tensors["C_bias"], self.tensors["B_bias"], angles)
            values = v[:, :, part, None] * value_proj[None, :, None]
            prev_values = prev_v.unsqueeze(-2) * value_proj
            mixed, state = mamba3_block(
                q, k, values, decay[:, :, part], dt[:, :, part], trap[:, :, part], state, prev_k, prev_values, length,
            )
            outputs.append(mimo_readout(mixed, values, z[:, :, part], gate_proj, output_proj, self.tensors["D"]))
            prev_k = k[:, :, -1]
            prev_v = v[:, :, part][:, :, -1]
        self.save_state(layer, slots, (angle, state, prev_k, prev_v))
        return torch.cat(outputs, dim = 1).to(out_dtype or torch.float32)
