"""FP32 Mamba-3 MIMO prefill and decode kernels."""
import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


def mimo_readout(mixed, values, z, gate_proj, out_proj, skip):
    """Gate and sum the MIMO ranks directly into token-major output."""
    gate = torch.nn.functional.silu(z.unsqueeze(-2) * gate_proj[None, :, None])
    output = ((mixed + skip[None, :, None, None, None] * values) * gate * out_proj[None, :, None]).sum(dim = -2)
    return output.transpose(1, 2).flatten(-2)


def block_weights(cumulative, current, previous):
    """Causal trapezoid weights for each pair of tokens in a block."""
    length = cumulative.shape[-1]
    indices = torch.arange(length, device = cumulative.device)
    causal = indices[:, None] >= indices
    earlier = indices[:, None] > indices
    following = torch.nn.functional.pad(previous[..., 1:], (0, 1))
    decay = (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).masked_fill(~causal, -torch.inf).exp()
    return decay * (current.unsqueeze(-2) + earlier * following.unsqueeze(-2))


def scan_block_states(state, boundary, end_update, carried):
    """Return each block's incoming state and the final state, in recurrence order."""
    incoming = []
    for i in range(boundary.shape[2]):
        start_state = state + boundary[:, :, i]
        incoming.append(start_state)
        state = carried[:, :, i, -1, None, None] * start_state + end_update[:, :, i]
    return torch.stack(incoming, dim = 2), state


def apply_block_state(query, state, local, decay, rank):
    """Add the decayed state projection to the local output for each block."""
    from_state = query.unflatten(-2, (-1, rank)) @ state.mT.unsqueeze(-3)
    return local + (from_state * decay[..., None, None]).flatten(-3, -2)


# H is the number of heads, P the value width, N the state width and R the MIMO rank.
# This pass prepares Q and coefficients per head and saves the previous K and V
# before updating the slot. The state pass can then tile the value axis independently.
@triton.jit
def _prepare_decode(Projected, Dynamic, DtBias, BBias, CBias, BNorm, CNorm,
                    Slots, Angle, PrevKey, PrevValue, Query, OldKey, OldValue, Coeff,
                    H: tl.constexpr, P: tl.constexpr, N: tl.constexpr, R: tl.constexpr,
                    Eps: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(Slots + batch).to(tl.int32)
    a: tl.constexpr = N // 4
    proj = Projected + (batch * H + head) * (2 * P + 3)
    dyn = Dynamic + batch * (2 * R * N + a)
    cache_head = slot * H + head
    scratch_head = batch * H + head
    dt_x = tl.load(proj + 2 * P) + tl.load(DtBias + head)
    dt = tl.where(dt_x > 20., dt_x, libdevice.log1p(tl.exp(dt_x)))
    decay_x = tl.load(proj + 2 * P + 1)
    decay = tl.exp(-tl.maximum(tl.where(decay_x > 20., decay_x, libdevice.log1p(tl.exp(decay_x))), 1.e-4) * dt)
    trap = tl.sigmoid(tl.load(proj + 2 * P + 2))
    tl.store(Coeff + scratch_head * 3, decay)
    tl.store(Coeff + scratch_head * 3 + 1, trap * dt)
    tl.store(Coeff + scratch_head * 3 + 2, (1. - trap) * dt)

    n = tl.arange(0, N)
    r = tl.arange(0, R)
    rn = r[:, None] * N + n[None, :]
    rotate = (n % (N // 2)) < a
    partner = n ^ (N // 2)
    old_angle = tl.load(Angle + cache_head * a + n % a)
    theta = old_angle + (libdevice.tanh(tl.load(dyn + 2 * R * N + n % a)) * dt) * math.pi
    theta_mod = libdevice.fmod(theta, 2 * math.pi)
    theta_mod = tl.where(theta_mod < 0, theta_mod + 2 * math.pi, theta_mod)
    tl.store(Angle + cache_head * a + n, theta_mod, n < a)
    cosine = tl.where(rotate, tl.cos(theta), 1.)
    sine = tl.where(rotate, tl.sin(theta) * tl.where(n < N // 2, -1., 1.), 0.)

    for is_query in tl.static_range(2):
        src = dyn + is_query * R * N
        bias = (CBias if is_query else BBias) + head * R * N
        norm = CNorm if is_query else BNorm
        raw = tl.load(src + rn)
        inv = tl.rsqrt(tl.sum(raw * raw, 1) / N + Eps)
        direct = (raw * inv[:, None]) * (1. + tl.load(norm + n).to(tl.float32))[None, :] + tl.load(bias + rn)
        paired_idx = r[:, None] * N + partner[None, :]
        paired = (tl.load(src + paired_idx) * inv[:, None]) * (
            1. + tl.load(norm + partner).to(tl.float32)
        )[None, :] + tl.load(bias + paired_idx)
        rotated = direct * cosine[None, :] + paired * sine[None, :]
        if is_query:
            tl.store(Query + scratch_head * R * N + rn, rotated)
        else:
            old = tl.load(PrevKey + cache_head * R * N + rn)
            tl.store(OldKey + scratch_head * R * N + rn, old)
            tl.store(PrevKey + cache_head * R * N + rn, rotated)

    p = tl.arange(0, P)
    old_v = tl.load(PrevValue + cache_head * P + p)
    tl.store(OldValue + scratch_head * P + p, old_v)
    tl.store(PrevValue + cache_head * P + p, tl.load(proj + P + p))


@triton.jit
def _update_decode(Projected, Slots, State, NewKey, Query, OldKey, OldValue, Coeff,
                   XProj, ZProj, OutProj, Skip, Output,
                   H: tl.constexpr, P: tl.constexpr, N: tl.constexpr, R: tl.constexpr,
                   Tile: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    p = tl.program_id(2) * Tile + tl.arange(0, Tile)
    n = tl.arange(0, N)
    slot = tl.load(Slots + b).to(tl.int32)
    sh = b * H + h
    ch = slot * H + h
    proj = Projected + sh * (2 * P + 3)
    z = tl.load(proj + p)
    v = tl.load(proj + P + p)
    old_v = tl.load(OldValue + sh * P + p)
    decay = tl.load(Coeff + sh * 3)
    current_weight = tl.load(Coeff + sh * 3 + 1)
    previous_weight = tl.load(Coeff + sh * 3 + 2)
    state_ptr = State + (ch * P + p[:, None]) * N + n[None, :]
    state = tl.load(state_ptr)
    current = tl.full((Tile, N), 0, tl.float32)
    previous = tl.full((Tile, N), 0, tl.float32)
    for r in tl.static_range(R):
        k = tl.load(NewKey + (ch * R + r) * N + n)
        old_k = tl.load(OldKey + (sh * R + r) * N + n)
        x_proj = tl.load(XProj + (h * R + r) * P + p)
        current = current + (v * x_proj)[:, None] * k[None, :]
        previous = previous + (old_v * x_proj)[:, None] * old_k[None, :]
    state = decay * (state + previous_weight * previous) + current_weight * current
    tl.store(state_ptr, state)

    output = tl.full((Tile,), 0, tl.float32)
    skip = tl.load(Skip + h)
    for r in tl.static_range(R):
        q = tl.load(Query + (sh * R + r) * N + n)
        mixed = tl.sum(state * q[None, :], 1)
        rp = (h * R + r) * P + p
        value = v * tl.load(XProj + rp)
        gate = z * tl.load(ZProj + rp)
        output = output + ((mixed + skip * value) * (gate * tl.sigmoid(gate))) * tl.load(OutProj + rp)
    tl.store(Output + sh * P + p, output)


def mamba3_decode(projected, dynamic, weights, slots, state, heads, width, size, rank, eps):
    """Advance the selected FP32 slots in place and return (batch, 1, heads * width)."""
    batch = projected.shape[0]
    angle, ssm, key, value = state
    query = projected.new_empty((batch, heads, rank, size))
    old_key = torch.empty_like(query)
    old_value = projected.new_empty((batch, heads, width))
    coeff = projected.new_empty((batch, heads, 3))
    output = projected.new_empty((batch, 1, heads * width))
    _prepare_decode[(batch, heads)](
        projected, dynamic, weights["dt_bias"], weights["B_bias"], weights["C_bias"],
        weights["B_norm.norm.weight"], weights["C_norm.norm.weight"], slots, angle, key, value,
        query, old_key, old_value, coeff, heads, width, size, rank, eps, enable_fp_fusion=False,
    )
    tile = 8
    _update_decode[(batch, heads, width // tile)](
        projected, slots, ssm, key, query, old_key, old_value, coeff,
        weights["in_proj_mimo_x"], weights["in_proj_mimo_z"], weights["out_proj_mimo"], weights["D"],
        output, heads, width, size, rank, tile, enable_fp_fusion=False,
    )
    return output
