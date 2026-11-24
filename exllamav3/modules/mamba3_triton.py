"""FP32 Mamba-3 MIMO prefill and decode kernels."""
import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ..util.backend import gfx_arch


@triton.jit
def _bias_rotate(Input, Bias, Cos, Sin, Output, Length: tl.constexpr, Heads: tl.constexpr,
                 Rank: tl.constexpr, Size: tl.constexpr, BatchStride, TimeStride,
                 Tile: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    pair = tl.arange(0, Tile)
    half: tl.constexpr = Size // 2
    angles: tl.constexpr = Size // 4
    channel = pair % half
    left = pair // half * Size + channel
    valid = pair < Rank * half
    src = Input + batch * BatchStride + token * TimeStride
    bias = Bias + head * Rank * Size
    x = tl.load(src + left, valid, 0) + tl.load(bias + left, valid, 0).to(tl.float32)
    y = tl.load(src + left + half, valid, 0) + tl.load(bias + left + half, valid, 0).to(tl.float32)
    row = (batch * Heads + head) * Length + token
    rotate = channel < angles
    c = tl.load(Cos + row * angles + channel, rotate, 0)
    s = tl.load(Sin + row * angles + channel, rotate, 0)
    dst = Output + row * Rank * Size
    tl.store(dst + left, tl.where(rotate, x * c - y * s, x), valid)
    tl.store(dst + left + half, tl.where(rotate, x * s + y * c, y), valid)


def bias_rotate(x, bias, cosine, sine):
    """Broadcast per-head biases and rotate Q or K directly into its block GEMM layout."""
    batch, length, rank, size = x.shape
    heads = bias.shape[0]
    output = x.new_empty((batch, heads, length, rank, size))
    _bias_rotate[(length, heads, batch)](
        x, bias, cosine, sine, output, length, heads, rank, size, x.stride(0), x.stride(1),
        triton.next_power_of_2(rank * (size // 2)), enable_fp_fusion=False,
    )
    return output


@triton.jit
def _mimo_readout(Mixed, Values, Z, GateProj, OutProj, Skip, Output,
                  Length: tl.constexpr, Heads: tl.constexpr, Rank: tl.constexpr, Width: tl.constexpr,
                  StrideVB, StrideVH, StrideVT, StrideZB, StrideZH, StrideZT, TileT: tl.constexpr, TileP: tl.constexpr):
    t = tl.program_id(0) * TileT + tl.arange(0, TileT)
    h = tl.program_id(1)
    b = tl.program_id(2)
    p = tl.arange(0, TileP)
    valid = (t[:, None] < Length) & (p[None, :] < Width)
    z = tl.load(Z + b * StrideZB + h * StrideZH + t[:, None] * StrideZT + p[None, :], valid, 0)
    skip = tl.load(Skip + h)
    row = (b * Heads + h) * Length + t
    acc = tl.full((TileT, TileP), 0, tl.float32)
    for r in tl.static_range(Rank):
        weight = (h * Rank + r) * Width + p
        idx = (row[:, None] * Rank + r) * Width + p[None, :]
        vptr = Values + b * StrideVB + h * StrideVH + t[:, None] * StrideVT
        v = tl.load(vptr + r * Width + p[None, :], valid, 0)
        mixed = tl.load(Mixed + idx, valid, 0)
        gate = z * tl.load(GateProj + weight, p < Width, 0)[None, :]
        gate = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
        out = tl.load(OutProj + weight, p < Width, 0)[None, :]
        acc += ((mixed + skip * v) * gate) * out
    dst = ((b * Length + t) * Heads + h) * Width
    tl.store(Output + dst[:, None] + p[None, :], acc, valid)


def mimo_readout(mixed, values, z, gate_proj, out_proj, skip):
    """Gate and sum the MIMO ranks directly into token-major output."""
    batch, heads, length, rank, width = mixed.shape
    if not (
        mixed.is_cuda and gfx_arch(mixed.device) == "gfx1100" and rank == 4 and width == 64
        and mixed.dtype == values.dtype == z.dtype == torch.float32
        and mixed.is_contiguous() and values.stride()[-2:] == (width, 1) and z.stride(-1) == 1
        and all(t.is_contiguous() for t in (gate_proj, out_proj, skip))
    ):
        gate = torch.nn.functional.silu(z.unsqueeze(-2) * gate_proj[None, :, None])
        output = ((mixed + skip[None, :, None, None, None] * values) * gate * out_proj[None, :, None]).sum(dim = -2)
        return output.transpose(1, 2).flatten(-2)

    output = mixed.new_empty((batch, length, heads * width))
    _mimo_readout[(triton.cdiv(length, 8), heads, batch)](
        mixed, values, z, gate_proj, out_proj, skip, output, length, heads, rank, width,
        *values.stride()[:3], *z.stride()[:3], 8, triton.next_power_of_2(width), enable_fp_fusion=False,
    )
    return output


@triton.jit
def _block_weights(Cumulative, Current, Previous, Output, Heads: tl.constexpr, Blocks: tl.constexpr,
                   Length: tl.constexpr, StrideB, StrideH, StrideBlock, StrideT, Tile: tl.constexpr):
    block = tl.program_id(0)
    row = tl.arange(0, Tile)
    col = tl.arange(0, Tile)
    offset = block * Length
    cumulative = tl.load(Cumulative + offset + row, row < Length, 0)
    base = block // (Heads * Blocks) * StrideB + (block // Blocks % Heads) * StrideH + block % Blocks * StrideBlock
    current = tl.load(Current + base + col * StrideT, col < Length, 0)
    following = tl.load(Previous + base + (col + 1) * StrideT, col + 1 < Length, 0)
    delta = cumulative[:, None] - cumulative[None, :]
    decay = libdevice.exp(tl.where(row[:, None] >= col[None, :], delta, -float("inf")))
    weight = current[None, :] + tl.where(row[:, None] > col[None, :], following[None, :], 0.)
    tl.store(Output + (offset + row[:, None]) * Length + col[None, :], decay * weight,
             (row[:, None] < Length) & (col[None, :] < Length))


def block_weights(cumulative, current, previous):
    """Causal trapezoid weights for each pair of tokens in a block."""
    _, heads, blocks, length = cumulative.shape
    if (cumulative.is_cuda and gfx_arch(cumulative.device) == "gfx1100" and length <= 32
            and cumulative.is_contiguous() and current.stride() == previous.stride()
            and cumulative.dtype == current.dtype == previous.dtype == torch.float32):
        weights = cumulative.new_empty((*cumulative.shape, length))
        _block_weights[(cumulative.numel() // length,)](
            cumulative, current, previous, weights, heads, blocks, length, *current.stride(),
            triton.next_power_of_2(length), enable_fp_fusion=False,
        )
        return weights

    indices = torch.arange(length, device = cumulative.device)
    causal = indices[:, None] >= indices
    earlier = indices[:, None] > indices
    following = torch.nn.functional.pad(previous[..., 1:], (0, 1))
    decay = (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).masked_fill(~causal, -torch.inf).exp()
    return decay * (current.unsqueeze(-2) + earlier * following.unsqueeze(-2))


@triton.jit
def _scan_block_states(State, Boundary, EndUpdate, Decay, Incoming, Final,
                       Heads: tl.constexpr, Blocks: tl.constexpr, Length: tl.constexpr, Elements: tl.constexpr,
                       StrideB: tl.constexpr, StrideH: tl.constexpr, StrideBlock: tl.constexpr,
                       Tile: tl.constexpr):
    head = tl.program_id(1)  # flattened batch and head
    i = tl.program_id(0) * Tile + tl.arange(0, Tile)
    valid = i < Elements
    state = tl.load(State + head * Elements + i, valid, 0)
    boundary = Boundary + (head // Heads) * StrideB + (head % Heads) * StrideH
    for block in range(Blocks):
        row = head * Blocks + block
        p = row * Elements + i
        state += tl.load(boundary + block * StrideBlock + i, valid, 0)
        tl.store(Incoming + p, state, valid)
        decay = tl.load(Decay + (row + 1) * Length - 1)
        state = decay * state + tl.load(EndUpdate + p, valid, 0)
    tl.store(Final + head * Elements + i, state, valid)


def scan_block_states(state, boundary, end_update, carried):
    """Return each block's incoming state and the final state, in recurrence order."""
    if (
        state.is_cuda and gfx_arch(state.device) == "gfx1100"
        and all(t.is_contiguous() and t.dtype == torch.float32 for t in (state, end_update, carried))
        and boundary.dtype == torch.float32 and boundary.stride()[-2:] == (boundary.shape[-1], 1)
    ):
        batch, heads, blocks, length = carried.shape
        elements = state.shape[-2] * state.shape[-1]
        incoming = torch.empty_like(end_update)
        final = torch.empty_like(state)
        tile = 256
        _scan_block_states[(triton.cdiv(elements, tile), batch * heads)](
            state, boundary, end_update, carried, incoming, final, heads, blocks, length, elements,
            *boundary.stride()[:3], tile,
            enable_fp_fusion=False,
        )
        return incoming, final

    incoming = []
    for i in range(boundary.shape[2]):
        start_state = state + boundary[:, :, i]
        incoming.append(start_state)
        state = carried[:, :, i, -1, None, None] * start_state + end_update[:, :, i]
    return torch.stack(incoming, dim = 2), state


@triton.jit
def _apply_block_state(Q, State, Local, Decay, Output,
                       Rows: tl.constexpr, Rank: tl.constexpr, Width: tl.constexpr, Size: tl.constexpr,
                       TileM: tl.constexpr, TileP: tl.constexpr, TileN: tl.constexpr):
    block = tl.program_id(0)  # flattened batch, head and time block
    m = tl.program_id(1) * TileM + tl.arange(0, TileM)
    p = tl.program_id(2) * TileP + tl.arange(0, TileP)
    n = tl.arange(0, TileN)
    q = Q + (block * Rows + m[:, None]) * Size + n[None, :]
    state = State + (block * Width + p[:, None]) * Size + n[None, :]
    acc = tl.full((TileM, TileP), 0, tl.float32)
    for k in range(Size // TileN):
        q_tile = tl.load(q + k * TileN)
        state_tile = tl.load(state + k * TileN)
        acc = tl.dot(q_tile, tl.trans(state_tile), acc, input_precision = "ieee")

    offsets = (block * Rows + m[:, None]) * Width + p[None, :]
    decay = tl.load(Decay + (block * Rows + m) // Rank)
    tl.store(Output + offsets, tl.load(Local + offsets) + acc * decay[:, None])


def apply_block_state(query, state, local, decay, rank):
    """Add the decayed state projection to the local output for each block."""
    rows, size = query.shape[-2:]
    width = local.shape[-1]
    tile_m, tile_p, tile_n = 16, 64, 32
    if not (
        query.is_cuda and gfx_arch(query.device) == "gfx1100"
        and rows % tile_m == 0 and width % tile_p == 0 and size % tile_n == 0
        and all(t.is_contiguous() and t.dtype == torch.float32 for t in (query, state, local, decay))
    ):
        # Preserve per-token FP32 accumulation order.
        from_state = query.unflatten(-2, (-1, rank)) @ state.mT.unsqueeze(-3)
        return local + (from_state * decay[..., None, None]).flatten(-3, -2)

    output = torch.empty_like(local)
    grid = (query.numel() // (rows * size), rows // tile_m, width // tile_p)
    _apply_block_state[grid](
        query, state, local, decay, output, rows, rank, width, size, tile_m, tile_p, tile_n,
        enable_fp_fusion=False,
    )
    return output


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
