"""FP32 Mamba-3 recurrence and in-place decode against torch references."""

import math

import pytest
import torch

from exllamav3.modules.mamba3 import mamba3_block


@pytest.mark.nogpu
@pytest.mark.parametrize("length, block_size", [(1, 1), (7, 7), (32, 32), (64, 16)])
def test_mamba3_block_matches_sequential_recurrence(length, block_size):
    rng = torch.Generator().manual_seed(123)
    batch, heads, rank, width, state_dim = 2, 3, 4, 8, 12

    def rand(*shape):
        return torch.randn(*shape, generator = rng)

    query = rand(batch, heads, length, rank, state_dim)
    key = rand(batch, heads, length, rank, state_dim)
    value = rand(batch, heads, length, rank, width)
    dt = rand(batch, heads, length).sigmoid()
    log_decay = -rand(batch, heads, length).abs() * dt
    trap = rand(batch, heads, length).sigmoid()
    state = rand(batch, heads, width, state_dim)
    previous_key = rand(batch, heads, rank, state_dim)
    previous_value = rand(batch, heads, rank, width)
    actual, actual_state = mamba3_block(
        query, key, value, log_decay, dt, trap, state, previous_key, previous_value, block_size,
    )

    expected = []
    for i in range(length):
        decay = log_decay[..., i].exp()[..., None, None]
        gamma = (trap[..., i] * dt[..., i])[..., None, None]
        beta = ((1 - trap[..., i]) * dt[..., i])[..., None, None] * decay
        current_key, current_value = key[:, :, i], value[:, :, i]
        state = decay * state + gamma * (current_value.mT @ current_key)
        state = state + beta * (previous_value.mT @ previous_key)
        expected.append(query[:, :, i] @ state.mT)
        previous_key, previous_value = current_key, current_value

    torch.testing.assert_close(actual, torch.stack(expected, dim = 2), atol = 2e-5, rtol = 2e-5)
    torch.testing.assert_close(actual_state, state, atol = 2e-5, rtol = 2e-5)


def test_fused_mamba3_updates_selected_fp32_slots(device):
    from exllamav3.modules.mamba3 import rotate_half
    from exllamav3.modules.mamba3_triton import mamba3_decode
    from exllamav3.modules.rmsnorm import zero_centered_rms_norm

    rng = torch.Generator(device = device).manual_seed(17)
    batch, heads, rank, width, size = 2, 3, 4, 64, 128
    eps = 1e-6

    def rand(*shape):
        return torch.randn(*shape, generator = rng, device = device)

    params = {
        "dt_bias": rand(heads),
        "B_bias": rand(heads, rank, size),
        "C_bias": rand(heads, rank, size),
        "B_norm.norm.weight": rand(size).bfloat16(),
        "C_norm.norm.weight": rand(size).bfloat16(),
        "in_proj_mimo_x": rand(heads, rank, width),
        "in_proj_mimo_z": rand(heads, rank, width),
        "out_proj_mimo": rand(heads, rank, width),
        "D": rand(heads),
    }
    state = (
        rand(4, heads, size // 4).remainder(2 * math.pi), rand(4, heads, width, size),
        rand(4, heads, rank, size), rand(4, heads, width),
    )
    slots = torch.tensor([3, 1], dtype = torch.int32, device = device)
    untouched = [t[[0, 2]].clone() for t in state]
    angle, ssm, prev_key, prev_value = [t[slots] for t in state]
    for _ in range(3):
        p = rand(batch, heads, 2 * width + 3)
        d = rand(batch, 2 * rank * size + size // 4)
        z, v, dt, decay, trap = p.split((width, width, 1, 1, 1), dim = -1)
        kin, qin, raw_angle = d.split((rank * size, rank * size, size // 4), dim = -1)
        dt = torch.nn.functional.softplus(dt.squeeze(-1) + params["dt_bias"])
        decay = -torch.nn.functional.softplus(decay.squeeze(-1)).clamp_min(1e-4) * dt
        trap = trap.squeeze(-1).sigmoid()
        angles = angle + raw_angle[:, None].tanh() * dt[..., None] * math.pi
        q = zero_centered_rms_norm(qin.view(batch, rank, size), params["C_norm.norm.weight"], eps)
        k = zero_centered_rms_norm(kin.view(batch, rank, size), params["B_norm.norm.weight"], eps)
        q = rotate_half(q[:, None] + params["C_bias"], angles.unsqueeze(-2))
        k = rotate_half(k[:, None] + params["B_bias"], angles.unsqueeze(-2))
        v_proj = params["in_proj_mimo_x"]
        values = v[..., None, :] * v_proj
        mixed, ssm = mamba3_block(
            q.unsqueeze(2), k.unsqueeze(2), values.unsqueeze(2), decay.unsqueeze(-1),
            dt.unsqueeze(-1), trap.unsqueeze(-1), ssm, prev_key, prev_value[..., None, :] * v_proj,
        )
        gate = torch.nn.functional.silu(z[..., None, :] * params["in_proj_mimo_z"])
        expected = ((mixed.squeeze(2) + params["D"][None, :, None, None] * values)
                    * gate * params["out_proj_mimo"]).sum(dim = -2).flatten(1)
        actual = mamba3_decode(
            p.view(batch, 1, -1), d.unsqueeze(1), params, slots, state, heads, width, size, rank, eps,
        )
        torch.testing.assert_close(actual[:, 0], expected, atol = 4e-4, rtol = 4e-5)
        angle = angles.remainder(2 * math.pi)
        prev_key, prev_value = k, v
        for cached, ref, unchanged in zip(state, (angle, ssm, prev_key, prev_value), untouched, strict = True):
            torch.testing.assert_close(cached[slots], ref, atol = 2e-5, rtol = 2e-5)
            assert torch.equal(cached[[0, 2]], unchanged)


def test_mamba3_prefill_with_strided_token_slices(device):
    from exllamav3.modules.mamba3 import rotate_half, rotate_qk
    from exllamav3.modules.mamba3_triton import mimo_readout, scan_block_states

    batch, heads, length, rank, size, width = 2, 3, 11, 4, 128, 64
    rng = torch.Generator(device = device).manual_seed(92)

    def rand(*shape):
        return torch.randn(*shape, generator = rng, device = device)

    # A tail chunk of a multi-sequence prefill has gaps between batch rows.
    q, k = [rand(batch, 2 * length, rank, size)[:, length:] for _ in range(2)]
    q_bias, k_bias = [rand(heads, rank, size) for _ in range(2)]
    angles = rand(batch, heads, length, size // 4)
    actual = rotate_qk(q, k, q_bias, k_bias, angles)
    for rotated, source, bias in zip(actual, (q, k), (q_bias, k_bias)):
        ref = rotate_half(source[:, None] + bias[None, :, None], angles.unsqueeze(-2))
        torch.testing.assert_close(rotated, ref, atol = 0, rtol = 0)

    state = rand(batch, heads, 7, 12)
    # The boundary product can be block-major even when the end update is head-major.
    boundary = rand(batch, 3, heads, 7, 12).transpose(1, 2)
    update = rand(batch, heads, 3, 7, 12)
    decay = rand(batch, heads, 3, length).sigmoid()
    incoming, final = scan_block_states(state, boundary, update, decay)
    for i in range(3):
        state = state + boundary[:, :, i]
        torch.testing.assert_close(incoming[:, :, i], state, atol = 0, rtol = 0)
        state = state * decay[:, :, i, -1, None, None] + update[:, :, i]
    torch.testing.assert_close(final, state, atol = 0, rtol = 0)

    mixed = rand(batch, heads, length, rank, width)
    v = rand(batch, length * 2, heads, rank, width)[:, length:].transpose(1, 2)
    z = rand(batch, length * 2, heads, width)[:, length:].transpose(1, 2)
    g, o = [rand(heads, rank, width) for _ in range(2)]
    skip = rand(heads)
    gate = torch.nn.functional.silu(z.unsqueeze(-2) * g[None, :, None])
    ref = ((mixed + v * skip[None, :, None, None, None]) * gate * o[None, :, None]).sum(-2)
    torch.testing.assert_close(mimo_readout(mixed, v, z, g, o, skip), ref.transpose(1, 2).flatten(-2))
