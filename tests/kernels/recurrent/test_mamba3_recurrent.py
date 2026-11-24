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
