"""Torch reference operations for FP32 Mamba-3 MIMO prefill."""

import torch


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
