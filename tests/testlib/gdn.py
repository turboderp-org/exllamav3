"""
The GDN int8 recurrent-state pool for the state kernel tests: int8 values plus fp16 scales, one
absmax per 32-wide k slice (SUBK = 4 scales per v row at the 128 head dim int8 pools allow),
values = round(state * 127 / amax).

These helpers work in the model's basis, i.e. on a state as the rule computes it. The pool's own
basis is the model basis rotated per 32-wide k slice (see gated_delta_rule.pack_state_row, which
takes a row in the model basis and writes the pool basis)
"""

import torch


def pool_from_state(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(rows, 1, nv, kd, vd) fp32 in the model's basis -> int8 pool of the same shape + fp16 scales
    (rows, 1, nv, vd, kd // 32): one absmax per 32-wide k slice, values = round(state * 127 / amax)"""
    b = state.float().view(*state.shape[:-2], state.shape[-2] // 32, 32, state.shape[-1])
    amax = b.abs().amax(dim = -2, keepdim = True)
    inv = torch.where(amax > 0, 127.0 / amax, 0.0)
    q = torch.round(b * inv).to(torch.int8).view(*state.shape)
    scales = amax.squeeze(-2).permute(0, 1, 2, 4, 3)
    return q, scales.contiguous().to(torch.float16)


def state_from_pool(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """int8 pool (rows, 1, nv, kd, vd) + scales (rows, 1, nv, vd, kd // 32) -> fp32 state in the
    model's basis. The stored scale is the slice's absmax, the step is amax / 127"""
    v = values.float().view(*values.shape[:-2], values.shape[-2] // 32, 32, values.shape[-1])
    s = scales.float().permute(0, 1, 2, 4, 3).unsqueeze(-2)
    return (v * s * (1.0 / 127.0)).view(*values.shape)
