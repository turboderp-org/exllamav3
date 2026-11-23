"""
An independent bit-layout reference for the GDN packed recurrent-state pool used by the state
kernel tests: a torch.int8 (char) payload of W-bit values (2 <= W <= 12), v-major
(rows, 1, nv, dv, dk * W // 8), plus fp16 scales, one absmax per 32-wide k slice. The k-th value
of a v column sits at bit k * W of the column's byte stream (little-endian). This module implements
that layout on its own (64-bit windows on the pack side, 16-bit units on the unpack side), so it
shares no code with the kernel-mirroring converters it is tested against.

These helpers work in the model's basis, i.e. on a state as the rule computes it (the scan kernel
dequantizes the pool values and uses them directly, rotating q/k warp-locally).
"""

import torch


def pool_from_state(state: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(rows, 1, nv, dk, dv) fp32 in the model's basis -> packed int pool (rows, 1, nv, dv,
    dk * bits // 8) + fp16 scales (rows, 1, nv, dv, dk // 32): one absmax per 32-wide k slice,
    value = round(state * Q / amax) where Q = 2^(bits-1) - 1"""
    rows, _, nv, dk, dv = state.shape
    b = bits
    subk = dk // 32
    q_bound = (1 << (b - 1)) - 1
    m = state.float()
    amax = m.view(rows, 1, nv, subk, 32, dv).abs().amax(dim = 4)      # (rows, 1, nv, subk, dv)
    inv = torch.where(amax > 0, q_bound / amax, 0.0)
    scales = amax.squeeze(1).permute(0, 1, 3, 2).contiguous().to(torch.float16)     # (rows, nv, dv, subk)
    q = torch.round(m.view(rows, 1, nv, subk, 32, dv) * inv.unsqueeze(4)).view(rows, 1, nv, dk, dv).to(torch.int16)
    values = _pack_row(q.view(rows, nv, dk, dv), b)
    return values.view(rows, 1, nv, dv, dk * b // 8), scales.view(rows, 1, nv, dv, subk)


def state_from_pool(values: torch.Tensor, scales: torch.Tensor, bits: int) -> torch.Tensor:
    """packed int pool (rows, 1, nv, dv, dk * bits // 8) + scales (rows, 1, nv, dv, dk // 32) ->
    fp32 state in the model's basis. The stored scale is the slice's absmax, the step is amax / Q"""
    rows, _, nv, dv, dbits = values.shape
    b = bits
    dk = dbits * 8 // b
    subk = dk // 32
    q_bound = (1 << (b - 1)) - 1
    q = _unpack_row(values.view(rows, nv, dv, dbits), b)              # (rows, nv, dk, dv)
    s = scales.float().squeeze(1).permute(0, 1, 3, 2)                        # (rows, nv, subk, dv)
    q = q.view(rows, nv, subk, 32, dv) * s.unsqueeze(3) / q_bound
    return q.view(rows, 1, nv, dk, dv)


def _pack_row(q: torch.Tensor, b: int) -> torch.Tensor:
    """(rows, nv, dk, dv) signed quantized values -> (rows, nv, dv, dk * b // 8) packed bytes.
    Value e of the column at bit e * b; each value's b-bit two's-complement pattern ORs into the
    64-bit window holding that offset (and the next window when it straddles a 64-bit boundary)"""
    rows, nv, dk, dv = q.shape
    win = torch.zeros(rows, nv, dv, 2 * b, dtype = torch.int64, device = q.device)
    mask = (1 << b) - 1
    for e in range(dk):
        base = e * b
        v = q[:, :, e, :].to(torch.int64) & mask
        lo = base // 64
        sh = base % 64
        win[:, :, :, lo] = win[:, :, :, lo] | (v << sh)
        if sh + b > 64:
            win[:, :, :, lo + 1] = win[:, :, :, lo + 1] | (v >> (64 - sh))
    return win.view(torch.int8)


def _unpack_row(values: torch.Tensor, b: int) -> torch.Tensor:
    """(rows, nv, dv, dk * b // 8) packed bytes -> (rows, nv, dk, dv) signed values. The mirror of
    _pack_row: value e sits at bit e * b of the column's stream; the 16-bit unit p = (e * b) >> 4
    holds its low bits at bit s16 = (e * b) & 15, the next unit the high bits when the value leaves
    p (the units stay unsigned, a 64-bit shift would sign-extend a negative window)"""
    rows, nv, dv, dbits = values.shape
    dk = dbits * 8 // b
    r = values.view(rows, nv, dv, dbits // 2, 2).to(torch.int32)
    u = (r[:, :, :, :, 0] & 0xff) | ((r[:, :, :, :, 1] & 0xff) << 8)    # (rows, nv, dv, dbits // 2)
    out = torch.zeros(rows, nv, dk, dv, dtype = torch.int64, device = values.device)
    mask = (1 << b) - 1
    for e in range(dk):
        base = e * b
        s16 = base & 15
        p = base >> 4
        if s16 + b > 16:
            code = (u[:, :, :, p] >> s16) | ((u[:, :, :, p + 1] << (16 - s16)) & 0xffff)
        else:
            code = u[:, :, :, p] >> s16
        code = code & mask
        out[:, :, e, :] = torch.where(code >= (1 << (b - 1)), code - (1 << b), code)
    return out
