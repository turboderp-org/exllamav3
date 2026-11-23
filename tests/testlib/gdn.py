"""Independent bit-layout reference for packed GDN recurrent states.

Packed rows contain signed two's-complement fields at consecutive bit offsets in little-endian byte order,
with one fp16 absmax scale per 32-value k slice. Packed values represent recurrent state in the H32-rotated
domain; these helpers remain in that domain and do not rotate.
"""

import torch


def quantize_pool(state: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a (rows, 1, nv, dk, dv) state in the H32-rotated domain and return packed values and scales."""
    rows, _, num_v_heads, k_head_dim, v_head_dim = state.shape
    num_k_slices = k_head_dim // 32
    quantization_bound = (1 << (bits - 1)) - 1
    rotated_state = state.float()
    absmax = rotated_state.view(rows, 1, num_v_heads, num_k_slices, 32, v_head_dim).abs().amax(dim = 4)
    quantization_scale = torch.where(absmax > 0, quantization_bound / absmax, 0.0)
    scales = absmax.squeeze(1).permute(0, 1, 3, 2).contiguous().to(torch.float16)
    codes = torch.round(
        rotated_state.view(rows, 1, num_v_heads, num_k_slices, 32, v_head_dim) * quantization_scale.unsqueeze(4)
    ).view(rows, 1, num_v_heads, k_head_dim, v_head_dim).to(torch.int16)
    packed_values = _pack_row(codes.view(rows, num_v_heads, k_head_dim, v_head_dim), bits)
    return packed_values.view(rows, 1, num_v_heads, v_head_dim, k_head_dim * bits // 8), \
        scales.view(rows, 1, num_v_heads, v_head_dim, num_k_slices)


def dequantize_pool(packed_values: torch.Tensor, scales: torch.Tensor, bits: int) -> torch.Tensor:
    """Dequantize packed values and scales to a (rows, 1, nv, dk, dv) fp32 state in the H32-rotated domain."""
    rows, _, num_v_heads, v_head_dim, packed_bytes = packed_values.shape
    k_head_dim = packed_bytes * 8 // bits
    num_k_slices = k_head_dim // 32
    quantization_bound = (1 << (bits - 1)) - 1
    codes = _unpack_row(packed_values.view(rows, num_v_heads, v_head_dim, packed_bytes), bits)
    expanded_scales = scales.float().squeeze(1).permute(0, 1, 3, 2)
    state = codes.view(rows, num_v_heads, num_k_slices, 32, v_head_dim) \
        * expanded_scales.unsqueeze(3) / quantization_bound
    return state.view(rows, 1, num_v_heads, k_head_dim, v_head_dim)


def _pack_row(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack signed codes into 64-bit windows."""
    rows, num_v_heads, k_head_dim, v_head_dim = codes.shape
    windows = torch.zeros(rows, num_v_heads, v_head_dim, 2 * bits, dtype = torch.int64, device = codes.device)
    code_mask = (1 << bits) - 1
    for value_index in range(k_head_dim):
        bit_base = value_index * bits
        code = codes[:, :, value_index, :].to(torch.int64) & code_mask
        window_index = bit_base // 64
        bit_offset = bit_base % 64
        windows[:, :, :, window_index] = windows[:, :, :, window_index] | (code << bit_offset)
        if bit_offset + bits > 64:
            windows[:, :, :, window_index + 1] = windows[:, :, :, window_index + 1] | (code >> (64 - bit_offset))
    return windows.view(torch.int8)


def _unpack_row(packed_values: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack little-endian bytes into signed codes."""
    rows, num_v_heads, v_head_dim, packed_bytes = packed_values.shape
    k_head_dim = packed_bytes * 8 // bits
    byte_pairs = packed_values.view(rows, num_v_heads, v_head_dim, packed_bytes // 2, 2).to(torch.int32)
    units = (byte_pairs[..., 0] & 0xff) | ((byte_pairs[..., 1] & 0xff) << 8)
    unpacked = torch.zeros(
        rows, num_v_heads, k_head_dim, v_head_dim, dtype = torch.int64, device = packed_values.device
    )
    code_mask = (1 << bits) - 1
    for value_index in range(k_head_dim):
        bit_base = value_index * bits
        bit_offset = bit_base & 15
        unit_index = bit_base >> 4
        if bit_offset + bits > 16:
            code = (units[:, :, :, unit_index] >> bit_offset) | \
                ((units[:, :, :, unit_index + 1] << (16 - bit_offset)) & 0xffff)
        else:
            code = units[:, :, :, unit_index] >> bit_offset
        code = code & code_mask
        unpacked[:, :, value_index, :] = torch.where(code >= (1 << (bits - 1)), code - (1 << bits), code)
    return unpacked
