import os
import torch
from ...ext import exllamav3_ext as ext
from ...util.tensor import get_for_device, buffered_arange
from ...util.hadamard import get_hadamard_dt

# Packed rows contain signed two's-complement fields at consecutive bit offsets in little-endian byte
# order, with one fp16 absmax scale per 32-value k slice. Packed values represent recurrent state in
# the H32-rotated domain.
_K_SLICE_SIZE = 32
_K_HEAD_DIM = 128


def _hadamard_32(device):
    """Normalized fp32 32x32 Sylvester Hadamard matrix."""
    return get_hadamard_dt(32, device, torch.float, 32 ** -0.5)


def _unpack_bit_codes(units: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack unsigned 16-bit units into signed codes."""
    k_head_dim = units.shape[2] * 16 // bits
    unpacked = torch.zeros(*units.shape[:2], k_head_dim, dtype = torch.int64, device = units.device)
    code_mask = (1 << bits) - 1
    for value_index in range(k_head_dim):
        bit_base = value_index * bits
        bit_offset = bit_base & 15
        unit_index = bit_base >> 4
        if bit_offset + bits > 16:
            code = (
                (units[:, :, unit_index] >> bit_offset)
                | ((units[:, :, unit_index + 1] << (16 - bit_offset)) & 0xffff)
            )
        else:
            code = units[:, :, unit_index] >> bit_offset
        code = code & code_mask
        unpacked[:, :, value_index] = torch.where(code >= (1 << (bits - 1)), code - (1 << bits), code)
    return unpacked


def _pack_bit_row(codes: torch.Tensor, packed_values: torch.Tensor, bits: int):
    """Pack signed codes into a row's little-endian byte stream in place."""
    num_v_heads, k_head_dim, v_head_dim = codes.shape
    num_units = packed_values.shape[2] // 2
    units = torch.zeros(num_v_heads, v_head_dim, num_units, dtype = torch.int32, device = codes.device)
    code_mask = (1 << bits) - 1
    for value_index in range(k_head_dim):
        bit_base = value_index * bits
        code = codes[:, value_index, :].to(torch.int32) & code_mask
        bit_offset = bit_base & 15
        unit_index = bit_base >> 4
        units[:, :, unit_index] = units[:, :, unit_index] | (code << bit_offset)
        if bit_offset + bits > 16:
            units[:, :, unit_index + 1] = units[:, :, unit_index + 1] | (code >> (16 - bit_offset))
    flat_units = units.view(num_v_heads * v_head_dim, num_units)
    low_bytes = (flat_units & 0xff).to(torch.int8)
    high_bytes = ((flat_units >> 8) & 0xff).to(torch.int8)
    packed_values.view(num_v_heads * v_head_dim, num_units, 2).copy_(torch.stack([low_bytes, high_bytes], dim = -1))


def _unpack_quantized_state_row(packed_values: torch.Tensor, scales: torch.Tensor, bits: int) -> torch.Tensor:
    """Dequantize a packed row and rotate out of the H32-rotated domain."""
    num_v_heads, v_head_dim, packed_bytes = packed_values.shape
    k_head_dim = _K_HEAD_DIM
    assert packed_bytes == k_head_dim * bits // 8, "packed state width does not match its bit count"
    byte_pairs = packed_values.view(num_v_heads, v_head_dim, packed_bytes // 2, 2).to(torch.int32)
    units = (byte_pairs[:, :, :, 0] & 0xff) | ((byte_pairs[:, :, :, 1] & 0xff) << 8)
    codes = _unpack_bit_codes(units, bits)
    quantization_bound = (1 << (bits - 1)) - 1
    expanded_scales = scales.float().permute(0, 2, 1).unsqueeze(2)
    rotated_state = (
        codes.permute(0, 2, 1).reshape(
            num_v_heads, _K_HEAD_DIM // _K_SLICE_SIZE, _K_SLICE_SIZE, v_head_dim
        ) * expanded_scales / quantization_bound
    )
    hadamard = _hadamard_32(packed_values.device)
    return torch.matmul(rotated_state.transpose(2, 3), hadamard).transpose(2, 3).reshape(
        num_v_heads, k_head_dim, v_head_dim
    )


def _pack_quantized_state_row(state: torch.Tensor, packed_values: torch.Tensor, scales: torch.Tensor, bits: int):
    """Rotate an (nv, dk, dv) fp32 state into the H32-rotated domain and write packed values and scales in place."""
    num_v_heads, k_head_dim, v_head_dim = state.shape
    assert k_head_dim == _K_HEAD_DIM and packed_values.shape[2] == k_head_dim * bits // 8, \
        "packed state width does not match its bit count"
    hadamard = _hadamard_32(state.device)
    quantization_bound = (1 << (bits - 1)) - 1
    state_slices = state.reshape(num_v_heads, _K_HEAD_DIM // _K_SLICE_SIZE, _K_SLICE_SIZE, v_head_dim)
    rotated_state = torch.matmul(state_slices.transpose(2, 3), hadamard).transpose(2, 3)
    absmax = rotated_state.abs().amax(dim = 2)
    quantization_scale = torch.where(absmax > 0, quantization_bound / absmax, 0.0)
    scales.copy_(absmax.permute(0, 2, 1).half())
    codes = (rotated_state * quantization_scale.unsqueeze(2)).round().to(torch.int16).reshape(
        num_v_heads, k_head_dim, v_head_dim
    )
    _pack_bit_row(codes, packed_values, bits)


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    """
    For reference, not used
    """

    def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
        inv_norm = 1 / torch.sqrt(
            (x * x).sum(dim = dim, keepdim = True)
            + eps
        )
        return x * inv_norm

    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    batch_size, sequence_length, num_heads, k_head_dim = key.shape

    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query

    core_attn_out = torch.zeros(batch_size, sequence_length, num_heads, v_head_dim).to(value)

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    query = query.float()
    key = key.float()
    value = value.float()
    beta = beta.float()
    g = g.float()

    for i in range(sequence_length):
        q_t = query[:, i, :]
        k_t = key[:, i, :]
        v_t = value[:, i, :]
        g_t = g[:, i, :].exp().unsqueeze(-1)
        beta_t = beta[:, i, :].unsqueeze(-1)
        kv_mem = last_recurrent_state * k_t.unsqueeze(-1)
        kv_mem = kv_mem.sum(dim = -2)
        v_t = v_t - kv_mem * g_t
        upd = k_t.unsqueeze(-1) * v_t.unsqueeze(-2) * beta_t.unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t.unsqueeze(-1) + upd
        core_attn_out[:, i, :] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2) * scale

    if not output_final_state:
        last_recurrent_state = None
    return core_attn_out, last_recurrent_state


def torch_recurrent_kda(q, k, v, g, beta, state):
    """Sequential per-channel-decay (KDA) delta rule, fp32, mirroring the GLM5.3/Kimi
    reference. q/k/v: (b, s, h, d); g: (b, s, h, d) log-decay per k-channel; beta: (b, s, h);
    state: (b, h, dk, dv), updated in place. Returns (b, s, h, dv) bf16."""
    b, s, h, dk = k.shape
    dv = v.shape[-1]
    q, k, v, g, beta = [t.float() for t in (q, k, v, g, beta)]

    def l2norm(x):
        return x * torch.rsqrt((x * x).sum(dim = -1, keepdim = True) + 1e-6)
    q = l2norm(q) * (dk ** -0.5)
    k = l2norm(k)

    out = torch.empty((b, s, h, dv), dtype = torch.float, device = v.device)
    for i in range(s):
        g_i = g[:, i].exp().unsqueeze(-1)                       # (b, h, dk, 1)
        state *= g_i
        kv_mem = (state * k[:, i].unsqueeze(-1)).sum(dim = -2)  # (b, h, dv)
        delta = (v[:, i] - kv_mem) * beta[:, i].unsqueeze(-1)
        state += k[:, i].unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, i] = (state * q[:, i].unsqueeze(-1)).sum(dim = -2)
    return out.to(torch.bfloat16)


# Prefill sub-range for the fla chunk kernels. The chunked scan is run over consecutive
# sub-ranges of the sequence with the fp32 state carried between them, which is bit-identical
# to one pass over the whole range: the kernels' T-shaped temporaries (w, u, A, the per-64-token
# chunk states h, v_new, o: ~112 KB/token at 48 v-heads x 128) then scale with the sub-range
# instead of the chunk. 2048 is also the fastest point (-18% vs one 16k pass, -12% vs one 8k
# pass on a 5090: better program counts than a long pass, and 1024 already costs +38%)
_scan_sub_range = int(os.environ.get("EXL3_GDN_SUB_CHUNK", "2048"))


def _chunked_scan(fn, q, k, v, g, beta, state, save_state):
    """
    fn (chunk_gated_delta_rule / chunk_kda) over one batch row in sub-ranges of the sequence.
    q, k: (1, T, H, K); v: (1, T, HV, V); state: (1, HV, K, V) fp32 or None. Returns the
    concatenated output (1, T, HV, V) in fn's output dtype and the final state (None unless
    save_state)
    """
    T = q.shape[1]
    sub = _scan_sub_range if _scan_sub_range > 0 else T
    if T <= sub:
        return fn(q, k, v, g = g, beta = beta, initial_state = state, output_final_state = save_state,
                  use_qk_l2norm_in_kernel = True)
    out = None
    for t0 in range(0, T, sub):
        t1 = min(t0 + sub, T)
        o, state = fn(q[:, t0:t1], k[:, t0:t1], v[:, t0:t1], g = g[:, t0:t1], beta = beta[:, t0:t1],
                      initial_state = state, output_final_state = True, use_qk_l2norm_in_kernel = True)
        if out is None:
            out = torch.empty((1, T) + o.shape[2:], dtype = o.dtype, device = o.device)
        out[:, t0:t1] = o
    return out, state if save_state else None


# Vendored FLA imports query devices and Triton, so load chunk kernels on first use.
def gated_delta_rule_fn(
    mixed_qkv: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    recurrent_state: torch.Tensor,
    recurrent_slots: torch.Tensor,
    history: bool,
    save_state: bool,
    num_k_heads: int,
    num_v_heads: int,
    k_dim: int,
    v_dim: int,
    k_head_dim: int,
    v_head_dim: int,
    params: dict = None,
    channelwise_g: bool = False,
    slots_in: torch.Tensor | None = None,
    recurrent_scales: torch.Tensor | None = None,
    recurrent_bits: int = 0,
):
    if params is None:
        params = {}

    bsz, seqlen, _ = mixed_qkv.shape

    # recurrent_state rows: recurrent_slots are the rows the pass writes (and reads, except that
    # a speculative pass -- history -- reads each row's initial state from slots_in and leaves it
    # untouched; see GDNState). The chunk kernels take the initial and final state as separate
    # tensors, so they serve both cases; the CUDA kernel takes slots_in
    def chunked_rows():
        rows_out = get_for_device(params, "recurrent_slots_scan", "cpu", None)
        if rows_out is None:
            rows_out = buffered_arange(bsz, mixed_qkv.device)
        rows_in = get_for_device(params, "recurrent_slots_scan_in", "cpu", None) if history else None
        rows_out = rows_out.tolist()
        rows_in = rows_in.tolist() if rows_in is not None else rows_out
        return rows_out, rows_in

    # KDA (per-k-channel decay, g shaped (b, s, h, dk)): fla chunk kernel for long passes, the
    # channelwise CUDA recurrent kernel (in-kernel q/k l2norm) otherwise
    if channelwise_g:
        if seqlen >= num_v_heads:
            from ...vendor.fla import chunk_kda
            q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
            q = q.view(bsz, seqlen, -1, k_head_dim)
            k = k.view(bsz, seqlen, -1, k_head_dim)
            v = v.view(bsz, seqlen, -1, v_head_dim)

            rows_out, rows_in = chunked_rows()
            core_attn_out = []
            for i, (so, si) in enumerate(zip(rows_out, rows_in)):
                # .float(): the chunk kernels take and return fp32; an fp16 pool is converted on the way in and
                # by the copy_ on the way out
                state = recurrent_state[si, 0].float().unsqueeze(0) if recurrent_state is not None else None
                core_attn, new_state = _chunked_scan(
                    chunk_kda, q[i:i + 1], k[i:i + 1], v[i:i + 1], g[i:i + 1], beta[i:i + 1],
                    state, save_state,
                )
                if save_state and state is not None:
                    recurrent_state[so, 0].copy_(new_state.squeeze(0))
                core_attn_out.append(core_attn)
            return torch.cat(core_attn_out, dim = 0).to(torch.bfloat16)

        core_attn_out = torch.empty(
            (bsz, seqlen, num_v_heads, v_head_dim),
            dtype = torch.bfloat16,
            device = mixed_qkv.device,
        )
        if recurrent_state is None:
            recurrent_state = torch.zeros(
                (bsz, 1, num_v_heads, k_head_dim, v_head_dim),
                dtype = torch.float,
                device = mixed_qkv.device
            )
        ext.cuda_recurrent_gated_delta_rule(
            mixed_qkv.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            recurrent_state,
            core_attn_out,
            num_k_heads,
            num_v_heads,
            k_head_dim,
            v_head_dim,
            recurrent_slots,
            history,
            slots_in,
            recurrent_scales,
            recurrent_bits,
        )
        return core_attn_out

    # Chunked rule
    if seqlen >= num_v_heads:
        from ...vendor.fla import chunk_gated_delta_rule

        q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
        q = q.view(bsz, seqlen, -1, k_head_dim)
        k = k.view(bsz, seqlen, -1, k_head_dim)
        v = v.view(bsz, seqlen, -1, v_head_dim)

        # (Grouped attn supported in fla-core now)
        rows_out, rows_in = chunked_rows()
        core_attn_out = []
        # The chunked scan uses the unrotated state.
        for i, (so, si) in enumerate(zip(rows_out, rows_in)):
            if recurrent_state is None:
                state = None
            elif recurrent_scales is not None:
                state = _unpack_quantized_state_row(
                    recurrent_state[si, 0], recurrent_scales[si, 0], recurrent_bits
                ).unsqueeze(0)
            else:
                state = recurrent_state[si, 0].float().unsqueeze(0)
            core_attn, new_state = _chunked_scan(
                chunk_gated_delta_rule, q[i:i + 1], k[i:i + 1], v[i:i + 1], g[i:i + 1], beta[i:i + 1],
                state, save_state,
            )
            if save_state and state is not None:
                if recurrent_scales is not None:
                    _pack_quantized_state_row(
                        new_state.squeeze(0), recurrent_state[so, 0], recurrent_scales[so, 0], recurrent_bits
                    )
                else:
                    recurrent_state[so, 0].copy_(new_state.squeeze(0))
            core_attn_out.append(core_attn)

        core_attn_out = torch.cat(core_attn_out, dim = 0)

    # Fused recurrent rule
    else:
        core_attn_out = torch.empty(
            (bsz, seqlen, num_v_heads, v_head_dim),
            dtype = torch.bfloat16,
            device = mixed_qkv.device,
        )

        if recurrent_state is None:
            recurrent_state = torch.zeros(
                (bsz, 1, num_v_heads, k_head_dim, v_head_dim),
                dtype = torch.float,
                device = mixed_qkv.device
            )

        ext.cuda_recurrent_gated_delta_rule(
            mixed_qkv,
            g,
            beta,
            recurrent_state,
            core_attn_out,
            num_k_heads,
            num_v_heads,
            k_head_dim,
            v_head_dim,
            recurrent_slots,
            history,
            slots_in,
            recurrent_scales,
            recurrent_bits,
        )

    return core_attn_out
