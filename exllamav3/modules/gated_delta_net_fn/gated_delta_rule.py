import os
import torch
from ...ext import exllamav3_ext as ext
from ...util.tensor import get_for_device, buffered_arange
from ...util.hadamard import get_hadamard_dt

# The vendored fla chunk kernels (exllamav3.vendor.fla) are imported on first use rather than with
# the library: they query the devices and Triton at import

# Rotation block of the int pool: 32-wide k slices, each stored rotated through the Sylvester H32.
# gdn.cu reads and writes the row in that basis; the chunked path converts it in and out
_ROT_BLOCK = 32


def _gdn_h32(device):
    """The normalized fp32 Sylvester H32 the kernel's butterfly emits"""
    return get_hadamard_dt(32, device, torch.float, 32 ** -0.5)


def _vcol_units(row: torch.Tensor) -> torch.Tensor:
    """A (nv, dv, 16W) packed pool row -> (nv, dv, 8W) little-endian uint16 units (as int32), the
    slots the kernel assembles (sp[2i] | sp[2i+1] << 8)"""
    r = row.view(row.shape[0], row.shape[1], row.shape[2] // 2, 2).to(torch.int32)
    return (r[:, :, :, 0] & 0xff) | ((r[:, :, :, 1] & 0xff) << 8)


def _unpack_bit_codes(u: torch.Tensor, bits: int) -> torch.Tensor:
    """(nv, dv, 8W) little-endian uint16 units of one v column -> the 128 signed W-bit values. A
    value sits at bit e * W of the column's 128W-bit stream: unit p = (e * W) >> 4 holds its low
    bits at bit s16 = (e * W) & 15, the next unit the high bits when the value leaves p. The units
    stay unsigned (a 64-bit shift would sign-extend a negative window)"""
    b = bits
    out = torch.zeros(*u.shape[:2], 128, dtype = torch.int64, device = u.device)
    mask = (1 << b) - 1
    for e in range(128):
        base = e * b
        s16 = base & 15
        p = base >> 4
        if s16 + b > 16:
            code = (u[:, :, p] >> s16) | ((u[:, :, p + 1] << (16 - s16)) & 0xffff)
        else:
            code = u[:, :, p] >> s16
        code = code & mask
        out[:, :, e] = torch.where(code >= (1 << (b - 1)), code - (1 << b), code)
    return out


def _dequant_bit_row(values: torch.Tensor, scales: torch.Tensor, bits: int) -> torch.Tensor:
    """A (nv, dv, dk * W // 8) packed int pool row + (nv, dv, dk // 32) fp16 scales -> (nv, dk, dv)
    fp32 in the pool's basis: extract each value's W bits of its v column, dequantize against the
    slice's absmax. The CUDA scan reads this exact value (state_ld for int), so this is the kernel's
    load, mirrored on the CPU"""
    nv, dv, dbits = values.shape
    q_bound = (1 << (bits - 1)) - 1
    raw = _unpack_bit_codes(_vcol_units(values), bits)                # (nv, dv, 128)
    sc = scales.float().permute(0, 2, 1).unsqueeze(2)                # (nv, dk//32, 1, dv)
    return (raw.permute(0, 2, 1).reshape(nv, 128 // 32, 32, dv) * sc / q_bound).reshape(nv, 128, dv)


def _pack_bit_row(q: torch.Tensor, values: torch.Tensor, bits: int):
    """(nv, dk, dv) signed quantized values -> a (nv, dv, dk * W // 8) packed int pool row, written
    in place. Value e of the column sits at bit e * W of the column's 128W-bit stream: assemble its
    W-bit two's-complement pattern into the 16-bit units holding those bits (the kernel's store,
    mirrored; the kernel reads the units little-endian as sp[2i] | sp[2i+1] << 8)"""
    nv, dk, dv = q.shape
    b = bits
    units = torch.zeros(nv, dv, 8 * b, dtype = torch.int32, device = q.device)   # 128W bits = 8W 16-bit units
    mask = (1 << b) - 1
    for e in range(128):
        base = e * b
        v = q[:, e, :].to(torch.int32) & mask     # the value's W-bit two's-complement pattern
        s16 = base & 15
        p = base >> 4
        units[:, :, p] = units[:, :, p] | (v << s16)
        if s16 + b > 16:
            units[:, :, p + 1] = units[:, :, p + 1] | (v >> (16 - s16))
    # little-endian byte reassembly, matching the kernel's store (unit i -> sp[2i] | sp[2i+1] << 8)
    flat = units.view(nv * dv, 8 * b)
    lo = (flat & 0xff).to(torch.int8)
    hi = ((flat >> 8) & 0xff).to(torch.int8)
    values.view(nv * dv, 8 * b, 2).copy_(torch.stack([lo, hi], dim = -1))


def unpack_int_state_row(values: torch.Tensor, scales: torch.Tensor, bits: int) -> torch.Tensor:
    """A (nv, dv, dk * W // 8) packed int pool row and its (nv, dv, dk // 32) fp16 scales in the
    pool's basis -> (nv, dk, dv) fp32 in the model's basis: dequantize the pool values, then rotate
    each 32-wide k slice back to the model basis (the chunked path computes in the model basis)"""
    assert values.shape[2] == 128 * bits // 8, "packed pool width does not match its bit count"
    x = _dequant_bit_row(values, scales, bits)
    h32 = _gdn_h32(str(values.device))
    x = x.reshape(values.shape[0], 128 // 32, 32, values.shape[1])
    return torch.matmul(x.transpose(2, 3), h32).transpose(2, 3).reshape(values.shape[0], 128, values.shape[1])


def pack_int_state_row(state: torch.Tensor, values: torch.Tensor, scales: torch.Tensor, bits: int):
    """(nv, dk, dv) fp32 in the model's basis -> a (nv, dv, dk * W // 8) packed int pool row and
    (nv, dv, dk // 32) fp16 scales in the pool's basis, written in place. Rotate each 32-wide k
    slice to the pool basis, quantize each slice against its absmax (one scale word), and assemble
    each value's W bits into the v column's 16-bit units (the mirror of unpack's unit extraction)"""
    assert values.shape[2] == 128 * bits // 8, "packed pool width does not match its bit count"
    nv, dk, dv = state.shape
    b = bits
    h32 = _gdn_h32(str(state.device))
    q_bound = (1 << (b - 1)) - 1
    x = state.reshape(nv, dk // 32, 32, dv)
    x = torch.matmul(x.transpose(2, 3), h32).transpose(2, 3)     # pool basis, (nv, dk, dv)
    amax = x.abs().amax(dim = 2)                                  # (nv, dk//32, dv)
    inv = torch.where(amax > 0, q_bound / amax, 0.0)
    scales.copy_(amax.permute(0, 2, 1).half())
    q = (x * inv.unsqueeze(2)).round().to(torch.int16).reshape(nv, dk, dv)
    _pack_bit_row(q, values, b)


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
    # recurrent_scales: the int pool's fp16 scales (one per 32-wide k slice), None for an fp32 or fp16
    # pool. With scales, the chunked branch unpacks each rotated row into the model's basis and repacks
    # it on the store; the CUDA scan reads and writes the rotated basis itself. recurrent_bits is the
    # packed pool's width W (0 for an fp pool)
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
        for i, (so, si) in enumerate(zip(rows_out, rows_in)):
            if recurrent_state is None:
                state = None
            elif recurrent_scales is not None:
                state = unpack_int_state_row(recurrent_state[si, 0], recurrent_scales[si, 0], recurrent_bits).unsqueeze(0)
            else:
                state = recurrent_state[si, 0].float().unsqueeze(0)
            core_attn, new_state = _chunked_scan(
                chunk_gated_delta_rule, q[i:i + 1], k[i:i + 1], v[i:i + 1], g[i:i + 1], beta[i:i + 1],
                state, save_state,
            )
            if save_state and state is not None:
                if recurrent_scales is not None:
                    pack_int_state_row(new_state.squeeze(0), recurrent_state[so, 0], recurrent_scales[so, 0], recurrent_bits)
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
