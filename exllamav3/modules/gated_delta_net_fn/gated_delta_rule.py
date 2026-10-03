import os
import torch
from ...ext import exllamav3_ext as ext
from ...util.tensor import get_for_device, buffered_arange

# The vendored fla chunk kernels (exllamav3.vendor.fla) query the devices and Triton at import, so
# they are imported on first use rather than with the library

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
):
    if params is None:
        params = {}

    bsz, seqlen, _ = mixed_qkv.shape

    # KDA (per-k-channel decay, g shaped (b, s, h, dk)): fla chunk kernel for prefill, the
    # channelwise CUDA recurrent kernel (in-kernel q/k l2norm, history-capable) otherwise
    if channelwise_g:
        if seqlen >= num_v_heads and not history:
            from ...vendor.fla import chunk_kda
            q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
            q = q.view(bsz, seqlen, -1, k_head_dim)
            k = k.view(bsz, seqlen, -1, k_head_dim)
            v = v.view(bsz, seqlen, -1, v_head_dim)

            recurrent_slots_cpu = get_for_device(params, "recurrent_slots", "cpu", None)
            if recurrent_slots_cpu is None:
                recurrent_slots_cpu = buffered_arange(bsz, mixed_qkv.device)
            core_attn_out = []
            for i, s in enumerate(recurrent_slots_cpu.tolist()):
                state = recurrent_state[s, 0].unsqueeze(0) if recurrent_state is not None else None
                core_attn, new_state = _chunked_scan(
                    chunk_kda, q[i:i + 1], k[i:i + 1], v[i:i + 1], g[i:i + 1], beta[i:i + 1],
                    state, save_state,
                )
                if save_state and state is not None:
                    state.copy_(new_state)
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
        )
        return core_attn_out

    # Chunked rule
    if seqlen >= num_v_heads and not history:
        from ...vendor.fla import chunk_gated_delta_rule

        q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim = -1)
        q = q.view(bsz, seqlen, -1, k_head_dim)
        k = k.view(bsz, seqlen, -1, k_head_dim)
        v = v.view(bsz, seqlen, -1, v_head_dim)

        # (Grouped attn supported in fla-core now)
        recurrent_slots_cpu = get_for_device(params, "recurrent_slots", "cpu", None)
        if recurrent_slots_cpu is None:
            recurrent_slots_cpu = buffered_arange(bsz, mixed_qkv.device)
        core_attn_out = []
        for i, s in enumerate(recurrent_slots_cpu.tolist()):
            state = recurrent_state[s, 0].unsqueeze(0) if recurrent_state is not None else None
            core_attn, new_state = _chunked_scan(
                chunk_gated_delta_rule, q[i:i + 1], k[i:i + 1], v[i:i + 1], g[i:i + 1], beta[i:i + 1],
                state, save_state,
            )
            if save_state and state is not None:
                state.copy_(new_state)
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
        )

    return core_attn_out
