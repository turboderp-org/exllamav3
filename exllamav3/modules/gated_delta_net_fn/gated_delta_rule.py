import torch
from ...ext import exllamav3_ext as ext
from ...util.tensor import get_for_device, buffered_arange

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ModuleNotFoundError:
    chunk_gated_delta_rule = None

# """
# fla wrapper, reduce overhead by bypassing input_guard and torch custom ops stuff
# """
#
# def fused_recurrent_gated_delta_rule(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     g: torch.Tensor,
#     beta: torch.Tensor,
#     initial_state: torch.Tensor = None,
#     output_final_state: bool = False,
#     use_qk_l2norm_in_kernel: bool = False,
# ):
#     from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd
#
#     scale = k.shape[-1] ** -0.5
#     with torch.cuda.device(q.device.index):
#         o, final_state = fused_recurrent_gated_delta_rule_fwd(
#             q,
#             k,
#             v.contiguous(),
#             g,
#             None,
#             None,
#             beta,
#             scale,
#             initial_state.contiguous() if initial_state is not None else None,
#             output_final_state,
#             use_qk_l2norm_in_kernel,
#             None,
#         )
#     return o, final_state


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
):
    if params is None:
        params = {}

    bsz, seqlen, _ = mixed_qkv.shape

    # Chunked rule
    if seqlen >= num_v_heads and chunk_gated_delta_rule is not None:
        # History only used for short (spec decode) sequences
        assert not history

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
            core_attn, new_state = chunk_gated_delta_rule(
                q[i:i+1], k[i:i+1], v[i:i+1],
                g = g,
                beta = beta,
                initial_state = state,
                output_final_state = save_state,
                use_qk_l2norm_in_kernel = True,
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
