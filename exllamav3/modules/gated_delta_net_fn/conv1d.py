import torch
import torch.nn.functional as F
from ...util.tensor import get_for_device, buffered_arange
from ...ext import exllamav3_ext as ext

# Above this length the triton kernel splits into separate output/state kernels and its launch
# overhead is amortized anyway; the CUDA kernel keeps the conv window in registers per thread
# so it only makes sense for short sequences (decode and SD verification steps)
MAX_CUDA_SEQLEN = 32
MAX_CUDA_K = 16

try:
    import triton
    import triton.language as tl
    has_triton = True
except ImportError:
    has_triton = False

    class _DummyTritonLanguage:
        constexpr = object()

    class _DummyTriton:
        @staticmethod
        def jit(fn):
            return fn

    triton = _DummyTriton()
    tl = _DummyTritonLanguage()


@triton.jit
def _causal_conv1d_update_slotted_kernel(
    x,
    conv_state,
    slots,
    weight,
    bias,
    out,
    dim: tl.constexpr,
    seq_len,             # runtime: chunk length varies per job; only masks and address math
    state_size: tl.constexpr,
    conv_kernel_size: tl.constexpr,
    history: tl.constexpr,
    has_bias: tl.constexpr,
    transpose_output: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_s = tl.arange(0, BLOCK_S)
    offs_k = tl.arange(0, BLOCK_K)
    offs_state = tl.arange(0, BLOCK_STATE)

    slot = tl.load(slots + pid_b)
    mask_d = offs_d < dim
    mask_s = offs_s < seq_len
    mask_state = offs_state < state_size

    acc = tl.zeros((BLOCK_D, BLOCK_S), dtype = tl.float32)

    for k in range(BLOCK_K):
        if k < conv_kernel_size:
            src_t = offs_s + k + 1
            from_x = src_t >= conv_kernel_size
            x_t = src_t - conv_kernel_size

            state_vals = tl.load(
                conv_state + (slot * dim + offs_d[:, None]) * state_size + src_t[None, :],
                mask = mask_d[:, None] & mask_s[None, :] & (src_t[None, :] < conv_kernel_size),
                other = 0.0,
            )
            x_vals = tl.load(
                x + (pid_b * dim + offs_d[:, None]) * seq_len + x_t[None, :],
                mask = mask_d[:, None] & mask_s[None, :] & from_x[None, :] & (x_t[None, :] >= 0),
                other = 0.0,
            )
            vals = tl.where(from_x[None, :], x_vals, state_vals)
            w = tl.load(weight + offs_d * conv_kernel_size + k, mask = mask_d, other = 0.0)
            acc += vals * w[:, None]

    if has_bias:
        b = tl.load(bias + offs_d, mask = mask_d, other = 0.0)
        acc += b[:, None]

    acc = acc * tl.sigmoid(acc)
    if transpose_output:
        tl.store(
            out + (pid_b * seq_len + offs_s[None, :]) * dim + offs_d[:, None],
            acc,
            mask = mask_d[:, None] & mask_s[None, :],
        )
    else:
        tl.store(
            out + (pid_b * dim + offs_d[:, None]) * seq_len + offs_s[None, :],
            acc,
            mask = mask_d[:, None] & mask_s[None, :],
        )

    history_write_size = tl.where(state_size < conv_kernel_size + seq_len, state_size, conv_kernel_size + seq_len)
    state_write_size = tl.where(history, history_write_size, conv_kernel_size)
    dst_start = tl.where(history, state_size - state_write_size, 0)
    history_src_start = tl.where(conv_kernel_size + seq_len > state_size, conv_kernel_size + seq_len - state_size, 0)
    no_history_start = seq_len
    src_t = tl.where(history, history_src_start + offs_state - dst_start, no_history_start + offs_state)
    valid_state = (offs_state >= dst_start) & (offs_state < dst_start + state_write_size)
    from_x = src_t >= conv_kernel_size
    x_t = src_t - conv_kernel_size
    state_vals = tl.load(
        conv_state + (slot * dim + offs_d[:, None]) * state_size + src_t[None, :],
        mask = mask_d[:, None] & valid_state[None, :] & (src_t[None, :] >= 0) & (src_t[None, :] < conv_kernel_size),
        other = 0.0,
    )
    x_vals = tl.load(
        x + (pid_b * dim + offs_d[:, None]) * seq_len + x_t[None, :],
        mask = mask_d[:, None] & valid_state[None, :] & from_x[None, :] & (x_t[None, :] >= 0),
        other = 0.0,
    )
    new_state = tl.where(from_x[None, :], x_vals, state_vals)
    tl.store(
        conv_state + (slot * dim + offs_d[:, None]) * state_size + offs_state[None, :],
        new_state,
        mask = mask_d[:, None] & mask_state[None, :] & valid_state[None, :],
    )


@triton.jit
def _causal_conv1d_update_slotted_output_kernel(
    x,
    conv_state,
    slots,
    weight,
    bias,
    out,
    dim: tl.constexpr,
    seq_len,             # runtime: chunk length varies per job; only masks and address math
    state_size: tl.constexpr,
    conv_kernel_size: tl.constexpr,
    has_bias: tl.constexpr,
    transpose_output: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    slot = tl.load(slots + pid_b)
    mask_d = offs_d < dim
    mask_s = offs_s < seq_len

    acc = tl.zeros((BLOCK_D, BLOCK_S), dtype = tl.float32)

    for k in range(BLOCK_K):
        if k < conv_kernel_size:
            src_t = offs_s + k + 1
            from_x = src_t >= conv_kernel_size
            x_t = src_t - conv_kernel_size

            state_vals = tl.load(
                conv_state + (slot * dim + offs_d[:, None]) * state_size + src_t[None, :],
                mask = mask_d[:, None] & mask_s[None, :] & (src_t[None, :] < conv_kernel_size),
                other = 0.0,
            )
            x_vals = tl.load(
                x + (pid_b * dim + offs_d[:, None]) * seq_len + x_t[None, :],
                mask = mask_d[:, None] & mask_s[None, :] & from_x[None, :] & (x_t[None, :] >= 0),
                other = 0.0,
            )
            vals = tl.where(from_x[None, :], x_vals, state_vals)
            w = tl.load(weight + offs_d * conv_kernel_size + k, mask = mask_d, other = 0.0)
            acc += vals * w[:, None]

    if has_bias:
        b = tl.load(bias + offs_d, mask = mask_d, other = 0.0)
        acc += b[:, None]

    acc = acc * tl.sigmoid(acc)
    if transpose_output:
        tl.store(
            out + (pid_b * seq_len + offs_s[None, :]) * dim + offs_d[:, None],
            acc,
            mask = mask_d[:, None] & mask_s[None, :],
        )
    else:
        tl.store(
            out + (pid_b * dim + offs_d[:, None]) * seq_len + offs_s[None, :],
            acc,
            mask = mask_d[:, None] & mask_s[None, :],
        )


@triton.jit
def _causal_conv1d_update_slotted_state_kernel(
    x,
    conv_state,
    slots,
    dim: tl.constexpr,
    seq_len,             # runtime: chunk length varies per job; only masks and address math
    state_size: tl.constexpr,
    conv_kernel_size: tl.constexpr,
    history: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_state = tl.arange(0, BLOCK_STATE)

    slot = tl.load(slots + pid_b)
    mask_d = offs_d < dim
    mask_state = offs_state < state_size

    history_write_size = tl.where(state_size < conv_kernel_size + seq_len, state_size, conv_kernel_size + seq_len)
    state_write_size = tl.where(history, history_write_size, conv_kernel_size)
    dst_start = tl.where(history, state_size - state_write_size, 0)
    history_src_start = tl.where(conv_kernel_size + seq_len > state_size, conv_kernel_size + seq_len - state_size, 0)
    no_history_start = seq_len
    src_t = tl.where(history, history_src_start + offs_state - dst_start, no_history_start + offs_state)
    valid_state = (offs_state >= dst_start) & (offs_state < dst_start + state_write_size)
    from_x = src_t >= conv_kernel_size
    x_t = src_t - conv_kernel_size
    state_vals = tl.load(
        conv_state + (slot * dim + offs_d[:, None]) * state_size + src_t[None, :],
        mask = mask_d[:, None] & valid_state[None, :] & (src_t[None, :] >= 0) & (src_t[None, :] < conv_kernel_size),
        other = 0.0,
    )
    x_vals = tl.load(
        x + (pid_b * dim + offs_d[:, None]) * seq_len + x_t[None, :],
        mask = mask_d[:, None] & valid_state[None, :] & from_x[None, :] & (x_t[None, :] >= 0),
        other = 0.0,
    )
    new_state = tl.where(from_x[None, :], x_vals, state_vals)
    tl.store(
        conv_state + (slot * dim + offs_d[:, None]) * state_size + offs_state[None, :],
        new_state,
        mask = mask_d[:, None] & mask_state[None, :] & valid_state[None, :],
    )


def causal_conv1d_update_slotted_triton(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    slots: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    transpose_output: bool = False,
    history: bool = False,
) -> torch.Tensor:
    if not x.is_cuda:
        raise RuntimeError("causal_conv1d_update_slotted_triton requires CUDA tensors")
    if not x.is_contiguous() or not conv_state.is_contiguous() or not weight.is_contiguous():
        raise RuntimeError("causal_conv1d_update_slotted_triton requires contiguous x, conv_state, and weight")
    if conv_state.device != x.device:
        raise RuntimeError(f"conv_state is on {conv_state.device}, expected {x.device}")
    if slots.device != x.device:
        raise RuntimeError(f"slots is on {slots.device}, expected {x.device}")
    if weight.device != x.device:
        raise RuntimeError(f"weight is on {weight.device}, expected {x.device}")
    if bias is not None and bias.device != x.device:
        raise RuntimeError(f"bias is on {bias.device}, expected {x.device}")

    bsz, dim, seq_len = x.shape
    state_size = conv_state.shape[-1]
    conv_kernel_size = weight.shape[-1]
    if slots.shape != (bsz,):
        raise ValueError("slots must be [bsz]")
    if conv_state.dim() != 3 or conv_state.size(1) != dim:
        raise ValueError("conv_state must be [num_slots, dim, state_size]")
    if weight.shape[0] != dim:
        raise ValueError("weight must be [dim, conv_kernel_size]")
    if bias is not None and bias.shape != (dim,):
        raise ValueError("bias must be [dim]")
    if conv_kernel_size > 16:
        raise ValueError("causal_conv1d_update_slotted_triton supports conv_kernel_size <= 16")
    if state_size < conv_kernel_size:
        raise ValueError("conv_state must have at least conv_kernel_size entries")
    out_shape = (bsz, seq_len, dim) if transpose_output else (bsz, dim, seq_len)
    out = torch.empty(out_shape, dtype = x.dtype, device = x.device)
    block_d = 32
    block_k = triton.next_power_of_2(conv_kernel_size)
    block_state = triton.next_power_of_2(state_size)

    with torch.cuda.device(x.device):
        if seq_len <= 256:
            block_s = triton.next_power_of_2(seq_len)
            grid = (bsz, triton.cdiv(dim, block_d))
            _causal_conv1d_update_slotted_kernel[grid](
                x,
                conv_state,
                slots,
                weight,
                bias if bias is not None else weight,
                out,
                dim,
                seq_len,
                state_size,
                conv_kernel_size,
                history,
                bias is not None,
                transpose_output,
                BLOCK_D = block_d,
                BLOCK_S = block_s,
                BLOCK_K = block_k,
                BLOCK_STATE = block_state,
                num_warps = 4,
            )
        else:
            block_s = 256
            output_grid = (bsz, triton.cdiv(dim, block_d), triton.cdiv(seq_len, block_s))
            _causal_conv1d_update_slotted_output_kernel[output_grid](
                x,
                conv_state,
                slots,
                weight,
                bias if bias is not None else weight,
                out,
                dim,
                seq_len,
                state_size,
                conv_kernel_size,
                bias is not None,
                transpose_output,
                BLOCK_D = block_d,
                BLOCK_S = block_s,
                BLOCK_K = block_k,
                num_warps = 4,
            )
            state_grid = (bsz, triton.cdiv(dim, block_d))
            _causal_conv1d_update_slotted_state_kernel[state_grid](
                x,
                conv_state,
                slots,
                dim,
                seq_len,
                state_size,
                conv_kernel_size,
                history,
                BLOCK_D = block_d,
                BLOCK_STATE = block_state,
                num_warps = 4,
            )
    return out


def causal_conv1d_update_function_torch(
    x,
    conv_state,
    weight,
    bias = None,
    history: bool = False,
):
    bsz, dim, seq_len = x.shape
    state_size = conv_state.shape[-1]
    conv_kernel_size = weight.shape[-1]

    y = torch.cat([conv_state[:, :, :conv_kernel_size], x], dim = -1).to(weight.dtype)
    if history:
        write_size = min(state_size, y.shape[-1])
        conv_state[:, :, -write_size:].copy_(y[:, :, -write_size:])
    else:
        conv_state[:, :, :conv_kernel_size].copy_(y[:, :, -conv_kernel_size:])
    y = F.conv1d(y, weight.unsqueeze(1), bias, padding = 0, groups = dim)
    y = F.silu(y[:, :, -seq_len:])
    y = y.to(x.dtype)
    return y


def causal_conv1d_update(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_slots: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor,
    history: bool = False,
    params: dict = None
):
    bsz, dim, seqlen = mixed_qkv.shape

    if params is None:
        params = {}

    if conv_state is None:
        assert not history
        conv_state = torch.zeros((bsz, dim, conv1d_weight.shape[-1]), dtype = torch.bfloat16, device = mixed_qkv.device)
        dummy_slots = True
    else:
        dummy_slots = False

    if (
        mixed_qkv.is_cuda and
        seqlen <= MAX_CUDA_SEQLEN and
        conv1d_weight.shape[-1] <= MAX_CUDA_K and
        mixed_qkv.dtype == torch.bfloat16 and
        conv_state.dtype == torch.bfloat16 and
        conv1d_weight.dtype == torch.bfloat16 and
        (conv1d_bias is None or conv1d_bias.dtype == torch.bfloat16)
    ):
        out = torch.empty((bsz, seqlen, dim), dtype = torch.bfloat16, device = mixed_qkv.device)
        ext.cuda_causal_conv1d_update(
            mixed_qkv,
            conv_state,
            None if dummy_slots else recurrent_slots,
            conv1d_weight,
            conv1d_bias,
            out,
            True,
            history,
        )
        return out

    if has_triton:
        if dummy_slots:
            recurrent_slots = buffered_arange(bsz, mixed_qkv.device)
        mixed_qkv = causal_conv1d_update_slotted_triton(
            mixed_qkv,
            conv_state,
            recurrent_slots,
            conv1d_weight,
            conv1d_bias,
            transpose_output = True,
            history = history,
        )
    else:
        recurrent_slots_cpu = get_for_device(params, "recurrent_slots", "cpu") \
            if not dummy_slots else buffered_arange(bsz, "cpu")
        mixed_qkv_conv = []
        for i, s in enumerate(recurrent_slots_cpu.tolist()):
            mixed_qkv_conv.append(
                causal_conv1d_update_function_torch(
                    mixed_qkv[i].unsqueeze(0),
                    conv_state[s].unsqueeze(0),  # Updated inplace
                    conv1d_weight,
                    conv1d_bias,
                    history = history,
                )
            )
        mixed_qkv = torch.cat(mixed_qkv_conv, dim = 0)
        mixed_qkv = mixed_qkv.transpose(1, 2).contiguous()

    return mixed_qkv
