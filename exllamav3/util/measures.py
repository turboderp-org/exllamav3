import torch
import torch.nn.functional as F

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


def sqnr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    a_flat = a.view(a.shape[0], -1)
    b_flat = b.view(b.shape[0], -1)
    signal_power = torch.sum(b_flat ** 2, dim = 1)
    noise_power = torch.sum((a_flat - b_flat) ** 2, dim = 1) + eps
    return 10.0 * torch.log10(signal_power / noise_power).mean().item() # dB


def cosine_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8):
    a_flat = a.view(a.shape[0], -1)
    b_flat = b.view(b.shape[0], -1)
    cos_sim = F.cosine_similarity(a_flat, b_flat, dim = 1, eps = eps)
    return 1.0 - cos_sim.mean().item()


@triton.jit
def _target_logprob_partial_kernel(
    logits,
    target_ids,
    partial_max,
    partial_sum,
    partial_target,
    row_stride: tl.constexpr,
    vocab_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_block = tl.program_id(1)

    offs = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size
    row_base = pid_row.to(tl.int64) * row_stride
    vocab_offs = offs.to(tl.int64) * vocab_stride
    vals = tl.load(logits + row_base + vocab_offs, mask = mask, other = -float("inf")).to(tl.float32)

    block_max = tl.max(vals, axis = 0)
    block_sum = tl.sum(tl.exp(vals - block_max), axis = 0)

    target = tl.load(target_ids + pid_row)
    target_in_block = (target >= pid_block * BLOCK_SIZE) & (target < (pid_block + 1) * BLOCK_SIZE) & (target < vocab_size)
    target_val = tl.load(logits + row_base + target.to(tl.int64) * vocab_stride, mask = target_in_block, other = -float("inf")).to(tl.float32)

    out_idx = pid_row.to(tl.int64) * num_blocks + pid_block
    tl.store(partial_max + out_idx, block_max)
    tl.store(partial_sum + out_idx, block_sum)
    tl.store(partial_target + out_idx, target_val)


def _target_log_probs_torch(logits: torch.Tensor, target_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = logits[..., :vocab_size].float()
    log_probs = F.log_softmax(logits, dim = -1)
    return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).flatten()


def _flatten_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3 and logits.shape[0] == 1:
        return logits[0]
    elif logits.dim() == 2:
        return logits
    else:
        return logits.flatten(0, -2)


def compute_target_log_probs(logits: torch.Tensor, target_ids: torch.Tensor, vocab_size: int, block_size: int = 1024) -> torch.Tensor:
    """
    Compute log_softmax(logits)[target_ids] without materializing full-vocab FP32 log-probs.
    The Triton path stores only [tokens, ceil(vocab / block_size)] partial reductions.
    """

    if not has_triton or not logits.is_cuda:
        return _target_log_probs_torch(logits, target_ids.to(logits.device), vocab_size)

    logits_2d = _flatten_logits(logits)

    assert logits_2d.shape[-1] >= vocab_size
    target_ids = target_ids.to(device = logits.device).flatten()
    num_rows = logits_2d.shape[0]
    assert target_ids.numel() == num_rows

    num_blocks = triton.cdiv(vocab_size, block_size)
    partial_shape = (num_rows, num_blocks)
    partial_max = torch.empty(partial_shape, dtype = torch.float32, device = logits.device)
    partial_sum = torch.empty(partial_shape, dtype = torch.float32, device = logits.device)
    partial_target = torch.empty(partial_shape, dtype = torch.float32, device = logits.device)

    _target_logprob_partial_kernel[(num_rows, num_blocks)](
        logits_2d,
        target_ids,
        partial_max,
        partial_sum,
        partial_target,
        logits_2d.stride(0),
        logits_2d.stride(1),
        vocab_size,
        num_blocks,
        block_size,
        num_warps = 4,
    )

    row_max = partial_max.max(dim = 1).values
    row_sum = (partial_sum * torch.exp(partial_max - row_max[:, None])).sum(dim = 1)
    target_logits = partial_target.max(dim = 1).values
    return target_logits - row_max - torch.log(row_sum)


@triton.jit
def _kl_div_stats_kernel(
    input_logits,
    target_logits,
    partial_input_max,
    partial_input_sum,
    partial_target_max,
    partial_target_sum,
    input_row_stride: tl.constexpr,
    input_vocab_stride: tl.constexpr,
    target_row_stride: tl.constexpr,
    target_vocab_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_block = tl.program_id(1)

    offs = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size

    input_vals = tl.load(
        input_logits + pid_row.to(tl.int64) * input_row_stride + offs.to(tl.int64) * input_vocab_stride,
        mask = mask,
        other = -float("inf")
    ).to(tl.float32)
    target_vals = tl.load(
        target_logits + pid_row.to(tl.int64) * target_row_stride + offs.to(tl.int64) * target_vocab_stride,
        mask = mask,
        other = -float("inf")
    ).to(tl.float32)

    input_max = tl.max(input_vals, axis = 0)
    target_max = tl.max(target_vals, axis = 0)

    out_idx = pid_row.to(tl.int64) * num_blocks + pid_block
    tl.store(partial_input_max + out_idx, input_max)
    tl.store(partial_input_sum + out_idx, tl.sum(tl.exp(input_vals - input_max), axis = 0))
    tl.store(partial_target_max + out_idx, target_max)
    tl.store(partial_target_sum + out_idx, tl.sum(tl.exp(target_vals - target_max), axis = 0))


@triton.jit
def _kl_div_partial_kernel(
    input_logits,
    target_logits,
    input_log_z,
    target_log_z,
    partial_kl,
    input_row_stride: tl.constexpr,
    input_vocab_stride: tl.constexpr,
    target_row_stride: tl.constexpr,
    target_vocab_stride: tl.constexpr,
    vocab_size: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_block = tl.program_id(1)

    offs = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size

    input_vals = tl.load(
        input_logits + pid_row.to(tl.int64) * input_row_stride + offs.to(tl.int64) * input_vocab_stride,
        mask = mask,
        other = -float("inf")
    ).to(tl.float32)
    target_vals = tl.load(
        target_logits + pid_row.to(tl.int64) * target_row_stride + offs.to(tl.int64) * target_vocab_stride,
        mask = mask,
        other = -float("inf")
    ).to(tl.float32)

    input_lz = tl.load(input_log_z + pid_row)
    target_lz = tl.load(target_log_z + pid_row)
    target_log_probs = target_vals - target_lz
    target_probs = tl.exp(target_log_probs)
    input_log_probs = input_vals - input_lz
    kl_vals = target_probs * (target_log_probs - input_log_probs)
    kl_vals = tl.where(mask, kl_vals, 0.0)

    tl.store(partial_kl + pid_row.to(tl.int64) * num_blocks + pid_block, tl.sum(kl_vals, axis = 0))


def _kl_div_torch(input_logits: torch.Tensor, target_logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    input_logits = _flatten_logits(input_logits[..., :vocab_size]).float()
    target_logits = _flatten_logits(target_logits[..., :vocab_size]).float()
    input_log_probs = F.log_softmax(input_logits, dim = -1)
    target_probs = F.softmax(target_logits, dim = -1)
    return F.kl_div(input_log_probs, target_probs, reduction = "none").sum(dim = -1)


def compute_kl_div(
    input_logits: torch.Tensor,
    target_logits: torch.Tensor,
    vocab_size: int,
    block_size: int = 1024,
) -> torch.Tensor:
    """
    Compute per-row KL(softmax(target_logits) || softmax(input_logits)).
    This matches F.kl_div(log_softmax(input_logits), softmax(target_logits), reduction="none").sum(-1)
    while avoiding full-vocab FP32 probability/log-prob tensors on CUDA.
    """

    if not has_triton or not input_logits.is_cuda or not target_logits.is_cuda:
        return _kl_div_torch(input_logits, target_logits, vocab_size)

    input_logits_2d = _flatten_logits(input_logits)
    target_logits_2d = _flatten_logits(target_logits)
    assert input_logits_2d.shape[0] == target_logits_2d.shape[0]
    assert input_logits_2d.shape[-1] >= vocab_size
    assert target_logits_2d.shape[-1] >= vocab_size

    num_rows = input_logits_2d.shape[0]
    num_blocks = triton.cdiv(vocab_size, block_size)
    partial_shape = (num_rows, num_blocks)
    partial_input_max = torch.empty(partial_shape, dtype = torch.float32, device = input_logits.device)
    partial_input_sum = torch.empty(partial_shape, dtype = torch.float32, device = input_logits.device)
    partial_target_max = torch.empty(partial_shape, dtype = torch.float32, device = input_logits.device)
    partial_target_sum = torch.empty(partial_shape, dtype = torch.float32, device = input_logits.device)

    _kl_div_stats_kernel[(num_rows, num_blocks)](
        input_logits_2d,
        target_logits_2d,
        partial_input_max,
        partial_input_sum,
        partial_target_max,
        partial_target_sum,
        input_logits_2d.stride(0),
        input_logits_2d.stride(1),
        target_logits_2d.stride(0),
        target_logits_2d.stride(1),
        vocab_size,
        num_blocks,
        block_size,
        num_warps = 4,
    )

    input_max = partial_input_max.max(dim = 1).values
    target_max = partial_target_max.max(dim = 1).values
    input_log_z = input_max + torch.log((partial_input_sum * torch.exp(partial_input_max - input_max[:, None])).sum(dim = 1))
    target_log_z = target_max + torch.log((partial_target_sum * torch.exp(partial_target_max - target_max[:, None])).sum(dim = 1))

    partial_kl = torch.empty(partial_shape, dtype = torch.float32, device = input_logits.device)
    _kl_div_partial_kernel[(num_rows, num_blocks)](
        input_logits_2d,
        target_logits_2d,
        input_log_z,
        target_log_z,
        partial_kl,
        input_logits_2d.stride(0),
        input_logits_2d.stride(1),
        target_logits_2d.stride(0),
        target_logits_2d.stride(1),
        vocab_size,
        num_blocks,
        block_size,
        num_warps = 4,
    )

    return partial_kl.sum(dim = 1)
