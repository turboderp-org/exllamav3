"""
Block-packed symmetric accumulation for the calibration Hessians: H += x^T x stored as the upper triangle in
PACK_BLOCK x PACK_BLOCK blocks (diagonal blocks kept whole), which halves the memory of a captured Hessian and
lets the accumulation skip the lower-triangle tiles. The full matrix is only reconstructed when a tensor is
finalized for quantization (see finalize_capture_H).

Block (bi, bj) with bi <= bj lives at index bi * nb - bi * (bi - 1) / 2 + (bj - bi), row-major over the upper
triangle of the nb x nb block grid.
"""

import torch
import triton
import triton.language as tl

PACK_BLOCK = 64


def num_blocks(n: int) -> int:
    nb = -(-n // PACK_BLOCK)
    return nb * (nb + 1) // 2


def packed_numel(n: int) -> int:
    return num_blocks(n) * PACK_BLOCK * PACK_BLOCK


# Fixed launch geometry: the candidates measured within ~10% of each other, and autotuning an in-place
# accumulation would need its trial runs undone (restore_value) on every new (n, T) key, with T varying
# per call once non-finite rows are dropped
_BK = 32
_FLUSH = 64
_NUM_WARPS = 4
_NUM_STAGES = 3


@triton.jit
def _syrk_packed_kernel(
    x_ptr, out_ptr, T, n, nb, stride_t,
    B: tl.constexpr, BK: tl.constexpr, FLUSH: tl.constexpr, IEEE: tl.constexpr,
):
    # One program per upper-triangular block pair, enumerated row-major over the packed index
    pid = tl.program_id(0)
    bi = 0
    base = 0
    for r in range(0, nb):
        row_len = nb - r
        take = (pid >= base) & (pid < base + row_len)
        bi = tl.where(take, r, bi)
        base = base + row_len
    base_i = bi * nb - (bi * (bi - 1)) // 2
    bj = bi + (pid - base_i)
    offs_i = bi * B + tl.arange(0, B)
    offs_j = bj * B + tl.arange(0, B)
    offs_k = tl.arange(0, BK)
    mask_i = offs_i < n
    mask_j = offs_j < n

    # The MMA accumulator truncates when chained over the whole token range (relative error ~1e-5 with a
    # negative bias). Chaining only FLUSH rows and adding the partial into a separate fp32 accumulator with
    # a rounded add keeps the result at fp32-GEMM accuracy. FLUSH must exceed BK, or the compiler folds the
    # add back into the chain
    acc = tl.zeros((B, B), dtype = tl.float32)
    part = tl.zeros((B, B), dtype = tl.float32)
    for k0 in range(0, T, BK):
        kk = k0 + offs_k
        mask_k = kk < T
        a = tl.load(x_ptr + kk[:, None] * stride_t + offs_i[None, :], mask = mask_k[:, None] & mask_i[None, :], other = 0.0)
        b = tl.load(x_ptr + kk[:, None] * stride_t + offs_j[None, :], mask = mask_k[:, None] & mask_j[None, :], other = 0.0)
        if IEEE:
            part = tl.dot(tl.trans(a), b, part, input_precision = "ieee")
        else:
            part = tl.dot(tl.trans(a), b, part)
        if (k0 // BK) % (FLUSH // BK) == (FLUSH // BK) - 1:
            acc += part
            part = tl.zeros((B, B), dtype = tl.float32)
    acc += part

    oi = tl.arange(0, B)
    out = out_ptr + pid.to(tl.int64) * B * B + oi[:, None] * B + oi[None, :]
    tl.store(out, tl.load(out) + acc)


def accumulate_(packed: torch.Tensor, x: torch.Tensor):
    """packed += x^T x over the rows of x (T, n), fp32 accumulation. Half/bfloat16 inputs run on the tensor
    cores (their products are exact in fp32); fp32 inputs take the IEEE path"""
    assert x.dim() == 2 and packed.dtype == torch.float and packed.is_cuda
    T, n = x.shape
    assert packed.numel() == packed_numel(n), "packed Hessian size"
    if x.dtype not in (torch.half, torch.bfloat16, torch.float):
        x = x.float()
    if x.stride(1) != 1:
        x = x.contiguous()
    if T == 0:
        return
    with torch.cuda.device(packed.device):
        _syrk_packed_kernel[(num_blocks(n),)](
            x, packed, T, n, -(-n // PACK_BLOCK), x.stride(0),
            B = PACK_BLOCK, BK = _BK, FLUSH = _FLUSH, IEEE = x.dtype == torch.float,
            num_warps = _NUM_WARPS, num_stages = _NUM_STAGES,
        )


def unpack(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Full symmetric (n, n) matrix from the block-packed upper triangle, built block row by block row into
    one allocation (the matrices this serves can run to gigabytes)"""
    B = PACK_BLOCK
    nb = -(-n // B)
    assert packed.numel() == packed_numel(n), "packed Hessian size"
    blocks = packed.view(nb * (nb + 1) // 2, B, B)
    full = torch.empty((nb * B, nb * B), dtype = packed.dtype, device = packed.device)
    fv = full.view(nb, B, nb, B)
    start = 0
    for bi in range(nb):
        row = blocks[start : start + nb - bi]                        # blocks (bi, bi..nb-1)
        fv[bi, :, bi:, :] = row.permute(1, 0, 2)
        if bi + 1 < nb:
            fv[bi + 1:, :, bi, :] = row[1:].transpose(1, 2)           # mirrored into the lower triangle
        start += nb - bi
    return full if nb * B == n else full[:n, :n].contiguous()


def pack(full: torch.Tensor) -> torch.Tensor:
    """Block-packed upper triangle of a symmetric (n, n) matrix (the inverse of unpack)"""
    B = PACK_BLOCK
    n = full.shape[0]
    nb = -(-n // B)
    if nb * B != n:
        padded = torch.zeros((nb * B, nb * B), dtype = full.dtype, device = full.device)
        padded[:n, :n] = full
        full = padded
    fv = full.view(nb, B, nb, B)
    packed = torch.empty(packed_numel(n), dtype = full.dtype, device = full.device)
    blocks = packed.view(-1, B, B)
    start = 0
    for bi in range(nb):
        blocks[start : start + nb - bi] = fv[bi, :, bi:, :].permute(1, 0, 2)
        start += nb - bi
    return packed
