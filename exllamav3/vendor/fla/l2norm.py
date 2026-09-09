# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors


import torch
import torch.nn as nn
import triton
import triton.language as tl
from .utils import IS_AMD
from .utils import autotune_cache_kwargs

BT_LIST = [8, 16, 32, 64, 128]
NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16] if IS_AMD else [1, 2, 4, 8, 16, 32]


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    rstd,
    eps,
    D,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D

    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)
    tl.store(rstd + i_t, b_rstd)


@triton.autotune(
    configs=[triton.Config({"BT": BT}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16] for BT in BT_LIST],
    key=["D", "NB"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def l2norm_fwd_kernel(
    x,
    y,
    rstd,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_x = m_t[:, None] & (o_d[None, :] < D)
    p_x = x + o_t[:, None] * D + o_d[None, :]
    p_y = y + o_t[:, None] * D + o_d[None, :]
    p_rstd = rstd + o_t

    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]

    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_x)
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), mask=m_t)


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    if D <= 512:
        # NOTE(tylerr): Avoid excessive recompilation and autotuning by tolerating a larger range
        # of T before recompiling the kernel.
        # NB = triton.cdiv(T, 2048)
        NB = triton.cdiv(T, 2048 * 32)

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_fwd_kernel[grid](
            x=x,
            y=y,
            rstd=rstd,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            NB=NB,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x=x,
            y=y,
            rstd=rstd,
            eps=eps,
            D=D,
            BD=BD,
        )
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def l2norm(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)


l2_norm = l2norm
