#pragma once

#include <ATen/Tensor.h>

// Flash-attention forward on sm_75 HMMA for head_dim 256 (prefill): q [Tq, Hq, 256], k/v [Tkv, Hkv, 256]
// (16-byte aligned row/head strides, contiguous last dim), o [Tq, Hq, 256] contiguous, fp16. causal is
// bottom-right aligned (query i sees keys j <= i + Tkv - Tq); GQA by Hq / Hkv
void fa75_fwd
(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor o,
    double scale,
    bool causal
);

// One slice of the bounded staged prefill window. q [Tq, Hq, HD] fp16, k/v [Tkv, Hkv, HD] fp16 are the
// scratch slice of one sequence; the first key is absolute (frame_len - offs) in a frame of frame_len keys
// ending at this q chunk (causal: key j is visible to row i iff j <= i + offs, offs may be negative). Writes
// the unnormalized (oacc, m, l) into the combine's slot (m_tile * n_bh + bh), resuming that slot when
// resume != 0: one 64 * 256-float slot of fp32 partial_o (64 * 2 of partial_ml, m at 2 * row, l at 2 * row + 1)
// per (query block, bh), so the buffers hold q_blocks * n_bh slots with q_blocks = ceil(Tq / 64);
// bh = bh0 + q head, bh0 = batch * Hq, HD = 256, BM = 64
void fa75_fwd_slice
(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor partial_o,
    at::Tensor partial_ml,
    int64_t n_bh,
    int64_t bh0,
    double scale,
    int64_t offs,
    bool resume
);
