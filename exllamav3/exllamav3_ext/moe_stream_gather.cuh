#pragma once

#include <ATen/Tensor.h>

void moe_stream_gather
(
    const at::Tensor& dst,          // int16 CUDA staging buffer, >= sel.numel() * expert_bytes bytes
    const at::Tensor& chunk_base,   // int64 [chunks]: device-visible base address of each pinned arena chunk
    const at::Tensor& blk_chunk,    // int32 [E]: arena chunk holding each expert's block
    const at::Tensor& blk_off,      // int64 [E]: byte offset of the block within its chunk
    const at::Tensor& sel,          // int64 [slots]: expert id per staging slot, -1 = leave the slot untouched
    int64_t expert_bytes,
    const c10::optional<at::Tensor>& aux_ptrs,  // int64 [rows, E]: per-expert pointer tables (suh/svh)
    const c10::optional<at::Tensor>& aux_out    // int64 [rows, slots]: receives aux_ptrs[:, sel]
);
