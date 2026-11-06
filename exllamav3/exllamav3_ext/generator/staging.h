#pragma once

#include <vector>
#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

void generator_stage_decode
(
    at::Tensor arena,
    int64_t width,
    const std::vector<at::Tensor>& seq_block_index,
    const std::vector<int64_t>& kv_position,
    const std::vector<int64_t>& rope_offset,
    const c10::optional<at::Tensor>& batch_ids,
    const std::vector<at::Tensor>& seq_ids,
    const std::vector<at::Tensor>& page_seq,
    const std::vector<int64_t>& page_offset
);
