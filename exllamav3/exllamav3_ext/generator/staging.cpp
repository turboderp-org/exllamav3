#include "staging.h"
#include <cstring>
#include <algorithm>
#include <c10/util/Exception.h>

// Fill one generator step's pinned staging arena (int32, one row per batch sequence):
//
//   [0, B * width)            block index rows, each sequence's pages then zero padding
//   [B * width, + B)          cache_seqlens = kv_position
//   [B * width + B, + B)      positions = kv_position + rope_offset (only when rope_offset is given)
//
// With seq_ids given (single-token decode rows), also gather each row's next input token into
// batch_ids [B, 1] (int64) and record it in the row's current cache page at page_offset. The
// caller keeps the page bookkeeping (page kv_position, revert state); this does only the stores,
// which in Python cost a tensor dispatch each

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
)
{
    const int64_t B = static_cast<int64_t>(seq_block_index.size());
    const bool with_pos = !rope_offset.empty();
    TORCH_CHECK(arena.dtype() == at::kInt && arena.is_contiguous(), "staging arena must be contiguous int32");
    TORCH_CHECK(arena.numel() >= B * width + B * (with_pos ? 2 : 1), "staging arena too small");
    TORCH_CHECK(static_cast<int64_t>(kv_position.size()) == B, "kv_position length mismatch");
    TORCH_CHECK(!with_pos || static_cast<int64_t>(rope_offset.size()) == B, "rope_offset length mismatch");

    int32_t* bi = arena.data_ptr<int32_t>();
    int32_t* cs = bi + B * width;
    int32_t* ps = with_pos ? cs + B : nullptr;
    for (int64_t b = 0; b < B; ++b)
    {
        const at::Tensor& sbi = seq_block_index[b];
        TORCH_CHECK(sbi.dtype() == at::kInt && sbi.is_contiguous(), "block index must be contiguous int32");
        int32_t* row = bi + b * width;
        const int64_t n = std::min<int64_t>(sbi.numel(), width);
        std::memcpy(row, sbi.data_ptr<int32_t>(), static_cast<size_t>(n) * sizeof(int32_t));
        if (n < width) std::memset(row + n, 0, static_cast<size_t>(width - n) * sizeof(int32_t));
        cs[b] = static_cast<int32_t>(kv_position[b]);
        if (ps) ps[b] = static_cast<int32_t>(kv_position[b] + rope_offset[b]);
    }

    if (seq_ids.empty()) return;
    TORCH_CHECK(batch_ids.has_value(), "batch_ids required with seq_ids");
    TORCH_CHECK(static_cast<int64_t>(seq_ids.size()) == B && static_cast<int64_t>(page_seq.size()) == B &&
                static_cast<int64_t>(page_offset.size()) == B, "decode row list length mismatch");
    const at::Tensor& ids = *batch_ids;
    TORCH_CHECK(ids.dtype() == at::kLong && ids.size(0) >= B, "batch_ids must be int64 with a row per sequence");
    int64_t* out = ids.data_ptr<int64_t>();
    const int64_t out_stride = ids.stride(0);
    for (int64_t b = 0; b < B; ++b)
    {
        const at::Tensor& s = seq_ids[b];
        const at::Tensor& p = page_seq[b];
        TORCH_CHECK(s.dtype() == at::kLong && p.dtype() == at::kLong, "token tensors must be int64");
        TORCH_CHECK(kv_position[b] < s.size(-1) && page_offset[b] < p.size(-1), "token index out of range");
        const int64_t tok = s.data_ptr<int64_t>()[kv_position[b] * s.stride(-1)];
        out[b * out_stride] = tok;
        p.data_ptr<int64_t>()[page_offset[b] * p.stride(-1)] = tok;
    }
}
