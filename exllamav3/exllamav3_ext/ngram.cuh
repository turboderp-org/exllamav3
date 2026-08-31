#pragma once

#include <ATen/Tensor.h>

int64_t ngram_hash_cpu
(
    const at::Tensor& ids,
    int64_t seq_len,
    const at::Tensor& multipliers,
    const at::Tensor& offsets,
    const at::Tensor& sizes,
    int64_t heads_per_ngram,
    int64_t eos_token,
    at::Tensor uids,
    at::Tensor inverse,
    at::Tensor heads
);

void ngram_gather_cpu
(
    int64_t fd,
    int64_t base_offset,
    int64_t row_bytes,
    const at::Tensor& uids,
    int64_t uid_base,
    at::Tensor out
);

void ngram_dequant
(
    const at::Tensor& packed,
    int64_t K,
    const at::Tensor& heads,
    const at::Tensor& bias,
    at::Tensor out
);
