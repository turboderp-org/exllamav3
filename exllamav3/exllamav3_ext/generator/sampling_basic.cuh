#pragma once

#include <ATen/Tensor.h>

void argmax_sample
(
    const at::Tensor& logits,
    at::Tensor& ids,
    int max_logit
);

void gumbel_sample
(
    const at::Tensor& logits,
    at::Tensor& ids,
    int max_logit,
    uint32_t random
);