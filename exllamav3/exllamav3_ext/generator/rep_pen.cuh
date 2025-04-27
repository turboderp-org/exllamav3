#pragma once

#include <ATen/Tensor.h>

void apply_rep_pens
(
    const at::Tensor& in_logits,
    const at::Tensor& out_logits,
    const at::Tensor& past_ids,
    float rep_p,
    int sustain_range,
    int decay_range
);

void apply_pres_freq_pens
(
    const at::Tensor& in_logits,
    const at::Tensor& out_logits,
    const at::Tensor& past_ids,
    float pres_p,
    float freq_p,
    int sustain_range,
    int decay_range
);
