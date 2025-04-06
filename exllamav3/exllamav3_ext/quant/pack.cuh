#pragma once

#include <ATen/Tensor.h>

void pack_trellis
(
    at::Tensor packed,
    at::Tensor unpacked,
    int K
);

void unpack_trellis
(
    at::Tensor unpacked,
    at::Tensor packed,
    int K
);

void pack_signs
(
    at::Tensor packed,
    at::Tensor unpacked
);
