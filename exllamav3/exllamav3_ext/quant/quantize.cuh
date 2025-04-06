#pragma once

#include <ATen/Tensor.h>

void quantize_tiles
(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    int K
);

void decode
(
    at::Tensor input_indices,
    at::Tensor output_tiles
);
