#pragma once

#include <ATen/Tensor.h>
#include <tuple>

std::tuple<bool, int64_t> quantize_tiles_scratch(int device, int K, bool mcg, bool mul1, int L);

void quantize_tiles
(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    int K,
    bool mcg,
    bool mul1
);

void decode
(
    at::Tensor input_indices,
    at::Tensor output_tiles,
    bool mcg,
    bool mul1
);

void test_distribution
(
    at::Tensor& input,
    at::Tensor& dist_output,
    const c10::optional<at::Tensor>& ref_output,
    float min_value,
    float max_value,
    bool mcg,
    bool mul1
);
void quantize_tiles_frac(at::Tensor input_tiles, at::Tensor output_tiles, at::Tensor output_indices, at::Tensor temp_costs, at::Tensor temp_edges, int KA, int64_t MASK);
