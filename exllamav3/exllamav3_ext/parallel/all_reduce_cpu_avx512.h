#pragma once
#include <cstdint>
#include <cstddef>
#include "context.cuh"

// Runtime detection (defined in avx512_target.cpp)
bool is_avx512_supported();

// AVX-512 versions
void enable_fast_fp_avx512();

void bf16_add_inplace_avx512(
    uint16_t* __restrict a,
    const uint16_t* __restrict b,
    size_t count
);

void perform_cpu_reduce_avx512(
    PGContext* ctx,
    size_t data_size,
    uint32_t device_mask,
    uint8_t* shbuf_ptr,
    size_t shbuf_size
);
