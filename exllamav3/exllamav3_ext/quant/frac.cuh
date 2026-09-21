#pragma once
#include <ATen/Tensor.h>

// Fractional-bitrate trellis storage (K = KA + popcount(MASK)/16, half-integer in practice): pack /
// unpack the 16-bit windows to the (kb, nb, 16K)-uint16 tile layout and reconstruct fp16 weights
void pack_trellis_frac(at::Tensor packed, at::Tensor unpacked, int KA, int64_t MASK);
void unpack_trellis_frac(at::Tensor unpacked, at::Tensor packed, int KA, int64_t MASK);
