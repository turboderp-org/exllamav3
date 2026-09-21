#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

typedef void (*fp_quantize_tiles_kernel)(const float*, float*, uint16_t*, half*, uint16_t*, const half2*);

fp_quantize_tiles_kernel quantize_tiles_kernel_k1_cb0(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k1_cb1(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k1_cb2(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k2_cb0(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k2_cb1(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k2_cb2(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k3_cb0(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k3_cb1(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k3_cb2(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k4_cb0(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k4_cb1(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k4_cb2(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k5_cb0(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k5_cb1(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k5_cb2(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k6_cb0(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k6_cb1(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k6_cb2(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k7_cb0(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k7_cb1(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k7_cb2(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k8_cb0(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k8_cb1(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k8_cb2(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k1_cb2_l160(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k2_cb2_l160(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k3_cb2_l160(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k4_cb2_l160(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k5_cb2_l160(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k6_cb2_l160(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k7_cb2_l160(bool optimized = false);
fp_quantize_tiles_kernel quantize_tiles_kernel_k8_cb2_l160(bool optimized = false);


// Fractional-rate instances (mul1 codebook only): quantize_tiles_frac_kernel<KA, MASK>, rate KA + popcount(MASK)/16
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a1_maaaa();
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a2_maaaa();
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a3_maaaa();
