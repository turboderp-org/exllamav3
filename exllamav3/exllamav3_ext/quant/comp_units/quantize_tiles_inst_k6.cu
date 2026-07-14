#include "quantize_tiles_instances.cuh"
#include "../quantize_tiles_kernel.cuh"

fp_quantize_tiles_kernel quantize_tiles_kernel_k6_cb0() { return quantize_tiles_kernel<6, 0>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k6_cb1() { return quantize_tiles_kernel<6, 1>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k6_cb2() { return quantize_tiles_kernel<6, 2>; }
