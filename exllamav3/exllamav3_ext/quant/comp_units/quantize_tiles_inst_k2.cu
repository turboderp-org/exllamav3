#include "quantize_tiles_instances.cuh"
#include "../quantize_tiles_kernel.cuh"

fp_quantize_tiles_kernel quantize_tiles_kernel_k2_cb0() { return quantize_tiles_kernel<2, 0>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k2_cb1() { return quantize_tiles_kernel<2, 1>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k2_cb2() { return quantize_tiles_kernel<2, 2>; }
