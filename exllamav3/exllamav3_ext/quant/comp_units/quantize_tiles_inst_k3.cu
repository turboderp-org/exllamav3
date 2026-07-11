#include "quantize_tiles_instances.cuh"
#include "../quantize_tiles_kernel.cuh"

fp_quantize_tiles_kernel quantize_tiles_kernel_k3_cb0() { return quantize_tiles_kernel<3, 0>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k3_cb1() { return quantize_tiles_kernel<3, 1>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k3_cb2() { return quantize_tiles_kernel<3, 2>; }
