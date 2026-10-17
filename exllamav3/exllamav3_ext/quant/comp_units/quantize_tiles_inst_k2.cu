#include "quantize_tiles_instances.cuh"
#include "../quantize_tiles_optimized.cuh"

fp_quantize_tiles_kernel quantize_tiles_kernel_k2_cb0(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<2, 0> : quantize_tiles_kernel<2, 0>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k2_cb1(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<2, 1> : quantize_tiles_kernel<2, 1>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k2_cb2(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<2, 2> : quantize_tiles_kernel<2, 2>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k2_cb2_l160(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<2, 2, 160> : quantize_tiles_kernel<2, 2, 160>; }
