#include "quantize_tiles_instances.cuh"
#include "../quantize_tiles_optimized.cuh"

fp_quantize_tiles_kernel quantize_tiles_kernel_k7_cb0(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<7, 0> : quantize_tiles_kernel<7, 0>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k7_cb1(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<7, 1> : quantize_tiles_kernel<7, 1>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k7_cb2(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<7, 2> : quantize_tiles_kernel<7, 2>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k7_cb2_l160(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<7, 2, 160> : quantize_tiles_kernel<7, 2, 160>; }
