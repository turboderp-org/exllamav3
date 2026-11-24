#include "quantize_tiles_instances.cuh"
#include "../quantize_tiles_optimized.cuh"

fp_quantize_tiles_kernel quantize_tiles_kernel_k6_cb0(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<6, 0> : quantize_tiles_kernel<6, 0>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k6_cb1(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<6, 1> : quantize_tiles_kernel<6, 1>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k6_cb2(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<6, 2> : quantize_tiles_kernel<6, 2>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k6_cb2_row(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<6, 2, 0> : quantize_tiles_kernel<6, 2, 0>; }
