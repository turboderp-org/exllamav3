#include "quantize_tiles_instances.cuh"
#include "../quantize_tiles_optimized.cuh"

fp_quantize_tiles_kernel quantize_tiles_kernel_k8_cb0(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<8, 0> : quantize_tiles_kernel<8, 0>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k8_cb1(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<8, 1> : quantize_tiles_kernel<8, 1>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k8_cb2(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<8, 2> : quantize_tiles_kernel<8, 2>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k8_cb2_l160(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<8, 2, 160> : quantize_tiles_kernel<8, 2, 160>; }
