#include "quantize_tiles_instances.cuh"
#include "../quantize_tiles_optimized.cuh"

fp_quantize_tiles_kernel quantize_tiles_kernel_k3_cb0(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<3, 0> : quantize_tiles_kernel<3, 0>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k3_cb1(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<3, 1> : quantize_tiles_kernel<3, 1>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k3_cb2(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<3, 2> : quantize_tiles_kernel<3, 2>; }
fp_quantize_tiles_kernel quantize_tiles_kernel_k3_cb2_l160(bool optimized) { return optimized ? quantize_tiles_optimized_kernel<3, 2, 160> : quantize_tiles_kernel<3, 2, 160>; }
