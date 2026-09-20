#include "quantize_tiles_instances.cuh"
#include "../quantize_tiles_frac_kernel.cuh"

fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a1_maaaa() { return quantize_tiles_frac_kernel<1, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a2_maaaa() { return quantize_tiles_frac_kernel<2, 0xaaaau>; }
fp_quantize_tiles_kernel quantize_tiles_frac_kernel_a3_maaaa() { return quantize_tiles_frac_kernel<3, 0xaaaau>; }
