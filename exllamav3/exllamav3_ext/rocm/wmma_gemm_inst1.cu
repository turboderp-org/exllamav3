// WMMA GEMM launchers, unit 1 of 4 (configs i with i % 4 == 1)
#define EXL3_WMMA_INSTANTIATE
#include "wmma_gemm.cuh"

EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c1, 1, 2, 4, 3, 4, 1, 8, 256, 3)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c5, 1, 8, 6, 2, 2, 1, 8, 128, 1)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c9, 2, 1, 4, 4, 1, 1, 8, 256, 1)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c13, 2, 4, 3, 4, 2, 1, 8, 128, 1)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c17, 4, 2, 1, 2, 2, 0, 8, 128, 3)
