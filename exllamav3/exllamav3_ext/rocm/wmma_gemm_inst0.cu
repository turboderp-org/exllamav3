// WMMA GEMM launchers, unit 0 of 4 (configs i with i % 4 == 0)
#define EXL3_WMMA_INSTANTIATE
#include "wmma_gemm.cuh"

EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c0, 1, 2, 3, 4, 2, 1, 8, 256, 1)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c4, 1, 8, 3, 4, 2, 1, 8, 256, 1)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c8, 2, 1, 3, 4, 2, 1, 8, 128, 1)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c12, 2, 4, 3, 2, 2, 1, 8, 128, 3)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c16, 2, 8, 3, 4, 2, 1, 8, 128, 1)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c20, 8, 2, 2, 6, 4, 1, 8, 128, 1)
