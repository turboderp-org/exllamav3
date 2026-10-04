// WMMA GEMM launchers, unit 2 of 4 (configs i with i % 4 == 2)
#define EXL3_WMMA_INSTANTIATE
#include "wmma_gemm.cuh"

EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c2, 1, 4, 4, 2, 2, 0, 8, 128, 3)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c6, 1, 8, 8, 2, 1, 0, 8, 128, 1)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c10, 2, 2, 4, 4, 2, 1, 8, 128, 3)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c14, 2, 4, 8, 2, 1, 1, 8, 256, 3)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c18, 4, 2, 4, 3, 2, 1, 8, 256, 3)
