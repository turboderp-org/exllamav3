// WMMA GEMM launchers, unit 3 of 4 (configs i with i % 4 == 3)
#define EXL3_WMMA_INSTANTIATE
#include "wmma_gemm.cuh"

EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c3, 1, 4, 8, 2, 2, 1, 8, 128, 3)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c7, 2, 1, 3, 2, 1, 1, 8, 128, 1)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c11, 2, 2, 4, 4, 2, 1, 8, 256, 2)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c15, 2, 4, 8, 2, 4, 1, 8, 128, 3)
EXL3_WMMA_LAUNCHER(exl3_wmma_launch_c19, 4, 4, 4, 3, 4, 1, 8, 256, 1)
