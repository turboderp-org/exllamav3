#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../../util.h"
#include "../../util.cuh"
#include "../../ptx.cuh"
#include "../exl3_gemm_kernel.cuh"

// 3.5 bpw, mul1 codebook
EXL3_KERNEL_INSTANCES_H(3)
