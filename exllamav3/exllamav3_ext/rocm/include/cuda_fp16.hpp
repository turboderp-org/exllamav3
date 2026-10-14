// ROCm stand-in for cuda_fp16.hpp (the implementation half of cuda_fp16.h, which hipify does not map). On the
// HIP include path only.

#pragma once
#include <hip/hip_fp16.h>
