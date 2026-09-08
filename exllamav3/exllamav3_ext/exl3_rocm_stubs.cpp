// HIP definitions for the two warp-mma GEMV engine launchers, excluded from the
// ROCm build (build_config.py). Both always decline; the split-K streaming inner
// covers those shapes instead.

#include <ATen/Tensor.h>
#include "quant/exl3_gemm.cuh"
#include "quant/exl3_gemv.cuh"
#include "quant/exl3_gemv_int8.cuh"
#include "graph.cuh"

#if defined(USE_ROCM)

// always declines; A-hadamard staging stays the caller's responsibility
bool exl3_gemv_try_launch
(
    void**,
    int, int, int, int, int,
    bool, bool,
    int,
    cudaStream_t,
    void**,
    bool
)
{
    return false;
}

void exl3_gemv
(
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    bool,
    bool
)
{
    TORCH_CHECK(false, "exl3_gemv: direct GEMV entry point is not built on ROCm (the regular exl3_gemm kernel covers these shapes)");
}

bool exl3_gemv_int8_enabled() { return false; }

bool exl3_gemv_int8
(
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    cudaStream_t,
    Graph*
)
{
    return false;
}

int exl3_gemv_int8_max_k(int) { return 0; }

#endif
