// Test-only instantiations of the EXL3 GEMM at every TILEBLOCKS_K the kernel can be built at.
//
// The shipping shape table only reaches TILEBLOCKS_K <= 2, where the threadblock k-reduction runs
// a single store/add pair, so no shipping shape exercises the staging regions the reduction
// interleaves at higher k-tile counts. TILEBLOCKS_K == 1 runs no threadblock reduction at all and
// is the reference the reduction cannot influence.

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "util.h"
#include "util.cuh"
#include "ptx.cuh"
#include "quant/exl3_kernel_map.cuh"
#include "quant/exl3_gemm_kernel.cuh"

// (TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES). TILESIZE_M is fixed at 16 by the kernel.
#define PROBE_SHAPES(X, bits) \
    X(bits, 16, 128, 6, 5) \
    X(bits, 32, 128, 4, 3) \
    X(bits, 64, 128, 4, 3)

#define PROBE_BITS(X) X(2) X(3) X(4) X(5) X(6) X(7) X(8)

static int probe_gemm
(
    at::Tensor A,
    at::Tensor B,
    at::Tensor C,
    at::Tensor suh,
    at::Tensor A_had,
    at::Tensor svh,
    at::Tensor locks,
    int64_t tilesize_k
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int size_m = A.size(0);
    int size_k = A.size(1);
    int size_n = B.size(1) * 16;
    int bits = B.size(2) / 16;

    const half* A_ptr = (const half*) A.data_ptr();
    const uint16_t* B_ptr = (const uint16_t*) B.data_ptr();
    void* C_ptr = (void*) C.data_ptr();
    const half* suh_ptr = (const half*) suh.data_ptr();
    half* A_had_ptr = (half*) A_had.data_ptr();
    const half* svh_ptr = (const half*) svh.data_ptr();
    int* locks_ptr = (int*) locks.data_ptr();

    void* kernelArgs[] =
    {
        (void*) &A_ptr, (void*) &B_ptr, (void*) &C_ptr,
        (void*) &size_m, (void*) &size_k, (void*) &size_n,
        (void*) &locks_ptr, (void*) &suh_ptr, (void*) &A_had_ptr, (void*) &svh_ptr
    };

    fp_exl3_gemm_kernel kernel = nullptr;
    int tilesize_n = 0;

    #define PROBE_CASE(_bits, _tk, _tn, _shs, _frs) \
        if (bits == _bits && tilesize_k == _tk) \
        { \
            kernel = exl3_gemm_kernel<_bits, true, 0, 16, _tk, _tn, _shs, _frs>; \
            tilesize_n = _tn; \
        }
    #define PROBE_BITS_CASE(_bits) PROBE_SHAPES(PROBE_CASE, _bits)
    PROBE_BITS(PROBE_BITS_CASE)
    #undef PROBE_BITS_CASE
    #undef PROBE_CASE

    TORCH_CHECK(kernel, "probe_gemm: no instantiation for bits/tilesize_k");

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int max_slices = MAX(size_k / (int) tilesize_k * size_n / tilesize_n, 1);
    int num_sms = MAX(MIN(max_slices, prop.multiProcessorCount), 1);
    int block_dim = EXL3_GEMM_BASE_THREADS * (int) tilesize_k / 16;

    cudaFuncSetAttribute((void*) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
    cudaLaunchCooperativeKernel((void*) kernel, num_sms, block_dim, kernelArgs, SMEM_MAX, stream);
    C10_CUDA_CHECK(cudaPeekAtLastError());
    return num_sms;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("probe_gemm", &probe_gemm, "EXL3 GEMM at a forced TILESIZE_K");
}
