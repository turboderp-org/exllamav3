#pragma once

// Launch of the grid-synchronizing EXL3 GEMM kernels. ROCm's cooperative launch faults in the runtime when these
// are submitted back to back, so ROCm launches them as plain kernels and the kernels synchronize through a device
// barrier instead (EXL3_GRID_SYNC, exl3_gemm_kernel.cuh). Every candidate grid is at most one block per
// multiprocessor, so all blocks are resident either way
#if defined(USE_ROCM)
    #define EXL3_COOP_LAUNCH cudaLaunchKernel
#else
    #define EXL3_COOP_LAUNCH cudaLaunchCooperativeKernel
#endif

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

struct CoopAutotuneCandidate
{
    void* kernel;
    int block_dim;
    int max_num_sms;
    int max_concurrency;
    int total_sms;
    int tag;
};

struct CoopAutotuneLaunch
{
    void* kernel;
    int block_dim;
    int num_sms;
    int concurrency;
    int tag;
};

class CoopKernelAutotuner
{
public:
    static bool launch_locked
    (
        uint64_t hash,
        void** kernel_args,
        size_t smem,
        cudaStream_t stream,
        CoopAutotuneLaunch* launch_config = nullptr
    );

    static CoopAutotuneLaunch launch
    (
        uint64_t hash,
        const std::vector<CoopAutotuneCandidate>& candidates,
        void** kernel_args,
        size_t smem,
        cudaStream_t stream,
        size_t numel_B = 1e9
    );
};
