#pragma once

#include <algorithm>
#include <cstdint>

#include "../util.h"

// Group mappings are immutable launch metadata, not activation data. Passing a
// validated, fixed-size value into the kernel keeps the mapping graph-safe and
// avoids a device-side trap (which would poison the CUDA context on bad input).
constexpr int EXL3_BF16_MAX_GROUPED_MATRICES = 256;
struct Exl3Bf16HadGroupIds
{
    uint8_t values[EXL3_BF16_MAX_GROUPED_MATRICES];
};

// Validate the device-level requirements shared by the BF16 I/O kernels and
// return the maximum number of blocks that may participate in one cooperative
// launch. Keeping this check next to launch selection makes force_num_sms=0 a
// portable default and turns unsupported devices into a clear host-side error.
inline int exl3_bf16_cooperative_capacity
(
    void* kernel,
    int device,
    int block_dim,
    int smem_bytes
)
{
    int major = 0;
    int cooperative = 0;
    int max_smem = 0;
    cuda_check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    cuda_check(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
    cuda_check(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    TORCH_CHECK(major >= 8, "EXL3 BF16 I/O requires compute capability 8.0 or newer");
    TORCH_CHECK(cooperative, "EXL3 BF16 I/O requires cooperative launch support");
    TORCH_CHECK(
        smem_bytes <= max_smem,
        "EXL3 BF16 I/O requires ", smem_bytes,
        " bytes of dynamic shared memory, but the device supports ", max_smem
    );

    cuda_check(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    int blocks_per_sm = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel, block_dim, smem_bytes));
    TORCH_CHECK(
        blocks_per_sm > 0,
        "EXL3 BF16 I/O kernel cannot be resident with the requested launch resources"
    );

    int num_sms = 0;
    cuda_check(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
    return blocks_per_sm * num_sms;
}

inline int exl3_bf16_select_sms
(
    int force_num_sms,
    int total_sms,
    int cooperative_capacity,
    int resident_groups
)
{
    TORCH_CHECK(resident_groups > 0, "EXL3 BF16 I/O requires at least one resident group");
    TORCH_CHECK(force_num_sms >= 0, "EXL3 BF16 I/O force_num_sms must be nonnegative");
    int num_sms = force_num_sms > 0
        ? force_num_sms
        : std::min(total_sms, cooperative_capacity / resident_groups);
    TORCH_CHECK(
        num_sms > 0 && num_sms <= total_sms,
        "invalid EXL3 BF16 I/O SM count: ", num_sms,
        " for a device with ", total_sms, " SMs"
    );
    TORCH_CHECK(
        num_sms * resident_groups <= cooperative_capacity,
        "EXL3 BF16 I/O cooperative grid exceeds device residency: ",
        num_sms * resident_groups, " blocks requested, ",
        cooperative_capacity, " supported"
    );
    return num_sms;
}
