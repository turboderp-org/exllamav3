#pragma once

#include <c10/util/Exception.h>

#if defined(USE_ROCM)

#include <hip/hip_runtime_api.h>

struct CudaDrv
{
    decltype(&hipModuleLoadData)                module_load_data;
    decltype(&hipModuleUnload)                  module_unload;
    decltype(&hipModuleGetFunction)             module_get_function;
    decltype(&hipFuncSetAttribute)              func_set_attribute;
    decltype(&hipModuleLaunchKernel)            launch_kernel;

    static const CudaDrv& instance();
};

#define cuda_check_drv(res) \
do \
{ \
    hipError_t res_ = (res); \
    if (res_ != hipSuccess) \
    { \
        fprintf(stderr, "HIP driver error %d: %s %d\n", static_cast<int>(res_), __FILE__, __LINE__); \
        TORCH_CHECK(false, "HIP driver error"); \
    } \
} \
while(false)

#else

#include <cuda.h>

// CUDA driver API entry points, resolved at runtime from the driver library so the extension
// never links against libcuda (only the runtime is needed at build time). Symbol names are
// stringified after macro expansion, so versioned entry points (cuGraphKernelNodeGetParams_v2
// etc.) resolve to the same ABI the headers were compiled against.

struct CudaDrv
{
    decltype(&cuModuleLoadData)                 module_load_data;
    decltype(&cuModuleUnload)                   module_unload;
    decltype(&cuModuleGetFunction)              module_get_function;
    decltype(&cuFuncSetAttribute)               func_set_attribute;
    decltype(&cuLaunchKernel)                   launch_kernel;
    decltype(&cuGraphKernelNodeGetParams)       graph_kernel_node_get_params;
    decltype(&cuGraphExecKernelNodeSetParams)   graph_exec_kernel_node_set_params;

    static const CudaDrv& instance();
};

#define cuda_check_drv(res) \
do \
{ \
    CUresult res_ = (res); \
    if (res_ != CUDA_SUCCESS) \
    { \
        fprintf(stderr, "CUDA driver error %d: %s %d\n", (int) res_, __FILE__, __LINE__); \
        TORCH_CHECK(false, "CUDA driver error"); \
    } \
} \
while(false)

#endif
