#pragma once

#if defined(USE_ROCM)
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#endif

// CUDA driver API entry points, resolved at runtime from the driver library so the extension
// never links against libcuda (only the runtime is needed at build time). Symbol names are
// stringified after macro expansion, so versioned entry points (cuGraphKernelNodeGetParams_v2
// etc.) resolve to the same ABI the headers were compiled against.
//
// On ROCm the same role is filled by libamdhip64: hipModule* and hipGraph* are exported by
// the runtime library itself, so they are resolved with the same lazy-dlopen pattern.

struct CudaDrv
{
#if defined(USE_ROCM)
    decltype(&hipModuleLoadData)                module_load_data;
    decltype(&hipModuleUnload)                  module_unload;
    decltype(&hipModuleGetFunction)             module_get_function;
    decltype(&hipFuncSetAttribute)              func_set_attribute;
    decltype(&hipModuleLaunchKernel)            launch_kernel;
    decltype(&hipGraphKernelNodeGetParams)      graph_kernel_node_get_params;
    decltype(&hipGraphExecKernelNodeSetParams)  graph_exec_kernel_node_set_params;
#else
    decltype(&cuModuleLoadData)                 module_load_data;
    decltype(&cuModuleUnload)                   module_unload;
    decltype(&cuModuleGetFunction)              module_get_function;
    decltype(&cuFuncSetAttribute)               func_set_attribute;
    decltype(&cuLaunchKernel)                   launch_kernel;
    decltype(&cuGraphKernelNodeGetParams)       graph_kernel_node_get_params;
    decltype(&cuGraphExecKernelNodeSetParams)   graph_exec_kernel_node_set_params;
#endif

    static const CudaDrv& instance();
};

#if defined(USE_ROCM)
#define cuda_check_drv(res) \
do \
{ \
    hipError_t res_ = (res); \
    if (res_ != hipSuccess) \
    { \
        fprintf(stderr, "HIP error %d (%s): %s %d\n", (int) res_, hipGetErrorName(res_), __FILE__, __LINE__); \
        TORCH_CHECK(false, "HIP driver/runtime error"); \
    } \
} \
while(false)
#else
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
