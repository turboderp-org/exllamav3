#pragma once

// Remap the cu* driver names to the HIP runtime functions (same handle types).
// Macros, not hipify renames: hipify maps the five module/func/launch names but
// not the two cuGraph* kernel-node functions, and DRV_STR stringifies after
// expansion.

#if defined(USE_ROCM)

#include <hip/hip_runtime_api.h>

#define cuModuleLoadData                 hipModuleLoadData
#define cuModuleUnload                   hipModuleUnload
#define cuModuleGetFunction              hipModuleGetFunction
#define cuFuncSetAttribute              hipFuncSetAttribute
#define cuLaunchKernel                   hipModuleLaunchKernel
#define cuGraphKernelNodeGetParams       hipGraphKernelNodeGetParams
#define cuGraphExecKernelNodeSetParams   hipGraphExecKernelNodeSetParams

// driver types hipify does not map
using CUgraphNode          = hipGraphNode_t;
using CUgraphExec          = hipGraphExec_t;
using CUmodule             = hipModule_t;
using CUfunction           = hipFunction_t;
using cudaKernelNodeParams = hipKernelNodeParams;
using CUDA_KERNEL_NODE_PARAMS = hipKernelNodeParams;

#endif
