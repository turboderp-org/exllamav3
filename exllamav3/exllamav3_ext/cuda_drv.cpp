#include <cstdio>
#include <c10/util/Exception.h>
#include "cuda_drv.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define DRV_STR2(x) #x
#define DRV_STR(x) DRV_STR2(x)

#if defined(USE_ROCM)

// HIP has no separate driver library: the module/graph entry points are exported by
// the HIP runtime the extension already links against.

const CudaDrv& CudaDrv::instance()
{
    static CudaDrv d = []
    {
        CudaDrv d{};
        d.module_load_data                  = cuModuleLoadData;
        d.module_unload                     = cuModuleUnload;
        d.module_get_function               = cuModuleGetFunction;
        d.func_set_attribute                = cuFuncSetAttribute;
        d.launch_kernel                     = cuLaunchKernel;
        d.graph_kernel_node_get_params      = cuGraphKernelNodeGetParams;
        d.graph_exec_kernel_node_set_params = cuGraphExecKernelNodeSetParams;
        return d;
    }
    ();
    return d;
}

#else

static void* drv_sym(void* lib, const char* name)
{
    #ifdef _WIN32
        void* fp = (void*) GetProcAddress((HMODULE) lib, name);
    #else
        void* fp = dlsym(lib, name);
    #endif
    TORCH_CHECK(fp, "CUDA driver symbol not found: ", name);
    return fp;
}

const CudaDrv& CudaDrv::instance()
{
    static CudaDrv d = []
    {
        #ifdef _WIN32
            void* lib = (void*) LoadLibraryA("nvcuda.dll");
        #else
            void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
            if (!lib) lib = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
        #endif
        TORCH_CHECK(lib, "Could not load the CUDA driver library");

        CudaDrv d{};
        d.module_load_data                  = (decltype(&cuModuleLoadData))               drv_sym(lib, DRV_STR(cuModuleLoadData));
        d.module_unload                     = (decltype(&cuModuleUnload))                 drv_sym(lib, DRV_STR(cuModuleUnload));
        d.module_get_function               = (decltype(&cuModuleGetFunction))            drv_sym(lib, DRV_STR(cuModuleGetFunction));
        d.func_set_attribute                = (decltype(&cuFuncSetAttribute))             drv_sym(lib, DRV_STR(cuFuncSetAttribute));
        d.launch_kernel                     = (decltype(&cuLaunchKernel))                 drv_sym(lib, DRV_STR(cuLaunchKernel));
        d.graph_kernel_node_get_params      = (decltype(&cuGraphKernelNodeGetParams))     drv_sym(lib, DRV_STR(cuGraphKernelNodeGetParams));
        d.graph_exec_kernel_node_set_params = (decltype(&cuGraphExecKernelNodeSetParams)) drv_sym(lib, DRV_STR(cuGraphExecKernelNodeSetParams));
        return d;
    }
    ();
    return d;
}

#endif
