#include <Python.h>
#include "triton_kernel.h"
#include <c10/util/Exception.h>
#include <cstdio>
#include "cuda_drv.h"

TritonKernel::TritonKernel(py::bytes cubin, std::string _name, int _num_warps, int _shared_bytes) :
    name(std::move(_name)),
    num_warps(_num_warps),
    shared_bytes(_shared_bytes)
{
    // Ensure the primary context of the current device is initialized and current for the
    // driver API before loading the module
    cudaFree(nullptr);

    std::string data = cubin;
    const CudaDrv& drv = CudaDrv::instance();
    cuda_check_drv(drv.module_load_data(&mod, data.data()));
    cuda_check_drv(drv.module_get_function(&fn, mod, name.c_str()));
    if (shared_bytes > 48 * 1024)
        cuda_check_drv(drv.func_set_attribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes));
}

TritonKernel::~TritonKernel()
{
    if (mod) CudaDrv::instance().module_unload(mod);
}

void TritonKernel::launch(int gx, int gy, int gz, std::vector<void*>& args, cudaStream_t stream) const
{
    // Two hidden trailing params: global scratch and profile scratch, both unused (size 0)
    args.push_back(nullptr);
    args.push_back(nullptr);
    std::vector<void*> arg_ptrs(args.size());
    for (size_t i = 0; i < args.size(); ++i)
        arg_ptrs[i] = &args[i];

    cuda_check_drv(
        CudaDrv::instance().launch_kernel(
            fn,
            gx, gy, gz,
            32 * num_warps, 1, 1,
            shared_bytes,
            stream,
            arg_ptrs.data(),
            nullptr
        )
    );
}
