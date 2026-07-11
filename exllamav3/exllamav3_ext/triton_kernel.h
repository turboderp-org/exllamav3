#pragma once

#include <ATen/Tensor.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

// A Triton kernel compiled ahead of time (triton.compile -> asm["cubin"]), loadable and
// launchable from C++ through the driver API. Launch args are the kernel's non-constexpr
// parameters in signature order; Triton appends two hidden trailing pointer params (global and
// profile scratch), which launch() supplies as null (scratch sizes must be 0, which holds for
// all kernels used here). Compatible with Graph::record_param: kernel nodes captured from
// cuLaunchKernel are identified by the CUfunction handle().
//
// The module is loaded into the primary context of the device that is current at construction;
// construct under the target torch device.

struct TritonKernel
{
    CUmodule mod = nullptr;
    CUfunction fn = nullptr;
    std::string name;
    int num_warps;
    int shared_bytes;

    TritonKernel(py::bytes cubin, std::string name, int num_warps, int shared_bytes);
    ~TritonKernel();
    TritonKernel(const TritonKernel&) = delete;
    TritonKernel& operator=(const TritonKernel&) = delete;

    // args = kernel parameter values (device pointers and 32-bit ints, one value per 8-byte
    // slot), without the trailing scratch params
    void launch(int gx, int gy, int gz, std::vector<void*>& args, cudaStream_t stream) const;

    void* handle() const { return (void*) fn; }
};
