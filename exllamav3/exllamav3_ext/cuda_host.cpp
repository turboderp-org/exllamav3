#include "cuda_host.h"
#include <ATen/cuda/CUDAContext.h>
#include "util.h"

// Shared host buffers (the TP arena, the MoE offload segment) are pinned through cudart so CUDA
// copies can use them and, where the platform allows it, so kernels can address them directly.
//
// These calls previously went through ctypes against a cudart shared library located at runtime
// by name. That meant guessing DLL/SONAME versions, searching PATH, and potentially binding a
// second runtime version alongside the one torch had already loaded. Calling from the extension
// removes the search entirely, and lets the compiler supply the cudaError_t values instead of the
// hardcoded constants the ctypes version used (which did not match cudaError_t, so the benign
// cases handled below were raising rather than being tolerated).

void cuda_host_register(uintptr_t ptr, size_t nbytes, unsigned int flags)
{
    cudaError_t cr = cudaHostRegister(reinterpret_cast<void*>(ptr), nbytes, flags);

    // A region can only be pinned once per process, but ranks sharing an arena may each try to
    // register it. The runtime also records the returned error as the last error, so clear it or
    // the next cudaGetLastError() elsewhere (torch's launch checks) attributes it to something
    // unrelated
    if (cr == cudaErrorHostMemoryAlreadyRegistered)
    {
        cudaGetLastError();
        return;
    }

    TORCH_CHECK(
        cr == cudaSuccess,
        "cudaHostRegister(", reinterpret_cast<void*>(ptr), ", ", nbytes, ") failed: ",
        cudaGetErrorString(cr)
    );
}

void cuda_host_unregister(uintptr_t ptr)
{
    cudaError_t cr = cudaHostUnregister(reinterpret_cast<void*>(ptr));

    // Teardown is racy by nature: the region may already have been released, or the runtime may
    // be unloading while shared segments are still being torn down. Both are benign here
    if (cr == cudaErrorHostMemoryNotRegistered || cr == cudaErrorCudartUnloading)
    {
        cudaGetLastError();
        return;
    }

    TORCH_CHECK(
        cr == cudaSuccess,
        "cudaHostUnregister(", reinterpret_cast<void*>(ptr), ") failed: ", cudaGetErrorString(cr)
    );
}

uintptr_t cuda_host_get_device_pointer(uintptr_t ptr)
{
    // Device-side alias of memory registered with cudaHostRegisterMapped. Equal to the host
    // pointer under UVA (Linux desktop), but distinct under WDDM (native Windows, and potentially
    // WSL2), where the host pointer is not usable in kernels and this alias must be passed instead
    void* dev_ptr = nullptr;
    cudaError_t cr = cudaHostGetDevicePointer(&dev_ptr, reinterpret_cast<void*>(ptr), 0);

    TORCH_CHECK(
        cr == cudaSuccess,
        "cudaHostGetDevicePointer(", reinterpret_cast<void*>(ptr), ") failed: ",
        cudaGetErrorString(cr)
    );

    return reinterpret_cast<uintptr_t>(dev_ptr);
}

int cuda_device_get_attribute(int attr, int device)
{
    int value = 0;
    cudaError_t cr = cudaDeviceGetAttribute(&value, static_cast<cudaDeviceAttr>(attr), device);

    TORCH_CHECK(
        cr == cudaSuccess,
        "cudaDeviceGetAttribute(", attr, ", ", device, ") failed: ", cudaGetErrorString(cr)
    );

    return value;
}
