#pragma once

#include <cstddef>
#include <cstdint>
#include <ATen/Tensor.h>

// Host-memory registration, used by the tensor-parallel shared arena and by the CPU expert
// offload segment

void cuda_host_register(uintptr_t ptr, size_t nbytes, unsigned int flags);
void cuda_host_unregister(uintptr_t ptr);
uintptr_t cuda_host_get_device_pointer(uintptr_t ptr);
// cudaMemcpyAsync on the current stream between raw pointers; kind: 1 host->device, 2 device->host.
// For host memory registered with cuda_host_register, which torch's copy_ does not always recognize as
// pinned (it then copies synchronously through a staging buffer)
void cuda_memcpy_async(uintptr_t dst, uintptr_t src, size_t nbytes, int kind);
int cuda_device_get_attribute(int attr, int device);
at::Tensor pinned_cuda_view(const at::Tensor& t, int64_t device);
