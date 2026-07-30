#pragma once

#include <cstddef>
#include <cstdint>

// Host-memory registration, used by the tensor-parallel shared arena and by the CPU expert
// offload segment. These are thin wrappers over cudart, called from the extension so they always
// bind the same CUDA runtime the rest of the extension (and torch) is already using.

void cuda_host_register(uintptr_t ptr, size_t nbytes, unsigned int flags);
void cuda_host_unregister(uintptr_t ptr);
uintptr_t cuda_host_get_device_pointer(uintptr_t ptr);
int cuda_device_get_attribute(int attr, int device);
