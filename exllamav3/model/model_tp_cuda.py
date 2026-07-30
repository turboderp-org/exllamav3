from __future__ import annotations
from ..ext import exllamav3_ext as ext

# Host-memory registration used by the tensor-parallel shared arena and the CPU expert offload
# segment. The calls themselves live in the extension (see exllamav3_ext/cuda_host.cpp), which is
# linked against the same CUDA runtime torch loads. An earlier version of this module located a
# cudart shared library at runtime and called it through ctypes, which meant guessing versioned
# DLL/SONAME names, searching PATH on Windows, and risking a second runtime version being loaded
# into the process alongside torch's.

# cudaHostRegister flags, from driver_types.h
CUDA_HOST_REGISTER_DEFAULT = 0x00
CUDA_HOST_REGISTER_PORTABLE = 0x01
CUDA_HOST_REGISTER_MAPPED = 0x02

# cudaDeviceAttr values, from driver_types.h
CUDA_DEV_ATTR_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91


def cuda_host_register(ptr: int, nbytes: int, flags: int = 0) -> None:
    """
    Pin a host allocation so CUDA copies can use it, and (with CUDA_HOST_REGISTER_MAPPED) so it can
    be addressed from kernels. Registering a region that is already pinned is a no-op.
    """
    ext.cuda_host_register(ptr, nbytes, flags)


def cuda_host_unregister(ptr: int) -> None:
    """
    Unpin a host allocation. Unregistering a region that is not (or no longer) registered is a
    no-op, as is unregistering while the CUDA runtime is unloading, so this is safe during teardown.
    """
    ext.cuda_host_unregister(ptr)


def cuda_host_get_device_pointer(ptr: int) -> int:
    """
    Return the device-side alias of host memory registered with cudaHostRegisterMapped. On Linux
    desktop the alias equals the host pointer, but under WDDM (native Windows, and potentially
    WSL2) the host pointer is not directly usable in kernels and this alias must be passed instead.
    """
    return ext.cuda_host_get_device_pointer(ptr)


def cuda_device_get_attribute(attr: int, device: int) -> int:
    """
    Query a cudaDeviceAttr value for the given device ordinal.
    """
    return ext.cuda_device_get_attribute(attr, device)
