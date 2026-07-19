import ctypes
from functools import lru_cache
import os

CUDA_SUCCESS = 0
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 34
CUDA_HOST_REGISTER_PORTABLE = 1
CUDA_HOST_REGISTER_MAPPED = 2
CUDA_ERROR_CUDART_UNLOADING = 13
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 719
CUDA_DEV_ATTR_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91

@lru_cache(maxsize = 1)
def _cudart():
    """
    Load and cache the CUDA runtime library used for host-memory registration calls.

    Tensor-parallel shared-memory buffers are pinned through cudart so CUDA copies can use them efficiently. The
    lookup handles the common Windows DLL names and Linux SONAMEs, then falls back to ctypes' library search.
    """

    # Windows: Try to find cudart64_*.dll in known CUDA installation paths
    if os.name == "nt":
        candidates = []
        for v in ("130","120","118","117","116","110","101","100"):
            candidates.append(f"cudart64_{v}.dll")
        # Search known CUDA installation directories only
        cuda_paths = []
        for env_var in ("CUDA_PATH", "CUDA_PATH_V12_0", "CUDA_PATH_V11_8"):
            val = os.getenv(env_var)
            if val:
                cuda_paths.append(os.path.join(val, "bin"))
        # Fallback to Program Files
        for pf in ("ProgramFiles", "ProgramW6432"):
            base = os.environ.get(pf)
            if base:
                for d in os.listdir(base) if os.path.isdir(base) else []:
                    if d.lower().startswith("cuda"):
                        cuda_paths.append(os.path.join(base, d, "bin"))
        for cp in cuda_paths:
            if os.path.isdir(cp):
                for name in candidates:
                    full = os.path.join(cp, name)
                    if os.path.isfile(full):
                        try:
                            return ctypes.WinDLL(full)
                        except OSError:
                            pass
        raise OSError("Could not load cudart64_*.dll; ensure CUDA runtime is installed")

    # Linux: try unversioned and common SONAMEs, then ctypes.util.find_library
    else:
        for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11"):
            try:
                return ctypes.CDLL(name)  # cdecl
            except OSError:
                pass
        from ctypes.util import find_library
        path = find_library("cudart")
        if path:
            return ctypes.CDLL(path)
        raise OSError("Could not load libcudart; set LD_LIBRARY_PATH to your CUDA runtime")


def _cuda_error_string(code: int) -> str:
    lib = _cudart()
    fn = lib.cudaGetErrorString
    fn.argtypes = [ctypes.c_int]
    fn.restype  = ctypes.c_char_p
    return fn(code).decode(errors="replace")


def cuda_host_register(ptr: int, nbytes: int, flags: int = 0) -> None:
    lib = _cudart()
    fn = lib.cudaHostRegister
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    fn.restype  = ctypes.c_int
    err = fn(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(flags))
    if err not in (CUDA_SUCCESS, CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED):
        raise RuntimeError(f"cudaHostRegister({hex(ptr)}, {nbytes}) failed: {err} ({_cuda_error_string(err)})")


def cuda_host_get_device_pointer(ptr: int) -> int:
    """
    Return the device-side alias of host memory registered with cudaHostRegisterMapped. On Linux desktop the alias
    equals the host pointer, but under WDDM (native Windows, and potentially WSL2) the host pointer is not directly
    usable in kernels and this alias must be passed instead.
    """
    lib = _cudart()
    fn = lib.cudaHostGetDevicePointer
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    fn.restype  = ctypes.c_int
    dptr = ctypes.c_void_p()
    err = fn(ctypes.byref(dptr), ctypes.c_void_p(ptr), 0)
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cudaHostGetDevicePointer({hex(ptr)}) failed: {err} ({_cuda_error_string(err)})")
    return dptr.value or 0


def cuda_device_get_attribute(attr: int, device: int) -> int:
    lib = _cudart()
    fn = lib.cudaDeviceGetAttribute
    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    fn.restype  = ctypes.c_int
    out = ctypes.c_int()
    err = fn(ctypes.byref(out), ctypes.c_int(attr), ctypes.c_int(device))
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"cudaDeviceGetAttribute({attr}, {device}) failed: {err} ({_cuda_error_string(err)})")
    return out.value


def cuda_host_unregister(ptr: int) -> None:
    lib = _cudart()
    fn = lib.cudaHostUnregister
    fn.argtypes = [ctypes.c_void_p]
    fn.restype  = ctypes.c_int
    err = fn(ctypes.c_void_p(ptr))
    # During teardown (or if already unregistered) treat as benign:
    if err not in (CUDA_SUCCESS, CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, CUDA_ERROR_CUDART_UNLOADING):
        raise RuntimeError(f"cudaHostUnregister({hex(ptr)}) failed: {err} ({_cuda_error_string(err)})")
