import ctypes
from functools import lru_cache
import os, glob

CUDA_SUCCESS = 0
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 34
CUDA_HOST_REGISTER_PORTABLE = 1
CUDA_ERROR_CUDART_UNLOADING = 13
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 719

@lru_cache(maxsize = 1)
def _cudart():

    # Windows: Try to find cudart64_*.dll in common paths
    # TODO: Test that this actually works
    if os.name == "nt":
        candidates = [f"cudart64_{v}.dll" for v in ("130","120","118","117","116","110","101","100")]
        for p in os.getenv("PATH","").split(os.pathsep):
            candidates += glob.glob(os.path.join(p, "cudart64_*.dll"))
        last_err = None
        for name in candidates:
            try:
                return ctypes.WinDLL(name)  # __stdcall
            except OSError as e:
                last_err = e
        raise OSError("Could not load cudart64_*.dll; ensure CUDA runtime is on PATH") from last_err

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


def cuda_host_unregister(ptr: int) -> None:
    lib = _cudart()
    fn = lib.cudaHostUnregister
    fn.argtypes = [ctypes.c_void_p]
    fn.restype  = ctypes.c_int
    err = fn(ctypes.c_void_p(ptr))
    # During teardown (or if already unregistered) treat as benign:
    if err not in (CUDA_SUCCESS, CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, CUDA_ERROR_CUDART_UNLOADING):
        raise RuntimeError(f"cudaHostUnregister({hex(ptr)}) failed: {err} ({_cuda_error_string(err)})")
