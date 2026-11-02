"""Dedicated-memory queries for Windows CUDA devices using WDDM."""
import atexit
import ctypes
from functools import lru_cache
import os
from pathlib import Path


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def _check(result, operation):
    if result != 0:
        raise RuntimeError(f"{operation} failed with NVML error {result}")


@lru_cache(maxsize = 1)
def _init_nvml():
    try:
        # DCH drivers install NVML in System32; standard drivers use the NVSMI directory.
        path = Path(os.environ["SystemRoot"]) / "System32" / "nvml.dll"
        if not path.is_file():
            path = Path(os.environ["ProgramW6432"]) / "NVIDIA Corporation" / "NVSMI" / "nvml.dll"
        nvml = ctypes.CDLL(str(path))
        for name, args in (
            ("nvmlInit_v2", []),
            ("nvmlShutdown", []),
            ("nvmlDeviceGetHandleByUUID", [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]),
            ("nvmlDeviceGetDriverModel", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint),
                                         ctypes.POINTER(ctypes.c_uint)]),
            ("nvmlDeviceGetMemoryInfo", [ctypes.c_void_p, ctypes.POINTER(_Memory)]),
        ):
            function = getattr(nvml, name)
            function.argtypes = args
            function.restype = ctypes.c_int
    except (OSError, AttributeError, KeyError) as error:
        raise RuntimeError("Cannot load NVIDIA NVML (nvml.dll) for Windows memory queries") from error
    _check(nvml.nvmlInit_v2(), "nvmlInit_v2")
    atexit.register(nvml.nvmlShutdown)
    return nvml


@lru_cache(maxsize = None)
def _get_device(uuid: str):
    nvml = _init_nvml()
    handle = ctypes.c_void_p()
    # CUDA ordinals can differ from NVML ordinals, especially with CUDA_VISIBLE_DEVICES.
    uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    _check(nvml.nvmlDeviceGetHandleByUUID(uuid.encode("ascii"), ctypes.byref(handle)),
           "nvmlDeviceGetHandleByUUID")
    current, pending = ctypes.c_uint(), ctypes.c_uint()
    _check(nvml.nvmlDeviceGetDriverModel(handle, ctypes.byref(current), ctypes.byref(pending)),
           "nvmlDeviceGetDriverModel")
    return handle, current.value == 0  # NVML_DRIVER_WDDM


def get_wddm_free_memory(uuid: str) -> int | None:
    handle, wddm = _get_device(uuid)
    if not wddm:
        return None
    value = _Memory()
    _check(_init_nvml().nvmlDeviceGetMemoryInfo(handle, ctypes.byref(value)), "nvmlDeviceGetMemoryInfo")
    if value.total == 0 or value.free > value.total:
        raise RuntimeError(f"Invalid NVML memory information: free={value.free}, total={value.total}")
    return value.free
