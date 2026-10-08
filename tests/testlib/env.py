"""Hardware and software queries for skip conditions. Every function is safe to call without CUDA."""

import functools
import importlib.util
import os
import sys

import torch


def cuda_available() -> bool:
    return torch.cuda.is_available()


def is_rocm() -> bool:
    return bool(torch.version.hip)


def num_devices() -> int:
    return torch.cuda.device_count() if cuda_available() else 0


def compute_capability(device = None) -> tuple[int, int]:
    """(major, minor) of a CUDA device, (0, 0) without CUDA"""
    if not cuda_available():
        return 0, 0
    return torch.cuda.get_device_capability(torch.device(device) if device is not None else None)


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@functools.cache
def cpu_flags() -> frozenset[str]:
    """CPU feature flags as reported by /proc/cpuinfo (empty elsewhere)"""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return frozenset(line.split(":", 1)[1].split())
    except OSError:
        pass
    return frozenset()


def platform() -> str:
    """"linux" | "windows" | "darwin" """
    return {"win32": "windows"}.get(sys.platform, sys.platform)


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v not in ("", "0", "false", "False")


def get_test_device() -> "torch.device":
    """The configured test device (--device / $EXL3_TEST_DEVICE, pinned per xdist worker), for test modules whose
    helpers need it at module level. Tests should prefer the `device` fixture"""
    return torch.device(os.environ.get("EXL3_TEST_DEVICE", "cuda:0"))
