# Vendored from flash-linear-attention (https://github.com/fla-org/flash-linear-attention),
# v0.5.2, MIT license (see LICENSE in this directory). Replaces fla/utils with just the device
# probes and decorators the forward kernels in this package need: no autograd helpers, no backend
# dispatch, no autotune result cache beyond Triton's own.

import contextlib
import functools
import inspect
import os
from collections import deque

import torch
import triton


def _triton_version() -> tuple[int, int, int]:
    parts = []
    for p in triton.__version__.split("+")[0].split(".")[:3]:
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


TRITON_VERSION = _triton_version()
TRITON_ABOVE_3_4_0 = TRITON_VERSION >= (3, 4, 0)

# Triton >= 3.4 can persist autotune results in its own cache dir; fla enables this by default
SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
autotune_cache_kwargs = {"cache_results": True} if SUPPORTS_AUTOTUNE_CACHE else {}


class DeviceAutotune(triton.KernelInterface):
    """
    triton.autotune keeps one in-memory table of winning configs per kernel, keyed on the key
    arguments and dtypes but not on the device. On a mixed rig the first device to run a shape
    benches and picks, and that config is then launched on every other device: a tile that fits
    a 100 KB device dies at launch on a 64 KB one, since the bench loop catches OutOfResources
    and the launch does not. This keeps one Autotuner per device kind (name and capability) and
    dispatches on the device of the first tensor argument, so each kind benches its own choice.
    Triton's on-disk autotune cache is already keyed on the compile target.

    `configs` may be a callable taking the DeviceKind and returning the config list, so the
    candidates can depend on that device's shared memory and architecture rather than on
    whichever device fla would probe at import (or on the smallest visible one)

    TODO: Track upstream issue and remove this workaround when it's no longer needed,
          https://github.com/triton-lang/triton/issues/12222
    """
    def __init__(self, fn, kwargs):
        self.fn = fn
        self.arg_names = fn.arg_names
        self.keys = kwargs.get("key")
        self._kwargs = kwargs
        self._tuners = {}

    def _tuner(self, args, kwargs):
        idx = None
        for a in args:
            if isinstance(a, torch.Tensor):
                idx = a.device.index
                break
        if idx is None:
            for a in kwargs.values():
                if isinstance(a, torch.Tensor):
                    idx = a.device.index
                    break
        if idx is None:
            idx = torch.cuda.current_device()
        kind = device_kind(idx)
        tuner = self._tuners.get(kind.key)
        if tuner is None:
            kw = self._kwargs
            if callable(kw["configs"]):
                kw = {**kw, "configs": kw["configs"](kind)}
            tuner = self._tuners[kind.key] = triton.autotune(**kw)(self.fn)
        return tuner

    def run(self, *args, **kwargs):
        return self._tuner(args, kwargs).run(*args, **kwargs)


def autotune(**kwargs):
    """Drop-in for triton.autotune (same keyword arguments) with a per-device-kind config table;
    configs may be a callable of the DeviceKind"""
    return lambda fn: DeviceAutotune(fn, kwargs)


@functools.cache
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except Exception:
        return "cuda"


device_platform = get_available_device()
IS_AMD = (device_platform == "hip")
IS_NVIDIA = (device_platform == "cuda")

# fla probes device 0 / the current device only. The kernel code paths these pick are chosen
# once at import for the whole process, so here the flags describe every visible device: a
# workaround flag is set if any device needs it, a capability flag only if all devices have it.
# Autotune config lists are per device instead, see DeviceKind
_caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())] if IS_NVIDIA else []
IS_NVIDIA_BLACKWELL = IS_NVIDIA and any(c[0] in (10, 12) for c in _caps)
IS_TF32_SUPPORTED = IS_NVIDIA and all(c[0] >= 8 for c in _caps)
IS_GATHER_SUPPORTED = hasattr(triton.language, "gather")
IS_TMA_SUPPORTED = False   # fla only enables TMA with FLA_USE_TMA=1; the kernels keep their non-TMA path

if IS_NVIDIA and not IS_TF32_SUPPORTED:
    # Triton defaults to tf32 for fp32 dots, which pre-Ampere cards don't have
    os.environ["TRITON_F32_DEFAULT"] = "ieee"


def _default_alloc_fn(size: int, alignment: int, stream: int | None):
    return torch.empty(size, device = "cuda", dtype = torch.int8)


if IS_NVIDIA_BLACKWELL:
    # Blackwell (SM100 / SM120): the Triton compiler may emit global_scratch for autotuned
    # kernels even without TMA, which needs an allocator. See triton-lang/triton#10002
    triton.set_allocator(_default_alloc_fn)


@functools.cache
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    try:
        return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)["multiprocessor_count"]
    except Exception:
        return 1


@functools.cache
def get_max_shared_mem(tensor_idx: int = 0) -> int:
    try:
        return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)["max_shared_mem"]
    except Exception:
        return -1


# Shared memory per SM that fla associates with each architecture name
_SHARED_MEM = {
    "ada": 101376,      # RTX 4090
    "ampere": 166912,   # A100
    "hopper": 232448,   # H100
}


class DeviceKind:
    """What a kernel's autotune config list may depend on, for one device index. Devices with
    the same name and capability share a kind and so a tuner"""
    def __init__(self, idx: int):
        p = torch.cuda.get_device_properties(idx)
        self.key = (p.name, p.major, p.minor)
        self.major, self.minor = p.major, p.minor
        self.max_shared_mem = get_max_shared_mem(idx)
        self.hopper = IS_NVIDIA and p.major == 9
        self.blackwell = IS_NVIDIA and p.major in (10, 12)

    def shared_mem(self, arch: str = "none") -> bool:
        """True if this device has at least the shared memory fla associates with `arch`
        (fla's check_shared_mem, per device)"""
        return self.max_shared_mem >= _SHARED_MEM.get(arch, 102400)


@functools.cache
def device_kind(idx: int) -> DeviceKind:
    return DeviceKind(idx)


def input_guard(fn):
    """Make all tensor arguments contiguous and run on the device of the first tensor argument"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = [a.contiguous() if isinstance(a, torch.Tensor) else a for a in args]
        kwargs = {k: (v.contiguous() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
        t = next((a for a in [*args, *kwargs.values()] if isinstance(a, torch.Tensor)), None)
        if t is not None and t.device.index is not None:
            ctx = torch.cuda.device(t.device.index)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            return fn(*args, **kwargs)
    return wrapper


def tensor_cache(fn):
    """Memoize the most recent results by argument identity (used by the varlen index helpers)"""
    cached: deque = deque(maxlen = 4)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for cached_args, cached_kwargs, cached_result in cached:
            if len(args) != len(cached_args) or len(kwargs) != len(cached_kwargs):
                continue
            if all(a is b for a, b in zip(args, cached_args)) and \
                    all(k in cached_kwargs and v is cached_kwargs[k] for k, v in kwargs.items()):
                return cached_result
        result = fn(*args, **kwargs)
        cached.append((args, kwargs, result))
        return result
    return wrapper
