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


@functools.cache
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except Exception:
        return "cuda"


device_platform = get_available_device()
IS_AMD = (device_platform == "hip")
IS_NVIDIA = (device_platform == "cuda")

# fla probes device 0 / the current device only. Kernel code paths and config lists are chosen
# once at import for the whole process, so here the flags describe every visible device: a
# workaround flag is set if any device needs it, a capability flag only if all devices have it
_caps = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())] if IS_NVIDIA else []
IS_NVIDIA_HOPPER = IS_NVIDIA and any(c[0] == 9 for c in _caps)
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
def get_all_max_shared_mem() -> list[int]:
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)["max_shared_mem"]
            for i in range(torch.cuda.device_count())
        ]
    except Exception:
        return [-1]


# Shared memory per SM that fla associates with each architecture name
_SHARED_MEM = {
    "ada": 101376,      # RTX 4090
    "ampere": 166912,   # A100
    "hopper": 232448,   # H100
}


@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    """True if every visible device has at least the shared memory of `arch` (fla checks device
    `tensor_idx` only; the tile lists derived from this are process-wide, so the smallest device
    has to fit)"""
    try:
        return min(get_all_max_shared_mem()) >= _SHARED_MEM.get(arch, 102400)
    except Exception:
        return False


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
