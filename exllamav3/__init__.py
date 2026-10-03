try:
    import torch
except ImportError as e:
    raise RuntimeError(
        "PyTorch is required but not installed. exllamav3 deliberately does not install "
        "a default torch because it must match your CUDA setup; install a matching build "
        "first (for example `uv pip install torch --torch-backend=auto`, or select a CUDA "
        "flavor extra such as `uv sync --extra cu130`). "
        "See the README (\"Building from source\") for all install variants. "
        "https://github.com/turboderp-org/exllamav3"
    ) from e


def _default_allocator_settings():
    """
    Expandable segments for the CUDA caching allocator, unless the user configured the
    allocator themselves.

    The runtime setter works before CUDA is initialized and also applies to every segment
    created after it, so hosts that already touched CUDA before importing the library still
    get it for the model.

    Skipped on Windows (virtual-memory API support is uneven there), on ROCm (the HIP allocator
    accepts the option, but where the driver cannot reserve virtual address ranges every later
    allocation fails as out of memory) and on torch builds that reject the option.
    """
    import os, sys
    if "PYTORCH_CUDA_ALLOC_CONF" in os.environ or sys.platform == "win32":
        return
    if torch.version.hip:
        return
    if os.environ.get("EXL3_EXPANDABLE_SEGMENTS", "1") == "0":
        return
    try:
        # torch >= 2.13 exposes the setter on the accelerator API and deprecates the cuda one
        setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None) \
            or torch.cuda.memory._set_allocator_settings
        setter("expandable_segments:True")
    except Exception:
        pass

_default_allocator_settings()


def _default_triton_backend():
    """
    On a ROCm torch, point Triton at its AMD backend. With an NVIDIA driver also installed, both
    Triton backends report themselves active and Triton refuses to choose between them.
    """
    import os
    if torch.version.hip:
        os.environ.setdefault("TRITON_DEFAULT_BACKEND", "amd")

_default_triton_backend()

from .model.config import Config
from .model.model import Model
from .tokenizer import Tokenizer, MMEmbedding
from .cache import Cache, CacheLayer_fp16, CacheLayer_quant
from .generator import Generator, Job, AsyncGenerator, AsyncJob, Filter, FormatronFilter, LLGuidanceFilter
from .generator.sampler import *