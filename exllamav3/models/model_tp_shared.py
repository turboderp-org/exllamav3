from __future__ import annotations
import torch
import numpy as np
from multiprocessing import shared_memory
import uuid
import ctypes
from functools import lru_cache
import os, glob

DEFAULT_BUFFER_SIZE = 2 * 1024 ** 3

_torch_dtypes = {
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.bfloat16": torch.bfloat16,
}

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


def _cuda_host_register(ptr: int, nbytes: int, flags: int = 0) -> None:
    lib = _cudart()
    fn = lib.cudaHostRegister
    fn.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    fn.restype  = ctypes.c_int
    err = fn(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(flags))
    if err not in (CUDA_SUCCESS, CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED):
        raise RuntimeError(f"cudaHostRegister({hex(ptr)}, {nbytes}) failed: {err} ({_cuda_error_string(err)})")


def _cuda_host_unregister(ptr: int) -> None:
    lib = _cudart()
    fn = lib.cudaHostUnregister
    fn.argtypes = [ctypes.c_void_p]
    fn.restype  = ctypes.c_int
    err = fn(ctypes.c_void_p(ptr))
    # During teardown (or if already unregistered) treat as benign:
    if err not in (CUDA_SUCCESS, CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED, CUDA_ERROR_CUDART_UNLOADING):
        raise RuntimeError(f"cudaHostUnregister({hex(ptr)}) failed: {err} ({_cuda_error_string(err)})")


class SMProducer:
    def __init__(
        self,
        shm_name: str | None = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ):
        self.shm_name = shm_name or uuid.uuid4().hex
        self.buffer_size = buffer_size

        # Create SHM handle and numpy buffer
        self.shm = shared_memory.SharedMemory(create = True, size = self.buffer_size, name = self.shm_name)
        self.buf = np.ndarray((self.buffer_size,), dtype = np.uint8, buffer = self.shm.buf)
        self.buf_is_pinned = False
        self.next_offset = 0

        # Pre-touch buffer to avoid page faults later
        self.buf[: self.buffer_size: 4096] = 0

    def export(self, device):
        return {
            "shm_name": self.shm_name,
            "buffer_size": self.buffer_size,
        }

    def send(self, tensor: torch.Tensor | None) -> dict:

        # None tensor
        if tensor is None:
            return {
                "method": "none_tensor",
            }

        # Bytes to export
        nbytes = tensor.element_size() * tensor.numel()
        nbytes_align = (nbytes + 127) // 128 * 128

        # Fall back on slow sharing if buffer too small
        if self.next_offset + nbytes_align >= self.buffer_size:
            tensor.share_memory_()
            return {
                "method": "share_memory",
                "shared_tensor": tensor,
            }

        # Allocate space
        offset = self.next_offset
        self.next_offset += nbytes_align

        # Copy to shared buffer
        t_cpu = tensor.cpu().contiguous()
        src = t_cpu.numpy().view(np.uint8).ravel()
        dst = np.ndarray((nbytes,), dtype = np.uint8, buffer = self.shm.buf, offset = offset)
        np.copyto(dst, src, casting = "no")

        # Data is now buffered in shared memory space, store metadata and offset
        return {
            "method": "buffer",
            "offset": offset,
            "nbytes": nbytes,
            "dtype": str(tensor.dtype),
            "shape": tuple(tensor.shape),
        }

    def clear(self):
        self.next_offset = 0

    def close(self):
        self.shm.close()
        self.shm.unlink()


class SMConsumer:

    def __init__(
        self,
        producer_imp: dict | SMProducer,
        device: int | None = None,
        pin_memory: bool = False,
    ):
        self.pin_memory = pin_memory
        self.device = device
        torch.cuda.set_device(self.device)

        def get_local_tensor(shm_buf, _buffer_size):
            # Create local uint8 tensor mapping the entire shared buffer
            np_view = np.ndarray(
                shape = (_buffer_size,),
                dtype = np.uint8,
                buffer = shm_buf,
                offset = 0,
            )
            return torch.as_tensor(np_view)

        # Remote process consumer
        if isinstance(producer_imp, dict):
            self.producer = None
            self.shm_name = producer_imp["shm_name"]
            self.buffer_size = producer_imp["buffer_size"]
            self.shm = shared_memory.SharedMemory(name = self.shm_name)
            self.arena = get_local_tensor(self.shm.buf, self.buffer_size)

            # Optionally pin memory
            if pin_memory:
                torch.cuda.set_device(self.device)
                _cuda_host_register(self.arena.data_ptr(), self.arena.numel(), flags = CUDA_HOST_REGISTER_PORTABLE)

        # Local consumer
        else:
            assert isinstance(producer_imp, SMProducer)
            self.producer = producer_imp
            self.shm_name = producer_imp.shm_name
            self.buffer_size = producer_imp.buffer_size
            self.shm = producer_imp.shm
            self.arena = get_local_tensor(self.shm.buf, self.buffer_size)

            # Optionally pin memory
            if pin_memory:
                torch.cuda.set_device(self.device)
                assert not self.producer.buf_is_pinned, "Only one local consumer can pin arena"
                _cuda_host_register(self.arena.data_ptr(), self.arena.numel(), flags = CUDA_HOST_REGISTER_PORTABLE)
                self.producer.buf_is_pinned = True

    def recv(
        self,
        imp: dict,
        cuda: bool = False,
        slice_dim: int | None = None,
        first: int | None = None,
        last: int | None = None,
    ) -> torch.Tensor | None:

        if cuda:
            torch.cuda.set_device(self.device)

        # Send was None
        if imp["method"] == "none_tensor":
            return None

        # Fallback method
        if imp["method"] == "share_memory":
            tensor = imp["shared_tensor"]

        # Construct Torch tensor in shared memory
        else:
            offset = imp["offset"]
            nbytes = imp["nbytes"]
            dtype = _torch_dtypes[imp["dtype"]]
            shape = imp["shape"]
            tensor = self.arena.narrow(0, offset, nbytes).view(dtype).view(shape)

        # Slice before cloning
        if slice_dim is not None:
            tensor = tensor.narrow(slice_dim, first, last - first)

        # Move to GPU or clone to unshared memory
        if cuda:
            tensor = tensor.to(
                self.device,
                non_blocking = self.pin_memory,
                copy = True,
                memory_format = torch.contiguous_format
            )
        else:
            tensor = tensor.clone(memory_format = torch.contiguous_format)

        return tensor


    def close(self):
        if self.pin_memory:
            _cuda_host_unregister(self.arena.data_ptr())
        if self.producer is not None:
            self.shm.close()
