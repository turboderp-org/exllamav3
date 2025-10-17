from __future__ import annotations
import torch
import numpy as np
from multiprocessing import shared_memory
import uuid
from .model_tp_cuda import cuda_host_register, cuda_host_unregister, CUDA_HOST_REGISTER_PORTABLE

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

    def export(self):
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
        tensor_d = tensor.view((1,)) if len(tensor.shape) == 0 else tensor
        t_cpu = tensor_d.cpu().contiguous()
        src = t_cpu.view(torch.uint8).numpy().view(np.uint8).ravel()
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
                cuda_host_register(self.arena.data_ptr(), self.arena.numel(), flags = CUDA_HOST_REGISTER_PORTABLE)

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
                cuda_host_register(self.arena.data_ptr(), self.arena.numel(), flags = CUDA_HOST_REGISTER_PORTABLE)
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
            cuda_host_unregister(self.arena.data_ptr())
        if self.producer is not None:
            self.shm.close()
