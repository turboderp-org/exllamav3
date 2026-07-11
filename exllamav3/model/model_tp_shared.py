from __future__ import annotations
import torch
import numpy as np
from multiprocessing import shared_memory
import uuid
from .model_tp_cuda import cuda_host_register, cuda_host_unregister, CUDA_HOST_REGISTER_PORTABLE

DEFAULT_BUFFER_SIZE = 2 * 1024 ** 3
MAX_CACHE_PER_PROCESS = 4 * 1024**3

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
        """
        Create the producer side of a shared-memory tensor transfer arena.

        The parent process writes serialized CPU tensor bytes into this arena and sends only small metadata records
        over multiprocessing pipes. The arena is pre-touched to avoid page faults during load/dispatch, and large
        CPU tensors can be cached by cache_id so repeated exports do not recopy the same payload.
        """
        self.shm_name = shm_name or uuid.uuid4().hex
        self.buffer_size = buffer_size

        # Create SHM handle and numpy buffer
        self.shm = shared_memory.SharedMemory(create = True, size = self.buffer_size, name = self.shm_name)
        self.buf = np.ndarray((self.buffer_size,), dtype = np.uint8, buffer = self.shm.buf)
        self.buf_is_pinned = False
        self.next_offset = 0

        # Pre-touch buffer to avoid page faults later
        self.buf[: self.buffer_size: 4096] = 0

        # Cache
        self.cached_cpu_tensors = {}
        self.cache_size = 0

    def export(self):
        """
        Return the metadata needed for another process to open this shared-memory arena.
        """
        return {
            "shm_name": self.shm_name,
            "buffer_size": self.buffer_size,
        }

    def send(self, tensor: torch.Tensor | None, cache_id: int = None) -> dict:
        """
        Copy a tensor into shared memory and return a compact import descriptor.

        Small metadata describes whether the payload is absent, already cached by the consumer, stored in this
        arena at a byte offset, or exported through torch's slower share_memory_ fallback when the arena is full.
        """

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

        # Cache
        if nbytes > MAX_CACHE_PER_PROCESS:
            cache_id = None

        if cache_id is not None:
            if cache_id in self.cached_cpu_tensors:
                # print("sending cache ref:", cache_id)
                return {
                    "method": "cached",
                    "cache_id": cache_id,
                }
            while self.cache_size + nbytes > MAX_CACHE_PER_PROCESS:
                self.cached_cpu_tensors.pop(next(iter(self.cached_cpu_tensors)))
            self.cached_cpu_tensors[cache_id] = tensor
            # print("caching send:", cache_id)

        # Data is now buffered in shared memory space, store metadata and offset
        return {
            "method": "buffer",
            "offset": offset,
            "nbytes": nbytes,
            "dtype": str(tensor.dtype),
            "shape": tuple(tensor.shape),
            "cache_id": cache_id
        }

    def clear(self):
        """
        Reset the arena write pointer so subsequent sends can reuse the buffer.
        """
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
        """
        Open the consumer side of a shared-memory tensor transfer arena.

        Remote workers receive producer metadata and open the named shared-memory object; the local pseudo-worker
        can consume the producer directly. When pin_memory is enabled, the arena is registered with CUDA so recv()
        can perform non-blocking copies to the worker device.
        """
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

        # Cache
        self.cached_cpu_tensors = {}
        self.cache_size = 0


    def recv(
        self,
        imp: dict,
        cuda: bool = False,
        slice_dim: int | None = None,
        first: int | None = None,
        last: int | None = None,
    ) -> torch.Tensor | None:
        """
        Reconstruct a tensor from a producer descriptor.

        The descriptor may refer to None, a cached CPU tensor, a torch shared-memory fallback tensor, or a byte
        range in the shared arena. Optional slicing is applied before the final clone or CUDA copy, which lets TP
        workers import only their shard of a larger exported tensor.
        """

        if cuda:
            torch.cuda.set_device(self.device)

        # Get method
        method = imp.get("method")

        # Send was None
        if method == "none_tensor":
            return None

        # Send was cached
        cache_id = imp.get("cache_id")  # Always initialize
        if method == "cached":
            # print("receiving cached:", cache_id)
            assert not cuda, "Cannot share cached tensor for CUDA"
            return self.cached_cpu_tensors[cache_id]

        # Fallback method
        if method == "share_memory":
            tensor = imp["shared_tensor"]

        # Construct Torch tensor in shared memory
        else:
            offset = imp["offset"]
            nbytes = imp["nbytes"]
            dtype = _torch_dtypes[imp["dtype"]]
            shape = imp["shape"]
            tensor = self.arena.narrow(0, offset, nbytes).view(dtype).view(shape)
            if cache_id is not None:
                # print("caching recv:", cache_id)
                assert not cuda, "Cannot share cached tensor for CUDA"
                while self.cache_size + nbytes > MAX_CACHE_PER_PROCESS:
                    self.cached_cpu_tensors.pop(next(iter(self.cached_cpu_tensors)))
                self.cached_cpu_tensors[cache_id] = tensor.clone(memory_format = torch.contiguous_format)

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
        elif imp["method"] != "share_memory" or not tensor.is_contiguous():
            tensor = tensor.clone(memory_format = torch.contiguous_format)

        return tensor


    def close(self):
        if self.pin_memory:
            cuda_host_unregister(self.arena.data_ptr())
        if self.producer is not None:
            self.shm.close()


class TPTensorWrapper:

    @staticmethod
    def tp_export(t: torch.Tensor, plan, producer):
        return {
            "cls": TPTensorWrapper,
            "weight": producer.send(t),
            "device": t.device,
        }

    @staticmethod
    def tp_import_split(local_context, exported, plan, split, dbg = False):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_w = exported["weight"]

        assert split is not None
        _, first, last = split
        w = consumer.recv(id_w, cuda = True, slice_dim = 0, first = first, last = last)
        return w


    @staticmethod
    def tp_import_split_3(local_context, exported, plan, split_0, split_1, split_2, dbg = False):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_w = exported["weight"]

        w_ = []

        for split in [split_0, split_1, split_2]:
            assert split is not None
            _, first, last = split
            w = consumer.recv(id_w, cuda = True, slice_dim = 0, first = first, last = last)
            w_.append(w)

        w = torch.cat(w_, dim = 0).contiguous()
        return w
