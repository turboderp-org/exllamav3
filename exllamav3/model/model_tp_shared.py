from __future__ import annotations
import torch
import numpy as np
from multiprocessing import shared_memory, resource_tracker
import sys
import uuid
from .model_tp_cuda import cuda_host_register, cuda_host_unregister, CUDA_HOST_REGISTER_PORTABLE
from ..util.shm import check_shm_capacity

DEFAULT_BUFFER_SIZE = 2 * 1024 ** 3
MAX_CACHE_PER_PROCESS = 4 * 1024**3

def _open_shm(name: str) -> shared_memory.SharedMemory:
    """
    Open an existing shared-memory segment by name without registering it with this process's resource
    tracker: the creating process owns the segment and unlinks it, so a consumer handle must not count as a
    leak when the consumer exits (Python 3.13 has a flag for this; earlier versions register in the
    constructor and have to be told to forget)
    """
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name = name, track = False)
    shm = shared_memory.SharedMemory(name = name)
    resource_tracker.unregister(shm._name, "shared_memory")
    return shm


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

def _cache_insert(owner: SMProducer | SMConsumer, cache_id: int, tensor: torch.Tensor):
    """
    Add a tensor to one side's CPU tensor cache, evicting the oldest entries to stay within the byte limit.

    The producer and every consumer run this same rule over the same sequence of sizes, and a consumer only ever
    sees a subset of what the producer cached, so a consumer never drops an entry the producer still believes is
    cached. That relies on both sides ordering entries by their latest insert: an id the producer evicted and then
    sends again goes to the back of the consumer's queue too, rather than keeping its old position.
    """
    cache = owner.cached_cpu_tensors
    old = cache.pop(cache_id, None)
    if old is not None:
        owner.cache_size -= old.element_size() * old.numel()
    nbytes = tensor.element_size() * tensor.numel()
    while cache and owner.cache_size + nbytes > MAX_CACHE_PER_PROCESS:
        evicted = cache.pop(next(iter(cache)))
        owner.cache_size -= evicted.element_size() * evicted.numel()
    cache[cache_id] = tensor
    owner.cache_size += nbytes


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
        check_shm_capacity(self.buffer_size, "The tensor-parallel shared arena")
        self.shm = shared_memory.SharedMemory(create = True, size = self.buffer_size, name = self.shm_name)
        self.buf = np.ndarray((self.buffer_size,), dtype = np.uint8, buffer = self.shm.buf)
        self.buf_is_pinned = False
        self.next_offset = 0
        # Segments created for tensors that did not fit in the arena, alive until the next clear()
        self.overflow: list[shared_memory.SharedMemory] = []

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
        arena at a byte offset, or stored in an overflow segment created for it because the arena is full.

        There is deliberately no fallback to torch's own storage sharing (tensor.share_memory_() pickled over the
        pipe). torch's file_system strategy keeps a refcount inside the segment that is balanced per pickle: the
        sender increfs once when it pickles, each receiver decrefs once when it rebuilds, and each receiver's
        close() decrements again. A command pickled once and sent to several ranks therefore loses a reference
        per extra rank, and the segment is unlinked as soon as two ranks have finished with it, while the main
        process still owns the tensor and ranks that are behind (one is mid-decode while a prefill arrives) have
        not opened it yet: "unable to open shared memory object ... No such file or directory" in every late
        worker (issue #530, image embeddings larger than the arena). Overflow segments are owned by this producer
        like the arena is, and released by clear() once the forward's acks are in, so a rank can be arbitrarily
        late within a forward and still find its data.
        """

        # None tensor
        if tensor is None:
            return {
                "method": "none_tensor",
            }

        # Bytes to export
        nbytes = tensor.element_size() * tensor.numel()
        nbytes_align = (nbytes + 127) // 128 * 128

        # Cache: a tensor the consumers already hold is sent by reference only
        if nbytes > MAX_CACHE_PER_PROCESS:
            cache_id = None
        if cache_id is not None and cache_id in self.cached_cpu_tensors:
            return {
                "method": "cached",
                "cache_id": cache_id,
            }

        # Destination: the arena, or a segment of its own when the arena cannot hold it
        if self.next_offset + nbytes_align < self.buffer_size:
            offset = self.next_offset
            self.next_offset += nbytes_align
            dst = np.ndarray((nbytes,), dtype = np.uint8, buffer = self.shm.buf, offset = offset)
            desc = {
                "method": "buffer",
                "offset": offset,
                "nbytes": nbytes,
            }
        else:
            check_shm_capacity(nbytes, f"A tensor-parallel transfer of {nbytes / 2**20:.1f} MiB (larger than the arena)")
            segment = shared_memory.SharedMemory(create = True, size = max(nbytes, 1), name = uuid.uuid4().hex)
            self.overflow.append(segment)
            dst = np.ndarray((nbytes,), dtype = np.uint8, buffer = segment.buf)
            desc = {
                "method": "overflow",
                "shm_name": segment.name,
                "nbytes": nbytes,
            }

        # Copy to shared memory
        tensor_d = tensor.view((1,)) if len(tensor.shape) == 0 else tensor
        t_cpu = tensor_d.cpu().contiguous()
        src = t_cpu.view(torch.uint8).numpy().view(np.uint8).ravel()
        np.copyto(dst, src, casting = "no")
        if desc["method"] == "overflow" and sys.platform != "win32":
            # The segment stays reachable by name until clear() unlinks it, so the handle is not needed past the
            # copy: one held per tensor ran a large module's export out of file descriptors. (On Windows a named
            # mapping only lives while a handle to it is open)
            del dst
            segment.close()

        if cache_id is not None:
            _cache_insert(self, cache_id, tensor)

        desc.update({
            "dtype": str(tensor.dtype),
            "shape": tuple(tensor.shape),
            "cache_id": cache_id,
        })
        return desc

    def clear(self):
        """
        Reset the arena write pointer so subsequent sends can reuse the buffer, and release the overflow
        segments of the last round. Called before the next forward's sends, after the previous forward's acks
        have been drained, so every consumer has copied its data out by now.
        """
        self.next_offset = 0
        self._release_overflow()

    def _release_overflow(self):
        for segment in self.overflow:
            segment.close()
            segment.unlink()
        self.overflow = []

    def close(self):
        self._release_overflow()
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

        The descriptor may refer to None, a cached CPU tensor, a byte range in the shared arena, or an overflow
        segment of its own. Optional slicing is applied before the final clone or CUDA copy, which lets TP
        workers import only their shard of a larger exported tensor. Shared memory is only ever viewed here and
        copied out before returning: the producer reuses the arena and unlinks the overflow segments once the
        forward is acknowledged.
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

        # View the bytes in place: the arena, or the overflow segment made for this tensor
        nbytes = imp["nbytes"]
        dtype = _torch_dtypes[imp["dtype"]]
        shape = imp["shape"]
        segment = None
        if method == "overflow":
            segment = _open_shm(imp["shm_name"])
            flat = torch.as_tensor(np.ndarray((nbytes,), dtype = np.uint8, buffer = segment.buf))
            pinned = False
        else:
            flat = self.arena.narrow(0, imp["offset"], nbytes)
            pinned = self.pin_memory
        tensor = flat.view(dtype).view(shape)
        if cache_id is not None:
            # print("caching recv:", cache_id)
            assert not cuda, "Cannot share cached tensor for CUDA"
            _cache_insert(self, cache_id, tensor.clone(memory_format = torch.contiguous_format))

        # Slice before cloning
        if slice_dim is not None:
            tensor = tensor.narrow(slice_dim, first, last - first)

        # Move to GPU or clone to unshared memory. Only the arena is pinned, so only it can be copied
        # asynchronously; an overflow segment is unmapped as soon as the copy is done
        if cuda:
            tensor = tensor.to(
                self.device,
                non_blocking = pinned,
                copy = True,
                memory_format = torch.contiguous_format
            )
        else:
            tensor = tensor.clone(memory_format = torch.contiguous_format)

        if segment is not None:
            del flat
            segment.close()

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
        return TPTensorWrapper.tp_import_split_n(local_context, exported, plan, [split_0, split_1, split_2], dbg)


    @staticmethod
    def tp_import_split_n(local_context, exported, plan, splits, dbg = False):
        consumer = local_context["consumer"]
        device = local_context["device"]
        id_w = exported["weight"]

        w_ = []

        for split in splits:
            assert split is not None
            _, first, last = split
            w = consumer.recv(id_w, cuda = True, slice_dim = 0, first = first, last = last)
            w_.append(w)

        w = torch.cat(w_, dim = 0).contiguous()
        return w
