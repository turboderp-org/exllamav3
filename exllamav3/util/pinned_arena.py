from __future__ import annotations
import mmap
import os
import weakref
import torch
from .memory import check_host_memory

# cudaHostRegister / hipHostRegister flags. Portable: usable from every device's context. Mapped: addressable
# from kernels through its device pointer (zero-copy aliases)
_HOST_REGISTER_PORTABLE = 0x01
_HOST_REGISTER_MAPPED = 0x02


def _unregister(ptr: int):
    from ..ext import exllamav3_ext as ext
    torch.cuda.synchronize()
    ext.cuda_host_unregister(ptr)


class PinnedArena:
    """
    One page-locked host region of exactly the requested size, for a bank of fixed-size buffers that would
    otherwise be pinned one tensor at a time.

    Torch's pinned allocator rounds every request up to a power of two and keeps freed blocks for reuse, so
    a bank of odd-sized pinned tensors locks up to twice its nominal size and holds on to it after the
    tensors are gone. The arena is an anonymous mapping registered with the driver instead: its size is the
    budget, it is committed and locked when constructed (so running out of memory surfaces here rather than
    in the middle of a transfer), and close() hands it straight back to the OS. Tensors sliced from
    `tensor` report is_pinned() and take the asynchronous copy paths like any other pinned tensor.

    An arena that is dropped without close() unlocks itself when collected, so the mapping never goes away
    while the driver still holds it registered.

    If the region cannot be locked (RLIMIT_MEMLOCK, driver refusal) it stays pageable: transfers still work,
    but synchronously and at lower bandwidth, so that is warned about once. With `required`, for memory a
    device will address directly (`mapped`), that is an error instead.
    """

    def __init__(self, nbytes: int, what: str, mapped: bool = False, required: bool = False):
        page = mmap.PAGESIZE
        self.size = (int(nbytes) + page - 1) // page * page
        self.what = what
        check_host_memory(self.size, what)
        if os.name == "nt":
            self.map = mmap.mmap(-1, self.size)
        else:
            self.map = mmap.mmap(-1, self.size, flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self.tensor = torch.frombuffer(self.map, dtype = torch.uint8)
        self.ptr = self.tensor.data_ptr()
        self._unregister = None
        try:
            from ..ext import exllamav3_ext as ext
            torch.cuda.init()
            ext.cuda_host_register(
                self.ptr, self.size, _HOST_REGISTER_PORTABLE | (_HOST_REGISTER_MAPPED if mapped else 0)
            )
            self._unregister = weakref.finalize(self, _unregister, self.ptr)
            self._unregister.atexit = False
        except Exception as e:
            if required:
                self.tensor = None
                self.map.close()
                raise RuntimeError(f"{what}: could not page-lock {self.size >> 20} MiB of host memory ({e})") from e
            print(f" !! {what}: could not page-lock {self.size >> 20} MiB ({str(e).splitlines()[0]}), "
                  f"falling back to pageable memory", flush = True)
            # Commit the pages now all the same, which locking would have done
            self.tensor.zero_()


    @property
    def pinned(self) -> bool:
        return self._unregister is not None


    def close(self):
        """
        Unlock and unmap. Transfers still queued against the region are waited for first. Tensors sliced
        from the arena must be dropped by the caller beforehand; if one is still alive the mapping stays
        valid (pageable) until it is gone.
        """
        if self.map is None:
            return
        if self._unregister is not None:
            self._unregister()
            self._unregister = None
        self.tensor = None
        try:
            self.map.close()
        except BufferError:
            pass
        self.map = None
