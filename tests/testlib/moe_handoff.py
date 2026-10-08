"""
The CPU expert-offload handoff segment (cpu/moe_handoff.h) from the parent's side, for driving
exl3_moe_cpu_worker_run and the GPU flag ops without a model. Layout constants are transcribed from moe_handoff.h
(not imported from moe_cpu_host.py, whose copy they are checked against).

    seg = Segment(num_slots, cap_rows, max_hi, max_ho, max_topk, num_wslots, wslot_size)
    seg.start(threads = 2, stage_threads = 2)      # worker loop on a thread (the binding releases the GIL)
    seg.push_job(seq, layer, rows, topk, slot)     # descriptor, then seg.data_ready[slot] = seq (host or GPU)
    seg.wait(seg.done, slot, seq)
    seg.stop()                                     # quit flag, join

Slot sections are torch views (x fp16 [cap_rows, max_hi], sel int32 [cap_rows, max_topk], w fp16, out fp32
[cap_rows, max_ho]); flags are u32 words at a 64-byte stride. `register()` pins and maps the segment for GPU
flag ops / copies and returns the device alias of its base.
"""

import threading
import time
from multiprocessing import shared_memory

import numpy as np
import torch

MOE_JOB_RING = 256
MOE_MAX_SLOTS = 8
MOE_MAX_WSLOTS = 8
MOE_JOB_MAX_EXPERTS = 256
MOE_JOB_BYTES = 4 * (7 + MOE_JOB_MAX_EXPERTS + 1)     # sizeof(MoeJob)
MOE_CTRL_JOBS_OFFSET = 384
MOE_SLOT_FLAGS_OFFSET = MOE_CTRL_JOBS_OFFSET + MOE_JOB_RING * MOE_JOB_BYTES
MOE_FLAGS_SIZE = 3 * 64 * MOE_MAX_SLOTS + 2 * 64 * MOE_MAX_WSLOTS
MOE_STAGE_RING = 64
MOE_STAGE_TAIL_OFFSET = MOE_SLOT_FLAGS_OFFSET + MOE_FLAGS_SIZE
MOE_STAGE_HEAD_OFFSET = MOE_STAGE_TAIL_OFFSET + 64
MOE_STAGE_JOBS_OFFSET = MOE_STAGE_TAIL_OFFSET + 128
MOE_CTRL_SIZE = MOE_STAGE_JOBS_OFFSET + MOE_STAGE_RING * MOE_JOB_BYTES

KIND_COMPUTE, KIND_STAGE, KIND_COMPUTE_GATED = 0, 1, 2

# Control words (byte offsets)
QUIT, PASS_WAKE, ABORT, READY, JOBS_TAIL, JOBS_HEAD = 0, 64, 128, 192, 256, 320
# Flag banks (byte offset of entry 0; entries 64 bytes apart)
DATA_READY = MOE_SLOT_FLAGS_OFFSET
DONE = DATA_READY + 64 * MOE_MAX_SLOTS
CONSUMED = DONE + 64 * MOE_MAX_SLOTS
STAGE_DONE = CONSUMED + 64 * MOE_MAX_SLOTS
PINNED_FREE = STAGE_DONE + 64 * MOE_MAX_WSLOTS


def align64(x: int) -> int:
    return (x + 63) & ~63


class Segment:

    def __init__(self, num_slots, cap_rows, max_hi, max_ho, max_topk, num_wslots = 0, wslot_size = 0,
                 sentinel = 0x7B):
        self.num_slots, self.cap_rows = num_slots, cap_rows
        self.max_hi, self.max_ho, self.max_topk = max_hi, max_ho, max_topk
        self.off_x = 0
        self.off_sel = align64(cap_rows * max_hi * 2)
        self.off_w = align64(self.off_sel + cap_rows * max_topk * 4)
        self.off_out = align64(self.off_w + cap_rows * max_topk * 2)
        self.slot_size = align64(self.off_out + cap_rows * max_ho * 4)
        self.wstage_off = MOE_CTRL_SIZE + num_slots * self.slot_size
        self.num_wslots, self.wslot_size = num_wslots, wslot_size
        self.size = self.wstage_off + num_wslots * wslot_size
        self.shm = shared_memory.SharedMemory(create = True, size = self.size)
        self.u8 = np.ndarray((self.size,), dtype = np.uint8, buffer = self.shm.buf)
        self.u8[:MOE_CTRL_SIZE] = 0
        self.u8[MOE_CTRL_SIZE:] = sentinel
        self.u32 = np.ndarray((self.size // 4,), dtype = np.uint32, buffer = self.shm.buf)
        self.base = self.u8.ctypes.data
        self.dev_base = None
        self.thread = None
        self.error = []

    # Views

    def word(self, off: int) -> int:
        return int(self.u32[off // 4])

    def set_word(self, off: int, v: int):
        self.u32[off // 4] = v & 0xFFFFFFFF

    def flag(self, bank: int, idx: int) -> int:
        return self.word(bank + 64 * idx)

    def set_flag(self, bank: int, idx: int, v: int):
        self.set_word(bank + 64 * idx, v)

    def section(self, slot: int, name: str) -> torch.Tensor:
        off, count, dtype, cols = dict(
            x = (self.off_x, self.cap_rows * self.max_hi, torch.half, self.max_hi),
            sel = (self.off_sel, self.cap_rows * self.max_topk, torch.int32, self.max_topk),
            w = (self.off_w, self.cap_rows * self.max_topk, torch.half, self.max_topk),
            out = (self.off_out, self.cap_rows * self.max_ho, torch.float, self.max_ho),
        )[name]
        return torch.frombuffer(self.shm.buf, dtype = dtype, count = count,
                                offset = MOE_CTRL_SIZE + slot * self.slot_size + off).view(self.cap_rows, cols)

    def slot_bytes(self, slot: int) -> np.ndarray:
        b = MOE_CTRL_SIZE + slot * self.slot_size
        return self.u8[b : b + self.slot_size]

    def wslot_bytes(self, ws: int) -> np.ndarray:
        b = self.wstage_off + ws * self.wslot_size
        return self.u8[b : b + self.wslot_size]

    # Job rings

    def _write_job(self, ring_off, index, seq, layer, rows, topk, slot, kind, prev_seq = 0, experts = ()):
        j = np.ndarray((MOE_JOB_BYTES // 4,), dtype = np.uint32, buffer = self.shm.buf,
                       offset = ring_off + index * MOE_JOB_BYTES)
        j[:7] = [seq, layer, rows, topk, slot, kind, prev_seq]
        j[7 : 7 + len(experts)] = experts

    def push_job(self, seq, layer, rows, topk, slot, kind = KIND_COMPUTE):
        """Compute job descriptor, published by advancing jobs_tail (before the slot's data_ready flag)"""
        tail = self.word(JOBS_TAIL)
        assert tail - self.word(JOBS_HEAD) < MOE_JOB_RING
        self._write_job(MOE_CTRL_JOBS_OFFSET, tail % MOE_JOB_RING, seq, layer, rows, topk, slot, kind)
        self.set_word(JOBS_TAIL, tail + 1)

    def push_stage(self, seq, layer, experts, wslot, prev_seq):
        tail = self.word(MOE_STAGE_TAIL_OFFSET)
        assert tail - self.word(MOE_STAGE_HEAD_OFFSET) < MOE_STAGE_RING
        self._write_job(MOE_STAGE_JOBS_OFFSET, tail % MOE_STAGE_RING, seq, layer, len(experts), 0, wslot, KIND_STAGE,
                        prev_seq, experts)
        self.set_word(MOE_STAGE_TAIL_OFFSET, tail + 1)

    def wait(self, bank, idx, seq, timeout = 30.0):
        """Poll a flag until it reaches seq (cyclic >=), as the GPU wait does"""
        deadline = time.monotonic() + timeout
        while ((self.flag(bank, idx) - seq) & 0xFFFFFFFF) >= 0x80000000:
            if self.error:
                raise self.error[0]
            if time.monotonic() > deadline:
                raise TimeoutError(f"flag {bank}+{idx}: {self.flag(bank, idx)}, waiting for {seq}")
            time.sleep(0.0002)

    # Worker

    def start(self, threads = 2, stage_threads = 2, timeout = 60.0):
        from exllamav3.ext import exllamav3_ext as ext

        def body():
            try:
                ext.exl3_moe_cpu_worker_run(self.base, self.num_slots, self.slot_size, self.cap_rows, self.max_hi,
                                            self.max_ho, self.max_topk, self.wstage_off, self.num_wslots,
                                            self.wslot_size, threads, stage_threads)
            except Exception as e:
                self.error.append(e)

        self.thread = threading.Thread(target = body, daemon = True)
        self.thread.start()
        deadline = time.monotonic() + timeout
        while not self.word(READY):
            if self.error:
                raise self.error[0]
            assert time.monotonic() < deadline, "worker did not signal ready"
            time.sleep(0.001)

    def stop(self, timeout = 10.0) -> bool:
        """Set quit; True if the worker loop (and its stager) returned within the timeout"""
        self.set_word(QUIT, 1)
        if self.thread is None:
            return True
        self.thread.join(timeout)
        return not self.thread.is_alive()

    # GPU access

    def register(self) -> int:
        from exllamav3.model.model_tp_cuda import (CUDA_HOST_REGISTER_MAPPED, CUDA_HOST_REGISTER_PORTABLE,
                                                   cuda_host_get_device_pointer, cuda_host_register)
        cuda_host_register(self.base, self.size, flags = CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED)
        self.dev_base = cuda_host_get_device_pointer(self.base)
        return self.dev_base

    def close(self):
        if self.dev_base is not None:
            from exllamav3.model.model_tp_cuda import cuda_host_unregister
            torch.cuda.synchronize()
            cuda_host_unregister(self.base)
            self.dev_base = None
        del self.u8, self.u32
        try:
            self.shm.close()
        except BufferError:
            pass   # torch views of the slots still alive; the mapping goes with the process
        self.shm.unlink()
