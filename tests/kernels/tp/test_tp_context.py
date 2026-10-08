"""
Native TP backend host-side state (parallel/context.cu, all_reduce_cpu.cu), without GPUs:

- pg_init_context(ctx) initializes every PGContext word the collectives and the CPU reduce queue read
  (sync_timeout 0, barrier and LL-broadcast epochs 1, every per-device stage counter, sequence word and queue
  index 0) and writes nothing past sizeof(PGContext). Reference: the struct layout of parallel/context.cuh,
  checked on a sentinel-filled region (whether the collectives work from that state alone is
  test_tp_collectives.py::test_context_init_alone).
- end_cpu_reduce_jobs(ctx) pushes one end marker on the CPU-reduce job ring (MAX_REDUCE_JOBS = 2048 entries, one
  kept free: 2047 pending entries fit, the next push raises "queue is full"); run_cpu_reduce_jobs(ctx, ...)
  consumes jobs in FIFO order and returns at the first end marker, blocking until one arrives (also across
  processes, the helper being a separate process). Head and tail wrap modulo the ring size.

run_cpu_reduce_jobs sets FTZ/DAZ on its calling thread and holds the GIL, so it runs on helper threads (only ever
returning at once, with a marker queued) or in a child process.
"""

import os
import subprocess
import sys
import threading
import time
from multiprocessing import shared_memory

import numpy as np
import pytest

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.nogpu

GLOBALS_SIZE = 128 * 1024
MAX_DEVICES = 16
MB_BLOCKS = 4          # CPUREDUCE_MB_BLOCKS
STAGE_STRIDE = 16      # REDUCE_STAGE_STRIDE (u32 words)
MAX_REDUCE_JOBS = 2048

# Byte offsets in PGContext (alignas(64) struct, parallel/context.cuh)
OFF = dict(
    sync_timeout = 0, barrier_epoch = 4,
    barrier_epoch_device = 16, broadcast_stage_device = 80, reduce_stage_produced = 144,
    reduce_stage_consumed = 208, gather_stage_produced = 272, gather_stage_consumed = 336,
    broadcast_ll_epoch = 448, broadcast_ll_sequence_device = 512,
    reduce_jobs_head = 576, reduce_jobs_tail = 640,
    cpusum_stage_device = 704, cpusum_stage_device_mb = 1728, cpusum_stage_recv = 5824,
    cpusum_stage_recv_mb = 6848, cpusum_stage_cpu = 10944, reduce_jobs = 11008,
)
SIZEOF_CONTEXT = OFF["reduce_jobs"] + 16 * MAX_REDUCE_JOBS


def initialized_words() -> dict[int, int]:
    """{byte offset: value} of every word pg_init_context must set"""
    w = {OFF["sync_timeout"]: 0, OFF["barrier_epoch"]: 1, OFF["broadcast_ll_epoch"]: 1,
         OFF["reduce_jobs_head"]: 0, OFF["reduce_jobs_tail"]: 0, OFF["cpusum_stage_cpu"]: 0}
    for i in range(MAX_DEVICES):
        for f in ("barrier_epoch_device", "broadcast_stage_device", "reduce_stage_produced", "reduce_stage_consumed",
                  "gather_stage_produced", "gather_stage_consumed", "broadcast_ll_sequence_device"):
            w[OFF[f] + 4 * i] = 0
        for f in ("cpusum_stage_device", "cpusum_stage_recv"):
            w[OFF[f] + 4 * STAGE_STRIDE * i] = 0
        for j in range(MB_BLOCKS):
            for f in ("cpusum_stage_device_mb", "cpusum_stage_recv_mb"):
                w[OFF[f] + 4 * STAGE_STRIDE * (i * MB_BLOCKS + j)] = 0
    return w


def u32_at(buf: np.ndarray, off: int) -> int:
    return int(buf[off : off + 4].view(np.uint32)[0])


def test_init_context_fields():
    buf = np.full(GLOBALS_SIZE, 0x5A, dtype = np.uint8)
    ext.pg_init_context(buf.ctypes.data)
    for off, v in initialized_words().items():
        assert u32_at(buf, off) == v, f"word at byte {off}: {u32_at(buf, off):#x}, want {v}"
    tail = buf[SIZEOF_CONTEXT:]
    assert (tail == 0x5A).all(), f"pg_init_context wrote past sizeof(PGContext) at byte {SIZEOF_CONTEXT + int(np.argmax(tail != 0x5A))}"


@pytest.fixture
def ctx_buf():
    buf = np.zeros(GLOBALS_SIZE, dtype = np.uint8)
    ext.pg_init_context(buf.ctypes.data)
    return buf


def run_on_thread(ptr: int, calls: int = 1, timeout: float = 10.0):
    """run_cpu_reduce_jobs `calls` times on a helper thread (FTZ/DAZ stay off the test thread)"""
    err = []

    def body():
        try:
            for _ in range(calls):
                ext.run_cpu_reduce_jobs(ptr, 0, 0)
        except Exception as e:
            err.append(e)

    t = threading.Thread(target = body, daemon = True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "run_cpu_reduce_jobs did not return at a queued end marker"
    assert not err, err


def head_tail(buf):
    return u32_at(buf, OFF["reduce_jobs_head"]), u32_at(buf, OFF["reduce_jobs_tail"])


def test_end_marker_fifo(ctx_buf):
    ptr = ctx_buf.ctypes.data
    for _ in range(3):
        ext.end_cpu_reduce_jobs(ptr)
    assert head_tail(ctx_buf) == (0, 3)
    # Each run consumes exactly one marker
    run_on_thread(ptr)
    assert head_tail(ctx_buf) == (1, 3)
    run_on_thread(ptr, 2)
    assert head_tail(ctx_buf) == (3, 3)
    # A marker is a zero-size job entry
    for i in range(3):
        job = ctx_buf[OFF["reduce_jobs"] + 16 * i : OFF["reduce_jobs"] + 16 * (i + 1)]
        assert (job == 0).all(), f"job {i}: {job}"


def test_ring_capacity_and_wrap(ctx_buf):
    ptr = ctx_buf.ctypes.data
    for _ in range(MAX_REDUCE_JOBS - 1):
        ext.end_cpu_reduce_jobs(ptr)
    with pytest.raises(RuntimeError, match = "queue is full"):
        ext.end_cpu_reduce_jobs(ptr)
    assert head_tail(ctx_buf) == (0, MAX_REDUCE_JOBS - 1)
    run_on_thread(ptr, MAX_REDUCE_JOBS - 1, timeout = 60)
    assert head_tail(ctx_buf) == (MAX_REDUCE_JOBS - 1, MAX_REDUCE_JOBS - 1)
    # Wrap around the end of the ring
    for _ in range(5):
        ext.end_cpu_reduce_jobs(ptr)
    assert head_tail(ctx_buf) == (MAX_REDUCE_JOBS - 1, 4)
    run_on_thread(ptr, 5)
    assert head_tail(ctx_buf) == (4, 4)


_CHILD = """
import sys
from multiprocessing import resource_tracker, shared_memory
resource_tracker.register = resource_tracker.unregister = lambda *a, **k: None
import numpy as np
from exllamav3.ext import exllamav3_ext as ext
shm = shared_memory.SharedMemory(name = sys.argv[1])
buf = np.ndarray((shm.size,), dtype = np.uint8, buffer = shm.buf)
buf[-1] = 1  # entered
ext.run_cpu_reduce_jobs(buf.ctypes.data, 0, 0)
buf[-1] = 2  # returned
del buf
shm.close()
"""


def test_run_blocks_until_marker_across_processes():
    shm = shared_memory.SharedMemory(create = True, size = GLOBALS_SIZE + 64)
    try:
        buf = np.ndarray((shm.size,), dtype = np.uint8, buffer = shm.buf)
        buf[:] = 0
        ext.pg_init_context(buf.ctypes.data)
        p = subprocess.Popen([sys.executable, "-c", _CHILD, shm.name], env = dict(os.environ, CUDA_VISIBLE_DEVICES = ""))
        try:
            deadline = time.monotonic() + 60
            while buf[-1] == 0:
                assert p.poll() is None, f"helper exited early ({p.returncode})"
                assert time.monotonic() < deadline, "helper did not start"
                time.sleep(0.01)
            time.sleep(0.5)
            assert p.poll() is None and buf[-1] == 1, "run_cpu_reduce_jobs returned with an empty queue"
            ext.end_cpu_reduce_jobs(buf.ctypes.data)
            assert p.wait(timeout = 10) == 0
            assert buf[-1] == 2
            assert head_tail(buf) == (1, 1)
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
        del buf
    finally:
        shm.close()
        shm.unlink()
