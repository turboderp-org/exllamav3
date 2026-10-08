"""
GPU-side flag ops of the CPU expert-offload handoff (cpu/moe_handoff.cu): exl3_moe_flag_write(addr, value) and
exl3_moe_flag_wait(addr, value, abort_addr) on mapped, registered host memory, in both implementations
exl3_moe_cpu_set_memops selects (stream memory operations, or the fallback kernels).

Contracts:
- flag_write stores (uint32) value at addr in stream order on the current torch stream: not before the stream's
  earlier work, and after everything the stream copied to host memory earlier (a reader that sees the flag sees
  the data). Other streams are not ordered behind it.
- flag_wait holds the current stream until the flag reaches value in the cyclic sense ((int32)(flag - value) >= 0,
  so 3 satisfies a wait for 0xFFFFFFF0 and 0x80000005 does not satisfy a wait for 5); later work on the stream
  (including copies from host memory written before the flag) runs only afterwards. Values are taken mod 2^32.
- The two together carry the parent/worker handshake: data out + flag, flag in + data, with the counterpart a
  host thread or another process polling and writing the same words (the CPU worker process in MoeCpuHost).
- Fallback kernel only: a wait still unsatisfied after its timeout (30 s) sets *abort_addr = 1 and lets the stream
  continue (the memop wait has no timeout; MoeCpuHost's watchdog unblocks it). slow.
"""

import os
import subprocess
import sys
import threading
import time
from multiprocessing import shared_memory

import numpy as np
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.model.model_tp_cuda import (CUDA_HOST_REGISTER_MAPPED, CUDA_HOST_REGISTER_PORTABLE,
                                           cuda_host_get_device_pointer, cuda_host_register, cuda_host_unregister)

SLEEP_CYCLES = 200_000_000   # torch.cuda._sleep: ~0.1 s
DATA_OFF = 4096              # data area after the flag words
DATA_BYTES = 1 << 16


class Region:
    """Registered, mapped shared memory: u32 flag words at a 64-byte stride from byte 0, a data area from DATA_OFF"""

    def __init__(self, size = DATA_OFF + 2 * DATA_BYTES):
        self.shm = shared_memory.SharedMemory(create = True, size = size)
        self.u8 = np.ndarray((size,), dtype = np.uint8, buffer = self.shm.buf)
        self.u8[:] = 0
        self.u32 = np.ndarray((size // 4,), dtype = np.uint32, buffer = self.shm.buf)
        self.base = self.u8.ctypes.data
        cuda_host_register(self.base, size, flags = CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED)
        self.dev = cuda_host_get_device_pointer(self.base)

    def addr(self, i):
        return self.dev + 64 * i

    def get(self, i):
        return int(self.u32[16 * i])

    def set(self, i, v):
        self.u32[16 * i] = v & 0xFFFFFFFF

    def data(self, which, dtype = torch.int32):
        """Torch view of data area `which` (0, 1), pinned through the registration"""
        return torch.frombuffer(self.shm.buf, dtype = dtype, count = DATA_BYTES // torch.tensor([], dtype = dtype).element_size(),
                                offset = DATA_OFF + which * DATA_BYTES)

    def close(self):
        torch.cuda.synchronize()
        cuda_host_unregister(self.base)
        del self.u8, self.u32
        try:
            self.shm.close()
        except BufferError:
            pass
        self.shm.unlink()


@pytest.fixture(params = [True, False], ids = ["memops", "kernels"])
def memops(request):
    ext.exl3_moe_cpu_set_memops(request.param)
    yield request.param
    ext.exl3_moe_cpu_set_memops(os.environ.get("EXL3_MOE_MEMOPS", "1") != "0")


@pytest.fixture
def region(device):
    r = Region()
    yield r
    # Unblock any wait a failing test left on the stream before tearing down
    for i in range(8):
        r.set(i, 0x7FFFFFFF)
    r.close()


def wait_event(ev: torch.cuda.Event, timeout = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not ev.query():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.001)
    return True


def test_write_stream_ordered(device, memops, region):
    # Warm-up: under lazy module loading the first launch of the fallback kernel blocks the host until the device
    # is idle, which would hide whether the write itself waits for the stream
    ext.exl3_moe_flag_write(region.addr(4), 1)
    ext.exl3_moe_flag_wait(region.addr(4), 1, region.addr(7))
    torch.cuda.synchronize()
    torch.cuda._sleep(SLEEP_CYCLES)
    ext.exl3_moe_flag_write(region.addr(1), 7)
    early = region.get(1)
    torch.cuda.synchronize()
    assert early == 0, "flag written before the stream's earlier work finished"
    assert region.get(1) == 7
    ext.exl3_moe_flag_write(region.addr(1), (1 << 32) + 5)
    ext.exl3_moe_flag_write(region.addr(2), -1)
    torch.cuda.synchronize()
    assert region.get(1) == 5 and region.get(2) == 0xFFFFFFFF
    # Untouched neighbours
    assert region.get(0) == 0 and region.get(3) == 0 and region.get(5) == 0


def test_write_follows_current_stream(device, memops, region):
    side = torch.cuda.Stream(device)
    torch.cuda._sleep(SLEEP_CYCLES * 5)   # default stream busy
    with torch.cuda.stream(side):
        ext.exl3_moe_flag_write(region.addr(0), 3)
    side.synchronize()
    seen = region.get(0)
    torch.cuda.synchronize()
    assert seen == 3, "flag write on a side stream waited for the default stream"


@pytest.mark.parametrize("flag, value, satisfied", [
    (5, 5, True), (9, 5, True), (4, 5, False), (3, 0xFFFFFFF0, True), (3, -16, True), (0x80000005, 5, False),
    (0, 0x80000001, True), (5, (1 << 32) + 5, True),
], ids = ["equal", "above", "below", "wrapped", "wrapped_negative_arg", "half_ring_behind", "half_ring_ahead",
          "value_mod_2_32"])
def test_wait_predicate(device, memops, region, flag, value, satisfied):
    region.set(0, flag)
    ext.exl3_moe_flag_wait(region.addr(0), value, region.addr(7))
    ev = torch.cuda.Event()
    ev.record()
    if satisfied:
        assert wait_event(ev), f"wait({value:#x}) did not pass with the flag at {flag:#x}"
    else:
        time.sleep(0.2)
        assert not ev.query(), f"wait({value:#x}) passed with the flag at {flag:#x}"
        region.set(0, value)
        assert wait_event(ev), "wait did not pass once the flag reached the value"
    assert region.get(7) == 0, "abort flag set"


def test_wait_orders_later_copies(device, memops, region):
    """Data the host writes before raising the flag is what the stream reads after the wait"""
    src = region.data(0)
    dst = torch.zeros(src.numel(), dtype = torch.int32, device = device)
    ext.exl3_moe_flag_wait(region.addr(0), 1, region.addr(7))
    dst.copy_(src, non_blocking = True)
    ev = torch.cuda.Event()
    ev.record()
    time.sleep(0.2)
    assert not ev.query()
    src.copy_(torch.arange(src.numel(), dtype = torch.int32))
    region.set(0, 1)
    assert wait_event(ev)
    assert torch.equal(dst.cpu(), torch.arange(src.numel(), dtype = torch.int32))


ROUNDS = 200


def gpu_side(region, device):
    """Per round i: data_in := i (D2H), ping = i; wait pong >= i; out[i] := data_out (H2D). All enqueued at once"""
    n = region.data(0).numel()
    outs = torch.zeros(ROUNDS, n, dtype = torch.int32, device = device)
    src = torch.arange(n, dtype = torch.int32, device = device)
    for i in range(1, ROUNDS + 1):
        region.data(0).copy_(src + i, non_blocking = True)
        ext.exl3_moe_flag_write(region.addr(0), i)
        ext.exl3_moe_flag_wait(region.addr(1), i, region.addr(7))
        outs[i - 1].copy_(region.data(1), non_blocking = True)
    return outs


def host_side(u32_in, u32_out, ping, pong, rounds, deadline_s = 60.0):
    """Per round i: wait ping >= i; check data_in == arange + i; data_out := 3 * data_in; pong = i. Returns the
    rounds whose input differed"""
    bad = []
    base = np.arange(u32_in.size, dtype = np.uint32)
    deadline = time.monotonic() + deadline_s
    for i in range(1, rounds + 1):
        while ((int(ping[0]) - i) & 0xFFFFFFFF) >= 0x80000000:
            if time.monotonic() > deadline:
                pong[0] = 0x7FFFFFFF
                return bad + ["timeout"]
            time.sleep(0)
        if not np.array_equal(u32_in, base + i):
            bad.append(i)
        u32_out[:] = u32_in * 3
        pong[0] = i
    return bad


def check_outs(outs, n):
    want = (torch.arange(n, dtype = torch.int32).unsqueeze(0) + torch.arange(1, ROUNDS + 1, dtype = torch.int32).unsqueeze(1)) * 3
    return torch.equal(outs.cpu(), want)


def views(buf):
    u32 = np.ndarray((len(buf) // 4,), dtype = np.uint32, buffer = buf)
    n = DATA_BYTES // 4
    return (u32[DATA_OFF // 4 : DATA_OFF // 4 + n], u32[DATA_OFF // 4 + n : DATA_OFF // 4 + 2 * n],
            u32[0:1], u32[16:17])


def test_handshake_with_thread(device, memops, region):
    d_in, d_out, ping, pong = views(region.shm.buf)
    result = []
    t = threading.Thread(target = lambda: result.append(host_side(d_in, d_out, ping, pong, ROUNDS)), daemon = True)
    t.start()
    outs = gpu_side(region, device)
    torch.cuda.synchronize()
    t.join(10)
    assert result == [[]], f"rounds where the host saw stale data: {result}"
    assert check_outs(outs, d_in.size), "the stream read back stale host data"


_CHILD = """
import sys
from multiprocessing import resource_tracker, shared_memory
resource_tracker.register = resource_tracker.unregister = lambda *a, **k: None
sys.path.insert(0, sys.argv[3])
from test_moe_flags import host_side, views
shm = shared_memory.SharedMemory(name = sys.argv[1])
bad = host_side(*views(shm.buf), int(sys.argv[2]))
print("BAD", bad, flush = True)
sys.exit(1 if bad else 0)
"""


def test_handshake_with_process(device, memops, region):
    """The counterpart in another process, polling and writing the shared words as the CPU worker does"""
    here = os.path.dirname(os.path.abspath(__file__))
    p = subprocess.Popen([sys.executable, "-c", _CHILD, region.shm.name, str(ROUNDS), here],
                         stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True,
                         env = dict(os.environ, CUDA_VISIBLE_DEVICES = ""))
    try:
        outs = gpu_side(region, device)
        torch.cuda.synchronize()
        out, _ = p.communicate(timeout = 60)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
    assert p.returncode == 0, out
    assert check_outs(outs, DATA_BYTES // 4), "the stream read back stale host data"


@pytest.mark.slow
def test_kernel_wait_timeout_sets_abort(device, region):
    ext.exl3_moe_cpu_set_memops(False)
    try:
        ext.exl3_moe_flag_wait(region.addr(0), 1, region.addr(7))
        ev = torch.cuda.Event()
        ev.record()
        assert wait_event(ev, timeout = 90), "fallback wait kernel did not time out"
        assert region.get(7) == 1, "abort flag not set on timeout"
        assert region.get(0) == 0
    finally:
        ext.exl3_moe_cpu_set_memops(os.environ.get("EXL3_MOE_MEMOPS", "1") != "0")
