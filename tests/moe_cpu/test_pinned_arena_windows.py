"""
Windows backend of the CPU MoE pinned arena (EXL3_MOE_PINNED_ARENA=1): the worker's chunks are named
pagefile-backed sections the parent opens by name and page-locks, published over the worker pipe as
("chunk", index, size, name); Linux keeps memfd + SCM_RIGHTS with name = None. The MEM_LARGE_PAGES size
negotiation of private chunks runs on every platform against a recording kernel32 stand-in (the Windows branch
of _new_chunk is selected by os.name); the named-section tests need Windows.
"""

import ctypes
import mmap
import multiprocessing
import os
import socket
import subprocess
import sys
from multiprocessing import shared_memory
from types import SimpleNamespace

import pytest
import torch

from exllamav3.model import moe_cpu_host as m
from testlib.env import cuda_available

pytestmark = pytest.mark.nogpu

MiB = 1 << 20


class _CapturePipe:
    def __init__(self):
        self.messages = []

    def send(self, msg):
        self.messages.append(msg)


def _fresh_interpreter(code, *flags, **env):
    return subprocess.run([sys.executable, "-B", *flags, "-c", code], capture_output = True,
                          text = True, env = {**os.environ, **env})


@pytest.mark.platform("windows")
def test_windows_memory_status_reports_physical_and_commit_headroom():
    from exllamav3.util.memory import windows_memory_status
    avail_phys, avail_commit = windows_memory_status()
    assert 0 < avail_phys
    assert 0 < avail_commit


def test_pinned_arena_flag_is_honoured_on_this_platform():
    r = _fresh_interpreter("from exllamav3.model.moe_cpu_host import TUNING; print(TUNING.pinned_arena)",
                           EXL3_MOE_PINNED_ARENA = "1", EXL3_MOE_ARENA_HUGE = "")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "True"


@pytest.mark.platform("windows")
@pytest.mark.parametrize("huge", ["2m", "1g"])
@pytest.mark.parametrize("flags", [(), ("-O",)], ids = ["normal", "optimized"])
def test_windows_rejects_arena_huge(huge, flags):
    r = _fresh_interpreter("import exllamav3.model.moe_cpu_host", *flags,
                           EXL3_MOE_PINNED_ARENA = "1", EXL3_MOE_ARENA_HUGE = huge)
    assert r.returncode != 0
    assert "RuntimeError" in r.stderr and "EXL3_MOE_ARENA_HUGE" in r.stderr


def _small_arena(monkeypatch, module, conn = None):
    monkeypatch.setenv("EXL3_HOST_MEM_RESERVE_MB", "0")
    monkeypatch.setattr(module._HugeArena, "CHUNK_BYTES", 4 * MiB)
    return module._HugeArena(shared = True, conn = conn)


def _small_private_arena(monkeypatch, module):
    monkeypatch.setenv("EXL3_HOST_MEM_RESERVE_MB", "0")
    monkeypatch.setattr(module._HugeArena, "CHUNK_BYTES", 4 * MiB)
    return module._HugeArena()


@pytest.mark.platform("windows")
def test_windows_chunk_is_a_named_section_the_parent_can_open(monkeypatch):
    pipe = _CapturePipe()
    arena = _small_arena(monkeypatch, m, pipe)
    arena._new_chunk(1)
    (kind, index, size, name), = pipe.messages
    assert (kind, index, size, name) == ("chunk", 0, 4 * MiB, f"exl3_moe_arena_{os.getpid()}_0")
    arena.cur[:4] = b"exl3"
    section = shared_memory.SharedMemory(name = name)
    assert bytes(section.buf[:4]) == b"exl3"
    section.close()
    arena.cur.close()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name = name)


@pytest.mark.platform("windows")
@pytest.mark.parametrize("status, limit", [((1 * MiB, 1 << 40), "physical RAM"),
                                           ((1 << 40, 1 * MiB), "commit")])
def test_windows_chunk_blows_the_fuse_before_creating_a_section(monkeypatch, status, limit):
    monkeypatch.setattr(m, "windows_memory_status", lambda: status)
    arena = _small_arena(monkeypatch, m, _CapturePipe())
    with pytest.raises(RuntimeError, match = limit):
        arena._new_chunk(1)
    assert arena.chunks == [] and arena.conn.messages == []
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name = f"exl3_moe_arena_{os.getpid()}_0")


@pytest.mark.platform("windows")
def test_windows_section_creation_failure_names_the_chunk_and_the_switch(monkeypatch):
    def refuse(fileno, length, *a, **k):
        raise OSError(1455, "The paging file is too small for this operation to complete")
    monkeypatch.setattr(mmap, "mmap", refuse)
    arena = _small_arena(monkeypatch, m, _CapturePipe())
    with pytest.raises(RuntimeError, match = "chunk 0.*paging file.*EXL3_MOE_PINNED_ARENA"):
        arena._new_chunk(1)
    assert arena.chunks == [] and arena.conn.messages == []


@pytest.mark.platform("windows")
def test_private_chunk_falls_back_to_mmap_when_large_pages_unavailable(monkeypatch):
    monkeypatch.setattr(m, "_win32_large_page_alloc", lambda size, min_size: None)
    monkeypatch.setattr(m.TUNING, "arena_hugepage", True)
    arena = _small_private_arena(monkeypatch, m)
    arena._new_chunk(1)
    assert isinstance(arena.cur, mmap.mmap) and arena.win32_large_bytes == 0
    arena.cur.close()


@pytest.mark.platform("windows")
def test_private_chunk_uses_the_large_page_buffer_when_virtualalloc_succeeds(monkeypatch):
    made = (ctypes.c_uint8 * (4 * MiB))()
    monkeypatch.setattr(m, "_win32_large_page_alloc", lambda size, min_size: made)
    monkeypatch.setattr(m.TUNING, "arena_hugepage", True)
    arena = _small_private_arena(monkeypatch, m)
    arena._new_chunk(1)
    assert arena.cur is made and arena.win32_large_bytes == len(made)
    arena.cur[:4] = b"exl3"                    # the buffer-protocol write path rehome() uses
    assert bytes(memoryview(arena.cur)[:4]) == b"exl3"


@pytest.mark.platform("windows")
def test_shared_chunk_never_takes_the_large_page_path(monkeypatch):
    # Ordering invariant: a MEM_LARGE_PAGES VirtualAlloc chunk cannot be opened by name by
    # the parent, so the pinned/shared path must stay on named pagefile sections
    def boom(size, min_size):
        raise AssertionError("_win32_large_page_alloc must not run for shared arenas")
    monkeypatch.setattr(m, "_win32_large_page_alloc", boom)
    monkeypatch.setattr(m.TUNING, "arena_hugepage", True)
    arena = _small_arena(monkeypatch, m, _CapturePipe())
    arena._new_chunk(1)
    assert isinstance(arena.cur, mmap.mmap) and arena.win32_large_bytes == 0
    arena.cur.close()


@pytest.mark.platform("windows")
def test_private_chunk_skips_large_pages_when_the_flag_is_off(monkeypatch):
    def boom(size, min_size):
        raise AssertionError("_win32_large_page_alloc must not run with HUGEPAGE=0")
    monkeypatch.setattr(m, "_win32_large_page_alloc", boom)
    monkeypatch.setattr(m.TUNING, "arena_hugepage", False)
    arena = _small_private_arena(monkeypatch, m)
    arena._new_chunk(1)
    assert isinstance(arena.cur, mmap.mmap) and arena.win32_large_bytes == 0
    arena.cur.close()


@pytest.mark.platform("linux")
def test_linux_chunk_message_carries_no_name_and_one_descriptor(monkeypatch):
    parent, child = multiprocessing.get_context("spawn").Pipe(duplex = True)
    arena = _small_arena(monkeypatch, m, child)
    arena._new_chunk(1)
    assert parent.recv() == ("chunk", 0, 4 * MiB, None)
    with socket.socket(fileno = os.dup(parent.fileno())) as sock:
        _, fds, _, _ = socket.recv_fds(sock, 1, 1)
    assert len(fds) == 1
    os.close(fds[0])
    arena.cur.close()


def _host(module):
    return module.MoeCpuHost(SimpleNamespace(directory = None, infer_params = SimpleNamespace()))


@pytest.mark.platform("windows")
@pytest.mark.skipif(not cuda_available(), reason = "needs a CUDA device")
def test_parent_registers_named_section_for_dma_and_releases_it_on_shutdown(monkeypatch, device):
    pipe = _CapturePipe()
    arena = _small_arena(monkeypatch, m, pipe)
    arena._new_chunk(1)
    _, index, size, name = pipe.messages[0]
    arena.cur[:2] = (1234).to_bytes(2, "little")

    host = _host(m)
    host._attach_chunk(index, size, name)
    assert len(host.arena_maps) == len(host.arena_views) == 1
    assert host.arena_views[0].numel() == size // 2
    assert int(host.arena_views[0][0]) == 1234

    # DMA straight out of the registered section on a side stream, as streamed prefill does
    dev = torch.empty(size // 2, dtype = torch.int16, device = device)
    s = torch.cuda.Stream(device)
    with torch.cuda.stream(s):
        dev.copy_(host.arena_views[0], non_blocking = True)
    s.synchronize()
    assert int(dev[0]) == 1234

    host.shutdown()
    assert host.arena_maps == [] and host.arena_views == []
    arena.cur.close()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name = name)


@pytest.mark.platform("windows")
def test_short_section_is_closed_even_while_the_traceback_is_held(monkeypatch):
    pipe = _CapturePipe()
    arena = _small_arena(monkeypatch, m, pipe)
    arena._new_chunk(1)
    _, index, size, name = pipe.messages[0]
    host = _host(m)
    with pytest.raises(RuntimeError, match = f"chunk {index}") as excinfo:
        host._attach_chunk(index, size * 2, name)    # advertised twice the real section
    assert host.arena_maps == [] and host.arena_views == []
    arena.cur.close()
    with pytest.raises(FileNotFoundError):            # excinfo (and its frames) still alive here
        shared_memory.SharedMemory(name = name)
    del excinfo


@pytest.mark.platform("windows")
def test_registration_failure_closes_the_section_while_the_traceback_is_held(monkeypatch):
    pipe = _CapturePipe()
    arena = _small_arena(monkeypatch, m, pipe)
    arena._new_chunk(1)
    _, index, size, name = pipe.messages[0]
    def refuse(ptr, n, flags):
        raise RuntimeError("cudaHostRegister injected failure")
    monkeypatch.setattr(m, "cuda_host_register", refuse)
    host = _host(m)
    with pytest.raises(RuntimeError, match = "injected failure.*EXL3_MOE_PINNED_ARENA") as excinfo:
        host._attach_chunk(index, size, name)
    assert host.arena_maps == [] and host.arena_views == []
    arena.cur.close()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name = name)
    del excinfo


class _FakeKernel32:
    """kernel32 stand-in for _win32_large_page_alloc: honours VirtualAlloc only up to `max_ok`
    bytes and records every attempted size, so the size-negotiation ladder is observable.
    Functions are wrapped so the restype/argtypes assignments the caller does succeed."""
    class _Func:
        def __init__(self, fn):
            self.fn = fn
            self.restype = None
            self.argtypes = None
        def __call__(self, *a):
            return self.fn(*a)

    def __init__(self, max_ok, granularity = 2 * MiB):
        self.max_ok = max_ok
        self.attempts = []
        self.buffers = []
        self.GetLargePageMinimum = self._Func(lambda *a: granularity)
        self.VirtualFree = self._Func(lambda *a: None)
        self.VirtualAlloc = self._Func(self._virtual_alloc)
        self._ctypes = ctypes

    def _virtual_alloc(self, addr, size, flags, protect):
        self.attempts.append(size)
        if size > self.max_ok:
            return None
        self.buffers.append((self._ctypes.c_uint8 * size)())
        return self._ctypes.addressof(self.buffers[-1])


def _fake_kernel32(monkeypatch, module, max_ok):
    k32 = _FakeKernel32(max_ok)
    monkeypatch.setattr("ctypes.WinDLL", lambda *a, **k: k32, raising = False)
    monkeypatch.setattr(module, "_WIN32_LARGE_PAGE_SUPPORT", True)
    return k32


@pytest.mark.platform("windows")
def test_large_page_alloc_halves_the_request_until_one_succeeds(monkeypatch):
    k32 = _fake_kernel32(monkeypatch, m, max_ok = 128 * MiB)
    buf = m._win32_large_page_alloc(1 << 30, 64 * MiB)
    assert k32.attempts == [1 << 30, 512 * MiB, 256 * MiB, 128 * MiB]
    assert buf is not None and len(buf) == 128 * MiB
    buf[:4] = b"exl3"
    assert bytes(buf[:4]) == b"exl3"


@pytest.mark.platform("windows")
def test_large_page_alloc_stops_at_min_size_and_returns_none(monkeypatch):
    k32 = _fake_kernel32(monkeypatch, m, max_ok = 0)
    assert m._win32_large_page_alloc(1 << 30, 64 * MiB) is None
    assert k32.attempts == [1 << 30, 512 * MiB, 256 * MiB, 128 * MiB, 64 * MiB]


@pytest.mark.platform("windows")
def test_large_page_alloc_still_tries_an_unaligned_floor(monkeypatch):
    # A floor that is not a power-of-two halving step is itself a valid (granularity-aligned)
    # request, so it gets one attempt before giving up
    k32 = _fake_kernel32(monkeypatch, m, max_ok = 0)
    assert m._win32_large_page_alloc(1 << 30, 300 * MiB) is None
    assert k32.attempts == [1 << 30, 512 * MiB, 300 * MiB]


@pytest.mark.platform("windows")
def test_private_chunk_floors_the_large_page_floor_at_64mib_or_the_need(monkeypatch):
    # The chunk bounds the negotiated floor at 64 MiB, raised to the size it must still hold
    floors = []
    def capture(size, min_size):
        floors.append(min_size)
        return None
    monkeypatch.setattr(m, "_win32_large_page_alloc", capture)
    monkeypatch.setattr(m.TUNING, "arena_hugepage", True)
    arena = _small_private_arena(monkeypatch, m)
    arena._new_chunk(1)
    arena2 = _small_private_arena(monkeypatch, m)
    arena2._new_chunk(100 * MiB)
    assert floors == [64 * MiB, 100 * MiB]
    arena.cur.close()
    arena2.cur.close()


@pytest.mark.platform("windows")
def test_private_chunk_keeps_full_size_when_falling_back_to_mmap(monkeypatch):
    # The plain-mapping fallback has no contiguity constraint, so it keeps the full chunk size
    sizes = []
    real_mmap = mmap.mmap
    def spy(fileno, length, *a, **k):
        sizes.append(length)
        return real_mmap(fileno, length, *a, **k)
    monkeypatch.setattr(m, "_win32_large_page_alloc", lambda size, min_size: None)
    monkeypatch.setattr(mmap, "mmap", spy)
    monkeypatch.setattr(m.TUNING, "arena_hugepage", True)
    arena = _small_private_arena(monkeypatch, m)
    arena._new_chunk(1)
    assert sizes == [4 * MiB]
    arena.cur.close()


def _windows_private_arena(monkeypatch, module, max_ok):
    """Private arena on the Windows branch of _new_chunk with a fake kernel32, on any platform:
    the branch is selected by os.name and only needs an anonymous mmap besides the allocator"""
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(module, "check_host_memory", lambda *a, **k: None)
    monkeypatch.setattr(module.TUNING, "arena_hugepage", True)
    return module._HugeArena(), _fake_kernel32(monkeypatch, module, max_ok)


def test_large_page_ladder_resumes_at_the_size_the_last_chunk_was_served_at(monkeypatch):
    arena, k32 = _windows_private_arena(monkeypatch, m, max_ok = 256 * MiB)
    arena._new_chunk(1)
    assert k32.attempts == [1 << 30, 512 * MiB, 256 * MiB]
    arena._new_chunk(1)
    arena._new_chunk(1)
    assert k32.attempts[3:] == [256 * MiB, 256 * MiB]
    assert [len(c) for c in arena.chunks] == [256 * MiB] * 3
    assert arena.win32_large_bytes == 3 * 256 * MiB


def test_large_page_attempts_stop_once_the_smallest_size_has_failed(monkeypatch):
    arena, k32 = _windows_private_arena(monkeypatch, m, max_ok = 0)
    arena._new_chunk(1)
    assert k32.attempts == [1 << 30, 512 * MiB, 256 * MiB, 128 * MiB, 64 * MiB]
    arena._new_chunk(1)
    assert len(k32.attempts) == 5
    # Regular pages have no contiguity constraint, so the fallback keeps the full chunk size
    assert all(isinstance(c, mmap.mmap) and len(c) == 1 << 30 for c in arena.chunks)
    assert arena.win32_large_bytes == 0
    for c in arena.chunks:
        c.close()


def test_large_page_ladder_follows_the_supply_down_and_skips_placements_above_it(monkeypatch):
    arena, k32 = _windows_private_arena(monkeypatch, m, max_ok = 512 * MiB)
    arena._new_chunk(1)
    assert k32.attempts == [1 << 30, 512 * MiB]
    # Supply shrinks mid-load: the next chunk starts at the remembered size and steps down
    k32.max_ok = 128 * MiB
    arena._new_chunk(1)
    assert k32.attempts[2:] == [512 * MiB, 256 * MiB, 128 * MiB]
    # A placement that needs more than any size that still works goes straight to a plain
    # mapping, and does not disturb what is remembered for the chunks after it
    arena._new_chunk(200 * MiB)
    assert len(k32.attempts) == 5 and isinstance(arena.cur, mmap.mmap)
    arena._new_chunk(1)
    assert k32.attempts[5:] == [128 * MiB]
    arena.chunks[2].close()


@pytest.mark.parametrize("max_ok, expected", [
    (1 << 30, "on large pages (locked in RAM"),
    (0, "no large pages could be allocated"),
])
def test_worker_reports_large_page_use_without_the_debug_flag(monkeypatch, capsys, max_ok, expected):
    monkeypatch.delenv("EXL3_MOE_ARENA_DEBUG", raising = False)
    arena, _ = _windows_private_arena(monkeypatch, m, max_ok)
    arena._new_chunk(1)
    arena.promote_hugepages()
    out = capsys.readouterr().out
    assert expected in out and out.count("\n") == 1
    if max_ok == 0:
        arena.cur.close()


def test_worker_stays_silent_when_the_account_lacks_the_privilege(monkeypatch, capsys):
    monkeypatch.delenv("EXL3_MOE_ARENA_DEBUG", raising = False)
    arena, _ = _windows_private_arena(monkeypatch, m, max_ok = 1 << 30)
    monkeypatch.setattr(m, "_WIN32_LARGE_PAGE_SUPPORT", False)
    arena._new_chunk(1)
    arena.promote_hugepages()
    assert capsys.readouterr().out == "" and arena.win32_large_bytes == 0
    arena.cur.close()
