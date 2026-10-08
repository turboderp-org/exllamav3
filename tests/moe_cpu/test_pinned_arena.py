"""
Pinned expert arena (EXL3_MOE_PINNED_ARENA), Linux backend: memfd-backed chunks published over a pipe with the
descriptor attached, mapped on the receiving side and seeing the same bytes (checked through a second mapping);
contiguous expert block reservation; the memfd_create fallbacks for interpreters without os.memfd_create
(libc symbol, raw syscall), checked against os.memfd_create's constants and behaviour. No GPU needed.
"""

import mmap
import multiprocessing
import os
import socket

import pytest
import torch

from exllamav3.model import moe_cpu_host as m

pytestmark = [pytest.mark.nogpu, pytest.mark.platform("linux")]


def _map_published(parent):
    msg = parent.recv()
    assert msg[0] == "chunk"
    with socket.socket(fileno = os.dup(parent.fileno())) as sock:
        _, fds, _, _ = socket.recv_fds(sock, 1, 1)
    fd = fds[0]
    mapping = mmap.mmap(fd, msg[2], mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    os.close(fd)
    return msg[1], mapping


def test_shared_arena_publishes_and_shares_pages():
    parent, child = multiprocessing.get_context("spawn").Pipe(duplex = True)
    arena = m._HugeArena(shared = True, conn = child)
    arena.CHUNK_BYTES = 4 << 20
    t = torch.arange(0, 1024, dtype = torch.int16).view(4, 16, 16)
    v = arena.rehome(t)
    idx, mapping = _map_published(parent)
    assert idx == 0 and len(mapping) == arena.CHUNK_BYTES
    # Same pages: the mapping on the receiving side sees what rehome wrote, and later writes
    other = torch.frombuffer(mapping, dtype = torch.int16)[: t.numel()].view(t.shape)
    assert torch.equal(other, t)
    v.fill_(7)
    assert int(other[0, 0, 0]) == 7
    mapping.close()
    parent.close()
    child.close()


def test_reserve_places_experts_contiguously():
    arena = m._HugeArena()
    arena.CHUNK_BYTES = 4 << 20
    g = torch.full((2, 8, 64), 1, dtype = torch.int16)   # 2 KiB
    u = torch.full((2, 8, 64), 2, dtype = torch.int16)
    d = torch.full((8, 2, 64), 3, dtype = torch.int16)
    nb = g.numel() * 2
    ci, off = arena.reserve(3 * nb)
    for t in (g, u, d):
        arena.rehome(t)
    buf = torch.frombuffer(arena.chunks[ci], dtype = torch.int16)[off // 2 : off // 2 + 3 * nb // 2]
    assert torch.equal(buf[: nb // 2], torch.full((nb // 2,), 1, dtype = torch.int16))
    assert torch.equal(buf[nb // 2 : nb], torch.full((nb // 2,), 2, dtype = torch.int16))
    assert torch.equal(buf[nb:], torch.full((nb // 2,), 3, dtype = torch.int16))
    # A reservation that does not fit the current chunk opens a new one
    ci2, off2 = arena.reserve(arena.CHUNK_BYTES - 64)
    assert ci2 == 1 and off2 == 0


def test_hugetlb_request_fails_cleanly_without_reservation():
    try:
        with open("/proc/sys/vm/nr_hugepages") as f:
            free = int(f.read())
    except OSError:
        pytest.skip("no hugetlb sysctl")
    if free:
        pytest.skip("hugepages are reserved on this host")
    arena = m._HugeArena(shared = True, huge = "2m")
    with pytest.raises(RuntimeError, match = "hugetlb"):
        arena.reserve(64)


# --- memfd_create fallbacks (PR #341: conda / manylinux interpreters lack os.memfd_create) ---

def _check_memfd(fd):
    """A usable memfd: sized, mappable shared, readable back through a second mapping"""
    os.ftruncate(fd, 1 << 20)
    a = mmap.mmap(fd, 1 << 20, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    b = mmap.mmap(fd, 1 << 20, mmap.MAP_SHARED, mmap.PROT_READ)
    a[:4] = b"exl3"
    assert bytes(b[:4]) == b"exl3"
    a.close(); b.close(); os.close(fd)


def test_memfd_fallbacks_match_os_memfd_create():
    assert m.MFD_HUGETLB == getattr(os, "MFD_HUGETLB", m.MFD_HUGETLB)
    assert m.MFD_HUGE_2MB == getattr(os, "MFD_HUGE_2MB", m.MFD_HUGE_2MB)
    assert m.MFD_HUGE_1GB == getattr(os, "MFD_HUGE_1GB", m.MFD_HUGE_1GB)
    _check_memfd(m._memfd_via_libc("t", 0))
    _check_memfd(m._memfd_via_syscall("t", 0))
    _check_memfd(m._memfd_create("t", m.MFD_CLOEXEC))
    # errors surface as OSError with the kernel's errno, not as a negative descriptor
    with pytest.raises(OSError):
        m._memfd_via_syscall("t", 0xffff)   # invalid flags -> EINVAL
    with pytest.raises(OSError):
        m._memfd_via_libc("t", 0xffff)


def test_shared_arena_without_os_memfd_create(monkeypatch):
    """The whole publish path on an interpreter without os.memfd_create: libc symbol, then the
    raw syscall, then a clear error naming the switch"""
    monkeypatch.delattr(os, "memfd_create", raising = False)
    for c in ("MFD_CLOEXEC", "MFD_HUGETLB", "MFD_HUGE_2MB", "MFD_HUGE_1GB"):
        monkeypatch.delattr(os, c, raising = False)
    monkeypatch.setenv("EXL3_HOST_MEM_RESERVE_MB", "0")
    monkeypatch.setattr(m._HugeArena, "CHUNK_BYTES", 4 << 20)

    def publish():
        parent, child = multiprocessing.get_context("spawn").Pipe(duplex = True)
        arena = m._HugeArena(shared = True, conn = child)
        arena._new_chunk(1)
        arena.cur[:4] = b"exl3"
        _, view = _map_published(parent)
        assert bytes(view[:4]) == b"exl3"
        view.close(); arena.cur.close()

    publish()                                                   # libc symbol
    def no_symbol(n, f):
        raise AttributeError("no symbol")
    monkeypatch.setattr(m, "_memfd_via_libc", no_symbol)
    publish()                                                   # raw syscall
    monkeypatch.setattr(m, "_MEMFD_SYSCALL", {})
    with pytest.raises(RuntimeError, match = "EXL3_MOE_PINNED_ARENA"):
        publish()                                               # neither
