# CPU MoE offload host-side bookkeeping, model-free: the shared expert arena's /dev/shm
# preflight and name release, and the streamed-prefill profiler's per-pass normalization.

import os
import pytest
from multiprocessing import shared_memory
from exllamav3.model.moe_cpu_host import _SharedArena, _stream_prof_line


class _Statvfs:
    def __init__(self, avail):
        self.f_bavail = avail // 4096
        self.f_frsize = 4096


def test_arena_preflight_rejects_small_shm(monkeypatch):
    monkeypatch.setattr(os, "statvfs", lambda path: _Statvfs(64 << 20), raising = False)
    arena = _SharedArena()
    with pytest.raises(RuntimeError, match = r"/dev/shm.*shm-size"):
        arena.reserve(1)
    assert arena.chunks == []


def test_arena_preflight_accepts_large_shm(monkeypatch):
    monkeypatch.setattr(os, "statvfs", lambda path: _Statvfs(1 << 40), raising = False)
    arena = _SharedArena()
    arena.reserve(1)
    assert len(arena.chunks) == 1
    for c in arena.chunks:
        c.close()
        c.unlink()


@pytest.mark.skipif(os.name == "nt", reason = "shared memory names are not persistent on Windows")
def test_arena_unlink_releases_names():
    arena = _SharedArena()
    arena.reserve(1)
    name = arena.chunks[0].name
    shared_memory.SharedMemory(name = name).close()
    arena.unlink()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name = name)


def _prof(**kw):
    pr = dict(n = 0, sync = 0.0, host = 0.0, gpu = 0.0, gpu_n = 0, batches = 0,
              stagewait = 0.0, dma = 0.0, compute = 0.0, ev = None, pending = [])
    pr.update(kw)
    return pr


def test_stream_prof_line_single_layer_no_gpu_samples():
    line = _stream_prof_line(_prof(n = 1, sync = 0.001, host = 0.002, batches = 3), 1, 4096)
    assert "gpu-span 0.00" in line
    assert "batches/layer 3.0" in line


def test_stream_prof_line_normalizes_by_harvested_events():
    pr = _prof(n = 4, gpu = 60.0, gpu_n = 3, stagewait = 3.0, dma = 30.0, compute = 15.0)
    line = _stream_prof_line(pr, 4, 4096)
    assert "gpu-span 20.00" in line
    assert "stage-wait 1.00" in line
    assert "dma 10.00" in line
    assert "compute 5.00" in line
