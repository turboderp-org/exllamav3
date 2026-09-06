# CPU MoE offload host-side bookkeeping, model-free: the shared expert arena's /dev/shm
# preflight and name release, and the streamed-prefill profiler's per-pass normalization.

import os
import types
import multiprocessing
import pytest
from multiprocessing import shared_memory
from exllamav3.model.moe_cpu_host import (
    MoeCpuHost, TUNING, _SharedArena, _stream_prof_line, _proj_swizzled, _stream_per_slot,
)


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
              rawwait = 0.0, dma = 0.0, compute = 0.0, ev = None, pending = [])
    pr.update(kw)
    return pr


def test_stream_prof_line_single_layer_no_gpu_samples():
    line = _stream_prof_line(_prof(n = 1, sync = 0.001, host = 0.002, batches = 3), 1, 4096)
    assert "gpu-span 0.00" in line
    assert "batches/layer 3.0" in line


def test_stream_prof_line_normalizes_by_harvested_events():
    pr = _prof(n = 4, gpu = 60.0, gpu_n = 3, rawwait = 3.0, dma = 30.0, compute = 15.0)
    line = _stream_prof_line(pr, 4, 4096)
    assert "gpu-span 20.00" in line
    assert "raw-slot-wait 1.00" in line
    assert "dma 10.00" in line
    assert "compute 5.00" in line


# ---- layout contract: the child decides the physical trellis order once and reports it ----

def test_proj_swizzled_rule():
    assert _proj_swizzled(True, 4) is True
    assert _proj_swizzled(True, 8) is False
    assert _proj_swizzled(False, 4) is False


def _config():
    return types.SimpleNamespace(directory = "no-such-model", infer_params = types.SimpleNamespace())


def test_host_snapshots_swizzle_at_construction(monkeypatch):
    monkeypatch.setattr(TUNING, "swizzle", False)
    host = MoeCpuHost(_config())
    monkeypatch.setattr(TUNING, "swizzle", True)
    assert host.swizzle is False
    assert MoeCpuHost(_config()).swizzle is True


class _FakeProcess:
    def __init__(self, target, args, daemon):
        self.target, self.args, self.daemon = target, args, daemon
        self.alive = True
    def start(self):
        pass
    def is_alive(self):
        return self.alive
    def join(self, timeout = None):
        self.alive = False
    def terminate(self):
        self.alive = False
    def kill(self):
        self.alive = False


class _FakeContext:
    def __init__(self):
        self.proc = None
    def Pipe(self, duplex):
        return multiprocessing.Pipe(duplex = duplex)
    def Process(self, target, args, daemon):
        self.proc = _FakeProcess(target, args, daemon)
        return self.proc


def test_spawn_passes_swizzle_snapshot_to_child(monkeypatch):
    monkeypatch.setattr(TUNING, "swizzle", False)
    host = MoeCpuHost(_config())
    ctx = _FakeContext()
    monkeypatch.setattr(multiprocessing, "get_context", lambda method: ctx)
    host._spawn()
    try:
        assert ctx.proc.args[1:] == (host.model_dir, host.threads, False)
    finally:
        host.shutdown()
    # A never-started host still joins its worker on shutdown
    assert ctx.proc.alive is False


# ---- profiler batch count: the loop batches by slot capacity, not batch_experts alone ----

def test_stream_per_slot_is_capacity_limited():
    assert _stream_per_slot(wslot_size = 3 * 1000, exp_b = 1000, batch_experts = 24) == 3
    assert _stream_per_slot(wslot_size = 100 * 1000, exp_b = 1000, batch_experts = 24) == 24


# ---- arena ownership: chunk names reach the parent as soon as they exist ----

def test_arena_publishes_chunk_names_on_creation(monkeypatch):
    monkeypatch.setattr(os, "statvfs", lambda path: _Statvfs(1 << 40), raising = False)
    parent, child = multiprocessing.Pipe(duplex = True)
    arena = _SharedArena(conn = child)
    try:
        arena.reserve(1)
        assert parent.poll(1.0)
        assert parent.recv() == ("chunk", arena.chunks[0].name, arena.chunks[0].size)
    finally:
        for c in arena.chunks:
            c.close()
            c.unlink()
        parent.close()
        child.close()


def _pumping_host():
    host = MoeCpuHost(_config())
    parent, child = multiprocessing.Pipe(duplex = True)
    host.conn = parent
    host.proc = _FakeProcess(None, (), True)
    return host, child


def test_pump_records_chunks_and_acks_layers_separately():
    host, child = _pumping_host()
    try:
        child.send(("chunk", "exl3_test_a", 64))
        child.send(("ok",))
        assert host._pump(1.0) and host._pump(1.0)
        assert host.arena_names == ["exl3_test_a"]
        assert host.acked == 1
    finally:
        child.close()
        host.shutdown()


def test_pump_rejects_unknown_message():
    host, child = _pumping_host()
    try:
        child.send(("bogus",))
        with pytest.raises(RuntimeError, match = "bogus"):
            host._pump(1.0)
    finally:
        child.close()
        host.shutdown()


def test_pump_worker_error_shuts_down_before_raising():
    host, child = _pumping_host()
    child.send(("err", "boom"))
    try:
        with pytest.raises(RuntimeError, match = "boom"):
            host._pump(1.0)
        assert host.proc is None
        assert host.conn is None
    finally:
        child.close()


def test_pump_worker_death_shuts_down_before_raising():
    host, child = _pumping_host()
    host.proc.alive = False
    try:
        with pytest.raises(RuntimeError, match = "died"):
            host._pump(0.01)
        assert host.proc is None
    finally:
        child.close()


@pytest.mark.skipif(os.name == "nt", reason = "shared memory names are not persistent on Windows")
def test_shutdown_unlinks_published_chunks_it_never_mapped():
    shm = shared_memory.SharedMemory(create = True, size = 4096)
    name = shm.name
    host = MoeCpuHost(_config())
    host.arena_names = [name]
    host.shutdown()
    shm.close()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name = name)
