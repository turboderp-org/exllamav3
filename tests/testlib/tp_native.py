"""
Harness for the native tensor-parallel backend (TPBackendNative, parallel/*.cu): rank processes that each run the
real backend over the real shared-memory regions, plus the CPU reduce helper process, as model_tp.py sets them up.

    results = run_ranks(program, ranks, *args, output_rank = -1, rank_envs = None, env = None)

`ranks` is the list of CUDA ordinals (as each rank's process sees them) forming active_devices; rank i runs on
ranks[i]. One process per rank, plus the CPU helper (device -1, cpu = True), each a fresh interpreter. The rank on
ranks[output_rank] is the master (creates the shared memory and calls pg_init_context), as the output device's
pseudo-worker is in model_tp.py. Every participant first meets at a host barrier, so no collective starts before
the master has initialized the context.

`program` is a module-level function of a test module, called in every rank process as
program(ctx, *args) inside torch.inference_mode(); ctx is a RankContext (backend, device, devices, rank, world,
output_device, master, sync(), end_round()). Its return value (picklable, CPU tensors) comes back as results[rank].

The CPU helper calls run_cpu_reduce_jobs() whenever the job queue is non-empty, as model_tp.py dispatches it
once per forward pass; each call returns at the end marker the master pushes in backend.end_cpu_reduce_jobs().
ctx.end_round() ends a round (every rank calls it, as every rank calls end_cpu_reduce_jobs at the end of a forward).
The harness ends the last round after the program returns.

rank_envs: per-rank environment overrides (e.g. CUDA_VISIBLE_DEVICES for two ranks on one physical device,
see single_device_ranks). env: overrides for every process, the CPU helper included (e.g. EXL3_TP_NO_FP16_WIRE).
scramble_context: the master fills the context region (G) with random bytes after creating it and re-initializes
it with pg_init_context alone (TPBackendNative zeroes it first, which would hide an incomplete init).

A failure in any participant terminates the others and raises AssertionError with the output tails.
"""

import importlib.util
import inspect
import os
import pickle
import subprocess
import sys
import tempfile
import time
import uuid as uuid_mod
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np
import torch

_MODULE_NAME = "_exl3_tp_native_module"
_SYNC_STRIDE = 16   # u32 words between participants' barrier slots (one cache line each)
# Byte offsets of PGContext::reduce_jobs_head / reduce_jobs_tail (parallel/context.cuh)
_CTX_JOBS_HEAD = 576
_CTX_JOBS_TAIL = 640


@dataclass
class RankContext:
    backend: object
    device: int
    devices: list
    rank: int
    world: int
    output_device: int
    master: bool
    _sync: object = None

    def end_round(self):
        """End a CPU-reduce round (one forward pass): the master pushes the end marker"""
        self.backend.end_cpu_reduce_jobs()

    def sync(self):
        """torch.cuda.synchronize() on this rank, then a host barrier over all ranks (not the CPU helper)"""
        torch.cuda.synchronize()
        self._sync.wait(self.world)


class _HostBarrier:
    """Sense-free generation barrier over a shared u32 array: each participant owns one slot (single writer)"""

    def __init__(self, name, index, participants):
        self.shm = shared_memory.SharedMemory(name = name)
        self.u32 = np.ndarray((participants * _SYNC_STRIDE,), dtype = np.uint32, buffer = self.shm.buf)
        self.index = index
        self.participants = participants
        self.gen = {}

    def wait(self, count = None, timeout = 120.0):
        """Barrier over participants 0..count-1 (default: all). Each subset keeps its own generation counter"""
        count = count or self.participants
        g = self.gen.get(count, 0) + 1
        self.gen[count] = g
        # Separate word per (subset, participant): subset k uses word k-1 of the participant's line
        word = self.index * _SYNC_STRIDE + (count - 1)
        self.u32[word] = g
        deadline = time.monotonic() + timeout
        while True:
            if all(self.u32[i * _SYNC_STRIDE + (count - 1)] >= g for i in range(count)):
                return
            if time.monotonic() > deadline:
                raise TimeoutError("host barrier timeout")
            time.sleep(0.0005)

    def request_stop(self):
        self.u32[_SYNC_STRIDE - 1] = 1

    def stop_requested(self) -> bool:
        return bool(self.u32[_SYNC_STRIDE - 1])

    def close(self):
        del self.u32
        self.shm.close()


def single_device_ranks(device, other) -> tuple[list[int], list[dict]]:
    """Ranks [0, 1] that both run on the physical GPU `device`: the kernels index the shared context by CUDA
    ordinal, so the two rank processes need distinct ordinals for the same GPU. Rank 0 sees only that GPU (as
    ordinal 0); rank 1 sees `other` first and that GPU second (as ordinal 1) and never touches `other`.
    Returns (ranks, rank_envs)"""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    phys = lambda d: visible.split(",")[torch.device(d).index].strip() if visible else str(torch.device(d).index)
    p, q = phys(device), phys(other)
    assert p != q
    return [0, 1], [{"CUDA_VISIBLE_DEVICES": p}, {"CUDA_VISIBLE_DEVICES": f"{q},{p}"}]


def run_ranks(program, ranks: list[int], *args, output_rank: int = -1, rank_envs: list[dict] | None = None,
              env: dict | None = None, scramble_context: bool = False, timeout: float = 300.0,
              **kwargs) -> list:
    path = os.path.abspath(inspect.getsourcefile(program))
    name = program.__name__
    assert program.__qualname__ == name, "program must be a module-level function"
    world = len(ranks)
    output_device = ranks[output_rank]
    uid = "exl3t_" + uuid_mod.uuid4().hex[:16]
    participants = world + 1
    sync = shared_memory.SharedMemory(create = True, size = participants * _SYNC_STRIDE * 4, name = uid + "_sync")
    np.ndarray((participants * _SYNC_STRIDE,), dtype = np.uint32, buffer = sync.buf)[:] = 0
    procs = []
    try:
        with tempfile.TemporaryDirectory(prefix = "exl3_tp_") as tmp:
            spec = dict(path = path, name = name, args = args, kwargs = kwargs, ranks = ranks, uuid = uid,
                        output_device = output_device, participants = participants,
                        scramble_context = scramble_context)
            spec_file = os.path.join(tmp, "spec.pkl")
            with open(spec_file, "wb") as f:
                pickle.dump(spec, f)
            roles = [str(i) for i in range(world)] + ["cpu"]
            for i, role in enumerate(roles):
                e = dict(os.environ, **{k: str(v) for k, v in (env or {}).items()})
                if role == "cpu":
                    # The helper never launches kernels
                    e["CUDA_VISIBLE_DEVICES"] = ""
                elif rank_envs:
                    e.update({k: str(v) for k, v in rank_envs[i].items()})
                log = open(os.path.join(tmp, f"log_{role}.txt"), "w")
                procs.append((role, subprocess.Popen(
                    [sys.executable, "-c", "from testlib.tp_native import _child_main; _child_main()",
                     spec_file, role, os.path.join(tmp, f"out_{role}.pt")],
                    env = e, stdout = log, stderr = subprocess.STDOUT), log))

            def tails():
                out = []
                for role, _, log in procs:
                    log.flush()
                    with open(os.path.join(tmp, f"log_{role}.txt")) as f:
                        out.append(f"--- {role} ---\n" + f.read()[-4000:])
                return "\n".join(out)

            deadline = time.monotonic() + timeout
            failed = None
            while True:
                codes = [p.poll() for _, p, _ in procs]
                bad = [(r, c) for (r, _, _), c in zip(procs, codes) if c not in (None, 0)]
                if bad:
                    failed = f"participant(s) {bad} failed"
                    break
                if all(c == 0 for c in codes):
                    break
                if time.monotonic() > deadline:
                    failed = f"timeout after {timeout:.0f} s (exit codes {codes})"
                    break
                time.sleep(0.05)
            if failed:
                for _, p, _ in procs:
                    if p.poll() is None:
                        p.kill()
                for _, p, _ in procs:
                    p.wait()
                raise AssertionError(f"native TP harness: {failed}\n{tails()}")
            return [torch.load(os.path.join(tmp, f"out_{i}.pt"), weights_only = False) for i in range(world)]
    finally:
        for _, p, log in procs:
            if p.poll() is None:
                p.kill()
                p.wait()
            log.close()
        sync.close()
        sync.unlink()
        # Segments of a master that did not get to close()
        for suffix in ("_g", "_b", "_r", "_s", "_ll"):
            try:
                shared_memory.SharedMemory(name = uid + suffix).unlink()
            except FileNotFoundError:
                pass


def _child_main():
    spec_file, role, out_file = sys.argv[1:4]
    with open(spec_file, "rb") as f:
        spec = pickle.load(f)
    ranks, uid, output_device = spec["ranks"], spec["uuid"], spec["output_device"]
    world = len(ranks)
    index = world if role == "cpu" else int(role)

    if role != "cpu":
        torch.cuda.set_device(ranks[index])
    from exllamav3.model.model_tp_backend import TPBackendNative
    device = ranks[index] if role != "cpu" else -1
    master = device == output_device
    # Each child is a fresh interpreter with its own resource tracker, which would unlink the segments it merely
    # opened when it exits (model_tp.py's spawned workers share the parent's tracker instead). The master unlinks
    # its segments in close(); run_ranks removes what a crashed run leaves behind
    from multiprocessing import resource_tracker
    resource_tracker.register = resource_tracker.unregister = lambda *a, **k: None
    barrier = _HostBarrier(uid + "_sync", index, spec["participants"])

    if role == "cpu":
        backend = TPBackendNative(-1, ranks, output_device, "", master = False, uuid = uid, cpu = True)
        barrier.wait()
        # As in model_tp.py, where the helper is dispatched at the start of each forward pass, run_cpu_reduce_jobs
        # is entered only when a round has work queued (its wait timeout counts from the call)
        u32 = backend.tensor_g.view(torch.int32)
        head, tail = _CTX_JOBS_HEAD // 4, _CTX_JOBS_TAIL // 4
        while True:
            while u32[head].item() == u32[tail].item() and not barrier.stop_requested():
                time.sleep(0.0005)
            if u32[head].item() == u32[tail].item():
                break
            backend.run_cpu_reduce_jobs()
        barrier.wait()
        backend.close()
        barrier.close()
        torch.save(None, out_file)
        return

    backend = TPBackendNative(device, ranks, output_device, "", master = master, uuid = uid)
    if master and spec["scramble_context"]:
        # Garbage over the whole context region, then pg_init_context alone (no participant has touched the
        # region yet: they all wait at the barrier below)
        from exllamav3.ext import exllamav3_ext as ext
        g = torch.Generator().manual_seed(0)
        backend.tensor_g.copy_(torch.randint(0, 256, backend.tensor_g.shape, dtype = torch.uint8, generator = g))
        ext.pg_init_context(backend.ptr_g)
    barrier.wait()

    mod_spec = importlib.util.spec_from_file_location(_MODULE_NAME, spec["path"])
    module = importlib.util.module_from_spec(mod_spec)
    sys.modules[_MODULE_NAME] = module
    mod_spec.loader.exec_module(module)
    ctx = RankContext(backend = backend, device = device, devices = ranks, rank = index, world = world,
                      output_device = output_device, master = master, _sync = barrier)
    with torch.inference_mode():
        result = getattr(module, spec["name"])(ctx, *spec["args"], **spec["kwargs"])
        torch.cuda.synchronize()
        # Every reduce has completed (the CPU helper processed them all); the stop word goes up before the final
        # end marker, so the helper's last run returns with it set
        if master:
            barrier.request_stop()
        backend.end_cpu_reduce_jobs()
        torch.cuda.synchronize()
    barrier.wait()
    backend.close()
    barrier.close()
    torch.save(result, out_file)
