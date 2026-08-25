from __future__ import annotations
import os
import multiprocessing
from multiprocessing import shared_memory
import numpy as np
import torch

from ..ext import exllamav3_ext as ext
from ..util.misc import Cleanupper, install_parent_death_signal
from .model_tp_cuda import (
    cuda_host_register,
    cuda_host_unregister,
    cuda_host_get_device_pointer,
    CUDA_HOST_REGISTER_PORTABLE,
    CUDA_HOST_REGISTER_MAPPED,
)

cleanupper = Cleanupper()

"""
Persistent-worker handoff for CPU-offloaded MoE experts, following the native TP backend's
CPU-helper pattern: one spawned child process owns the expert weights and consumes a job ring in
pinned shared memory. The parent's forward pass never blocks on the CPU: for each offloaded
layer it enqueues, onto the CUDA stream, contiguous D2H copies of the staged inputs, a kernel
publishing the job's sequence number, a blocking kernel that waits for the worker's completion
flag, and the contiguous H2D readback (strided GPU<->CPU copies are forbidden here: torch stages
them host-side at enqueue time, outside stream order, which reads/writes the slots at the wrong
time).

Loading is incremental: the child is spawned at the first layer registration and receives layer
specs over a pipe as the parent's loader reaches each MoE layer, loading the expert tensors
concurrently with the parent's GPU loading. After a module commits on the GPU side (no OOM
rollback), the loader waits for the child's ack for that layer, so the progress bar reflects
combined progress and there is no bulk stall at the end. Re-registering a key (autosplit
rollback retry) returns the existing layer index without reloading.

Shared-memory layout constants mirror cpu/moe_handoff.h.
"""

MOE_JOB_RING = 256
MOE_MAX_SLOTS = 8
MOE_JOB_MAX_EXPERTS = 256   # structural capacity of MoeJob.experts[]; see cpu/moe_handoff.h
# sizeof(MoeJob): 7 uint32 fields + experts[MOE_JOB_MAX_EXPERTS] + one uint32 pad, all 4-byte
# fields so no compiler padding -- recompute this if the struct's fixed fields ever change
MOE_JOB_BYTES = (7 + MOE_JOB_MAX_EXPERTS + 1) * 4
MOE_CTRL_JOBS_OFFSET = 384
MOE_SLOT_FLAGS_OFFSET = MOE_CTRL_JOBS_OFFSET + MOE_JOB_RING * MOE_JOB_BYTES
MOE_MAX_WSLOTS = 8
MOE_FLAGS_SIZE = 3 * 64 * MOE_MAX_SLOTS + 2 * 64 * MOE_MAX_WSLOTS
MOE_STAGE_RING = 64
MOE_STAGE_TAIL_OFFSET = MOE_SLOT_FLAGS_OFFSET + MOE_FLAGS_SIZE
MOE_STAGE_HEAD_OFFSET = MOE_STAGE_TAIL_OFFSET + 64
MOE_STAGE_JOBS_OFFSET = MOE_STAGE_TAIL_OFFSET + 128
MOE_CTRL_SIZE = MOE_STAGE_JOBS_OFFSET + MOE_STAGE_RING * MOE_JOB_BYTES


def _align64(x):
    return (x + 63) & ~63


class MoeCpuTuning:
    """
    Tunable knobs for the CPU MoE offload path, collected in one place: env vars are read once
    here (at import time) instead of scattered os.environ.get calls, so a config-file migration
    or an automated sweep only has one object to touch. MoeCpuHost copies the values it needs at
    construction time (plain scalars -- this class is never bound into C++; anything the native
    side needs is propagated explicitly as a call or pipe-message parameter).

    For a same-process sweep, mutate fields on the module-level TUNING singleton before
    constructing each MoeCpuHost (each model load constructs a fresh one); env vars only matter
    at the first import.
    """

    def __init__(self):
        # --- CPU worker / staging ---
        self.num_slots = int(os.environ.get("EXL3_MOE_CPU_SLOTS", 4))
        self.cap_rows = int(os.environ.get("EXL3_MOE_CPU_SLOT_ROWS", 64))
        # Thread count fallback chain ends here; config.infer_params.moe_cpu_threads (or the
        # draft/MTP equivalent) takes precedence per host when set (MoeCpuHost.__init__)
        self.threads = int(os.environ.get("EXL3_MOE_CPU_THREADS", max(1, (os.cpu_count() or 2) // 2)))
        self.num_wslots = min(int(os.environ.get("EXL3_MOE_CPU_WSLOTS", 2)), MOE_MAX_WSLOTS)
        self.wslot_size = int(os.environ.get("EXL3_MOE_CPU_WSLOT_MB", 32)) * 1024 * 1024
        self.stage_threads = int(os.environ.get("EXL3_MOE_CPU_STAGE_THREADS", 4))
        # madvise(MADV_HUGEPAGE) on the expert-weight arena chunks: with defrag=madvise (the
        # common default), the kernel does SYNCHRONOUS compaction on first touch of a hinted
        # region once easily-compactable free memory runs low, which can stall loading badly
        self.arena_hugepage = os.environ.get("EXL3_MOE_ARENA_HUGEPAGE", "1") != "0"
        # Band-contiguous ("swizzled") expert trellis layout: repacked at arena rehome so each
        # 8-tile output band streams sequentially from DRAM. Only applied when the VBMI kernel
        # tier is active. EXL3_MOE_CPU_SWIZZLE=0 restores the native layout.
        self.swizzle = os.environ.get("EXL3_MOE_CPU_SWIZZLE", "1") != "0"

        # --- GPU-streaming prefill ---
        self.stream_t_explicit = "EXL3_MOE_STREAM_T" in os.environ
        self.stream_t = int(os.environ.get("EXL3_MOE_STREAM_T", 16))
        self.stream_fused_t = int(os.environ.get("EXL3_MOE_STREAM_FUSED_T", 512))
        self.stream_min_rows = int(os.environ.get("EXL3_MOE_STREAM_MIN_ROWS", 32))
        self.batch_experts = max(1, min(
            int(os.environ.get("EXL3_MOE_STREAM_BATCH_EXPERTS", 24)), MOE_JOB_MAX_EXPERTS))

        # --- debug / kill switches ---
        self.stream_debug = bool(os.environ.get("EXL3_MOE_STREAM_DEBUG"))
        self.cpu_prof = bool(os.environ.get("EXL3_MOE_CPU_PROF"))
        self.memops = os.environ.get("EXL3_MOE_MEMOPS", "1") != "0"


TUNING = MoeCpuTuning()
ext.exl3_moe_cpu_set_memops(TUNING.memops)


class _HugeArena:
    """
    Growable pool of large (default 1 GiB) anonymous mmap chunks that expert weights are copied
    into, so hugepage promotion (see promote_hugepages) has few, large regions to work with
    instead of thousands of separate small (sub-2MB) loader allocations
    """
    CHUNK_BYTES = 1 << 30   # 1 GiB

    def __init__(self):
        self.chunks = []
        self.cur = None
        self.cur_off = 0

    def _new_chunk(self, min_bytes):
        import mmap, os
        size = max(self.CHUNK_BYTES, (min_bytes + (2 << 20) - 1) & ~((2 << 20) - 1))
        if os.name == "nt":
            # mmap.MAP_PRIVATE / mmap.PROT_* don't exist on Windows; an anonymous mapping is
            # writable by default there. Hugepage promotion doesn't apply (promote_hugepages
            # is already a no-op via its try/except), the arena still serves its pooling role
            m = mmap.mmap(-1, size)
        else:
            m = mmap.mmap(-1, size, mmap.MAP_PRIVATE, mmap.PROT_READ | mmap.PROT_WRITE)
        self.chunks.append(m)
        self.cur = m
        self.cur_off = 0
        if os.environ.get("EXL3_MOE_ARENA_DEBUG"):
            total = sum(len(c) for c in self.chunks)
            print(f" -- arena: new chunk {size/1e6:.1f} MB, {len(self.chunks)} chunks, "
                  f"{total/1e9:.3f} GB total", flush = True)

    def promote_hugepages(self):
        """One-shot MADV_COLLAPSE (Linux 6.1+) over each chunk, meant to run once after all
        expert weights are loaded. Deliberately NOT done via a live MADV_HUGEPAGE hint during the
        per-tensor writes in rehome(): with the common defrag=madvise policy, that hint makes the
        kernel do SYNCHRONOUS compaction on every first touch of a hinted region once easily-
        compactable free memory runs low, which turned into multi-second stalls per offloaded
        layer partway through a large model's load. A single explicit collapse pass after
        loading gets the same steady-state throughput benefit without blocking incremental
        per-layer progress. Best-effort: silently leaves chunks at 4K pages if collapse fails or
        the kernel doesn't support it."""
        import mmap, os
        if not TUNING.arena_hugepage:
            return
        collapse = getattr(mmap, "MADV_COLLAPSE", 25)
        for c in self.chunks:
            try:
                c.madvise(collapse)
            except Exception:
                pass
        if os.environ.get("EXL3_MOE_ARENA_DEBUG"):
            print(f" -- arena: MADV_COLLAPSE issued on {len(self.chunks)} chunks", flush = True)

    def rehome(self, tensor, band_swizzle = False):
        """Copy `tensor` into the arena and return a same-dtype/shape view over the copy. The
        arena outlives every tensor it hands out (held for the process lifetime), so the
        returned view stays valid.

        band_swizzle: repack a [k/16, n/16, 16K] trellis tensor band-contiguous during the
        copy -- physical order becomes (group n/128, k-tile, member, tile), one strided copy_.
        The returned view keeps the original logical shape; only the byte order differs
        (consumed by the swz-aware kernels in moe_mul1.cpp)."""
        import torch
        if tensor is None or tensor.numel() == 0:
            return tensor
        nbytes = tensor.numel() * tensor.element_size()
        aligned = (nbytes + 63) & ~63
        if self.cur is None or self.cur_off + aligned > len(self.cur):
            self._new_chunk(aligned)
        off = self.cur_off
        self.cur_off += aligned
        buf = memoryview(self.cur)[off : off + nbytes]
        dst = torch.frombuffer(buf, dtype = torch.uint8)
        if band_swizzle:
            tk, tn, ps = tensor.shape
            dst.view(tensor.dtype).view(tn // 8, tk, 8, ps) \
               .copy_(tensor.view(tk, tn // 8, 8, ps).permute(1, 0, 2, 3))
        else:
            dst.copy_(tensor.contiguous().view(torch.uint8).reshape(-1))
        return dst.view(tensor.dtype).view(tensor.shape)


def _moe_cpu_child_main(conn, model_dir, threads, stage_threads):
    """
    Child entry point: receives ("layer", spec) messages, loading each layer's expert tensors
    (deferred, multithreaded) and acking, until ("start", shm_name, layout) switches it into the
    worker loop. Errors are reported over the pipe before exiting.
    """
    import ctypes
    import signal
    import traceback
    import torch  # noqa: F401
    from ..ext import exllamav3_ext as cext
    from ..loader.safetensors import SafetensorsCollection
    from ..util.misc import install_parent_death_signal as ipds

    # Terminal Ctrl-C is delivered to the whole foreground process group; shutdown is
    # orchestrated by the parent (quit flag) or the kernel (PDEATHSIG), never by SIGINT
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ipds()

    if os.name == "nt":
        # Hold 1 ms timer resolution for this process (per-process since Windows 10 2004): the
        # worker's poll loops back off to 50 us sleeps, which otherwise round up to the default
        # 15.6 ms quantum -- one quantum per job-ring poll miss lands directly on token latency
        try:
            ctypes.WinDLL("winmm").timeBeginPeriod(1)
        except Exception:
            pass

    shm = None
    try:
        stc = SafetensorsCollection(model_dir)
        cpu = torch.device("cpu")
        arena = _HugeArena()

        def fetch(keys):
            out = []
            for k in keys:
                trellis = stc.get_tensor(k + ".trellis", cpu)
                suh = stc.get_tensor(k + ".suh", cpu, float2half = True)
                svh = stc.get_tensor(k + ".svh", cpu, float2half = True)
                bias = stc.get_tensor(k + ".bias", cpu, optional = True, float2half = True)
                out.append((trellis, suh, svh, bias))
            return out

        # Swizzle the trellis copies band-contiguous when the VBMI kernel tier will consume them
        swz = TUNING.swizzle and cext.exl3_moe_cpu_has_avx512_vbmi()

        def rehome_trellis(t):
            return arena.rehome(t, band_swizzle = swz and t.shape[2] // 16 != 8)

        def rehome_all(ts):
            return [(rehome_trellis(t[0]), arena.rehome(t[1]), arena.rehome(t[2]), arena.rehome(t[3]))
                   for t in ts]

        def biases(ts):
            return [t[3] for t in ts] if ts and ts[0][3] is not None else []

        # Per-layer, per-expert arena views retained for runtime expert installs (dynamic
        # placement): the arena memory is what the compute kernels read, so an in-place copy
        # into these views (with the same swizzle transform rehome applied) replaces an
        # expert's weights. The parent quiesces (full sync, ring drained) before sending an
        # install, so the workers never observe a torn write
        layer_views = []

        while True:
            msg = conn.recv()
            if msg[0] == "layer":
                spec = msg[1]
                stc.begin_deferred_load()
                g = fetch(spec["gate_keys"])
                u = fetch(spec["up_keys"])
                d = fetch(spec["down_keys"])
                stc.end_deferred_load()
                # Copy into the hugepage-backed arena now that the deferred reads have actually
                # populated these tensors
                g, u, d = rehome_all(g), rehome_all(u), rehome_all(d)
                cext.exl3_moe_cpu_make_layer(
                    [t[0] for t in g], [t[1] for t in g], [t[2] for t in g],
                    [t[0] for t in u], [t[1] for t in u], [t[2] for t in u],
                    [t[0] for t in d], [t[1] for t in d], [t[2] for t in d],
                    biases(g), biases(u), biases(d),
                    spec["activation"], spec["act_limit"],
                    1 if swz else 0,
                )
                layer_views.append((g, u, d))
                # Reclaim this layer's now-discarded loader tensors immediately (rehome_all copied
                # everything into the arena)
                try:
                    ctypes.CDLL(None).malloc_trim(0)
                except Exception:
                    pass
                conn.send(("ok",))
            elif msg[0] == "start":
                shm_name, layout = msg[1], msg[2]
                break
            elif msg[0] == "quit":
                return

        arena.promote_hugepages()

        stc.close()
        shm = shared_memory.SharedMemory(name = shm_name)
        base = np.frombuffer(shm.buf, dtype = np.uint8).ctypes.data
        cext.exl3_moe_cpu_set_prof(layout.get("cpu_prof", False))

        def install(li, ei, keys):
            """Replace expert ei of layer li with the checkpoint tensors at `keys` (one key
            per projection, gate omitted for gateless layers), in place in the arena."""
            views = layer_views[li]
            projs = views if len(keys) == 3 else views[1:]
            for key, plist in zip(keys, projs):
                v_tr, v_suh, v_svh, v_bias = plist[ei]
                new = stc.get_tensor(key + ".trellis", cpu)
                assert new.shape == v_tr.shape, f"install shape mismatch: {key}"
                if swz and v_tr.shape[2] // 16 != 8:
                    tk, tn, ps = new.shape
                    v_tr.view(tn // 8, tk, 8, ps) \
                        .copy_(new.view(tk, tn // 8, 8, ps).permute(1, 0, 2, 3))
                else:
                    v_tr.copy_(new)
                v_suh.copy_(stc.get_tensor(key + ".suh", cpu, float2half = True))
                v_svh.copy_(stc.get_tensor(key + ".svh", cpu, float2half = True))
                if v_bias is not None:
                    v_bias.copy_(stc.get_tensor(key + ".bias", cpu, float2half = True))

        # The compute loop runs on its own thread (worker_run releases the GIL); the main
        # thread keeps serving the pipe for runtime installs
        import threading
        worker = threading.Thread(
            target = cext.exl3_moe_cpu_worker_run,
            args = (
                base,
                layout["num_slots"], layout["slot_size"], layout["cap_rows"],
                layout["max_hi"], layout["max_ho"], layout["max_topk"],
                layout["wstage_off"], layout["num_wslots"], layout["wslot_size"],
                threads, stage_threads,
            ),
            daemon = True,
        )
        worker.start()

        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break
            if msg[0] == "install":
                try:
                    install(msg[1], msg[2], msg[3])
                    conn.send(("ok",))
                except Exception:
                    conn.send(("err", traceback.format_exc()))
            elif msg[0] == "quit":
                break
        worker.join(timeout = 2.0)
    except Exception:
        try:
            conn.send(("err", traceback.format_exc()))
        except Exception:
            pass
        raise
    finally:
        if shm is not None:
            shm.close()


class MoeCpuHost:

    def __init__(self, config):
        self.config = config
        self.model_dir = config.directory
        self.specs = []
        self.by_key = {}
        self.live_layers = 0
        self.acked = 0
        self.started = False
        self.shm = None
        self.proc = None
        self.conn = None
        self.seq = 0
        self.next_slot = 0
        self.slot_last_seq = [0] * MOE_MAX_SLOTS
        self.num_slots = TUNING.num_slots
        self.cap_rows = TUNING.cap_rows
        # Per-component thread override: config.infer_params.moe_cpu_threads for the main model,
        # draft_moe_cpu_threads for anything else (MTP head / draft model); falls back to the
        # tuning default (EXL3_MOE_CPU_THREADS env, else cpu_count/2)
        comp = getattr(config.infer_params, "moe_cpu_component", "text")
        cfg_threads = getattr(config.infer_params,
            "moe_cpu_threads" if comp == "text" else "draft_moe_cpu_threads", None)
        self.threads = cfg_threads or TUNING.threads
        self.stage_threads = TUNING.stage_threads
        # GPU-streaming prefill: experts with at least stream_t assigned tokens are streamed to
        # the GPU (weights DMA'd through a pinned staging ring) while the tail stays on the CPU
        self.num_wslots = TUNING.num_wslots
        self.wslot_size = TUNING.wslot_size
        self.stream_t = TUNING.stream_t
        self.stream_min_rows = TUNING.stream_min_rows
        self.batch_experts = TUNING.batch_experts
        self.wseq = 0
        self.next_wslot = 0
        self.wslot_prev_seq = [0] * MOE_MAX_WSLOTS
        self.aux = {}

    def _spawn(self):
        if self.proc is not None:
            return
        ctx = multiprocessing.get_context("spawn")
        self.conn, child_conn = ctx.Pipe(duplex = True)
        self.proc = ctx.Process(
            target = _moe_cpu_child_main,
            args = (child_conn, self.model_dir, self.threads, self.stage_threads),
            daemon = True,
        )
        self.proc.start()
        child_conn.close()
        # Cleanupper fires at the end of the __main__ scope, before interpreter teardown breaks
        # the shm views and pipe machinery that shutdown() needs; PDEATHSIG in the child covers
        # the paths where no Python hook runs at all
        cleanupper.register_atexit(self.shutdown)

    def _pump(self, timeout):
        """Receive one message from the child, surfacing errors and death"""
        if self.conn.poll(timeout):
            msg = self.conn.recv()
            if msg[0] == "err":
                raise RuntimeError(f"CPU MoE worker failed:\n{msg[1]}")
            if msg[0] == "ok":
                self.acked += 1
            return True
        if not self.proc.is_alive():
            raise RuntimeError("CPU MoE worker process died")
        return False

    def register_layer(self, key, gate_keys, up_keys, down_keys, activation, act_limit, hi, ho, topk,
                       proj_dims = None, aux = None):
        if key in self.by_key:
            # Autosplit rollback retry: the child keeps its copy, reuse the index, but take
            # the re-fetched aux tensors: the retry runs on a different device, and the stored
            # copies live on the one the layer just rolled back from. Streamed-prefill dequant
            # builds raw pointer tables from these, so stale entries are device-A addresses
            # handed to kernels on device B (illegal memory access on the first  multi-chunk
            # prefill after a cross-device rollback)
            idx = self.by_key[key]
            self.live_layers += 1
            if aux is not None:
                self.aux[idx] = aux
            return idx
        assert not self.started, "cannot register layers after the worker has started"
        self._spawn()
        spec = dict(
            gate_keys = gate_keys, up_keys = up_keys, down_keys = down_keys,
            activation = activation, act_limit = act_limit,
            hi = hi, ho = ho, topk = topk,
            num_experts = len(up_keys),
            proj_dims = proj_dims,
        )
        if proj_dims is not None:
            # Deterministic per-expert byte layout (gate, up, down), mirrored by the worker's
            # stage function
            def tb(d):
                k, n, K = d
                return (k // 16) * (n // 16) * 16 * K * 2
            gb = tb(proj_dims["g"]) if proj_dims.get("g") else 0
            ub, db = tb(proj_dims["u"]), tb(proj_dims["d"])
            spec["proj_bytes"] = (gb, ub, db)
            spec["expert_bytes"] = gb + ub + db
        self.specs.append(spec)
        self.live_layers += 1
        idx = len(self.specs) - 1
        self.by_key[key] = idx
        if aux is not None:
            self.aux[idx] = aux
        self.conn.send(("layer", {k: v for k, v in spec.items() if k != "proj_dims"}))
        return idx

    def commit_module(self, module_key):
        """
        Called by the loader after a top-level module has committed on the GPU side: block until
        the child has loaded every layer registered under that module, keeping load progress
        honest and the child at most one layer behind.
        """
        if self.proc is None:
            return
        idxs = [i for k, i in self.by_key.items()
                if k == module_key or k.startswith(module_key + ".")]
        if not idxs:
            return
        need = max(idxs) + 1
        while self.acked < need:
            self._pump(1.0)

    def ensure_started(self):
        if self.started or not self.specs:
            return
        while self.acked < len(self.specs):
            self._pump(1.0)

        max_hi = max(s["hi"] for s in self.specs)
        max_ho = max(s["ho"] for s in self.specs)
        max_topk = max(s["topk"] for s in self.specs)
        # All staged sections must be contiguous 2D blocks: torch implements strided GPU<->CPU
        # copies with a host-side staging pass at enqueue time, which breaks stream ordering (the
        # readback would ship the slot's *previous* contents). Uniform dims keep every copy a
        # full-width contiguous slice
        assert all(s["hi"] == max_hi and s["ho"] == max_ho and s["topk"] == max_topk
                   for s in self.specs), "CPU MoE offload requires uniform expert dims and top-k"

        off_x = 0
        off_sel = _align64(off_x + self.cap_rows * max_hi * 2)
        off_w = _align64(off_sel + self.cap_rows * max_topk * 4)
        off_out = _align64(off_w + self.cap_rows * max_topk * 2)
        slot_size = _align64(off_out + self.cap_rows * max_ho * 4)
        self.layout = dict(
            num_slots = self.num_slots, slot_size = slot_size, cap_rows = self.cap_rows,
            max_hi = max_hi, max_ho = max_ho, max_topk = max_topk,
        )

        self.layout["wstage_off"] = MOE_CTRL_SIZE + self.num_slots * slot_size
        self.layout["num_wslots"] = self.num_wslots
        self.layout["wslot_size"] = self.wslot_size
        self.layout["cpu_prof"] = TUNING.cpu_prof
        size = MOE_CTRL_SIZE + self.num_slots * slot_size + self.num_wslots * self.wslot_size
        self.shm = shared_memory.SharedMemory(create = True, size = size)
        buf = np.frombuffer(self.shm.buf, dtype = np.uint8)
        buf[:MOE_CTRL_SIZE] = 0
        self.base_ptr = buf.ctypes.data
        cuda_host_register(self.base_ptr, size,
                           flags = CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED)
        # GPU-visible alias of the registered buffer: equal to base_ptr under UVA (Linux
        # desktop), but a distinct address under WDDM (Windows), where the host VA is not a
        # valid device pointer
        self.gpu_base_ptr = cuda_host_get_device_pointer(self.base_ptr)

        u32 = np.frombuffer(self.shm.buf, dtype = np.uint32)
        self.v_quit = u32[0:1]
        self.v_pass_wake = u32[16:17]
        self.v_abort = u32[32:33]
        self.v_ready = u32[48:49]
        self.v_jobs_tail = u32[64:65]
        self.v_jobs_head = u32[80:81]
        self.v_jobs = np.frombuffer(
            self.shm.buf, dtype = np.uint32,
            offset = MOE_CTRL_JOBS_OFFSET, count = MOE_JOB_RING * (MOE_JOB_BYTES // 4)).reshape(MOE_JOB_RING, MOE_JOB_BYTES // 4)

        self.slots = []
        for s in range(self.num_slots):
            sbase = MOE_CTRL_SIZE + s * slot_size
            def view(off, count, dtype):
                return torch.frombuffer(self.shm.buf, dtype = dtype, count = count,
                                        offset = sbase + off)
            # Device-visible aliases of the slot's data sections, for the fused issue/collect
            # kernels' zero-copy accesses (same mapped registration as the flag words)
            gpu_data = self.gpu_base_ptr + sbase
            self.slots.append(dict(
                x = view(off_x, self.cap_rows * max_hi, torch.half).view(self.cap_rows, max_hi),
                sel = view(off_sel, self.cap_rows * max_topk, torch.int32).view(self.cap_rows, max_topk),
                w = view(off_w, self.cap_rows * max_topk, torch.half).view(self.cap_rows, max_topk),
                out = view(off_out, self.cap_rows * max_ho, torch.float).view(self.cap_rows, max_ho),
                x_dev = gpu_data + off_x,
                sel_dev = gpu_data + off_sel,
                w_dev = gpu_data + off_w,
                out_dev = gpu_data + off_out,
                data_ready = self.gpu_base_ptr + MOE_SLOT_FLAGS_OFFSET + s * 64,
                done = self.gpu_base_ptr + MOE_SLOT_FLAGS_OFFSET + 64 * MOE_MAX_SLOTS + s * 64,
                consumed = self.gpu_base_ptr + MOE_SLOT_FLAGS_OFFSET + 2 * 64 * MOE_MAX_SLOTS + s * 64,
            ))
        # Per-device empty-job gates for the fused kernels (issue writes, collect reads)
        self.dev_count = {}

        # Weight-staging views, flags and streams for GPU-streamed prefill
        self.wviews = []
        for s in range(self.num_wslots):
            self.wviews.append(torch.frombuffer(
                self.shm.buf, dtype = torch.int16, count = self.wslot_size // 2,
                offset = self.layout["wstage_off"] + s * self.wslot_size))
        # Wslot flag banks sit after data_ready/done/consumed (keep in sync with
        # moe_handoff.h's layout comment and the watchdog unblock above)
        fl = self.gpu_base_ptr + MOE_SLOT_FLAGS_OFFSET + 3 * 64 * MOE_MAX_SLOTS
        self.stage_done_addr = [fl + s * 64 for s in range(MOE_MAX_WSLOTS)]
        self.pinned_free_addr = [fl + 64 * MOE_MAX_WSLOTS + s * 64 for s in range(MOE_MAX_WSLOTS)]
        self.v_stage_tail = u32[MOE_STAGE_TAIL_OFFSET // 4 : MOE_STAGE_TAIL_OFFSET // 4 + 1]
        self.v_stage_head = u32[MOE_STAGE_HEAD_OFFSET // 4 : MOE_STAGE_HEAD_OFFSET // 4 + 1]
        self.v_stage_jobs = np.frombuffer(
            self.shm.buf, dtype = np.uint32,
            offset = MOE_STAGE_JOBS_OFFSET, count = MOE_STAGE_RING * (MOE_JOB_BYTES // 4)).reshape(MOE_STAGE_RING, MOE_JOB_BYTES // 4)
        # Per-device CUDA state for the streamed-prefill path (copy stream, VRAM ring, events)
        self.sstate = {}

        self.conn.send(("start", self.shm.name, self.layout))

        import time
        t0 = time.time()
        while not self.v_ready[0]:
            if not self.proc.is_alive():
                raise RuntimeError("CPU MoE worker process died during startup")
            if time.time() - t0 > 60:
                raise RuntimeError("CPU MoE worker startup timeout")
            time.sleep(0.005)
        self.started = True
        self._flags_u32 = u32
        self._start_watchdog()
        kern = "avx512-vbmi" if ext.exl3_moe_cpu_has_avx512_vbmi() else \
               ("avx512-vnni" if ext.exl3_moe_cpu_has_avx512_vnni() else \
               ("avx2" if ext.exl3_moe_cpu_has_avx2() else "scalar"))
        print(f" -- CPU MoE worker started: {len(self.specs)} layers, {kern}, {self.threads} threads")

    def _start_watchdog(self):
        """
        GPU-side waits go through stream memops (no timeout, unlike the fallback kernel's 30s
        abort), so a dead worker would otherwise hang the stream forever. Detect it host-side
        and unblock every pending GEQ wait by writing satisfying values into the flags, then set
        the abort flag so the next begin_pass raises.
        """
        import threading, time

        def wd():
            while True:
                if not self.started or self.proc is None:
                    return
                if not self.proc.is_alive():
                    try:
                        u32 = self._flags_u32
                        seq, wseq = self.seq + 1, self.wseq + 1
                        done0 = MOE_SLOT_FLAGS_OFFSET + 64 * MOE_MAX_SLOTS
                        cons0 = MOE_SLOT_FLAGS_OFFSET + 2 * 64 * MOE_MAX_SLOTS
                        for s in range(MOE_MAX_SLOTS):
                            u32[(done0 + s * 64) // 4] = seq
                            u32[(cons0 + s * 64) // 4] = seq
                        fl = MOE_SLOT_FLAGS_OFFSET + 3 * 64 * MOE_MAX_SLOTS
                        for s in range(MOE_MAX_WSLOTS):
                            u32[(fl + s * 64) // 4] = wseq                          # stage_done
                            u32[(fl + 64 * MOE_MAX_WSLOTS + s * 64) // 4] = wseq    # pinned_free
                        self.v_abort[0] = 1
                    except Exception:
                        pass
                    return
                time.sleep(0.5)

        threading.Thread(target = wd, daemon = True).start()

    def begin_pass(self):
        if not self.started:
            self.ensure_started()
        if not self.started:
            return
        if self.v_abort[0]:
            raise RuntimeError("CPU MoE worker timed out (abort flag set)")
        self.v_pass_wake[0] += 1

    def submit(self, layer_idx, y, selected_experts, routing_weights):
        """
        Enqueue the routed-expert computation for one offloaded layer onto the current CUDA
        stream; returns the (asynchronously filled) float output tensor. Never synchronizes the
        host with the stream.
        """
        # Device guard: the flag kernels launch on the *current* device's current stream, which
        # need not match the layer's device (e.g. a model loaded entirely on cuda:1)
        with torch.cuda.device(y.device):
            spec = self.specs[layer_idx]
            h = y.shape[1]
            out = torch.empty((y.shape[0], h), dtype = torch.float, device = y.device)
            if os.environ.get("EXL3_MOE_SUBMIT_PROF"):
                if not hasattr(self, "_prof_ev"):
                    self._prof_ev = []
                ev0 = torch.cuda.Event(enable_timing = True)
                ev1 = torch.cuda.Event(enable_timing = True)
                ev0.record()
                jobs, rtmp = self._issue_compute(layer_idx, y, selected_experts, routing_weights, spec, out, h)
                self._collect_compute(jobs, out, rtmp, h)
                ev1.record()
                self._prof_ev.append((ev0, ev1))
                if len(self._prof_ev) >= 64:
                    done_ms = [a.elapsed_time(b) for a, b in self._prof_ev[:-1] if b.query()]
                    if done_ms:
                        done_ms.sort()
                        print(f" -- submit prof ({len(done_ms)} brackets, stream ms): "
                              f"med {done_ms[len(done_ms) // 2]:.3f} "
                              f"p90 {done_ms[int(len(done_ms) * 0.9)]:.3f} "
                              f"max {done_ms[-1]:.3f}", flush = True)
                    self._prof_ev = self._prof_ev[-1:]
            else:
                jobs, rtmp = self._issue_compute(layer_idx, y, selected_experts, routing_weights, spec, out, h)
                self._collect_compute(jobs, out, rtmp, h)
        return out

    def submit_issue(self, layer_idx, y, selected_experts, routing_weights):
        """
        Two-phase submit for the per-layer expert split: stage this layer's inputs and publish
        the compute job(s) nwo, defer the stream-side waits and readbacks to submit_collect so
        the caller can enqueue its own GPU expert work in between. That work then executes
        concurrently with the worker instead of behind the flag wait.
        """
        with torch.cuda.device(y.device):
            spec = self.specs[layer_idx]
            h = y.shape[1]
            out = torch.empty((y.shape[0], h), dtype = torch.float, device = y.device)
            jobs, rtmp = self._issue_compute(layer_idx, y, selected_experts, routing_weights, spec, out, h)
        return (jobs, rtmp, out, h, y.device)

    def submit_collect(self, handle):
        """Enqueue the deferred flag waits and readbacks; returns the (asynchronously
        filled) output tensor. Never synchronizes the host."""
        jobs, rtmp, out, h, dev = handle
        with torch.cuda.device(dev):
            self._collect_compute(jobs, out, rtmp, h)
        return out

    def submit_issue_fused(self, layer_idx, y, selected_experts, routing_weights,
                           split_map, split_hist, first_cpu):
        """
        Kernel-fused variant of submit_issue for decode-size jobs: ONE kernel launch stages
        sel/x/w straight into the pinned slot with zero-copy stores (replacing the int32
        cast, the zero-pad and three cudaMemcpyAsync launches), doubling as moe_split_map in
        dynamic-placement mode (in-place sel translate + hit histogram). When no selected
        expert is CPU-resident, the activation/weight payload is skipped entirely, the
        worker skips the compute, and the fused collect skips the readback: an inactive
        layer costs two flag memops and two near-empty kernels, and no PCIe payload.

        Returns None if the job shape does not fit the single-slot fast path (caller falls
        back to the copy path). selected_experts must hold RAW router ids here; translation
        (dynamic map or static tail offset) happens inside the kernel.
        """
        rows = y.shape[0]
        spec = self.specs[layer_idx]
        if rows > self.cap_rows or not (y.is_contiguous() and routing_weights.is_contiguous()):
            return None
        h_ = y.shape[1]
        hi = spec["hi"]
        dev = y.device
        with torch.cuda.device(dev):
            counts = self.dev_count.get(dev)
            if counts is None:
                counts = self.dev_count[dev] = \
                    torch.zeros((self.num_slots,), dtype = torch.int32, device = dev)
            slot_idx = self.next_slot
            self.next_slot = (self.next_slot + 1) % self.num_slots
            self.seq += 1
            seq = self.seq
            slot = self.slots[slot_idx]

            tail = int(self.v_jobs_tail[0])
            if tail - int(self.v_jobs_head[0]) >= MOE_JOB_RING - 4:
                import time
                while tail - int(self.v_jobs_head[0]) >= MOE_JOB_RING - 4:
                    if self.v_abort[0] or not self.proc.is_alive():
                        raise RuntimeError("CPU MoE worker failed (ring stall)")
                    time.sleep(0.0002)
            job = self.v_jobs[tail % MOE_JOB_RING]
            job[0] = seq
            job[1] = layer_idx
            job[2] = rows
            job[3] = spec["topk"]
            job[4] = slot_idx
            job[5] = 2    # MOE_JOB_KIND_COMPUTE_GATED
            self.v_jobs_tail[0] = tail + 1

            if self.slot_last_seq[slot_idx]:
                ext.exl3_moe_flag_wait(slot["consumed"], self.slot_last_seq[slot_idx],
                                       self.gpu_base_ptr + 128)
            ext.moe_split_issue(
                selected_experts.view(-1), split_map, split_hist,
                y, routing_weights,
                slot["sel_dev"], slot["x_dev"], slot["w_dev"],
                counts, slot_idx, hi, first_cpu,
            )
            ext.exl3_moe_flag_write(slot["data_ready"], seq)
            self.slot_last_seq[slot_idx] = seq
        return (seq, slot_idx, rows, h_, spec["ho"], dev, counts)

    def submit_collect_fused(self, handle, final_2d):
        """Fold the worker's partial into final_2d (rows, h) in place, straight from the
        pinned slot, or, for a job the issue kernel recorded as empty, do nothing (no
        PCIe reads). Never synchronizes the host."""
        seq, slot_idx, rows, h_, ho, dev, counts = handle
        slot = self.slots[slot_idx]
        with torch.cuda.device(dev):
            ext.exl3_moe_flag_wait(slot["done"], seq, self.gpu_base_ptr + 128)
            ext.moe_split_collect_add(final_2d, slot["out_dev"], counts, slot_idx, ho)
            ext.exl3_moe_flag_write(slot["consumed"], seq)

    def _issue_compute(self, layer_idx, y, selected_experts, routing_weights, spec, out, h):
        """
        Stage inputs and publish one compute job per cap_rows chunk (descriptors, D2H copies
        and data_ready flags only). Waits and readbacks are deferred to _collect_compute so
        other GPU work can be enqueued in between. BUT! Only up to num_slots jobs deep: a
        deeper batch inline-collects the (i - num_slots)-th job before reusing its slot.
        Without the window, slot reuse inside one batch waits on a consumed flag whose write
        (in the deferred collect) sits BEHIND the wait on the same stream (an in-stream
        deadlock (second-prompt freeze)) and even under the old done-flag gating the
        worker's next compute could overwrite a slot output the deferred readback had not
        fetched yet. Caller holds device guard.
        """
        rows = y.shape[0]
        h_ = y.shape[1]
        hi = spec["hi"]
        ho = spec["ho"]
        sel32 = selected_experts.to(torch.int32)
        # Zero-pad up to the quantized input width on the GPU so every D2H below is a contiguous
        # full-width block (see the assert in ensure_started for why this matters)
        y_pad = torch.nn.functional.pad(y, (0, hi - h_)) if hi != h_ else y
        rtmp = torch.empty((min(self.cap_rows, rows), ho), dtype = torch.float, device = y.device) \
            if ho != h_ else None
        jobs = []
        for a in range(0, rows, self.cap_rows):
            if len(jobs) >= self.num_slots:
                self._collect_one(jobs.pop(0), out, rtmp, h)
            b = min(a + self.cap_rows, rows)
            n = b - a
            slot_idx = self.next_slot
            self.next_slot = (self.next_slot + 1) % self.num_slots
            self.seq += 1
            seq = self.seq
            slot = self.slots[slot_idx]

            # Descriptor first (host-visible before the GPU can publish the data flag). Throttle
            # against ring overflow: a long prefill enqueues every job of the pass with no host
            # sync, and overwriting unconsumed descriptors corrupts the whole stream
            tail = int(self.v_jobs_tail[0])
            if tail - int(self.v_jobs_head[0]) >= MOE_JOB_RING - 4:
                import time
                while tail - int(self.v_jobs_head[0]) >= MOE_JOB_RING - 4:
                    if self.v_abort[0] or not self.proc.is_alive():
                        raise RuntimeError("CPU MoE worker failed (ring stall)")
                    time.sleep(0.0002)
            job = self.v_jobs[tail % MOE_JOB_RING]
            job[0] = seq
            job[1] = layer_idx
            job[2] = n
            job[3] = spec["topk"]
            job[4] = slot_idx
            job[5] = 0    # MOE_JOB_KIND_COMPUTE
            self.v_jobs_tail[0] = tail + 1

            # Serialize on the previous tenant's output having been read back (consumed flag,
            # written by the collecting stream after its D2H), not merely computed (done flag):
            # with offloaded layers spread over multiple devices, the previous tenant's collect
            # may sit queued on a different stream than this issue
            if self.slot_last_seq[slot_idx]:
                ext.exl3_moe_flag_wait(slot["consumed"], self.slot_last_seq[slot_idx], self.gpu_base_ptr + 128)
            slot["x"][:n].copy_(y_pad[a:b], non_blocking = True)
            slot["sel"][:n].copy_(sel32[a:b], non_blocking = True)
            slot["w"][:n].copy_(routing_weights[a:b], non_blocking = True)
            ext.exl3_moe_flag_write(slot["data_ready"], seq)
            self.slot_last_seq[slot_idx] = seq
            jobs.append((seq, slot_idx, a, b, n))
        return jobs, rtmp

    def _collect_one(self, job, out, rtmp, h):
        seq, slot_idx, a, b, n = job
        slot = self.slots[slot_idx]
        ext.exl3_moe_flag_wait(slot["done"], seq, self.gpu_base_ptr + 128)
        if rtmp is None:
            out[a:b].copy_(slot["out"][:n], non_blocking = True)
        else:
            rtmp[:n].copy_(slot["out"][:n], non_blocking = True)
            out[a:b] = rtmp[:n, :h]
        # Publish consumption AFTER the readback on this same stream: the slot's next tenant
        # (possibly issuing from another device) gates on this
        ext.exl3_moe_flag_write(slot["consumed"], seq)

    def _collect_compute(self, jobs, out, rtmp, h):
        """Wait for each still-pending job and read its output back; the padded width is
        trimmed on the GPU. Jobs beyond the slot count were already collected inline by the
        issue loop's sliding window. Caller holds the device guard."""
        for job in jobs:
            self._collect_one(job, out, rtmp, h)

    def _ensure_stream_state(self, device):
        key = torch.device(device).index or 0
        st = self.sstate.get(key)
        if st is not None:
            return st
        # Reconstruct scratch sized for the largest projection
        mx = 0
        for s in self.specs:
            pd = s.get("proj_dims")
            if pd:
                for k in ("g", "u", "d"):
                    if pd.get(k):
                        mx = max(mx, pd[k][0] * pd[k][1])
        st = dict(
            copy_stream = torch.cuda.Stream(device = device),
            vram_slots = [torch.empty(self.wslot_size // 2, dtype = torch.int16, device = device)
                          for _ in range(self.num_wslots)],
            wready_ev = [torch.cuda.Event() for _ in range(self.num_wslots)],
            wconsumed_ev = [torch.cuda.Event() for _ in range(self.num_wslots)],
            wslot_used = [False] * self.num_wslots,
            w_scratch = torch.empty(mx, dtype = torch.half, device = device) if mx else None,
            # Fused-tier (exl3_moe) temp buffers, allocated lazily per (hidden, intermediate)
            # shape: the kernel reads both dims from the buffers, so they must match the layer
            fused_t = TUNING.stream_fused_t,
            fused_bufs = {},
        )

        # Probe pinned->device bandwidth once: the break-even assignment count for streaming an
        # expert scales inversely with the link's bandwidth, so a chipset-attached x4 card needs
        # a much hotter expert to justify the weight DMA than a CPU-direct x16 one. An explicit
        # EXL3_MOE_STREAM_T overrides the scaling.
        probe = min(self.wslot_size, 16 << 20)
        ev0, ev1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        with torch.cuda.stream(st["copy_stream"]):
            for _ in range(2):   # warm-up: first transfer pays wakeup/pagetable costs
                st["vram_slots"][0][:probe // 2].copy_(self.wviews[0][:probe // 2],
                                                       non_blocking = True)
            ev0.record(st["copy_stream"])
            st["vram_slots"][0][:probe // 2].copy_(self.wviews[0][:probe // 2],
                                                   non_blocking = True)
            ev1.record(st["copy_stream"])
        ev1.synchronize()
        bw = probe / (ev0.elapsed_time(ev1) * 1e-3) / 1e9   # GB/s
        st["bw"] = bw
        if TUNING.stream_t_explicit:
            st["stream_t"] = self.stream_t
        else:
            st["stream_t"] = max(self.stream_t, int(self.stream_t * 25.0 / max(bw, 0.5)))
        if TUNING.stream_debug:
            print(f" -- stream state cuda:{key}: pinned->device {bw:.1f} GB/s, "
                  f"stream_t {st['stream_t']}")
        self.sstate[key] = st
        return st

    def _dq_linear(self, x, trellis_view, dims, suh, svh, bias, w_scratch):
        """reconstruct-path linear: had_in(x * suh) @ W -> had_out * svh (+ bias)"""
        k, n, K = dims
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, suh, None, 1.0)
        w = w_scratch[:k * n].view(k, n)
        ext.reconstruct(w, trellis_view, K, False, True)
        y = torch.empty((x.shape[0], n), dtype = torch.half, device = x.device)
        ext.hgemm(xh, w, y)
        ext.had_r_128(y, y, None, svh, 1.0)
        if bias is not None:
            y += bias
        return y

    def _act(self, spec, g, u):
        act = spec["activation"]
        if act in (0, 1):
            # Nonzero act_limit clamps up symmetrically and the activated gate from above,
            # before the multiply (mirrors the act_mul kernels; DS4 ships swiglu_limit = 10
            # with plain silu)
            fn = torch.nn.functional.silu if act == 0 else torch.nn.functional.gelu
            av, uf = fn(g.float()), u.float()
            lim = spec["act_limit"]
            if lim:
                av = av.clamp(max = lim)
                uf = uf.clamp(-lim, lim)
            return (av * uf).half()
        if act == 3:
            lim = spec["act_limit"]
            gf = g.float().clamp(max = lim)
            uf = u.float().clamp(-lim, lim)
            return ((uf + 1.0) * gf * torch.sigmoid(1.702 * gf)).half()
        uf = torch.nn.functional.relu(u.float())
        return (uf * uf).half()

    def install_expert(self, layer_idx, local_idx, keys):
        """Dynamic placement: replace worker expert `local_idx` of `layer_idx` with the
        checkpoint tensors at `keys` (per-projection prefixes, gate first when gated). The
        caller must have quiesced (full stream sync => job ring drained, worker idle) before
        calling; blocks until the child acks the in-place arena copy."""
        self.conn.send(("install", layer_idx, local_idx, keys))
        msg = self.conn.recv()
        if msg[0] != "ok":
            raise RuntimeError(f"CPU MoE worker expert install failed: {msg[1] if len(msg) > 1 else msg}")

    def submit_prefill(self, layer_idx, y, selected_experts, routing_weights):
        """
        Split the routed-expert workload by per-expert token count: hot experts (count >=
        stream_t) have their weights staged by the worker, DMA'd through the pinned ring to a
        small VRAM ring on a copy stream, and computed on the GPU via the reconstruct path,
        while the cold tail runs on the CPU, compressed to the rows that still have at least one
        unmasked assignment. Tail jobs are issued before the streamed batches and collected
        after them, so the CPU works the tail while the GPU streams. Falls back to the plain CPU
        path when nothing qualifies.
        """
        spec = self.specs[layer_idx]
        rows = y.shape[0]
        if (rows < self.stream_min_rows or spec.get("expert_bytes") is None
                or spec["expert_bytes"] > self.wslot_size or layer_idx not in self.aux):
            return self.submit(layer_idx, y, selected_experts, routing_weights)

        with torch.cuda.device(y.device):
            st = self._ensure_stream_state(y.device)
            E = spec["num_experts"]
            flat = selected_experts.reshape(-1)
            # Shifted histogram so any -1 sentinels land in bin 0 instead of polluting expert 0.
            # scatter_add, not torch.bincount: bincount hides two blocking min/max reductions
            # (negative-input validation and output sizing), leaving the tolist below as the only
            # sync before the tail-row compression
            shifted = flat + 1
            counts1 = torch.zeros(E + 1, dtype = torch.long, device = flat.device)
            counts1.scatter_add_(0, shifted, torch.ones_like(shifted))
            counts1_h = counts1.tolist()
            neg, counts_h = counts1_h[0], counts1_h[1:]
            streamed = [e for e in range(E) if counts_h[e] >= st["stream_t"]]
            if TUNING.stream_debug:
                n_str = sum(counts_h[e] for e in streamed)
                print(f" -- stream L{layer_idx}: rows {rows}, streamed experts "
                      f"{len(streamed)}/{E}, assignments {n_str}/{sum(counts_h)}")
            if not streamed:
                return self.submit(layer_idx, y, selected_experts, routing_weights)
            return self._submit_prefill_streamed(
                layer_idx, y, selected_experts, routing_weights, spec, streamed, st,
                counts_h, flat, shifted, neg)

    def _submit_prefill_streamed(self, layer_idx, y, selected_experts, routing_weights, spec,
                                 streamed, st, counts_h, flat, shifted, neg):
        rows = y.shape[0]
        h = y.shape[1]
        E = spec["num_experts"]
        topk = selected_experts.shape[1]
        out = torch.zeros((rows, h), dtype = torch.float, device = y.device)

        # Group assignments by expert once: every expert's token segment is then a slice at
        # host-known prefix offsets. Anything per-expert/per-batch from here on is sync-free —
        # per-expert nonzero() would pin the host to the stream position and collapse the
        # copy-stream lookahead into lockstep with compute
        order = torch.argsort(flat)
        token_sorted = torch.div(order, topk, rounding_mode = "floor")
        weight_sorted = routing_weights.reshape(-1).index_select(0, order)
        offs = [neg]
        for c in counts_h:
            offs.append(offs[-1] + c)

        # CPU tail: mask streamed assignments, then compress to rows that still carry work (a
        # nearly-fully-streamed layer otherwise pays a full cap_rows-chunked pass of no-ops).
        # Issue only; the waits and readbacks come after the streamed batches are enqueued.
        # The streamed table is built host-side and indexed with the shifted ids (entry 0 is the
        # -1 sentinel, always False), replacing the index_put/clamp/compare/and kernel chain
        table = np.zeros(E + 1, dtype = np.bool_)
        for e in streamed:
            table[e + 1] = True
        smask1 = torch.from_numpy(table).to(y.device, non_blocking = True)
        is_streamed = smask1.index_select(0, shifted)
        sel_tail = flat.masked_fill(is_streamed, -1).view(rows, topk)
        tidx = (sel_tail >= 0).any(dim = 1).nonzero(as_tuple = True)[0]
        n_tail = tidx.shape[0]
        tail_jobs = None
        if n_tail:
            out_t = torch.empty((n_tail, h), dtype = torch.float, device = y.device)
            tail_jobs, rtmp = self._issue_compute(
                layer_idx, y.index_select(0, tidx), sel_tail.index_select(0, tidx),
                routing_weights.index_select(0, tidx), spec, out_t, h)

        aux = self.aux[layer_idx]
        pd = spec["proj_dims"]
        gb, ub, db = spec["proj_bytes"]
        exp_b = spec["expert_bytes"]
        per_slot = min(self.wslot_size // exp_b, self.batch_experts)
        gated = pd.get("g") is not None
        abort = self.gpu_base_ptr + 128
        copy_stream = st["copy_stream"]

        # Mid-tier experts (count <= fused_t) run through the fused MoE kernel per staged batch;
        # experts too hot for the temp buffers take the per-expert reconstruct path. Same
        # eligibility as support_fused on the GPU side: mul1 (given), silu/gelu gated or relu2
        # gateless, no per-expert biases, no padded dims
        fused_t = st["fused_t"] if (
            spec["activation"] in (0, 1, 2) and spec["hi"] == h and spec["ho"] == h
            and not any(aux.get(b) is not None for b in ("bias_g", "bias_u", "bias_d"))
        ) else 0
        fbufs = None
        if fused_t and any(counts_h[e] <= fused_t for e in streamed):
            key = (spec["hi"], pd["u"][1])
            fbufs = st["fused_bufs"].get(key)
            if fbufs is None:
                conc = ext.exl3_moe_max_concurrency(torch.device(y.device).index or 0)
                fbufs = tuple(
                    torch.empty((conc, st["fused_t"], dim), dtype = torch.half, device = y.device)
                    for dim in (key[0], key[0], key[1], key[1]))
                st["fused_bufs"][key] = fbufs

        for i0 in range(0, len(streamed), per_slot):
            batch = streamed[i0:i0 + per_slot]
            ws = self.next_wslot
            self.next_wslot = (self.next_wslot + 1) % self.num_wslots
            self.wseq += 1
            seq = self.wseq

            # Stage job: its own ring, consumed by the worker's dedicated stager thread, so the
            # weight memcpys overlap the compute pool's work on the tail
            stail = int(self.v_stage_tail[0])
            while stail - int(self.v_stage_head[0]) >= MOE_STAGE_RING - 2:
                import time
                if self.v_abort[0] or not self.proc.is_alive():
                    raise RuntimeError("CPU MoE worker failed (stage ring stall)")
                time.sleep(0.0002)
            job = self.v_stage_jobs[stail % MOE_STAGE_RING]
            job[0] = seq
            job[1] = layer_idx
            job[2] = len(batch)
            job[3] = 0
            job[4] = ws
            job[5] = 1    # MOE_JOB_KIND_STAGE
            job[6] = self.wslot_prev_seq[ws]
            for bi, e in enumerate(batch):
                job[7 + bi] = e
            self.v_stage_tail[0] = stail + 1
            self.wslot_prev_seq[ws] = seq

            used = (len(batch) * exp_b) // 2
            with torch.cuda.stream(copy_stream):
                if st["wslot_used"][ws]:
                    copy_stream.wait_event(st["wconsumed_ev"][ws])
                ext.exl3_moe_flag_wait(self.stage_done_addr[ws], seq, abort)
                st["vram_slots"][ws][:used].copy_(self.wviews[ws][:used], non_blocking = True)
                ext.exl3_moe_flag_write(self.pinned_free_addr[ws], seq)
                st["wready_ev"][ws].record(copy_stream)
            st["wslot_used"][ws] = True

            # Compute the batch on the current stream once the DMA lands
            torch.cuda.current_stream().wait_event(st["wready_ev"][ws])
            vslot = st["vram_slots"][ws]
            per_e = [(bi, e, token_sorted[offs[e] : offs[e] + counts_h[e]],
                      weight_sorted[offs[e] : offs[e] + counts_h[e]])
                     for bi, e in enumerate(batch)]

            # Mid tier: one fused kernel over the batch's cooler experts. Heavy experts stay in
            # the descriptor (the kernel skips counts above the temp-row capacity) so the
            # token_sorted segments line up with expert_count
            n_fused = sum(1 for _, e, _, _ in per_e if counts_h[e] <= fused_t) if fused_t else 0
            if TUNING.stream_debug:
                print(f" --   batch L{layer_idx} ws{ws}: {len(batch)} experts, fused_t {fused_t}, "
                      f"n_fused {n_fused}, counts {[counts_h[e] for e in batch]}")
            if n_fused:
                base = vslot.data_ptr()
                tbl = [[] for _ in range(9)]
                for bi, e, _, _ in per_e:
                    bb = bi * exp_b
                    if gated:
                        tbl[0].append(base + bb)
                        tbl[1].append(aux["suh_g"][e].data_ptr())
                        tbl[2].append(aux["svh_g"][e].data_ptr())
                    tbl[3].append(base + bb + gb)
                    tbl[4].append(aux["suh_u"][e].data_ptr())
                    tbl[5].append(aux["svh_u"][e].data_ptr())
                    tbl[6].append(base + bb + gb + ub)
                    tbl[7].append(aux["suh_d"][e].data_ptr())
                    tbl[8].append(aux["svh_d"][e].data_ptr())
                if not gated:
                    # Placeholder gate tables, never dereferenced (gate GEMM is skipped)
                    for i in (0, 1, 2):
                        tbl[i] = tbl[i + 3]
                tblt = torch.tensor(tbl, dtype = torch.int64).to(y.device, non_blocking = True)
                ec = torch.tensor([counts_h[e] for _, e, _, _ in per_e] + [0],
                                  dtype = torch.long).to(y.device, non_blocking = True)
                tok = torch.cat([seg for _, _, seg, _ in per_e])
                wts = torch.cat([wseg for _, _, _, wseg in per_e]).half()
                Ku, Kd = pd["u"][2], pd["d"][2]
                Kg = pd["g"][2] if gated else Ku
                ext.exl3_moe(
                    y, out, ec, tok, wts,
                    fbufs[0], fbufs[1], fbufs[2], fbufs[3],
                    spec["activation"], Kg, Ku, Kd,
                    tblt[0], tblt[1], tblt[2], tblt[3], tblt[4], tblt[5],
                    tblt[6], tblt[7], tblt[8],
                    False, True, False, True, False, True,
                    float(spec["act_limit"] or 0.0), n_fused)

            # Heavy tier: per-expert reconstruct
            for bi, e, idx, wseg in per_e:
                if fused_t and counts_h[e] <= fused_t:
                    continue
                boff = (bi * exp_b) // 2
                xg = y.index_select(0, idx)
                # Zero-pad to the quantized input width (the had transform requires it)
                hi = spec["hi"]
                if xg.shape[1] != hi:
                    xg = torch.nn.functional.pad(xg, (0, hi - xg.shape[1]))
                we = wseg.float().unsqueeze(1)
                def tview(off_b, dims):
                    k, n, K = dims
                    numel = (k // 16) * (n // 16) * 16 * K
                    return vslot[boff + off_b // 2 : boff + off_b // 2 + numel] \
                        .view(k // 16, n // 16, 16 * K)
                if gated:
                    gy = self._dq_linear(xg, tview(0, pd["g"]), pd["g"],
                                         aux["suh_g"][e], aux["svh_g"][e],
                                         aux["bias_g"][e] if aux.get("bias_g") else None,
                                         st["w_scratch"])
                uy = self._dq_linear(xg, tview(gb, pd["u"]), pd["u"],
                                     aux["suh_u"][e], aux["svh_u"][e],
                                     aux["bias_u"][e] if aux.get("bias_u") else None,
                                     st["w_scratch"])
                a = self._act(spec, gy if gated else None, uy) if gated else self._act(spec, None, uy)
                dy = self._dq_linear(a, tview(gb + ub, pd["d"]), pd["d"],
                                     aux["suh_d"][e], aux["svh_d"][e],
                                     aux["bias_d"][e] if aux.get("bias_d") else None,
                                     st["w_scratch"])
                out.index_add_(0, idx, dy[:, :h].float() * we)
            st["wconsumed_ev"][ws].record(torch.cuda.current_stream())

        # Collect the CPU tail (by now usually complete) and merge
        if tail_jobs:
            self._collect_compute(tail_jobs, out_t, rtmp, h)
            out.index_add_(0, tidx, out_t)
        return out

    def unregister(self):
        # Called per offloaded layer on unload; shut down when the model releases the last
        # one. A LIVE COUNT, not specs.pop(): register_layer hands out stable indices into
        # specs (and returns cached indices on autosplit rollback retries), so removing
        # entries desyncs every later layer's index and the ack bookkeeping — under tight
        # autosplit budgets with split layers on every device this hung the load
        if self.live_layers == 0:
            return
        self.live_layers -= 1
        if self.live_layers == 0:
            self.shutdown()

    def shutdown(self):
        if self.proc is not None:
            try:
                if self.started and self.shm is not None:
                    self.v_quit[0] = 1
                    # The child's main thread serves the pipe at runtime (expert installs);
                    # the flag stops its compute thread, the message unblocks the recv loop
                    if self.conn is not None:
                        self.conn.send(("quit",))
                elif self.conn is not None:
                    self.conn.send(("quit",))
                self.proc.join(timeout = 5)
                if self.proc.is_alive():
                    self.proc.terminate()
                    self.proc.join(timeout = 2)
                if self.proc.is_alive():
                    self.proc.kill()
            except Exception:
                pass
            self.proc = None
        # Full reset: a later re-registration must not resolve stale indices against a child
        # that no longer exists
        self.specs = []
        self.by_key = {}
        self.aux = {}
        self.acked = 0
        self.live_layers = 0
        cleanupper.unregister_atexit(self.shutdown)
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
        if self.shm is not None:
            try:
                cuda_host_unregister(self.base_ptr)
            except Exception:
                pass
            # Drop every view over the buffer before closing, or mmap refuses to unmap
            self.slots = None
            self.wviews = None
            self.sstate = None
            self.v_quit = self.v_pass_wake = self.v_abort = self.v_ready = None
            self.v_jobs_tail = self.v_jobs_head = self.v_jobs = None
            self.v_stage_tail = self.v_stage_head = self.v_stage_jobs = None
            self._flags_u32 = None
            import gc
            gc.collect()
            try:
                self.shm.close()
                self.shm.unlink()
            except Exception:
                pass
            self.shm = None
        self.started = False
        self.by_key = {}
