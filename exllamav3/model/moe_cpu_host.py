from __future__ import annotations
import os
import sys
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

The child's expert-weight arena is a set of shared-memory chunks that the parent also maps
and page-locks: GPU-streamed prefill DMAs expert blocks straight out of the arena into a VRAM
ring (no host-side staging copy, so the only DRAM traffic per streamed byte is the DMA read)
and repacks the band-swizzled tiles into native order on the GPU.

Shared-memory layout constants mirror cpu/moe_handoff.h.
"""

MOE_JOB_RING = 256
MOE_MAX_SLOTS = 8
MOE_JOB_BYTES = 8 * 4   # sizeof(MoeJob): six uint32 fields + two uint32 pad
MOE_CTRL_JOBS_OFFSET = 384
MOE_SLOT_FLAGS_OFFSET = MOE_CTRL_JOBS_OFFSET + MOE_JOB_RING * MOE_JOB_BYTES
MOE_FLAGS_SIZE = 3 * 64 * MOE_MAX_SLOTS
MOE_CTRL_SIZE = MOE_SLOT_FLAGS_OFFSET + MOE_FLAGS_SIZE
MOE_ARENA_CHUNK = 1 << 30


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
        assert 1 <= self.num_slots <= 8, "EXL3_MOE_CPU_SLOTS must be 1..8 (MOE_MAX_SLOTS in moe_handoff.h)"
        self.cap_rows = int(os.environ.get("EXL3_MOE_CPU_SLOT_ROWS", 64))
        # Thread count fallback chain ends here; config.infer_params.moe_cpu_threads (or the
        # draft/MTP equivalent) takes precedence per host when set (MoeCpuHost.__init__)
        self.threads = int(os.environ.get("EXL3_MOE_CPU_THREADS", max(1, (os.cpu_count() or 2) // 2)))
        # VRAM ring for streamed expert weights: num_wslots compute slots (native tile order)
        # plus two raw DMA landing slots, each wslot_size bytes
        self.num_wslots = int(os.environ.get("EXL3_MOE_CPU_WSLOTS", 2))
        self.wslot_size = int(os.environ.get("EXL3_MOE_CPU_WSLOT_MB", 32)) * 1024 * 1024
        # Band-contiguous ("swizzled") expert trellis layout: repacked at arena rehome so each
        # 8-tile output band streams sequentially from DRAM. Only applied when the VBMI kernel
        # tier is active. EXL3_MOE_CPU_SWIZZLE=0 restores the native layout.
        self.swizzle = os.environ.get("EXL3_MOE_CPU_SWIZZLE", "1") != "0"

        # --- GPU-streaming prefill ---
        self.stream_t_explicit = "EXL3_MOE_STREAM_T" in os.environ
        self.stream_t = int(os.environ.get("EXL3_MOE_STREAM_T", 8))
        self.stream_fused_t = int(os.environ.get("EXL3_MOE_STREAM_FUSED_T", 512))
        self.stream_min_rows = int(os.environ.get("EXL3_MOE_STREAM_MIN_ROWS", 32))
        self.batch_experts = max(1, int(os.environ.get("EXL3_MOE_STREAM_BATCH_EXPERTS", 24)))

        # --- debug / kill switches ---
        self.stream_debug = bool(os.environ.get("EXL3_MOE_STREAM_DEBUG"))
        # Per-layer streamed-prefill timing: router sync wait, host enqueue wall time, and the
        # compute stream's span for the layer, averaged over 48 layers
        self.stream_prof = bool(os.environ.get("EXL3_MOE_STREAM_PROF"))
        self.cpu_prof = bool(os.environ.get("EXL3_MOE_CPU_PROF"))
        # Stream memops lose to the kernel waits under WDDM (measured 23-26 vs 28-30 tok/s
        # decode on an RTX 4090), so Windows defaults to the kernel path
        self.memops = os.environ.get("EXL3_MOE_MEMOPS", "0" if os.name == "nt" else "1") != "0"


TUNING = MoeCpuTuning()
ext.exl3_moe_cpu_set_memops(TUNING.memops)


def _stream_prof_line(pr, L, rows):
    """One EXL3_MOE_STREAM_PROF report. Host-side figures cover the L layers of the pass; the
    GPU-side figures are harvested one layer late (at the next router sync), so they are
    normalized by the number of layers actually harvested this period (gpu_n), which lags the
    pass by one and would otherwise divide by zero at L == 1."""
    g = max(pr["gpu_n"], 1)
    return (f" -- stream prof ({pr['n']} layers, rows {rows}, ms/layer): router-sync "
            f"{pr['sync'] / L * 1e3:.2f} host-enqueue {pr['host'] / L * 1e3:.2f} "
            f"gpu-span {pr['gpu'] / g:.2f} | batches/layer {pr['batches'] / L:.1f} | "
            f"per layer: raw-slot-wait {pr['rawwait'] / g:.2f} "
            f"dma {pr['dma'] / g:.2f} compute {pr['compute'] / g:.2f}")


def _proj_swizzled(arena_swz, K):
    """Physical order of one trellis projection in the arena: band-swizzled when the worker
    swizzles at all, except K8 matrices, which always stay in native tile order. The worker
    reports arena_swz once at startup; both sides derive every projection's layout from it"""
    return bool(arena_swz) and K != 8


def _stream_per_slot(wslot_size, exp_b, batch_experts):
    """Experts per streamed batch: bounded by the compute slot's capacity and the batch cap"""
    return min(wslot_size // exp_b, batch_experts)


class _SharedArena:
    """
    Growable pool of 1 GiB shared-memory chunks that expert weights are copied into. The
    chunks outlive every tensor handed out (held for the process lifetime). The parent maps
    the same chunks by name and page-locks them, so the tensors' (chunk, offset) locations
    double as DMA sources for streamed prefill.
    """

    def __init__(self, conn = None):
        self.chunks = []
        self.cur = None
        self.cur_off = 0
        # Parent-side pipe: every chunk's name is published the moment it exists, so the
        # parent can release it however the worker ends
        self.conn = conn

    def _new_chunk(self, min_bytes):
        size = max(MOE_ARENA_CHUNK, (min_bytes + (2 << 20) - 1) & ~((2 << 20) - 1))
        self._check_shm_capacity(size)
        shm = shared_memory.SharedMemory(create = True, size = size)
        self.chunks.append(shm)
        if self.conn is not None:
            self.conn.send(("chunk", shm.name, shm.size))
        self.cur = shm
        self.cur_off = 0
        if os.environ.get("EXL3_MOE_ARENA_DEBUG"):
            total = sum(c.size for c in self.chunks)
            print(f" -- arena: new chunk {size/1e6:.1f} MB, {len(self.chunks)} chunks, "
                  f"{total/1e9:.3f} GB total", flush = True)

    def _check_shm_capacity(self, size):
        """POSIX shared memory is a tmpfs (/dev/shm) and ftruncate succeeds lazily, so an
        undersized mount (Docker's default is 64 MiB) would only surface as SIGBUS when the
        pages are first written. Fail here instead, with the sizing the deployment needs."""
        if not hasattr(os, "statvfs"):
            return
        try:
            st = os.statvfs("/dev/shm")
        except OSError:
            return
        avail = st.f_bavail * st.f_frsize
        if size <= avail:
            return
        have = sum(c.size for c in self.chunks)
        raise RuntimeError(
            f"CPU MoE shared arena: /dev/shm has {avail / 2**30:.2f} GiB available, the next "
            f"{size / 2**30:.2f} GiB chunk does not fit ({have / 2**30:.2f} GiB allocated so "
            f"far). Every CPU-offloaded expert lives in /dev/shm, which must hold the whole "
            f"offloaded set; enlarge it (Docker: --shm-size, Compose: shm_size)")

    def reserve(self, nbytes):
        """Make sure the next `nbytes` of rehomes land contiguously in one chunk; returns the
        (chunk index, offset) they will start at"""
        if self.cur is None or self.cur_off + nbytes > self.cur.size:
            self._new_chunk(nbytes)
        return len(self.chunks) - 1, self.cur_off

    def rehome(self, tensor, band_swizzle = False):
        """Copy `tensor` into the arena and return a same-dtype/shape view over the copy.

        band_swizzle: repack a [k/16, n/16, 16K] trellis tensor band-contiguous during the
        copy -- physical order becomes (group n/128, k-tile, member, tile), one strided copy_.
        The returned view keeps the original logical shape; only the byte order differs
        (consumed by the swz-aware kernels in moe_mul1.cpp, undone on the GPU by
        moe_unswizzle_trellis for streamed prefill)."""
        if tensor is None or tensor.numel() == 0:
            return tensor
        nbytes = tensor.numel() * tensor.element_size()
        aligned = _align64(nbytes)
        self.reserve(aligned)
        off = self.cur_off
        self.cur_off += aligned
        dst = torch.frombuffer(self.cur.buf[off : off + nbytes], dtype = torch.uint8)
        if band_swizzle:
            tk, tn, ps = tensor.shape
            dst.view(tensor.dtype).view(tn // 8, tk, 8, ps) \
               .copy_(tensor.view(tk, tn // 8, 8, ps).permute(1, 0, 2, 3))
        else:
            dst.copy_(tensor.contiguous().view(torch.uint8).reshape(-1))
        return dst.view(tensor.dtype).view(tensor.shape)


def _moe_cpu_child_main(conn, model_dir, threads, swizzle):
    """
    Child entry point: receives ("layer", spec) messages, loading each layer's expert tensors
    (deferred, multithreaded) into the shared arena (publishing each chunk's name as ("chunk",
    name, size) on creation) and acking, until ("start", shm_name, layout) switches it into the
    worker loop, after replying ("arena", per-layer expert block locations, swizzled) so the
    parent can map the arena and knows the physical trellis order. `swizzle` is the parent's
    snapshot of the tuning request; the actual order also depends on this CPU's VBMI support,
    which is why the child reports it back. Errors are reported over the pipe before exiting.
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

    def leave():
        """Clean exit without interpreter teardown: the arena chunks stay exported to the
        extension's layer tensors for the process lifetime, so releasing them (at function
        return or at teardown) would have SharedMemory.__del__ fail to unmap each one
        (BufferError). The parent holds its own mapping; the kernel releases this one"""
        if shm is not None:
            shm.close()
        conn.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    try:
        stc = SafetensorsCollection(model_dir)
        cpu = torch.device("cpu")
        # The parent releases the chunks at shutdown: it learns each name on creation
        arena = _SharedArena(conn)

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
        swz = bool(swizzle and cext.exl3_moe_cpu_has_avx512_vbmi())

        def rehome_trellis(t):
            return arena.rehome(t, band_swizzle = _proj_swizzled(swz, t.shape[2] // 16))

        def nbytes(t):
            return t.numel() * t.element_size()

        def rehome_experts(g, u, d):
            """Per expert: the gate/up/down trellis tensors back to back (one contiguous
            expert block, the DMA unit for streamed prefill), then the small aux tensors.
            Returns the per-projection lists and the per-expert (chunk, offset) of the block."""
            gs, us, ds, blocks = [], [], [], []
            for e in range(len(u)):
                projs = ([g[e]] if g else []) + [u[e], d[e]]
                total = sum(nbytes(t[0]) for t in projs)
                assert all(nbytes(t[0]) % 64 == 0 for t in projs), "trellis size not 64-byte aligned"
                blocks.append(arena.reserve(total))
                trellis = [rehome_trellis(t[0]) for t in projs]
                aux = [(arena.rehome(t[1]), arena.rehome(t[2]), arena.rehome(t[3])) for t in projs]
                if g:
                    gs.append((trellis[0],) + aux[0])
                us.append((trellis[-2],) + aux[-2])
                ds.append((trellis[-1],) + aux[-1])
            return gs, us, ds, blocks

        def biases(ts):
            return [t[3] for t in ts] if ts and ts[0][3] is not None else []

        # Per-layer, per-expert arena views retained for runtime expert installs (dynamic
        # placement): the arena memory is what the compute kernels read, so an in-place copy
        # into these views (with the same swizzle transform rehome applied) replaces an
        # expert's weights. The parent quiesces (full sync, ring drained) before sending an
        # install, so the workers never observe a torn write
        layer_views = []
        layer_blocks = []

        while True:
            msg = conn.recv()
            if msg[0] == "layer":
                spec = msg[1]
                stc.begin_deferred_load()
                g = fetch(spec["gate_keys"])
                u = fetch(spec["up_keys"])
                d = fetch(spec["down_keys"])
                stc.end_deferred_load()
                # Copy into the shared arena now that the deferred reads have actually
                # populated these tensors
                g, u, d, blocks = rehome_experts(g, u, d)
                cext.exl3_moe_cpu_make_layer(
                    [t[0] for t in g], [t[1] for t in g], [t[2] for t in g],
                    [t[0] for t in u], [t[1] for t in u], [t[2] for t in u],
                    [t[0] for t in d], [t[1] for t in d], [t[2] for t in d],
                    biases(g), biases(u), biases(d),
                    spec["activation"], spec["act_limit"],
                    1 if swz else 0,
                )
                layer_views.append((g, u, d))
                layer_blocks.append(blocks)
                # Reclaim this layer's now-discarded loader tensors immediately (rehome_experts
                # copied everything into the arena)
                try:
                    ctypes.CDLL(None).malloc_trim(0)
                except Exception:
                    pass
                conn.send(("ok",))
            elif msg[0] == "start":
                shm_name, layout = msg[1], msg[2]
                break
            elif msg[0] == "quit":
                leave()

        conn.send(("arena", layer_blocks, swz))

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
                if _proj_swizzled(swz, v_tr.shape[2] // 16):
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
                threads,
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
        leave()
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
        self.v_quit = None
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
        # GPU-streaming prefill: experts with at least stream_t assigned tokens are streamed to
        # the GPU (weights DMA'd straight from the page-locked arena) while the tail stays on
        # the CPU
        self.num_wslots = TUNING.num_wslots
        self.wslot_size = TUNING.wslot_size
        self.stream_t = TUNING.stream_t
        self.stream_min_rows = TUNING.stream_min_rows
        self.batch_experts = TUNING.batch_experts
        # Requested trellis order, frozen here: a same-process tuning change after this host
        # exists must not reinterpret bytes the worker already packed. The worker reports the
        # actual order (arena_swz) at startup
        self.swizzle = TUNING.swizzle
        self.arena_swz = False
        self.next_wslot = 0
        self.next_rslot = 0
        self.aux = {}
        # Parent-side arena mappings (filled at start): uint8 views per chunk and, per layer,
        # per-expert (chunk, offset) of the contiguous [gate | up | down] trellis block
        self.arena = []
        self.arena_shm = []
        # Names of every arena chunk the worker has created, published on creation: the parent
        # owns their release regardless of how the worker ends
        self.arena_names = []
        self.blocks = []
        # EXL3_MOE_STREAM_PROF accumulators (per-layer timings of the streamed prefill path)
        self._sprof = None

    def _spawn(self):
        if self.proc is not None:
            return
        ctx = multiprocessing.get_context("spawn")
        self.conn, child_conn = ctx.Pipe(duplex = True)
        self.proc = ctx.Process(
            target = _moe_cpu_child_main,
            args = (child_conn, self.model_dir, self.threads, self.swizzle),
            daemon = True,
        )
        self.proc.start()
        child_conn.close()
        # Cleanupper fires at the end of the __main__ scope, before interpreter teardown breaks
        # the shm views and pipe machinery that shutdown() needs; PDEATHSIG in the child covers
        # the paths where no Python hook runs at all
        cleanupper.register_atexit(self.shutdown)

    def _pump(self, timeout):
        """Receive one message from the child. A worker error or death releases everything
        this host owns before raising, so a failed load leaves nothing behind"""
        if self.conn.poll(timeout):
            msg = self.conn.recv()
            if msg[0] == "ok":
                self.acked += 1
            elif msg[0] == "chunk":
                self.arena_names.append(msg[1])
            elif msg[0] == "err":
                self.shutdown()
                raise RuntimeError(f"CPU MoE worker failed:\n{msg[1]}")
            else:
                self.shutdown()
                raise RuntimeError(f"unexpected CPU MoE worker message {msg[0]!r}")
            return True
        if not self.proc.is_alive():
            self.shutdown()
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
        try:
            self._start()
        except BaseException:
            # Whatever failed (control block, a registration, the worker, the layout report):
            # release every registration, mapping and chunk name before surfacing it
            self.shutdown()
            raise

    def _start(self):
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

        self.layout["cpu_prof"] = TUNING.cpu_prof
        size = MOE_CTRL_SIZE + self.num_slots * slot_size
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

        # Per-device CUDA state for the streamed-prefill path (copy stream, VRAM ring, events)
        self.sstate = {}

        self.conn.send(("start", self.shm.name, self.layout))

        # Map and page-lock the child's arena so streamed prefill can DMA expert blocks
        # straight out of it. Registration of the whole arena is required: a chunk that
        # cannot be locked would silently fall back to a synchronous pageable copy
        while not self.conn.poll(1.0):
            if not self.proc.is_alive():
                raise RuntimeError("CPU MoE worker process died during startup")
        msg = self.conn.recv()
        if msg[0] == "err":
            raise RuntimeError(f"CPU MoE worker failed:\n{msg[1]}")
        if msg[0] != "arena":
            raise RuntimeError(f"unexpected CPU MoE worker message {msg[0]!r}")
        _, self.blocks, self.arena_swz = msg
        for name in self.arena_names:
            chunk = shared_memory.SharedMemory(name = name)
            self.arena_shm.append(chunk)
            view = torch.frombuffer(chunk.buf, dtype = torch.uint8)
            # Appended only once pinned: shutdown unregisters exactly what registered
            cuda_host_register(view.data_ptr(), chunk.size)
            self.arena.append(view)

        import time
        t0 = time.monotonic()
        while not self.v_ready[0]:
            if not self.proc.is_alive():
                raise RuntimeError("CPU MoE worker process died during startup")
            if time.monotonic() - t0 > 60:
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
                        seq = self.seq + 1
                        done0 = MOE_SLOT_FLAGS_OFFSET + 64 * MOE_MAX_SLOTS
                        cons0 = MOE_SLOT_FLAGS_OFFSET + 2 * 64 * MOE_MAX_SLOTS
                        for s in range(MOE_MAX_SLOTS):
                            u32[(done0 + s * 64) // 4] = seq
                            u32[(cons0 + s * 64) // 4] = seq
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
        # Experts arrive in the order the worker reported at startup (band-swizzled when the
        # VBMI tier owns them, K8 excepted per matrix); the GPU restores the native tile order
        # into the compute ring after each DMA
        swz = self.arena_swz
        st = dict(
            copy_stream = torch.cuda.Stream(device = device),
            # Compute slots hold native-order expert blocks; raw slots are the DMA landing
            # zone for the arena's bytes (repacked raw -> compute by moe_unswizzle_trellis)
            vram_slots = [torch.empty(self.wslot_size, dtype = torch.uint8, device = device)
                          for _ in range(self.num_wslots)],
            raw_slots = [torch.empty(self.wslot_size, dtype = torch.uint8, device = device)
                         for _ in range(2)],
            wready_ev = [torch.cuda.Event() for _ in range(2)],
            rfree_ev = [torch.cuda.Event() for _ in range(2)],
            rslot_used = [False] * 2,
            swz = swz,
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
        import time
        probe = min(self.wslot_size, 16 << 20)
        src = self.arena[0][:probe]
        ev0, ev1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        bw = 0.0
        with torch.cuda.stream(st["copy_stream"]):
            # An idle PCIe link sits in a low power state (the Windows driver drops it to Gen1
            # after a few idle seconds) and only retrains under sustained traffic. The retrain
            # arrives as a single step rather than a ramp, and its latency is not bounded by
            # anything we control: measured ~160 ms on an RTX 4090 under WDDM, on a link that
            # then holds 26.5 GB/s. A fixed warm-up budget is therefore a coin flip whenever the
            # step lands near it, and losing that flip is expensive -- the probe reads a Gen1/
            # Gen2 link, stream_t is scaled far too high, and the result is cached for the life
            # of the loaded model (measured: 6.8 GB/s read, stream_t 15 instead of 8, -18.1%
            # prefill, no recovery short of a reload).
            #
            # So time every copy and keep the best over a window long enough to contain the
            # retrain. A plateau alone cannot be trusted to mean "this link is slow": the
            # retrain passes through intermediate generations, and an intermediate plateau is
            # indistinguishable from a genuinely slow link by bandwidth alone (the failure
            # above read 6.8 GB/s, a Gen2 plateau, not the Gen1 floor). Hence a floor on total
            # observation time, with the settle window only allowed to end the probe after it.
            # A genuinely slow link pays the floor once per loaded model and keeps its low
            # reading and its high stream_t, which is what the calibration wants.
            floor = 2.0         # observe at least this long: ~12x the observed retrain latency
            settle = 0.5        # after the floor, best unimproved this long -> at the ceiling
            cap = 5.0           # bound for a link that never settles
            t0 = time.perf_counter()
            t_improved = t0
            while True:
                ev0.record(st["copy_stream"])
                st["raw_slots"][0][:probe].copy_(src, non_blocking = True)
                ev1.record(st["copy_stream"])
                ev1.synchronize()
                sample = probe / (ev0.elapsed_time(ev1) * 1e-3) / 1e9   # GB/s
                now = time.perf_counter()
                if sample > bw * 1.02:      # 2%: ignore sample noise, catch a link generation
                    t_improved = now
                bw = max(bw, sample)
                if now - t0 >= floor and now - t_improved >= settle:
                    break
                if now - t0 >= cap:
                    # Still climbing at the cap: the reading is known-bad rather than merely
                    # low, and stream_t derived from it will be too high. Say so -- silently
                    # caching it is what makes this failure invisible.
                    print(f" !! CPU MoE: pinned->device probe on cuda:{key} did not settle "
                          f"({bw:.1f} GB/s and still rising after {cap:.1f} s). stream_t may be "
                          f"too high; pin EXL3_MOE_STREAM_T to override.")
                    break
        st["bw"] = bw
        if TUNING.stream_t_explicit:
            st["stream_t"] = self.stream_t
        else:
            # Calibrated on Qwen3.8-Flash-Next (512 experts, 410 on the CPU, 12 threads) over
            # gen5 x16 (57 GB/s), gen5 x8 (29 GB/s) and gen4 x4 (6.7 GB/s) links: 8 was best
            # on both gen5 links (4 and 16 both slower), 16 on gen4 x4 (32 no better). The
            # break-even count grows with the square root of the bandwidth deficit, not
            # linearly: the tail's CPU cost falls with the same rows the streaming gains
            st["stream_t"] = max(self.stream_t, int(round(self.stream_t * (25.0 / max(bw, 0.5)) ** 0.5)))
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
        stream_t) have their weights DMA'd straight from the page-locked arena into a small
        VRAM ring on a copy stream, repacked to native tile order, and computed on the GPU
        (fused kernel or the reconstruct path), while the cold tail runs on the CPU, compressed
        to the rows that still have at least one unmasked assignment. Tail jobs are issued before the streamed batches and collected
        after them, so the CPU works the tail while the GPU streams. Falls back to the plain CPU
        path when nothing qualifies.
        """
        spec = self.specs[layer_idx]
        rows = y.shape[0]
        if (rows < self.stream_min_rows or spec.get("expert_bytes") is None
                or spec["expert_bytes"] > self.wslot_size or layer_idx not in self.aux):
            return self.submit(layer_idx, y, selected_experts, routing_weights)
        assert spec["expert_bytes"] % 16 == 0, "streamed expert block must be 16-byte aligned"

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
            if TUNING.stream_prof:
                import time
                tp0 = time.perf_counter()
            counts1_h = counts1.tolist()
            if TUNING.stream_prof:
                self._sprof_sync = time.perf_counter() - tp0
            neg, counts_h = counts1_h[0], counts1_h[1:]
            streamed = [e for e in range(E) if counts_h[e] >= st["stream_t"]]
            if TUNING.stream_debug:
                n_str = sum(counts_h[e] for e in streamed)
                print(f" -- stream L{layer_idx}: rows {rows}, streamed experts "
                      f"{len(streamed)}/{E}, assignments {n_str}/{sum(counts_h)}")
            if not streamed:
                return self.submit(layer_idx, y, selected_experts, routing_weights)
            if not TUNING.stream_prof:
                return self._submit_prefill_streamed(
                    layer_idx, y, selected_experts, routing_weights, spec, streamed, st,
                    counts_h, flat, shifted, neg)
            import time
            if self._sprof is None:
                self._sprof = dict(n = 0, sync = 0.0, host = 0.0, gpu = 0.0, gpu_n = 0,
                                   batches = 0, rawwait = 0.0, dma = 0.0, compute = 0.0,
                                   ev = None, pending = [])
            pr = self._sprof
            if pr["ev"] is not None:
                # Previous layer's events completed at the router sync above
                e0, e1 = pr["ev"]
                pr["gpu"] += e0.elapsed_time(e1)
                pr["gpu_n"] += 1
                for pe in pr["pending"]:
                    pr["rawwait"] += pe[0].elapsed_time(pe[1])
                    pr["dma"] += pe[1].elapsed_time(pe[2])
                    pr["compute"] += pe[3].elapsed_time(pe[4])
                pr["pending"] = []
            ev0 = torch.cuda.Event(enable_timing = True)
            ev1 = torch.cuda.Event(enable_timing = True)
            ev0.record()
            th0 = time.perf_counter()
            out = self._submit_prefill_streamed(
                layer_idx, y, selected_experts, routing_weights, spec, streamed, st,
                counts_h, flat, shifted, neg)
            pr["host"] += time.perf_counter() - th0
            ev1.record()
            pr["ev"] = (ev0, ev1)
            pr["sync"] += self._sprof_sync
            pr["n"] += 1
            per_slot = _stream_per_slot(self.wslot_size, spec["expert_bytes"], self.batch_experts)
            pr["batches"] += -(-len(streamed) // per_slot)
            # Report once per full pass over the registered layers
            L = len(self.specs)
            if pr["n"] % L == 0:
                print(_stream_prof_line(pr, L, rows), flush = True)
                pr.update(sync = 0.0, host = 0.0, gpu = 0.0, gpu_n = 0, batches = 0,
                          rawwait = 0.0, dma = 0.0, compute = 0.0)
            return out

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
        per_slot = _stream_per_slot(self.wslot_size, exp_b, self.batch_experts)
        gated = pd.get("g") is not None
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

        blocks = self.blocks[layer_idx]
        for i0 in range(0, len(streamed), per_slot):
            batch = streamed[i0:i0 + per_slot]
            ws = self.next_wslot
            self.next_wslot = (self.next_wslot + 1) % self.num_wslots
            rs = self.next_rslot
            self.next_rslot = (self.next_rslot + 1) % 2

            # DMA each expert's arena block into the raw slot on the copy stream; the arena is
            # page-locked so these are true async copies from the parent's mapping
            raw = st["raw_slots"][rs]
            prof = self._sprof if TUNING.stream_prof else None
            with torch.cuda.stream(copy_stream):
                # Timing bracket around the raw-slot reuse wait: pe[0] before, pe[1] after,
                # so the interval is the exposed copy-stream stall (zero when the slot is free)
                if prof is not None:
                    pe = [torch.cuda.Event(enable_timing = True) for _ in range(4)]
                    pe[0].record(copy_stream)
                if st["rslot_used"][rs]:
                    copy_stream.wait_event(st["rfree_ev"][rs])
                if prof is not None:
                    pe[1].record(copy_stream)
                for bi, e in enumerate(batch):
                    ci, off = blocks[e]
                    raw[bi * exp_b : (bi + 1) * exp_b].copy_(
                        self.arena[ci][off : off + exp_b], non_blocking = True)
                st["wready_ev"][rs].record(copy_stream)
                if prof is not None:
                    pe[2].record(copy_stream)
            st["rslot_used"][rs] = True

            # Repack raw -> compute slot on the current stream once the DMA lands, then
            # release the raw slot for the next DMA into it
            cur = torch.cuda.current_stream()
            cur.wait_event(st["wready_ev"][rs])
            if st["wslot_used"][ws]:
                cur.wait_event(st["wconsumed_ev"][ws])
            if prof is not None:
                pe[3].record()
                prof["pending"].append(pe)
            vslot = st["vram_slots"][ws]
            # One launch per projection over the whole batch (K8 matrices and the native
            # layout were never swizzled: plain copy)
            for name, off in (("g", 0), ("u", gb), ("d", gb + ub)):
                if pd.get(name):
                    k, n, K = pd[name]
                    ext.moe_unswizzle_trellis(raw, vslot, len(batch), exp_b, off,
                                              k // 16, n // 16, K, _proj_swizzled(st["swz"], K))
            st["rfree_ev"][rs].record(cur)
            st["wslot_used"][ws] = True
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
                xg = y.index_select(0, idx)
                # Zero-pad to the quantized input width (the had transform requires it)
                hi = spec["hi"]
                if xg.shape[1] != hi:
                    xg = torch.nn.functional.pad(xg, (0, hi - xg.shape[1]))
                we = wseg.float().unsqueeze(1)
                def tview(off_b, dims):
                    k, n, K = dims
                    numel = (k // 16) * (n // 16) * 16 * K
                    return vslot[bi * exp_b + off_b : bi * exp_b + off_b + numel * 2] \
                        .view(torch.int16).view(k // 16, n // 16, 16 * K)
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
            st["wconsumed_ev"][ws].record(cur)
            if prof is not None:
                pe4 = torch.cuda.Event(enable_timing = True)
                pe4.record()
                prof["pending"][-1].append(pe4)

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
            # The child's main thread serves the pipe at runtime (expert installs); the flag
            # stops its compute thread, the message unblocks the recv loop. A broken pipe
            # (worker already gone) must not skip reaping it
            try:
                if self.v_quit is not None:
                    self.v_quit[0] = 1
                if self.conn is not None:
                    self.conn.send(("quit",))
            except Exception:
                pass
            try:
                self.proc.join(timeout = 5)
                if self.proc.is_alive():
                    self.proc.terminate()
                    self.proc.join(timeout = 2)
                if self.proc.is_alive():
                    self.proc.kill()
                    self.proc.join(timeout = 2)
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
        # Unpin exactly what was pinned: the control block, and each arena chunk whose view
        # was appended after its registration succeeded
        if self.shm is not None:
            try:
                cuda_host_unregister(self.base_ptr)
            except Exception:
                pass
        for view in self.arena:
            try:
                cuda_host_unregister(view.data_ptr())
            except Exception:
                pass
        # Drop every view over the buffers before closing, or mmap refuses to unmap
        self.slots = None
        self.sstate = None
        self.arena = []
        self.blocks = []
        self.v_quit = self.v_pass_wake = self.v_abort = self.v_ready = None
        self.v_jobs_tail = self.v_jobs_head = self.v_jobs = None
        self._flags_u32 = None
        import gc
        gc.collect()
        if self.shm is not None:
            try:
                self.shm.close()
                self.shm.unlink()
            except Exception:
                pass
            self.shm = None
        # Release every chunk the worker published, mapped here or not: with the child gone
        # this is the single unlink
        mapped = {chunk.name: chunk for chunk in self.arena_shm}
        for name in self.arena_names:
            try:
                chunk = mapped.get(name) or shared_memory.SharedMemory(name = name)
                chunk.close()
                chunk.unlink()
            except Exception:
                pass
        self.arena_shm = []
        self.arena_names = []
        self.started = False
        self.by_key = {}
