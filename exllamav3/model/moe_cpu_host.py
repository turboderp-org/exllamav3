from __future__ import annotations
import os
import time
import multiprocessing
from multiprocessing import shared_memory
import numpy as np
import torch

from ..ext import exllamav3_ext as ext
from ..util.device_copy import host_to_device
from ..util.misc import Cleanupper, install_parent_death_signal
from ..util.shm import check_shm_capacity
from ..util.memory import check_host_memory, windows_memory_status
from .moe_cpu_affinity import (plan_host_cpus, default_worker_threads, apply_process_affinity, process_cpus,
                               numa_topology, pci_bdf_numa_node, parse_numa_nodes, bind_memory_to_node,
                               check_node_memory)
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
# sizeof(MoeJob): 7 uint32 fields + experts[MOE_JOB_MAX_EXPERTS] + seq_hi, all 4-byte
# fields so no compiler padding -- recompute this if the struct's fixed fields ever change
MOE_JOB_BYTES = (7 + MOE_JOB_MAX_EXPERTS + 1) * 4
MOE_JOB_SEQ_HI = 7 + MOE_JOB_MAX_EXPERTS   # u32 index of MoeJob.seq_hi
# Flag width (MOE_FLAG_BITS in cpu/moe_handoff.h): ROCm's stream wait has no cyclic compare, so its
# flags are 64-bit and never wrap; CUDA's are 32-bit and compared cyclically
MOE_FLAG_BITS = 64 if torch.version.hip else 32
MOE_CTRL_JOBS_OFFSET = 384
MOE_SLOT_FLAGS_OFFSET = MOE_CTRL_JOBS_OFFSET + MOE_JOB_RING * MOE_JOB_BYTES
MOE_MAX_WSLOTS = 8
MOE_FLAGS_SIZE = 3 * 64 * MOE_MAX_SLOTS + 2 * 64 * MOE_MAX_WSLOTS
MOE_STAGE_RING = 64
MOE_STAGE_TAIL_OFFSET = MOE_SLOT_FLAGS_OFFSET + MOE_FLAGS_SIZE
MOE_STAGE_HEAD_OFFSET = MOE_STAGE_TAIL_OFFSET + 64
MOE_STAGE_JOBS_OFFSET = MOE_STAGE_TAIL_OFFSET + 128
MOE_CTRL_SIZE = MOE_STAGE_JOBS_OFFSET + MOE_STAGE_RING * MOE_JOB_BYTES
# Multi-NUMA N-split: workers 1..W-1 each own an extension block (their rings) appended at the
# layout's wext_off, then one bank of arrival counters; worker 0 keeps the legacy block above, and
# every flag stays where a single worker has it (moe_handoff.h, MoeWorkerCtrl)
MOE_MAX_WORKERS = 8
MOE_WEXT_READY = 0
MOE_WEXT_JOBS_TAIL = 64
MOE_WEXT_JOBS_HEAD = 128
MOE_WEXT_JOBS = 192
MOE_WEXT_STAGE_TAIL = MOE_WEXT_JOBS + MOE_JOB_RING * MOE_JOB_BYTES
MOE_WEXT_STAGE_HEAD = MOE_WEXT_STAGE_TAIL + 64
MOE_WEXT_STAGE_JOBS = MOE_WEXT_STAGE_HEAD + 64
MOE_WEXT_SIZE = MOE_WEXT_STAGE_JOBS + MOE_STAGE_RING * MOE_JOB_BYTES
MOE_ARRIVE_SIZE = 64 * (MOE_MAX_SLOTS + MOE_MAX_WSLOTS)


def _worker_ctrl_offsets(w: int, wext_off: int) -> dict:
    """Byte offsets of worker w's (w >= 1) control words and rings, in its extension block"""
    b = wext_off + (w - 1) * MOE_WEXT_SIZE
    return dict(
        ready = b + MOE_WEXT_READY, jobs_tail = b + MOE_WEXT_JOBS_TAIL, jobs_head = b + MOE_WEXT_JOBS_HEAD,
        jobs = b + MOE_WEXT_JOBS, stage_tail = b + MOE_WEXT_STAGE_TAIL,
        stage_head = b + MOE_WEXT_STAGE_HEAD, stage_jobs = b + MOE_WEXT_STAGE_JOBS,
    )


def _set_job_seq(job, seq: int):
    """Sequence number into a job descriptor (u32 view) as its two 32-bit halves"""
    job[0] = seq & 0xFFFFFFFF
    job[MOE_JOB_SEQ_HI] = seq >> 32


def _write_flag(u32: np.ndarray, offset: int, value: int):
    """Host-side write of a handoff flag at a byte offset into the region"""
    if MOE_FLAG_BITS == 64:
        u32[offset // 4 : offset // 4 + 2].view(np.uint64)[0] = value
    else:
        u32[offset // 4] = value & 0xFFFFFFFF


def _align64(x):
    return (x + 63) & ~63


def _slice_ranges(interm: int, workers: int) -> list[tuple[int, int]]:
    """
    Intermediate-dim slice [lo, hi) per worker for the multi-NUMA N-split. Cuts at whole
    128-column Hadamard blocks, so every per-worker tensor stays a legal input to the unmodified
    kernels and packed layouts: gate/up are sliced along n, down along k (worker w's down input
    is its own gate/up output slice). Remainder blocks go to the lowest-indexed workers. Parent
    and workers both call this, so the rule lives in one place.
    """
    if interm % 128:
        raise ValueError(f"CPU MoE workers: intermediate dimension {interm} is not a multiple of "
                         f"the 128-column N-split block")
    blocks = interm // 128
    if workers < 1 or workers > blocks:
        raise ValueError(f"CPU MoE workers: {workers} workers do not fit the intermediate dimension "
                         f"{interm} ({blocks} 128-column blocks)")
    base, extra = divmod(blocks, workers)
    out, b = [], 0
    for w in range(workers):
        nb = base + (1 if w < extra else 0)
        out.append((b * 128, (b + nb) * 128))
        b += nb
    return out


def _slice_expert_proj(t, n_sliced, lo, hi, drop_bias = False, trellis_presliced = False):
    """One worker's slice of an expert projection (trellis, suh, svh, bias) in native order,
    to intermediate columns [lo, hi). Gate/up (n_sliced) cut trellis dim 1, svh and bias (the
    n side); suh (the hidden input side) stays whole. Down cuts trellis dim 0 and suh; svh (the
    hidden output side) stays whole, and so does its bias, which only worker 0 keeps (drop_bias
    elsewhere): every worker adding it would count it W - 1 extra times in the summed partials.
    trellis_presliced: the trellis was already read ranged (DiskTensorHandle.read_range /
    read_dim1_range), only the small aux tensors remain to cut."""
    tr, suh, svh, bias = t
    if not trellis_presliced:
        tr = tr[:, lo // 16:hi // 16, :] if n_sliced else tr[lo // 16:hi // 16, :, :]
    suh = suh if n_sliced else suh[lo:hi]
    svh = svh[lo:hi] if n_sliced else svh
    if drop_bias:
        bias = None
    elif bias is not None and n_sliced:
        bias = bias[lo:hi]
    return (tr, suh, svh, bias)


def _nsplit_pieces(proj_dims, layout, workers):
    """
    How the workers' N-split slices of one expert tile its full packed trellis block (the
    [gate | up | down] block a single worker stores, streams and promotes). Every packed layout
    permutes whole 16x16 tiles -- tile (kt, nt) to (nt / group, kt, nt % group), or native
    (kt, nt) order for group 0 -- and transforms bytes only within a tile, and every slice
    boundary is a multiple of 8 tiles (a multiple of any group), so each worker's slice of each
    projection is one 2D block of the full one:

        gate/up (n sliced, tiles [a, b)): packed: contiguous at tile a * tk; native: tk rows of
            b - a tiles, full pitch tn
        down (k sliced, tiles [a, b)): packed: tn / group bands of (b - a) * group tiles, full
            pitch tk * group, from tile a * group; native: contiguous at tile a * tn

    Returns (worker_bytes, pieces): each worker's expert block size, and per worker the pieces
    (src offset in the worker's block, dst offset in the full block, width, height, src pitch,
    dst pitch), all in bytes. A copy plan for any direction is built from these.
    """
    ranges = _slice_ranges(proj_dims["u"][1], workers)
    names = [nm for nm in ("g", "u", "d") if proj_dims.get(nm)]
    worker_bytes, pieces = [], []
    for lo, hi in ranges:
        a, b = lo // 16, hi // 16
        wp, w_off, f_off = [], 0, 0
        for nm in names:
            k, n, K = proj_dims[nm]
            tk, tn, tb = k // 16, n // 16, int(16 * K) * 2
            group = layout[("g", "u", "d").index(nm)][0]
            if nm != "d":
                w_bytes = tk * (b - a) * tb
                if group:
                    assert a % group == 0 and b % group == 0
                    wp.append((w_off, f_off + a * tk * tb, w_bytes, 1, w_bytes, w_bytes))
                else:
                    row = (b - a) * tb
                    wp.append((w_off, f_off + a * tb, row, tk, row, tn * tb))
            else:
                w_bytes = (b - a) * tn * tb
                if group:
                    assert tn % group == 0
                    row = (b - a) * group * tb
                    wp.append((w_off, f_off + a * group * tb, row, tn // group, row, tk * group * tb))
                else:
                    wp.append((w_off, f_off + a * tn * tb, w_bytes, 1, w_bytes, w_bytes))
            w_off += w_bytes
            f_off += tk * tn * tb
        # Rows that already sit back to back on both sides are one plain copy
        wp = [(s, d, wd * h, 1, wd * h, wd * h) if h > 1 and wd == sp == dp else (s, d, wd, h, sp, dp)
              for s, d, wd, h, sp, dp in wp]
        worker_bytes.append(w_off)
        pieces.append(wp)
    return worker_bytes, pieces


def _gather_plan(pieces, dst_bases, src_bases):
    """cuda_memcpy2d_async_batch rows gathering worker slices into full expert blocks:
    dst_bases[i] is expert i's full block, src_bases[i][w] its block in worker w"""
    rows = []
    for dst, srcs in zip(dst_bases, src_bases):
        for w, wp in enumerate(pieces):
            for s_off, d_off, width, height, spitch, dpitch in wp:
                rows.append((dst + d_off, dpitch, srcs[w] + s_off, spitch, width, height))
    return torch.tensor(rows, dtype = torch.int64)


def _scatter_plan(pieces, src_base, dst_bases):
    """The inverse of _gather_plan for one expert: its full block at src_base into worker w's
    block at dst_bases[w]"""
    rows = []
    for w, wp in enumerate(pieces):
        for s_off, d_off, width, height, spitch, dpitch in wp:
            rows.append((dst_bases[w] + s_off, spitch, src_base + d_off, dpitch, width, height))
    return torch.tensor(rows, dtype = torch.int64)


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
        # Physical cores kept free of pool workers for the host process, whether the pool pins
        # its workers (EXL3_MOE_CPU_PIN) or lets them float; the host is confined to them once
        # the worker has started, see moe_cpu_affinity.py. 0 disables the reservation
        self.host_cores = int(os.environ.get("EXL3_MOE_HOST_CORES", 1))
        # Thread count fallback chain ends here; config.infer_params.moe_cpu_threads (or the
        # draft/MTP equivalent) takes precedence per host when set (MoeCpuHost.__init__).
        # Default: physical cores minus host_cores; cpu_count/2 when host_cores is 0 or the
        # topology is unreadable
        order, n_phys = ext.exl3_moe_cpu_core_order()
        # The pools' placement (moe_mul1.cpp): the main one pins the physical-core prefix of the core order, the
        # others its SMT siblings, wrapping around when short
        self.n_phys, self.n_lps = n_phys, len(order)
        self.threads = int(os.environ.get(
            "EXL3_MOE_CPU_THREADS",
            default_worker_threads(n_phys, self.host_cores) if n_phys and self.host_cores > 0
            else max(1, (os.cpu_count() or 2) // 2)))
        self.num_wslots = min(int(os.environ.get("EXL3_MOE_CPU_WSLOTS", 2)), MOE_MAX_WSLOTS)
        self.wslot_size = int(os.environ.get("EXL3_MOE_CPU_WSLOT_MB", 32)) * 1024 * 1024
        self.stage_threads = int(os.environ.get("EXL3_MOE_CPU_STAGE_THREADS", 4))
        # madvise(MADV_HUGEPAGE) on the expert-weight arena chunks: with defrag=madvise (the
        # common default), the kernel does SYNCHRONOUS compaction on first touch of a hinted
        # region once easily-compactable free memory runs low, which can stall loading badly.
        # On Windows the same flag makes each arena chunk attempt MEM_LARGE_PAGES at
        # VirtualAlloc time (no post-hoc promotion exists there), negotiating the request
        # size down per chunk and falling back to a plain mapping per chunk
        self.arena_hugepage = os.environ.get("EXL3_MOE_ARENA_HUGEPAGE", "1") != "0"
        # Packed ("swizzled") expert trellis layout: repacked at arena rehome so the banded
        # CPU kernels' tile groups stream sequentially from DRAM. The group is per matrix and
        # ISA-dependent -- exl3_moe_cpu_swizzle_group is the single rule (band-8 on the
        # AVX-512 tiers, band-2 + planar dwords/halfwords on AVX2, native on scalar).
        # EXL3_MOE_CPU_SWIZZLE=0 restores the native layout everywhere.
        self.swizzle = os.environ.get("EXL3_MOE_CPU_SWIZZLE", "1") != "0"
        # Experts read per deferred-load pass when the worker loads a layer. Each pass is read
        # into loader tensors and then copied into the arena, so this bounds the transient host
        # memory on top of the arena to a slice of a layer instead of the whole layer (~1.2 GiB
        # per layer on a 512-expert model). 0 loads the whole layer in one pass
        self.load_batch_experts = int(os.environ.get("EXL3_MOE_CPU_LOAD_BATCH", 32))

        # --- GPU-streaming prefill ---
        self.stream_t_explicit = "EXL3_MOE_STREAM_T" in os.environ
        self.stream_t = int(os.environ.get("EXL3_MOE_STREAM_T", 8))
        self.stream_fused_t = int(os.environ.get("EXL3_MOE_STREAM_FUSED_T", 256))
        self.stream_min_rows = int(os.environ.get("EXL3_MOE_STREAM_MIN_ROWS", 32))
        self.batch_experts = max(1, min(
            int(os.environ.get("EXL3_MOE_STREAM_BATCH_EXPERTS", 24)), MOE_JOB_MAX_EXPERTS))

        # --- pinned expert arena (experimental, opt-in) ---
        # EXL3_MOE_PINNED_ARENA=1: the worker's expert arena is memfd-backed and shared with the
        # parent, which maps and page-locks it, so streamed prefill DMAs each expert's block
        # straight out of the arena on the copy stream instead of having the worker's stager
        # thread memcpy it into the pinned handoff ring first (the stager is the prefill
        # bottleneck on fully offloaded models). Costs: the arena is registered with CUDA
        # (~0.2 s per GiB at load), and shmem pages only get transparent hugepages when
        # /sys/kernel/mm/transparent_hugepage/shmem_enabled allows it (advise/always/
        # within_size), so on a default (never) system the CPU kernels run on 4K pages;
        # EXL3_MOE_ARENA_HUGE=2m|1g backs the memfd with hugetlbfs pages instead (requires
        # vm.nr_hugepages / hugepages-1048576kB reservations); Windows uses named sections, no
        # hugepage variant.
        self.pinned_arena = os.environ.get("EXL3_MOE_PINNED_ARENA", "0") != "0"
        self.arena_huge = os.environ.get("EXL3_MOE_ARENA_HUGE", "").strip().lower()
        assert self.arena_huge in ("", "2m", "1g"), "EXL3_MOE_ARENA_HUGE must be 2m or 1g"
        if self.arena_huge and os.name == "nt":
            raise RuntimeError("EXL3_MOE_ARENA_HUGE is Linux-only (hugetlbfs memfd); unset it on Windows")
        # Batched reconstruct tier for the streamed heavy experts (see moe_batch_recon.py):
        # experts too hot for the fused kernel are dequantized in groups with one launch per
        # projection and run through padded bmm. EXL3_MOE_STREAM_BATCH_RECON=0 restores the
        # per-expert loop
        self.stream_batch_recon = os.environ.get("EXL3_MOE_STREAM_BATCH_RECON", "1") != "0"
        # Fused-tier row tiles (32 / 64-row kernel instances per expert range), as EXL3_MOE_MTILE
        # on the GPU side
        self.mtile = os.environ.get("EXL3_MOE_MTILE", "1") != "0"

        # --- debug / kill switches ---
        self.stream_debug = bool(os.environ.get("EXL3_MOE_STREAM_DEBUG"))
        self.cpu_prof = bool(os.environ.get("EXL3_MOE_CPU_PROF"))
        self.memops = os.environ.get("EXL3_MOE_MEMOPS", "1") != "0"


TUNING = MoeCpuTuning()
ext.exl3_moe_cpu_set_memops(TUNING.memops)

# Host placement is per process: planned for the first started host's worker count and kept
# for later hosts (MTP head, draft model, reload). Workers spawned after it restore the
# pre-pin mask (_HOST_ORIG_CPUS) so their pool can pin outside the host's LPs
_HOST_AFFINITY_THREADS: int | None = None
_HOST_ORIG_CPUS: list[int] | None = None


def _apply_host_affinity(threads: int, host_cores: int, cpus: list[int] | None = None):
    """Confine the host process to the CPUs the worker pool leaves free. Never raises: an OS
    failure leaves the host unpinned with a notice. `cpus`: the multi-NUMA plan's host LPs
    (MoeCpuHost._worker_plan) instead of the single-worker tail of the pool's core order"""
    global _HOST_AFFINITY_THREADS, _HOST_ORIG_CPUS
    if _HOST_AFFINITY_THREADS is not None:
        if threads > _HOST_AFFINITY_THREADS:
            print(f" -- CPU MoE host affinity: placement planned for {_HOST_AFFINITY_THREADS} workers, "
                  f"kept for a {threads}-thread worker")
        return
    _HOST_AFFINITY_THREADS = threads
    multi_numa = cpus is not None
    if not multi_numa:
        order, n_phys = ext.exl3_moe_cpu_core_order()
        cpus = plan_host_cpus(order, n_phys, threads, host_cores)
    if cpus is None:
        if host_cores > 0 and n_phys:
            print(" -- CPU MoE host affinity: no LP free of workers in one processor group; "
                  "host threads left unpinned")
        return
    try:
        orig = process_cpus()
    except OSError as e:
        print(f" !! CPU MoE host affinity: {e}; host threads left unpinned")
        return
    err = apply_process_affinity(cpus)
    if err is not None:
        print(f" !! CPU MoE host affinity: {err}; host threads left unpinned")
        return
    _HOST_ORIG_CPUS = orig
    if multi_numa:
        # Torch sized its intra-op pool from the machine's CPU count; confined to a core or two, the
        # host's CPU-side ops fork-join that whole pool onto them (GLM-5.3-Flash on 4 nodes:
        # streamed prefill at 500 instead of 700 tok/s)
        torch.set_num_threads(max(1, len(cpus)))
    print(f" -- CPU MoE host affinity: host on LPs {[enc & 0xFFFF for enc in cpus]}")


# memfd_create only ships in CPython when the interpreter was built against glibc >= 2.27; conda
# and manylinux-built interpreters lack it (and the os.MFD_* constants) even on kernels that
# have had the syscall since 4.17 (PR #341 discussion). Resolve it at runtime instead: os first,
# then libc's symbol, then the raw syscall by architecture. The MFD_* values are kernel ABI
MFD_CLOEXEC = 0x0001
MFD_HUGETLB = 0x0004
MFD_HUGE_2MB = 21 << 26
MFD_HUGE_1GB = 30 << 26
_MEMFD_SYSCALL = {"x86_64": 319, "aarch64": 279, "riscv64": 279, "loongarch64": 279,
                  "ppc64le": 360, "ppc64": 360, "s390x": 350, "i686": 356, "armv7l": 385}


def _memfd_via_libc(name: str, flags: int) -> int:
    """libc's memfd_create symbol (any glibc >= 2.27 / musl >= 1.1.20 at runtime)"""
    import ctypes
    libc = ctypes.CDLL(None, use_errno = True)
    fn = libc.memfd_create   # AttributeError when the runtime libc predates it
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    fd = fn(name.encode(), flags)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


def _memfd_via_syscall(name: str, flags: int) -> int:
    """Raw memfd_create syscall, for a runtime libc without the wrapper"""
    import ctypes
    nr = _MEMFD_SYSCALL.get(os.uname().machine)
    if nr is None:
        raise RuntimeError(f"no memfd_create syscall number known for {os.uname().machine}")
    libc = ctypes.CDLL(None, use_errno = True)
    libc.syscall.restype = ctypes.c_long
    fd = libc.syscall(ctypes.c_long(nr), ctypes.c_char_p(name.encode()), ctypes.c_uint(flags))
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return int(fd)


def _memfd_create(name: str, flags: int = 0) -> int:
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, flags)
    try:
        return _memfd_via_libc(name, flags)
    except AttributeError:
        pass
    try:
        return _memfd_via_syscall(name, flags)
    except RuntimeError as e:
        raise RuntimeError(
            f"CPU MoE pinned arena: this Python build has no os.memfd_create and no fallback "
            f"applies ({e}). Unset EXL3_MOE_PINNED_ARENA to use the staged path.") from e


# Windows large pages: there is no madvise/promotion path, so MEM_LARGE_PAGES must be requested
# at VirtualAlloc time. That requires SeLockMemoryPrivilege enabled on this process's token
# (the privilege is granted-but-disabled by default for accounts that have it), and the size
# must be a multiple of GetLargePageMinimum() (2 MiB on x64). Large pages are committed and
# non-pageable, so allocation can fail on a fragmented or busy system -- callers fall back to
# a plain anonymous mapping per chunk. A failed request is expensive to repeat (the kernel
# searches for contiguous memory each time) and contiguity does not come back during a load,
# so the arena remembers how far down the size ladder it had to go (see _HugeArena._new_chunk).

_WIN32_LARGE_PAGE_SUPPORT = None


def _win32_enable_lock_memory_privilege() -> bool:
    """Best-effort SeLockMemoryPrivilege enable on the current process token. Returns True
    only when the privilege ends up enabled."""
    import ctypes
    from ctypes import wintypes

    TOKEN_ADJUST_PRIVILEGES = 0x0020
    SE_PRIVILEGE_ENABLED = 0x00000002

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Luid", LUID), ("Attributes", wintypes.DWORD)]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [("PrivilegeCount", wintypes.DWORD),
                    ("Privileges", LUID_AND_ATTRIBUTES * 1)]

    advapi32 = ctypes.WinDLL("advapi32", use_last_error = True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    # GetCurrentProcess returns the -1 pseudo-handle; without a pointer-sized restype ctypes
    # truncates it to 32 bits and OpenProcessToken fails with ERROR_INVALID_HANDLE
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi32.LookupPrivilegeValueW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.POINTER(LUID)]
    advapi32.AdjustTokenPrivileges.argtypes = [
        wintypes.HANDLE, wintypes.BOOL, ctypes.POINTER(TOKEN_PRIVILEGES),
        wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
            kernel32.GetCurrentProcess(),
            TOKEN_ADJUST_PRIVILEGES,
            ctypes.byref(token)):
        return False
    try:
        luid = LUID()
        if not advapi32.LookupPrivilegeValueW(None, "SeLockMemoryPrivilege", ctypes.byref(luid)):
            return False
        tp = TOKEN_PRIVILEGES()
        tp.PrivilegeCount = 1
        tp.Privileges[0] = LUID_AND_ATTRIBUTES(luid, SE_PRIVILEGE_ENABLED)
        advapi32.AdjustTokenPrivileges(token, False, ctypes.byref(tp), 0, None, None)
        # AdjustTokenPrivileges returns success even when it silently dropped privileges it
        # couldn't assign; GetLastError distinguishes full assignment from partial
        return ctypes.get_last_error() == 0
    finally:
        kernel32.CloseHandle(token)


def _win32_large_page_alloc(size: int, min_size: int):
    """Allocate MEM_LARGE_PAGES memory of up to `size` bytes, halving the request on each
    failure down to `min_size`, and return the largest buffer obtained (a ctypes byte array
    bound to the allocation -- buffer-protocol compatible, so it drops into the same
    memoryview / torch.frombuffer consumers as an mmap object), or None when even `min_size`
    cannot be supplied. A large request needs that many physically contiguous 2 MiB regions,
    which a fragmented or busy system often cannot supply even when smaller runs exist, so a
    shrunken chunk is still a win over falling back to 4K pages outright. A weakref.finalize
    on the array issues VirtualFree(MEM_RELEASE) when the last reference dies, giving the
    chunk the same free-on-GC lifetime semantics as an mmap object."""
    import ctypes
    import weakref

    global _WIN32_LARGE_PAGE_SUPPORT
    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    kernel32.GetLargePageMinimum.restype = ctypes.c_size_t

    if _WIN32_LARGE_PAGE_SUPPORT is None:
        large_page_min = kernel32.GetLargePageMinimum()
        _WIN32_LARGE_PAGE_SUPPORT = bool(large_page_min) and _win32_enable_lock_memory_privilege()
        if not _WIN32_LARGE_PAGE_SUPPORT and os.environ.get("EXL3_MOE_ARENA_DEBUG"):
            print(" -- arena: MEM_LARGE_PAGES unavailable "
                  f"(GetLargePageMinimum={large_page_min}, SeLockMemoryPrivilege "
                  "not enabled); arena chunks will use regular pages", flush = True)
    if not _WIN32_LARGE_PAGE_SUPPORT:
        return None

    granularity = kernel32.GetLargePageMinimum()
    MEM_RESERVE = 0x2000
    MEM_COMMIT = 0x1000
    MEM_LARGE_PAGES = 0x20000000
    PAGE_READWRITE = 0x04
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualAlloc.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
    kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]

    want = (size + granularity - 1) // granularity * granularity
    floor = (min_size + granularity - 1) // granularity * granularity
    addr = None
    while True:
        addr = kernel32.VirtualAlloc(
            None, want, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE)
        if addr or want <= floor:
            break
        want = max(want >> 1, floor)
    if not addr:
        return None
    buf = (ctypes.c_uint8 * want).from_address(addr)
    weakref.finalize(buf, kernel32.VirtualFree, ctypes.c_void_p(addr), 0, 0x8000)  # MEM_RELEASE
    if want < size and os.environ.get("EXL3_MOE_ARENA_DEBUG"):
        print(f" -- arena: MEM_LARGE_PAGES negotiated down to {want >> 20} MiB "
              f"(asked {size >> 20} MiB)", flush = True)
    return buf


def _fractional_native_subset(native, q, p, e):
    """Native halfwords for planes[..., p::2, e::2], retaining the tile-prefix strides."""
    # q is odd: the even plane/lane steps leave xor 1 entirely in the start index.
    return native.as_strided(
        native.shape[:-1] + ((q - p + 1) // 2, 4),
        native.stride()[:-1] + (2, 2 * q),
        native.storage_offset() + ((q * e + p) ^ 1))


def _copy_repacked(dst, tensor, group, planar):
    """Copy a [k/16, n/16, 16K] trellis tensor into a same-shape view, applying the packed
    CPU layout (exl3_moe_cpu_swizzle_group / exl3_moe_cpu_planar_layout): tile (kt, nt) moves
    to (nt/group, kt, nt%group). Integer planar tiles transpose dword w to 8*(w%bits) + w//bits.
    Fractional planar tiles use Q = 2*bits+1 halfword planes, with chronological halfword j
    packed at 8*(j%Q) + j//Q from native[j ^ 1] (no byte swap within a halfword).
    Copies write directly into `dst` on the caller's CPU/CUDA stream, without a full-tensor
    temporary. Group 0 is a plain copy."""
    tk, tn, ps = tensor.shape
    if not group:
        dst.copy_(tensor)
    elif planar and ps % 16 == 8:
        assert group == 2, "fractional planar layout requires group 2"
        q = ps // 8
        native = tensor.view(tk, tn // group, group, ps).permute(1, 0, 2, 3)
        planes = dst.view(tn // group, tk, group, q, 8)
        if q == 3 and dst.device.type == "cpu" and tensor.device.type == "cpu":
            # Measurements favor six single-plane copies for 1.5-bit CPU-to-CPU.
            # The even 2*q step keeps xor 1 in start, preserving the byte mapping
            # in bounded four-halfword slices.
            for p in range(q):
                for e in range(2):
                    start = (q * e + p) ^ 1
                    planes[..., p, e::2].copy_(native[..., start::2 * q])
        else:
            for p in range(2):
                for e in range(2):
                    planes[..., p::2, e::2].copy_(_fractional_native_subset(native, q, p, e))
    elif planar:
        wd = ps // 2
        dst.view(torch.int32).view(tn // group, tk, group, wd // 8, 8).copy_(
            tensor.view(torch.int32).view(tk, tn // group, group, 8, wd // 8).permute(1, 0, 2, 4, 3))
    else:
        dst.view(tn // group, tk, group, ps).copy_(
            tensor.view(tk, tn // group, group, ps).permute(1, 0, 2, 3))


def _copy_unpacked(dst, packed, group, planar):
    """The inverse of _copy_repacked: a packed-layout [k/16, n/16, 16K] view back to native
    tile order (and integer dword / fractional halfword order) directly in `dst`.
    Fractional native halfword h comes from packed[8*((h ^ 1)%Q) + (h ^ 1)//Q]."""
    tk, tn, ps = dst.shape
    if not group:
        dst.copy_(packed)
    elif planar and ps % 16 == 8:
        assert group == 2, "fractional planar layout requires group 2"
        q = ps // 8
        native = dst.view(tk, tn // group, group, ps)
        planes = packed.view(tn // group, tk, group, q, 8).permute(1, 0, 2, 3, 4)
        if q == 3 and dst.device.type == "cpu" and packed.device.type == "cpu":
            for p in range(q):
                for e in range(2):
                    start = (q * e + p) ^ 1
                    native[..., start::2 * q].copy_(planes[..., p, e::2])
        else:
            for p in range(2):
                for e in range(2):
                    _fractional_native_subset(native, q, p, e).copy_(planes[..., p::2, e::2])
    elif planar:
        wd = ps // 2
        dst.view(torch.int32).view(tk, tn // group, group, 8, wd // 8).copy_(
            packed.view(torch.int32).view(tn // group, tk, group, wd // 8, 8).permute(1, 0, 2, 4, 3))
    else:
        dst.view(tk, tn // group, group, ps).copy_(
            packed.view(tn // group, tk, group, ps).permute(1, 0, 2, 3))


class _HugeArena:
    """
    Growable pool of large (default 1 GiB) anonymous mmap chunks that expert weights are copied
    into, so hugepage promotion (see promote_hugepages) has few, large regions to work with
    instead of thousands of separate small (sub-2MB) loader allocations
    """
    CHUNK_BYTES = 1 << 30   # 1 GiB
    WIN32_LARGE_FLOOR = 64 << 20   # smallest MEM_LARGE_PAGES chunk worth having (Windows)
    CHECK_STEP = 256 << 20   # host-memory guard granularity for lazily committed chunks

    def __init__(self, shared = False, huge = "", conn = None, node = None):
        """shared: back each chunk with shared memory and publish it over `conn` as
        ("chunk", index, size, name) so the parent can map the same pages and page-lock them
        for DMA: Linux sends a memfd descriptor after the message (SCM_RIGHTS, name = None),
        Windows a named pagefile section. huge: "2m"/"1g" requests hugetlbfs memfds (Linux).
        node: the NUMA node a multi-NUMA worker's memory is bound to (bind_memory_to_node), for the
        per-node memory check."""
        self.shared = shared
        self.huge = huge
        self.conn = conn
        self.node = node
        self.chunks = []
        self.cur = None
        self.cur_off = 0
        # A private anonymous chunk (Linux, not shared) only takes RAM for the pages rehome()
        # writes, and the last chunk of a load is usually mostly unused, so the host-memory
        # guard runs on bytes written (in CHECK_STEP slices) rather than on whole chunks.
        # Shared, hugetlb and Windows chunks are committed up front and keep the per-chunk check
        self.lazy = not shared and os.name != "nt"
        self.written = 0
        self.checked = 0
        self.win32_large_bytes = 0   # bytes of chunks backed by MEM_LARGE_PAGES (Windows)
        # Largest MEM_LARGE_PAGES request still worth making (Windows), None until the first
        # attempt: the size the last chunk was served at, or below the smallest size that failed
        self.win32_large_ceiling = None

    def _new_chunk(self, min_bytes):
        import mmap, os
        size = max(self.CHUNK_BYTES, (min_bytes + (2 << 20) - 1) & ~((2 << 20) - 1))
        if not self.lazy:
            check_host_memory(size, f"CPU MoE expert arena chunk {len(self.chunks)} "
                                    f"({(sum(len(c) for c in self.chunks) + size) >> 20} MiB in total)")
            check_node_memory(size, self.node, f"CPU MoE expert arena chunk {len(self.chunks)}")
        if self.shared and os.name == "nt":
            # Named pagefile-backed section as a plain mmap (a SharedMemory owner's finalizer
            # trips on the layer tensors' exports at worker exit). It charges commit and gets
            # page-locked by the parent, so both free RAM and commit must cover the chunk
            index = len(self.chunks)
            avail_phys, avail_commit = windows_memory_status()
            if size > min(avail_phys, avail_commit):
                raise RuntimeError(
                    f"CPU MoE pinned arena: chunk {index} needs {size >> 20} MiB, but only "
                    f"{avail_phys >> 20} MiB of physical RAM and {avail_commit >> 20} MiB of commit "
                    f"are available. Free RAM, offload fewer experts, or unset EXL3_MOE_PINNED_ARENA.")
            name = f"exl3_moe_arena_{os.getpid()}_{index}"
            try:
                m = mmap.mmap(-1, size, tagname = name)
            except OSError as e:
                raise RuntimeError(
                    f"CPU MoE pinned arena: cannot create the {size >> 20} MiB section for chunk "
                    f"{index} ({e.strerror}). Free RAM or commit, or unset EXL3_MOE_PINNED_ARENA.") from e
            if self.conn is not None:
                self.conn.send(("chunk", index, size, name))
        elif self.shared:
            flags = 0
            if self.huge == "1g":
                size = (size + (1 << 30) - 1) & ~((1 << 30) - 1)
                flags = MFD_HUGETLB | MFD_HUGE_1GB
            elif self.huge == "2m":
                flags = MFD_HUGETLB | MFD_HUGE_2MB
            fd = _memfd_create(f"exl3_moe_arena_{len(self.chunks)}", flags)
            try:
                # Preallocate: a hugetlb memfd without enough reserved pages fails here with
                # ENOMEM instead of SIGBUS on first touch
                os.posix_fallocate(fd, 0, size)
                m = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            except OSError as e:
                os.close(fd)
                raise RuntimeError(
                    f"CPU MoE pinned arena: cannot allocate a {size >> 20} MiB "
                    f"{'hugetlb ' if self.huge else ''}memfd chunk ({e.strerror}). "
                    + ("Reserve hugepages (vm.nr_hugepages / hugepages-1048576kB) or unset "
                       "EXL3_MOE_ARENA_HUGE." if self.huge else
                       "Check the memory cgroup limit, or unset EXL3_MOE_PINNED_ARENA.")) from e
            if not self.huge and TUNING.arena_hugepage:
                # Honoured only where shmem_enabled permits (advise/always/within_size)
                try:
                    m.madvise(mmap.MADV_HUGEPAGE)
                except Exception:
                    pass
            if self.conn is not None:
                # SCM_RIGHTS over the pipe's socketpair. socket.send_fds rather than
                # multiprocessing.reduction.send_handle: the latter blocks for an
                # acknowledgement byte, which would stall the worker's loading until the
                # parent next pumps the pipe
                import socket
                self.conn.send(("chunk", len(self.chunks), size, None))
                with socket.socket(fileno = os.dup(self.conn.fileno())) as sock:
                    socket.send_fds(sock, [b"F"], [fd])
            os.close(fd)   # the mapping keeps the pages alive
        elif os.name == "nt":
            # mmap.MAP_PRIVATE / mmap.PROT_* don't exist on Windows; an anonymous mapping is
            # writable by default there. Windows has no post-hoc hugepage promotion, so the
            # EXL3_MOE_ARENA_HUGEPAGE knob is honoured here instead: each chunk first tries a
            # MEM_LARGE_PAGES VirtualAlloc (needs SeLockMemoryPrivilege and physically
            # contiguous 2 MiB regions), halving the request down to 64 MiB -- and no lower
            # than what the placement needs -- before falling back to a plain mapping, still
            # at full chunk size since regular pages have no contiguity constraint. Later
            # chunks start at the size the previous one was served at instead of walking the
            # ladder from the top again, and stop asking once the smallest size has failed
            m = None
            if TUNING.arena_hugepage:
                floor = max(self.WIN32_LARGE_FLOOR, min_bytes)
                ceiling = self.win32_large_ceiling
                if ceiling is None or ceiling >= floor:
                    m = _win32_large_page_alloc(size if ceiling is None else min(size, ceiling), floor)
                    self.win32_large_ceiling = len(m) if m is not None else floor // 2
            if m is not None:
                self.win32_large_bytes += len(m)
            else:
                m = mmap.mmap(-1, size)
        else:
            m = mmap.mmap(-1, size, mmap.MAP_PRIVATE, mmap.PROT_READ | mmap.PROT_WRITE)
        self.chunks.append(m)
        self.cur = m
        self.cur_off = 0
        if os.environ.get("EXL3_MOE_ARENA_DEBUG"):
            total = sum(len(c) for c in self.chunks)
            print(f" -- arena: new chunk {len(m)/1e6:.1f} MB, {len(self.chunks)} chunks, "
                  f"{total/1e9:.3f} GB total", flush = True)

    def reserve(self, nbytes):
        """Make sure the next `nbytes` of rehomes land contiguously in the current chunk;
        returns the (chunk index, byte offset) they will start at"""
        aligned = (nbytes + 63) & ~63
        if self.cur is None or self.cur_off + aligned > len(self.cur):
            self._new_chunk(aligned)
        return len(self.chunks) - 1, self.cur_off

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
        import mmap, os, time
        if not TUNING.arena_hugepage:
            return
        if os.name == "nt":
            # MEM_LARGE_PAGES is decided at VirtualAlloc time (see _new_chunk); there is no
            # promotion step to run here, only coverage to report. Large pages are locked in
            # RAM, which the user did not ask for explicitly, so say so whenever they are in
            # use; an account without the privilege gets regular pages and no message
            total = sum(len(c) for c in self.chunks)
            if self.win32_large_bytes:
                print(f" -- CPU MoE arena: {self.win32_large_bytes/1e9:.2f} GB of {total/1e9:.2f} GB "
                      f"on large pages (locked in RAM, never paged out; set "
                      f"EXL3_MOE_ARENA_HUGEPAGE=0 to use regular pages)", flush = True)
            elif _WIN32_LARGE_PAGE_SUPPORT and total:
                print(" -- CPU MoE arena: no large pages could be allocated, using regular pages "
                      "(physical memory is too fragmented; large pages are usually available "
                      "again after a reboot)", flush = True)
            return
        collapse = getattr(mmap, "MADV_COLLAPSE", 25)
        t0 = time.perf_counter()
        for c in self.chunks:
            try:
                c.madvise(collapse)
            except Exception:
                pass
        if os.environ.get("EXL3_MOE_ARENA_DEBUG"):
            print(f" -- arena: MADV_COLLAPSE issued on {len(self.chunks)} chunks "
                  f"in {time.perf_counter() - t0:.1f} s", flush = True)

    def rehome(self, tensor, band_swizzle = 0, planar = False):
        """Copy `tensor` into the arena and return a same-dtype/shape view over the copy. The
        arena outlives every tensor it hands out (held for the process lifetime), so the
        returned view stays valid.

        band_swizzle/planar: repack a [k/16, n/16, 16K] trellis tensor with the given packed
        layout during the copy (see _copy_repacked). The returned view keeps the original
        logical shape; only the byte order differs (consumed by the swz-aware kernels in
        moe_mul1.cpp)."""
        import torch
        if tensor is None or tensor.numel() == 0:
            return tensor
        nbytes = tensor.numel() * tensor.element_size()
        aligned = (nbytes + 63) & ~63
        if self.lazy:
            if self.written + aligned > self.checked:
                step = max(aligned, self.CHECK_STEP)
                check_host_memory(step, f"CPU MoE expert arena ({(self.written + step) >> 20} MiB in total)")
                check_node_memory(step, self.node, "CPU MoE expert arena")
                self.checked = self.written + step
            self.written += aligned
        if self.cur is None or self.cur_off + aligned > len(self.cur):
            self._new_chunk(aligned)
        off = self.cur_off
        self.cur_off += aligned
        buf = memoryview(self.cur)[off : off + nbytes]
        dst = torch.frombuffer(buf, dtype = torch.uint8)
        if band_swizzle or planar:
            _copy_repacked(dst.view(tensor.dtype).view(tensor.shape), tensor, band_swizzle, planar)
        else:
            dst.copy_(tensor.contiguous().view(torch.uint8).reshape(-1))
        return dst.view(tensor.dtype).view(tensor.shape)


def _moe_cpu_child_main(conn, model_dir, threads, stage_threads, pinned = False, huge = "", cpus = None,
                        pin_siblings = False, node = None, worker_idx = 0, num_workers = 1, core_order = None):
    """
    Child entry point: receives ("layer", spec) messages, loading each layer's expert tensors
    (deferred, multithreaded) and acking, until ("start", shm_name, layout) switches it into the
    worker loop. Errors are reported over the pipe before exiting.

    Multi-NUMA (num_workers > 1): this is worker `worker_idx` of NUMA node `node`, its pool pinned
    along `core_order` (the node's LPs but the host's) and its memory bound to the node, and
    it loads only its slice of every expert's intermediate dimension (_slice_ranges).
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

    if cpus is not None:
        # Spawned after the host was confined: drop the inherited host mask, or the pool's
        # worker pins fail (Windows refuses a thread mask outside the process mask)
        err = apply_process_affinity(cpus)
        if err is not None:
            print(f" !! CPU MoE worker affinity: {err}; worker confined to the host's LPs")
    if pin_siblings:
        # A draft component's pool beside the main model's: share its physical cores, not its LPs
        cext.exl3_moe_cpu_pool_pin_siblings()
    if num_workers > 1:
        # Before anything is allocated: a bound allocation reclaims the node's page cache instead of
        # spilling onto another node, and every thread started from here on inherits the policy
        if not bind_memory_to_node(node):
            print(f" !! CPU MoE worker {worker_idx}: could not bind memory to NUMA node {node}")
        cext.exl3_moe_cpu_set_core_order(core_order)
        # Torch sizes its intra-op pool (the loader's repacks) from the machine's CPU count, one
        # thread per LP for a single worker; every worker sized that way oversubscribes the machine
        # W times over (GLM-5.3-Flash on 4 nodes: 44 s load instead of 600 s)
        torch.set_num_threads(threads)

    if os.name == "nt":
        # Hold 1 ms timer resolution for this process (per-process since Windows 10 2004): the
        # worker's poll loops back off to 50 us sleeps, which otherwise round up to the default
        # 15.6 ms quantum -- one quantum per job-ring poll miss lands directly on token latency
        try:
            ctypes.WinDLL("winmm").timeBeginPeriod(1)
        except Exception:
            pass

    shm = None
    t_entry = time.time()
    try:
        stc = SafetensorsCollection(model_dir)
        if os.environ.get("EXL3_MOE_ARENA_DEBUG"):
            print(f" -- worker: entered main at {t_entry:.3f}, checkpoint index ready at {time.time():.3f} "
                  f"({time.time() - t_entry:.2f} s)", flush = True)
        cpu = torch.device("cpu")
        # Pinned mode: memfd-backed chunks, published to the parent as they are created
        arena = _HugeArena(shared = pinned, huge = huge, conn = conn if pinned else None, node = node)

        def fetch(keys, ranged = None):
            """Per expert (trellis, suh, svh, bias). ranged = (n_sliced, lo, hi): read only this
            multi-NUMA worker's slice of the trellis from disk (see _slice_expert_proj)"""
            out = []
            for k in keys:
                if ranged is None:
                    trellis = stc.get_tensor(k + ".trellis", cpu)
                else:
                    n_sliced, lo, hi = ranged
                    hnd = stc.get_tensor_handle(k + ".trellis")
                    try:
                        trellis = hnd.read_dim1_range(lo // 16, hi // 16) if n_sliced \
                            else hnd.read_range(lo // 16, hi // 16)
                    finally:
                        hnd.close()
                suh = stc.get_tensor(k + ".suh", cpu, float2half = True)
                svh = stc.get_tensor(k + ".svh", cpu, float2half = True)
                bias = stc.get_tensor(k + ".bias", cpu, optional = True, float2half = True)
                out.append((trellis, suh, svh, bias))
            return out

        # Repack the trellis copies into the packed layout the kernel tier wants for each
        # matrix's bitrate (exl3_moe_cpu_swizzle_group / exl3_moe_cpu_planar_layout; group 0
        # keeps native order: scalar tier, half rates on AVX2 unless explicitly enabled,
        # K8 on the AVX-512 tiers)
        swz = TUNING.swizzle

        def rehome_trellis(t):
            grp = cext.exl3_moe_cpu_swizzle_group(t.shape[2] / 16.0) if swz else 0
            pln = grp and cext.exl3_moe_cpu_planar_layout(t.shape[2] / 16.0)
            return arena.rehome(t, band_swizzle = grp, planar = pln)

        def nbytes(t):
            return t.numel() * t.element_size()

        def rehome_experts(g, u, d):
            """Per expert: the gate/up/down trellis tensors back to back (one contiguous
            expert block in (g, u, d) order, the layout the stager produces and the DMA unit
            of the pinned-arena streamed path), then the small aux tensors. Returns the
            per-projection lists and the per-expert (chunk, byte offset) of each block."""
            gs, us, ds, blocks, auxlocs = [], [], [], [], []
            for e in range(len(u)):
                projs = ([g[e]] if g else []) + [u[e], d[e]]
                total = sum(nbytes(t[0]) for t in projs)
                assert all(nbytes(t[0]) % 64 == 0 for t in projs), "trellis size not 64-byte aligned"
                blocks.append(arena.reserve(total))
                trellis = [rehome_trellis(t[0]) for t in projs]
                # The aux tensors' (chunk, offset, bytes) go to the parent with the block: a demotion
                # (promote-on-miss) writes an expert's suh/svh/bias into the arena from the GPU
                aux, locs = [], []
                for t in projs:
                    pa, pl = [], []
                    for a in (t[1], t[2], t[3]):
                        if a is None or a.numel() == 0:
                            pl.append(None)
                        else:
                            pl.append(arena.reserve(nbytes(a)) + (nbytes(a),))
                        pa.append(arena.rehome(a))
                    aux.append(tuple(pa))
                    locs.append(pl)
                auxlocs.append(locs)
                if g:
                    gs.append((trellis[0],) + aux[0])
                us.append((trellis[-2],) + aux[-2])
                ds.append((trellis[-1],) + aux[-1])
            return gs, us, ds, blocks, auxlocs

        def biases(ts):
            return [t[3] for t in ts] if ts and ts[0][3] is not None else []

        # Per-layer, per-expert arena views retained for runtime expert installs (dynamic
        # placement): the arena memory is what the compute kernels read, so an in-place copy
        # into these views (with the same swizzle transform rehome applied) replaces an
        # expert's weights. The parent quiesces (full sync, ring drained) before sending an
        # install, so the workers never observe a torn write
        layer_views = []
        # Per layer: this worker's intermediate slice under the N-split (installs re-slice with it)
        layer_ranges = []

        # The pipe is drained by its own thread into a queue: the parent registers layers as
        # fast as it loads its own, and a send that had to wait for a child busy rehoming the
        # previous layer would couple the two loads through the socket buffer
        import queue
        import threading
        inbox = queue.Queue()
        def receive():
            while True:
                try:
                    m = conn.recv()
                except EOFError:
                    m = ("quit",)
                inbox.put(m)
                if m[0] == "quit":
                    return
        threading.Thread(target = receive, daemon = True).start()
        # Rehoming one batch overlaps the deferred fetch of the next (the loader releases the
        # GIL while its engine runs); a single worker thread keeps the arena writes in order
        from concurrent.futures import ThreadPoolExecutor
        rehome_pool = ThreadPoolExecutor(max_workers = 1)
        debug = bool(os.environ.get("EXL3_MOE_ARENA_DEBUG"))
        if debug:
            print(f" -- worker: affinity {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else '?'} cpus, "
                  f"torch threads {torch.get_num_threads()}, loader stage threads {stage_threads}", flush = True)
        while True:
            t_wait = time.perf_counter()
            msg = inbox.get()
            t_wait = time.perf_counter() - t_wait
            if msg[0] == "layer":
                t_layer = time.perf_counter()
                spec = msg[1]
                num_experts = len(spec["up_keys"])
                batch = TUNING.load_batch_experts or num_experts
                g, u, d, blocks, auxlocs = [], [], [], [], []
                pending = None
                def collect(fut):
                    bg_, bu_, bd_, bb, ba = fut.result()
                    g.extend(bg_)
                    u.extend(bu_)
                    d.extend(bd_)
                    blocks.extend(bb)
                    auxlocs.extend(ba)
                # Multi-NUMA: this worker's slice of the intermediate dim, from the up trellis's
                # header (dim 1 is n / 16), known before any trellis is read so the reads are ranged
                rng = None
                if num_workers > 1:
                    hnd = stc.get_tensor_handle(spec["up_keys"][0] + ".trellis")
                    rng = _slice_ranges(hnd.shape[1] * 16, num_workers)[worker_idx]
                    hnd.close()
                layer_ranges.append(rng)
                ranged_n = (True,) + rng if rng is not None else None
                ranged_k = (False,) + rng if rng is not None else None
                for e0 in range(0, num_experts, batch):
                    sl = slice(e0, e0 + batch)
                    stc.begin_deferred_load()
                    bg = fetch(spec["gate_keys"][sl], ranged_n)
                    bu = fetch(spec["up_keys"][sl], ranged_n)
                    bd = fetch(spec["down_keys"][sl], ranged_k)
                    stc.end_deferred_load()
                    if rng is not None:
                        # The aux tensors cut to the slice (the trellis was read ranged); down's bias
                        # belongs to worker 0 alone
                        lo, hi = rng
                        bg = [_slice_expert_proj(t, True, lo, hi, trellis_presliced = True) for t in bg]
                        bu = [_slice_expert_proj(t, True, lo, hi, trellis_presliced = True) for t in bu]
                        bd = [_slice_expert_proj(t, False, lo, hi, drop_bias = worker_idx != 0,
                                                 trellis_presliced = True) for t in bd]
                    # Copy into the hugepage-backed arena now that the deferred reads have
                    # actually populated these tensors; the loader tensors die with this batch.
                    # The copy runs on the rehome thread while the next batch is fetched
                    if pending is not None:
                        collect(pending)
                    pending = rehome_pool.submit(rehome_experts, bg, bu, bd)
                if pending is not None:
                    collect(pending)
                cext.exl3_moe_cpu_make_layer(
                    [t[0] for t in g], [t[1] for t in g], [t[2] for t in g],
                    [t[0] for t in u], [t[1] for t in u], [t[2] for t in u],
                    [t[0] for t in d], [t[1] for t in d], [t[2] for t in d],
                    biases(g), biases(u), biases(d),
                    spec["activation"], spec["act_limit"],
                    1 if swz else 0,
                )
                layer_views.append((g, u, d))
                if debug:
                    print(f" -- worker: layer {len(layer_views) - 1}: waited {t_wait * 1000:.0f} ms for the spec, "
                          f"loaded in {(time.perf_counter() - t_layer) * 1000:.0f} ms, done at {time.time():.3f}", flush = True)
                # Reclaim the discarded loader tensors (rehome_experts copied everything into the
                # arena); every few layers rather than every layer, the trim itself takes a
                # noticeable slice of a layer's load
                if len(layer_views) % 4 == 3:
                    try:
                        ctypes.CDLL(None).malloc_trim(0)
                    except Exception:
                        pass
                # The ack carries the expert block locations (only meaningful to a parent that
                # maps the arena)
                conn.send(("ok", blocks if pinned else None, auxlocs if pinned else None))
            elif msg[0] == "start":
                shm_name, layout = msg[1], msg[2]
                break
            elif msg[0] == "quit":
                return

        stc.close()
        shm = shared_memory.SharedMemory(name = shm_name)
        base = np.frombuffer(shm.buf, dtype = np.uint8).ctypes.data
        cext.exl3_moe_cpu_set_prof(layout.get("cpu_prof", False))

        def install(li, ei, keys):
            """Replace expert ei of layer li with the checkpoint tensors at `keys` (one key
            per projection, gate omitted for gateless layers), in place in the arena using
            the same packed representation as the initial rehome."""
            views = layer_views[li]
            projs = views if len(keys) == 3 else views[1:]
            rng = layer_ranges[li]
            for pi, (key, plist) in enumerate(zip(keys, projs)):
                v_tr, v_suh, v_svh, v_bias = plist[ei]
                # Multi-NUMA: the same slice as the load (the last projection is the down)
                ranged = None if rng is None else (pi < len(projs) - 1,) + rng
                new = fetch([key], ranged)[0]
                if ranged is not None:
                    new = _slice_expert_proj(new, *ranged, trellis_presliced = True)
                new_tr, new_suh, new_svh, new_bias = new
                assert new_tr.shape == v_tr.shape, f"install shape mismatch: {key}"
                grp = cext.exl3_moe_cpu_swizzle_group(new_tr.shape[2] / 16.0) if swz else 0
                if grp:
                    _copy_repacked(v_tr, new_tr, grp, cext.exl3_moe_cpu_planar_layout(new_tr.shape[2] / 16.0))
                else:
                    v_tr.copy_(new_tr)
                v_suh.copy_(new_suh)
                v_svh.copy_(new_svh)
                if v_bias is not None:
                    v_bias.copy_(new_bias)

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
                worker_idx, num_workers, layout.get("wext_off", 0),
            ),
            daemon = True,
        )
        worker.start()

        # Hugepage promotion runs off the startup path: MADV_COLLAPSE is synchronous and copies
        # the whole arena into 2 MiB pages (tens of seconds for a 50+ GiB arena, minutes when
        # free memory is fragmented and the kernel has to compact first), and the parent's
        # startup wait must not depend on it. Page migration is transparent to the compute
        # threads, so the worker serves requests on 4K pages until each chunk lands
        threading.Thread(target = arena.promote_hugepages, daemon = True).start()

        while True:
            msg = inbox.get()
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


def probe_bandwidth(timed_copy, floor_s = 0.5, cap_s = 2.0, asleep_gbs = 5.0):
    """Sustained pinned->device rate in GB/s from repeated `timed_copy()` calls. An idle link
    sits at Gen1 (the Windows driver drops it after a few idle seconds) and retrains only after
    0.2-0.3 s of sustained traffic, so: copy for at least floor_s, then until the last 8 copies
    are within 5% of the best, up to cap_s while the rate still looks asleep. Median of the
    last 8, so one copy straddling the retrain step cannot win."""
    import time
    t0 = time.perf_counter()
    best, trace = 0.0, []
    while True:
        rate = timed_copy()
        trace.append(rate)
        best = max(best, rate)
        t = time.perf_counter() - t0
        tail = trace[-8:]
        steady = len(tail) == 8 and min(tail) >= 0.95 * best
        if t >= cap_s or (t >= floor_s and steady and best >= asleep_gbs):
            return sorted(tail)[len(tail) // 2]


def _pinning_exhausted(e: Exception) -> bool:
    """Whether a cudaHostRegister failure means the process has no page-locked memory left
    (cudaErrorMemoryAllocation, reported as "out of memory"), as opposed to a bad mapping"""
    return "out of memory" in str(e).lower()


def _pools_share_lps(hosts):
    """Whether the main model's pool and the other components' would run on the same LPs: no SMT, fewer SMT siblings
    than draft threads, or a main pool larger than the physical cores"""
    text = hosts.get("text")
    others = sum(host.threads for comp, host in hosts.items() if comp != "text")
    return text is not None and others > 0 and \
        (text.threads > TUNING.n_phys or others > TUNING.n_lps - TUNING.n_phys)


# Hot polling trades busy cores for the wake-up of a napping worker. That wake-up is only slow enough to be worth the
# trade on Windows; elsewhere a worker is back before polling would have saved anything
HOT_POLLING = os.name == "nt"


def set_moe_cpu_hot(config, text, draft):
    """Hot polling of a config's CPU MoE workers across the MTP draft/verify handoff: `text` for the main model's,
    `draft` for every other component's. Does nothing where hot polling is off (HOT_POLLING)"""
    if not HOT_POLLING:
        return
    hosts = getattr(config, "moe_cpu_hosts", {})
    if text and draft and _pools_share_lps(hosts):
        # Two hot pools on the same LPs time-slice each other (2-3x slower steps): only the drafting one polls
        text = False
    for comp, host in hosts.items():
        host.set_hot(text if comp == "text" else draft)


class _ExtraWorker:
    """Process and bookkeeping of multi-NUMA worker w >= 1. Worker 0's live in MoeCpuHost's own
    fields of the same names, so code taking a worker state (MoeCpuHost._wst) runs the single
    worker on exactly the fields it always used"""

    def __init__(self):
        self.proc = None
        self.conn = None
        self._send_queue = None
        self._sender = None
        self.acked = 0
        self.layer_blocks = []
        self.layer_auxloc = []
        self.arena_maps = []
        self.arena_views = []
        self.arena_registered = None
        self._arena_pending = None


class MoeCpuHost:

    # Single-worker defaults (also for hosts built without __init__)
    workers = 1
    _xw = ()

    @classmethod
    def for_config(cls, config):
        """The worker host of this config's component (infer_params.moe_cpu_component), created
        on first use and shared by every offloaded layer of the component"""
        comp = getattr(config.infer_params, "moe_cpu_component", "text")
        hosts = getattr(config, "moe_cpu_hosts", None)
        if hosts is None:
            hosts = {}
            config.moe_cpu_hosts = hosts
        host = hosts.get(comp)
        if host is None:
            host = cls(config)
            hosts[comp] = host
        return host

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
        # tuning default (EXL3_MOE_CPU_THREADS env, else physical cores minus EXL3_MOE_HOST_CORES)
        comp = getattr(config.infer_params, "moe_cpu_component", "text")
        self.component = comp
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
        self._dev_bufs = {}      # per device: persistent streamed-prefill buffers (see _device_buffers)
        # Pinned arena (config.infer_params.moe_cpu_pinned_arena, EXL3_MOE_PINNED_ARENA its
        # default): parent-side mappings of the worker's memfd chunks (mmap, int16 view) indexed
        # like the worker's chunk list, and per layer the per-expert (chunk, byte offset) of its
        # contiguous [gate | up | down] trellis block
        self.pinned = bool(getattr(config.infer_params, "moe_cpu_pinned_arena", TUNING.pinned_arena))
        self.arena_maps = []
        self.arena_views = []
        # Chunks are registered in order; the first one the driver refuses for lack of page-locked
        # memory (WDDM and WSL2 budget the pinnable memory at about half of RAM) ends the pinning,
        # and experts in it and every later chunk stream through the staged path instead
        self.arena_registered = None
        # The chunk the worker may still be writing: (index, size, name, mapping), page-locked
        # once the next chunk appears or loading is done (see _attach_chunk)
        self._arena_pending = None
        self.layer_blocks = []
        self.layer_auxloc = []
        # Promote-on-miss transfers (see block_sparse_mlp_cpu._split_promote_decide): per device a
        # stream of its own and two expert-sized staging buffers (packed and native order)
        self._promo = {}
        self.batch_recon = TUNING.stream_batch_recon

        # Multi-NUMA (infer_params.moe_cpu_numa_nodes, Linux): one worker process per NUMA node,
        # each holding a slice of every CPU expert's intermediate dimension, pinned to its node's
        # physical cores with its memory bound there; the collecting stream sums the workers'
        # partial outputs. A single worker (the default) runs exactly the single-process path
        self.numa_nodes = self._resolve_numa_nodes(getattr(config.infer_params, "moe_cpu_numa_nodes", None))
        self.workers = len(self.numa_nodes) if self.numa_nodes else 1
        self._xw = [_ExtraWorker() for _ in range(1, self.workers)]
        self.threads_explicit = cfg_threads is not None or "EXL3_MOE_CPU_THREADS" in os.environ
        self.host_cpus = None
        self.worker_threads = [self.threads]
        self.wext_off = 0
        self.partial_stride = 0

    # -- multi-NUMA helpers ------------------------------------------------------------------------

    def _resolve_numa_nodes(self, value):
        """
        The NUMA nodes of a moe_cpu_numa_nodes value (see parse_numa_nodes), one worker each, or
        None for a single worker: unset, a component other than the main model (an MTP head keeps
        the single worker), or "all" with fewer than two nodes this process may run on. Listed
        nodes must be nodes this process may run on, at least two and at most MOE_MAX_WORKERS.
        """
        nodes = parse_numa_nodes(value)
        if nodes is None or self.component != "text":
            return None
        usable = sorted(self._usable_topology())
        if nodes == "all":
            if len(usable) < 2:
                print(f" -- CPU MoE: moe_cpu_numa_nodes = all, but this process may run on {len(usable)} "
                      f"NUMA node(s); using one worker", flush = True)
                return None
            nodes = usable
        else:
            missing = [n for n in nodes if n not in usable]
            if missing:
                raise ValueError(f"moe_cpu_numa_nodes: NUMA node(s) {missing} are not available to this "
                                 f"process (available: {usable})")
            if len(nodes) < 2:
                raise ValueError(f"moe_cpu_numa_nodes needs at least two NUMA nodes (got {nodes})")
        if len(nodes) > MOE_MAX_WORKERS:
            raise ValueError(f"moe_cpu_numa_nodes: at most {MOE_MAX_WORKERS} NUMA nodes (got {nodes})")
        return nodes

    def _wst(self, w):
        """Worker w's process/bookkeeping fields: the host itself for worker 0, see _ExtraWorker"""
        return self if w == 0 else self._xw[w - 1]

    def _states(self):
        return [self] + list(self._xw)

    def _alive(self):
        return all(st.proc is not None and st.proc.is_alive() for st in self._states())

    def _send_all(self, msg):
        for st in self._states():
            st._send_queue.put(msg)

    @staticmethod
    def _usable_topology():
        """NUMA node -> its LPs this process may run on (Linux sysfs): the pre-pin mask once a host
        has confined it (a reload, an MTP head), so a later load sees the same nodes as the first"""
        try:
            allowed = {c & 0xFFFF for c in _HOST_ORIG_CPUS} if _HOST_ORIG_CPUS is not None \
                else set(process_cpus())
        except OSError:
            allowed = None
        topo = {n: [c for c in cpus if allowed is None or c in allowed] for n, cpus in numa_topology().items()}
        return {n: cpus for n, cpus in topo.items() if cpus}

    def _worker_plan(self):
        """
        Per worker, (the LPs its pool pins to, its physical core count): the pool's own pin order
        (physical cores first, then their SMT siblings) restricted to the worker's NUMA node, less
        the host's LPs. The host's are planned as for a single worker (plan_host_cpus), but on the
        current CUDA device's node (the first worker's when that is unknown), and kept in
        self.host_cpus -- or, when an earlier host in this process already placed it, are the LPs
        it is confined to. As with a single worker, nothing else in a worker is confined.
        """
        topo = self._usable_topology()
        order, n_phys = ext.exl3_moe_cpu_core_order()

        def node_order(n, exclude = ()):
            lps = set(topo[n]) - set(exclude)
            cores = [c for c in order[:n_phys] if c in lps]
            return cores + [c for c in order[n_phys:] if c in lps], len(cores)

        host_node = None
        try:
            p = torch.cuda.get_device_properties(torch.cuda.current_device())
            host_node = pci_bdf_numa_node(f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0")
        except Exception:
            pass
        if host_node not in topo:
            host_node = self.numa_nodes[0]
        if _HOST_ORIG_CPUS is not None:
            try:
                host = process_cpus()
            except OSError:
                host = []
            self.host_cpus = None
        else:
            hord, hn = node_order(host_node)
            host = plan_host_cpus(hord, hn, hn - TUNING.host_cores, TUNING.host_cores) or []
            self.host_cpus = host or None
        plan = []
        for n in self.numa_nodes:
            pins, cores = node_order(n, host)
            if not cores:
                raise RuntimeError(f"CPU MoE worker on NUMA node {n} has no CPU left after the host's "
                                   f"reserved cores; lower EXL3_MOE_HOST_CORES")
            plan.append((pins, cores))
        print(f" -- CPU MoE: {self.workers} workers on NUMA nodes {self.numa_nodes}, "
              f"{[c for _, c in plan]} cores, host on LPs {sorted(host)}", flush = True)
        return plan

    def _spawn(self):
        if self.proc is not None:
            return
        ctx = multiprocessing.get_context("spawn")
        huge = TUNING.arena_huge if self.pinned else ""
        if self.workers > 1:
            # The thread count as for a single worker: an explicit one as given (past the node's
            # physical cores onto their SMT siblings), by default the node's physical cores
            plan = self._worker_plan()
            self.worker_threads = [self.threads if self.threads_explicit else cores for _, cores in plan]
        for w in range(self.workers):
            st = self._wst(w)
            st.conn, child_conn = ctx.Pipe(duplex = True)
            if self.workers > 1:
                args = (child_conn, self.model_dir, self.worker_threads[w], self.stage_threads,
                        self.pinned, huge, _HOST_ORIG_CPUS, False, self.numa_nodes[w], w, self.workers,
                        plan[w][0])
            else:
                args = (child_conn, self.model_dir, self.threads, self.stage_threads,
                        self.pinned, huge, _HOST_ORIG_CPUS, self.component != "text")
            st.proc = ctx.Process(target = _moe_cpu_child_main, args = args, daemon = True)
            st.proc.start()
            child_conn.close()
            # Parent -> child messages go through a queue and a sender thread: a layer spec is tens
            # of KB, and a direct send would block the loading thread until the child's receiver
            # takes it, which it only gets to do between its own loader calls
            import queue
            import threading
            st._send_queue = queue.Queue()
            def sender(st = st):
                while True:
                    msg = st._send_queue.get()
                    if msg is None:
                        return
                    try:
                        st.conn.send(msg)
                    except Exception:
                        return
            st._sender = threading.Thread(target = sender, daemon = True)
            st._sender.start()
        # Cleanupper fires at the end of the __main__ scope, before interpreter teardown breaks
        # the shm views and pipe machinery that shutdown() needs; PDEATHSIG in the child covers
        # the paths where no Python hook runs at all
        cleanupper.register_atexit(self.shutdown)

    def _pump(self, timeout):
        """Receive one message from the child, surfacing errors and death. With several workers:
        one message from each worker that has one, waiting up to `timeout` for the first"""
        if self._xw:
            from multiprocessing.connection import wait
            ready = wait([st.conn for st in self._states()], timeout)
            got = False
            for w, st in enumerate(self._states()):
                if st.conn in ready:
                    got |= self._pump_worker(w, 0)
            if not got and not self._alive():
                raise RuntimeError("CPU MoE worker process died")
            return got
        return self._pump_worker(0, timeout)

    def _pump_worker(self, w, timeout):
        st = self._wst(w)
        tag = f"CPU MoE worker {w}" if self._xw else "CPU MoE worker"
        if st.conn.poll(timeout):
            msg = st.conn.recv()
            if msg[0] == "err":
                raise RuntimeError(f"{tag} failed:\n{msg[1]}")
            if msg[0] == "ok":
                # Layer acks arrive in registration order: entry i describes specs[i]
                st.layer_blocks.append(msg[1] if len(msg) > 1 else None)
                st.layer_auxloc.append(msg[2] if len(msg) > 2 else None)
                st.acked += 1
            elif msg[0] == "chunk":
                self._attach_chunk(msg[1], msg[2], msg[3], w)
            return True
        if not st.proc.is_alive():
            raise RuntimeError(f"{tag} process died")
        return False

    def _register_handoff(self, size):
        """Page-lock the handoff segment. It is registered after the arena chunks (its size depends
        on the layers), so when the pinned budget ran out on them it can run out here too; the
        segment is what the staged path runs on, so arena chunks are given back, last first, until
        it fits"""
        while True:
            try:
                cuda_host_register(self.base_ptr, size,
                                   flags = CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED)
                return
            except Exception as e:
                # The worker with the most page-locked chunks gives one back (any with several)
                cand = [(max((i for i, v in enumerate(st.arena_views) if v is not None), default = -1), w)
                        for w, st in enumerate(self._states())]
                last, w = max(cand)
                if not _pinning_exhausted(e) or last < 0:
                    raise
                st = self._wst(w)
                cuda_host_unregister(st.arena_views[last].data_ptr())
                st.arena_views[last] = None
                st.arena_maps[last].close()
                st.arena_maps[last] = None
                st.arena_registered = last
                which = f"worker {w} chunk" if self._xw else "chunk"
                print(f" !! CPU MoE pinned arena: {which} {last} released to page-lock the handoff "
                      f"segment; experts in it and the remaining chunks use the staged path", flush = True)


    def _attach_chunk(self, index, size, name, w = 0):
        """Pinned arena: map arena chunk `index` (Linux: descriptor sent right after the
        ("chunk", ...) message; Windows: section `name`) and page-lock the chunk before it. The
        worker fills its chunks strictly in order, so a chunk is complete once its successor is
        published; the last one is page-locked when loading is done (ensure_started). A chunk is
        never page-locked while the worker writes it: on Windows the driver's per-page probe
        rewrites each page's first byte, and a concurrent write there is lost. Registration still
        overlaps the rest of the load. Any failure closes the mapping before the error leaves
        this frame."""
        import mmap
        import socket
        st = self._wst(w)
        self._pin_pending_chunk(w)
        assert index == len(st.arena_maps), "arena chunk published out of order"
        if name is not None:
            # Opening by name fails loudly if the worker already dropped the section
            m = shared_memory.SharedMemory(name = name)
        else:
            with socket.socket(fileno = os.dup(st.conn.fileno())) as sock:
                _, fds, _, _ = socket.recv_fds(sock, 1, 1)
            assert len(fds) == 1, "arena chunk descriptor missing"
            fd = fds[0]
            try:
                m = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            finally:
                os.close(fd)
        if st.arena_registered is not None:
            # Page-locked memory ran out at an earlier chunk: nothing past it is pinned, and the
            # mapping is not needed either (the staged path reads the arena in the worker)
            m.close()
            st.arena_maps.append(None)
            st.arena_views.append(None)
            return
        st._arena_pending = (index, size, name, m)

    def _pin_pending_chunk(self, w = 0):
        """Page-lock the chunk published last (see _attach_chunk), once the worker has finished it"""
        st = self._wst(w)
        if st._arena_pending is None:
            return
        index, size, name, m = st._arena_pending
        st._arena_pending = None
        view = None
        try:
            # count = size // 2: a section opened by name reports its page-rounded size, and
            # frombuffer rejects a buffer shorter than the advertised chunk
            view = torch.frombuffer(m.buf if name is not None else m, dtype = torch.int16,
                                    count = size // 2)
            cuda_host_register(view.data_ptr(), size, flags = CUDA_HOST_REGISTER_PORTABLE)
        except Exception as e:
            view = None   # no tensor over an unmapped region may survive in the traceback's frame
            m.close()
            if _pinning_exhausted(e):
                # The driver has no page-locked memory left for this process (WDDM and WSL2 cap
                # it at about half of RAM; the KV tiers and other processes share the budget).
                # Not a fault: keep what is pinned, stage the rest
                st.arena_registered = index
                st.arena_maps.append(None)
                st.arena_views.append(None)
                print(f" !! CPU MoE pinned arena: no page-locked memory left for chunk {index} "
                      f"({index * (size >> 20)} MiB registered); experts in the remaining chunks "
                      f"use the staged path", flush = True)
                return
            raise RuntimeError(
                f"CPU MoE pinned arena: chunk {index} ({size >> 20} MiB) could not be attached "
                f"({e}). Unset EXL3_MOE_PINNED_ARENA to use the staged path.") from e
        st.arena_maps.append(m)
        st.arena_views.append(view)
        if os.environ.get("EXL3_MOE_ARENA_DEBUG"):
            print(f" -- pinned arena: mapped + registered {f'worker {w} ' if w else ''}chunk {index} "
                  f"({size >> 20} MiB)", flush = True)

    def register_layer(self, key, gate_keys, up_keys, down_keys, activation, act_limit, hi, ho, topk,
                       proj_dims = None, aux = None, interm_fp32 = False):
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
            interm_fp32 = interm_fp32,      # resident experts' gate/up output dtype (BlockSparseMLP interm_dtype)
        )
        if proj_dims is not None:
            # Deterministic per-expert byte layout (gate, up, down), mirrored by the worker's
            # stage function
            def tb(d):
                k, n, K = d
                return (k // 16) * (n // 16) * int(16 * K) * 2
            gb = tb(proj_dims["g"]) if proj_dims.get("g") else 0
            ub, db = tb(proj_dims["u"]), tb(proj_dims["d"])
            spec["proj_bytes"] = (gb, ub, db)
            spec["expert_bytes"] = gb + ub + db
            if self.workers > 1:
                # The N-split must fit this layer (128-column blocks, at least one per worker);
                # fail before the workers load anything
                _slice_ranges(proj_dims["u"][1], self.workers)
        self.specs.append(spec)
        self.live_layers += 1
        idx = len(self.specs) - 1
        self.by_key[key] = idx
        if aux is not None:
            self.aux[idx] = aux
        self._send_all(("layer", {k: v for k, v in spec.items() if k != "proj_dims"}))
        return idx

    def commit_module(self, module_key):
        """
        Called by the loader after a top-level module has committed on the GPU side: take whatever
        the child has sent so far (acks, pinned-arena chunks, errors) without waiting for it. The
        child loads behind the parent at its own pace and the parent only waits for it in
        ensure_started, before the first forward; an error in the child still surfaces within a
        module of happening
        """
        if self.proc is None or self.conn is None:
            return
        for w, st in enumerate(self._states()):
            while st.conn.poll(0):
                self._pump_worker(w, 0)

    def ensure_started(self):
        if self.started or not self.specs:
            return
        while any(st.acked < len(self.specs) for st in self._states()):
            self._pump(1.0)
        for w in range(self.workers):
            self._pin_pending_chunk(w)

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
        if self.workers > 1:
            # One output section per worker; worker w's partial sits partial_stride floats after
            # worker w-1's (moe_handoff.h)
            out_stride = _align64(self.cap_rows * max_ho * 4)
            self.partial_stride = out_stride // 4
            slot_size = _align64(off_out + self.workers * out_stride)
        self.layout = dict(
            num_slots = self.num_slots, slot_size = slot_size, cap_rows = self.cap_rows,
            max_hi = max_hi, max_ho = max_ho, max_topk = max_topk,
        )

        self.layout["wstage_off"] = MOE_CTRL_SIZE + self.num_slots * slot_size
        self.layout["num_wslots"] = self.num_wslots
        self.layout["wslot_size"] = self.wslot_size
        self.layout["cpu_prof"] = TUNING.cpu_prof
        size = MOE_CTRL_SIZE + self.num_slots * slot_size + self.num_wslots * self.wslot_size
        wstage_offs = [self.layout["wstage_off"]]
        if self.workers > 1:
            # Workers 1..W-1: their control blocks, the arrival counters, then their own
            # weight-staging regions
            self.wext_off = _align64(size)
            self.layout["wext_off"] = self.wext_off
            xstage = self.wext_off + (self.workers - 1) * MOE_WEXT_SIZE + MOE_ARRIVE_SIZE
            wstage_offs += [xstage + (w - 1) * self.num_wslots * self.wslot_size for w in range(1, self.workers)]
            size = xstage + (self.workers - 1) * self.num_wslots * self.wslot_size
        check_shm_capacity(size, "The CPU MoE offload handoff segment")
        self.shm = shared_memory.SharedMemory(create = True, size = size)
        buf = np.frombuffer(self.shm.buf, dtype = np.uint8)
        buf[:MOE_CTRL_SIZE] = 0
        if self.workers > 1:
            buf[self.wext_off:self.wext_off + (self.workers - 1) * MOE_WEXT_SIZE + MOE_ARRIVE_SIZE] = 0
        self.base_ptr = buf.ctypes.data
        self._register_handoff(size)
        # GPU-visible alias of the registered buffer: equal to base_ptr under UVA (Linux
        # desktop), but a distinct address under WDDM (Windows), where the host VA is not a
        # valid device pointer
        self.gpu_base_ptr = cuda_host_get_device_pointer(self.base_ptr)

        u32 = np.frombuffer(self.shm.buf, dtype = np.uint32)
        self.v_quit = u32[0:1]
        self.v_pass_wake = u32[16:17]
        self.v_hot = u32[17:18]
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

        if self.workers > 1:
            # Workers 1..W-1: control words, rings and staging views, as above for worker 0 (the
            # flags are shared)
            for w, x in enumerate(self._xw, start = 1):
                o = _worker_ctrl_offsets(w, self.wext_off)
                x.v_ready = u32[o["ready"] // 4 : o["ready"] // 4 + 1]
                x.v_jobs_tail = u32[o["jobs_tail"] // 4 : o["jobs_tail"] // 4 + 1]
                x.v_jobs_head = u32[o["jobs_head"] // 4 : o["jobs_head"] // 4 + 1]
                x.v_jobs = np.frombuffer(
                    self.shm.buf, dtype = np.uint32, offset = o["jobs"],
                    count = MOE_JOB_RING * (MOE_JOB_BYTES // 4)).reshape(MOE_JOB_RING, MOE_JOB_BYTES // 4)
                x.v_stage_tail = u32[o["stage_tail"] // 4 : o["stage_tail"] // 4 + 1]
                x.v_stage_head = u32[o["stage_head"] // 4 : o["stage_head"] // 4 + 1]
                x.v_stage_jobs = np.frombuffer(
                    self.shm.buf, dtype = np.uint32, offset = o["stage_jobs"],
                    count = MOE_STAGE_RING * (MOE_JOB_BYTES // 4)).reshape(MOE_STAGE_RING, MOE_JOB_BYTES // 4)
                x.wviews = [torch.frombuffer(self.shm.buf, dtype = torch.int16, count = self.wslot_size // 2,
                                             offset = wstage_offs[w] + s_ * self.wslot_size)
                            for s_ in range(self.num_wslots)]
            # Collect gates for the plain path's sum, which has no issue-kernel counts
            self._collect_ones = {}

        self._send_queue.put(("start", self.shm.name, self.layout))
        for w, x in enumerate(self._xw, start = 1):
            x._send_queue.put(("start", self.shm.name, dict(self.layout, wstage_off = wstage_offs[w])))

        import time
        t0 = time.time()
        # Startup is now just the shared-memory attach, layer registration and thread spawn
        # (hugepage promotion runs in the worker's background); the limit stays generous for
        # slow hosts and is overridable
        timeout = float(os.environ.get("EXL3_MOE_CPU_START_TIMEOUT", "60"))
        while not self.v_ready[0] or not all(x.v_ready[0] for x in self._xw):
            if not self._alive():
                raise RuntimeError("CPU MoE worker process died during startup")
            if time.time() - t0 > timeout:
                raise RuntimeError(
                    f"CPU MoE worker startup timeout ({timeout:.0f} s; EXL3_MOE_CPU_START_TIMEOUT overrides)")
            time.sleep(0.005)
        self.started = True
        self._flags_u32 = u32
        self._start_watchdog()
        kern = "avx512-vbmi" if ext.exl3_moe_cpu_has_avx512_vbmi() else \
               ("avx512-vnni" if ext.exl3_moe_cpu_has_avx512_vnni() else \
               ("avx512-bw" if ext.exl3_moe_cpu_has_avx512_bw() else \
               ("avx2" if ext.exl3_moe_cpu_has_avx2() else "scalar")))
        if self.workers > 1:
            print(f" -- CPU MoE workers started: {len(self.specs)} layers, {kern}, {self.workers} workers, "
                  f"{self.worker_threads} threads")
            # Planned with the workers (_worker_plan): on the GPU's node, or the placement an earlier
            # host in this process already made
            if self.host_cpus:
                _apply_host_affinity(self.threads, TUNING.host_cores, cpus = self.host_cpus)
            return
        print(f" -- CPU MoE worker started: {len(self.specs)} layers, {kern}, {self.threads} threads")
        # The worker was spawned before this and pins its own threads; keep the host off them
        _apply_host_affinity(self.threads, TUNING.host_cores)

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
                if not self._alive():
                    try:
                        u32 = self._flags_u32
                        seq, wseq = self.seq + 1, self.wseq + 1
                        done0 = MOE_SLOT_FLAGS_OFFSET + 64 * MOE_MAX_SLOTS
                        cons0 = MOE_SLOT_FLAGS_OFFSET + 2 * 64 * MOE_MAX_SLOTS
                        for s in range(MOE_MAX_SLOTS):
                            _write_flag(u32, done0 + s * 64, seq)
                            _write_flag(u32, cons0 + s * 64, seq)
                        fl = MOE_SLOT_FLAGS_OFFSET + 3 * 64 * MOE_MAX_SLOTS
                        for s in range(MOE_MAX_WSLOTS):
                            _write_flag(u32, fl + s * 64, wseq)                         # stage_done
                            _write_flag(u32, fl + 64 * MOE_MAX_WSLOTS + s * 64, wseq)   # pinned_free
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
        dev = y.device
        counts = self.dev_count.get(dev)
        if counts is None:
            counts = self.dev_count[dev] = \
                torch.zeros((self.num_slots,), dtype = torch.int32, device = dev)
        slot_idx = self.next_slot
        self.next_slot = (self.next_slot + 1) % self.num_slots
        self.seq += 1
        seq = self.seq
        slot = self.slots[slot_idx]
        # One extension call: the job descriptor into the shared ring, the slot's consumed wait,
        # the issue kernel and the data_ready flag, all on the stream of y's device. It declines
        # (touching nothing) only when the job ring is near overflow, which a long prefill can
        # reach; then throttle on the worker's progress and retry
        args = (self.base_ptr, seq, layer_idx, rows, spec["topk"], slot_idx, self.slot_last_seq[slot_idx],
                selected_experts.view(-1), split_map, split_hist, y, routing_weights,
                slot["sel_dev"], slot["x_dev"], slot["w_dev"], counts, spec["hi"], first_cpu,
                slot["data_ready"], slot["consumed"], self.gpu_base_ptr + 128,
                self.workers, self.wext_off)
        if not ext.exl3_moe_split_issue_fused(*args):
            import time
            while any(int(st.v_jobs_tail[0]) - int(st.v_jobs_head[0]) >= MOE_JOB_RING - 4
                      for st in self._states()):
                if self.v_abort[0] or not self._alive():
                    raise RuntimeError("CPU MoE worker failed (ring stall)")
                time.sleep(0.0002)
            ext.exl3_moe_split_issue_fused(*args)
        self.slot_last_seq[slot_idx] = seq
        return (seq, slot_idx, rows, h_, spec["ho"], dev, counts)

    def submit_collect_fused(self, handle, final_2d):
        """Fold the worker's partial into final_2d (rows, h) in place, straight from the
        pinned slot, or, for a job the issue kernel recorded as empty, do nothing (no
        PCIe reads). Never synchronizes the host."""
        seq, slot_idx, rows, h_, ho, dev, counts = handle
        slot = self.slots[slot_idx]
        ext.exl3_moe_split_collect_fused(
            final_2d, slot["out_dev"], counts, slot_idx, ho,
            slot["done"], seq, slot["consumed"], self.gpu_base_ptr + 128,
            self.workers, self.partial_stride)

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
            if ho != h_ and self.workers == 1 else None
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
            # sync, and overwriting unconsumed descriptors corrupts the whole stream. Every
            # worker of an N-split gets the same descriptor
            for st in self._states():
                tail = int(st.v_jobs_tail[0])
                if tail - int(st.v_jobs_head[0]) >= MOE_JOB_RING - 4:
                    import time
                    while tail - int(st.v_jobs_head[0]) >= MOE_JOB_RING - 4:
                        if self.v_abort[0] or not self._alive():
                            raise RuntimeError("CPU MoE worker failed (ring stall)")
                        time.sleep(0.0002)
                job = st.v_jobs[tail % MOE_JOB_RING]
                _set_job_seq(job, seq)
                job[1] = layer_idx
                job[2] = n
                job[3] = spec["topk"]
                job[4] = slot_idx
                job[5] = 0    # MOE_JOB_KIND_COMPUTE
                st.v_jobs_tail[0] = tail + 1

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
        if self.workers > 1:
            # N-split: once the last worker has published done, every worker's partial, summed in
            # worker order straight from the slot (the kernel trims a padded width); all-ones
            # gate counts, since this path has no issue-kernel counts
            ext.exl3_moe_flag_wait(slot["done"], seq, self.gpu_base_ptr + 128)
            ones = self._collect_ones.get(out.device)
            if ones is None:
                ones = self._collect_ones[out.device] = \
                    torch.ones((self.num_slots,), dtype = torch.int32, device = out.device)
            ext.moe_split_collect_add(out[a:b], slot["out_dev"], ones, slot_idx, slot["out"].shape[1],
                                      self.workers, self.partial_stride, False)
            ext.exl3_moe_flag_write(slot["consumed"], seq)
            return
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

    def _max_proj_numel(self):
        mx = 0
        for s in self.specs:
            pd = s.get("proj_dims")
            if pd:
                for k in ("g", "u", "d"):
                    if pd.get(k):
                        mx = max(mx, pd[k][0] * pd[k][1])
        return mx

    def _stream_layout(self, spec):
        """Packed trellis layout of a layer's streamed projections as ((group, planar) for g, u,
        d), plus whether any of them needs the GPU restore pass. The arena holds the bytes in
        the layout the CPU kernel tier wants per bitrate (exl3_moe_cpu_swizzle_group /
        exl3_moe_cpu_planar_layout, the same rule the child loader and expert replacement
        apply), and the GPU kernels only read native order, so every packed projection is
        restored after its DMA. Rates are fixed per layer, so the policy is cached on the spec;
        a layer with nothing packed reads vram_slots directly (no ring, no copy pass)"""
        lay = spec.get("stream_layout")
        if lay is None:
            pd = spec["proj_dims"]
            lay = []
            for nm in ("g", "u", "d"):
                if pd.get(nm) and TUNING.swizzle:
                    K = pd[nm][2]
                    lay.append((int(ext.exl3_moe_cpu_swizzle_group(K)),
                                int(ext.exl3_moe_cpu_planar_layout(K))))
                else:
                    lay.append((0, 0))
            lay = spec["stream_layout"] = (tuple(lay), any(g or p for g, p in lay))
        return lay

    def _device_buffers(self, device, spec = None):
        """The persistent device-side buffers of the streamed-prefill path (VRAM weight ring,
        native-order ring for restored experts, reconstruct scratch, fused-tier buffers, batched
        tier statics), created on first request per device. The autosplit loader requests them
        during a layer's measured load (prefill_worst_case_parts), before the worker starts, so
        they count as allocated when the split is planned instead of appearing on the first real
        prefill; _ensure_stream_state adopts them. The native-order ring exists only once a
        layer whose experts arrive packed asks for it (spec given), so hosts whose tier packs
        nothing never allocate it; once allocated it stays for later layers to account for"""
        key = torch.device(device).index or 0
        d = self._dev_bufs.get(key)
        mx = self._max_proj_numel()
        if d is None:
            d = dict(
                vram_slots = [torch.empty(self.wslot_size // 2, dtype = torch.int16, device = device)
                              for _ in range(self.num_wslots)],
                native_slots = None,
                w_scratch = None,
                fused_bufs = {},
                recon = {},
            )
            self._dev_bufs[key] = d
        if spec is not None and d["native_slots"] is None and (self._stream_layout(spec)[1] or self.workers > 1):
            d["native_slots"] = [torch.empty(self.wslot_size // 2, dtype = torch.int16, device = device)
                                 for _ in range(self.num_wslots)]
        # Reconstruct scratch sized for the largest projection registered so far; grows if a
        # later layer is larger
        if mx and (d["w_scratch"] is None or d["w_scratch"].numel() < mx):
            d["w_scratch"] = torch.empty(mx, dtype = torch.half, device = device)
        return d

    def _ensure_stream_state(self, device):
        key = torch.device(device).index or 0
        st = self.sstate.get(key)
        if st is not None:
            return st
        bufs = self._device_buffers(device)
        st = dict(
            copy_stream = torch.cuda.Stream(device = device),
            vram_slots = bufs["vram_slots"],
            native_slots = bufs["native_slots"],
            wready_ev = [torch.cuda.Event() for _ in range(self.num_wslots)],
            wconsumed_ev = [torch.cuda.Event() for _ in range(self.num_wslots)],
            wslot_used = [False] * self.num_wslots,
            w_scratch = bufs["w_scratch"],
            # Fused-tier (exl3_moe) temp buffers per (hidden, intermediate) shape and the
            # batched reconstruct tier state per layer, shared with the preallocated buffers
            fused_t = TUNING.stream_fused_t,
            fused_bufs = bufs["fused_bufs"],
            # Batched reconstruct tier state per layer (moe_batch_recon.BatchReconLayer)
            recon = bufs["recon"],
        )

        # Probe pinned->device bandwidth once: the break-even assignment count for streaming an
        # expert scales inversely with the link's bandwidth, so a chipset-attached x4 card needs
        # a much hotter expert to justify the weight DMA than a CPU-direct x16 one. An explicit
        # EXL3_MOE_STREAM_T overrides the scaling.
        probe = min(self.wslot_size, 16 << 20)
        ev0, ev1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        def timed_copy():
            ev0.record(st["copy_stream"])
            st["vram_slots"][0][:probe // 2].copy_(self.wviews[0][:probe // 2], non_blocking = True)
            ev1.record(st["copy_stream"])
            ev1.synchronize()
            return probe / (ev0.elapsed_time(ev1) * 1e-3) / 1e9   # GB/s
        with torch.cuda.stream(st["copy_stream"]):
            # streaming keeps the link awake, so the sustained awake rate is the one to use
            bw = probe_bandwidth(timed_copy)
        st["bw"] = bw
        if st["native_slots"] is not None:
            # The ring was carved on the compute stream during the measured load; the copy
            # stream writes into it, so its first write is ordered after that stream (the
            # allocator may have recycled a block whose last user is still queued there)
            st["copy_stream"].wait_stream(torch.cuda.current_stream())
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

    def _dq_linear(self, x, trellis_view, dims, suh, svh, bias, w_scratch, out_dtype = torch.half):
        """reconstruct-path linear: had_in(x * suh) @ W -> had_out * svh (+ bias). out_dtype
        follows the resident experts (fp32 for the down projection, the model's interm_dtype
        for gate/up): on models with massive activations the output-side Hadamard concentrates
        a 128-block past the fp16 range, and an in-place fp16 transform overflows to inf"""
        k, n, K = dims
        xh = torch.empty_like(x)
        ext.had_r_128(x, xh, suh, None, 1.0)
        w = w_scratch[:k * n].view(k, n)
        ext.reconstruct(w, trellis_view, K, False, True)
        y = torch.empty((x.shape[0], n), dtype = out_dtype, device = x.device)
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

    # -- promote-on-miss transfers ---------------------------------------------------------------
    #
    # With the pinned arena the parent maps the worker's expert memory, so an expert moves between
    # the GPU slot tensors and its arena block by DMA on a side stream: no checkpoint read, no
    # worker message, no quiesce. The arena holds the bytes in the CPU tier's packed layout
    # (_stream_layout), the GPU slots in native order, so a transfer repacks through the staging
    # buffers with the same bounded strided copies the child loader uses, including fractional
    # halfword planes. The caller orders the transfers against the compute streams
    # (block_sparse_mlp_cpu._split_promote_decide).

    def set_hot(self, hot):
        """Keep the worker polling between passes (the next one is a draft/verify handoff away, sooner than a nap's
        wake-up), or let it nap again"""
        if self.started:
            self.v_hot[0] = 1 if hot else 0


    def promote_ready(self, layer_idx, cpu_local):
        """Whether worker expert `cpu_local` of `layer_idx` can be moved by DMA: the arena is
        mapped here and the expert's chunks were page-locked (the pinning may have stopped short),
        in every worker"""
        if not self.pinned or layer_idx >= len(self.layer_blocks) or self.layer_blocks[layer_idx] is None:
            return False
        if layer_idx not in self.aux or self.specs[layer_idx].get("expert_bytes") is None:
            return False
        for st in self._states():
            chunks = {st.layer_blocks[layer_idx][cpu_local][0]}
            for proj in st.layer_auxloc[layer_idx][cpu_local]:
                chunks |= {loc[0] for loc in proj if loc is not None}
            if not all(c < len(st.arena_views) and st.arena_views[c] is not None for c in chunks):
                return False
        return True


    def _promo_state(self, device):
        key = torch.device(device).index or 0
        P = self._promo.get(key)
        if P is None:
            eb = max(sp.get("expert_bytes") or 0 for sp in self.specs)
            P = self._promo[key] = dict(
                stream = torch.cuda.Stream(device = device),
                raw = torch.empty(eb // 2, dtype = torch.int16, device = device),
            )
        return P


    def promo_stream(self, device):
        return self._promo_state(device)["stream"]


    def promo_drain(self):
        """Wait for the in-flight promote-on-miss transfers, ahead of releasing the GPU slots and arena pages they write"""
        for P in self._promo.values():
            P["stream"].synchronize()


    def _promo_parts(self, layer_idx, linears):
        """(projection index in the worker's (g, u, d) order, name, Linear, byte offset, bytes) of
        the expert's projections; `linears` is (gate or None, up, down)"""
        spec = self.specs[layer_idx]
        gb, ub, db = spec["proj_bytes"]
        parts, li, off = [], 0, 0
        for name, lin, nb in zip(("g", "u", "d"), linears, (gb, ub, db)):
            if lin is None:
                continue
            parts.append((li, name, lin, off, nb))
            li += 1
            off += nb
        return parts


    def promote_expert(self, layer_idx, cpu_local, linears):
        """Copy worker expert `cpu_local` into the GPU slot `linears` (gate or None, up, down):
        the trellis block by DMA from the arena, repacked to native order, the aux tensors from
        their GPU copies. Enqueued on the device's promotion stream; returns the event after it"""
        spec = self.specs[layer_idx]
        eb = spec["expert_bytes"]
        layout, _ = self._stream_layout(spec)
        aux = self.aux[layer_idx]
        up = linears[1]
        P = self._promo_state(up.trellis.device)
        stream = P["stream"]
        with torch.cuda.stream(stream):
            raw = P["raw"][:eb // 2]
            if self.workers > 1:
                # N-split: the workers' slices gathered into the full packed block
                ext.cuda_memcpy2d_async_batch(_gather_plan(
                    self._nsplit(spec)[1], [raw.data_ptr()], [self._expert_blocks(layer_idx, cpu_local)]), 1)
            else:
                ci, off = self.layer_blocks[layer_idx][cpu_local]
                src = self.arena_views[ci][off // 2 : (off + eb) // 2]
                # Raw memcpy: torch's copy_ does not take the registered arena for pinned memory and
                # would copy synchronously (the same for the device-to-host copies of demote_expert)
                ext.cuda_memcpy_async(raw.data_ptr(), src.data_ptr(), eb, 1)
            for li, name, lin, o, nb in self._promo_parts(layer_idx, linears):
                dst = lin.trellis.view(torch.int16)
                piece = raw[o // 2 : (o + nb) // 2]
                group, planar = layout[("g", "u", "d").index(name)]
                if group:
                    _copy_unpacked(dst, piece.view(dst.shape), group, planar)
                else:
                    dst.view(-1).copy_(piece)
                for an, attr in (("suh", "suh"), ("svh", "svh"), ("bias", "bias")):
                    lst = aux.get(f"{an}_{name}")
                    t = getattr(lin, attr, None)
                    if lst is None or lst[cpu_local] is None or t is None:
                        continue
                    t.copy_(lst[cpu_local], non_blocking = True)
            ev = torch.cuda.Event()
            ev.record(stream)
        return ev


    def demote_expert(self, layer_idx, cpu_local, linears):
        """Copy the GPU slot `linears` into worker expert `cpu_local`: the trellis block repacked
        and DMA'd into the arena, the aux tensors into both their GPU copies (the streamed
        prefill's) and the arena (the CPU kernel's). The worker slot must be unused (unmapped)
        until the returned event has landed"""
        spec = self.specs[layer_idx]
        eb = spec["expert_bytes"]
        if self.workers == 1:
            ci, off = self.layer_blocks[layer_idx][cpu_local]
            dst_block = self.arena_views[ci][off // 2 : (off + eb) // 2]
        layout, _ = self._stream_layout(spec)
        aux = self.aux[layer_idx]
        locs = self.layer_auxloc[layer_idx][cpu_local]
        up = linears[1]
        P = self._promo_state(up.trellis.device)
        stream = P["stream"]
        hold = []
        with torch.cuda.stream(stream):
            raw = P["raw"][:eb // 2]
            for li, name, lin, o, nb in self._promo_parts(layer_idx, linears):
                srct = lin.trellis.view(torch.int16)
                piece = raw[o // 2 : (o + nb) // 2]
                group, planar = layout[("g", "u", "d").index(name)]
                if group:
                    _copy_repacked(piece.view(srct.shape), srct, group, planar)
                else:
                    piece.copy_(srct.view(-1))
                for j, (an, attr) in enumerate((("suh", "suh"), ("svh", "svh"), ("bias", "bias"))):
                    lst = aux.get(f"{an}_{name}")
                    t = getattr(lin, attr, None)
                    if lst is None or lst[cpu_local] is None or t is None:
                        continue
                    lst[cpu_local].copy_(t, non_blocking = True)
                    if self.workers > 1:
                        tc = t.contiguous()
                        if tc is not t:
                            hold.append(tc)
                        self._demote_aux_nsplit(layer_idx, cpu_local, li, j, name, tc)
                        continue
                    loc = locs[li][j]
                    if loc is not None:
                        lc, lo, ln = loc
                        tc = t.contiguous()
                        if tc is not t:
                            hold.append(tc)      # a temporary: kept alive until the copy has landed
                        ext.cuda_memcpy_async(self.arena_views[lc].data_ptr() + lo, tc.data_ptr(), ln, 2)
            if self.workers > 1:
                # N-split: the packed block scattered back into the workers' slices
                ext.cuda_memcpy2d_async_batch(_scatter_plan(
                    self._nsplit(spec)[1], raw.data_ptr(), self._expert_blocks(layer_idx, cpu_local)), 2)
            else:
                ext.cuda_memcpy_async(dst_block.data_ptr(), raw.data_ptr(), eb, 2)
            ev = torch.cuda.Event()
            ev.record(stream)
            if hold:
                # Temporaries live until their copies have landed; landed entries are dropped on the
                # next call (no synchronization on the transfer stream)
                P["hold"] = [(e, ts) for e, ts in P.get("hold", ()) if not e.query()] + [(ev, hold)]
        return ev


    def install_expert(self, layer_idx, local_idx, keys):
        """Dynamic placement: replace worker expert `local_idx` of `layer_idx` with the
        checkpoint tensors at `keys` (per-projection prefixes, gate first when gated). The
        caller must have quiesced (full stream sync => job ring drained, worker idle) before
        calling; blocks until the child acks the in-place arena copy (every worker its slice)."""
        self._send_all(("install", layer_idx, local_idx, keys))
        # Every worker's ack before raising: an unread one would answer the next install
        msgs = [st.conn.recv() for st in self._states()]
        for w, msg in enumerate(msgs):
            if msg[0] != "ok":
                tag = f"CPU MoE worker {w}" if self._xw else "CPU MoE worker"
                raise RuntimeError(f"{tag} expert install failed: {msg[1] if len(msg) > 1 else msg}")

    def _expert_blocks(self, layer_idx, cpu_local):
        """Host address of expert `cpu_local`'s arena block in each worker (pinned arena)"""
        out = []
        for wst in self._states():
            ci, off = wst.layer_blocks[layer_idx][cpu_local]
            out.append(wst.arena_views[ci].data_ptr() + off)
        return out

    def _nsplit_aux_slices(self, name, j, numel):
        """Per worker, the [lo, hi) element range of a projection's aux tensor j (suh, svh, bias)
        that worker holds, None where it holds none. Gate/up are cut along their output (svh,
        bias), down along its input (suh); down's bias stays with worker 0"""
        ranges = None
        out = []
        for w in range(self.workers):
            sliced = (name != "d" and j in (1, 2)) or (name == "d" and j == 0)
            if not sliced:
                out.append(None if (name == "d" and j == 2 and w != 0) else (0, numel))
                continue
            if ranges is None:
                ranges = _slice_ranges(numel, self.workers)
            out.append(ranges[w])
        return out

    def _demote_aux_nsplit(self, layer_idx, cpu_local, li, j, name, t):
        """A demoted expert's aux tensor j of projection `name` (GPU, contiguous) into each
        worker's arena copy of its slice"""
        es = t.element_size()
        for w, (wst, rng) in enumerate(zip(self._states(), self._nsplit_aux_slices(name, j, t.numel()))):
            loc = wst.layer_auxloc[layer_idx][cpu_local][li][j]
            if loc is None or rng is None:
                continue
            lc, lo, ln = loc
            assert ln == (rng[1] - rng[0]) * es, "N-split aux slice size mismatch"
            ext.cuda_memcpy_async(wst.arena_views[lc].data_ptr() + lo, t.data_ptr() + rng[0] * es, ln, 2)

    def arena_expert(self, layer_idx, cpu_local):
        """Expert `cpu_local`'s packed trellis block as a single worker holds it (a view of the
        arena; a copy gathered from the workers' slices under the N-split), for verification"""
        spec = self.specs[layer_idx]
        eb = spec["expert_bytes"]
        if self.workers == 1:
            ci, off = self.layer_blocks[layer_idx][cpu_local]
            return self.arena_views[ci][off // 2 : (off + eb) // 2]
        out = torch.empty(eb, dtype = torch.uint8)
        srcs = [wst.arena_views[wst.layer_blocks[layer_idx][cpu_local][0]].view(torch.uint8)
                [wst.layer_blocks[layer_idx][cpu_local][1]:] for wst in self._states()]
        for w, wp in enumerate(self._nsplit(spec)[1]):
            for s_off, d_off, width, height, spitch, dpitch in wp:
                for r in range(height):
                    out[d_off + r * dpitch : d_off + r * dpitch + width] = \
                        srcs[w][s_off + r * spitch : s_off + r * spitch + width]
        return out.view(torch.int16)

    def arena_aux(self, layer_idx, cpu_local, li, j, name, numel):
        """An expert's aux tensor j of projection `name` as the CPU kernels see it, int16 (a view of
        the arena; a copy reassembled from the workers' slices under the N-split); None if not stored"""
        if self.workers == 1:
            loc = self.layer_auxloc[layer_idx][cpu_local][li][j]
            if loc is None:
                return None
            lc, lo, ln = loc
            return self.arena_views[lc][lo // 2 : (lo + ln) // 2]
        out = torch.empty(numel, dtype = torch.int16)
        have = False
        for wst, rng in zip(self._states(), self._nsplit_aux_slices(name, j, numel)):
            loc = wst.layer_auxloc[layer_idx][cpu_local][li][j]
            if loc is None or rng is None:
                continue
            lc, lo, ln = loc
            out[rng[0]:rng[1]] = wst.arena_views[lc][lo // 2 : (lo + ln) // 2]
            have = True
        return out if have else None

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

    def _stream_fused_t(self, spec, aux, h):
        """Fused-kernel row capacity for a streamed layer, 0 when the layer isn't eligible
        (same rule as support_fused on the GPU side: silu/gelu gated or relu2 gateless, no
        per-expert biases, no padded dims)"""
        return TUNING.stream_fused_t if (
            spec["activation"] in (0, 1, 2) and spec["hi"] == h and spec["ho"] == h
            and not any(aux.get(b) is not None for b in ("bias_g", "bias_u", "bias_d"))
        ) else 0

    def _stream_recon_layer(self, st, layer_idx, spec, aux, device):
        """Batched reconstruct tier state for a streamed layer, built once per (device,
        layer); None when disabled or the experts carry biases (a batched add would be needed)"""
        if not self.batch_recon or any(aux.get(b) is not None for b in ("bias_g", "bias_u", "bias_d")):
            return None
        recon = st["recon"].get(layer_idx)
        if recon is None:
            from ..modules.moe_batch_recon import BatchReconLayer
            pd = spec["proj_dims"]
            gated = pd.get("g") is not None
            scales = {p: (aux["suh_" + p], aux["svh_" + p])
                      for p in (("g", "u", "d") if gated else ("u", "d"))}
            recon = BatchReconLayer(
                pd.get("g"), pd["u"], pd["d"], (False, True), (False, True), (False, True),
                spec["activation"], spec["act_limit"], device, scales)
            st["recon"][layer_idx] = recon
        return recon

    def _stream_fused_bufs(self, st, spec, device):
        """Fused-tier temp buffers for a streamed layer's (hidden, intermediate) shape, kept
        per device (the kernel reads both dims from the buffers, so they must match the layer)"""
        key = (spec["hi"], spec["proj_dims"]["u"][1])
        fbufs = st["fused_bufs"].get(key)
        if fbufs is None:
            conc = ext.exl3_moe_max_concurrency(torch.device(device).index or 0)
            fbufs = tuple(
                torch.empty((conc, TUNING.stream_fused_t, dim), dtype = torch.half, device = device)
                for dim in (key[0], key[0], key[1], key[1]))
            st["fused_bufs"][key] = fbufs
        return fbufs

    def prefill_worst_case_parts(self, layer_idx, rows, device, assignments):
        """Autosplit worst case for one layer's prefill on `device` with `assignments` routed
        rows on CPU-resident experts, as (fixed, variable) bytes: the outputs and lists that
        live for the whole call, and the larger of one batched-reconstruct group and the
        per-expert path (freed before the GPU-side tiers allocate theirs). The persistent
        device buffers are allocated for real here (_device_buffers) so the load accounts for
        them"""
        spec = self.specs[layer_idx]
        h, hi = spec["ho"], spec["hi"]
        A = assignments
        # Plain CPU path: the fp32 output and the readback staging
        fixed = rows * h * 4 + min(self.cap_rows, rows) * h * 4
        if (rows < self.stream_min_rows or spec.get("expert_bytes") is None
                or spec["expert_bytes"] > self.wslot_size or layer_idx not in self.aux):
            return fixed, 0
        aux = self.aux[layer_idx]
        pd = spec["proj_dims"]
        gated = pd.get("g") is not None
        n = pd["u"][1]
        # Streamed path: padded fp32 output, the tail's compressed output, the input copy with
        # its zero row, and the sorted assignment lists
        fixed += (rows + 1) * h * 4 + rows * h * 4 + (rows + 1) * hi * 2 + A * 32
        with torch.cuda.device(device):
            bufs = self._device_buffers(device, spec)
            if self._stream_fused_t(spec, aux, h):
                self._stream_fused_bufs(bufs, spec, device)
            recon = self._stream_recon_layer(bufs, layer_idx, spec, aux, device)
        r = min(rows, A)
        per_expert = r * (hi * 2 + n * 2 * (2 if gated else 1) + h * 4)
        batched = recon.worst_case_bytes(A, slot_mode = False) if recon is not None else 0
        return fixed, max(per_expert, batched)

    def _nsplit(self, spec):
        """(worker block bytes, pieces) of a layer's experts under the N-split, cached on the spec"""
        ns = spec.get("nsplit")
        if ns is None:
            ns = spec["nsplit"] = _nsplit_pieces(spec["proj_dims"], self._stream_layout(spec)[0], self.workers)
        return ns

    def _nsplit_restore(self, spec):
        """Per projection of a layer: the N-split's tile boundaries (along n for gate/up, along k
        for down) and each worker's byte offset of the projection within its expert block"""
        ns = spec.get("nsplit_restore")
        if ns is None:
            pd = spec["proj_dims"]
            ranges = _slice_ranges(pd["u"][1], self.workers)
            split = [lo // 16 for lo, _ in ranges] + [ranges[-1][1] // 16]
            names = [nm for nm in ("g", "u", "d") if pd.get(nm)]
            pieces = self._nsplit(spec)[1]
            ns = spec["nsplit_restore"] = {nm: (split, [pieces[w][i][0] for w in range(self.workers)])
                                           for i, nm in enumerate(names)}
        return ns

    def _nsplit_pinned(self, layer_idx, experts):
        """Whether every worker's arena block of these experts is mapped and page-locked here (the
        pinning may stop short in any worker; those experts take the staged path)"""
        for wst in self._states():
            blocks = wst.layer_blocks[layer_idx] if layer_idx < len(wst.layer_blocks) else None
            if blocks is None:
                return False
            reg = wst.arena_registered
            for e in experts:
                ci = blocks[e][0]
                if (reg is not None and ci >= reg) or ci >= len(wst.arena_views) or wst.arena_views[ci] is None:
                    return False
        return True

    def _push_stage_job(self, st, seq, layer_idx, batch, ws):
        """Stage job into one worker's stage ring (its stager copies its slices of `batch` into
        its staging slot ws once the slot's previous tenant has been DMA'd out)"""
        stail = int(st.v_stage_tail[0])
        while stail - int(st.v_stage_head[0]) >= MOE_STAGE_RING - 2:
            import time
            if self.v_abort[0] or not self._alive():
                raise RuntimeError("CPU MoE worker failed (stage ring stall)")
            time.sleep(0.0002)
        job = st.v_stage_jobs[stail % MOE_STAGE_RING]
        _set_job_seq(job, seq)
        job[1] = layer_idx
        job[2] = len(batch)
        job[3] = 0
        job[4] = ws
        job[5] = 1    # MOE_JOB_KIND_STAGE
        job[6] = self.wslot_prev_seq[ws] & 0xFFFFFFFF   # compared cyclically by the worker
        for bi, e in enumerate(batch):
            job[7 + bi] = e
        st.v_stage_tail[0] = stail + 1

    def _submit_prefill_streamed(self, layer_idx, y, selected_experts, routing_weights, spec,
                                 streamed, st, counts_h, flat, shifted, neg):
        from ..modules.moe_batch_recon import plan_groups
        rows = y.shape[0]
        h = y.shape[1]
        E = spec["num_experts"]
        topk = selected_experts.shape[1]
        # One spare row: the batched reconstruct tier's padding sink (never read back)
        out_ext = torch.zeros((rows + 1, h), dtype = torch.float, device = y.device)
        out = out_ext[:rows]

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
        smask1 = host_to_device(torch.from_numpy(table), y.device)
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
        # Pinned arena: per-expert (chunk, offset) of the DMA source; None = staged path. Experts
        # in chunks past the registered ones (see _attach_chunk) go through the staged path too,
        # decided per batch below
        blocks = self.layer_blocks[layer_idx] if self.pinned else None
        pinned_chunks = self.arena_registered

        # Packed-layout policy of this layer's projections (see _stream_layout): packed bytes
        # are restored to native order after the DMA, so the kernels below always see native
        layout, restore = self._stream_layout(spec)
        # N-split: every worker holds a slice of each expert, DMA'd verbatim and reassembled into
        # the full native expert by the restore pass (moe_unswizzle_trellis_nsplit), so the GPU
        # tiers below run as for a single worker
        nsplit = self._nsplit(spec) if self.workers > 1 else None
        if nsplit is not None:
            nsplit_off = [sum(nsplit[0][:w]) for w in range(self.workers)]
            nsplit_restore = self._nsplit_restore(spec)

        # Mid-tier experts (count <= fused_t) run through the fused MoE kernel per staged batch;
        # experts too hot for the temp buffers take the per-expert reconstruct path. Same
        # eligibility as support_fused on the GPU side: mul1 (given), silu/gelu gated or relu2
        # gateless, no per-expert biases, no padded dims
        fused_t = self._stream_fused_t(spec, aux, h)
        recon = self._stream_recon_layer(st, layer_idx, spec, aux, y.device)
        recon_ctx = None
        fbufs = None
        if fused_t and any(counts_h[e] <= fused_t for e in streamed):
            fbufs = self._stream_fused_bufs(st, spec, y.device)

        for i0 in range(0, len(streamed), per_slot):
            batch = streamed[i0:i0 + per_slot]
            ws = self.next_wslot
            self.next_wslot = (self.next_wslot + 1) % self.num_wslots
            self.wseq += 1
            seq = self.wseq

            if nsplit is not None and self.pinned and self._nsplit_pinned(layer_idx, batch):
                # Pinned arena, N-split: each worker's block of each expert DMA'd verbatim (one 2D
                # copy per worker when the batch's blocks sit at a regular stride in its arena) to
                # the expert's slot, worker blocks back to back; the restore below reassembles
                # the full native expert
                with torch.cuda.stream(copy_stream):
                    if st["wslot_used"][ws]:
                        copy_stream.wait_event(st["wconsumed_ev"][ws])
                    raw = st["vram_slots"][ws].data_ptr()
                    dma = []
                    for w, wst in enumerate(self._states()):
                        locs = [wst.layer_blocks[layer_idx][e] for e in batch]
                        addrs = [wst.arena_views[ci].data_ptr() + off for ci, off in locs]
                        # Fold into one 2D copy only within one arena chunk: the pitched span is
                        # what the driver checks for page-locking, and separate chunks can be far
                        # apart (or out of order) in the address space
                        stride = locs[1][1] - locs[0][1] if len(locs) > 1 else 0
                        if len(locs) > 1 and all(ci == locs[0][0] for ci, _ in locs) and \
                                stride >= nsplit[0][w] and all(b[1] - a[1] == stride for a, b in zip(locs, locs[1:])):
                            dma.append((raw + nsplit_off[w], exp_b, addrs[0], stride, nsplit[0][w], len(batch)))
                        else:
                            dma += [(raw + bi * exp_b + nsplit_off[w], 0, a, 0, nsplit[0][w], 1)
                                     for bi, a in enumerate(addrs)]
                    ext.cuda_memcpy2d_async_batch(torch.tensor(dma, dtype = torch.int64), 1)
                nsplit_src = (nsplit_off, [exp_b] * self.workers)
            elif nsplit is not None:
                # Staged, N-split: every worker's stager writes its block of each expert into its
                # own staging region; once the last has (it publishes stage_done), one DMA per
                # worker, then pinned_free releases the slot to every stager
                wexp = nsplit[0]
                for wst in self._states():
                    self._push_stage_job(wst, seq, layer_idx, batch, ws)
                self.wslot_prev_seq[ws] = seq
                with torch.cuda.stream(copy_stream):
                    if st["wslot_used"][ws]:
                        copy_stream.wait_event(st["wconsumed_ev"][ws])
                    ext.exl3_moe_flag_wait(self.stage_done_addr[ws], seq, abort)
                    raw = st["vram_slots"][ws].data_ptr()
                    dma = [(raw + len(batch) * nsplit_off[w], 0, wst.wviews[ws].data_ptr(), 0,
                             len(batch) * wexp[w], 1) for w, wst in enumerate(self._states())]
                    ext.cuda_memcpy2d_async_batch(torch.tensor(dma, dtype = torch.int64), 1)
                    ext.exl3_moe_flag_write(self.pinned_free_addr[ws], seq)
                nsplit_src = ([len(batch) * o for o in nsplit_off], wexp)
            elif blocks is not None and (pinned_chunks is None or all(blocks[e][0] < pinned_chunks for e in batch)):
                # Pinned arena: DMA each expert's contiguous block straight out of the
                # registered arena mapping on the copy stream (no worker involvement, the
                # stager thread stays idle)
                with torch.cuda.stream(copy_stream):
                    if st["wslot_used"][ws]:
                        copy_stream.wait_event(st["wconsumed_ev"][ws])
                    raw = st["vram_slots"][ws]
                    for bi, e in enumerate(batch):
                        ci, off = blocks[e]
                        raw[bi * exp_b // 2 : (bi + 1) * exp_b // 2].copy_(
                            self.arena_views[ci][off // 2 : (off + exp_b) // 2],
                            non_blocking = True)
            else:
                # Stage job: its own ring, consumed by the worker's dedicated stager thread, so
                # the weight memcpys overlap the compute pool's work on the tail
                stail = int(self.v_stage_tail[0])
                while stail - int(self.v_stage_head[0]) >= MOE_STAGE_RING - 2:
                    import time
                    if self.v_abort[0] or not self.proc.is_alive():
                        raise RuntimeError("CPU MoE worker failed (stage ring stall)")
                    time.sleep(0.0002)
                job = self.v_stage_jobs[stail % MOE_STAGE_RING]
                _set_job_seq(job, seq)
                job[1] = layer_idx
                job[2] = len(batch)
                job[3] = 0
                job[4] = ws
                job[5] = 1    # MOE_JOB_KIND_STAGE
                job[6] = self.wslot_prev_seq[ws] & 0xFFFFFFFF   # compared cyclically by the worker
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

            if (restore or nsplit is not None) and st["native_slots"] is None:
                # First batch on this device that needs native-order bytes: adopt the ring the
                # measured load allocated, or allocate it now (manual loads) on the compute
                # stream. The caching allocator ties the block to the stream it was carved on,
                # so the copy stream's first write into it is ordered after that stream
                st["native_slots"] = self._device_buffers(y.device, spec)["native_slots"]
                copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(copy_stream):
                if nsplit is not None:
                    # N-split: reassemble every full native projection from the workers' blocks
                    # (and restore the native order of packed ones) on the copy stream, one
                    # launch per projection over the whole batch
                    w_off, w_stride = nsplit_src
                    for pi, (name, off) in enumerate((("g", 0), ("u", gb), ("d", gb + ub))):
                        if not pd.get(name):
                            continue
                        k, n, K = pd[name]
                        split, w_proj = nsplit_restore[name]
                        ext.moe_unswizzle_trellis_nsplit(
                            st["vram_slots"][ws], st["native_slots"][ws], len(batch), exp_b, off,
                            k // 16, n // 16, K, layout[pi][0], layout[pi][1], name == "d",
                            split, w_off, w_stride, w_proj)
                elif restore:
                    # Restore native tile and dword/halfword order on the copy stream, one launch per
                    # projection over the whole batch (group-0 projections in an otherwise
                    # packed layer take the plain-copy path in the same kernel)
                    for pi, (name, off) in enumerate((("g", 0), ("u", gb), ("d", gb + ub))):
                        if not pd.get(name):
                            continue
                        k, n, K = pd[name]
                        ext.moe_unswizzle_trellis(
                            st["vram_slots"][ws], st["native_slots"][ws], len(batch), exp_b, off,
                            k // 16, n // 16, K, layout[pi][0], layout[pi][1])
                st["wready_ev"][ws].record(copy_stream)
            st["wslot_used"][ws] = True

            # Compute the batch on the current stream once the DMA lands
            torch.cuda.current_stream().wait_event(st["wready_ev"][ws])
            vslot = st["native_slots"][ws] if (restore or nsplit is not None) else st["vram_slots"][ws]
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
                tblt = host_to_device(torch.tensor(tbl, dtype = torch.int64), y.device)
                ec = host_to_device(torch.tensor([counts_h[e] for _, e, _, _ in per_e] + [0], dtype = torch.long), y.device)
                tok = torch.cat([seg for _, _, seg, _ in per_e])
                wts = torch.cat([wseg for _, _, _, wseg in per_e]).half()
                Ku, Kd = pd["u"][2], pd["d"][2]
                Kg = pd["g"][2] if gated else Ku
                # Row-tile tiers as on the GPU side (block_sparse_mlp): one launch per tile over
                # its expert range. Streamed experts are mul1 by construction
                fc = [counts_h[e] for _, e, _, _ in per_e if counts_h[e] <= fused_t]
                t1 = sum(1 for c in fc if 16 < c <= 32)
                t2 = sum(1 for c in fc if c > 32)
                tiers = [(t2, 33, fused_t, 64), (t1, 17, 32, 32), (len(fc) - t1 - t2, 1, 16, 16)] \
                    if TUNING.mtile and (t1 or t2) else [(n_fused, 1, fused_t, 16)]
                for n_act, lo, hi, mt in tiers:
                    if not n_act:
                        continue
                    ext.exl3_moe(
                        y, out, ec, tok, wts,
                        fbufs[0], fbufs[1], fbufs[2], fbufs[3],
                        spec["activation"], Kg, Ku, Kd,
                        tblt[0], tblt[1], tblt[2], tblt[3], tblt[4], tblt[5],
                        tblt[6], tblt[7], tblt[8],
                        False, True, False, True, False, True,
                        float(spec["act_limit"] or 0.0), n_act, None, None, lo, hi, mt
                    )

            # Heavy tier: batched reconstruct (groups of experts, a handful of launches per
            # group; see moe_batch_recon.py) when eligible, else per expert
            heavy = [(bi, e) for bi, e, _, _ in per_e if not (fused_t and counts_h[e] <= fused_t)]
            if recon is not None:
                # Experts above the batched tier's row cap stay on the per-expert loop (large
                # slabs pad and stream more than they save in launches)
                single = [(bi, e) for bi, e in heavy if counts_h[e] > recon.max_rows]
                heavy = [(bi, e) for bi, e in heavy if counts_h[e] <= recon.max_rows]
            else:
                # No batched tier (EXL3_MOE_STREAM_BATCH_RECON=0 or biased experts): every
                # heavy expert runs on the per-expert loop below
                single = heavy
                heavy = []
            if heavy:
                if recon_ctx is None:
                    # Padding sources / sinks: a zero input row, an output sink row (out is a
                    # view of the first `rows` rows of out_ext) and sentinel entries after the
                    # sorted assignment lists
                    hi = spec["hi"]
                    y_in = y if y.shape[1] == hi else torch.nn.functional.pad(y, (0, hi - y.shape[1]))
                    recon_ctx = (
                        torch.cat([y_in, torch.zeros((1, hi), dtype = y.dtype, device = y.device)]),
                        torch.cat([token_sorted, torch.full((1,), rows, dtype = token_sorted.dtype,
                                                            device = y.device)]),
                        torch.cat([weight_sorted.half(), torch.zeros((1,), dtype = torch.half,
                                                                     device = y.device)]),
                    )
                y_ext, tok_ext, w_ext = recon_ctx
                base = vslot.data_ptr()
                slot_of = {e: bi for bi, e in heavy}
                for grp in plan_groups([e for _, e in heavy], lambda e: counts_h[e], recon.cap):
                    # Trellis addresses inside the VRAM slot, per projection
                    bb = [base + slot_of[e] * exp_b for e in grp]
                    recon.run_group(
                        y_ext, out_ext, tok_ext, w_ext,
                        grp, [offs[e] for e in grp], [counts_h[e] for e in grp],
                        ptrs = (bb if gated else None, [b + gb for b in bb], [b + gb + ub for b in bb]))
                heavy = []
            single_ids = {e for _, e in single} if recon is not None else None
            for bi, e, idx, wseg in per_e:
                if fused_t and counts_h[e] <= fused_t:
                    continue
                if single_ids is not None and e not in single_ids:
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
                    numel = (k // 16) * (n // 16) * int(16 * K)
                    return vslot[boff + off_b // 2 : boff + off_b // 2 + numel] \
                        .view(k // 16, n // 16, int(16 * K))
                # Same output dtypes as the resident experts: the intermediate as the model's
                # interm_dtype, the down projection fp32 (the activation casts to half for the
                # down GEMM either way)
                idt = torch.float if spec.get("interm_fp32") else torch.half
                if gated:
                    gy = self._dq_linear(xg, tview(0, pd["g"]), pd["g"],
                                         aux["suh_g"][e], aux["svh_g"][e],
                                         aux["bias_g"][e] if aux.get("bias_g") else None,
                                         st["w_scratch"], out_dtype = idt)
                uy = self._dq_linear(xg, tview(gb, pd["u"]), pd["u"],
                                     aux["suh_u"][e], aux["svh_u"][e],
                                     aux["bias_u"][e] if aux.get("bias_u") else None,
                                     st["w_scratch"], out_dtype = idt)
                a = self._act(spec, gy if gated else None, uy) if gated else self._act(spec, None, uy)
                dy = self._dq_linear(a, tview(gb + ub, pd["d"]), pd["d"],
                                     aux["suh_d"][e], aux["svh_d"][e],
                                     aux["bias_d"][e] if aux.get("bias_d") else None,
                                     st["w_scratch"], out_dtype = torch.float)
                out.index_add_(0, idx, dy[:, :h] * we)
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
            if self.started and self.shm is not None:
                # The flag stops every worker's compute thread
                self.v_quit[0] = 1
            for st in self._states():
                if st.proc is None:
                    continue
                try:
                    # The child's main thread serves the pipe at runtime (expert installs); the
                    # message unblocks its recv loop
                    if st.conn is not None:
                        st._send_queue.put(("quit",))
                    st._send_queue.put(None)
                    st._sender.join(timeout = 5)
                except Exception:
                    pass
            for st in self._states():
                if st.proc is None:
                    continue
                try:
                    st.proc.join(timeout = 5)
                    if st.proc.is_alive():
                        st.proc.terminate()
                        st.proc.join(timeout = 2)
                    if st.proc.is_alive():
                        st.proc.kill()
                except Exception:
                    pass
                st.proc = None
        # Full reset: a later re-registration must not resolve stale indices against a child
        # that no longer exists
        self.specs = []
        self.by_key = {}
        self.aux = {}
        self.acked = 0
        self.live_layers = 0
        self.seq = self.wseq = 0
        self.next_slot = self.next_wslot = 0
        self.slot_last_seq = [0] * MOE_MAX_SLOTS
        self.wslot_prev_seq = [0] * MOE_MAX_WSLOTS
        cleanupper.unregister_atexit(self.shutdown)
        for st in self._states():
            if st.conn is not None:
                try:
                    st.conn.close()
                except Exception:
                    pass
                st.conn = None
        for x in self._xw:
            # Drop every view over the segment before it is closed below
            for nm in ("v_ready", "v_jobs_tail", "v_jobs_head", "v_jobs", "v_stage_tail", "v_stage_head",
                       "v_stage_jobs", "wviews"):
                setattr(x, nm, None)
        if self.shm is not None:
            try:
                cuda_host_unregister(self.base_ptr)
            except Exception:
                pass
            # Drop every view over the buffer before closing, or mmap refuses to unmap
            self.slots = None
            self.wviews = None
            self.sstate = None
            self._dev_bufs = {}
            self.v_quit = self.v_pass_wake = self.v_hot = self.v_abort = self.v_ready = None
            self.v_jobs_tail = self.v_jobs_head = self.v_jobs = None
            self.v_stage_tail = self.v_stage_head = self.v_stage_jobs = None
            self._flags_u32 = None
            import gc
            gc.collect()
        # Pinned arena mappings: unpin, drop the views, unmap. The pages themselves die with
        # the worker (memfd, no name to unlink; a Windows section goes with its last handle)
        maps = []
        for st in self._states():
            for view in st.arena_views:
                if view is None:
                    continue
                try:
                    cuda_host_unregister(view.data_ptr())
                except Exception:
                    pass
            st.arena_views = []
            st.layer_blocks = []
            st.layer_auxloc = []
            maps += st.arena_maps
            st.arena_maps = []
            if st._arena_pending is not None:
                maps.append(st._arena_pending[3])
                st._arena_pending = None
        self._promo = {}
        if maps:
            import gc
            gc.collect()
            for m in maps:
                if m is None:
                    continue
                try:
                    m.close()
                except Exception:
                    pass
        if self.shm is not None:
            try:
                self.shm.close()
                self.shm.unlink()
            except Exception:
                pass
            self.shm = None
        self.started = False
        self.by_key = {}
        # Workers 1..W-1 start a later load from scratch (worker 0's state is reset above)
        self._xw = [_ExtraWorker() for _ in range(1, self.workers)]
