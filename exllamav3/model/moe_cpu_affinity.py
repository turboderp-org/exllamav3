"""
Host-side CPU placement for the CPU MoE offload path.

The worker pool (cpu/moe_mul1.cpp Pool) pins one compute thread per physical core, in
physical_core_order(): one LP per core first, then every SMT sibling. The parent process (the
thread driving the forward pass, CUDA's driver threads, an API server's executor threads) is
not pinned, so the scheduler can drop it on a worker's LP. A pinned worker cannot move away,
so it becomes the straggler at every per-phase barrier of the job: measured on a 12-core
Ryzen 9 7900X / RTX 4090 running Qwen3.8-Flash-Next with 408 of 512 experts per layer on the
CPU, decode swung 11-32 tok/s with the host unpinned and settled at 30-36 tok/s once the host
was kept off the worker LPs. This module plans and applies that placement.
"""
import os
import sys


def _lp(enc: int) -> int:
    return enc & 0xFFFF


def default_worker_threads(n_physical: int, host_cores: int) -> int:
    """Pool size when nothing explicit is configured: every physical core minus the reserved
    host cores, never below one."""
    return max(1, n_physical - max(0, host_cores))


def plan_host_cpus(order: list[int], n_physical: int, threads: int, host_cores: int) -> list[int] | None:
    """
    LPs (encoded as the pool encodes them) the host process should be confined to, or None
    when no placement is possible or wanted.

    Workers take order[:threads]. Reserved cores are the last `host_cores` physical cores no
    worker occupies; the host gets those cores' LPs and their SMT siblings. When the workers
    cover every core the host gets the sibling LPs no worker took instead (it shares cores but
    never an LP with a worker). Multi-group Windows encodings are refused: the process-wide
    affinity API addresses one processor group.
    """
    if host_cores <= 0 or n_physical <= 0 or not order:
        return None
    if any(enc >> 16 for enc in order):
        return None
    primaries = order[:n_physical]
    siblings = order[n_physical:]
    if threads < n_physical:
        chosen = range(threads, n_physical)[-host_cores:]
        # x86 SMT is 2-way and the pool appends siblings in core order: core k's sibling is
        # siblings[k] (absent for cores without SMT, e.g. hybrid E-cores enumerated last)
        return [primaries[k] for k in chosen] + [siblings[k] for k in chosen if k < len(siblings)]
    return order[threads:] or None


def process_cpus() -> list[int]:
    """LPs this process may currently run on (Windows: its processor group's mask). Raises
    OSError when the OS refuses."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        proc_mask, sys_mask = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.byref(proc_mask), ctypes.byref(sys_mask)):
            raise ctypes.WinError(ctypes.get_last_error())
        return [lp for lp in range(64) if proc_mask.value >> lp & 1]
    return sorted(os.sched_getaffinity(0))


def apply_process_affinity(cpus: list[int]) -> str | None:
    """Confine every thread of this process to `cpus`. Returns None on success, else a reason.
    Threads created later inherit the mask (Windows: process mask; Linux: the creator's)."""
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            kernel32.SetProcessAffinityMask.restype = ctypes.c_int
            mask = 0
            for enc in cpus:
                mask |= 1 << _lp(enc)
            if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask):
                return f"SetProcessAffinityMask failed (error {ctypes.get_last_error()})"
            return None
        if hasattr(os, "sched_setaffinity"):
            cpu_set = {_lp(enc) for enc in cpus}
            for tid in os.listdir("/proc/self/task"):
                try:
                    os.sched_setaffinity(int(tid), cpu_set)
                except ProcessLookupError:
                    pass   # thread exited between listdir and the call
            return None
        return "no affinity API on this platform"
    except OSError as e:
        return str(e)
