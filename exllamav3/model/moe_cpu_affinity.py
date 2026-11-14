"""
Host-side CPU placement for the CPU MoE offload path.

The worker pool (cpu/moe_mul1.cpp Pool) runs one compute thread per physical core, pinned by
default in physical_core_order(): one LP per core first, then every SMT sibling. The parent
process (the thread driving the forward pass, CUDA's driver threads, an API server's executor
threads) is not pinned, so the scheduler can drop it on a worker's LP, where the worker becomes
the straggler at every per-phase barrier of the job. This module plans and applies the host
placement, with the workers pinned or floating alike (the pool reports its topology either way).

Scope rule: affinity is applied to the calling thread only (Linux; threads the caller creates
from then on inherit the mask) or to the process (Windows, where the process mask is the only
handle). Never sweep every existing thread of the process (/proc/self/task): in a server that
crushes dozens of unrelated threads -- CUDA driver threads, API executors, torch pools -- onto
the few host LPs, which poisons every non-worker phase of the process (loading, tokenization,
API handling) for the process lifetime.
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
    """LPs the calling thread may currently run on (Windows: its processor group's mask). Raises
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
    """Confine the calling thread to `cpus` (Linux) or the whole process (Windows). Returns None
    on success, else a reason. Linux: threads the caller creates from then on inherit the mask;
    existing threads are left untouched -- see the scope rule in the module docstring."""
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
            os.sched_setaffinity(0, {_lp(enc) for enc in cpus})
            return None
        return "no affinity API on this platform"
    except OSError as e:
        return str(e)


def _parse_cpu_list(s: str) -> list[int]:
    """"0-3,8,10-11" -> [0, 1, 2, 3, 8, 10, 11] (sysfs cpu list format)"""
    out = []
    for part in s.strip().split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def node_sibling_groups(node: int) -> dict[int, list[int]]:
    """SMT groups of one NUMA node from sysfs thread_siblings_list -- the kernel's own sibling
    relation, not the (package, core) heuristic whose key repeats across dies/tiles on some
    packages and silently collapses distinct cores. Returns {min sibling LP: [member LPs]}.
    Raises RuntimeError when a sibling list is unreadable, inconsistent between members of one
    core, or reaches outside the node: the worker picker refuses to guess SMT relations."""
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            cpus = _parse_cpu_list(f.read())
    except OSError as e:
        raise RuntimeError(f"node{node}/cpulist: {e}") from e
    cpu_set = set(cpus)
    groups: dict[int, list[int]] = {}
    declared: dict[int, list[int]] = {}
    for cpu in cpus:
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
                sib = sorted(set(_parse_cpu_list(f.read())))
        except OSError as e:
            raise RuntimeError(f"cpu{cpu}/thread_siblings_list: {e}") from e
        if not sib or cpu not in sib:
            raise RuntimeError(f"cpu{cpu}: malformed thread_siblings_list ({sib})")
        key = sib[0]
        if key in declared:
            if declared[key] != sib:
                raise RuntimeError(f"cpu{cpu} declares siblings {sib} but cpu{declared[key][0]} "
                                   f"in the same group declares {declared[key]}")
        else:
            declared[key] = sib
            groups[key] = []
        outside = [s for s in sib if s not in cpu_set]
        if outside:
            raise RuntimeError(f"cpu{cpu}: siblings {sib} reach outside node{node}'s cpulist")
        groups[key].append(cpu)
    return groups
