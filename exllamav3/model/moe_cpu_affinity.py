"""
Host-side CPU placement for the CPU MoE offload path.

The worker pool (cpu/moe_mul1.cpp Pool) runs one compute thread per physical core, pinned by
default in physical_core_order(): one LP per core first, then every SMT sibling. The parent
process (the thread driving the forward pass, CUDA's driver threads, an API server's executor
threads) is not pinned, so the scheduler can drop it on a worker's LP, where the worker becomes
the straggler at every per-phase barrier of the job. This module plans and applies the host
placement, with the workers pinned or floating alike (the pool reports its topology either way).
"""
import functools
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


# --- NUMA topology and placement (Linux) -------------------------------------------
#
# Used by the multi-NUMA CPU MoE offload (moe_cpu_numa_nodes): one worker process per NUMA node,
# its pool pinned to the node's cores and its memory bound there. A single worker uses none of it. Everything here reads sysfs and fails soft: an empty/None result means "no NUMA information".


def _cpulist(text: str) -> list[int]:
    """Parse a sysfs CPU list ("0-31,128-159")"""
    out = []
    for part in text.strip().split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


@functools.lru_cache(maxsize = None)
def numa_topology() -> dict[int, list[int]]:
    """NUMA node -> sorted LP list, from /sys/devices/system/node/nodeN/cpulist; nodes without
    CPUs left out. {} when sysfs exposes no nodes (non-NUMA kernel, restricted container)."""
    if sys.platform == "win32":
        return {}
    nodes: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/sys/devices/system/node")
    except OSError:
        return {}
    for node in entries:
        if not node.startswith("node"):
            continue
        try:
            n = int(node[4:])
            with open(f"/sys/devices/system/node/{node}/cpulist") as f:
                cpus = _cpulist(f.read())
        except (OSError, ValueError):
            continue
        if cpus:
            nodes[n] = sorted(cpus)
    return nodes


def pci_bdf_numa_node(bdf: str) -> int | None:
    """NUMA node of one PCI device by BDF ("0000:41:00.0" as cuDeviceGetPCIBusId reports it).
    None when the device or its node is unknown."""
    path = f"/sys/bus/pci/devices/{bdf.lower().strip()}/numa_node"
    try:
        with open(path) as f:
            n = int(f.read().strip())
        return n if n >= 0 else None
    except (OSError, ValueError):
        return None


def node_available_memory_kb(node: int) -> int | None:
    """Memory (kB) an allocation bound to one NUMA node can still get: MemFree plus the node's
    page cache and reclaimable slab, which a bound allocation reclaims (a model whose checkpoint
    was just read typically has much of it in page cache). Shared memory (the pinned arena's
    memfd pages among it) is not reclaimable and not counted. None when unknown."""
    fields = {}
    try:
        with open(f"/sys/devices/system/node/node{node}/meminfo") as f:
            for line in f:
                # "Node 0 MemFree:   144189540 kB"
                parts = line.split()
                if len(parts) == 5 and parts[4] == "kB":
                    fields[parts[2].rstrip(":")] = int(parts[3])
    except (OSError, ValueError, IndexError):
        return None
    if "MemFree" not in fields:
        return None
    cache = max(0, fields.get("FilePages", 0) - fields.get("Shmem", 0))
    slab = fields.get("KReclaimable", fields.get("SReclaimable", 0))
    return fields["MemFree"] + cache + slab


def check_node_memory(nbytes: int, node: int | None, what: str) -> None:
    """Fail before an allocation bound to NUMA node `node` that cannot fit there (no-op for None).

    check_host_memory only sees the machine-wide MemAvailable, and a bound allocation does not
    fall back to another node: the kernel OOM killer steps in instead. Keeps the same
    EXL3_HOST_MEM_RESERVE_MB reserve, and fails soft when the node's counters are unavailable.
    """
    if node is None:
        return
    avail_kb = node_available_memory_kb(node)
    reserve_mb = int(os.environ.get("EXL3_HOST_MEM_RESERVE_MB", 2048))
    if avail_kb is None or reserve_mb <= 0:
        return
    if nbytes + (reserve_mb << 20) > avail_kb * 1024:
        raise RuntimeError(
            f"{what} needs {nbytes >> 20} MiB on NUMA node {node}, but only {avail_kb >> 10} MiB "
            f"is available there and EXL3_HOST_MEM_RESERVE_MB = {reserve_mb} MiB is kept free. "
            f"Free memory on that node, offload fewer experts, or lower EXL3_HOST_MEM_RESERVE_MB.")


def parse_numa_nodes(value) -> str | list[int] | None:
    """A moe_cpu_numa_nodes value: None or "" (a single worker), "all", or node IDs (a list, or
    comma-separated)."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = str(value).strip().lower()
        if text in ("", "all"):
            return text or None
        items = text.split(",")
    try:
        nodes = [int(str(n).strip()) for n in items]
    except ValueError:
        raise ValueError(f"moe_cpu_numa_nodes must be 'all' or NUMA node IDs (got {value!r})") from None
    if len(set(nodes)) != len(nodes):
        raise ValueError(f"moe_cpu_numa_nodes lists a node twice (got {value!r})")
    return nodes


# set_mempolicy(2) syscall number (kernel ABI, like the memfd table in moe_cpu_host.py): x86_64 has
# its own numbering; everything asm-generic (arm64, riscv64, loongarch64) shares one table
_SET_MEMPOLICY_SYSCALL = {"x86_64": 238, "aarch64": 237, "riscv64": 237, "loongarch64": 237}
MPOL_BIND = 2


def bind_memory_to_node(node: int) -> bool:
    """Bind the calling thread's memory, and that of every thread it starts afterwards, to NUMA
    node `node` (set_mempolicy MPOL_BIND): pages fault in on the node, and a full node reclaims
    its page cache instead of spilling onto another. Covers tmpfs pages too (the pinned arena's
    memfd chunks), which a mapping's mbind would not. False when unavailable or refused."""
    nr = _SET_MEMPOLICY_SYSCALL.get(os.uname().machine) if hasattr(os, "uname") else None
    if nr is None or not 0 <= node < 1024:
        return False
    import ctypes
    mask = (ctypes.c_uint64 * 16)()   # 1024 nodes
    mask[node // 64] = 1 << (node % 64)
    try:
        libc = ctypes.CDLL(None, use_errno = True)
        # maxnode is one past the mask's bits (the kernel drops the last)
        return libc.syscall(ctypes.c_long(nr), ctypes.c_int(MPOL_BIND), ctypes.byref(mask),
                            ctypes.c_ulong(1025)) == 0
    except (OSError, AttributeError):
        return False
