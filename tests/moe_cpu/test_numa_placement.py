"""
NUMA helpers for the multi-NUMA CPU MoE offload (moe_cpu_affinity.py): the moe_cpu_numa_nodes
grammar, the per-node memory check, and (on a NUMA machine) that memory bound with
bind_memory_to_node lands on the node: a private arena chunk, and a pinned-arena memfd chunk
(posix_fallocate) from a thread started after the bind.
"""

import subprocess
import sys
import textwrap

import pytest

import exllamav3.model.moe_cpu_affinity as aff

pytestmark = pytest.mark.nogpu

LINUX = sys.platform.startswith("linux")
NUMA = LINUX and len(aff.numa_topology()) > 1


def test_parse_numa_nodes():
    assert aff.parse_numa_nodes(None) is None
    assert aff.parse_numa_nodes(" ") is None
    assert aff.parse_numa_nodes(" ALL ") == "all"
    assert aff.parse_numa_nodes("2, 0") == [2, 0]
    assert aff.parse_numa_nodes([1, 3]) == [1, 3]
    for bad in ("0,x", "auto", "0,0", [1, 1]):
        with pytest.raises(ValueError):
            aff.parse_numa_nodes(bad)


def test_check_node_memory(monkeypatch):
    monkeypatch.setenv("EXL3_HOST_MEM_RESERVE_MB", "1")
    monkeypatch.setattr(aff, "node_available_memory_kb", lambda n: 4096)
    aff.check_node_memory(2 << 20, 0, "test")
    with pytest.raises(RuntimeError, match = "NUMA node 0"):
        aff.check_node_memory(4 << 20, 0, "test")
    aff.check_node_memory(1 << 40, None, "test")          # not bound: nothing to check
    monkeypatch.setattr(aff, "node_available_memory_kb", lambda n: None)
    aff.check_node_memory(1 << 40, 0, "test")             # no counters: soft


_BOUND_CHILD = textwrap.dedent("""
    import ctypes, importlib.util, mmap, os, re, sys, threading
    spec = importlib.util.spec_from_file_location("aff", sys.argv[1])
    aff = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aff)
    node = int(sys.argv[2])
    assert aff.bind_memory_to_node(node)

    def nodes_of(m):
        view = ctypes.c_char.from_buffer(m)
        addr = ctypes.addressof(view)
        del view
        with open("/proc/self/numa_maps") as f:
            line = next(l for l in f if l.startswith(f"{addr:x} "))
        return {int(n) for n in re.findall(r" N(\\d+)=", line)}

    private = mmap.mmap(-1, 16 << 20, mmap.MAP_PRIVATE, mmap.PROT_READ | mmap.PROT_WRITE)
    # A VMA of its own: an adjacent anonymous mapping would merge into it, and numa_maps would
    # count that one's pages (faulted before the bind) too
    private.madvise(mmap.MADV_DONTFORK)
    private[:] = b"\\x01" * len(private)
    assert nodes_of(private) == {node}, nodes_of(private)

    got = []
    def memfd_chunk():
        fd = os.memfd_create("exl3_test_bind")
        os.posix_fallocate(fd, 0, 8 << 20)
        m = mmap.mmap(fd, 8 << 20, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(fd)
        m[:] = b"\\x01" * len(m)   # map the preallocated pages in, for numa_maps to count them
        got.append(nodes_of(m))
    t = threading.Thread(target = memfd_chunk)
    t.start()
    t.join()
    assert got == [{node}], got
""")


@pytest.mark.skipif(not NUMA, reason = "needs Linux NUMA topology")
def test_bound_memory_lands_on_the_node():
    # In a child process: the binding is process-lasting
    topo = aff.numa_topology()
    avail = {n: aff.node_available_memory_kb(n) or 0 for n in topo}
    node = max(avail, key = avail.get)
    if avail[node] < (64 << 10):
        pytest.skip("no node with enough memory")
    subprocess.run([sys.executable, "-c", _BOUND_CHILD, aff.__file__, str(node)], check = True)
