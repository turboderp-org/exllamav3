"""
Multi-NUMA CPU MoE (moe_cpu_numa_nodes): resolving the node list, the worker plan (one worker per
node, pinned along the pool's core order restricted to the node; the host's cores on the CUDA
device's node), the plan on a reload in a process whose host is already confined, and the
per-worker spawn arguments and thread counts. Synthetic topologies, no GPU, no model, no
processes.
"""

import pytest

import exllamav3.model.moe_cpu_host as mch

pytestmark = pytest.mark.nogpu


# 4 nodes x 4 cores, SMT siblings at +16: node n has primaries 4n..4n+3 and siblings 16+4n..; the
# pool's core order (exl3_moe_cpu_core_order) lists the 16 primaries, then their siblings
TOPO = {n: list(range(4 * n, 4 * n + 4)) + list(range(16 + 4 * n, 16 + 4 * n + 4)) for n in range(4)}


@pytest.fixture
def machine(monkeypatch):
    """TOPO with the CUDA device on `gpu_node` (None: unknown); returns a host factory"""
    monkeypatch.setattr(mch.ext, "exl3_moe_cpu_core_order", lambda: (list(range(32)), 16))
    monkeypatch.setattr(mch, "numa_topology", lambda: TOPO)
    monkeypatch.setattr(mch, "process_cpus", lambda: list(range(32)))
    monkeypatch.setattr(mch, "_HOST_ORIG_CPUS", None)
    monkeypatch.setattr(mch.TUNING, "host_cores", 1)
    monkeypatch.setattr(mch.torch.cuda, "current_device", lambda: 0)

    def make(nodes, gpu_node = 0, component = "text"):
        class Props:
            pci_domain_id = pci_bus_id = pci_device_id = 0
        monkeypatch.setattr(mch.torch.cuda, "get_device_properties", lambda d: Props())
        monkeypatch.setattr(mch, "pci_bdf_numa_node", lambda bdf: gpu_node)
        host = object.__new__(mch.MoeCpuHost)
        host.component = component
        host.numa_nodes = host._resolve_numa_nodes(nodes)
        host.workers = len(host.numa_nodes) if host.numa_nodes else 1
        host._xw = [mch._ExtraWorker() for _ in range(1, host.workers)]
        return host
    return make


def test_unset_and_other_components_never_touch_the_topology(monkeypatch):
    def boom():
        raise AssertionError("topology read for a single worker")
    monkeypatch.setattr(mch, "numa_topology", boom)
    host = object.__new__(mch.MoeCpuHost)
    host.component = "text"
    assert host._resolve_numa_nodes(None) is None
    assert host._resolve_numa_nodes("") is None
    host.component = "mtp"
    assert host._resolve_numa_nodes("all") is None


def test_all_is_every_usable_node(machine, monkeypatch, capsys):
    assert machine("all").numa_nodes == [0, 1, 2, 3]
    monkeypatch.setattr(mch, "process_cpus", lambda: TOPO[2])          # e.g. numactl --cpunodebind=2
    assert machine("all").numa_nodes is None
    assert "using one worker" in capsys.readouterr().out


def test_listed_nodes_are_validated(machine, monkeypatch):
    assert machine("2,0").numa_nodes == [2, 0]
    assert machine([1, 3]).numa_nodes == [1, 3]
    for bad in ("0,7", "1", "0,0", "0,x"):
        with pytest.raises(ValueError):
            machine(bad)
    monkeypatch.setattr(mch, "MOE_MAX_WORKERS", 3)
    with pytest.raises(ValueError, match = "at most 3"):
        machine("all")


def test_usable_nodes_come_from_the_pre_pin_mask(machine, monkeypatch):
    # A second load must not see only the confined host's LPs
    monkeypatch.setattr(mch, "_HOST_ORIG_CPUS", list(range(32)))
    monkeypatch.setattr(mch, "process_cpus", lambda: [3, 19])
    assert machine("all").numa_nodes == [0, 1, 2, 3]


def test_plan_one_worker_per_node_host_on_the_gpu_node(machine):
    host = machine("all", gpu_node = 2)
    plan = host._worker_plan()
    assert host.host_cpus == [11, 27]                     # node 2's last core, both siblings
    assert plan[2] == ([8, 9, 10, 24, 25, 26], 3)
    assert [plan[n] for n in (0, 1, 3)] == [(TOPO[n], 4) for n in (0, 1, 3)]


def test_plan_host_off_the_listed_nodes(machine):
    host = machine("1,3", gpu_node = 0)
    assert host._worker_plan() == [(TOPO[1], 4), (TOPO[3], 4)]
    assert host.host_cpus == [3, 19]


def test_plan_without_gpu_node_reserves_on_the_first_worker(machine):
    host = machine("1,3", gpu_node = None)
    assert host._worker_plan()[0] == ([4, 5, 6, 20, 21, 22], 3)
    assert host.host_cpus == [7, 23]


def test_plan_on_reload_keeps_the_placed_host_out(machine, monkeypatch):
    monkeypatch.setattr(mch, "_HOST_ORIG_CPUS", list(range(32)))
    monkeypatch.setattr(mch, "process_cpus", lambda: [3, 19])
    host = machine("all")
    plan = host._worker_plan()
    assert host.host_cpus is None
    assert plan[0] == ([0, 1, 2, 16, 17, 18], 3)


class _Fakes:
    def __init__(self, monkeypatch):
        self.spawned = []
        fakes = self

        class FakeConn:
            def close(self): pass
            def send(self, msg): pass

        class FakeProcess:
            def __init__(self, target, args, daemon):
                fakes.spawned.append(args)
            def start(self): pass

        class FakeCtx:
            Process = FakeProcess
            @staticmethod
            def Pipe(duplex): return FakeConn(), FakeConn()

        monkeypatch.setattr(mch.multiprocessing, "get_context", lambda method: FakeCtx())
        monkeypatch.setattr(mch.cleanupper, "register_atexit", lambda fn: None)


def _spawn(host, threads = 12, explicit = False):
    host.proc, host.model_dir, host.threads, host.stage_threads, host.pinned = None, "m", threads, 1, False
    host.threads_explicit = explicit
    host._spawn()


def test_spawn_single_worker_is_unchanged(machine, monkeypatch):
    fk = _Fakes(monkeypatch)
    _spawn(machine(None))
    assert [a[1:] for a in fk.spawned] == [("m", 12, 1, False, "", None, False)]


def test_spawn_one_process_per_node(machine, monkeypatch):
    fk = _Fakes(monkeypatch)
    host = machine("all")
    _spawn(host)
    assert host.worker_threads == [3, 4, 4, 4]           # by default each node's physical cores
    for w, args in enumerate(fk.spawned):
        assert args[2] == host.worker_threads[w]
        # The process mask as a single worker's (here: inherited), the pool's pins per node
        assert args[6:] == (None, False, w, w, 4, host._worker_plan()[w][0])
    assert all(st.proc is not None and st._send_queue is not None for st in host._states())


def test_explicit_thread_count_applies_to_each_worker(machine, monkeypatch):
    # As for a single worker: past the physical cores the pins continue onto their SMT siblings
    _Fakes(monkeypatch)
    host = machine("all")
    _spawn(host, threads = 6, explicit = True)
    assert host.worker_threads == [6, 6, 6, 6]
