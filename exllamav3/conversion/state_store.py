"""
Tiered storage for measurement state snapshots: VRAM first, system RAM second, swap only
as a genuine last resort — with all budgets autodetected so the user never has to size
anything.

Motivation (measured on gpt-oss-120b, 36L/128-expert MoE, 96 GB GPU / 123 GB RAM):

  * measure_model.py snapshots one full calibration state per (tensor-group, candidate)
    per pass. These snapshots were stored via ``.cpu()``, so a pass's working set lived
    entirely in system RAM while tens of GB of VRAM sat idle (the GPU only holds one
    layer's modules at a time during measurement).
  * The old ``-ms`` chunking formula under-budgets the working set by a factor of
    ``num_candidates`` (states are stored per candidate, not per target), which makes
    OOM-by-formula easy: peak = targets_in_pass x num_candidates x state_size.
  * The final module is special: every candidate forward materializes a
    ``[rows*cols, vocab]`` logits tensor in VRAM regardless of where snapshots live
    (~25 GB at cal 32x2048 over a 200k vocab). VRAM reserved for snapshots must leave
    room for it, or the run dies at the finish line.

The store keeps snapshots in VRAM while ``free - size > reserve`` holds (checked live,
per store, so module loads and transients are automatically respected), spilling to
pageable system RAM otherwise. If RAM is exhausted the OS pages to swap — degraded but
alive, and only reachable after both real tiers are full.

Correctness notes:

  * ``store()`` CLONES device tensors. ``.cpu()`` used to provide snapshot semantics as
    a side effect of the device move; a zero-copy VRAM "store" aliases the live residual
    buffer, which module.forward() mutates in place — every snapshot silently becomes
    the same trampled tensor.
  * ``load()`` returns a PRIVATE device tensor. ``prepare_for_device()`` fast-paths
    same-device inputs with no copy, so a VRAM-resident snapshot fed straight to
    ``forward()`` would be corrupted for its next read (the base state is re-read once
    per target). CPU-resident snapshots get their copy for free via ``.to(device)``.

Run ``python -m exllamav3.conversion.state_store`` for a self-test of both properties.
"""

import os
import torch


def ram_available_bytes() -> int:
    """Best-effort available system RAM (reclaimable-aware on Linux)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return 16 << 30  # conservative fallback; store() degrades gracefully anyway


def auto_kldiv_batch(vocab_size: int, budget_elements: int = 32 << 20) -> int:
    """
    Softmax chunk rows for the final KL-div step, sized so a single kernel stays small
    enough for display-watchdog systems (a 512 x 200k fp32 softmax is one long kernel
    launch and gets killed on desktops). Scales with vocab: ~128 rows at 200k vocab,
    512 at 32k.
    """
    return max(16, min(512, budget_elements // max(1, vocab_size)))


class TieredStateStore:

    def __init__(self, device: torch.device, reserve_bytes: int, ram_floor_bytes: int = 4 << 30):
        """
        :param device:
            CUDA device snapshots may reside on.
        :param reserve_bytes:
            VRAM to keep free at all times (module loads + forward activations + the
            final-module logits transient). Snapshots never intrude on it.
        :param ram_floor_bytes:
            When available RAM falls below this, warn once that the OS is about to
            page (the store still proceeds — swap is the last resort, not an error).
        """
        self.device = device
        self.reserve = reserve_bytes
        self.ram_floor = ram_floor_bytes
        self.kept_vram = 0
        self.spilled_ram = 0
        self._warned_swap = False

    def store(self, t: torch.Tensor) -> torch.Tensor:
        """Snapshot a tensor: private copy in VRAM if the reserve allows, else RAM."""
        if t.device.type == "cuda":
            nbytes = t.numel() * t.element_size()
            free, _ = torch.cuda.mem_get_info(t.device)
            if free - nbytes > self.reserve:
                self.kept_vram += 1
                return t.clone()  # MUST copy: forward() mutates its input in place
            self.spilled_ram += 1
            if not self._warned_swap and ram_available_bytes() < self.ram_floor:
                self._warned_swap = True
                print(" !! System RAM nearly exhausted; the OS may start paging state "
                      "data to swap (run continues, but slowly)")
        return t.cpu()

    def load(self, t: torch.Tensor) -> torch.Tensor:
        """
        Materialize a snapshot on-device as a PRIVATE tensor safe to hand to a forward
        pass that mutates its input. CPU tensors are copied by the transfer; device
        tensors must be cloned explicitly.
        """
        if t.device == self.device:
            return t.clone()
        return t  # prepare_for_device() copies host tensors on upload

    def stats(self) -> str:
        return f"{self.kept_vram} state snapshots in VRAM, {self.spilled_ram} spilled to RAM"


def _selftest():
    if not torch.cuda.is_available():
        print("CUDA unavailable; self-test skipped")
        return
    dev = torch.device("cuda:0")
    store = TieredStateStore(dev, reserve_bytes = 2 << 30)
    buf = torch.zeros(64 << 20, dtype = torch.half, device = dev)   # 128 MB
    snap = store.store(buf)
    assert snap.data_ptr() != buf.data_ptr(), "snapshot aliases live buffer"
    buf += 1
    assert snap.abs().sum().item() == 0, "snapshot trampled by in-place mutation"
    loaded = store.load(snap)
    assert loaded.data_ptr() != snap.data_ptr(), "load() returned the stored snapshot itself"
    loaded += 7
    assert snap.abs().sum().item() == 0, "stored snapshot corrupted through load()"
    tiny = TieredStateStore(dev, reserve_bytes = 1 << 60)           # impossible reserve
    spilled = tiny.store(buf)
    assert spilled.device.type == "cpu", "reserve floor not respected"
    print("state_store self-test passed:", store.stats())


if __name__ == "__main__":
    _selftest()
