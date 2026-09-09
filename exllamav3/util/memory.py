from dataclasses import dataclass
from collections import deque
import torch
import gc
import sys
from pydantic import PydanticUserError

# @lru_cache
# def init_pynvml():
#     pynvml.nvmlInit()

# Try to make sure device is live for correct measurement of free VRAM
def touch_device(device: int):
    d = torch.empty((32, 32), device = device, dtype = torch.float)
    d = d @ d
    d = d + d


# Touch device and measure VRAM (child process)
def touch_device_measure_vram(local_context: dict):
    device = local_context["device"]
    touch_device(device)
    return torch.cuda.mem_get_info(device)


# Reserve byte amount on device
def set_memory_fraction_reserve(
    reserve: int,
    device: int
):
    touch_device(device)
    free, total = torch.cuda.mem_get_info(device)
    # mem_get_info reports memory free *after* whatever this process has already reserved, but
    # set_per_process_memory_fraction limits the process's *cumulative* reserved bytes. Add the
    # current reservation back, or memory held by an earlier load in the same process (a draft
    # model, a vision tower) is subtracted from the budget twice.
    current = torch.cuda.memory_reserved(device)
    fraction = (current + free - reserve) / total
    fraction = min(1.0, max(0.01, fraction))
    torch.cuda.set_per_process_memory_fraction(fraction, device = device)


# Allow byte amount to be used on device, PER LOAD: the budget is headroom for the load
# being planned, on top of whatever this process already holds. set_per_process_memory_fraction
# caps the process's CUMULATIVE reserved bytes, so the current reservation is added back —
# otherwise a second component model (draft, vision tower) loaded with use_per_device would
# have its budget silently reduced by the first component's footprint (same double-count the
# reserve variant above corrects)
def set_memory_fraction_use(
    use: int,
    device: int
):
    touch_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    current = torch.cuda.memory_reserved(device)
    fraction = min((current + use) / total, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, device = device)


# Un-reserve VRAM
def unset_memory_fraction(active_devices: list[int]):
    for i in active_devices:
        torch.cuda.set_per_process_memory_fraction(1.0, device = i)


# Free unused VRAM
def free_mem():
    gc.collect()
    torch.cuda.empty_cache()


# Host-allocation churn (recurrent checkpoint stashes, per-layer fp32 copies during
# conversion, ...) leaves freed memory stranded in glibc's arenas: interleaved lifetimes
# fragment the heap and RSS ratchets up even though nothing is referenced. malloc_trim
# returns what can be returned
_libc = None

def malloc_trim():
    global _libc
    if _libc is False:
        return
    try:
        if _libc is None:
            import ctypes
            _libc = ctypes.CDLL("libc.so.6")
        _libc.malloc_trim(0)
    except Exception:
        _libc = False


def list_gpu_tensors(min_size: int = 1, cuda_only: bool = True):
    """
    Search the current process for referenced CUDA tensors and list them.

    :param min_size:
        Ignore tensors smaller than this size, in megabytes

    :param cuda_only:
        Only list CUDA tensors
    """

    import threading
    import warnings
    from tabulate import tabulate

    # Suppress FutureWarning from Torch every time we try to access certain objects
    warnings.simplefilter(action = 'ignore', category = FutureWarning)

    @dataclass
    class Result:
        paths: list[str]
        shape: tuple
        dtype: torch.dtype
        device: str
        size: int

    results = {}
    visited = set()

    # Helper function to filter and collect items
    def collect(path, item):
        nonlocal results

        # Only collect CUDA tensors
        if not isinstance(item, torch.Tensor) or (cuda_only and not item.is_cuda):
            return

        # Tensor size in MB, filter anything smaller than the minimum size
        size = item.nelement() * item.element_size() // (1024**2)
        if size < min_size:
            return

        # Skip tensors in paths containing specific debug substrings
        if any(x in path for x in [
            ".stderr.dbg.",
            "dbg.value_resolve_thread_list",
            "global_vars[",
            "local_vars[",
            "updated_globals[",
        ]):
            return

        # Adjust the path display for objects defined in __main__
        if ".__main__." in path:
            path = path[path.find(".__main__.") + 10:]

        # If tensor is already recorded, just record the additional path
        obj_id = id(item)
        if obj_id in results and path not in results[obj_id].paths:
            results[obj_id].paths.append(path)
        else:
            results[obj_id] = Result(
                paths = [path],
                shape = item.shape,
                dtype = item.dtype,
                device = str(item.device),
                size = size
            )

    # Queue of items to scan recursively
    queue = deque()

    # Collect items that are global variables, and add to the queue
    for name, obj in globals().items():
        collect(name, obj)
        queue.append((name, obj))

    # Traverse each thread's frame stack, collecting items and queueing items
    for thread_id, frame in sys._current_frames().items():
        prefix = ""

        # Skip the current frame for the current thread to avoid recursion issues
        if thread_id == threading.get_ident():
            frame = frame.f_back

        # Collect/queue each local variable in the frame, extend the relative path prefix
        # and walk the stack
        while frame:
            for name, obj in frame.f_locals.items():
                # We actually start three levels deep but want variables in the "current" frame
                # (i.e. the frame of the function calling list_gpu_tensors) to have a prefix of "."
                new_path = f"{prefix[2:]}.{name}"
                collect(new_path, obj)
                queue.append((name, obj))
            frame = frame.f_back
            prefix += "."

    # Process the queue by examining attributes, dictionary entries, and sequence items
    while queue:
        path, obj = queue.popleft()

        # Iterate over entries in object with __dict__ attribute
        try:
            if hasattr(obj, '__dict__'):
                for attr, value in obj.__dict__.items():
                    new_path = f"{path}.{attr}"
                    collect(new_path, value)
                    if id(value) not in visited:
                        visited.add(id(value))
                        queue.append((new_path, value))
        except PydanticUserError:
            pass

        # If object is a dictionary, iterate through all its items
        if isinstance(obj, dict):
            try:
                for key, value in obj.items():
                    new_path = f"{path}['{key}']"
                    collect(new_path, value)
                    if id(value) not in visited:
                        visited.add(id(value))
                        queue.append((new_path, value))
            except:
                pass

        # Same for list, tuple, set
        if isinstance(obj, (list, tuple, set)):
            for idx, item in enumerate(obj):
                new_path = f"{path}[{idx}]"
                collect(new_path, item)
                if id(item) not in visited:
                    visited.add(id(item))
                    queue.append((new_path, item))

    # Sort tensors by descending size
    items = list(results.values())
    items.sort(key = lambda x: -x.size)

    # Build output table, grouped by device
    devices: dict[str, list] = {}
    for v in items:
        if v.device not in devices:
            devices[v.device] = []
        dev = devices[v.device]
        dev.append([
            v.size,
            v.paths[0],
            tuple(v.shape),
            str(v.dtype).replace("torch.", "")
        ])
        for p in v.paths[1:]:
            dev.append([
                None,
                " + " + p,
                None,
                None
            ])

    # Print tables to console
    for k in sorted(devices.keys()):
        print()
        print(f"--------------")
        print(f"| {k:10} |")
        print(f"--------------")
        print()
        headers = ["size // MB", "path", "shape", "dtype"]
        print(tabulate(devices[k], headers = headers, tablefmt = "github", intfmt=","))


# ---- VRAM accounting -------------------------------------------------------------------------
#
# Attribute every live CUDA storage of a loaded model to an owner and reconcile against the
# caching allocator's counters and the driver's view of the device. Owners:
#
#   weights (arena)   loader slab blocks (128 MiB), split into used bytes and the unused tail
#   weights (direct)  weight tensors with their own allocation (> arena cap, or loaded outside
#                     a deferred-load bracket)
#   cache             K/V, latent or pool cache layers
#   recurrent         per-slot recurrent states (GDN/KDA/SWA rings)
#   statics           g_tensor_cache entries (graph statics, decode workspaces), by tag
#   generator         tensors reachable from the Generator that no module/cache owns
#   other             live storages nobody claims (transients alive at snapshot time, ...)
#
# Allocator: allocated (live), inactive-split (free space stranded inside segments; zero with
# expandable segments), cached-free (whole free blocks kept for reuse), reserved. Non-torch is
# the device's used memory beyond torch's reservation: CUDA context, cuBLAS/Triton modules,
# other libraries' allocations.

from dataclasses import field
import collections

ARENA_BLOCK_BYTES = 128 << 20


@dataclass
class VRAMDeviceReport:
    device: str
    device_name: str
    weights_arena_used: int = 0
    weights_arena_tail: int = 0
    arena_blocks: int = 0
    weights_direct: int = 0
    cache: int = 0
    recurrent: int = 0
    statics: int = 0
    statics_by_tag: dict = field(default_factory = dict)
    generator: int = 0
    other: int = 0
    other_examples: list = field(default_factory = list)   # (shape, dtype, bytes, path)
    live_total: int = 0
    allocated: int = 0
    reserved: int = 0
    inactive_split: int = 0
    cached_free: int = 0
    segments: int = 0
    peak_allocated: int = 0
    peak_reserved: int = 0
    non_torch: int = 0
    device_used: int = 0
    device_total: int = 0

    @property
    def weights(self) -> int:
        return self.weights_arena_used + self.weights_arena_tail + self.weights_direct

    def as_dict(self) -> dict:
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.__dict__.items()} | {"weights": self.weights}


def _walk_tensors(obj, out: list, seen: set, paths: dict, path: str = ""):
    """Collect tensors reachable from obj through attributes and containers, cycle-protected,
    recording the attribute path of the first sighting of each storage."""
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        out.append(obj)
        if obj.is_cuda:
            paths.setdefault(_storage_key(obj), path)
        return
    if obj is None or isinstance(obj, (str, bytes, int, float, bool, type)):
        return
    if isinstance(obj, (list, tuple, set, deque)):
        for i, x in enumerate(obj):
            _walk_tensors(x, out, seen, paths, f"{path}[{i}]")
    elif isinstance(obj, dict):
        for k, x in obj.items():
            _walk_tensors(x, out, seen, paths, f"{path}[{k!r}]")
    elif hasattr(obj, "__dict__"):
        for k, x in vars(obj).items():
            _walk_tensors(x, out, seen, paths, f"{path}.{k}" if path else f"{type(obj).__name__}.{k}")


def _storage_key(t: torch.Tensor):
    s = t.untyped_storage()
    return (s.data_ptr(), s.nbytes())


def vram_accounting(model, cache = None, generator = None, devices = None) -> list[VRAMDeviceReport]:
    """
    Full per-device VRAM accounting for a loaded model (see the module comment for the
    categories). Walks gc for live CUDA tensors, so it costs a few hundred ms on a large MoE
    model; call it between forwards, not inside one. `devices` defaults to every device the
    process has torch allocations on.
    """
    from .tensor import g_tensor_cache
    torch.cuda.synchronize()
    gc.collect()
    if devices is None:
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
                   if torch.cuda.memory_stats(i).get("allocated_bytes.all.current", 0) > 0]
    devices = [torch.device(d) for d in devices]
    dev_set = set(devices)

    # type(o) reads the type slot; isinstance() would read o.__class__ on every tracked object,
    # and some instrumented shims (torch.distributed.reduce_op) warn on any attribute access
    all_t = [o for o in gc.get_objects() if issubclass(type(o), torch.Tensor) and o.is_cuda and o.device in dev_set]

    # Owner marks per storage
    owners: dict = {}
    def mark(ts, label):
        for t in ts:
            if isinstance(t, torch.Tensor) and t.is_cuda and t.device in dev_set:
                owners.setdefault(_storage_key(t), set()).add(label)

    paths: dict = {}
    weight_t: list = []
    seen = {id(model.config), id(getattr(model.config, "stc", None)), id(cache), id(generator)}
    for m in model.modules:
        for sub in m:
            _walk_tensors(sub, weight_t, seen, paths)
    mark(weight_t, "weights")
    if cache is not None:
        for l in cache.layers.values():
            mark(l.get_tensors(), "cache")
        from ..modules import Module
        for rl in cache.recurrent_layers.values():
            ts = []
            if hasattr(rl, "get_state_tensors"):
                try: ts += list(rl.get_state_tensors())
                except Exception: pass
            # State objects point back at their module; stop the walk at module boundaries so
            # weights are not relabelled as state (recurrent outranks weights below)
            stop = {id(v) for v in vars(rl).values() if isinstance(v, Module)} | {id(model), id(cache)}
            _walk_tensors(rl, ts, stop, {})
            mark(ts, "recurrent")
    tag_of = {}
    for key, (refc, v) in g_tensor_cache.cache.items():
        if v.is_cuda and v.device in dev_set:
            mark([v], "statics"); tag_of[_storage_key(v)] = key.split("/")[-1]
    if generator is not None:
        ts = []
        _walk_tensors(generator, ts, {id(model), id(cache)}, {})
        mark(ts, "generator")

    arena_keys = {_storage_key(t) for t in all_t
                  if t.dtype == torch.uint8 and t.dim() == 1 and t.numel() == ARENA_BLOCK_BYTES}

    reports = []
    for dev in devices:
        r = VRAMDeviceReport(device = str(dev), device_name = torch.cuda.get_device_name(dev))
        keys = {_storage_key(t) for t in all_t if t.device == dev}
        arena_here = {k for k in keys if k in arena_keys}
        used_views = set()
        for t in weight_t:
            if t.is_cuda and t.device == dev and _storage_key(t) in arena_here:
                vk = (t.data_ptr(), t.numel() * t.element_size())
                if vk not in used_views:
                    used_views.add(vk); r.weights_arena_used += vk[1]
        r.arena_blocks = len(arena_here)
        r.weights_arena_tail = sum(k[1] for k in arena_here) - r.weights_arena_used
        by_tag = collections.Counter()
        other_keys = []
        for k in keys:
            nb = k[1]
            if k in arena_here:
                continue
            labs = owners.get(k, set())
            # explicit owners win: modules also reference their cache layers / states
            for lab in ("cache", "recurrent", "statics", "generator", "weights"):
                if lab in labs:
                    break
            else:
                other_keys.append(k); r.other += nb; continue
            if lab == "weights": r.weights_direct += nb
            elif lab == "cache": r.cache += nb
            elif lab == "recurrent": r.recurrent += nb
            elif lab == "generator": r.generator += nb
            else:
                r.statics += nb; by_tag[tag_of.get(k, "?")] += nb
        r.statics_by_tag = dict(by_tag.most_common())
        shapes = {}
        for t in all_t:
            if t.device == dev:
                k = _storage_key(t)
                if k in other_keys and k not in shapes:
                    shapes[k] = (tuple(t.shape), str(t.dtype).replace("torch.", ""))
        for k in sorted(other_keys, key = lambda k: -k[1])[:8]:
            sh, dt = shapes.get(k, ("?", "?"))
            r.other_examples.append((sh, dt, k[1], paths.get(k, "")))
        r.live_total = sum(k[1] for k in keys)
        ms = torch.cuda.memory_stats(dev)
        r.allocated = ms.get("allocated_bytes.all.current", 0)
        r.reserved = ms.get("reserved_bytes.all.current", 0)
        r.inactive_split = ms.get("inactive_split_bytes.all.current", 0)
        r.cached_free = max(r.reserved - r.allocated - r.inactive_split, 0)
        r.segments = ms.get("segment.all.current", 0)
        r.peak_allocated = ms.get("allocated_bytes.all.peak", 0)
        r.peak_reserved = ms.get("reserved_bytes.all.peak", 0)
        free, total = torch.cuda.mem_get_info(dev)
        r.device_total = total
        r.device_used = total - free
        r.non_torch = max(r.device_used - r.reserved, 0)
        reports.append(r)
    return reports


_ANSI = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m", "magenta": "\033[35m",
    "blue": "\033[34m", "red": "\033[31m", "white": "\033[37m",
}


def format_vram_report(reports: list[VRAMDeviceReport], color: bool = True, unit: str = "GiB",
                       totals: bool = True, previous: list[VRAMDeviceReport] | None = None) -> str:
    """Render accounting reports as aligned tables (ANSI colors optional). With `previous` (an
    earlier vram_accounting() result) every row gains a delta column against the report for
    the same device: red for growth, green for shrinkage, blank when unchanged (< 1 MiB)."""
    div = {"GiB": 1024 ** 3, "MiB": 1024 ** 2, "GB": 1e9, "MB": 1e6}[unit]
    def c(name, s):
        return f"{_ANSI[name]}{s}{_ANSI['reset']}" if color else s
    def q(nb, w = 9):
        return f"{nb / div:{w}.2f}"
    lines = []
    W = 30
    prev_by_dev = {r.device: r for r in previous} if previous else {}
    prev = {"r": None}          # report being diffed against, set per device / totals block
    def delta(nb, key):
        if prev["r"] is None:
            return ""
        d = nb - (prev["r"][key] if isinstance(prev["r"], dict) else getattr(prev["r"], key))
        if abs(d) < (1 << 20):
            return " " * 12
        sign = "+" if d > 0 else "-"
        s = f"{sign}{abs(d) / div:.2f} {unit}" if abs(d) >= div / 100 else f"{sign}{abs(d) / 2**20:.0f} MiB"
        return c("red" if d > 0 else "green", f"{s:>11} ")
    def row(label, nb, style = None, note = "", indent = 2, key = None):
        lab = " " * indent + f"{label:<{W - indent}}"
        val = q(nb) + f" {unit}"
        if style:
            lab, val = c(style, lab), c(style, val)
        d = delta(nb, key) if key else (" " * 12 if previous else "")
        lines.append(lab + val + d + (c("dim", "   " + note) if note else ""))
    def head(text):
        lines.append(c("bold", c("cyan", text)))
    def sub(text):
        lines.append(c("bold", "  " + text))

    for r in reports:
        prev["r"] = prev_by_dev.get(r.device)
        head(f"{r.device}  {r.device_name}" + (c("dim", "   (delta since previous report)") if prev["r"] else ""))
        sub("live tensors")
        row("weights", r.weights, "green",
            f"direct {q(r.weights_direct, 6).strip()} + arena {q(r.weights_arena_used, 6).strip()} used"
            + (f" + {q(r.weights_arena_tail, 6).strip()} tail ({r.arena_blocks} blocks)" if r.arena_blocks else ""),
            key = "weights")
        row("cache layers", r.cache, "green", key = "cache")
        row("recurrent states", r.recurrent, "green", key = "recurrent")
        tags = ", ".join(f"{t} {v / 2**20:.0f} MiB" for t, v in list(r.statics_by_tag.items())[:6])
        row("statics (g_tensor_cache)", r.statics, "green", tags, key = "statics")
        if prev["r"] is not None:
            # tags that appeared or grew since the previous report
            grown = [(t, v - prev["r"].statics_by_tag.get(t, 0)) for t, v in r.statics_by_tag.items()
                     if v - prev["r"].statics_by_tag.get(t, 0) >= (1 << 20)]
            if grown:
                lines.append(c("dim", "      new/grown statics: " + ", ".join(f"{t} +{d / 2**20:.0f} MiB" for t, d in grown[:8])))
        if r.generator:
            row("generator-owned", r.generator, "green", key = "generator")
        row("other / unattributed", r.other, "yellow" if r.other > (64 << 20) else "green", key = "other")
        for sh, dt, nb, path in r.other_examples[:3]:
            if nb >= (16 << 20):
                lines.append(c("dim", f"      {sh} {dt} {nb / 2**20:.0f} MiB {path}"))
        row("= live storages", r.live_total, "bold", f"torch allocated {q(r.allocated, 6).strip()}", key = "live_total")
        sub("allocator")
        row("inactive-split (fragmented)", r.inactive_split, "yellow" if r.inactive_split > (256 << 20) else None,
            f"{r.segments} segments" if r.segments else "expandable segments", key = "inactive_split")
        row("cached free blocks", r.cached_free, None,
            f"peak allocated {q(r.peak_allocated, 6).strip()}, peak reserved {q(r.peak_reserved, 6).strip()}",
            key = "cached_free")
        row("= torch reserved", r.reserved, "bold", key = "reserved")
        sub("device")
        row("non-torch (context, runtime)", r.non_torch, key = "non_torch")
        row("= device used", r.device_used, "bold", f"of {q(r.device_total, 6).strip()} {unit}", key = "device_used")
        lines.append("")
    if totals and len(reports) > 1:
        head(f"all {len(reports)} devices")
        attrs = (("weights", "weights"), ("cache layers", "cache"), ("recurrent states", "recurrent"),
                 ("statics", "statics"), ("other", "other"), ("torch allocated", "allocated"),
                 ("torch reserved", "reserved"), ("non-torch", "non_torch"), ("device used", "device_used"))
        prev["r"] = {a: sum(getattr(p, a) for p in previous) for _, a in attrs} if previous else None
        for label, attr in attrs:
            row(label, sum(getattr(r, attr) for r in reports), "bold" if label.startswith(("torch", "device")) else None, key = attr)
        lines.append("")
    return "\n".join(lines)
