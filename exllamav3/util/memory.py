from dataclasses import dataclass
from collections import deque
import torch
import gc
import sys

# @lru_cache
# def init_pynvml():
#     pynvml.nvmlInit()

# Try to make sure device is live for correct measurement of free VRAM
def touch_device(device: int):
    d = torch.empty((32, 32), device = device, dtype = torch.float)
    d = d @ d
    d = d + d


# Reserve byte amount on device
def set_memory_fraction_reserve(
    reserve: int,
    device: int
):
    touch_device(device)
    free, total = torch.cuda.mem_get_info(device)
    fraction = (free - reserve) / total
    torch.cuda.set_per_process_memory_fraction(fraction, device = device)


# Reserve all but byte amount on device
def set_memory_fraction_use(
    use: int,
    device: int
):
    touch_device(device)
    free, total = torch.cuda.mem_get_info(device)
    baseline = torch.cuda.memory_allocated(device)
    fraction = min((baseline + use) / total, 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, device = device)


# Un-reserve VRAM
def unset_memory_fraction(active_devices: list[int]):
    for i in active_devices:
        torch.cuda.set_per_process_memory_fraction(1.0, device = i)


# Free unused VRAM
def free_mem():
    gc.collect()
    torch.cuda.empty_cache()



def list_gpu_tensors(min_size: int = 1):
    import threading
    import warnings
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

    def collect(path, item):
        nonlocal results
        if not isinstance(item, torch.Tensor) or not item.is_cuda:
            return
        size = item.nelement() * item.element_size() // (1024**2)
        if size < min_size:
            return
        if ".stderr.dbg." in path:
            return
        if ".__main__." in path:
            path = path[path.find(".__main__.") + 10:]
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

    queue = deque()

    for name, obj in globals().items():
        collect(name, obj)
        queue.append((name, obj))

    for thread_id, frame in sys._current_frames().items():
        prefix = ""
        if thread_id == threading.get_ident():
            frame = frame.f_back
        while frame:
            for name, obj in frame.f_locals.items():
                new_path = f"{prefix[2:]}.{name}"
                collect(new_path, obj)
                queue.append((name, obj))
            frame = frame.f_back
            prefix += "."

    while queue:
        path, obj = queue.popleft()

        if hasattr(obj, '__dict__'):
            for attr, value in obj.__dict__.items():
                new_path = f"{path}.{attr}"
                collect(new_path, value)
                if id(value) not in visited:
                    visited.add(id(value))
                    queue.append((new_path, value))

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}['{key}']"
                collect(new_path, value)
                if id(value) not in visited:
                    visited.add(id(value))
                    queue.append((new_path, value))

        if isinstance(obj, (list, tuple, set)):
            for idx, item in enumerate(obj):
                new_path = f"{path}[{idx}]"
                collect(new_path, item)
                if id(item) not in visited:
                    visited.add(id(item))
                    queue.append((new_path, item))

    from tabulate import tabulate
    devices: dict[str, list] = {}
    items = list(results.values())
    items.sort(key = lambda x: -x.size)
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

    for k in sorted(devices.keys()):
        print()
        print(f"--------------")
        print(f"| {k:10} |")
        print(f"--------------")
        print()
        headers = ["size // MB", "path", "shape", "dtype"]
        print(tabulate(devices[k], headers = headers, tablefmt = "github", intfmt=","))

