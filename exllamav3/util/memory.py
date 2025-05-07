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
        if hasattr(obj, '__dict__'):
            for attr, value in obj.__dict__.items():
                new_path = f"{path}.{attr}"
                collect(new_path, value)
                if id(value) not in visited:
                    visited.add(id(value))
                    queue.append((new_path, value))

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

