import math
import threading
import time
import torch
import socket, contextlib
import weakref

lock = threading.RLock()

def synchronized(func):
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)
    return wrapper

def align_to(value, alignment):
    return int(math.ceil(value / alignment) * alignment)


class Timer:
    """
    Context manager to record duration
    """

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.interval = self.end_time - self.start_time


def cuda_sync_active():
    """
    Calling torch.cuda.synchronize() will create a CUDA context on CUDA:0 even if that device is not being used.
    This function synchronizes only devices actively used by Torch in the current process.
    """
    for device_id in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{device_id}')
        if torch.cuda.memory_allocated(device) > 0:
            torch.cuda.synchronize(device)


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def human_time(seconds: float) -> str:
    seconds = round(seconds)
    minutes = seconds // 60
    hours = minutes // 60
    minutes -= hours * 60
    if hours:
        if minutes:
            hs = "s" if hours > 1 else ""
            ms = "s" if minutes > 1 else ""
            return f"{hours} hour{hs}, {minutes} minute{ms}"
        else:
            hs = "s" if hours > 1 else ""
            return f"{hours} hour{hs}"
    elif minutes:
        ms = "s" if minutes > 1 else ""
        return f"{minutes} minute{ms}"
    else:
        return f"< 1 minute"


def first_not_none(*values):
    return next((v for v in values if v is not None), None)


def ratio_split(d, weights, chunk_size = 128):
    assert d % chunk_size == 0, "Total must be divisible by chunk size"
    total_chunks = d // chunk_size
    total_weight = sum(weights)
    ideal_chunks = [total_chunks * w / total_weight for w in weights]
    base_chunks = [int(c) for c in ideal_chunks]
    remainder = total_chunks - sum(base_chunks)
    residuals = [c - int(c) for c in ideal_chunks]
    for i in sorted(range(len(residuals)), key = lambda i: -residuals[i])[:remainder]:
        base_chunks[i] += 1
    final_alloc = [c * chunk_size for c in base_chunks]
    assert sum(final_alloc) == d
    return final_alloc


def find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class Cleanupper:
    """
    Utility class to call cleanup functions at the end of the __main__ scope. Similar functionality to
    atexit but called before Python starts tearing down objects/threads.
    """

    def __init__(self):
        self.atexit_fns = []
        weakref.finalize(self, self._shutdown)

    def register_atexit(self, fn):
        self.atexit_fns.append(fn)

    def unregister_atexit(self, fn):
        if fn in self.atexit_fns:
            self.atexit_fns.remove(fn)

    def _shutdown(self):
        for fn in self.atexit_fns:
            fn()
        self.atexit_fns = []


def set_process_priority_and_affinity():
    import psutil, os
    import multiprocessing as mp

    p = psutil.Process(os.getpid())
    # Try to bump priority slightly. May need sudo (?)
    try:
        p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS if os.name == "nt" else -5)
    except PermissionError:
        pass
    except Exception as e:
        pass

    # Pin to a core
    # TODO: Pick an idle core automatically?
    try:
        p.cpu_affinity([0])  # pick an isolated/quiet core if possible
    except AttributeError:
        pass
    except Exception as e:
        pass
