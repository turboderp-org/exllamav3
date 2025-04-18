import math
import threading
import time
import torch


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
