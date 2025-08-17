import os
from functools import lru_cache
import time

_col_log = "\u001b[33;1m"  # Yellow
_col_default = "\u001b[0m"

global_t0 = time.time()

@lru_cache
def dbg_timezero(s: str):
    return

@lru_cache
def dbg_enabled(s: str, t0 = None):
    global global_t0
    v = "EXLLAMA_DEBUGLOG_" + s.upper()
    if v in os.environ:
        return global_t0
    return None

def set_t0(s: str, t0):
    global global_t0
    global_t0 = t0

def log(s: str, t: str):
    tz = dbg_enabled(s)
    if tz:
        timestamp = f"{time.time() - tz:.3f}"
        print(f"{_col_log} -- {timestamp} s: {t}{_col_default}")

def log_tp(device: int | None, t: str):
    if device is None:
        log("TP", f"main process, {t}")
    elif device == -1:
        log("TP", f"CPU process, {t}")
    else:
        log("TP", f"device {device}, {t}")