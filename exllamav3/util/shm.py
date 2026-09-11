from __future__ import annotations
import os

# POSIX shared memory (multiprocessing.shared_memory) lives on the /dev/shm tmpfs on Linux. Creating
# a segment larger than that filesystem succeeds (ftruncate + mmap are lazy) and only fails when the
# pages past the limit are touched: SIGBUS from the CPU, or "invalid argument" from cudaHostRegister
# when the segment is pinned for CUDA. Docker containers default to a 64 MiB /dev/shm, so check the
# capacity up front and say what to do about it.

SHM_DIR = "/dev/shm"


def shm_free_bytes() -> tuple[int, int] | None:
    """(free, total) bytes of the shared-memory filesystem, or None where it can't be queried"""
    if os.name != "posix" or not os.path.isdir(SHM_DIR):
        return None
    try:
        st = os.statvfs(SHM_DIR)
    except OSError:
        return None
    return st.f_bavail * st.f_frsize, st.f_blocks * st.f_frsize


def check_shm_capacity(nbytes: int, purpose: str):
    """Raise a RuntimeError naming the limit if a shared-memory segment of nbytes cannot be backed"""
    fs = shm_free_bytes()
    if fs is None:
        return
    free, total = fs
    if nbytes <= free:
        return
    raise RuntimeError(
        f"{purpose} needs {nbytes / 2**20:.1f} MiB of POSIX shared memory ({SHM_DIR}), but only "
        f"{free / 2**20:.1f} MiB of {total / 2**20:.1f} MiB is free. A segment larger than the filesystem "
        f"can be created but its pages cannot be backed, so pinning it for CUDA fails with 'invalid "
        f"argument'. In Docker the default {SHM_DIR} is 64 MiB: start the container with a larger "
        f"--shm-size (e.g. --shm-size=1g, or shm_size: \"1gb\" in compose), or mount a larger tmpfs on "
        f"{SHM_DIR}."
    )
