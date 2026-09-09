from __future__ import annotations
import os
import threading
import torch

"""
Device-to-device tensor moves.

On some platforms the driver reports peer-to-peer access between two GPUs that the PCIe fabric
does not actually deliver (ACS redirection, an IOMMU not in passthrough mode, misreported
capability on some consumer boards). Torch trusts the driver, so a plain .to() between such
devices silently produces an empty or garbage tensor rather than an error. Every cross-device
move in the engine goes through to_device(), which decides per (source, destination) pair
whether the copy can go direct or has to bounce through host memory:

    EXLLAMA_NO_P2P_COPY unset  -> autodetect: the first move between a pair of devices probes
                                  it (a few random floats there and back, compared on the host)
                                  and the verdict is kept for the life of the process
    EXLLAMA_NO_P2P_COPY=1      -> always bounce (any value other than "0")
    EXLLAMA_NO_P2P_COPY=0      -> always direct, never probe

Both directions of a pair are probed together, since the fabric can fail one way only.
"""

_env = os.environ.get("EXLLAMA_NO_P2P_COPY")
FORCED: bool | None = None if _env is None else (_env != "0")   # None: autodetect

_lock = threading.Lock()
_verdicts: dict[tuple[int, int], bool] = {}    # (src index, dst index) -> needs bounce
stats = {"direct": 0, "bounced": 0, "probes": 0}


def _probe_direct(src: torch.device, dst: torch.device) -> tuple[bool, bool]:
    """Move random data src -> dst -> src. Returns (src->dst ok, dst->src ok), each judged on
    the host so the verdict never depends on a peer copy itself."""
    a = torch.rand(64, dtype = torch.float32, device = src)
    b = a.to(dst)
    c = b.to(src)
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)
    a_h = a.cpu()
    fwd_ok = torch.equal(a_h, b.cpu())
    bwd_ok = fwd_ok and torch.equal(a_h, c.cpu())
    return fwd_ok, bwd_ok


def needs_bounce(src: torch.device, dst: torch.device) -> bool:
    """Whether a move between two CUDA devices must go through host memory."""
    if FORCED is not None:
        return FORCED
    key = (src.index, dst.index)
    v = _verdicts.get(key)
    if v is not None:
        return v
    with _lock:
        v = _verdicts.get(key)
        if v is None:
            stats["probes"] += 1
            fwd_ok, bwd_ok = _probe_direct(src, dst)
            _verdicts[key] = not fwd_ok
            _verdicts[(dst.index, src.index)] = not bwd_ok
            if not (fwd_ok and bwd_ok):
                print(f" !! Direct copies between {src} and {dst} corrupt data "
                      f"({'->' if not fwd_ok else ''}{'<-' if not bwd_ok else ''}); "
                      f"routing them through system memory (see EXLLAMA_NO_P2P_COPY)")
            v = _verdicts[key]
    return v


def to_device(t: torch.Tensor, device: torch.device | str | int, non_blocking: bool = False) -> torch.Tensor:
    """
    t.to(device), with CUDA-to-CUDA moves bounced through host memory when the pair needs it.
    Host-to-device and device-to-host moves pass straight through (non_blocking honoured).
    """
    if device is None:
        # Emulate tensor.to(None): no-op
        return t
    device = torch.device(device)
    if t.device == device:
        return t
    if t.device.type == "cuda" and device.type == "cuda" and t.device.index != device.index:
        if needs_bounce(t.device, device):
            stats["bounced"] += 1
            return t.cpu().to(device)
        stats["direct"] += 1
    return t.to(device, non_blocking = non_blocking)


def reset_verdicts():
    """Forget probe results (tests)."""
    with _lock:
        _verdicts.clear()
