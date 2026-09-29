"""Bounded pinned staging for single-device recurrent checkpoints.

The LRU retains ordinary pageable storage. Unsupported layouts fall back to
the existing per-layer copies; no checkpoint interval or cache size changes.
"""

import threading

import torch

STAGING_BYTES = 128 * 1024**2
_staging = None
_lock = threading.RLock()


def _layout(groups, checkpoint_size):
    entries, devices = [], set()
    offset = 0
    for key, tensors in groups.items():
        items = []
        for tensor in tensors:
            if tensor.layout != torch.strided or tensor.device.type not in ("cpu", "cuda"):
                return None
            if tensor.is_cuda:
                devices.add(tensor.device)
            size = tensor.numel() * tensor.element_size()
            if offset % tensor.element_size():
                return None
            items.append((tensor, offset, size))
            offset += size
        entries.append((key, items))
    if offset != checkpoint_size or not 0 < offset <= STAGING_BYTES or len(devices) > 1:
        return None
    return entries, offset, next(iter(devices), None)


def _view(buffer, tensor, offset, size):
    return buffer[offset : offset + size].view(tensor.dtype).view(tensor.shape)


def _get_staging():
    global _staging
    if _staging is None:
        _staging = torch.empty(STAGING_BYTES, dtype = torch.uint8, device = "cpu", pin_memory = True)
    return _staging


def try_stash(groups, position, checkpoint_size):
    plan = _layout(groups, checkpoint_size)
    if plan is None or plan[2] is None:
        return None
    entries, size, device = plan
    with _lock:
        try:
            staging = _get_staging()[:size]
        except RuntimeError:
            # Pinned host allocation may be unavailable or constrained by the OS.
            return None
        stream = torch.cuda.current_stream(device)
        try:
            for _, items in entries:
                for source, offset, length in items:
                    _view(staging, source, offset, length).copy_(source, non_blocking = source.is_cuda)
        finally:
            # The arena cannot be published or reused with outstanding transfers.
            stream.synchronize()
        stored = torch.empty(size, dtype = torch.uint8, device = "cpu", pin_memory = False)
        stored.copy_(staging)
    result = {"position": position, "checkpoint_size": size, "_coalesced_slab": stored}
    for key, items in entries:
        result[key] = tuple(_view(stored, source, offset, length) for source, offset, length in items)
    return result


def restore(groups, position, checkpoint_size, saved):
    plan = _layout(groups, checkpoint_size)
    if plan is None or plan[2] is None or saved["position"] != position:
        raise ValueError("Incompatible coalesced recurrent checkpoint layout")
    entries, size, device = plan
    stored = saved["_coalesced_slab"]
    if (stored.device.type != "cpu" or stored.dtype != torch.uint8 or not stored.is_contiguous()
            or stored.ndim != 1 or stored.numel() != size or saved["checkpoint_size"] != size):
        raise ValueError("Incompatible coalesced recurrent checkpoint storage")
    for key, items in entries:
        tensors = saved[key]
        if len(tensors) != len(items) or any(t.dtype != target.dtype or t.shape != target.shape
                                          for t, (target, _, _) in zip(tensors, items)):
            raise ValueError("Incompatible coalesced recurrent checkpoint tensor")
    with _lock:
        staging = _get_staging()[:size]
        staging.copy_(stored)
        stream = torch.cuda.current_stream(device)
        try:
            for _, items in entries:
                for target, offset, length in items:
                    target.copy_(_view(staging, target, offset, length), non_blocking = target.is_cuda)
        finally:
            stream.synchronize()
