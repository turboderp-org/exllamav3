"""
Shared-memory budget for the Triton attention kernels.

The tile configs in this package are sized for the ~100 KB of dynamic shared memory that Ampere
and later grant a block. Older parts grant less (Turing: 64 KB), and Triton does not shrink a
tile on its own: a kernel whose compiled footprint exceeds the limit fails at launch with
OutOfResources. Guessing per-architecture tiles is unreliable because the footprint of the same
config differs between architectures (no cp.async pipelining below sm_80, for one), so each
launch site lists a ladder of configs instead -- the stock config first, then progressively
smaller tiles -- and pick_config() walks it with compile-only warmups, taking the first whose
measured footprint fits the device. Picks are cached per launch key, so the walk is paid once
per kernel family, and on a full-budget device the first candidate always fits, which is exactly
the compile the launch would have done anyway.
"""

import os
import torch

_limit = {}
# Test override: pretend the device grants at most this many bytes, to exercise the ladders on
# any GPU (the picks then reflect this device's footprints, not the smaller device's)
_env_limit = int(os.environ.get("EXL3_TRITON_SMEM_LIMIT", "0") or "0")
# EXL3_TRITON_SMEM_DEBUG=1 prints every pick with its measured footprint
_debug = bool(os.environ.get("EXL3_TRITON_SMEM_DEBUG"))
_picks = {}


class NoFittingConfig(RuntimeError):
    """No candidate in a ladder fits the device's shared memory. Callers with an alternative
    formulation of the same computation catch this and take it; the miss is cached, so the
    probe cost is paid once per launch key."""


def _dev_index(device) -> int:
    idx = device.index if hasattr(device, "index") else device
    return torch.cuda.current_device() if idx is None else idx


def smem_limit(device) -> int:
    """Dynamic shared memory per block the device grants (the opt-in maximum), in bytes."""
    idx = _dev_index(device)
    lim = _limit.get(idx)
    if lim is None:
        props = torch.cuda.get_device_properties(idx)
        lim = getattr(props, "shared_memory_per_block_optin", 0)
        if not lim:
            lim = 64 * 1024 if props.major < 8 else 96 * 1024
        if _env_limit:
            lim = min(lim, _env_limit)
        _limit[idx] = lim
    return lim


def pick_config(device, name: str, key, candidates, probe):
    """First config in `candidates` whose compiled kernel fits the device's shared memory.

    probe(cfg) compiles the kernel for that config without launching it and returns the
    footprint in bytes (see shared_bytes). `key` must cover every launch constant the footprint
    depends on; the pick is cached per (device architecture, limit, name, key). Raises
    NoFittingConfig when no candidate fits (also cached).
    """
    idx = _dev_index(device)
    limit = smem_limit(idx)
    full_key = (torch.cuda.get_device_capability(idx), limit, name, key)
    cfg = _picks.get(full_key, ...)
    if cfg is None:
        raise NoFittingConfig(f"{name}: no tile configuration fits the device's {limit} B of dynamic shared memory")
    if cfg is not ...:
        return cfg
    need = None
    for cand in candidates:
        need = probe(cand)
        if _debug:
            print(f" -- smem: {name} {key} config {cand}: {need} B ({'fits' if need <= limit else 'over'} {limit} B)", flush = True)
        if need <= limit:
            _picks[full_key] = cand
            return cand
    _picks[full_key] = None
    raise NoFittingConfig(
        f"{name}: no tile configuration fits the device's {limit} B of dynamic shared memory "
        f"(the smallest candidate needs {need} B)"
    )


def shared_bytes(kernel, args, **kwargs) -> int:
    """Compile-only warmup of a JIT kernel with the launch's real arguments; the launch grid does
    not affect the footprint. The compile is cached by Triton, so the eventual launch reuses it."""
    return kernel.run(*args, grid = (1,), warmup = True, **kwargs).metadata.shared


def tile_ladder(block_m: int, block_n: int, num_warps: int, num_stages: int,
                min_m: int = 16, min_n: int = 16):
    """Stock (block_m, block_n, num_warps, num_stages) first, then halving q tiles at the same
    kv tile, then the same at a halved kv tile. Warps follow the q tile so each warp keeps whole
    16-row MMA tiles."""
    cands = [(block_m, block_n, num_warps, num_stages)]
    bn = block_n
    while bn >= min_n:
        bm = block_m
        while bm >= min_m:
            c = (bm, bn, max(1, min(num_warps, bm // 16)), num_stages)
            if c not in cands:
                cands.append(c)
            bm //= 2
        bn //= 2
    return cands


def halving_ladder(value: int, minimum: int = 16):
    """value, value / 2, ... down to `minimum` (single-parameter ladders such as a kv tile). The
    stock value always leads, even when it is already below the minimum."""
    cands = [value]
    while value // 2 >= minimum:
        value //= 2
        cands.append(value)
    return cands
