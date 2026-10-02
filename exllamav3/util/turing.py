import os
import torch

# Turing (sm_75) fast paths. GeForce Turing parts have no bf16 tensor cores, run fp32-accumulate HMMA at
# half the fp16-accumulate rate, give a block 64 KB of shared memory, and Triton lowers tl.dot to scalar
# FMA there, so generic paths can leave most of the chip idle. Each path below is switched by an
# EXL3_<NAME> environment variable; when the variable is unset, it defaults to the value here on sm_75
# devices and to 0 (upstream behaviour) on every other architecture.

SM75_DEFAULTS = {
    "SDPA_PREFILL": 1,    # prefill attention on the dequantized window (PyTorch SDPA / fa75), not the packed cache
    "FA75": 1,            # flash-attention prefill kernel for head_dim 256 (needs SDPA_PREFILL)
}

_cc_cache = {}

def _capability(device) -> tuple[int, int]:
    if device is None:
        idx = torch.cuda.current_device()
    elif isinstance(device, torch.Tensor):
        idx = device.device.index
    elif isinstance(device, torch.device):
        idx = device.index if device.index is not None else torch.cuda.current_device()
    elif isinstance(device, str):
        d = torch.device(device)
        idx = d.index if d.index is not None else torch.cuda.current_device()
    else:
        idx = int(device)
    cc = _cc_cache.get(idx)
    if cc is None:
        cc = torch.cuda.get_device_capability(idx) if torch.version.cuda else (0, 0)
        _cc_cache[idx] = cc
    return cc


def turing_flag(name: str, device = None) -> int:
    """
    Level of the sm_75 fast path `name` for `device`: EXL3_<name> if set (0 disables, any other integer
    is taken as given), else SM75_DEFAULTS[name] on sm_75 and 0 elsewhere.
    """
    env = os.environ.get("EXL3_" + name)
    if env is not None:
        try:
            return int(env)
        except ValueError:
            return 0 if env.strip().lower() in ("", "false", "off", "no") else 1
    if not torch.cuda.is_available():
        return 0
    return SM75_DEFAULTS.get(name, 0) if _capability(device) == (7, 5) else 0
