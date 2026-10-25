"""
Pinned host memory on a model that leaves the first device unused (a manual split of [0, N]), each case in a fresh
process, since what matters is which devices hold a CUDA context:

- Linear.pin_linears registers the pinned weights with the layer's own device, whatever device the loading thread
  has current. Host memory belongs to the device current at registration, and a zero-copy alias for any other
  device is refused ("Specified device cuda:1 does not match device of data cuda:0"). A first load got away with
  it while the first device had no context (torch then never switches back to it); any later load in the same
  process failed.
- PinnedArena.close waits only for devices in use: a bare torch.cuda.synchronize() created a context on the
  first device when the CPU page cache was closed at unload.
"""

import pytest
import torch

from testlib.isolated import run_isolated

pytestmark = pytest.mark.cuda_only

KEY = "model.layers.0.mlp.up_proj"


def worker_pin_linears(directory, first, other, init_first):
    from exllamav3.modules import Linear
    from testlib.checkpoint import module_config
    from testlib.exl3 import checkpoint_tensors, generator

    if init_first:
        # What a previous load and unload leave behind: a context on the first device, current on this thread
        torch.zeros(1, device = f"cuda:{first}")
        torch.cuda.set_device(first)

    device = torch.device("cuda", other)
    with torch.inference_mode():
        t = checkpoint_tensors(KEY, 512, 256, 4, generator(0), codebook = "mul1")
        module = Linear(module_config(t, directory), KEY, 512, 256)
        module.load(device)
        x = torch.randn((1, 4, 512), generator = generator(1)).half().to(device)
        resident = module.forward(x, {}).cpu()
        module.pin_linears()
        pinned = module.forward(x, {}).cpu()
        result = {
            "alias_device": str(module.inner.trellis.device),
            "store_pinned": module.inner._pinned_store.is_pinned(),
            "max_diff": (pinned.float() - resident.float()).abs().max().item(),
        }
        module.unload()
    return result


def worker_arena_close(other):
    from exllamav3.util.pinned_arena import PinnedArena

    device = torch.device("cuda", other)
    torch.zeros(1, device = device)
    arena = PinnedArena(1 << 20, "test arena", mapped = True, required = True)
    host = arena.tensor[:4096]
    host.copy_(torch.arange(4096, dtype = torch.uint8))
    on_device = host.to(device, non_blocking = True)
    host.copy_(on_device, non_blocking = True)
    arena.close()
    has_context = torch._C._cuda_hasPrimaryContext
    return [i for i in range(torch.cuda.device_count()) if has_context(i)]


def two_devices(devices):
    if len(devices) < 2:
        pytest.skip("needs two GPUs")
    return devices[0].index, devices[1].index


@pytest.mark.parametrize("init_first", [False, True], ids = ["first_unused", "first_current"])
def test_pin_linears_on_another_device(tmp_path, devices, init_first):
    first, other = two_devices(devices)
    result = run_isolated(worker_pin_linears, str(tmp_path), first, other, init_first)
    assert result["alias_device"] == f"cuda:{other}"
    assert result["store_pinned"]
    # The same weights, read through the alias
    assert result["max_diff"] == 0.0


def test_arena_close_creates_no_context(devices):
    _, other = two_devices(devices)
    assert run_isolated(worker_arena_close, other) == [other]
