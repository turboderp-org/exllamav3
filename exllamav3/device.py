from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from .constants import PAGE_SIZE
from .models import Config

device_contexts = {}

class DeviceContext:

    def __init__(
        self,
        config: Config,
        device: torch.Device,
    ):
        self.reference_count = 0
        self.device = device
        self.config = config


def get_key(
    config: Config,
    device: torch.Device,
):
    return f"{str(config.uuid)},{str(device)}"


def get_device_context(config: Config, device: torch.device):
    key = get_key(config, device)
    if key not in device_contexts:
        device_contexts[key] = DeviceContext(config, device)
    dc = device_contexts[key]
    dc.reference_count += 1
    return dc


def release_device_context(config: Config, device: torch.device):
    key = get_key(config, device)
    assert key in device_contexts
    dc = device_contexts[key]
    dc.reference_count -= 1
    if dc.reference_count == 0:
        del device_contexts[key]
