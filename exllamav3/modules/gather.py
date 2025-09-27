from __future__ import annotations
from typing_extensions import override
from . import Module
import torch
import torch.distributed as dist

class OutputGather(Module):
    def __init__(
        self,
        config: None,
        key: str,
        device: int,
        output_device: int,
        gather_devices: list[int],
        ldims: list[int],
    ):
        super().__init__(config, key, None)
        self.device = device
        self.output_device = output_device
        self.gather_devices = gather_devices
        self.ldims = ldims
        self.odim = sum(ldims)
        self.active = device == output_device or device in gather_devices

    @override
    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def load(self, device: torch.Device, **kwargs):
        raise NotImplementedError()

    @override
    def unload(self):
        raise NotImplementedError()

    @override
    def get_tensors(self):
        raise NotImplementedError()

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor | None:

        if len(self.gather_devices) == 1 and self.device == self.output_device:
            return x

        if not self.active:
            return None

        backend = params["backend"]

        if self.output_device == self.device:
            out_shape = list(x.shape)
            out_shape[-1] = self.odim
            out_tensor = torch.empty(*out_shape, dtype = x.dtype, device = x.device)
        else:
            out_tensor = None

        # print(f"Gather:  device {self.device}, ldims {self.ldims}")

        backend.gather(x, out_tensor, self.gather_devices, self.output_device, self.ldims)
        return out_tensor
