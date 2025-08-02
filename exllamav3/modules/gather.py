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
        rank: int,
        world_size: int,
        output_rank: int,
        splits: list[(int, int)],
        gather_dim: int = -1
    ):
        super().__init__(config, key, None)
        self.rank = rank
        self.world_size = world_size
        self.output_rank = output_rank

        self.splits = splits
        self.output_dim = sum(s[1] - s[0] for s in splits)
        self.gather_dim = gather_dim

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
    ) -> torch.Tensor:

        if self.world_size == 1 and self.output_rank == self.rank:
            return x

        gather_dim = self.gather_dim % x.ndim

        # Allocate output buffer only on the gather rank
        if self.rank == self.output_rank:
            out_shape = list(x.shape)
            out_shape[gather_dim] = self.output_dim
            out = torch.empty(*out_shape, dtype = x.dtype, device = x.device)
        else:
            out = None

        # output_rank receives, other ranks send
        if self.rank == self.output_rank:
            for src, s in enumerate(self.splits):
                out_slice = out.narrow(gather_dim, s[0], s[1] - s[0])
                if src == self.rank:
                    out_slice.copy_(x)
                else:
                    rbuf = torch.empty_like(out_slice)
                    dist.recv(rbuf, src = src)
                    out_slice.copy_(rbuf)
        else:
            dist.send(x, dst = self.output_rank)

        return out