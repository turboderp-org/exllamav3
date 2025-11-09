from __future__ import annotations
from typing_extensions import override
from . import Module
from ..model.config import Config
import torch
import torch.distributed as dist

class DeepstackEmbed(Module):
    def __init__(
        self,
        config: Config,
        key: str,
        deepstack_index: int,
    ):
        super().__init__(config, key, None)
        self.deepstack_index = deepstack_index
        self.module_name = "DeepstackEmber"

    @override
    def optimizer_targets(self):
        return []

    @override
    def get_tensors(self):
        return {}

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor | None:

        emb = params.get("deepstack_emb")
        if emb is None:
            return x

        t = emb[self.deepstack_index].to(x.device)
        x += t
        return x