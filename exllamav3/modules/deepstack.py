from __future__ import annotations
from typing_extensions import override
from . import Module
from ..model.config import Config
import torch
from ..model.model_tp_alloc import TPAllocation

class DeepstackEmbed(Module):
    def __init__(
        self,
        config: Config | None,
        key: str,
        deepstack_index: int,
    ):
        super().__init__(config, key, None)
        self.deepstack_index = deepstack_index
        self.module_name = "DeepstackEmbed"

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

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        return []

    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."
        return {
            "cls": DeepstackEmbed,
            "kwargs": {
                "key": self.key,
                "deepstack_index": self.deepstack_index,
            },
            "device": self.device
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        module = DeepstackEmbed(
            config = None,
            **exported["kwargs"],
        )
        module.device = exported["device"]
        return module