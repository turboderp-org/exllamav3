from __future__ import annotations
from typing_extensions import override
import torch
import torch.nn.functional as F
from torch import nn
from ..model.config import Config
from ..util.tensor import to2
from . import Module
from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
from ..model.model_tp_alloc import TPAllocation

class Embedding(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        vocab_size: int,
        hidden_size: int,
        out_dtype: torch.dtype | None = torch.float,
        qmap: str | None = None,
        normalize: bool = False
    ):
        super().__init__(config, key, None)
        assert qmap is None, "No quant scheme for Embedding"

        self.key = key
        self.embedding = None
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.out_dtype = out_dtype
        self._numel = vocab_size * hidden_size
        self.normalize = normalize

        self.caps.update({
            "prefer_cpu": True,
        })

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        weight = self.config.stc.get_tensor(self.key + ".weight", self.device, float2half = True)
        self._numel = weight.numel()
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            device = "meta"
        )
        self.embedding.weight = nn.Parameter(weight)

    @override
    def unload(self):
        self.device = None
        self.embedding = None

    @override
    def get_tensors(self):
       return {
            f"{self.key}.weight": self.embedding.weight.data.contiguous()
        }

    @override
    def weights_numel(self):
        return self._numel
        
    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:

        indexed_emb = params.get("indexed_embeddings")
        input_ids = x
        out_dtype = out_dtype or self.out_dtype or x.dtype

        # Indexed embedding masks
        if indexed_emb:
            standard_mask = input_ids < FIRST_MM_EMBEDDING_INDEX
            indexed_masks = [
                (input_ids >= e.first_index) & (input_ids < (e.first_index + e.mm_length))
                for e in indexed_emb
            ]
            indexed_act = [im.any() for im in indexed_masks]
            use_indexed_emb = any(indexed_act)

        # Mixed embeddings when needed
        if indexed_emb and use_indexed_emb:
            bsz, seq_len = input_ids.shape
            combined_emb = torch.empty((bsz, seq_len, self.hidden_size), device = self.device, dtype = out_dtype)

            # Prepare deepstack embedding tensors
            if any(ie.deepstack_embeddings is not None for ie in indexed_emb) and indexed_act:
                assert all(ie.deepstack_embeddings is not None for ie in indexed_emb)
                num_layers = len(indexed_emb[0].deepstack_embeddings)
                assert all(num_layers == len(ie.deepstack_embeddings) is not None for ie in indexed_emb)
                deepstack_emb = [torch.zeros_like(combined_emb) for _ in range(num_layers)]
            else:
                deepstack_emb = None

            # Insert standard embeddings
            if standard_mask.any():
                for i in range(bsz):
                    standard_ids_row = input_ids[i][standard_mask[i]]
                    standard_emb_row = self.embedding(standard_ids_row)
                    combined_emb[i][standard_mask[i]] = standard_emb_row.to(out_dtype)

            # Only normalize standard embeddings
            if self.normalize:
                combined_emb *= combined_emb.shape[-1] ** 0.5

            # Insert indexed embeddings
            for im, ie, act in zip(indexed_masks, indexed_emb, indexed_act):
                if not act:
                    continue
                for i in range(bsz):
                    indexed_ids_row = input_ids[i][im[i]] - ie.first_index
                    combined_emb[i][im[i]] = ie.embeddings[indexed_ids_row].to(out_dtype)

                    # Prepare deepstack embeddings
                    if ie.deepstack_embeddings is not None:
                        for layer, de in enumerate(ie.deepstack_embeddings):
                            deepstack_emb[layer][i][im[i]] = de[indexed_ids_row].to(out_dtype)

            # Save deepstack embeddings to params
            if deepstack_emb is not None:
                params["deepstack_emb"] = deepstack_emb

            return combined_emb

        # No indexed embeddings, or none in current batch
        else:
            x = self.embedding.forward(x)
            x = to2(x, out_dtype, self.out_dtype)
            if self.normalize:
                x *= x.shape[-1] ** 0.5
            return x

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        return []

    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."
        return {
            "cls": Embedding,
            "kwargs": {
                "key": self.key,
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "out_dtype": self.out_dtype,
                "normalize": self.normalize,
            },
            "embedding.weight": producer.send(self.embedding.weight),
            "device": self.device
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        module = Embedding(
            config = None,
            **exported["kwargs"],
        )
        module.device = exported["device"]
        module.embedding = nn.Embedding(
            module.vocab_size,
            module.hidden_size,
            device = "meta"
        )
        emb = consumer.recv(exported["embedding.weight"], cuda = False)
        module.embedding.weight = nn.Parameter(emb)
        return module