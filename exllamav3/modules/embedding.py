from __future__ import annotations
from typing_extensions import override
import torch
from torch import nn
from ..model.config import Config
from ..util.tensor import to2
from ..util import emb8_kernel
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
        normalize: bool = False,
        multiplier: float = 1.0
    ):
        super().__init__(config, key, None)
        assert qmap is None, "No quant scheme for Embedding"

        self.key = key
        self.embedding = None
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.out_dtype = out_dtype
        self._pinned_staging = {}
        self.quant = None
        self._numel = vocab_size * hidden_size
        self.normalize = normalize
        self.multiplier = multiplier

        self.caps.update({
            "prefer_cpu": True,
        })

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        stc = self.config.stc
        if stc.has_tensor(self.key + ".weight_i8"):
            w8 = stc.get_tensor(self.key + ".weight_i8", device)
            scale = stc.get_tensor(self.key + ".scale", device)
            assert w8.dtype == torch.int8 and scale.dtype == torch.half
            assert w8.shape == (self.vocab_size, self.hidden_size)
            assert scale.shape == (self.vocab_size, self.hidden_size // 32)
            if getattr(stc, "new_tensors", None) is None and stc.has_tensor(self.key + ".weight"):
                print(f" !! {self.key}.weight (16-bit) found alongside int8 storage; ignoring (dead data)")
            self._numel = w8.numel()
            self.quant = {"weight_i8": w8, "scale": scale}
            self.embedding = None
            return
        weight = stc.get_tensor(self.key + ".weight", self.device, float2half = True, allow_bf16 = True)
        self._numel = weight.numel()
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            device = "meta"
        )
        self.embedding.weight = nn.Parameter(weight)

    def _gather(self, ids: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        if self.quant is not None:
            target = dtype or torch.half
            if target in (torch.half, torch.float) and emb8_kernel.available():
                x = emb8_kernel.dequant(self.quant["weight_i8"], self.quant["scale"], ids, target)
                return x.view(*ids.shape, self.hidden_size)
            q = self.quant["weight_i8"][ids].to(target)
            s = self.quant["scale"][ids].to(target)
            x = q.view(*ids.shape, self.hidden_size // 32, 32) * s.unsqueeze(-1)
            return x.view(*ids.shape, self.hidden_size)
        return self.embedding(ids)

    @override
    def unload(self):
        self.device = None
        self.embedding = None
        self.quant = None

    @override
    def get_tensors(self):
        if self.quant is not None:
            return {
                f"{self.key}.weight_i8": self.quant["weight_i8"].contiguous(),
                f"{self.key}.scale": self.quant["scale"].contiguous(),
            }
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

        # Ensure input IDs in params
        if "input_ids" not in params:
            params["input_ids"] = x

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
                    standard_emb_row = self._gather(standard_ids_row, out_dtype)
                    combined_emb[i][standard_mask[i]] = standard_emb_row.to(out_dtype)

            # Only normalize standard embeddings
            if self.normalize:
                combined_emb *= combined_emb.shape[-1] ** 0.5

            # Also only scale standard embeddings
            if self.multiplier != 1.0:
                combined_emb *= self.multiplier

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
            x = self._gather(x, out_dtype)
            if self.multiplier != 1.0:
                x *= self.multiplier
            x = to2(x, out_dtype, self.out_dtype)
            if self.normalize:
                x *= x.shape[-1] ** 0.5
            # When the embedding resides on the CPU, its output is uploaded to the first
            # device layer; staging it through a reused pinned buffer makes that upload
            # asynchronous. Only callers that guarantee a sync point between forward passes
            # (the generator's decode loop) may set the pinned_staging flag.
            if params.get("pinned_staging") and x.device.type == "cpu":
                key = (x.shape, x.dtype)
                buf = self._pinned_staging.get(key)
                if buf is None:
                    if len(self._pinned_staging) > 8:
                        self._pinned_staging.clear()
                    buf = torch.empty_like(x, pin_memory = True)
                    self._pinned_staging[key] = buf
                buf.copy_(x)
                x = buf
            return x

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        return []

    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."
        if self.quant is not None:
            exported_weights = {
                "quant": {
                    "weight_i8": producer.send(self.quant["weight_i8"]),
                    "scale": producer.send(self.quant["scale"]),
                }
            }
        else:
            exported_weights = {"embedding.weight": producer.send(self.embedding.weight)}
        return {
            "cls": Embedding,
            "kwargs": {
                "key": self.key,
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "out_dtype": self.out_dtype,
                "normalize": self.normalize,
                "multiplier": self.multiplier,
            },
            **exported_weights,
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
        if "quant" in exported:
            module.quant = {
                "weight_i8": consumer.recv(exported["quant"]["weight_i8"], cuda = False),
                "scale": consumer.recv(exported["quant"]["scale"], cuda = False),
            }
            module.embedding = None
        else:
            module.embedding = nn.Embedding(
                module.vocab_size,
                module.hidden_size,
                device = "meta"
            )
            emb = consumer.recv(exported["embedding.weight"], cuda = False)
            module.embedding.weight = nn.Parameter(emb)
        return module

    def convert_int8(self, block_size = 32):
        assert block_size == 32, "int8 embedding storage is fixed at block_size 32"
        assert self.embedding is not None, f"Cannot convert {self.key} to int8: embedding not loaded"
        w = self.embedding.weight.data
        assert w.dim() == 2 and w.shape[1] % block_size == 0
        assert w.dtype in (torch.half, torch.bfloat16)
        w = w.to(torch.half)
        nb = w.shape[1] // block_size
        wb = w.view(w.shape[0], nb, block_size)
        d = wb.abs().amax(dim = 2) / 127.0
        d = torch.where(d == 0, torch.ones_like(d), d)
        q = torch.round(wb / d.unsqueeze(2)).clamp(-127, 127).to(torch.int8)
        self.quant = {
            "weight_i8": q.view(w.shape).contiguous(),
            "scale": d.contiguous(),
        }
        self.embedding = None