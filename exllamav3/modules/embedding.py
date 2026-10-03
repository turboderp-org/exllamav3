from __future__ import annotations
from typing_extensions import override
import torch
from torch import nn
from ..model.config import Config
from ..util.tensor import to2
from ..ext import exllamav3_ext as ext
from . import Module
from .row_table import RowTable
from .quant.exl3_lib.ngram_codec import GROUP_DIM, mul1_codebook, dequant_rows
from ..tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
from ..model.model_tp_alloc import TPAllocation

"""
Token embedding table. Three storage forms, all behind the same lookup:

    <key>.weight                   -> resident nn.Embedding (the default)
    <key>.weight, streamed         -> rows gathered from disk per forward (RowTable)
    <key>.trellis + <key>.signs    -> quantized table (see ngram_codec), held in RAM or streamed,
                                      decoded on the device
"""

DEDUP_MIN_TOKENS = 16

class TableEmbedding:
    """Stands in for nn.Embedding when the table is a RowTable"""

    def __init__(self, table: RowTable, hidden_size: int, device: torch.device, signs: torch.Tensor | None):
        self.table = table
        self.hidden_size = hidden_size
        self.device = device = torch.device(device)     # (TP ranks name their device by index)
        self.signs = signs
        if signs is not None:
            self.groups = signs.shape[0]
            self.K = (table.row_words // self.groups - 1) // 16
            assert table.row_words == self.groups * (1 + 16 * self.K) and self.groups * GROUP_DIM >= hidden_size
            self.codebook = mul1_codebook(device) if device.type != "cuda" else None

    def resolve(self, ids: torch.Tensor, pin) -> tuple:
        # Decode-sized lookups gather their rows as they come; chunks gather each row once
        if ids.numel() <= DEDUP_MIN_TOKENS and len(self.table.stores) == 1:
            return ids.view(-1), True
        uids, inverse = torch.unique(ids, return_inverse = True)
        pin.inverse[:ids.numel()] = inverse.view(-1)
        return uids, False

    def decode(self, rows: torch.Tensor, pin, U: int) -> torch.Tensor:
        if self.signs is None:
            return rows if rows.dtype == torch.bfloat16 else rows.half()
        rings = rows.view(U * self.groups, -1)
        if rows.is_cuda:
            out = torch.empty((U * self.groups, GROUP_DIM), dtype = torch.half, device = rows.device)
            ext.ngram_dequant(rings, self.K, None, self.signs, out, True)
        else:
            out = dequant_rows(rings, self.K, self.codebook, signs = self.signs.repeat(U, 1)).half()
        out = out.view(U, -1)
        return out if out.shape[1] == self.hidden_size else out[:, :self.hidden_size]

    def forward(self, ids: torch.Tensor, synced: bool = False) -> torch.Tensor:
        shape = ids.shape
        ids = ids.to("cpu", torch.int64).contiguous()
        out = self.table.lookup(ids, ids.numel(), self.resolve, self.decode, self.device, synced)
        return out.view(*shape, self.hidden_size)

    __call__ = forward


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
        multiplier: float = 1.0,
        allow_table: bool = True,
    ):
        super().__init__(config, key, None)
        assert qmap is None, "No quant scheme for Embedding"

        self.key = key
        self.embedding = None
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.out_dtype = out_dtype
        self._pinned_staging = {}
        self._numel = vocab_size * hidden_size
        self.normalize = normalize
        self.multiplier = multiplier
        # False for tables whose weight is read directly (always a resident nn.Embedding)
        self.allow_table = allow_table
        self.compiled = None

        # A quantized table is decoded on the device, from ids that stay on the CPU
        quantized = allow_table and config is not None and config.stc.has_tensor(key + ".trellis")
        self.caps.update({
            "prefer_cpu": not quantized,
            "x_cpu": quantized,
        })

    @override
    def optimizer_targets(self):
        return []

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        stc = self.config.stc
        table = RowTable.find(stc, self.key, "trellis") if self.allow_table else None
        infer_params = getattr(self.config, "infer_params", None)
        stream_from_disk = self.allow_table and infer_params is not None and infer_params.embed_stream_from_disk
        if table is not None:
            signs = stc.get_tensor(self.key + ".signs", device, no_defer = True)
            table.open(stc, stream_from_disk)
        elif stream_from_disk and stc.get_tensor_handle(self.key + ".weight", optional = True) is not None:
            signs = None
            table = RowTable.find(stc, self.key, "weight")
            table.open(stc, True)
        if table is not None:
            self.embedding = TableEmbedding(table, self.hidden_size, device, signs)
            return
        weight = stc.get_tensor(self.key + ".weight", self.device, float2half = True, allow_bf16 = True)
        self._numel = weight.numel()
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            device = "meta"
        )
        self.embedding.weight = nn.Parameter(weight, requires_grad = False)

    @override
    def unload(self):
        if isinstance(self.embedding, TableEmbedding):
            self.embedding.table.close()
        self.device = None
        self.embedding = None

    @override
    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        return x if isinstance(self.embedding, TableEmbedding) else super().prepare_for_device(x, params)

    @override
    def get_tensors(self):
        if isinstance(self.embedding, TableEmbedding):
            table = self.embedding.table
            assert not table.on_disk and self.embedding.signs is not None
            return {
                f"{self.key}.trellis": table.stores[0],
                f"{self.key}.signs": self.embedding.signs,
            }
        return {
            f"{self.key}.weight": self.embedding.weight.data.contiguous()
        }

    @override
    def get_compile_sizes(self, stc):
        if self.compiled is not None:
            return [v.numel() * v.element_size() for v in self.compiled.values()]
        return super().get_compile_sizes(stc)

    @override
    def get_compile_tensors(self, stc):
        return self.compiled if self.compiled is not None else super().get_compile_tensors(stc)

    def prefetch_tokens(self, ids: list[int]):
        """Tokens of a coming forward, known ahead of it: start reading their rows"""
        if isinstance(self.embedding, TableEmbedding):
            self.embedding.table.advise(ids)

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
                    standard_emb_row = self.embedding(standard_ids_row)
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
                    combined_emb[i][im[i]] = ie.embeddings[indexed_ids_row].to(combined_emb.device, out_dtype)

                    # Prepare deepstack embeddings
                    if ie.deepstack_embeddings is not None:
                        for layer, de in enumerate(ie.deepstack_embeddings):
                            deepstack_emb[layer][i][im[i]] = de[indexed_ids_row].to(combined_emb.device, out_dtype)

            # Save deepstack embeddings to params
            if deepstack_emb is not None:
                params["deepstack_emb"] = deepstack_emb

            return combined_emb

        # No indexed embeddings, or none in current batch
        else:
            if isinstance(self.embedding, TableEmbedding):
                # pinned_staging is a buffer-reuse request, NOT a sync guarantee
                # (on master it only selects the pinned staging buffer; the port's
                # prefill sets it precisely because chunks run back-to-back with
                # no sync - see the resident-path comment below). RowTable.synced
                # asserts the opposite ("caller guarantees a sync point before the
                # next lookup"), and feeding pinned_staging into it disabled the
                # pin-set event guard during prefill: with the uploads async, a
                # reused staging set could be rewritten on the host before its
                # H2D landed (torn table gather). lookup() must always record the
                # event; the synchronize on reuse is free when a sync did happen.
                x = self.embedding.forward(x, False)
            else:
                x = self.embedding.forward(x)
            if self.multiplier != 1.0:
                x *= self.multiplier
            x = to2(x, out_dtype, self.out_dtype)
            if self.normalize:
                x *= x.shape[-1] ** 0.5
            # When the embedding resides on the CPU, its output is uploaded to the first
            # device layer; staging it through a pinned buffer makes that upload
            # asynchronous. Prefill callers may enqueue back-to-back chunks without a
            # sync, so each buffer pair alternates and the writer waits on the event
            # recorded after its previous upload was submitted.
            if params.get("pinned_staging") and x.device.type == "cpu":
                key = (x.shape, x.dtype)
                entry = self._pinned_staging.get(key)
                if entry is None:
                    if len(self._pinned_staging) > 8:
                        self._pinned_staging.clear()
                    # (two pinned buffers, two events, index of the last-used
                    # buffer, whether that use still needs its event recorded)
                    entry = [torch.empty_like(x, pin_memory = True),
                             torch.empty_like(x, pin_memory = True),
                             torch.cuda.Event(), torch.cuda.Event(), 0, False]
                    self._pinned_staging[key] = entry
                # Record the event for the buffer used on the previous call: its
                # H2D upload was enqueued on this stream since then, so this
                # event completes only after that copy has been read out.
                if entry[5]:
                    entry[2 + entry[4]].record(torch.cuda.current_stream())
                i = entry[4] ^ 1
                entry[4] = i
                entry[5] = True
                buf, ev = entry[i], entry[2 + i]
                ev.synchronize()
                buf.copy_(x)
                x = buf
            return x

    def make_tp_allocation(self, options: dict) -> list[TPAllocation]:
        return []

    def tp_export(self, plan, producer):
        assert self.device is not None, "Cannot export module for TP before loading."
        exported = {
            "cls": Embedding,
            "kwargs": {
                "key": self.key,
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "out_dtype": self.out_dtype,
                "normalize": self.normalize,
                "multiplier": self.multiplier,
                "allow_table": self.allow_table,
            },
            "device": self.device
        }
        if isinstance(self.embedding, TableEmbedding):
            # Every rank streams the table's rows from disk through its own handles
            signs = self.embedding.signs
            exported["table"] = self.embedding.table.export(self.config.stc)
            exported["signs"] = producer.send(signs) if signs is not None else None
        else:
            exported["embedding.weight"] = producer.send(self.embedding.weight)
        return exported

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        module = Embedding(
            config = None,
            **exported["kwargs"],
        )
        if "table" in exported:
            signs = exported["signs"]
            module.device = local_context["device"] if signs is not None else exported["device"]
            module.embedding = TableEmbedding(
                RowTable.from_export(exported["table"]), module.hidden_size, module.device,
                consumer.recv(signs, cuda = True) if signs is not None else None)
            return module
        module.device = exported["device"]
        module.embedding = nn.Embedding(
            module.vocab_size,
            module.hidden_size,
            device = "meta"
        )
        emb = consumer.recv(exported["embedding.weight"], cuda = False)
        module.embedding.weight = nn.Parameter(emb, requires_grad = False)
        return module