from __future__ import annotations
from typing_extensions import override
import torch
from ..model.config import Config
from ..ext import exllamav3_ext as ext
from . import Module
from .row_table import RowTable, PREFETCH_MIN_TOKENS
from .quant.exl3_lib.ngram_codec import ROW_DIM, mul1_codebook, dequant_rows, words_per_row

"""
Hashed n-gram embedding table (Qwen3.8-Flash-Next ple_embedding and kin): maps each token position
to (ngram_size - 1) * heads_per_ngram hash-table rows and concatenates them into one feature vector.

The table is enormous (tens of billions of parameters), so streaming is a first-class mode: by
default the table's tensors are never loaded, and the module gathers only the rows a forward
pass actually touches (see RowTable). The module is quantization-agnostic:

    <key>.trellis                  -> exl3_ngram_trellis format (util/convert_ngram.py)
    <key>.weight / .shard_N.weight -> unquantized source table

crossed with resident (RAM) or streamed (disk) storage gives four load modes, all returning
identical results for the same table contents.
"""


def _find_nth_prime_after(start: int, count: int) -> int:
    # mirrors the reference implementation used to derive the per-head vocab sizes
    def is_prime(v):
        if v < 2: return False
        if v % 2 == 0: return v == 2
        for d in range(3, int(v ** 0.5) + 1, 2):
            if v % d == 0: return False
        return True
    p = start
    for _ in range(count):
        p += 1
        while not is_prime(p):
            p += 1
    return p


class NGramEmbedding(Module):

    def __init__(
        self,
        config: Config | None,
        key: str,
        ngram_size: int,
        heads_per_ngram: int,
        ple_embed_dim: int,
        eos_token_id: int,
        stream_from_disk: bool | None = None,
        out_dtype: torch.dtype | None = torch.half,
        qmap: str | None = None,
    ):
        super().__init__(config, key, None)
        assert qmap is None, "NGramEmbedding quantizes via util/convert_ngram.py, not the qmap pipeline"

        self.ngram_size = ngram_size
        self.context_len = ngram_size - 1
        self.heads_per_ngram = heads_per_ngram
        self.num_heads = (ngram_size - 1) * heads_per_ngram
        self.ple_embed_dim = ple_embed_dim
        self.head_dim = ple_embed_dim // self.num_heads
        assert self.head_dim == ROW_DIM, f"expected {ROW_DIM}-D embedding rows, got {self.head_dim}"
        self.eos_token_id = eos_token_id
        # None: defer to config.infer_params.ngram_stream_from_disk at load time (the load-time
        # option; also EXL3_NGRAM_STREAM). An explicit bool here overrides it
        self.stream_from_disk = stream_from_disk
        self.out_dtype = out_dtype

        self.table = None
        self.K = None               # None for an unquantized table
        self.num_rows = 0
        self.head_bias = None
        self.head_offsets = None
        self.head_vocab_sizes = None
        self.layer_multipliers = None
        self.codebook = None

        self.caps.update({"prefer_cpu": True})

    @property
    def mode(self):
        """"trellis_disk" | "trellis_ram" | "fp16_disk" | "fp16_ram", None when not loaded"""
        if self.table is None:
            return None
        return ("trellis" if self.K else "fp16") + ("_disk" if self.table.on_disk else "_ram")

    @property
    def prefetch_stats(self):
        return self.table.prefetch_stats

    @override
    def optimizer_targets(self):
        return []

    def _load_aux(self, names: dict[str, str]):
        stc = self.config.stc
        def get(name, optional = False):
            # no_defer: these are consumed (copied) during load(), before a deferred load pass
            # would have filled them
            t = stc.get_tensor(name, "cpu", optional = True, allow_bf16 = True, no_defer = True)
            if t is None and not optional:
                raise ValueError(f"Required tensor {name} not found for {self.key}")
            return t
        # The hashing runs on the CPU (where the token ids live), so the hash parameters stay
        # host-side; only the dequant bias goes to the device
        self.head_offsets = get(names["offsets"]).long().contiguous()
        self.head_vocab_sizes = get(names["sizes"]).long().contiguous()
        self.layer_multipliers = get(names["multipliers"]).long().contiguous()
        bias = get(names["bias"], optional = True) if "bias" in names else None
        self.head_bias = bias.half().contiguous().to(self.device) if bias is not None else None
        assert self.head_offsets.shape[0] == self.num_heads
        # These were buffered reads of the table file. On Windows a buffered file object (even a
        # closed one, for a few seconds) throttles the unbuffered row gathers on the same file, so
        # release the loader's handle now rather than at the end of the load
        for f in {stc.tensor_file_map[n] for n in names.values() if n in stc.tensor_file_map}:
            stc.release_file(f)

    @override
    def load(self, device: torch.device, **kwargs):
        self.device = device
        stc = self.config.stc
        parent = self.key.rsplit(".", 1)[0]

        table = RowTable.find(stc, self.key, "trellis")
        quantized = table is not None
        if quantized:
            self._load_aux({
                "offsets": f"{self.key}.head_offsets",
                "sizes": f"{self.key}.head_vocab_sizes",
                "multipliers": f"{self.key}.layer_multipliers",
                "bias": f"{self.key}.head_bias",
            })
            self.codebook = mul1_codebook(device)
        else:
            # unquantized source table
            table = RowTable.find(stc, self.key, "weight")
            if table is None:
                raise ValueError(f"No .trellis, .weight or .shard_N.weight tensors found for {self.key}")
            self._load_aux({
                "offsets": f"{parent}.ngram_heads_offsets",
                "sizes": f"{parent}.ngram_heads_vocab_sizes",
                "multipliers": f"{parent}.layer_multipliers",
            })

        infer_params = getattr(self.config, "infer_params", None)
        lock = infer_params is not None and infer_params.ngram_lock
        stream_from_disk = self.stream_from_disk
        if stream_from_disk is None:
            stream_from_disk = infer_params.ngram_stream_from_disk if infer_params is not None else True
        table.open(stc, stream_from_disk and not lock, allow_bf16 = not quantized,
                   what = f"n-gram table {self.key} held in RAM (--ngram_ram)", lock = lock)
        self._set_table(table, quantized)

    def _set_table(self, table: RowTable, quantized: bool):
        self.table = table
        self.num_rows = table.num_rows
        self.K = (table.row_words - 1) * 16 // ROW_DIM if quantized else None
        assert table.row_words == (words_per_row(self.K) if quantized else ROW_DIM)

    @override
    def unload(self):
        if self.table is not None:
            self.table.close()
        self.table = None
        self.device = None
        self.K = None
        self.head_bias = None
        self.head_offsets = None
        self.head_vocab_sizes = None
        self.layer_multipliers = None
        self.codebook = None

    @override
    def get_tensors(self):
        # The table is never resident as a whole in the general case; export/compile of this
        # module is handled by the conversion pipeline (util/convert_ngram.py), not here
        return {}

    @override
    def weights_numel(self):
        return self.num_rows * ROW_DIM

    def tp_export(self, plan, producer):
        """
        Tensor-parallel: every rank streams rows from disk through its own handles (a per-rank
        RAM copy of a table this size is not an option, and the gather is a few hundred rows per
        forward), so the export carries the table's location plus the small hashing/dequant
        parameters.
        """
        assert self.table is not None, "Cannot export module for TP before loading."
        return {
            "cls": NGramEmbedding,
            "kwargs": {
                "key": self.key,
                "ngram_size": self.ngram_size,
                "heads_per_ngram": self.heads_per_ngram,
                "ple_embed_dim": self.ple_embed_dim,
                "eos_token_id": self.eos_token_id,
                "out_dtype": self.out_dtype,
            },
            "table": self.table.export(self.config.stc),
            "quantized": self.K is not None,
            "head_offsets": producer.send(self.head_offsets),
            "head_vocab_sizes": producer.send(self.head_vocab_sizes),
            "layer_multipliers": producer.send(self.layer_multipliers),
            "head_bias": producer.send(self.head_bias) if self.head_bias is not None else None,
            "device": self.device,
        }

    @staticmethod
    def tp_import(local_context, exported, plan):
        consumer = local_context["consumer"]
        device = local_context["device"]
        module = NGramEmbedding(config = None, **exported["kwargs"], stream_from_disk = True)
        module.device = device
        module._set_table(RowTable.from_export(exported["table"]), exported["quantized"])
        module.head_offsets = consumer.recv(exported["head_offsets"], cuda = False).long().contiguous()
        module.head_vocab_sizes = consumer.recv(exported["head_vocab_sizes"], cuda = False).long().contiguous()
        module.layer_multipliers = consumer.recv(exported["layer_multipliers"], cuda = False).long().contiguous()
        module.head_bias = consumer.recv(exported["head_bias"], cuda = True) if exported.get("head_bias") is not None else None
        if module.K:
            module.codebook = mul1_codebook(device)
        return module

    def fetch_rows(self, uids: torch.Tensor, out_dtype: torch.dtype = torch.half) -> torch.Tensor:
        """Unique row indices (any device) -> decoded (N, 160) rows on the module's device.
        Reference form of the row pipeline (torch codec); forward() runs the fast path."""
        raw = self.table.fetch(uids.to("cpu", torch.int64)).to(self.device)
        if self.K:
            heads = (torch.searchsorted(self.head_offsets.to(self.device),
                                        uids.to(self.device, torch.int64),
                                        right = True) - 1).clamp(0, self.num_heads - 1)
            rows = dequant_rows(raw, self.K, self.codebook, self.head_bias.float()[heads])
        else:
            rows = raw.float()
        return rows.to(out_dtype)

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        # mirrors the reference implementation: n-grams never span an eos boundary; positions
        # whose shifted source would cross one read eos instead
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device = token_ids.device)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim = 1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim = 1)
        position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim = 1, index = gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def compute_ngram_ids(self, token_history: torch.Tensor, out_len: int) -> torch.Tensor:
        """
        token_history: (bsz, context + seq_len) token ids including the (ngram_size - 1) tokens
        preceding the sequence (eos-padded at the start of a new sequence).
        Returns (bsz, out_len, num_heads) global table row indices for the last out_len positions.
        """
        th = token_history.long()
        shifted = [self._shift_right_ignore_eos(th, s) for s in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            lo = (ngram - 2) * self.heads_per_ngram
            hi = lo + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0].to(th.device)
            for position in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[position] * self.layer_multipliers[position].to(th.device))
            sizes = self.head_vocab_sizes[lo:hi].to(th.device)
            offsets = self.head_offsets[lo:hi].to(th.device)
            blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes.view(1, 1, -1)) + offsets.view(1, 1, -1))
        return torch.cat(blocks, dim = -1)[:, -out_len:]

    def embed_ids(self, ngram_ids: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """(bsz, seq_len, num_heads) row indices -> (bsz, seq_len, ple_embed_dim) on device.
        Reference form; forward() runs the fast path."""
        bsz, seq_len, H = ngram_ids.shape
        flat = ngram_ids.reshape(-1)
        uids, inverse = torch.unique(flat, return_inverse = True)
        rows = self.fetch_rows(uids, out_dtype or self.out_dtype or torch.half)
        out = rows[inverse.to(self.device)]
        return out.view(bsz, seq_len, H * ROW_DIM)

    def forward_reference(self, x: torch.Tensor, params: dict,
                          out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Pure-torch reference pipeline (hashing + codec), kept for tests and A/B."""
        out_len = x.shape[1] - self.context_len
        ngram_ids = self.compute_ngram_ids(x, out_len)
        return self.embed_ids(ngram_ids, out_dtype)

    # ---- fast path -----------------------------------------------------------------------------
    #
    # Hashing, eos segmentation and dedup run in one C++ call on the CPU; the table stages the
    # unique rows and the GPU trellis dequant kernel (or a plain upcast for unquantized tables)
    # decodes them. The model stages each chunk before its first layers are issued (prefetch()),
    # so a cold gather overlaps block 0 instead of stalling at this layer. Decode-sized inputs
    # stay inline: their 16-row gathers are already parallel preads.

    def _resolve(self, history: torch.Tensor, pin) -> tuple:
        U = ext.ngram_hash_cpu(
            history, history.shape[1] - self.context_len,
            self.layer_multipliers, self.head_offsets, self.head_vocab_sizes,
            self.heads_per_ngram, self.eos_token_id,
            pin.uids, pin.inverse, pin.heads)
        return pin.uids[:U], False

    def _decode(self, packed: torch.Tensor, pin, U: int) -> torch.Tensor:
        if not self.K:
            return packed.float()
        heads = pin.heads[:U].to(packed.device, non_blocking = True)
        rows = torch.empty((U, ROW_DIM), dtype = torch.half, device = packed.device)
        ext.ngram_dequant(packed, self.K, heads, self.head_bias, rows, False)
        return rows

    def prefetch(self, history: torch.Tensor):
        """
        Stage the rows for a coming forward over `history`, the exact (bsz, context + seq) id
        history that forward() will receive, on a worker thread. Decode-sized inputs are ignored.
        """
        if self.table is None or history.dim() != 2:
            return
        n = history.shape[0] * (history.shape[1] - self.context_len)
        if n >= PREFETCH_MIN_TOKENS:
            self.table.prefetch(history, n * self.num_heads, self._resolve)

    @override
    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """
        x: (bsz, context + seq_len) token history (CPU in the hot path); returns embeddings for
        the last seq_len = x.shape[1] - context_len positions, on the module's device.
        """
        out_len = x.shape[1] - self.context_len
        ids = x.to("cpu", torch.int64).contiguous()
        bsz = ids.shape[0]
        H = self.num_heads
        out = self.table.lookup(ids, bsz * out_len * H, self._resolve, self._decode, self.device)
        return out.view(bsz, out_len, H * ROW_DIM).to(out_dtype or self.out_dtype or torch.half)
