"""Graph replay for single-token decode batches with a fixed device-side cache layout."""

from dataclasses import dataclass
import logging
from weakref import ref

import torch

from ..constants import PAGE_SIZE
from ..util.tensor import g_tensor_cache, get_for_device

logger = logging.getLogger(__name__)


@dataclass
class _DecodeGraphEntry:
    hidden: torch.Tensor
    lengths: torch.Tensor
    blocks: torch.Tensor
    slots: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None
    logits: torch.Tensor | None = None

    @classmethod
    def allocate(cls, hidden, pages, device):
        batch_size = hidden.shape[0]
        return cls(
            torch.empty_like(hidden, device = device),
            torch.empty((batch_size,), dtype = torch.int32, device = device),
            torch.zeros((batch_size, pages), dtype = torch.int32, device = device),
            torch.empty((batch_size,), dtype = torch.int32, device = device),
        )

    def stage(self, hidden, params, device):
        self.hidden.copy_(hidden, non_blocking = True)
        self.lengths.copy_(get_for_device(params, "cache_seqlens", device))
        blocks = get_for_device(params, "block_table", device)
        # The generator pads its staging table even when the cache has fewer pages.
        pages = min(blocks.shape[1], self.blocks.shape[1])
        self.blocks[:, :pages].copy_(blocks[:, :pages])
        self.slots.copy_(get_for_device(params, "recurrent_slots", device))

    def run(self, model, cache):
        return model.run_section(self.hidden, {
            "cache": cache,
            "cache_seqlens": self.lengths,
            "block_table": self.blocks,
            "recurrent_slots": self.slots,
        }, 1, len(model.fwd_modules), False)


class DecodeGraph:
    """Graph the decoder after embedding. Model-specific callers check replay eligibility.

    Graphs borrow the live KV and recurrent pools. The first step in a bucket
    runs eagerly with the padded tables, warming the same kernel shapes that
    will be captured on the next step. Capture itself records the launches;
    replay executes them once for the current token.
    """

    def __init__(self):
        self._cache = None
        self._scratch_generation = None
        self._entries = {}

    def forward(self, model, input_ids, params, device):
        if torch.cuda.is_current_stream_capturing():
            return None

        cache = params["cache"]
        generation = g_tensor_cache.generation
        if self._cache is None or self._cache() is not cache or self._scratch_generation != generation:
            self._entries.clear()
            self._cache = ref(cache)
            self._scratch_generation = generation

        pages = params["block_table"].shape[1]
        lengths = params["cache_seqlens"]
        if lengths.device.type == "cpu":
            pages = min(pages, int(lengths.max()) // PAGE_SIZE + 1)
        # Bound the number of private graph pools while keeping short-context
        # attention launches small. Unused entries are masked by cache_seqlens.
        bucket = 4 ** (((pages - 1).bit_length() + 1) // 2)
        bucket = min(bucket, cache.max_num_tokens // PAGE_SIZE)
        key = (input_ids.shape[0], bucket)
        if key in self._entries and self._entries[key] is None:
            return None

        hidden = model.run_section(input_ids, params, 0, 1, False)
        if key not in self._entries:
            entry = _DecodeGraphEntry.allocate(hidden, bucket, device)
            self._entries[key] = entry
            entry.stage(hidden, params, device)
            return entry.run(model, cache)

        entry = self._entries[key]
        entry.stage(hidden, params, device)
        if entry.graph is None:
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, capture_error_mode = "thread_local"):
                    logits = entry.run(model, cache)
                entry.graph, entry.logits = graph, logits
            except RuntimeError as error:
                logger.debug("Decode graph capture failed, using eager decoding: %s", error)
                # A failed capture only affects this batch size and page bucket.
                self._entries[key] = None
                return model.run_section(hidden, params, 1, len(model.fwd_modules), False)

        entry.graph.replay()
        # Model.forward returns independent logits even when a later step reuses
        # the graph's output buffer.
        return entry.logits.clone()
