from __future__ import annotations
import torch

FIRST_MM_EMBEDDING_INDEX = 1000000000

# Assume no model will have more than one billion regular text tokens, and assign dynamic token IDs starting from
# that index.

class MMTokenAllocator:

    next_token_index: int

    def __init__(self):
        self.next_token_index = FIRST_MM_EMBEDDING_INDEX

    def allocate(self, num_tokens):
        idx = self.next_token_index
        self.next_token_index += num_tokens
        return idx

global_allocator = MMTokenAllocator()


class MMEmbedding:
    """
    Container for one embedding (image etc.) and associated metadata
    """

    def __init__(
        self,
        embeddings: torch.Tensor | None = None,
        token_string: torch.Tensor | None = None,
        text_alias: str | None = None,
        deepstack_embeddings: list[torch.Tensor] | None = None,
        grid_thw: tuple | None = None,
        mrope_merge_size: int | None = None,
        align: int = 0,
        align_phase: int = 0,
        align_lead: int = 0,
        imp: dict | None = None
    ):
        """
        :param embeddings:
            Embeddings, shape (num_tokens, input_dim)

        :param token_string:
            Tokenized representation, with -1 as a placeholder for the MM embeddings

        :param text_alias:
            Text string to represent this embedding for tokenizing

        :param align:
            Position alignment for embeddings whose block must start on a fixed phase of a
            model-side grouping (DeepSeek-V4: the CSA compressor's 4-token groups). The
            token_string then begins with align_lead spare rows (pads) and the tokenizer emits
            only the suffix that places row align_lead at a prompt position p with
            p % align == align_phase. 0 = no alignment.
        """

        if imp:
            self.metadata = imp["metadata"]
            self.full_length = imp["full_length"]
            self.mm_length = imp["mm_length"]
            self.first_index = imp["first_index"]
            self.last_index = imp["last_index"]
            self.text_alias = imp["text_alias"]
            self.grid_thw = imp["grid_thw"]
            self.mrope_merge_size = imp["mrope_merge_size"]
            self.embeddings = imp["embeddings"]
            self.deepstack_embeddings = imp["deepstack_embeddings"]
            self.align = imp.get("align", 0)
            self.align_phase = imp.get("align_phase", 0)
            self.align_lead = imp.get("align_lead", 0)
            self.token_string = None
            self.token_list = None
            return

        global global_allocator

        if deepstack_embeddings is not None:
            assert all(de.shape == embeddings.shape for de in deepstack_embeddings), \
                "Deepstack embeddings shape mismatch"

        self.metadata = {}
        self.full_length = token_string.shape[-1]
        self.mm_length = embeddings.shape[-2]
        self.first_index = global_allocator.allocate(self.mm_length)
        self.last_index = self.first_index + self.mm_length
        self.embeddings = embeddings
        self.deepstack_embeddings = deepstack_embeddings
        self.text_alias = text_alias or f"<$EMB_{self.first_index}$>"

        # MRoPE
        self.grid_thw = grid_thw
        self.mrope_merge_size = mrope_merge_size

        # Position alignment (see align)
        assert align == 0 or (0 <= align_phase < align and 0 <= align_lead < align and align_lead < self.full_length), \
            "MMEmbedding: align_lead / align_phase must be smaller than align"
        self.align = align
        self.align_phase = align_phase
        self.align_lead = align_lead

        # not exported for TP
        r = torch.arange(self.first_index, self.first_index + self.mm_length, dtype = torch.long)
        m = (token_string == -1)
        token_string.masked_scatter_(m, r)
        self.token_string = token_string
        self.token_list = token_string[0].tolist()

    def token_list_at(self, position: int) -> list[int]:
        """Token ids to splice in when the embedding's first emitted row lands at prompt
        position `position`: the whole token_list, or, for aligned embeddings, the suffix that
        skips the spare leading rows not needed to put row align_lead on the required phase."""
        if not self.align:
            return self.token_list
        n_pad = (self.align_phase - position) % self.align
        assert n_pad <= self.align_lead
        return self.token_list[self.align_lead - n_pad:]


def send_embeddings(producer, ies: list[MMEmbedding]):
    return {
        "method": "list",
        "data": [
            {
                "metadata": ie.metadata,
                "full_length": ie.full_length,
                "mm_length": ie.mm_length,
                "first_index": ie.first_index,
                "last_index": ie.last_index,
                "text_alias": ie.text_alias,
                "grid_thw": ie.grid_thw,
                "mrope_merge_size": ie.mrope_merge_size,
                "align": ie.align,
                "align_phase": ie.align_phase,
                "align_lead": ie.align_lead,
                "embeddings": producer.send(ie.embeddings, cache_id = id(ie.embeddings)),
                "deepstack_embeddings": [
                    producer.send(dse, cache_id = id(dse))
                    for dse in ie.deepstack_embeddings
                ] if ie.deepstack_embeddings is not None else None
            }
            for ie in ies
        ]
    }


def recv_embeddings(consumer, recv) -> list[MMEmbedding]:
    result = []
    assert recv.get("method") == "list", "Consumer expected list"
    for imp in recv["data"]:
        imp["embeddings"] = consumer.recv(imp["embeddings"])
        imp["deepstack_embeddings"] = [
            consumer.recv(dse) for dse in imp["deepstack_embeddings"]
        ] if imp.get("deepstack_embeddings") else None
        result.append(MMEmbedding(imp = imp))
    return result