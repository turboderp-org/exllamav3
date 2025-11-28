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
        imp: dict | None = None
    ):
        """
        :param embeddings:
            Embeddings, shape (num_tokens, input_dim)

        :param token_string:
            Tokenized representation, with -1 as a placeholder for the MM embeddings

        :param text_alias:
            Text string to represent this embedding for tokenizing
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

        # not exported for TP
        r = torch.arange(self.first_index, self.first_index + self.mm_length, dtype = torch.long)
        m = (token_string == -1)
        token_string.masked_scatter_(m, r)
        self.token_string = token_string
        self.token_list = token_string[0].tolist()


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