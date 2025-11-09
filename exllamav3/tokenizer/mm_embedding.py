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
        embeddings: torch.Tensor,
        token_string: torch.Tensor,
        text_alias: str | None = None,
        deepstack_embeddings: list[torch.Tensor] | None = None,
        grid_thw: tuple | None = None,
        mrope_merge_size: int | None = None
    ):
        """
        :param embeddings:
            Embeddings, shape (num_tokens, input_dim)

        :param token_string:
            Tokenized representation, with -1 as a placeholder for the MM embeddings

        :param text_alias:
            Text string to represent this embedding for tokenizing
        """

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

        r = torch.arange(self.first_index, self.first_index + self.mm_length, dtype = torch.long)
        m = (token_string == -1)
        token_string.masked_scatter_(m, r)
        self.token_string = token_string
        self.token_list = token_string[0].tolist()