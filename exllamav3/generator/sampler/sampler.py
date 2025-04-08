from abc import abstractmethod
import torch
from ...tokenizer import Tokenizer

class Sampler:
    def __init__(self):
        pass

    @abstractmethod
    def forward(
        self,
        logits,
        sequence_ids: torch.Tensor | None = None,
        rand_u32: int | None = None,
        tokenizer: Tokenizer | None = None,
        blocked_tokens: list[int] | None = None,
        allowed_tokens: list[int] | None = None,
    ):
        pass