from .filter import Filter
from ...tokenizer import Tokenizer
import torch
from functools import lru_cache

try:
    import kbnf
    from formatron.integrations.utils import get_original_characters, default_mask_logits_fn, get_bit_mask
    from formatron.formatter import FormatterBuilder
    from formatron.config import EngineGenerationConfig
    formatron_available = True
except ModuleNotFoundError:
    formatron_available = False
except ImportError:
    formatron_available = False


@lru_cache(10)
def create_engine_vocabulary(
    tokenizer: Tokenizer,
    vocab_processors: list[callable] | None = None
) -> kbnf.Vocabulary:
    vocab = tokenizer.get_vocab_dict()
    new_vocab = get_original_characters(vocab, vocab_processors)
    return kbnf.Vocabulary(
        {k: kbnf.Token(v) for k, v in new_vocab.items()},
        {v: k for k, v in vocab.items()}
    )

class FormatronFilter(Filter):

    def __init__(
        self,
        tokenizer: Tokenizer,
        trigger_token: int | None = None,
        prefix_str: str | None = None,
        eos_after_completed: bool = False,
        formatter_builder: FormatterBuilder = None,
        engine_config: EngineGenerationConfig = None,
        vocab_processors: list[callable] | None = None
    ):
        if not formatron_available:
            raise ValueError("Formatron package is not available.")

        super().__init__(tokenizer, trigger_token, prefix_str, eos_after_completed)
        assert formatter_builder is not None
        self._formatter = formatter_builder.build(
            create_engine_vocabulary(tokenizer, vocab_processors),
            lambda tokens: tokenizer.tokenizer.decode(tokens)
        )
        self._config = engine_config or EngineGenerationConfig()
        if self._config.read_prompt:
            prompt = prefix_str.encode("utf-8")
            self._formatter.accept_bytes(prompt)
        self._zeros = None

    def reset(self):
        self._formatter.reset()

    def accept_token(self, token: int):
        if self._formatter.is_completed():
            return
        self._formatter.accept_token(token)

    def get_next_logit_mask(self) -> torch.Tensor:
        self._formatter.compute_allowed_tokens()
        if self._zeros is None:
            self._zeros = torch.zeros((self.vocab_size,), dtype = self.logits_dtype, device = "cpu")
        mask = self._formatter.mask_logits(self._zeros).unsqueeze(0)
        # mask_logits() sometimes modifies in-place, so create a new zeros tensor in that case
        # TODO: See if it's possible to get bit mask from Formatron instead (then apply with custom kernel)
        if mask.untyped_storage().data_ptr() == self._zeros.untyped_storage().data_ptr():
            self._zeros = None
        # self._debug(mask)
        return mask

    def is_completed(self) -> bool:
        return self._formatter.is_completed()

    def _debug(self, mask):
        allowed = (mask.squeeze(0) == 0).nonzero(as_tuple = False).tolist()
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        for i in allowed:
            print(i[0], repr(id_to_piece[i[0]]))
        pass