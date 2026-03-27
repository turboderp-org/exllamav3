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

    # Some tokenizers (e.g. EXAONE BPE) have duplicate token strings: multiple
    # token IDs share the same piece text.  _get_vocab_dict() builds a
    # {str: int} dict comprehension where the last ID wins, so earlier IDs that
    # share a string are dropped from `vocab` and consequently from `new_vocab`.
    # When the model outputs one of these "lost" IDs, kbnf rejects it with
    # "The input token id is rejected", crashing generation.
    #
    # Fix: build a str->bytes lookup from new_vocab, then back-fill any token
    # ID absent from new_vocab by reusing the bytes of its duplicate string.
    id_to_str = {v: k for k, v in vocab.items()}  # {token_id: token_str}
    str_to_bytes = {
        id_to_str[id_]: b for id_, b in new_vocab.items() if id_ in id_to_str
    }
    full_size = tokenizer.tokenizer.get_vocab_size()
    for i in range(full_size):
        if i not in new_vocab:
            token_str = tokenizer.tokenizer.id_to_token(i)
            if token_str in str_to_bytes:
                new_vocab[i] = str_to_bytes[token_str]

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
        try:
            self._formatter.accept_token(token)
        except ValueError as e:
            # kbnf (Rust) raises ValueError when a token that passed the logit
            # mask is nonetheless rejected by the grammar engine's internal state
            # machine.  This indicates a mask/accept inconsistency inside kbnf —
            # compute_allowed_tokens() reported the token as valid, but
            # try_accept_new_token() disagrees.
            #
            # Known trigger: GPT2-style byte-level BPE tokenizers (e.g. EXAONE
            # 4.x) whose special tokens (e.g. [PAD], id=0) have literal-string
            # byte representations (b'[PAD]') that incidentally match valid
            # grammar byte sequences in certain states (e.g. inside a JSON string
            # value), causing the mask to permit them while kbnf's state machine
            # rejects them on accept.
            #
            # Re-raising propagates the error up through the exllamav3 generator
            # loop, which causes the in-flight job to be aborted cleanly instead
            # of crashing the calling process.
            #
            # This fallback remains safe and beneficial even after a kbnf-side fix:
            # it turns any future unforeseen mask/accept inconsistency into a
            # recoverable per-request error rather than a process crash.
            raise

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