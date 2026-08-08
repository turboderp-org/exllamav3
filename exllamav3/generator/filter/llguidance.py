from __future__ import annotations
from .filter import Filter
from ...tokenizer import Tokenizer
import torch
import numpy as np
import json
from functools import lru_cache

try:
    from llguidance import LLTokenizer, LLMatcher, grammar_from
    llguidance_available = True
except (ModuleNotFoundError, ImportError):
    llguidance_available = False
    LLTokenizer = LLMatcher = grammar_from = None


@lru_cache(10)
def create_ll_tokenizer(tokenizer: Tokenizer) -> LLTokenizer:
    """
    Build (and cache) an LLTokenizer from the underlying HF tokenizer. Construction takes on the
    order of a second for a 150k vocabulary, subsequent filters reuse the cached instance.
    """
    eos = None
    cfg = tokenizer.config
    if getattr(cfg, "eos_token_id_list", None):
        eos = list(cfg.eos_token_id_list)
    elif tokenizer.eos_token_id is not None:
        eos = tokenizer.eos_token_id
    return LLTokenizer(tokenizer.tokenizer.to_str(), eos_token = eos)


class LLGuidanceFilter(Filter):

    def __init__(
        self,
        tokenizer: Tokenizer,
        trigger_token: int | None = None,
        prefix_str: str | None = None,
        eos_after_completed: bool = False,
        json_schema: dict | str | None = None,
        regex: str | None = None,
        lark_grammar: str | None = None,
        gbnf_grammar: str | None = None,
        llg_grammar: str | None = None,
        consume_prefix: bool = False,
    ):
        """
        Constrained sampling filter backed by llguidance. Exactly one grammar source must be given:

        :param json_schema:
            JSON schema as dict or string, see https://github.com/guidance-ai/llguidance/blob/main/docs/json_schema.md

        :param regex:
            Regular expression, Rust regex syntax (no lookaround/backreferences)

        :param lark_grammar:
            Context-free grammar in Lark syntax, see https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md

        :param gbnf_grammar:
            Context-free grammar in llama.cpp GBNF syntax (converted to Lark internally)

        :param llg_grammar:
            Raw llguidance grammar definition (JSON string), for advanced use

        :param consume_prefix:
            If True, prefix_str is tokenized and consumed by the matcher when the filter (re)starts,
            i.e. the grammar is expected to match prefix_str + sampled tokens. If False (default),
            the grammar only constrains sampled tokens.

        Remaining parameters are as for Filter.
        """
        if not llguidance_available:
            raise ValueError("llguidance package is not available.")

        super().__init__(tokenizer, trigger_token, prefix_str, eos_after_completed)

        sources = [
            ("json_schema", json_schema),
            ("regex", regex),
            ("lark", lark_grammar),
            ("gbnf", gbnf_grammar),
            ("llguidance", llg_grammar),
        ]
        given = [(f, g) for f, g in sources if g is not None]
        assert len(given) == 1, \
            "Specify exactly one of json_schema, regex, lark_grammar, gbnf_grammar, llg_grammar"
        g_format, g_text = given[0]
        if isinstance(g_text, dict):
            g_text = json.dumps(g_text)

        self._ll_tokenizer = create_ll_tokenizer(tokenizer)
        self._grammar = grammar_from(g_format, g_text)
        err = LLMatcher.validate_grammar(self._grammar, self._ll_tokenizer)
        if err:
            raise ValueError(f"Invalid grammar: {err}")

        self._matcher = LLMatcher(self._ll_tokenizer, self._grammar)
        self._consume_prefix = consume_prefix
        self._consumed = 0
        self._bitmask = np.empty(((self._ll_tokenizer.vocab_size + 31) // 32,), dtype = np.int32)
        self._bitmask_torch = torch.from_numpy(self._bitmask).unsqueeze(0)
        self._start()

    def _start(self):
        if self._consume_prefix and self.prefix_str:
            tokens = self._ll_tokenizer.tokenize_str(self.prefix_str)
            self._matcher.consume_tokens(tokens)
        self._check_error()

    def _check_error(self):
        if self._matcher.is_error():
            raise RuntimeError(f"llguidance matcher error: {self._matcher.get_error()}")

    def reset(self):
        # A matcher error state survives reset(), so recover by recreating the matcher (cheap, the
        # compiled grammar is reused)
        if self._matcher.is_error():
            self._matcher = LLMatcher(self._ll_tokenizer, self._grammar)
        else:
            self._matcher.reset()
        self._consumed = 0
        self._start()

    def accept_token(self, token: int):
        if self._matcher.is_stopped():
            return
        if self._matcher.consume_token(token):
            self._consumed += 1
        self._check_error()

    def rollback_tokens(self, num_tokens: int) -> bool:
        # Tokens consumed on an already-stopped matcher are not recorded in its rollback history, so
        # only roll back tokens known to have been consumed; anything else falls back to the journal
        # replay in Filter.rewind(). Rolling back more tokens than the matcher consumed would leave
        # it in an unrecoverable error state.
        if num_tokens > self._consumed:
            return False
        self._matcher.rollback(num_tokens)
        if self._matcher.is_error():
            self._matcher = LLMatcher(self._ll_tokenizer, self._grammar)
            self._consumed = 0
            return False
        self._consumed -= num_tokens
        return True

    def get_next_logit_mask(self) -> torch.Tensor:
        # Returned as the packed int32 bitmask llguidance computes natively (32 tokens per word,
        # bit clear = masked out). The buffer is reused for the next mask once the job has
        # consumed this one, like the pinned mask buffers downstream.
        bm = self._bitmask
        self._matcher.unsafe_compute_mask_ptr(bm.ctypes.data, bm.size * bm.itemsize)
        self._check_error()
        return self._bitmask_torch

    def is_completed(self) -> bool:
        return self._matcher.is_stopped()

    def get_captures(self) -> dict:
        """
        Return any named captures (Lark rules like `name: /regex/` referenced with `cap_name` etc.,
        see llguidance docs) collected so far.
        """
        return self._matcher.get_captures()
