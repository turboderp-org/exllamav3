from ...tokenizer import Tokenizer
import torch

# Filter journal entry types, one entry per token fed while the job is generating. The journal makes
# filter state rewindable (banned strings): FJ_ACCEPT/FJ_COMPLETE mark tokens the underlying state
# machine consumed, FJ_TRIGGER marks (re)activation of a triggered filter, FJ_PASS marks tokens that
# passed through an inactive filter.
FJ_PASS = 0
FJ_TRIGGER = 1
FJ_ACCEPT = 2
FJ_COMPLETE = 3

class Filter:

    def __init__(
        self,
        tokenizer: Tokenizer,
        trigger_token: int | None,
        prefix_str: str | None,
        eos_after_completed: bool
    ):
        """
        :param tokenizer:
            Tokenizer

        :param trigger_token:
            Token that generator will look for before enabling this filter

        :param prefix_str:
            Initial string of characters that will be accepted by the filter before any sampling happens

        :param eos_after_completed:
            Make generator treat completing the filter as a stop condition. If False, filter will be deactivated
            after an end state is reached and sampling is unconstrained after that (or until the next trigger,
            upon which the filter is reset and reactivated.)
        """
        self.tokenizer = tokenizer
        self.trigger_token = trigger_token
        self.prefix_str = prefix_str
        self.eos_after_completed = eos_after_completed

        self.job = None
        self.generator = None
        self.vocab_size = None
        self.logits_dtype = torch.half

        self.is_active = False if trigger_token is not None else True
        self._journal = []

    def feed(self, token: int) -> bool:
        """
        Advance the filter on an emitted token: handles trigger activation, state machine advance and
        completion, and journals the event so the filter can be rewound later (banned strings).
        Returns True if the filter completed on this token and eos_after_completed is set.
        """
        if not self.is_active:
            if token == self.trigger_token:
                self.is_active = True
                self.reset()
                self._journal.append((FJ_TRIGGER, token))
            else:
                self._journal.append((FJ_PASS, token))
            return False
        self.accept_token(token)
        if self.is_completed():
            self.is_active = False
            self._journal.append((FJ_COMPLETE, token))
            return self.eos_after_completed
        self._journal.append((FJ_ACCEPT, token))
        return False

    def rewind(self, num_tokens: int):
        """
        Roll back the filter over the last num_tokens fed tokens, e.g. after a banned-string rewind.
        Uses the subclass's native rollback when available, otherwise rebuilds the state by replaying
        the retained journal through reset()/accept_token().
        """
        if num_tokens == 0:
            return
        assert num_tokens <= len(self._journal), \
            f"Cannot rewind filter by {num_tokens} tokens, only {len(self._journal)} journaled"
        popped = self._journal[-num_tokens:]
        del self._journal[-num_tokens:]
        if any(e == FJ_TRIGGER for e, _ in popped):
            # Rewound past a (re)activation; reconstruct from scratch
            self._rebuild()
            return
        n_accepted = sum(1 for e, _ in popped if e in (FJ_ACCEPT, FJ_COMPLETE))
        if any(e == FJ_COMPLETE for e, _ in popped):
            self.is_active = True
        if n_accepted and not self.rollback_tokens(n_accepted):
            self._rebuild()

    def _rebuild(self):
        """
        Reconstruct filter state by replaying the journal from the initial state.
        """
        self.is_active = self.trigger_token is None
        self.reset()
        for e, token in self._journal:
            if e == FJ_TRIGGER:
                self.is_active = True
                self.reset()
            elif e in (FJ_ACCEPT, FJ_COMPLETE):
                self.accept_token(token)
                if e == FJ_COMPLETE:
                    self.is_active = False

    def rollback_tokens(self, num_tokens: int) -> bool:
        """
        Natively roll back the underlying state machine by the last num_tokens accepted tokens.
        Return True on success. Default implementation reports no native rollback support, in which
        case rewind() falls back to rebuilding the state by replay, using only the public filter
        interface (reset/accept_token). Subclasses may override with an efficient implementation.
        """
        return False

    def reset(self):
        """
        Reset the filter to the initial state
        """
        raise NotImplementedError()

    def accept_token(self, token: int):
        """
        Accept a token and advance the underlying state machine. Token is assumed to be in the current valid set.
        Assume self.is_completed() is False upon calling. Accepting the final token in a schema should set the
        completed state to True.
        """
        raise NotImplementedError()

    def get_next_logit_mask(self) -> torch.Tensor:
        """
        Return a mask of valid tokens for the current state as a CPU tensor: either a dense additive
        half tensor of shape (1, vocab_size) with 0 for allowed tokens and -inf for masked tokens, or
        a packed int32 bitmask of shape (1, ceil(vocab / 32)) where bit (i & 31) of word (i >> 5) set
        means token i is allowed. Tokens at or beyond the mask width count as masked out in either
        format. Assume self.is_completed() is False
        """
        raise NotImplementedError()

    def is_completed(self) -> bool:
        """
        Return True if the filter has reached an end state
        """
        raise NotImplementedError()

    def use_background_worker(self) -> bool:
        """
        To indicate whether filter can/should run as a background thread. Should be True unless the filter has a
        special requirement to run in the main thread or does very little computation.
        """
        return True

    def attach(self, job):
        """
        Runs when job is started to link filter to job context
        """
        self.job = job
        self.generator = job.generator
        self.vocab_size = job.generator.padded_vocab_size
        self._journal.clear()