from ...tokenizer import Tokenizer
import torch

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
        Return boolean mask of valid tokens for the current state as CPU tensor. Assume self.is_completed() is
        False and self.compute_constraint() has been called
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