from .sampler import Sampler
import torch
from typing_extensions import override
from ...tokenizer import Tokenizer
from ...ext import exllamav3_ext as ext
from ...util import next_power_of_2
from ...util.tensor import buffered_arange
import random
from dataclasses import dataclass
from enum import Enum

class SS(Enum):
    INIT = 0  # only state.in_logits is valid
    DONE = 1  # finished, state.sample is valid
    LOGITS = 2  # state.logits is valid
    PROBS = 3  # state.probs is valid
    LOGITS_S = 4  # state.logits is valid, state.indices is valid
    PROBS_S = 5  # state.probs is valid but not normalized, indices are valid
    PROBS_N = 6  # state.probs is valid and normalized
    PROBS_N_S = 7  # state.probs is valid and normalized, indices are valid

@dataclass
class SamplingState:
    rand_u32: int
    bsz: int
    dim: int
    in_logits: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    sample: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    indices: torch.Tensor | None = None
    past_ids: torch.Tensor | None = None
    state: SS = SS.INIT

    def empty_sample(self):
        assert self.sample is None
        return torch.empty((self.bsz, 1), dtype = torch.long, device = self.in_logits.device)

    def empty_probs(self, reuse = True):
        if reuse and self.probs is not None:
            return self.probs
        return torch.empty((self.bsz, self.dim), dtype = torch.float, device = self.in_logits.device)

    def empty_logits(self, reuse = True):
        if reuse and self.logits is not None:
            return self.logits
        return torch.empty((self.bsz, self.dim), dtype = torch.float, device = self.in_logits.device)


class SS_Base:
    def run(self, state: SamplingState):
        raise NotImplementedError()
    def prep(self, in_state: SS):
        return None
    def alt(self):
        return None
    def reqs_past_ids(self):
        return False


class SS_NoOp(SS_Base):
    """
    Empty sampling step
    """
    def run(self, state: SamplingState):
        pass


class SS_Argmax(SS_Base):
    """
    Final sampling step: select most likely token
    """
    def run(self, state: SamplingState):
        match state.state:
            case SS.INIT:
                state.sample = torch.argmax(state.in_logits, dim = -1)
            case SS.LOGITS:
                state.sample = torch.argmax(state.logits, dim = -1)
            case SS.PROBS | SS.PROBS_N:
                state.sample = torch.argmax(state.probs, dim = -1)
            case SS.LOGITS_S:
                temp = torch.argmax(state.logits, dim = -1)
                state.state = state.indices[temp]
            case SS.PROBS_S | SS.PROBS_N_S:
                temp = torch.argmax(state.probs, dim = -1)
                state.state = state.indices[temp]
        state.state = SS.DONE


class SS_Sample(SS_Base):
    """
    Final sampling step: categorical sampling, randomly sample from (truncated and/or modified) distribution
    """
    def run(self, state: SamplingState):
        # TODO: Fused Gumbel noise + argmax kernel
        # TODO: Evaluate if multinomial sampling from sorted prob. distribution is more efficient
        match state.state:
            case SS.INIT:
                state.logits = torch.empty_like(state.in_logits)
                ext.gumbel_noise_f16(state.in_logits, state.logits, state.rand_u32)
                state.sample = torch.argmax(state.logits, dim = -1)
            case SS.LOGITS:
                ext.gumbel_noise_f32(state.logits, state.logits, state.rand_u32)
                state.sample = torch.argmax(state.logits, dim = -1)
            case SS.PROBS | SS.PROBS_N:
                ext.gumbel_noise_log(state.probs, state.probs, state.rand_u32)
                state.sample = torch.argmax(state.probs, dim = -1)
            case SS.LOGITS_S:
                ext.gumbel_noise_f32(state.logits, state.logits, state.rand_u32)
                temp = torch.argmax(state.logits, dim = -1)
                state.sample = state.indices[buffered_arange(state.bsz, state.in_logits.device), temp]
            case SS.PROBS_S | SS.PROBS_N_S:
                ext.gumbel_noise_log(state.probs, state.probs, state.rand_u32)
                temp = torch.argmax(state.probs, dim = -1)
                state.sample = state.indices[buffered_arange(state.bsz, state.in_logits.device), temp]
        state.state = SS.DONE


class SS_Sample_mn(SS_Sample):
    """
    Categorical sampling, but only using torch.multinomial (for testing/validation)
    """
    def run(self, state: SamplingState):
        match state.state:
            case SS.PROBS_N_S | SS.PROBS_N:
                state.sample = torch.multinomial(state.probs, num_samples = 1)
            case _:
                raise ValueError("Sampling logic error")
        state.state = SS.DONE

    def prep(self, in_state: SS):
        match in_state:
            case SS.INIT | SS.LOGITS | SS.PROBS | SS.LOGITS_S | SS.PROBS_S:
                return [SS_Normalize]
            case _:
                return None


class SS_Temperature(SS_Base):
    """
    Modify distribution with temperature scaling
    """
    def __init__(self, temperature: float):
        self.temperature = temperature

    def run(self, state: SamplingState):
        match state.state:
            case SS.INIT:
                state.logits = state.in_logits.float()
                state.logits /= self.temperature
                state.state = SS.LOGITS
            case SS.LOGITS:
                state.logits /= self.temperature
            case SS.PROBS | SS.PROBS_N:
                state.probs.pow_(1.0 / self.temperature)
                state.state = SS.PROBS
            case SS.LOGITS_S:
                state.logits /= self.temperature
            case SS.PROBS_S | SS.PROBS_N_S:
                state.probs.pow_(1.0 / self.temperature)
                state.state = SS.PROBS_S

    def alt(self):
        if self.temperature == 1.0:
            return SS_NoOp()
        return None


class SS_Normalize(SS_Base):
    """
    Normalize distribution
    """
    def run(self, state: SamplingState):
        match state.state:
            case SS.INIT:
                state.probs = torch.softmax(state.in_logits.float(), dim = -1)
                state.state = SS.PROBS_N
            case SS.LOGITS:
                state.probs = torch.softmax(state.logits, dim = -1)
                state.state = SS.PROBS_N
            case SS.PROBS:
                state.probs /= state.probs.sum(dim = -1, keepdim = True)
                state.state = SS.PROBS_N
            case SS.LOGITS_S:
                state.probs = torch.softmax(state.logits, dim = -1)
                state.state = SS.PROBS_N_S
            case SS.PROBS_S:
                state.probs /= state.probs.sum(dim = -1, keepdim = True)
                state.state = SS.PROBS_N_S
            case SS.PROBS_N | SS.PROBS_N_S:
                pass


class SS_Sort(SS_Base):
    """
    Sort tokens by descending probability. state.indices
    """
    def run(self, state: SamplingState):
        match state.state:
            case SS.INIT:
                logits = state.in_logits.to(torch.float, copy = True)
                state.logits, state.indices = torch.sort(logits, dim = -1, descending = True)
                state.state = SS.LOGITS_S
            case SS.LOGITS:
                state.logits, state.indices = torch.sort(state.logits, dim = -1, descending = True)
                state.state = SS.LOGITS_S
            case SS.PROBS:
                state.probs, state.indices = torch.sort(state.probs, dim = -1, descending = True)
                state.state = SS.PROBS_S
            case SS.PROBS_N:
                state.probs, state.indices = torch.sort(state.probs, dim = -1, descending = True)
                state.state = SS.PROBS_N_S
            case SS.LOGITS_S | SS.PROBS_S | SS.PROBS_N_S:
                pass


class SS_TopK(SS_Base):
    """
    Mask out all but the top K most likely tokens
    """
    def __init__(self, top_k: int):
        assert isinstance(top_k, int) or top_k.is_integer(), "top_k value must be integer"
        self.top_k = int(top_k)

    def run(self, state: SamplingState):
        match state.state:
            case SS.PROBS_S | SS.PROBS_N_S:
                state.probs[..., self.top_k:] = 0.0
                state.state = SS.PROBS_S
            case SS.LOGITS_S:
                state.logits[..., self.top_k:] = -float("inf")
            case _:
                raise ValueError("Sampling logic error")

    def prep(self, in_state: SS):
        match in_state:
            case SS.INIT | SS.LOGITS | SS.PROBS | SS.PROBS_N:
                return [SS_Sort]
            case _:
                return None

    def alt(self):
        if self.top_k < 1:
            return SS_NoOp()
        return None


class SS_TopP(SS_Base):
    """
    Identify the smallest set of top tokens with a cumulative probability greater than P, mask out all
    remainig tokens
    """
    def __init__(self, top_p: float):
        self.top_p = top_p
        assert 0.0 <= top_p <= 1.0

    def run(self, state: SamplingState):
        match state.state:
            case SS.PROBS_N_S:
                cumsum = state.probs.cumsum(dim = -1)
                mask = cumsum <= self.top_p
                state.probs[..., 1:] *= mask[..., 1:]
                state.state = SS.PROBS_S
            case _:
                raise ValueError("Sampling logic error")

    def prep(self, in_state: SS):
        match in_state:
            case SS.PROBS_N:
                return [SS_Sort]
            case SS.INIT | SS.LOGITS | SS.PROBS:
                return [SS_Normalize, SS_Sort]
            case SS.LOGITS_S | SS.PROBS_S:
                return [SS_Normalize]
            case _:
                return None

    def alt(self):
        if self.top_p == 1.0:
            return SS_NoOp()
        return None


class SS_MinP(SS_Base):
    """
    Mask out all tokens whose probability is less than the top token's probability times min_p
    """
    def __init__(self, min_p: float):
        self.min_p = min_p
        assert 0.0 <= min_p <= 1.0

    def run(self, state: SamplingState):
        match state.state:
            case SS.PROBS_N:
                threshold = state.probs.amax(dim = -1, keepdim = True) * self.min_p
                mask = state.probs >= threshold
                state.probs *= mask
                state.state = SS.PROBS
            case SS.PROBS_N_S:
                threshold = state.probs[:, :1] * self.min_p
                mask = state.probs >= threshold
                state.probs *= mask
                state.state = SS.PROBS_S
            case _:
                raise ValueError("Sampling logic error")

    def prep(self, in_state: SS):
        match in_state:
            case SS.INIT | SS.LOGITS | SS.PROBS | SS.LOGITS_S | SS.PROBS_S:
                return [SS_Normalize]
            case _:
                return None

    def alt(self):
        if self.min_p == 0.0:
            return SS_NoOp()
        return None


class SS_RepP(SS_Base):
    """
    Apply Transformers style repetition penalties based on past token IDs. Must be the first step in sampler
    chain.
    """
    def __init__(
        self,
        rep_p: float = 1.0,
        sustain_range: int = int(10e7),
        decay_range: int = 0
    ):
        """
        :param rep_p:
            Multiplicative penalty. rep_p = 1.0 means no penalty. Positive logits are divided by this value and
            negative ones are multiplied by it. Recreates the method from the Transformers generate() pipeline,
            following https://arxiv.org/pdf/1909.05858.pdf which relies on the assumption that logits output
            straight from the model are "centered" around zero.
         :param sustain_range:
            Number of most recent past tokens over which to apply full penalty
        :param decay_range:
            Number tokens (after sustain_range) over which the penalty gradually fades out
        """
        self.rep_p = rep_p
        self.sustain_range = sustain_range
        self.decay_range = decay_range

    def run(self, state: SamplingState):
        match state.state:
            case SS.INIT:
                state.logits = torch.empty_like(state.in_logits, dtype = torch.float)
                ext.apply_rep_pens(
                    state.in_logits,
                    state.logits,
                    state.past_ids,
                    self.rep_p,
                    self.sustain_range,
                    self.decay_range
                )
            case SS.LOGITS:
                ext.apply_rep_pens(
                    state.logits,
                    state.logits,
                    state.past_ids,
                    self.rep_p,
                    self.sustain_range,
                    self.decay_range
                )
            case _:
                raise ValueError("Sampling logic error")
        state.state = SS.LOGITS

    def alt(self):
        if self.rep_p == 1.0 or self.sustain_range + self.decay_range <= 0:
            return SS_NoOp()
        return None

    def reqs_past_ids(self):
        return True


class SS_PresFreqP(SS_Base):
    """
    Apply OAI-style presence and frequency penalties based on past token IDs. Must be the first step in the
    sampler chain.
    """
    def __init__(
        self,
        pres_p: float = 0.0,
        freq_p: float = 0.0,
        sustain_range: int = int(10e7),
        decay_range: int = 0
    ):
        """
        :param pres_p:
            Additive penalty, OAI style. 0.0 means no penalty. Added to logit once if a token appears in
            past_ids
        :param freq_p:
            Additive penalty, OAI style. 0.0 means no penalty. Added to logit for every time a token is
            encountered in past_ids
         :param sustain_range:
            Number of most recent past tokens over which to apply full penalty
        :param decay_range:
            Number tokens (after sustain_range) over which the penalty gradually fades out
        """
        self.pres_p = pres_p
        self.freq_p = freq_p
        self.sustain_range = sustain_range
        self.decay_range = decay_range

    def run(self, state: SamplingState):
        match state.state:
            case SS.INIT:
                state.logits = torch.empty_like(state.in_logits, dtype = torch.float)
                ext.apply_pres_freq_pens(
                    state.in_logits,
                    state.logits,
                    state.past_ids,
                    self.pres_p,
                    self.freq_p,
                    self.sustain_range,
                    self.decay_range
                )
            case SS.LOGITS:
                ext.apply_pres_freq_pens(
                    state.logits,
                    state.logits,
                    state.past_ids,
                    self.pres_p,
                    self.freq_p,
                    self.sustain_range,
                    self.decay_range
                )
            case _:
                raise ValueError("Sampling logic error")
        state.state = SS.LOGITS

    def alt(self):
        if (self.pres_p == 0.0 and self.freq_p == 0.0) or self.sustain_range + self.decay_range <= 0:
            return SS_NoOp()
        return None

    def reqs_past_ids(self):
        return True


class CustomSampler(Sampler):
    def __init__(
        self,
        steps: list[SS_Base]
    ):
        super().__init__()

        self.steps = []
        state = SS.INIT
        for step in steps:
            self.reqs_past_ids = self.reqs_past_ids or step.reqs_past_ids()
            alt = step.alt()
            if alt:
                step = alt
            prep_steps = step.prep(state)
            if prep_steps:
                for prep_step in prep_steps:
                    self.steps.append(prep_step())
            self.steps.append(step)

        # TODO: Identify and remove redundant sampling steps, add rules for fusing steps where possible

    @override
    @torch.inference_mode
    def forward(
        self,
        logits,
        sequence_ids: torch.Tensor | None = None,
        rand_u32: int | None = None,
        tokenizer: Tokenizer | None = None,
        blocked_tokens: list[int] | None = None,
        allowed_tokens: list[int] | None = None,
        return_state: bool = False
    ):
        out_shape = logits.shape[:-1]

        if tokenizer is not None:
            logits[..., tokenizer.actual_vocab_size:] = -float("inf")

        if rand_u32 is None:
            rand_u32 = random.randint(0, (1<<32) - 1)
        else:
            torch.manual_seed(rand_u32)
            random.seed(rand_u32)

        dim = logits.shape[-1]
        bsz = logits.numel() // dim

        # Prepare logit bias tensor

        # TODO: Extension function for this, combine with filter API when it's added
        if blocked_tokens is not None or allowed_tokens is not None:
            logits = logits.clone()
        if blocked_tokens is not None:
            logits[..., blocked_tokens] = float('-inf')
        if allowed_tokens is not None:
            mask = torch.zeros(logits.shape[-1], dtype = torch.bool, device = logits.device)
            mask[allowed_tokens] = True
            logits[..., ~mask] = float('-inf')

        state = SamplingState(
            rand_u32 = rand_u32,
            dim = dim,
            bsz = bsz,
            in_logits = logits.view(bsz, dim),
            past_ids = sequence_ids,
        )

        for ss in self.steps:
            assert state.state != SS.DONE, "Sampling logic error"
            ss.run(state)
        assert return_state or state.state == SS.DONE, "Sampling logic error"

        return state if return_state else state.sample.view(out_shape)