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


class SS_NoOp(SS_Base):
    def run(self, state: SamplingState):
        pass


class SS_Argmax(SS_Base):
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


class SS_Temperature(SS_Base):
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
    def __init__(self, top_k: int):
        assert top_k >= 1
        self.top_k = top_k

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


class SS_TopP(SS_Base):
    def __init__(self, top_p: float):
        self.top_p = top_p
        assert 0.0 < top_p <= 1.0
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


class CustomSampler(Sampler):
    def __init__(
        self,
        steps: list[SS_Base]
    ):
        super().__init__()

        self.steps = []
        state = SS.INIT
        for step in steps:
            alt = step.alt()
            if alt:
                step = alt
            prep_steps = step.prep(state)
            if prep_steps:
                for prep_step in prep_steps:
                    self.steps.append(prep_step())
            self.steps.append(step)

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
        )

        for ss in self.steps:
            assert state.state != SS.DONE, "Sampling logic error"
            ss.run(state)
        assert state.state == SS.DONE, "Sampling logic error"

        return state.sample.view(out_shape)