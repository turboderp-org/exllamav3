from .sampler import Sampler
import math
import os
import torch
from typing_extensions import override
from ...tokenizer import Tokenizer
from ...ext import exllamav3_ext as ext
from ...util import next_power_of_2
from ...util.tensor import buffered_arange
import random
from dataclasses import dataclass
from enum import Enum
from ...util import profile_opt
import torch.nn.functional as F

# Collapse eligible sampler stacks into the fused kernel path; EXL3_FUSED_SAMPLER=0 keeps the
# original step-by-step implementation (for testing/validation)
fused_sampler_enable = os.environ.get("EXL3_FUSED_SAMPLER", "1") != "0"

class SS(Enum):
    INIT = 0  # only state.in_logits is valid
    DONE = 1  # finished, state.sample is valid
    LOGITS = 2  # state.logits is valid
    PROBS = 3  # state.probs is valid
    LOGITS_S = 4  # state.logits is valid, state.indices is valid
    PROBS_S = 5  # state.probs is valid but not normalized, indices are valid
    PROBS_N = 6  # state.probs is valid and normalized
    PROBS_N_S = 7  # state.probs is valid and normalized, indices are valid

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def conditional(condition, a, b):
    return a if condition else b

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
    # Logit mask deferred into the fused kernel (fused-only stacks); fused_dim bounds the vocab
    # scan and doubles as the -inf padding of a mask narrower than the logits
    fused_mask: torch.Tensor | None = None
    fused_dim: int | None = None

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
    def reqs_torch_seed(self):
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
                state.sample = state.indices[temp]
            case SS.PROBS_S | SS.PROBS_N_S:
                temp = torch.argmax(state.probs, dim = -1)
                state.sample = state.indices[temp]
        state.state = SS.DONE


class SS_Sample(SS_Base):
    """
    Final sampling step: categorical sampling, randomly sample from (truncated and/or modified) distribution
    """
    def run(self, state: SamplingState):
        # TODO: Fused Gumbel noise + argmax kernel
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

    def reqs_torch_seed(self):
        return True


class SS_Fused(SS_Base):
    """
    Collapsed terminal step covering the common truncation/temperature/sample stacks in a few
    fused kernel calls, working directly in logit space:

        p_i >= min_p * p_max                    <=>  l_i >= max(l) + ln(min_p)
        softmax(l)^(1/T), renormalized          <=>  softmax(l / T)
        categorical sample from softmax(l / T)  ==   argmax(l_i / T + g_i),  g_i ~ Gumbel(0, 1)

    Top-K and top-P also reduce to a single logit threshold, since every one of these filters
    keeps a top segment of the same ordering. The threshold is found from a deterministic
    fixed-point histogram of the logits (counts for top-K, exp-mass for top-P, normalized over
    the top-K-truncated set), with boundary buckets refined to 1/32768 nat -- finer than fp16
    ULP, so the kept set matches the exact sort-based truncation for fp16 logits. Tokens tied
    exactly at a cutoff are all kept rather than truncated in sort order.

    Produced by CustomSampler's stack collapse; equivalent to the steps it replaces (same
    Gumbel noise stream for a given rand_u32, up to rounding at exact ties).
    """
    MODE_GREEDY = 0
    MODE_SAMPLE = 1
    MODE_SAMPLE_MINP = 2
    MODE_SAMPLE_FILTERS = 3

    F_TOPK = 1
    F_TOPP = 2
    F_MINP = 4

    def __init__(
        self,
        mode: int,
        temperature: float = 1.0,
        minp_log: float = 0.0,
        filters: int = 0,
        top_k: int = 0,
        top_p: float = 1.0,
        temp_first: bool = False,
    ):
        self.mode = mode
        self.inv_temp = 1.0 / temperature
        self.minp_log = minp_log
        self.filters = filters
        self.top_k = top_k
        self.top_p = top_p
        # Filters computed on the tempered distribution (temperature before truncation in the
        # stack) weigh the histogram mass with the sampling temperature
        self.inv_temp_filter = self.inv_temp if temp_first else 1.0
        self.workspaces = {}
        self.histograms = {}

    def run(self, state: SamplingState):
        match state.state:
            case SS.INIT:
                logits = state.in_logits
            case SS.LOGITS:
                logits = state.logits
            case _:
                raise ValueError("Sampling logic error")
        if logits.dtype not in (torch.half, torch.float):
            logits = logits.float()
        if not logits.is_contiguous():
            logits = logits.contiguous()

        ws_key = (logits.device, state.bsz)
        workspace = self.workspaces.get(ws_key)
        if workspace is None:
            workspace = torch.empty(
                (state.bsz * ext.FUSED_SAMPLER_MAX_BLOCKS * 3,),
                dtype = torch.float,
                device = logits.device,
            )
            self.workspaces[ws_key] = workspace

        histogram = None
        if self.mode == SS_Fused.MODE_SAMPLE_FILTERS:
            histogram = self.histograms.get(ws_key)
            if histogram is None:
                histogram = torch.empty(
                    (state.bsz * ext.FUSED_SAMPLER_HIST_STRIDE,),
                    dtype = torch.uint8,
                    device = logits.device,
                )
                self.histograms[ws_key] = histogram

        state.sample = state.empty_sample()
        ext.fused_sampler(
            logits,
            state.fused_mask,
            state.sample,
            workspace,
            state.fused_dim or state.dim,
            self.inv_temp,
            self.minp_log,
            state.rand_u32,
            self.mode,
            self.filters,
            self.top_k,
            self.top_p,
            self.inv_temp_filter,
            histogram,
        )
        state.state = SS.DONE


# Tail patterns that collapse into SS_Fused, matched against the alt()-simplified stack:
#
#     [Temperature?] [MinP?] [TopK?] [TopP?] [Temperature?] (Sample | Argmax)
#
# with at most one temperature step, and the filters in the canonical (ComboSampler) order.
# This covers all the preset samplers. A greedy tail absorbs preceding temperature/truncation
# steps (they never change the top token). Whether temperature precedes or follows the filters
# decides whether they truncate the tempered or the untempered distribution; both reduce to
# logit-space thresholds (min-P at max(l) + ln(min_p), scaled by T when tempered; top-K/top-P
# through the histogram select with mass at the filter temperature).

def _match_fused_tail(tail: list) -> SS_Fused | None:
    if not tail:
        return None
    last = tail[-1]
    if type(last) not in (SS_Sample, SS_Argmax):
        return None
    pre = tail[:-1]
    if any(isinstance(s, SS_Temperature) and s.temperature <= 0.0 for s in pre):
        return None

    if type(last) is SS_Argmax:
        if all(type(s) in (SS_Temperature, SS_MinP, SS_TopK, SS_TopP) for s in pre):
            return SS_Fused(SS_Fused.MODE_GREEDY)
        return None

    # At most one temperature step, leading or trailing the filter sequence
    temp_first = False
    temperature = 1.0
    if pre and type(pre[0]) is SS_Temperature:
        temp_first = True
        temperature = pre[0].temperature
        pre = pre[1:]
    if pre and type(pre[-1]) is SS_Temperature:
        if temp_first:
            return None
        temperature = pre[-1].temperature
        pre = pre[:-1]

    # Filters must be a subsequence of (MinP, TopK, TopP)
    sig = tuple(type(s) for s in pre)
    order = (SS_MinP, SS_TopK, SS_TopP)
    pos = -1
    for t in sig:
        if t not in order or order.index(t) <= pos:
            return None
        pos = order.index(t)

    min_p = next((s.min_p for s in pre if type(s) is SS_MinP), None)
    top_k = next((s.top_k for s in pre if type(s) is SS_TopK), None)
    top_p = next((s.top_p for s in pre if type(s) is SS_TopP), None)

    # Degenerate truncations keep only the top token
    if top_k == 1 or top_p == 0.0:
        return SS_Fused(SS_Fused.MODE_GREEDY)

    minp_log = (temperature if temp_first else 1.0) * math.log(min_p) if min_p else 0.0

    if top_k is None and top_p is None:
        if min_p is None:
            return SS_Fused(SS_Fused.MODE_SAMPLE, temperature)
        return SS_Fused(SS_Fused.MODE_SAMPLE_MINP, temperature, minp_log)

    filters = (
        (SS_Fused.F_MINP if min_p is not None else 0) |
        (SS_Fused.F_TOPK if top_k is not None else 0) |
        (SS_Fused.F_TOPP if top_p is not None else 0)
    )
    return SS_Fused(
        SS_Fused.MODE_SAMPLE_FILTERS,
        temperature,
        minp_log,
        filters,
        top_k or 0,
        top_p if top_p is not None else 1.0,
        temp_first,
    )


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
    Sort tokens by descending probability.
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


class SS_AdaptiveP(SS_Base):
    """
    Implements Adaptive-P sampler. Maintains state but does not remember past states (keeps future state in case
    of rollback).
    """
    def __init__(
        self,
        target: float = 1.0,
        decay: float = 0.0
    ):
        self.target = target
        self.decay = decay
        clamped_decay = max(min(decay, 0.99), 0.0)
        self.weighted_sum = target / (1.0 - clamped_decay)
        self.total_weight = 1.0 / (1.0 - clamped_decay)

        self.DISTRIBUTION_WIDTH = 0.3
        self.PEAK_LOGIT_VALUE = 5.0
        self.SHARPNESS = 10.0
        self.INV_WIDTH = 1.0 / self.DISTRIBUTION_WIDTH

        # self.log = []

    def run(self, state: SamplingState):
        match state.state:
            case SS.PROBS_N_S:
                target = clamp(self.target, 0.0, 1.0)
                adapted_target = conditional(
                    self.total_weight == 0.0,
                    target,
                    2.0 * target - (self.weighted_sum / self.total_weight)
                )
                adapted_target = clamp(adapted_target, 0.0, 1.0)

                state.logits = torch.empty_like(state.in_logits, dtype = torch.float)
                ext.adaptivep_gumbel_noise_f32(
                    state.probs,
                    state.logits,
                    state.rand_u32,
                    adapted_target,
                    self.INV_WIDTH,
                    self.PEAK_LOGIT_VALUE,
                    self.SHARPNESS
                )

                temp = torch.argmax(state.logits, dim = -1)
                state.sample = state.indices[buffered_arange(state.bsz, state.in_logits.device), temp]
                sampled_prob = state.probs[0, temp].item()

                # self.log.append((adapted_target, sampled_prob))
                # if len(self.log) == 300:
                #     print("\n\n\n")
                #     s = 0
                #     for i, (a, b) in enumerate(self.log):
                #         s += b
                #         m = s / (i + 1)
                #         print(f"{i};{a};{b};{m}")
                #     print("\n\n\n")

                self.weighted_sum = sampled_prob + self.decay * self.weighted_sum
                self.total_weight = 1.0 + self.decay * self.total_weight
            case _:
                raise ValueError("Sampling logic error")
        state.state = SS.DONE

    def prep(self, in_state: SS):
        match in_state:
            case SS.INIT | SS.LOGITS | SS.PROBS | SS.LOGITS_S | SS.PROBS_S:
                return [SS_Normalize, SS_Sort]
            case _:
                return None

    def alt(self):
        if self.target == 1.0:
            return SS_NoOp()
        return None


class CustomSampler(Sampler):
    def __init__(
        self,
        steps: list[SS_Base]
    ):
        super().__init__()

        # Simplify the stack (identity steps become no-ops), then collapse an eligible tail
        # into the fused kernel step. Leading penalty steps are kept as-is; they feed fp32
        # logits to the fused step. Ineligible stacks fall through to the step-by-step path.
        simplified = []
        for step in steps:
            self.reqs_past_ids = self.reqs_past_ids or step.reqs_past_ids()
            self.reqs_torch_seed = self.reqs_torch_seed or step.reqs_torch_seed()
            alt = step.alt()
            if alt:
                step = alt
            if not isinstance(step, SS_NoOp):
                simplified.append(step)

        head = []
        fused_tail = None
        if fused_sampler_enable:
            i = 0
            while i < len(simplified) and type(simplified[i]) in (SS_RepP, SS_PresFreqP):
                i += 1
            fused_tail = _match_fused_tail(simplified[i:])
            if fused_tail is not None:
                head = simplified[:i]

        if fused_tail is not None:
            self.steps = head + [fused_tail]
            self.fused_only = not head
        else:
            self.steps = []
            self.fused_only = False
            state = SS.INIT
            for step in simplified:
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
        logit_mask: torch.Tensor | None = None,
        return_state: bool = False
    ):
        out_shape = logits.shape[:-1]

        if tokenizer is not None and tokenizer.actual_vocab_size < logits.shape[-1]:
            logits[..., tokenizer.actual_vocab_size:] = -float("inf")

        if rand_u32 is None:
            rand_u32 = random.randint(0, (1<<32) - 1)
        else:
            if self.reqs_torch_seed:
                torch.manual_seed(rand_u32)
                random.seed(rand_u32)

        dim = logits.shape[-1]
        bsz = logits.numel() // dim

        # For a fully fused stack the mask is applied inside the kernel; a mask narrower than
        # the logits bounds the scan instead of being padded with -inf. Same for the padded
        # vocab region, which the in-place fill above has already masked.
        fused_mask = None
        fused_dim = None
        if (
            self.fused_only and
            (logit_mask is None or (
                logit_mask.dtype == torch.half and
                logit_mask.is_contiguous() and
                (logit_mask.shape[0] == 1 or
                    (logit_mask.shape[0] == bsz and logit_mask.shape[-1] == dim))
            ))
        ):
            fused_dim = dim
            if tokenizer is not None:
                fused_dim = min(fused_dim, tokenizer.actual_vocab_size)
            if logit_mask is not None:
                fused_mask = logit_mask.view(logit_mask.shape[0], -1)
                fused_dim = min(fused_dim, fused_mask.shape[-1])

        # Apply logit mask/bias tensor
        elif logit_mask is not None:
            pad = logits.shape[-1] - logit_mask.shape[-1]
            if pad > 0:
                logit_mask = F.pad(logit_mask, (0, pad), value = float("-inf"))
            logits = logits + logit_mask

        state = SamplingState(
            rand_u32 = rand_u32,
            dim = dim,
            bsz = bsz,
            in_logits = logits.view(bsz, dim),
            past_ids = sequence_ids,
            fused_mask = fused_mask,
            fused_dim = fused_dim,
        )

        for ss in self.steps:
            assert state.state != SS.DONE, "Sampling logic error"
            ss.run(state)
        assert return_state or state.state == SS.DONE, "Sampling logic error"

        return state if return_state else state.sample.view(out_shape)