"""
Batched speculative-window readback: batched path vs per-position serial loop.

Generator.iterate_gen reads back the sampled token of the speculative window either batched (launch
every position's sampler, stage the CUDA results into one pinned buffer, one synchronize) or with
the per-position serial loop. The batched path is taken when a job is batchable (single sequence,
no banned-string checkpoint, past the min_new_tokens window, no active filters, sampler does not
need past token IDs); setting _NO_BATCH_READBACK forces the serial path.

For greedy decoding both paths must emit the identical token sequence: the batched path only
changes when the window's readback happens, never what is decided. Each test below runs the same
job twice (flag off, flag on) on a generator with ngram draft mode and asserts the full sequences
are equal, with at least one draft window actually verified so the speculative path is exercised.
A sampling job is only required to complete on both paths: the batched path consumes one RNG draw
per draft position, so a sampled stream may legitimately diverge after the first rejected draft
position while the distribution is unchanged.

Model role "dense-llama".
"""

import pytest
import torch

import exllamav3.generator.generator as gen_mod
from exllamav3 import Generator, Job, Filter
from exllamav3.generator.sampler import (
    ArgmaxSampler,
    CategoricalSampler,
    CustomSampler,
    SS_Argmax,
    SS_BanTokens,
    SS_LogitBias,
    SS_RepP,
)
from testlib.e2e import load_model

pytestmark = pytest.mark.model("dense-llama")

# Heavily repeated prompt: the ngram suffix automaton matches long spans, so draft windows are
# wide enough to make the batched readback do real work.
PROMPT = ("The fox ran. The fox ran. The fox ran. The fox ran. The fox ran. "
          "The fox ran. The fox ran. The fox ran. The fox ran. The fox ran. ")

NEW_TOKENS = 64


class AllowSetFilter(Filter):
    """Allows only a fixed set of tokens, forever"""
    def __init__(self, tokenizer, allowed):
        super().__init__(tokenizer, None, None, False)
        self.allowed = list(allowed)
    def reset(self): pass
    def accept_token(self, token): pass
    def is_completed(self): return False
    def use_background_worker(self): return False
    def get_next_logit_mask(self):
        m = torch.full((1, self.vocab_size), float("-inf"), dtype = torch.half)
        m[:, self.allowed] = 0
        return m


@pytest.fixture(scope = "module")
def lm(model_registry, device):
    with load_model(model_registry.get("dense-llama").path, device, "-ngram", "4", cache_tokens = 8192) as lm:
        yield lm


@pytest.fixture(scope = "module")
def tok(lm):
    return lm.tokenizer


@pytest.fixture(scope = "module")
def ids(tok):
    return tok.encode(PROMPT, add_bos = True)


@pytest.fixture(scope = "module")
def ids1(ids):
    """The prompt as a flat list of ints, for picking real single-token ids"""
    return ids[0].tolist() if ids.dim() == 2 else ids.tolist()


@pytest.fixture
def gen(lm):
    gen = Generator(model = lm.model, cache = lm.cache, tokenizer = lm.tokenizer,
                    max_batch_size = 1, ngram_match_min = 4)
    yield gen
    gen.close()


@pytest.fixture
def gen2(lm):
    gen = Generator(model = lm.model, cache = lm.cache, tokenizer = lm.tokenizer,
                    max_batch_size = 2, ngram_match_min = 4)
    yield gen
    gen.close()


def run_job(gen, tok, input_ids, sampler, new_tokens = NEW_TOKENS, min_new_tokens = 0,
            filters = None, no_batch_readback = False):
    """
    Run one job to completion with the given batched-readback flag; returns (sequence ids,
    draft tokens verified, draft tokens rejected).
    """
    old = gen_mod._NO_BATCH_READBACK
    gen_mod._NO_BATCH_READBACK = no_batch_readback
    try:
        job = Job(input_ids = input_ids, max_new_tokens = new_tokens, sampler = sampler,
                  min_new_tokens = min_new_tokens, filters = filters)
        gen.enqueue(job)
        while gen.num_remaining_jobs():
            gen.iterate()
        return (job.sequences[0].sequence_ids.torch()[0].tolist(),
                job.accepted_draft_tokens,
                job.rejected_draft_tokens)
    finally:
        gen_mod._NO_BATCH_READBACK = old


def equivalent(gen, tok, **job_kw):
    """
    Run the same job on both readback paths; asserts the speculative window was verified at
    least once and that both paths produced the identical sequence.
    """
    seq_a, acc_a, rej_a = run_job(gen, tok, no_batch_readback = False, **job_kw)
    seq_b, acc_b, rej_b = run_job(gen, tok, no_batch_readback = True, **job_kw)
    assert acc_a + rej_a > 0, "no draft tokens were verified; the test prompt is not producing drafts"
    assert acc_b + rej_b > 0, "no draft tokens were verified on the serial path"
    assert seq_a == seq_b, f"readback paths diverged: batched={seq_a} serial={seq_b}"


def test_greedy_argmax(gen, tok, ids):
    equivalent(gen, tok, input_ids = ids, sampler = ArgmaxSampler())


def test_greedy_logit_bias(gen, tok, ids, ids1):
    # A real prompt token; the bias must not flip the greedy continuation or the prompt's
    # periodicity (and with it the drafts) is destroyed
    t = ids1[2]
    equivalent(gen, tok, input_ids = ids,
               sampler = CustomSampler([SS_LogitBias({t: 5.0}), SS_Argmax()]))


def test_greedy_banned_token(gen, tok, ids, ids1):
    # The ban must not be part of the prompt's repeated period: banning a period token makes the
    # model's greedy continuation non-periodic (it switches patterns), which kills the ngram
    # drafts this test needs. The step stays live in the chain either way
    e = tok.encode(" banana", add_bos = False)
    t = e.flatten()[0].item()
    equivalent(gen, tok, input_ids = ids,
               sampler = CustomSampler([SS_BanTokens([t]), SS_Argmax()]))


def test_greedy_rep_penalty(gen, tok, ids):
    # Needs past token IDs: both runs take the serial loop; the outputs must still agree. The penalty is
    # kept mild: a strong one breaks the repetition the n-gram drafter needs, and no window gets verified
    equivalent(gen, tok, input_ids = ids,
               sampler = CustomSampler([SS_RepP(1.05), SS_Argmax()]))


class TriggeredAllowSetFilter(AllowSetFilter):
    """Dormant until its trigger token is emitted, then allows only a fixed set of tokens"""
    def __init__(self, tokenizer, allowed, trigger):
        super().__init__(tokenizer, allowed)
        self.trigger_token = trigger
        self.is_active = False


def run_job_with_event(gen, tok, input_ids, after_tokens, event, no_batch_readback):
    """Run a greedy job, calling event(job) once after_tokens tokens have streamed; returns the sequence
    ids and the draft tokens verified"""
    old = gen_mod._NO_BATCH_READBACK
    gen_mod._NO_BATCH_READBACK = no_batch_readback
    try:
        job = Job(input_ids = input_ids, max_new_tokens = NEW_TOKENS, sampler = ArgmaxSampler())
        gen.enqueue(job)
        streamed, fired = 0, False
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r["stage"] == "streaming" and r.get("token_ids") is not None:
                    streamed += r["token_ids"].numel()
            if not fired and streamed >= after_tokens:
                event(job)
                fired = True
        return (job.sequences[0].sequence_ids.torch()[0].tolist(),
                job.accepted_draft_tokens + job.rejected_draft_tokens)
    finally:
        gen_mod._NO_BATCH_READBACK = old


def test_triggered_filter_mid_window(gen, tok, ids, ids1):
    # A filter installed dormant activates on a token that lands inside a verification window; the
    # positions after it are constrained on both paths, and the paths agree
    trigger = ids1[4]                        # the period: emitted every few tokens
    allowed = [ids1[2], ids1[3]]
    install = lambda job: job.set_filters([TriggeredAllowSetFilter(tok, allowed, trigger)])
    n = ids.shape[-1]
    seq_a, drafts_a = run_job_with_event(gen, tok, ids, 12, install, no_batch_readback = False)
    seq_b, drafts_b = run_job_with_event(gen, tok, ids, 12, install, no_batch_readback = True)
    assert drafts_a > 0 and drafts_b > 0, "no draft tokens were verified"
    gen_a = seq_a[n + 12:]
    assert trigger in gen_a, "the trigger was never emitted after the filter was installed"
    constrained = gen_a[gen_a.index(trigger) + 1:]
    assert constrained and all(t in allowed for t in constrained), "a token escaped the filter"
    assert seq_a == seq_b, f"readback paths diverged: batched={seq_a} serial={seq_b}"


def test_forced_output_mid_window(gen, tok, ids):
    # constrain_output_now while drafting: every forced token must come out, on both paths
    forced = " ZEBRA QUARTZ VIOLIN"
    inject = lambda job: job.constrain_output_now(forced)
    n = ids.shape[-1]
    seq_a, drafts_a = run_job_with_event(gen, tok, ids, 6, inject, no_batch_readback = False)
    seq_b, drafts_b = run_job_with_event(gen, tok, ids, 6, inject, no_batch_readback = True)
    assert drafts_a > 0 and drafts_b > 0, "no draft tokens were verified"
    assert forced in tok.decode(torch.tensor(seq_a[n:])), "forced output lost on the batched path"
    assert seq_a == seq_b, f"readback paths diverged: batched={seq_a} serial={seq_b}"


def test_min_new_tokens_window(gen, tok, ids):
    # Serial while new_tokens < min_new_tokens, batched after; the crossing window must match
    equivalent(gen, tok, input_ids = ids, sampler = ArgmaxSampler(), min_new_tokens = 32)


def test_filter_fallback(gen, tok, ids, ids1):
    # An active filter holds the mask between positions: both runs fall back to serial
    allowed = [ids1[3], ids1[4]]
    equivalent(gen, tok, input_ids = ids, sampler = ArgmaxSampler(),
               filters = [AllowSetFilter(tok, allowed)])


def run_two(gen, tok, ids, no_batch_readback):
    """Two concurrent jobs in one active batch; returns (both sequences, total draft tokens verified)"""
    old = gen_mod._NO_BATCH_READBACK
    gen_mod._NO_BATCH_READBACK = no_batch_readback
    try:
        jobs = [Job(input_ids = ids, max_new_tokens = NEW_TOKENS, sampler = ArgmaxSampler()) for _ in range(2)]
        for job in jobs:
            gen.enqueue(job)
        while gen.num_remaining_jobs():
            gen.iterate()
        return ([job.sequences[0].sequence_ids.torch()[0].tolist() for job in jobs],
                sum(job.accepted_draft_tokens + job.rejected_draft_tokens for job in jobs))
    finally:
        gen_mod._NO_BATCH_READBACK = old


def test_concurrent_jobs(gen2, tok, ids):
    # Two jobs in the active batch: b - a > 1, not batchable, serial on both paths
    seqs_a, n_a = run_two(gen2, tok, ids, False)
    seqs_b, n_b = run_two(gen2, tok, ids, True)
    assert n_a + n_b > 0, "no draft tokens were verified; the test prompt is not producing drafts"
    assert seqs_a == seqs_b, f"readback paths diverged: batched={seqs_a} serial={seqs_b}"


def test_forced_tokens(gen, tok, ids):
    # constrain_output_now forces CPU-staged tokens through the window
    def forced(no_batch_readback):
        old = gen_mod._NO_BATCH_READBACK
        gen_mod._NO_BATCH_READBACK = no_batch_readback
        try:
            job = Job(input_ids = ids, max_new_tokens = NEW_TOKENS, sampler = ArgmaxSampler())
            gen.enqueue(job)
            job.constrain_output_now(" fox ran")
            while gen.num_remaining_jobs():
                gen.iterate()
            return job.sequences[0].sequence_ids.torch()[0].tolist(), \
                   job.accepted_draft_tokens, job.rejected_draft_tokens
        finally:
            gen_mod._NO_BATCH_READBACK = old

    seq_a, acc_a, rej_a = forced(False)
    seq_b, acc_b, rej_b = forced(True)
    assert acc_a + rej_a + acc_b + rej_b > 0, "no draft tokens were verified; the test prompt is not producing drafts"
    assert seq_a == seq_b, f"readback paths diverged: batched={seq_a} serial={seq_b}"


def test_sampling_completes(gen, tok, ids):
    # Sampling jobs are not required to match between paths (see module docstring); they must
    # complete on both. A temp-1 continuation is not guaranteed to stay periodic, so no draft
    # assertion here — the greedy tests above own that
    for flag in (False, True):
        seq, acc, rej = run_job(gen, tok, no_batch_readback = flag,
                                input_ids = ids, sampler = CategoricalSampler(temperature = 1.0))
        assert len(seq) > len(ids), f"job produced no new tokens (flag={flag})"