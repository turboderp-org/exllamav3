"""
The EXL3_NAN_CHECK fail-fast in Job.receive_logits: non-finite logits at a checked step raise
NonFiniteLogits (sampling them silently yields a fixed garbage token -- argmax over NaN is id 0 --
that can loop and hide the fault), finite logits pass through to the sampler, and the mode
boundaries hold (first-token only by default, every token at 2, off at 0).

CPU-only throughout: the check is what's under test, so the sampler is stubbed with a plain
argmax instead of the fused CUDA one.
"""

import random

import pytest
import torch

from exllamav3.generator import job as job_module
from exllamav3.generator.job import Job, NonFiniteLogits
from exllamav3.generator.sampler import Sampler

pytestmark = pytest.mark.nogpu


class StubSampler(Sampler):
    def forward(self, logits, current_device_ids, rand_idx, tokenizer, logit_mask = None,
                return_state = False):
        return logits.argmax(dim = -1)


class StubGenerator:
    class tokenizer:  # noqa: N801 - duck-typed attribute
        actual_vocab_size = 1 << 60    # never triggers the vocab-crop path


def make_job(new_tokens):
    job = Job.__new__(Job)
    job.new_tokens = new_tokens
    job.identifier = "test-job"
    job.serial_number = 7
    job.forced_ids = None
    job.forced_ids_device = None
    job.forced_index = 0
    job.sampler = StubSampler()
    job.current_device_ids = None
    job.rng = random.Random(0)
    job.generator = StubGenerator()
    job.return_probs = False
    job.return_top_tokens = 0
    job.device_logit_mask = None
    return job


def bad_logits(value):
    t = torch.randn(1, 1, 128)
    t.view(-1)[3] = value
    return t


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_raises(mode, value, monkeypatch):
    monkeypatch.setattr(job_module, "_nan_check_mode", mode)
    job = make_job(new_tokens = 0)
    with pytest.raises(NonFiniteLogits, match = "EXL3_BC_MLA=0") as exc:
        job.receive_logits(bad_logits(value))
    assert "step 0" in str(exc.value)
    assert "test-job" in str(exc.value)
    assert "non-finite" in str(exc.value)


def test_first_token_only(monkeypatch):
    monkeypatch.setattr(job_module, "_nan_check_mode", 1)
    job = make_job(new_tokens = 5)
    job.receive_logits(bad_logits(float("nan")))    # later step: no raise


def test_every_token_mode_catches_later_steps(monkeypatch):
    monkeypatch.setattr(job_module, "_nan_check_mode", 2)
    job = make_job(new_tokens = 5)
    with pytest.raises(NonFiniteLogits, match = "step 5"):
        job.receive_logits(bad_logits(float("nan")))


def test_disabled_never_raises(monkeypatch):
    monkeypatch.setattr(job_module, "_nan_check_mode", 0)
    for step in (0, 3):
        job = make_job(new_tokens = step)
        job.receive_logits(bad_logits(float("nan")))


def test_healing_step_is_checked(monkeypatch):
    # new_tokens == -1 (token healing) counts as a first token: prefill logits are covered too
    monkeypatch.setattr(job_module, "_nan_check_mode", 1)
    job = make_job(new_tokens = -1)
    with pytest.raises(NonFiniteLogits):
        job.receive_logits(bad_logits(float("nan")))


@pytest.mark.parametrize("mode", [1, 2])
def test_finite_logits_pass_and_sample(mode, monkeypatch):
    monkeypatch.setattr(job_module, "_nan_check_mode", mode)
    job = make_job(new_tokens = 0)
    t = torch.zeros(1, 1, 128)
    t[0, 0, 42] = 10.0
    next_token, _, _, _ = job.receive_logits(t.clone())
    assert next_token.item() == 42
