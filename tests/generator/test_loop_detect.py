"""
LoopDetector contract on synthetic token streams (no model): exact and approximate loops are reported once they
reach the documented thresholds, varied text and repetitions separated by new text are not, transcribing distinct
prompt items is exempt while cycling through them again is a loop, and Job wires the detector up (enabling forms,
priming, carried across requeues).
"""

import random

import pytest
import torch

from exllamav3.generator.job import Job
from exllamav3.generator.loop_detect import LoopDetector

pytestmark = pytest.mark.nogpu


def _varied(rng, n):
    return [rng.randrange(1000, 60000) for _ in range(n)]


def _first_detection(det, tokens):
    for i, t in enumerate(tokens):
        if det.feed(t):
            return i
    return None


def test_exact_loop_reported_after_threshold():
    rng = random.Random(0)
    unit = _varied(rng, 40)
    lead = _varied(rng, 500)
    det = LoopDetector()
    hit = _first_detection(det, lead + unit * 200)
    assert hit is not None
    # Exact repetition: exact_reps periods and exact_len tokens past the start of the first copy
    assert hit - len(lead) >= det.exact_len
    assert hit - len(lead) < det.exact_len + 2 * len(unit)
    assert det.period == len(unit)


def test_varied_text_not_reported():
    det = LoopDetector()
    assert _first_detection(det, _varied(random.Random(1), 60000)) is None


def test_approximate_loop_reported():
    rng = random.Random(2)
    unit = _varied(rng, 300)
    tokens = _varied(rng, 500)
    for _ in range(60):
        tokens += [rng.choice(unit) if rng.random() < 0.02 else t for t in unit]   # 2% substitutions per copy
    det = LoopDetector()
    hit = _first_detection(det, tokens)
    assert hit is not None and hit - 500 >= det.min_len


def test_new_text_between_copies_breaks_run():
    # The same block repeated many times, but with fresh text (longer than the gap allowance) after each copy, as
    # when a model restates something between rounds of new work
    rng = random.Random(3)
    unit = _varied(rng, 300)
    tokens = []
    for _ in range(60):
        tokens += unit + _varied(rng, 40)
    assert _first_detection(LoopDetector(), tokens) is None


def test_prompt_transcription_exempt_cycling_reported():
    rng = random.Random(4)
    base = _varied(rng, 300)
    options = []
    for _ in range(4):
        o = list(base)
        for _ in range(5):
            o[rng.randrange(len(o))] = rng.randrange(1000, 60000)
        options.append(o)
    prompt = [t for o in options for t in [7, 8, 9] + o]

    # Transcribing each option once, then continuing: distinct prompt items, not a loop
    det = LoopDetector()
    det.prime(prompt)
    once = [t for o in options for t in [7, 8, 9] + o] + _varied(rng, 2000)
    assert _first_detection(det, once) is None

    # Cycling through the same options over and over is a loop
    det = LoopDetector()
    det.prime(torch.tensor([prompt]))
    cycling = [t for _ in range(12) for o in options for t in [7, 8, 9] + o]
    assert _first_detection(det, cycling) is not None


def test_tiny_vocabulary_needs_near_exact_repetition():
    # Long non-repeating text over four symbols recurs at short distances by structure alone
    rng = random.Random(5)
    assert _first_detection(LoopDetector(), [rng.choice((11, 12, 13, 14)) for _ in range(20000)]) is None
    # ... but an exact loop over the same alphabet is reported
    assert _first_detection(LoopDetector(), [11, 12, 13, 14, 12] * 1000) is not None


def test_feed_many_and_reset():
    det = LoopDetector()
    tokens = [5, 6, 7, 8] * 1000
    hits = det.feed_many(torch.tensor(tokens))
    assert hits is not None and len(hits) == 1 and det.detected
    assert det.feed_many([1, 2, 3]) == hits                 # detection latches
    det.reset()
    assert not det.detected and det.feed_many([1, 2, 3]) is None


def test_job_wiring():
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    assert Job(input_ids = ids).loop_detector is None
    assert Job(input_ids = ids, stop_on_loop = False).loop_detector is None

    job = Job(input_ids = ids, stop_on_loop = True)
    assert isinstance(job.loop_detector, LoopDetector) and job.loop_detector._bloom is not None   # primed

    assert isinstance(Job(input_ids = ids, stop_on_loop = (300, 3)).loop_detector, LoopDetector)  # legacy form
    assert Job(input_ids = ids, stop_on_loop = {"min_len": 4096}).loop_detector.min_len == 4096

    # A requeued job continues the same detector instead of starting (and priming) a new one
    det = job.loop_detector
    det.feed_many([9, 10, 11])
    requeued = Job(input_ids = ids, stop_on_loop = True, rq_state = {"loop_detector": det, "rq_new_tokens": 3})
    assert requeued.loop_detector is det and det._t == 2
