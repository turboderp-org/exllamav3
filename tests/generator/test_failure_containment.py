"""
Regression tests for per-job failure containment in Generator.

A job whose allocate_pages() or prefill() raises must not take the generator
down with it: the failure is contained by Generator.reap_failed_job, which
releases the job's resources, drops it from the active set and emits an error
result carrying the standard serial/stage/eos contract. Without containment
the exception escapes to AsyncGenerator._run_iteration, which latches
self.error permanently and makes the generator unusable for every subsequent
job.

No GPU or model is required: Generator is instantiated via __new__ and given
stub jobs, so the test exercises only the scheduling/cleanup logic.
"""

import logging

import pytest
import torch

from exllamav3 import ArgmaxSampler
from exllamav3.generator.generator import Generator
from exllamav3.generator.async_generator import AsyncGenerator

pytestmark = pytest.mark.nogpu


class StubPagetable:
    def num_unreferenced_pages(self):
        return 1 << 30


class StubSequence:
    pass


class StubJob:
    def __init__(self, name, fail_in = None):
        self.name = name
        self.fail_in = fail_in
        self.sequences = [StubSequence()]
        self.serial_number = None
        self.identifier = name
        self.deallocated = False

    def current_new_pages_required(self):
        return 1

    def activate(self):
        pass

    def allocate_pages(self):
        if self.fail_in == "allocate_pages":
            raise RuntimeError(f"allocation failure for {self.name}")

    def prefill(self, results):
        if self.fail_in == "prefill":
            raise RuntimeError(f"prefill failure for {self.name}")

    def deallocate_pages(self):
        if self.fail_in == "deallocate_pages":
            raise RuntimeError(f"deallocation failure for {self.name}")
        self.deallocated = True


def make_generator(pending, active = (), max_batch_size = 16):
    generator = Generator.__new__(Generator)
    generator.pagetable = StubPagetable()
    generator.pending_jobs = list(pending)
    generator.active_jobs = list(active)
    generator.max_batch_size = max_batch_size
    return generator


def test_reap_result_contract():
    job = StubJob("a")
    job.serial_number = 7
    generator = make_generator([], active = [job])
    results = []
    error = RuntimeError("boom")
    generator.reap_failed_job(job, error, results)
    assert len(results) == 1
    r = results[0]
    assert r["job"] is job
    assert r["serial"] == 7
    assert r["stage"] == "error"
    assert r["eos"]
    assert r["error"] is error
    assert job not in generator.active_jobs
    assert job.deallocated


def test_failing_allocate_pages_is_contained():
    bad = StubJob("bad", fail_in = "allocate_pages")
    good = StubJob("good")
    bad.serial_number = 1
    good.serial_number = 2
    generator = make_generator([bad, good])
    results = []
    generator.iterate_start_jobs(results)
    assert bad not in generator.active_jobs
    assert good in generator.active_jobs
    assert good not in generator.pending_jobs
    assert bad not in generator.pending_jobs
    error_results = [r for r in results if r.get("stage") == "error"]
    assert len(error_results) == 1
    assert error_results[0]["job"] is bad
    assert isinstance(error_results[0]["error"], RuntimeError)
    started = [r for r in results if r.get("stage") == "started"]
    assert len(started) == 1
    assert started[0]["job"] is good


def test_deallocate_failure_is_contained_and_logged():
    job = StubJob("a", fail_in = "deallocate_pages")
    job.serial_number = 3
    generator = make_generator([], active = [job])
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    capture = Capture()
    logging.getLogger("exllamav3.generator.generator").addHandler(capture)
    try:
        generator.reap_failed_job(job, RuntimeError("boom"), [])
    finally:
        logging.getLogger("exllamav3.generator.generator").removeHandler(capture)
    assert job not in generator.active_jobs
    assert any("stranded" in r.getMessage() for r in records)


def test_failing_prefill_is_contained_in_iterate():
    bad = StubJob("bad", fail_in = "prefill")
    good = StubJob("good")
    for i, job in enumerate((bad, good)):
        job.serial_number = i
    generator = make_generator([], active = [bad, good])
    generator.cache = type("C", (), {"initialized": True})()
    generator.draft_cache = None
    generator.recurrent_cache = None
    generator.draft_model = None
    generator.ngram_match_min = 0
    generator.visualizer = None
    generator.iterate_gen = lambda results: None
    results = generator.iterate()
    assert bad not in generator.active_jobs
    assert good in generator.active_jobs
    error_results = [r for r in results if r.get("stage") == "error"]
    assert len(error_results) == 1
    assert error_results[0]["job"] is bad


def test_async_deliver_error_as_raw_exception():
    job = object()
    delivered = []

    class StubAsyncJob:
        def put_result(self, result):
            delivered.append(result)

    wrapper = AsyncGenerator.__new__(AsyncGenerator)
    wrapper.jobs = {job: StubAsyncJob()}
    error = RuntimeError("boom")
    wrapper.deliver_results([{"job": job, "serial": 0, "stage": "error", "eos": True, "error": error}])
    assert len(delivered) == 1
    assert delivered[0] is error
    assert job not in wrapper.jobs

    # Normal results still pass through as dicts; eos removes the job
    job2 = object()
    wrapper.jobs = {job2: StubAsyncJob()}
    wrapper.deliver_results([{"job": job2, "serial": 1, "stage": "streaming", "eos": False}])
    assert isinstance(delivered[1], dict)
    assert job2 in wrapper.jobs
    wrapper.deliver_results([{"job": job2, "serial": 1, "stage": "eos", "eos": True}])
    assert job2 not in wrapper.jobs


def test_sync_generate_raises_on_error_result():
    generator = Generator.__new__(Generator)

    def fake_enqueue(job):
        job.serial_number = 0
        return 0

    remaining = iter([True, True, False])
    error = RuntimeError("sync-raise-probe")

    def fake_iterate():
        return [{
            "job": None,
            "serial": 0,
            "stage": "error",
            "eos": True,
            "error": error,
        }]

    generator.enqueue = fake_enqueue
    generator.num_remaining_jobs = lambda: next(remaining)
    generator.iterate = fake_iterate
    generator.clear_queue = lambda: None

    class StubTokenizer:
        def encode(self, p, **kwargs):
            return torch.tensor([[1, 2, 3]], dtype = torch.long)

    generator.tokenizer = StubTokenizer()

    with pytest.raises(RuntimeError) as ctx:
        generator.generate(
            "hello",
            max_new_tokens = 8,
            sampler = ArgmaxSampler(),
        )
    assert ctx.value is error
