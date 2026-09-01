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

import os
import sys
import unittest
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.generator.generator import Generator


class StubPagetable:
    def num_unreferenced_pages(self):
        return 1 << 30


class StubSequence:
    pass


class StubJob:
    def __init__(self, name, fail_in=None):
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


def make_generator(pending, active=(), max_batch_size=16):
    generator = Generator.__new__(Generator)
    generator.pagetable = StubPagetable()
    generator.pending_jobs = list(pending)
    generator.active_jobs = list(active)
    generator.max_batch_size = max_batch_size
    return generator


class FailureContainmentTest(unittest.TestCase):
    def test_reap_result_contract(self):
        job = StubJob("a")
        job.serial_number = 7
        generator = make_generator([], active=[job])
        results = []
        error = RuntimeError("boom")
        generator.reap_failed_job(job, error, results)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertIs(r["job"], job)
        self.assertEqual(r["serial"], 7)
        self.assertEqual(r["stage"], "error")
        self.assertTrue(r["eos"])
        self.assertIs(r["error"], error)
        self.assertNotIn(job, generator.active_jobs)
        self.assertTrue(job.deallocated)

    def test_failing_allocate_pages_is_contained(self):
        bad = StubJob("bad", fail_in="allocate_pages")
        good = StubJob("good")
        bad.serial_number = 1
        good.serial_number = 2
        generator = make_generator([bad, good])
        results = []
        generator.iterate_start_jobs(results)
        self.assertNotIn(bad, generator.active_jobs)
        self.assertIn(good, generator.active_jobs)
        self.assertNotIn(good, generator.pending_jobs)
        self.assertNotIn(bad, generator.pending_jobs)
        error_results = [r for r in results if r.get("stage") == "error"]
        self.assertEqual(len(error_results), 1)
        self.assertIs(error_results[0]["job"], bad)
        self.assertIsInstance(error_results[0]["error"], RuntimeError)
        started = [r for r in results if r.get("stage") == "started"]
        self.assertEqual(len(started), 1)
        self.assertIs(started[0]["job"], good)

    def test_deallocate_failure_is_contained_and_logged(self):
        job = StubJob("a", fail_in="deallocate_pages")
        job.serial_number = 3
        generator = make_generator([], active=[job])
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
        self.assertNotIn(job, generator.active_jobs)
        self.assertTrue(any("stranded" in r.getMessage() for r in records))


if __name__ == "__main__":
    unittest.main(verbosity=2)
