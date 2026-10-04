"""
Tests for fair scheduling between prompt ingestion and generation (Generator fair_gen_rounds and
fair_chunk_size).

No GPU or model is required: Generator is instantiated via __new__ with stub jobs and counters in
place of the model-facing steps, so only the scheduling logic in iterate() is exercised. The
Job.prefill() tests run the real chunk arithmetic over stub pages with a recording model.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.generator import Generator
from exllamav3.generator.job import Job


class StubJob:
    """Active job that is either generating (prefill done) or still ingesting its prompt."""

    def __init__(self, generating, progress = True):
        self.generating = generating
        self.progress = progress
        self.prefill_chunks = []

    def is_prefill_done(self):
        return self.generating

    def prefill(self, results, chunk_size = None):
        self.prefill_chunks.append(chunk_size)
        if not self.generating and self.progress:
            results.append({"job": self, "stage": "prefill", "eos": False})


class FinishingJob(StubJob):
    """Prefilling job whose prompt completes during the prefill round."""

    def prefill(self, results, chunk_size = None):
        super().prefill(results, chunk_size)
        self.generating = True


def make_generator(jobs, rounds = 1, chunk = None):
    counts = SimpleNamespace(start = 0, checkpoint = 0, draft = 0, gen = 0, leave_after_gen = None)
    generator = Generator.__new__(Generator)
    generator.cache = SimpleNamespace(initialized = True, owner_serial = 1)
    generator.cache_owner_serial = 1
    generator.draft_cache = None
    generator.pending_jobs = []
    generator.active_jobs = list(jobs)
    generator.recurrent_cache = object()
    generator.draft_model = object()
    generator.dflash_draft = False
    generator.mtp_draft = True
    generator.ngram_match_min = 0
    generator.visualizer = None
    generator.fair_gen_rounds = rounds
    generator.fair_chunk_size = chunk

    def start_jobs(results):
        counts.start += 1

    def checkpoint():
        counts.checkpoint += 1

    def draft(results):
        counts.draft += 1
        return None

    def generate(results, draft_tokens = None):
        counts.gen += 1
        results.append({"job": None, "stage": "streaming", "eos": False, "round": counts.gen})
        if counts.leave_after_gen == counts.gen:
            generator.active_jobs = [job for job in generator.active_jobs if not job.generating]

    generator.iterate_start_jobs = start_jobs
    generator.recurrent_checkpoint = checkpoint
    generator.iterate_draftmodel_mtp_gen = draft
    generator.iterate_gen = generate
    return generator, counts


class FairSchedulingIterateTest(unittest.TestCase):

    def test_defaults_run_one_round_under_contention(self):
        generator, counts = make_generator([StubJob(True), StubJob(False)])
        results = generator.iterate()
        self.assertEqual((counts.start, counts.checkpoint, counts.draft, counts.gen), (1, 1, 1, 1))
        self.assertEqual(len([r for r in results if r["stage"] == "streaming"]), 1)
        self.assertEqual([job.prefill_chunks for job in generator.active_jobs], [[None], [None]])

    def test_extra_rounds_while_another_job_ingests_its_prompt(self):
        generator, counts = make_generator([StubJob(True), StubJob(False)], rounds = 4)
        results = generator.iterate()
        self.assertEqual((counts.start, counts.checkpoint, counts.draft, counts.gen), (1, 4, 4, 4))
        self.assertEqual([r["round"] for r in results if r["stage"] == "streaming"], [1, 2, 3, 4])

    def test_single_round_without_a_job_mix(self):
        cases = {
            "one generating": [StubJob(True)],
            "two generating": [StubJob(True), StubJob(True)],
            "one prefilling": [StubJob(False)],
            "two prefilling": [StubJob(False), StubJob(False)],
        }
        for name, jobs in cases.items():
            with self.subTest(name):
                generator, counts = make_generator(jobs, rounds = 6, chunk = 256)
                generator.iterate()
                self.assertEqual(counts.gen, 1)
                self.assertTrue(all(job.prefill_chunks == [None] for job in jobs))

    def test_single_round_when_the_prefill_round_only_walked_cached_pages(self):
        # A requeued job or a prompt cache hit skips cached pages without a forward pass and emits no
        # progress; the generating job then gets one round, as without fair scheduling
        generator, counts = make_generator([StubJob(True), StubJob(False, progress = False)], rounds = 6)
        generator.iterate()
        self.assertEqual(counts.gen, 1)

    def test_extra_rounds_stop_once_the_generating_job_leaves(self):
        generator, counts = make_generator([StubJob(True), StubJob(False)], rounds = 8)
        counts.leave_after_gen = 2
        generator.iterate()
        self.assertEqual(counts.gen, 2)

    def test_extra_rounds_stop_once_the_other_prompt_is_complete(self):
        generator, counts = make_generator([StubJob(True), FinishingJob(False)], rounds = 8)
        generator.iterate()
        self.assertEqual(counts.gen, 1)

    def test_chunk_cap_is_passed_only_under_contention(self):
        generating, prefilling = StubJob(True), StubJob(False)
        generator, counts = make_generator([generating, prefilling], rounds = 3, chunk = 512)
        generator.iterate()
        self.assertEqual(prefilling.prefill_chunks, [512])
        self.assertEqual(generating.prefill_chunks, [512])
        self.assertEqual(counts.gen, 3)

        generator, counts = make_generator([StubJob(False)], rounds = 3, chunk = 512)
        generator.iterate()
        self.assertEqual(generator.active_jobs[0].prefill_chunks, [None])

    def test_failing_prefill_is_still_contained(self):
        class FailingJob(StubJob):
            def prefill(self, results, chunk_size = None):
                raise RuntimeError("prefill failed")

        failing = FailingJob(False)
        generator, counts = make_generator([StubJob(True), failing], rounds = 2, chunk = 512)
        reaped = []
        generator.reap_failed_job = lambda job, error, results: reaped.append((job, str(error)))
        generator.iterate()
        self.assertEqual(reaped, [(failing, "prefill failed")])
        self.assertEqual(counts.gen, 1)


class FairSchedulingArgumentsTest(unittest.TestCase):

    def make_generator(self, **kwargs):
        model = SimpleNamespace(config = SimpleNamespace(vocab_size = 256), caps = {})
        cache = SimpleNamespace(num_slots = 4, reset_states = Mock())
        with patch("exllamav3.generator.generator.PageTable", return_value = SimpleNamespace(max_pages = 16)), \
             patch("exllamav3.generator.generator.ThreadPoolExecutor"):
            return Generator(model, cache, tokenizer = None, **kwargs)

    def test_defaults(self):
        generator = self.make_generator()
        self.assertEqual(generator.fair_gen_rounds, 1)
        self.assertIsNone(generator.fair_chunk_size)

    def test_stores_settings(self):
        generator = self.make_generator(fair_gen_rounds = 6, fair_chunk_size = 512)
        self.assertEqual(generator.fair_gen_rounds, 6)
        self.assertEqual(generator.fair_chunk_size, 512)

    def test_rejects_zero_rounds(self):
        with self.assertRaisesRegex(AssertionError, "fair_gen_rounds"):
            self.make_generator(fair_gen_rounds = 0)

    def test_rejects_unaligned_chunk(self):
        for chunk in (0, 1, 255, 300, 1000):
            with self.subTest(chunk = chunk):
                with self.assertRaisesRegex(AssertionError, "fair_chunk_size"):
                    self.make_generator(fair_chunk_size = chunk)


class StubIds:
    def __init__(self, num_tokens):
        self.num_tokens = num_tokens

    def __len__(self):
        return self.num_tokens

    def torch_slice(self, a, b):
        return torch.arange(a, b, dtype = torch.long).unsqueeze(0)


class StubPage:
    def __init__(self, cached):
        self.kv_position = PAGE_SIZE if cached else 0
        self.can_revert = True
        self.prev_hash = None
        self.phash = b"page"
        self.sequence = torch.zeros((1, PAGE_SIZE), dtype = torch.long)


class StubSequence:
    def __init__(self, num_tokens, cached_pages):
        num_pages = (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE
        self.prefill_complete = False
        self.kv_position = 0
        self.sequence_ids = StubIds(num_tokens)
        self.allocated_pages = [StubPage(i < cached_pages) for i in range(num_pages)]
        self.page_hashes = [b"page"] * num_pages
        self.block_index_tensor = None
        self.multimodal_mask = [False] * num_tokens


def make_job(num_tokens, cached_pages, max_chunk_size = 1024, embeddings = ()):
    forwards = []

    def record_prefill(input_ids, params):
        forwards.append((int(params["cache_seqlens"][0]), input_ids.shape[-1]))

    job = Job.__new__(Job)
    job.time_first_prefill = None
    job.sequences = [StubSequence(num_tokens, cached_pages)]
    job.recurrent_state = None
    job.embeddings = list(embeddings)
    job.cached_pages = 0
    job.cached_tokens = 0
    job.identifier = None
    job.serial_number = 0
    job.alt_rope_freqs = None
    job.pagetable = SimpleNamespace(all_pages = [])
    job.generator = SimpleNamespace(
        max_chunk_size = max_chunk_size,
        model = SimpleNamespace(caps = {}, prefill = record_prefill),
        recurrent_cache = None,
        cache = None,
        draft_model = None,
        mtp_draft = False,
        dflash_draft = False,
    )
    return job, forwards


class FairChunkPrefillTest(unittest.TestCase):
    """Job.prefill() with a reduced chunk: eight cached pages followed by 2048 uncached prompt tokens."""

    def test_reduced_chunk_bounds_the_forward_pass_only(self):
        job, forwards = make_job(num_tokens = 4097, cached_pages = 8)
        seq = job.sequences[0]

        # The cached prefix is still walked one full max_chunk_size window (four pages) per call
        for expected_position in (1024, 2048):
            job.prefill([], chunk_size = 256)
            self.assertEqual(seq.kv_position, expected_position)
            self.assertEqual(forwards, [])

        # The forward pass is bounded by the reduced chunk
        results = []
        job.prefill(results, chunk_size = 256)
        self.assertEqual(forwards, [(2048, 256)])
        self.assertEqual(seq.kv_position, 2304)
        self.assertEqual([r["stage"] for r in results], ["prefill"])
        self.assertEqual(results[0]["curr_progress"], 2304)

    def test_full_chunk_without_a_cap(self):
        job, forwards = make_job(num_tokens = 4097, cached_pages = 8)
        for _ in range(2):
            job.prefill([])
        job.prefill([])
        self.assertEqual(forwards, [(2048, 1024)])

    def test_cap_above_max_chunk_size_is_a_no_op(self):
        job, forwards = make_job(num_tokens = 4097, cached_pages = 8)
        for _ in range(2):
            job.prefill([], chunk_size = 4096)
        job.prefill([], chunk_size = 4096)
        self.assertEqual(forwards, [(2048, 1024)])

    def test_prompts_with_embeddings_keep_the_full_chunk(self):
        job, forwards = make_job(num_tokens = 4097, cached_pages = 8, embeddings = [object()])
        for _ in range(2):
            job.prefill([], chunk_size = 256)
        job.prefill([], chunk_size = 256)
        self.assertEqual(forwards, [(2048, 1024)])

    def test_reduced_chunk_reaches_the_end_of_the_prompt(self):
        job, forwards = make_job(num_tokens = 4097, cached_pages = 8)
        seq = job.sequences[0]
        for _ in range(2 + 8):
            job.prefill([], chunk_size = 256)
        self.assertEqual([f for f in forwards], [(2048 + 256 * i, 256) for i in range(8)])
        self.assertEqual(seq.kv_position, 4096)
        self.assertTrue(seq.prefill_complete)


if __name__ == "__main__":
    unittest.main()
