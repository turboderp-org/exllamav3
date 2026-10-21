"""
Prompt ingestion alongside generating jobs (Generator shared_chunk_size and prefill_cooldown), and the
recurrent checkpoint rule that lets chunk sizes vary.

No GPU or model is required: Generator is instantiated via __new__ with stub jobs and stubbed model-facing
steps, so only the scheduling in iterate() is exercised; Job.is_checkpoint_boundary() runs on a stub job.
"""

from types import SimpleNamespace
import pytest
from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.generator import Generator
from exllamav3.generator.job import Job
from unittest.mock import Mock, patch

pytestmark = pytest.mark.nogpu


class StubJob:
    """Active job that is either generating (prefill done) or still ingesting its prompt"""

    def __init__(self, name, generating):
        self.name = name
        self.generating = generating
        self.chunks = []   # chunk_size argument of each prefill() call

    def is_prefill_done(self):
        return self.generating

    def prefill(self, results, chunk_size = None):
        self.chunks.append(chunk_size)


def make_generator(jobs, shared_chunk_size = 2048, prefill_cooldown = 0):
    counts = SimpleNamespace(gen = 0)
    g = Generator.__new__(Generator)
    g.cache = SimpleNamespace(initialized = True, owner_serial = None)
    g.draft_cache = None
    g.pending_jobs = []
    g.active_jobs = list(jobs)
    g.recurrent_cache = None
    g.draft_model = None
    g.dflash_draft = False
    g.mtp_draft = False
    g.ngram_match_min = 0
    g.hybrid_draft_tokens = 0
    g.visualizer = None
    g.disk_page_cache = None
    g.max_chunk_size = 2048
    g.shared_chunk_size = shared_chunk_size
    g.prefill_cooldown = prefill_cooldown
    g.prefill_cooldown_left = 0
    g.prefill_rotation = 0
    g.iterate_start_jobs = lambda results: None
    def generate(results, draft_tokens = None):
        counts.gen += 1
    g.iterate_gen = generate
    return g, counts


def run(g, iterations):
    for _ in range(iterations):
        g.iterate()


def test_defaults_prefill_every_job_every_iteration():
    a, b, c = StubJob("a", True), StubJob("b", False), StubJob("c", False)
    g, counts = make_generator([a, b, c])
    run(g, 3)
    assert counts.gen == 3
    assert a.chunks == [None] * 3 and b.chunks == [None] * 3 and c.chunks == [None] * 3


def test_shared_chunk_serves_one_prompt_per_iteration_in_turns():
    a, b, c = StubJob("a", True), StubJob("b", False), StubJob("c", False)
    g, counts = make_generator([a, b, c], shared_chunk_size = 512)
    run(g, 4)
    assert counts.gen == 4
    assert b.chunks == [512, 512] and c.chunks == [512, 512]
    assert a.chunks == [512] * 4   # a generating job keeps its bookkeeping call


def test_cooldown_spaces_prefill_passes_between_generation_steps():
    a, b = StubJob("a", True), StubJob("b", False)
    g, counts = make_generator([a, b], prefill_cooldown = 2)
    run(g, 7)
    # One pass, then two generation-only iterations, repeated
    assert counts.gen == 7
    assert b.chunks == [None, None, None] and a.chunks == [None] * 7


def test_no_generating_job_means_full_chunks_for_everyone():
    b, c = StubJob("b", False), StubJob("c", False)
    g, counts = make_generator([b, c], shared_chunk_size = 512, prefill_cooldown = 3)
    run(g, 2)
    assert b.chunks == [None, None] and c.chunks == [None, None]


def test_contention_ending_restores_full_chunks_and_clears_cooldown():
    a, b = StubJob("a", True), StubJob("b", False)
    g, counts = make_generator([a, b], shared_chunk_size = 512, prefill_cooldown = 3)
    run(g, 1)
    assert b.chunks == [512]
    g.active_jobs.remove(a)   # the generating job finishes during the cooldown
    run(g, 2)
    assert b.chunks == [512, None, None]
    g.active_jobs.append(StubJob("d", True))
    run(g, 1)
    assert b.chunks == [512, None, None, 512]   # a fresh contention starts with a pass, not a wait


def test_prefill_failure_is_still_contained():
    class Failing(StubJob):
        def prefill(self, results, chunk_size = None):
            raise RuntimeError("boom")
    a, b = StubJob("a", True), Failing("b", False)
    g, counts = make_generator([a, b], shared_chunk_size = 256)
    reaped = []
    g.reap_failed_job = lambda job, e, results: reaped.append((job, str(e)))
    run(g, 1)
    assert reaped == [(b, "boom")] and counts.gen == 1


def make_real_generator(**kwargs):
    model = SimpleNamespace(config = SimpleNamespace(vocab_size = 256), caps = {})
    cache = SimpleNamespace(num_slots = 4, max_history = 0, reset_states = Mock())
    with patch("exllamav3.generator.generator.PageTable", return_value = SimpleNamespace(max_pages = 16)), \
         patch("exllamav3.generator.generator.ThreadPoolExecutor"):
        return Generator(model, cache, tokenizer = None, **kwargs)


@pytest.mark.parametrize("max_chunk_size, expected", [(2048, 1024), (768, 512), (256, 256), (1000, 512)])
def test_default_shared_chunk_is_half_the_chunk_rounded_up_to_a_page(max_chunk_size, expected):
    g = make_real_generator(max_chunk_size = max_chunk_size)
    assert g.shared_chunk_size == expected and g.prefill_cooldown == 4


def test_explicit_shared_chunk_and_cooldown_are_kept_and_validated():
    g = make_real_generator(max_chunk_size = 2048, shared_chunk_size = 2048, prefill_cooldown = 0)
    assert g.shared_chunk_size == 2048 and g.prefill_cooldown == 0
    for bad in (0, 100, 4096):
        with pytest.raises(AssertionError, match = "shared_chunk_size"):
            make_real_generator(max_chunk_size = 2048, shared_chunk_size = bad)
    with pytest.raises(AssertionError, match = "prefill_cooldown"):
        make_real_generator(prefill_cooldown = -1)


def make_job(kv_position, prompt_len, last_checkpoint = None, cached_pages = 0, max_chunk_size = 2048,
             interval = 2048, interval_pp = 32768):
    return SimpleNamespace(
        sequences = [SimpleNamespace(kv_position = kv_position, sequence_ids = [0] * (prompt_len + 1))],
        generator = SimpleNamespace(
            max_chunk_size = max_chunk_size,
            recurrent_checkpoint_interval = interval,
            recurrent_checkpoint_interval_pp = interval_pp,
        ),
        last_recurrent_checkpoint_pos = last_checkpoint,
        cached_pages = cached_pages,
    )


def test_checkpoint_rule_triggers_once_the_interval_is_exceeded_from_the_last_checkpoint():
    prompt = 100000
    # Chunk ends that do not fall on the interval grid still qualify once the interval has elapsed
    assert not Job.is_checkpoint_boundary(make_job(768 * 42, prompt))         # 32256 < 32768
    assert Job.is_checkpoint_boundary(make_job(768 * 43, prompt))             # 33024
    assert not Job.is_checkpoint_boundary(make_job(768 * 43 + 768, prompt, last_checkpoint = 768 * 43))
    assert Job.is_checkpoint_boundary(make_job(768 * 86, prompt, last_checkpoint = 768 * 43))
    # Exact grid positions behave as before
    assert Job.is_checkpoint_boundary(make_job(32768, prompt))
    assert Job.is_checkpoint_boundary(make_job(65536, prompt, last_checkpoint = 32768))
    assert not Job.is_checkpoint_boundary(make_job(65536 - 2048, prompt, last_checkpoint = 32768))


def test_checkpoint_rule_measures_from_the_cached_prefix_and_needs_a_page_boundary():
    prompt = 100000
    assert not Job.is_checkpoint_boundary(make_job(32768, prompt, cached_pages = 4))
    assert Job.is_checkpoint_boundary(make_job(32768 + 4 * PAGE_SIZE, prompt, cached_pages = 4))
    assert not Job.is_checkpoint_boundary(make_job(32768 + 1, prompt))
    assert not Job.is_checkpoint_boundary(make_job(40000, prompt))


def test_checkpoint_rule_uses_the_generation_interval_near_the_end_of_the_prompt_and_override():
    prompt = 10000
    # Within two chunks of the prompt end the finer generation interval applies
    assert Job.is_checkpoint_boundary(make_job(8192, prompt, last_checkpoint = 6144))
    assert not Job.is_checkpoint_boundary(make_job(7936, prompt, last_checkpoint = 6144))
    # Early in the prompt the prompt-ingestion interval applies
    assert not Job.is_checkpoint_boundary(make_job(4096, 100000, last_checkpoint = 2048))
    # An explicit interval is a plain grid test
    assert Job.is_checkpoint_boundary(make_job(4096, prompt, last_checkpoint = 3840), override_interval = PAGE_SIZE)
