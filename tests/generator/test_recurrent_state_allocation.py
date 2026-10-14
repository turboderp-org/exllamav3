from asyncio import CancelledError
from types import SimpleNamespace

import pytest
import torch

from exllamav3.cache import Cache
from exllamav3.generator.generator import Generator
from exllamav3.generator.job import Job
from exllamav3.modules.gated_delta_net import GDNState
from exllamav3.util.tensor import SeqTensor

pytestmark = pytest.mark.nogpu

ALLOCATORS = ["get_new_state", "get_test_state", "new_from_stashed"]
ERRORS = [RuntimeError, MemoryError, torch.OutOfMemoryError, CancelledError, KeyboardInterrupt]


class SlotState:
    def __init__(self, cache, slot, position, **kwargs):
        self.cache = cache
        self.slot = slot
        self.position = position
        self.kwargs = kwargs
        self.frees = 0

    def free(self):
        self.frees += 1
        self.cache.release_state(self)


def make_cache(slots = 4):
    cache = Cache.__new__(Cache)
    cache.num_slots = slots
    cache.recurrent_state_cls = SlotState
    cache.reset_states()
    return cache


def allocate(cache, method):
    args = {"get_new_state": (), "get_test_state": (256,), "new_from_stashed": ({"position": 256}, 256)}
    return getattr(cache, method)(*args[method])


def fail_with(error):
    def fail(*args, **kwargs):
        raise error
    return fail


@pytest.mark.parametrize("method", ALLOCATORS)
@pytest.mark.parametrize("error_type", ERRORS)
def test_failed_construction_returns_slot(method, error_type):
    cache = make_cache()
    held = cache.get_new_state()
    before = list(cache.free_list)
    error = error_type("injected state construction failure")
    cache.recurrent_state_cls = fail_with(error)
    for _ in range(cache.num_slots + 2):
        with pytest.raises(error_type) as caught:
            allocate(cache, method)
        assert caught.value is error
        assert list(cache.free_list) == before
        assert held.slot not in cache.free_list
    cache.recurrent_state_cls = SlotState
    states = [allocate(cache, method) for _ in before]
    assert len({held.slot, *(s.slot for s in states)}) == cache.num_slots
    for state in [held, *states]:
        state.free()
    assert sorted(cache.free_list) == list(range(cache.num_slots))


@pytest.mark.parametrize("method", ALLOCATORS)
def test_successful_construction_preserves_arguments(method):
    cache = make_cache()
    state = allocate(cache, method)
    assert state.cache is cache
    assert state.slot == 0
    assert list(cache.free_list) == [1, 2, 3]
    assert state.position == (0 if method == "get_new_state" else 256)
    expected = {
        "get_new_state": {"clear": True},
        "get_test_state": {"clear": True, "test_state": True},
        "new_from_stashed": {"clear": False, "stashed": {"position": 256}},
    }
    assert state.kwargs == expected[method]
    state.free()
    assert list(cache.free_list) == [0, 1, 2, 3]


@pytest.mark.parametrize("method", ALLOCATORS)
def test_exhausted_pool_still_rejects_allocation(method):
    cache = make_cache(0)
    cache.recurrent_state_cls = fail_with(RuntimeError("constructor must not run"))
    with pytest.raises(AssertionError, match = "no available slots"):
        allocate(cache, method)
    assert list(cache.free_list) == []


def make_generator(target = True, draft = True):
    gen = Generator.__new__(Generator)
    gen.cache = make_cache()
    gen.draft_cache = make_cache()
    gen.target_recurrent = target
    gen.draft_recurrent = draft
    return gen


def allocate_pair(gen, restore):
    if restore:
        return gen.recurrent_state_from_stashed({"target": {"position": 256}, "draft": {"position": 256}}, 256)
    return gen.new_recurrent_state()


@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("side", ["target", "draft"])
@pytest.mark.parametrize("error_type", ERRORS)
def test_paired_allocation_failure_releases_both_sides(restore, side, error_type):
    gen = make_generator()
    held = allocate_pair(gen, restore)
    before = [list(c.free_list) for c in (gen.cache, gen.draft_cache)]
    error = error_type("injected paired state failure")
    cache = gen.cache if side == "target" else gen.draft_cache
    cache.recurrent_state_cls = fail_with(error)
    for _ in range(cache.num_slots + 2):
        with pytest.raises(error_type) as caught:
            allocate_pair(gen, restore)
        assert caught.value is error
        assert [list(c.free_list) for c in (gen.cache, gen.draft_cache)] == before
    cache.recurrent_state_cls = SlotState
    state = allocate_pair(gen, restore)
    assert state.target.slot != held.target.slot
    assert state.draft.slot != held.draft.slot
    state.free()
    held.free()
    assert sorted(gen.cache.free_list) == [0, 1, 2, 3]
    assert sorted(gen.draft_cache.free_list) == [0, 1, 2, 3]


@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("target,draft", [(True, True), (True, False), (False, True)])
def test_paired_allocation_success(restore, target, draft):
    gen = make_generator(target, draft)
    pair = allocate_pair(gen, restore)
    assert (pair.target is not None) == target
    assert (pair.draft is not None) == draft
    assert pair.position == (256 if restore else 0)
    assert len(gen.cache.free_list) == 4 - int(target)
    assert len(gen.draft_cache.free_list) == 4 - int(draft)
    pair.free()
    assert list(gen.cache.free_list) == [0, 1, 2, 3]
    assert list(gen.draft_cache.free_list) == [0, 1, 2, 3]


def test_real_gdn_partial_clear_failure_returns_slot():
    cache = make_cache()
    cache.model = SimpleNamespace(loaded_tp = False)
    cache.recurrent_state_cls = GDNState
    cleared = []
    error = RuntimeError("second layer clear failed")
    cache.recurrent_layers = {
        (0, 0): SimpleNamespace(clear = lambda slot: cleared.append(slot), get_checkpoint_size = lambda: 4),
        (1, 0): SimpleNamespace(clear = fail_with(error), get_checkpoint_size = lambda: 4),
    }
    with pytest.raises(RuntimeError) as caught:
        cache.get_new_state()
    assert caught.value is error
    assert cleared == [0]
    assert list(cache.free_list) == [0, 1, 2, 3]


@pytest.mark.parametrize("draft_failure", [False, True])
def test_job_reaping_after_construction_failure_preserves_slots(draft_failure):
    gen = make_generator(draft = draft_failure)
    gen.active_jobs = []
    error = RuntimeError("injected job allocation failure")
    failing = gen.draft_cache if draft_failure else gen.cache
    failing.recurrent_state_cls = fail_with(error)
    gen.recurrent_cache = object()
    gen.pagetable = SimpleNamespace(deallocate_pages = lambda pages: pages.clear())
    for i in range(6):
        job = Job(input_ids = torch.tensor([[1, 2, 3]]), max_new_tokens = 4)
        job.generator = gen
        job.pagetable = gen.pagetable
        job.all_unique_hashes = set()
        job.sequences[0].allocate_pages = lambda *args: (0, 0, 0, None)
        job.sequences[0].allocated_pages = []
        job.serial_number = i
        gen.active_jobs.append(job)
        with pytest.raises(RuntimeError) as caught:
            job.allocate_pages()
        results = []
        gen.reap_failed_job(job, caught.value, results)
        assert results[0]["error"] is error
        assert job.recurrent_state is None
        assert not gen.active_jobs
        assert sorted(gen.cache.free_list) == [0, 1, 2, 3]
        assert sorted(gen.draft_cache.free_list) == [0, 1, 2, 3]


@pytest.mark.parametrize("restore", [False, True])
def test_rewind_replacement_failure_does_not_double_free(restore):
    gen = make_generator(draft = False)
    gen.tokenizer = SimpleNamespace(get_id_to_piece_list = lambda _: ["", "x", "p"])
    gen.pagetable = SimpleNamespace(deallocate_pages = lambda pages: pages.clear())
    pair = gen.new_recurrent_state()
    pair.target.position = 3
    pair.target.rollback_capacity = lambda: 0
    job = Job(input_ids = torch.tensor([[2, 2]]), max_new_tokens = 16, banned_strings = ["x"])
    job.generator = gen
    job.pagetable = gen.pagetable
    job.recurrent_state = pair
    job.serial_number = 0
    job.held_tokens = SeqTensor((1, 0), dtype = torch.long, seq_dim = -1)
    job.held_probs = SeqTensor((1, 0), dtype = torch.float, seq_dim = -1)
    job.held_k_tokens = SeqTensor((1, 0, 0), dtype = torch.long, seq_dim = 1)
    job.held_k_probs = SeqTensor((1, 0, 0), dtype = torch.float, seq_dim = 1)
    job.held_logits = SeqTensor((1, 0, 3), dtype = torch.float, seq_dim = 1)
    job.new_tokens = 0
    job.max_rq_tokens = 64
    job.rq_margin = 1
    job.sequences[0].kv_position = 2
    job.sequences[0].allocated_pages = []
    job.find_recurrent_stash = lambda _: {"position": 0} if restore else None
    error = RuntimeError("replacement state failed")
    gen.new_recurrent_state = fail_with(error)
    gen.recurrent_state_from_stashed = fail_with(error)
    with pytest.raises(RuntimeError) as caught:
        job.receive_sample(None, torch.tensor([[1]]), None, None, None, [])
    assert caught.value is error
    job.deallocate_pages()
    assert pair.target.frees == 1
    assert job.recurrent_state is None
    assert sorted(gen.cache.free_list) == [0, 1, 2, 3]
