"""
Model-free lifecycle tests for CPUPageCache's tensor-parallel pool management, using fake TP models. The
pure-TP configuration allocates no local slabs and starts no threads, so construction and teardown can be
exercised entirely on CPU.
"""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.generator.cpu_cache import CPUPageCache
from exllamav3.model.model_tp import Model_TPMixin
from exllamav3.model.model_tp_fn import mp_host_cache_alloc, mp_host_cache_free
import exllamav3.model.model_tp_fn as model_tp_fn


class _FakeTPModel:
    """Records the worker-side pool commands a real Model_TPMixin would dispatch to its ranks."""
    loaded_tp = True

    def __init__(self, page_bytes = 4096, fail_alloc = False):
        self.page_bytes = page_bytes
        self.fail_alloc = fail_alloc
        self.allocated_pool_ids = []
        self.pending_frees = []
        self.freed_pool_ids = []

    def tp_host_cache_process_frees(self):
        while self.pending_frees:
            self.freed_pool_ids.append(self.pending_frees.pop())

    def tp_host_cache_page_bytes(self, cache_id):
        return self.page_bytes

    def tp_host_cache_alloc(self, cache_id, pool_id, num_slots):
        # Record before raising to model a TP fan-out where one rank allocated pool_id before another failed.
        self.allocated_pool_ids.append(pool_id)
        if self.fail_alloc:
            raise RuntimeError("injected worker allocation failure")

    def tp_host_cache_free_deferred(self, pool_id):
        self.pending_frees.append(pool_id)


class _FakeTPCache:
    def __init__(self, model):
        self.model = model
        self.layers = {}


def test_tp_pool_rollback_on_init_failure():
    # A failure while allocating the second cache's pools must queue frees for the pools that already
    # succeeded: no finalizer exists yet at that point, so without explicit rollback the first cache's
    # worker-side pinned memory would survive until model teardown.
    model_ok = _FakeTPModel()
    model_bad = _FakeTPModel(fail_alloc = True)
    with pytest.raises(RuntimeError, match = "injected worker allocation failure"):
        CPUPageCache([_FakeTPCache(model_ok), _FakeTPCache(model_bad)], max_size = 1 << 20)
    assert len(model_ok.allocated_pool_ids) == 1
    assert model_ok.freed_pool_ids == model_ok.allocated_pool_ids, \
        f"allocated {model_ok.allocated_pool_ids} but freed {model_ok.freed_pool_ids}"
    assert model_bad.freed_pool_ids == model_bad.allocated_pool_ids, \
        f"partially allocated {model_bad.allocated_pool_ids} but freed {model_bad.freed_pool_ids}"
    assert model_ok.pending_frees == [] and model_bad.pending_frees == []


def test_tp_allocation_fanout_is_drained_before_failure():
    # Worker allocation errors are returned as status values, so the parent sees every rank's reply, frees
    # the attempted pool everywhere, and only then raises. The fake dispatch returns the complete result list
    # that a real drain-safe fan-out produces.
    class _FakeFanout:
        active_devices = [0, 1]

        def __init__(self):
            self.commands = []

        def tp_worker_dispatch_wait_multi(self, devices, fn, args):
            self.commands.append((fn, args))
            if fn is mp_host_cache_alloc:
                return [
                    None,
                    {
                        "device": 1,
                        "type": "RuntimeError",
                        "message": "injected rank allocation failure",
                    },
                ]
            assert fn is mp_host_cache_free
            return [None, None]

    model = _FakeFanout()
    with pytest.raises(RuntimeError, match = "device 1.*injected rank allocation failure"):
        Model_TPMixin.tp_host_cache_alloc(model, cache_id = 10, pool_id = 20, num_slots = 30)
    assert model.commands == [
        (mp_host_cache_alloc, (10, 20, 30)),
        (mp_host_cache_free, (20,)),
    ]


def test_worker_allocation_failure_removes_partial_local_pool(monkeypatch):
    # A rank can fail after pinning some of its cache tensors. It must return an ordinary error status and
    # remove the partially built pool locally; raising would make the parent's multi-rank wait stop early.
    class _FakeLayer:
        def get_tensors(self):
            return [
                SimpleNamespace(shape = (4, 2), dtype = torch.float16),
                SimpleNamespace(shape = (4, 3), dtype = torch.float16),
            ]

    class _FakeModule:
        tp_cache_lookup = {10: _FakeLayer()}

    allocations = []

    def fake_empty(shape, dtype, pin_memory):
        if allocations:
            raise RuntimeError("injected second-tensor allocation failure")
        pool = SimpleNamespace(shape = shape, dtype = dtype, pin_memory = pin_memory)
        allocations.append(pool)
        return pool

    monkeypatch.setattr(model_tp_fn.torch, "empty", fake_empty)
    local_context = {
        "device": 1,
        "kv_modules": [_FakeModule()],
    }
    result = mp_host_cache_alloc(local_context, cache_id = 10, pool_id = 20, num_slots = 30)
    assert result == {
        "device": 1,
        "type": "RuntimeError",
        "message": "injected second-tensor allocation failure",
    }
    assert 20 not in local_context["host_cache_pools"]


def test_failed_deferred_free_remains_queued():
    class _FakeFanout:
        active_devices = [0, 1]
        tp_pending_host_cache_frees = [20]

        def __init__(self):
            self.fail = True

        def tp_worker_dispatch_wait_multi(self, devices, fn, args):
            assert fn is mp_host_cache_free
            assert args == (20,)
            if self.fail:
                raise RuntimeError("injected cleanup failure")
            return [None, None]

    model = _FakeFanout()
    with pytest.raises(RuntimeError, match = "injected cleanup failure"):
        Model_TPMixin.tp_host_cache_process_frees(model)
    assert model.tp_pending_host_cache_frees == [20]

    model.fail = False
    Model_TPMixin.tp_host_cache_process_frees(model)
    assert model.tp_pending_host_cache_frees == []


def test_closed_tier_is_terminal():
    # close() queues the worker pools for release; store()/fetch() afterwards must fail immediately rather
    # than dispatch page copies against pools the ranks may already have dropped.
    model = _FakeTPModel()
    cpc = CPUPageCache([_FakeTPCache(model)], max_size = 1 << 20)
    cpc.close()
    assert model.pending_frees == model.allocated_pool_ids

    class _FakePage:
        phash = b"\x01" * 16
        prev_hash = None
        page_index = 0
        sequence = torch.zeros((1, 4), dtype = torch.long)

    with pytest.raises(RuntimeError, match = "closed"):
        cpc.store(_FakePage(), serial = 1)
    with pytest.raises(RuntimeError, match = "closed"):
        cpc.fetch(b"\x01" * 16, page_index = 0, serial = 2)

    cpc.close()  # idempotent
    assert model.pending_frees == model.allocated_pool_ids
