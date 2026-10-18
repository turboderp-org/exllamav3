"""
A tensor larger than the shared arena travels in an overflow segment owned by the producer and released by its
next clear(), never through torch's share_memory_() pickled over the pipe (issue #530: that path keeps a refcount
balanced per pickle, and a command pickled once for several ranks unlinks the segment after two ranks are done
with it, so late ranks cannot open it).
"""

import multiprocessing as mp
import pickle
import time
from multiprocessing import shared_memory

import pytest
import torch

from exllamav3.model.model_tp_shared import SMConsumer, SMProducer


def make_producer(buffer_size = 64 * 1024):
    return SMProducer(buffer_size = buffer_size)


def segment_exists(name):
    try:
        shm = shared_memory.SharedMemory(name = name)
    except FileNotFoundError:
        return False
    shm.close()
    return True


def test_oversized_tensor_roundtrips_through_an_overflow_segment():
    producer = make_producer()
    consumer = SMConsumer(producer, device = 0)
    try:
        big = torch.randn(300, 1024, dtype = torch.bfloat16)   # 600 KiB against a 64 KiB arena
        small = torch.arange(10, dtype = torch.int32)
        d_big = producer.send(big)
        d_small = producer.send(small)
        assert d_big["method"] == "overflow" and d_small["method"] == "buffer"
        assert segment_exists(d_big["shm_name"])
        assert torch.equal(consumer.recv(d_big), big)
        assert torch.equal(consumer.recv(d_small), small)
        # Sliced import to the device, as the weight shards do
        part = consumer.recv(d_big, cuda = True, slice_dim = 0, first = 100, last = 150)
        assert part.device.type == "cuda" and torch.equal(part.cpu(), big[100:150])
        # The segment lives until the producer reuses the arena
        assert segment_exists(d_big["shm_name"])
        producer.clear()
        assert not segment_exists(d_big["shm_name"])
        assert producer.overflow == []
    finally:
        consumer.close()
        producer.close()


def test_overflow_tensor_is_cached_like_arena_tensors():
    producer = make_producer()
    consumer = SMConsumer(producer, device = 0)
    try:
        big = torch.randn(300, 1024, dtype = torch.bfloat16)
        d1 = producer.send(big, cache_id = id(big))
        assert d1["method"] == "overflow"
        got = consumer.recv(d1)
        producer.clear()
        d2 = producer.send(big, cache_id = id(big))
        assert d2["method"] == "cached"
        assert torch.equal(consumer.recv(d2), got)
    finally:
        consumer.close()
        producer.close()


def _late_rank(producer_imp, descriptor_bytes, delay, expected_sum, result_queue):
    """A rank that receives the command late, after earlier ranks have already copied and freed"""
    time.sleep(delay)
    try:
        consumer = SMConsumer(producer_imp, device = 0)
        tensor = consumer.recv(pickle.loads(descriptor_bytes), cuda = True)
        ok = abs(tensor.float().sum().item() - expected_sum) < 1e-3 * abs(expected_sum)
        del tensor
        consumer.close()
        result_queue.put(ok)
    except Exception as e:
        result_queue.put(repr(e))


def test_one_command_pickled_for_three_staggered_ranks():
    """The exact shape of the bug: one pickled descriptor, consumed by three ranks at different times while
    the producer still holds the tensor. Every rank must find the data."""

    producer = make_producer()
    big = torch.randn(300, 1024, dtype = torch.float32)
    desc = producer.send(big)
    assert desc["method"] == "overflow"
    payload = pickle.dumps(desc)
    expected = big.sum().item()

    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    ranks = [
        ctx.Process(target = _late_rank, args = (producer.export(), payload, delay, expected, results))
        for delay in (0.0, 1.5, 3.0)
    ]
    try:
        for r in ranks:
            r.start()
        outcomes = [results.get(timeout = 120) for _ in ranks]
        for r in ranks:
            r.join(timeout = 30)
        assert outcomes == [True, True, True], outcomes
    finally:
        for r in ranks:
            if r.is_alive():
                r.kill()
        producer.close()
