"""
Fakes for driving tensor-parallel export/import of bare modules without workers, weights or a config.

    DEVICE                      test device when CUDA is available, else CPU (the export/import plumbing runs on both)
    FakeChild(key)              stands in for a loaded submodule (Linear, norm) on both sides of export/import;
                                imports, whole or split, return a FakeChild with the exported key
    FakeProducer / FakeConsumer the tensor channel: send() wraps a tensor, recv() unwraps it onto DEVICE
    RecordingBackend            records collectives: calls = [("all_reduce", contribution) | ("broadcast", src)],
                                tensors = [(shape, dtype)] of each call's buffer
    ctx(**fields)               a worker local_context dict ({"device": DEVICE, "consumer": None} + fields)
    stub_loading(cls = None)    context manager: torch.cuda.synchronize and (given a class) cls.load_local become
                                no-ops, since loading needs a config and weights that the plumbing under test does not
"""

import contextlib
from unittest.mock import patch

import torch

from testlib.env import cuda_available, get_test_device

DEVICE = get_test_device() if cuda_available() else torch.device("cpu")


class FakeChild:
    caps = {}

    def __init__(self, key = "fake"):
        self.key = key
        self.device = DEVICE

    def __iter__(self):
        yield self

    def tp_export(self, plan, producer):
        return {"cls": FakeChild, "key": self.key}

    @staticmethod
    def tp_import(local_context, exported, plan):
        return FakeChild(exported["key"])

    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        return FakeChild(exported["key"])


class FakeProducer:

    def send(self, t):
        return None if t is None else {"t": t}


class FakeConsumer:

    def recv(self, e, cuda = False, **kwargs):
        return None if e is None else e["t"].to(DEVICE)


class RecordingBackend:

    def __init__(self):
        self.calls = []
        self.tensors = []

    def all_reduce(self, t, contribution = True):
        self.calls.append(("all_reduce", contribution))
        self.tensors.append((tuple(t.shape), t.dtype))

    def broadcast(self, t, src_device):
        self.calls.append(("broadcast", src_device))
        self.tensors.append((tuple(t.shape), t.dtype))


def ctx(**fields) -> dict:
    return {"device": DEVICE, "consumer": None} | fields


@contextlib.contextmanager
def stub_loading(cls = None):
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("torch.cuda.synchronize", lambda *args, **kwargs: None))
        if cls is not None:
            stack.enter_context(patch.object(cls, "load_local", lambda self, device, **kw: None))
        yield
