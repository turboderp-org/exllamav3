"""
Model_LSMixin._load_autosplit measures each module's transient VRAM as the peak over a
measuring forward. A first forward can leave persistent allocations behind (the CPU-offload
MoE host's per-device stream state, lazily built statics): they are in memory_allocated from
then on, so they must not also be carried as transient headroom for every later module on the
device. Regression: the double count closed a device ~186 MiB early, and the RuntimeError
raised at the last device dropped the reason.
"""
import os, sys
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.model.model_ls import Model_LSMixin

MiB = 1 << 20


class FakeModule:

    def __init__(self, key, transient, persist = 0):
        self.key = key
        self.caps = {}
        self.transient = transient
        self.persist = persist
        self.device = None
        self.weight = None
        self.persistent = None
        self.forwards = 0

    def can_defer_load(self):
        return False

    def load(self, device, max_chunk_size = None):
        self.device = device
        self.weight = torch.empty(64 * MiB, dtype = torch.uint8, device = device)

    def unload(self):
        self.weight = None
        self.persistent = None

    def prepare_for_device(self, x, params):
        return x

    def forward(self, x, params):
        self.forwards += 1
        if self.persist and self.persistent is None:
            self.persistent = torch.empty(self.persist, dtype = torch.uint8, device = self.device)
        t = torch.empty(self.transient, dtype = torch.uint8, device = self.device)
        del t
        return x

    def __iter__(self):
        return iter(())


class FakeModel(Model_LSMixin):

    def __init__(self, modules):
        self.modules = modules
        self.caps = {}

    def __iter__(self):
        return iter(self.modules)

    def get_layer_instances(self, layer_idx):
        return ()


def _load(modules, use):
    config = SimpleNamespace(
        infer_params = SimpleNamespace(vision_pinned = False),
        stc = SimpleNamespace(
            begin_deferred_load = lambda arena = True: None,
            end_deferred_load = lambda: None,
            abort_deferred_load = lambda: None,
            close = lambda: None,
        ),
    )
    model = FakeModel(modules)
    list(model._load_autosplit(
        progressbar = False,
        reserve_per_device = None,
        use_per_device = [use],
        active_devices = [0],
        max_chunk_size = 256,
        max_output_size = 32,
        max_output_factor = 1,
        callback_sync = None,
        generator = False,
        config = config,
        modules = modules,
        verbose = False,
        max_batch_size = 1,
        cache_weakrefs = {},
        autosplit_no_forward = False,
    ))
    return model


@pytest.fixture(autouse = True)
def _reset_allocator():
    # set_memory_fraction_use budgets on top of the current reservation; a failed load leaves
    # the fraction set
    torch.cuda.empty_cache()
    yield
    torch.cuda.set_per_process_memory_fraction(1.0, device = 0)
    torch.cuda.empty_cache()


def test_first_forward_persistent_allocation_is_not_transient():
    # Four 64 MiB modules with a 128 MiB transient; the first one also keeps 256 MiB from its
    # first forward. Budget covers weights + persistent + one honest transient + caching
    # allocator slack, but not the transient double-counted with the persistent allocation
    modules = [FakeModule("a", 128 * MiB, persist = 256 * MiB)] + \
        [FakeModule(k, 128 * MiB) for k in "bcd"]
    _load(modules, use = (4 * 64 + 256 + 128 + 128) * MiB)
    assert all(m.weight is not None for m in modules)


def test_insufficient_vram_error_names_the_cause():
    # The first module's transient is what closes the device, on a later module
    modules = [FakeModule("a", 128 * MiB)] + [FakeModule(k, 16 * MiB) for k in "bcd"]
    with pytest.raises(RuntimeError, match = "Insufficient VRAM.*no headroom left"):
        _load(modules, use = 300 * MiB)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
