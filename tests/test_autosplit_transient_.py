"""_load_autosplit must not count persistent allocations left by a measuring forward as transient
headroom, must keep the recurring autosplit_extra_measure transient, must not hold two dummy states
while re-measuring, must free the re-measure input when it OOMs, and must report the OOM cause."""
import os, sys
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
from exllamav3.model import model_ls
from exllamav3.model.model_ls import Model_LSMixin

MiB = 1 << 20


class FakeSubmodule:
    # Mirrors BlockSparseMLP.autosplit_extra_measure: one-time persistent state, transient every call

    def __init__(self, persist, transient):
        self.persist = persist
        self.transient = transient
        self.persistent = None
        self.device = None

    def autosplit_extra_measure(self, params):
        if self.persist and self.persistent is None:
            self.persistent = torch.empty(self.persist, dtype = torch.uint8, device = self.device)
        t = torch.empty(self.transient, dtype = torch.uint8, device = self.device)
        del t


class FakeModule:

    def __init__(self, key, transient, persist = 0, extra_transient = 0):
        self.key = key
        self.caps = {}
        self.transient = transient
        self.device = None
        self.weight = None
        self.sub = FakeSubmodule(persist, extra_transient)
        self.forwards = 0

    def can_defer_load(self):
        return False

    def load(self, device, max_chunk_size = None):
        self.device = device
        self.sub.device = device
        self.weight = torch.empty(64 * MiB, dtype = torch.uint8, device = device)

    def unload(self):
        self.weight = None
        self.sub.persistent = None

    def prepare_for_device(self, x, params):
        return x.to(self.device)

    def forward(self, x, params):
        self.forwards += 1
        t = torch.empty(self.transient, dtype = torch.uint8, device = self.device)
        del t
        # Real modules return a new state; the input dies when the caller drops it
        return x.clone()

    def __iter__(self):
        return iter((self.sub,))


class FakeModel(Model_LSMixin):

    def __init__(self, modules, state_bytes = 256):
        self.modules = modules
        self.caps = {}
        self.state_bytes = state_bytes

    def __iter__(self):
        return iter(self.modules)

    def get_layer_instances(self, layer_idx):
        return ()

    def default_load_shape_dtype(self, chunk_size):
        return (1, self.state_bytes), torch.uint8


def _load(modules, use, state_bytes = 256):
    config = SimpleNamespace(
        infer_params = SimpleNamespace(vision_pinned = False),
        stc = SimpleNamespace(
            begin_deferred_load = lambda arena = True: None,
            end_deferred_load = lambda: None,
            abort_deferred_load = lambda: None,
            close = lambda: None,
        ),
    )
    model = FakeModel(modules, state_bytes)
    list(model._load_autosplit(
        progressbar = False,
        reserve_per_device = None,
        use_per_device = [use],
        active_devices = [0],
        max_chunk_size = state_bytes,
        max_output_size = state_bytes,
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
    # set_memory_fraction_use budgets on top of the current reservation; a failed load leaves it set
    torch.cuda.empty_cache()
    yield
    torch.cuda.set_per_process_memory_fraction(1.0, device = 0)
    torch.cuda.empty_cache()


def test_extra_measure_persistent_allocation_is_not_transient():
    # Four 64 MiB modules, 128 MiB transient each; the first keeps 256 MiB from autosplit_extra_measure.
    # Budget: weights + persistent + one transient + slack, not the transient double-counted with it
    modules = [FakeModule("a", 128 * MiB, persist = 256 * MiB)] + \
        [FakeModule(k, 128 * MiB) for k in "bcd"]
    _load(modules, use = (4 * 64 + 256 + 128 + 128) * MiB)
    assert all(m.weight is not None for m in modules)
    assert modules[0].forwards == 2 and all(m.forwards == 1 for m in modules[1:])


def test_remeasure_does_not_hold_two_states():
    # 256 MiB state and persistent block (whole cached blocks, no allocator carving). Budget: weight +
    # persistent + running state + one forward's clone + slack; a second live state would not fit
    state = 256 * MiB
    modules = [FakeModule("a", 0, persist = 256 * MiB)]
    _load(modules, use = (64 + 256 + 2 * 256 + 64) * MiB, state_bytes = state)
    assert modules[0].forwards == 2


def test_insufficient_vram_error_names_the_cause():
    # The first module's transient is what closes the device, on a later module
    modules = [FakeModule("a", 128 * MiB)] + [FakeModule(k, 16 * MiB) for k in "bcd"]
    with pytest.raises(RuntimeError, match = "Insufficient VRAM.*no headroom left"):
        _load(modules, use = 300 * MiB)


def test_failed_remeasure_frees_its_input(monkeypatch):
    # Pass 2 OOMs inside forward. At the failure path's free_mem() the failed device must hold only what
    # unload() cannot drop; a live pass-2 input would be state-sized residue carried to the next device
    class OomOnSecondForward(FakeModule):
        def forward(self, x, params):
            if self.forwards == 1:
                self.transient = 1 << 40
            return super().forward(x, params)

    real_free_mem = model_ls.free_mem
    seen = []
    def spy_free_mem():
        seen.append(torch.cuda.memory_allocated(0))
        real_free_mem()
    monkeypatch.setattr(model_ls, "free_mem", spy_free_mem)

    modules = [OomOnSecondForward("a", 0, persist = 256 * MiB)]
    with pytest.raises(RuntimeError, match = "Insufficient VRAM"):
        _load(modules, use = 2048 * MiB, state_bytes = 256 * MiB)
    assert modules[0].forwards == 2
    assert seen and seen[0] < 16 * MiB


def test_remeasure_keeps_extra_measure_transient():
    # a: 64 MiB forward transient, 256 MiB persistent plus 128 MiB recurring from autosplit_extra_measure.
    # After b loads, 384 MiB resident + 128 MiB headroom exceeds 480 MiB only if the recurring part was kept
    modules = [FakeModule("a", 64 * MiB, persist = 256 * MiB, extra_transient = 128 * MiB), FakeModule("b", 16 * MiB)]
    with pytest.raises(RuntimeError, match = "Insufficient VRAM.*no headroom left"):
        _load(modules, use = 480 * MiB)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
