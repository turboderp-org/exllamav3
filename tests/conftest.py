"""
Suite-wide pytest configuration. See tests/README.md for the layout and conventions.

Options
    --device DEV        device for single-GPU tests (default: $EXL3_TEST_DEVICE or cuda:0)
    --devices LIST      comma-separated device indices, e.g. 0,2,3. Under pytest-xdist each worker is pinned to one
                        of them (gw0 -> first, gw1 -> second, ...) and sees it as cuda:0; without xdist the list is
                        what multi-GPU tests may use (default: all visible devices)
    --model-root DIR    root for relative paths in the model registry (default: $EXL3_TEST_MODEL_ROOT)
    --models FILE       extra model registry file (default: $EXL3_TEST_MODELS)
    --slow              also run tests marked slow

Markers (skip when the requirement is not met)
    nogpu               test does not need CUDA; every other test is skipped on machines without a GPU
    multi_gpu(n=2)      needs n devices
    cc(major, minor)    needs compute capability >= major.minor on the test device
    cc_max(major, minor) needs compute capability <= major.minor
    cuda_only / rocm_only
    model(id, ...)      needs these registry models (resolved by the model_dir / model_dirs fixtures)
    models(*tags)       parametrize over every registry model carrying all the tags (fixture: model_id); roles
                        tagged draft or hf_reference are left out unless a tag names them
    hf                  needs transformers
    cpu_flags(*flags)   needs these /proc/cpuinfo flags
    platform(name)      linux | windows
    slow                skipped unless --slow
    vram(gib)           needs at least this much device memory (model registry entries carry their own)

Every directory under tests/ also becomes a marker for the tests below it, so `-m attention` selects the
attention tests across kernels/, modules/, graph/ and e2e/ (files can add topic markers with pytestmark).
"""

import os
import sys

import pytest
import torch

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)

# The source tree (not an installed copy) is what's under test, also in subprocesses tests spawn
for p in (TESTS_DIR, REPO_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = os.pathsep.join([REPO_DIR, TESTS_DIR] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])

from testlib import env
from testlib.models import Registry
_SKIP_DIRS = {"testlib", "__pycache__", "__disk_lru_cache__", ".pytest_cache", ".benchmarks", "_retired"}


def pytest_addoption(parser):
    g = parser.getgroup("exllamav3")
    g.addoption("--device", default = os.environ.get("EXL3_TEST_DEVICE", "cuda:0"),
                help = "device for single-GPU tests (default: $EXL3_TEST_DEVICE or cuda:0)")
    g.addoption("--devices", default = None,
                help = "comma-separated device indices; one per xdist worker, or the set multi-GPU tests may use")
    g.addoption("--model-root", default = os.environ.get("EXL3_TEST_MODEL_ROOT"),
                help = "root directory for relative model registry paths")
    g.addoption("--models", default = os.environ.get("EXL3_TEST_MODELS"),
                help = "extra model registry file")
    g.addoption("--slow", action = "store_true", default = False, help = "also run tests marked slow")


# Registry tags left out of models(...) matrices unless the marker names them
MATRIX_EXCLUDED_TAGS = ("draft", "hf_reference")

_REQUIREMENT_MARKERS = [
    "nogpu: test does not need CUDA",
    "multi_gpu(n=2): needs n CUDA devices",
    "cc(major, minor): needs compute capability >= major.minor",
    "cc_max(major, minor): needs compute capability <= major.minor",
    "cuda_only: skipped on ROCm",
    "rocm_only: skipped on CUDA",
    "model(*ids): needs these registry models",
    "models(*tags, exclude=...): parametrize model_id over registry models with these tags (draft and hf_reference roles excluded unless named)",
    "hf: needs transformers",
    "cpu_flags(*flags): needs these CPU feature flags",
    "platform(name): linux | windows",
    "slow: skipped unless --slow",
    "vram(gib): needs a test device with at least this much memory",
]


def _topic_dirs():
    out = set()
    for dp, dn, _ in os.walk(TESTS_DIR):
        dn[:] = [d for d in dn if d not in _SKIP_DIRS and not d.startswith(".")]
        for d in dn:
            out.add(d)
    return sorted(out)


def _worker_index() -> int | None:
    w = os.environ.get("PYTEST_XDIST_WORKER")
    return int(w[2:]) if w and w.startswith("gw") and w[2:].isdigit() else None


def pytest_configure(config):
    for m in _REQUIREMENT_MARKERS:
        config.addinivalue_line("markers", m)
    reserved = {m.split(":")[0].split("(")[0] for m in _REQUIREMENT_MARKERS}
    for d in _topic_dirs():
        # A directory named like a requirement marker would silently change what -m selects (e.g. -m "not model")
        if d in reserved:
            raise pytest.UsageError(f"tests/ directory name '{d}' collides with the '{d}' marker; rename it")
        config.addinivalue_line("markers", f"{d}: tests under a '{d}' directory")

    # xdist: pin each worker to one device before anything initializes CUDA, so tests see it as cuda:0
    devs = config.getoption("--devices")
    widx = _worker_index()
    if devs and widx is not None:
        pick = [d.strip() for d in devs.split(",") if d.strip()]
        # Indices are torch device indices as the invoking shell sees them, so translate through any existing
        # CUDA_VISIBLE_DEVICES mapping rather than replacing it with raw indices
        visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
        sel = pick[widx % len(pick)]
        os.environ["CUDA_VISIBLE_DEVICES"] = visible[int(sel)] if visible else sel
        config.option.device = "cuda:0"
        config._exl3_devices = [0]
    elif devs:
        config._exl3_devices = [int(d) for d in devs.split(",") if d.strip()]
    else:
        config._exl3_devices = None   # all visible, resolved lazily

    # Module-level consumers (testlib.env.get_test_device) see the same device as the fixtures
    os.environ["EXL3_TEST_DEVICE"] = config.getoption("--device")

    config._exl3_registry = Registry(
        root = config.getoption("--model-root"),
        extra_files = [config.getoption("--models")],
    )


def _devices(config) -> list[int]:
    if config._exl3_devices is None:
        config._exl3_devices = list(range(env.num_devices()))
    return config._exl3_devices


def pytest_collection_modifyitems(config, items):
    for item in items:
        rel = os.path.relpath(str(item.path), TESTS_DIR)
        for part in rel.split(os.sep)[:-1]:
            if part not in _SKIP_DIRS:
                item.add_marker(part)
        # models(...) matrices need checkpoints too, so -m "not model" deselects them
        if item.get_closest_marker("models") and not item.get_closest_marker("model"):
            item.add_marker("model")


def _skip(reason):
    pytest.skip(reason)


def pytest_runtest_setup(item):
    config = item.config
    has_cuda = env.cuda_available()

    if item.get_closest_marker("slow") and not config.getoption("--slow"):
        _skip("slow test (run with --slow)")
    if not item.get_closest_marker("nogpu") and not has_cuda:
        _skip("needs a CUDA/ROCm device")
    if item.get_closest_marker("cuda_only") and env.is_rocm():
        _skip("CUDA-only test")
    if item.get_closest_marker("rocm_only") and not env.is_rocm():
        _skip("ROCm-only test")

    m = item.get_closest_marker("multi_gpu")
    if m:
        n = m.args[0] if m.args else m.kwargs.get("n", 2)
        if len(_devices(config)) < n:
            _skip(f"needs {n} devices")

    dev = config.getoption("--device")
    m = item.get_closest_marker("cc")
    if m and has_cuda and env.compute_capability(dev) < tuple(m.args):
        _skip(f"needs compute capability >= {m.args[0]}.{m.args[1]}")
    m = item.get_closest_marker("cc_max")
    if m and has_cuda and env.compute_capability(dev) > tuple(m.args):
        _skip(f"needs compute capability <= {m.args[0]}.{m.args[1]}")

    if item.get_closest_marker("hf") and not env.has_module("transformers"):
        _skip("needs transformers")
    m = item.get_closest_marker("cpu_flags")
    if m:
        missing = [f for f in m.args if f not in env.cpu_flags()]
        if missing:
            _skip(f"CPU lacks {', '.join(missing)}")
    m = item.get_closest_marker("platform")
    if m and env.platform() != m.args[0]:
        _skip(f"{m.args[0]}-only test")

    m = item.get_closest_marker("vram")
    if m and has_cuda:
        total = torch.cuda.get_device_properties(torch.device(dev)).total_memory / 2**30
        if total < m.args[0]:
            _skip(f"needs {m.args[0]:g} GiB of device memory, {dev} has {total:.0f}")

    m = item.get_closest_marker("model")
    if m:
        reg = config._exl3_registry
        for mid in m.args:
            e = reg.get(mid)
            if not e.available:
                _skip(f"test model '{mid}' not available ({e.path or 'no path / model root'})")


def pytest_generate_tests(metafunc):
    m = metafunc.definition.get_closest_marker("models")
    if m and "model_id" in metafunc.fixturenames:
        reg = metafunc.config._exl3_registry
        # Draft models and the unquantized checkpoints parity tests use as HF references are not standalone test
        # subjects; a matrix includes them only when asked for by tag
        exclude = set(m.kwargs.get("exclude", MATRIX_EXCLUDED_TAGS)) - set(m.args)
        ids = [i for i in reg.ids(*m.args) if not (reg.get(i).tags & exclude)]
        # Module scope: tests are grouped by model, and module-scoped fixtures may depend on model_id (one load
        # per model and file)
        metafunc.parametrize("model_id", ids or [pytest.param(None, marks = pytest.mark.skip("no registry model with tags " + ", ".join(m.args)))],
                             scope = "module")


# Fixtures

@pytest.fixture(scope = "session")
def device(pytestconfig) -> torch.device:
    """The device single-GPU tests run on (--device / $EXL3_TEST_DEVICE)"""
    return torch.device(pytestconfig.getoption("--device"))


@pytest.fixture(scope = "session")
def devices(pytestconfig) -> list[torch.device]:
    """Devices multi-GPU tests may use (--devices, default all visible)"""
    return [torch.device("cuda", i) for i in _devices(pytestconfig)]


@pytest.fixture(scope = "session")
def model_registry(pytestconfig) -> Registry:
    return pytestconfig._exl3_registry


@pytest.fixture
def model_dir(request, model_registry) -> str:
    """Directory of the model named by the test's model() marker (first id), or of model_id when parametrized"""
    mid = request.getfixturevalue("model_id") if "model_id" in request.fixturenames else None
    if mid is None:
        m = request.node.get_closest_marker("model")
        assert m and m.args, "model_dir needs a model(id) marker or a model_id parameter"
        mid = m.args[0]
    e = model_registry.get(mid)
    if not e.available:
        pytest.skip(f"test model '{mid}' not available ({e.path or 'no path / model root'})")
    _require_vram(e, request.config)
    return e.path


def _require_vram(entry, config):
    if entry.vram and env.cuda_available():
        dev = torch.device(config.getoption("--device"))
        total = torch.cuda.get_device_properties(dev).total_memory / 2**30
        if total < entry.vram:
            pytest.skip(f"test model '{entry.id}' needs {entry.vram:g} GiB of VRAM, {dev} has {total:.0f}")


@pytest.fixture
def model_dirs(request, model_registry) -> dict[str, str]:
    """{id: directory} for every id in the test's model() marker"""
    m = request.node.get_closest_marker("model")
    assert m and m.args, "model_dirs needs a model(id, ...) marker"
    return {mid: model_registry.get(mid).path for mid in m.args}


@pytest.fixture(autouse = True)
def _current_cuda_device(request):
    # Triton and some ext launches follow the *current* device rather than the tensors' device. Make the test
    # device current before each test and restore it after, so results don't depend on collection order or on a
    # test that called torch.cuda.set_device() without restoring it
    if env.cuda_available() and not request.node.get_closest_marker("nogpu"):
        dev = torch.device(request.config.getoption("--device"))
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
    yield
    if env.cuda_available():
        torch.cuda.set_device(torch.device(request.config.getoption("--device")))


@pytest.fixture(scope = "module", autouse = True)
def _release_gpu_memory_between_modules():
    # Test modules share one process (and under xdist one worker per GPU), so workspaces a module leaves in the
    # library's global tensor cache, and blocks held by the caching allocator, would otherwise count against every
    # later module, e.g. a model load in e2e/ after a kernel test that cached a prefill-sized workspace
    yield
    tensor = sys.modules.get("exllamav3.util.tensor")
    if tensor is not None:
        tensor.g_tensor_cache.drop_all()
    import gc
    gc.collect()
    if env.cuda_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
