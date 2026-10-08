"""
Running a test-module function in a fresh interpreter.

Some state is fixed per process: the CPU MoE instruction-set cap (EXL3_MOE_CPU_MAX_ISA, read once at static
init), layout switches read at import, the prompt cache of a loaded model. Tests that compare such settings run
each one in a child process:

    result = run_isolated(worker, arg1, arg2, env = {"EXL3_MOE_CPU_MAX_ISA": "avx2"})

`worker` must be a module-level function of a test module (or any importable file); the child loads that file
by path, calls the function with the (picklable) arguments and hands back its return value through torch.save,
so tensors and plain containers of them come back as they were. The child inherits the parent's environment
(incl. PYTHONPATH with the source tree and $EXL3_TEST_DEVICE), updated with `env`.

Children that load a model through model_init (which places layers over every visible device) are pinned to the
test device with `device_env(device)`.
"""

import importlib.util
import inspect
import os
import pickle
import subprocess
import sys
import tempfile

import torch

_MODULE_NAME = "_exl3_isolated_module"


def run_isolated(func, *args, env: dict | None = None, timeout: float | None = 1800, **kwargs):
    """Call func(*args, **kwargs) in a fresh interpreter with the environment updated by `env`, return its result.
    A failing child raises AssertionError with the tail of its output"""
    path = os.path.abspath(inspect.getsourcefile(func))
    name = func.__name__
    assert func.__qualname__ == name, f"{func.__qualname__}: only module-level functions can run isolated"
    with tempfile.TemporaryDirectory(prefix = "exl3_isolated_") as tmp:
        args_file = os.path.join(tmp, "args.pkl")
        out_file = os.path.join(tmp, "out.pt")
        with open(args_file, "wb") as f:
            pickle.dump((args, kwargs), f)
        child_env = dict(os.environ, **{k: str(v) for k, v in (env or {}).items()})
        r = subprocess.run(
            [sys.executable, "-c", "from testlib.isolated import _child_main; _child_main()",
             path, name, args_file, out_file],
            env = child_env, capture_output = True, text = True, timeout = timeout,
        )
        assert r.returncode == 0 and os.path.exists(out_file), \
            f"{name} failed in a child process (exit {r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-6000:]}"
        return torch.load(out_file, weights_only = False)


def _child_main():
    path, name, args_file, out_file = sys.argv[1:5]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    with open(args_file, "rb") as f:
        args, kwargs = pickle.load(f)
    torch.save(getattr(module, name)(*args, **kwargs), out_file)


def device_env(device) -> dict[str, str]:
    """Environment that makes a child process see only `device` (as cuda:0). Respects a CUDA_VISIBLE_DEVICES the
    parent already runs under (pytest-xdist workers)"""
    device = torch.device(device)
    assert device.type == "cuda", device
    index = device.index or 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = visible.split(",")[index].strip() if visible else str(index)
    return {"CUDA_VISIBLE_DEVICES": physical, "EXL3_TEST_DEVICE": "cuda:0"}
