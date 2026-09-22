"""Exercise recursive replay, factor equivalence and interrupted-output recovery on CPU."""

import argparse
import copy
import hashlib
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from util import yaqa_hessians as unrecursed
from util import yaqa_recursive as recursive


class LinearBlock(nn.Module):
    """A nonlinear linear stack with shared inputs and row-dependent metadata."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(8, 12, bias=False)
        self.up_proj = nn.Linear(8, 12, bias=False)
        self.down_proj = nn.Linear(12, 8, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        context: dict[str, list[torch.Tensor]],
        scale: float,
    ) -> tuple[torch.Tensor]:
        x = hidden_states + context["bias"][0] * scale
        update = self.down_proj(self.gate_proj(x).tanh() * self.up_proj(x).sigmoid())
        return (hidden_states + update,)


class LinearModel(nn.Module):
    """Five CPU blocks; alternating positional and keyword hidden-state calls."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.layers = nn.ModuleList([LinearBlock() for _ in range(5)])
        self.lm_head = nn.Linear(8, 16, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, input_ids: torch.Tensor, use_cache: bool) -> SimpleNamespace:
        assert not use_cache
        x = self.embedding(input_ids)
        context = {"bias": [input_ids.unsqueeze(-1).float() / 16]}
        for index, layer in enumerate(self.layers):
            if index % 2:
                x = layer(hidden_states=x, context=context, scale=(index + 1) / 5)[0]
            else:
                x = layer(x, context=context, scale=(index + 1) / 5)[0]
        return SimpleNamespace(logits=self.lm_head(x))


def collect(
    root: Path,
    model: LinearModel,
    rows: list[torch.Tensor],
    mode: str,
    sides: str = "out",
    unweighted: bool = False,
    resume_from: tuple[Path, ...] = (),
    layers: tuple[int, int] = (1, 4),
    head: bool | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run the real collector; replace only model loading and CUDA-only telemetry."""
    candidate = copy.deepcopy(model)
    if mode.startswith("recursive"):
        unrecursed.checkpoint_layers(candidate)
    args = argparse.Namespace(
        device="0",
        hessian_device=None,
        tf32=False,
        sides=sides,
        unweighted=unweighted,
        passes=None,
        samples=2,
        seed=3,
        temperature=0.8,
        model_dir=str(root),
        layers=layers,
        head=sides == "both" if head is None else head,
        pattern=unrecursed.default_pattern,
        key_prefix="model.",
        recursive=str(root / "boundaries") if mode.startswith("recursive") else None,
        dry_run=False,
        out_dir=str(root / mode),
        resume_from=[str(path) for path in resume_from],
        log_interval=100,
        cal_data="fixture",
    )
    with (
        patch.object(unrecursed, "load_model", return_value=candidate),
        patch.object(unrecursed, "load_rows", return_value=rows),
        patch.object(torch.cuda, "synchronize"),
        patch.object(torch.cuda, "max_memory_allocated", return_value=0),
    ):
        unrecursed.main(args)
    return {
        path.name: recursive.load_file(str(path))
        for path in (root / mode).glob("*.safetensors")
    }


def compare(sides: str, unweighted: bool) -> None:
    """Compare packed FP32 factors across odd splits, row lengths and Fisher samples."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = LinearModel().eval().requires_grad_(False)
        rows = [torch.randint(0, 16, (1, length)) for length in (11, 7)]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        expected = collect(root, model, rows, "reference", sides, unweighted)
        actual = collect(root, model, rows, "recursive", sides, unweighted)
        keys = {
            f"model.layers.{layer}.{projection}.safetensors"
            for layer in range(1, 4)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        if sides == "both":
            keys.add("lm_head.safetensors")
        assert actual.keys() == expected.keys() == keys
        for key in keys:
            assert actual[key].keys() == expected[key].keys()
            for side in actual[key]:
                torch.testing.assert_close(
                    actual[key][side], expected[key][side], rtol=0, atol=0
                )
        assert not list((root / "boundaries").iterdir())


def test_recursive_weighted_output_matches_unrecursed() -> None:
    """Preserve weighted Hout across odd splits and row-dependent replay arguments."""
    compare("out", False)


def test_recursive_alternating_factors_and_head_match_unrecursed() -> None:
    """Preserve alternating Hin/Hout updates and the head input factor."""
    compare("both", False)


def test_recursive_unweighted_output_matches_unrecursed() -> None:
    """Preserve the single-pass unweighted estimator without an input-factor pass."""
    compare("out", True)


def test_recursive_unweighted_head_matches_forward_only_input_factor() -> None:
    """Collect head Hin even when decoder weighting would otherwise select backward-only work."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        model = LinearModel().eval().requires_grad_(False)
        rows = [torch.randint(0, 16, (1, n)) for n in (11, 7)]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        expected = collect(root, model, rows, "recursive-head-weighted", layers=(5, 5), head=True)
        actual = collect(
            root, model, rows, "recursive-head-unweighted",
            unweighted=True, layers=(5, 5), head=True,
        )
        assert actual.keys() == expected.keys() == {"lm_head.safetensors"}
        head = actual["lm_head.safetensors"]
        assert head.keys() == {"hin"}
        torch.testing.assert_close(
            head["hin"], expected["lm_head.safetensors"]["hin"], rtol=0, atol=0,
        )


def test_missing_spilled_boundary_propagates_and_cleans_scratch() -> None:
    """Fail rather than continue with missing boundary data, and remove scratch state."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        model = LinearModel().eval().requires_grad_(False)
        rows = [torch.randint(0, 16, (1, 5))]
    load = recursive.load_file

    def remove_boundary(filename: str) -> dict[str, torch.Tensor]:
        path = Path(filename)
        path.unlink()
        return load(filename)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.object(recursive, "load_file", side_effect=remove_boundary):
            try:
                collect(root, model, rows, "recursive")
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("A missing spilled boundary must fail collection")
        assert not list((root / "boundaries").iterdir())


def killed_collection(root: Path) -> None:
    """Kill the real collector during its fourth output write, after one complete layer."""
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = LinearModel().eval().requires_grad_(False)
    rows = [torch.randint(0, 16, (1, n)) for n in (11, 7)]
    save = unrecursed.save_file
    writes = 0

    def interrupt(tensors: dict[str, torch.Tensor], filename: str) -> None:
        nonlocal writes
        writes += 1
        if writes == 4:
            Path(filename).write_bytes(b"interrupted safetensors payload")
            os.kill(os.getpid(), signal.SIGKILL)
        save(tensors, filename)

    with patch.object(unrecursed, "save_file", side_effect=interrupt):
        collect(root, model, rows, "recursive-killed", sides="both", head=False)


def test_killed_collection_resumes_identically_without_touching_saved_files() -> None:
    """Resume after a killed temporary write without changing any preserved file.

    Restarting at a nonzero layer must regenerate the same inputs and sampled gradients; the combined factors must match an uninterrupted reference bitwise, including the separately resumed head.
    """
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy,sys; from pathlib import Path; "
                "runpy.run_path(sys.argv[1])['killed_collection'](Path(sys.argv[2]))",
                str(Path(__file__).resolve()),
                str(root),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            timeout=60,
            check=False,
        )
        assert child.returncode == -signal.SIGKILL
        original = root / "recursive-killed"
        before = {
            p.relative_to(original): hashlib.sha256(p.read_bytes()).digest()
            for p in original.rglob("*")
            if p.is_file()
        }
        completed = {
            p.name: recursive.load_file(str(p)) for p in original.glob("*.safetensors")
        }
        assert set(completed) == {
            f"model.layers.1.{n}.safetensors"
            for n in ("gate_proj", "up_proj", "down_proj")
        }
        assert any(p.name == "result.tmp" for p in before)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            model = LinearModel().eval().requires_grad_(False)
            rows = [torch.randint(0, 16, (1, n)) for n in (11, 7)]
        expected = collect(root, model, rows, "reference", sides="both", head=True)
        remaining = collect(
            root,
            model,
            rows,
            "recursive-resumed",
            sides="both",
            resume_from=(original,),
            layers=(2, 4),
            head=True,
        )
        actual = completed | remaining
        assert actual.keys() == expected.keys()
        for key in expected:
            assert actual[key].keys() == expected[key].keys()
            for side in expected[key]:
                torch.testing.assert_close(
                    actual[key][side], expected[key][side], rtol=0, atol=0
                )
        # A second resume skips every complete tensor, including the head.
        collect(
            root, model, rows, "recursive-resumed", sides="both", resume_from=(original,), head=True
        )
        after = {
            p.relative_to(original): hashlib.sha256(p.read_bytes()).digest()
            for p in original.rglob("*")
            if p.is_file()
        }
        assert after == before


def test_existing_result_cannot_be_overwritten_or_mistaken_for_complete() -> None:
    """Reject malformed packed factors and refuse to replace their final filenames."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        module = nn.Linear(3, 2)
        path = root / "projection.safetensors"
        recursive.save_file({"hout": torch.ones(2)}, str(path))
        before = path.read_bytes()
        try:
            unrecursed.complete_output(path, "projection", module, "out")
        except ValueError:
            pass
        else:
            raise AssertionError("Wrong packed shape was accepted")
        collector = unrecursed.Collector(
            "projection", "projection", module, torch.device("cpu"), False, True
        )
        try:
            collector.save_atomic(str(root), "out")
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing output was overwritten")
        assert path.read_bytes() == before


def test_cli_head_uses_its_own_checkout_and_rejects_empty_output() -> None:
    """Resolve CLI imports to this checkout and reject an empty head as a completed result."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        shadow = root / "other-checkout" / "util"
        shadow.mkdir(parents=True)
        (shadow / "yaqa_hessians.py").write_text(
            "raise RuntimeError('Wrong collector checkout')\n"
        )
        out = root / "output"
        out.mkdir()
        command = (
            "import runpy,sys; from pathlib import Path; import torch; "
            "sys.path.insert(0,sys.argv[1]); runpy.run_path(sys.argv[2]); "
            "from util.yaqa_recursive import Collector; "
            "c=Collector('lm_head','lm_head',torch.nn.Linear(3,2),torch.device('cpu'),True,False); "
            "c.begin_pass('fwd'); c.forward('fwd',torch.ones(2,3),None); "
            "c.end_pass('fwd'); c.save_atomic(sys.argv[3],'out')"
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                command,
                str(shadow.parent),
                str(Path(__file__).resolve().parents[1] / "util" / "yaqa_hessians.py"),
                str(out),
            ],
            cwd=root,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            check=True,
            timeout=60,
        )
        saved = recursive.load_file(str(out / "lm_head.safetensors"))
        torch.testing.assert_close(saved["hin"], torch.ones(6), rtol=0, atol=0)
        empty = root / "lm_head.safetensors"
        recursive.save_file({}, str(empty))
        try:
            unrecursed.complete_output(empty, "lm_head", nn.Linear(3, 2), "out")
        except ValueError:
            pass
        else:
            raise AssertionError("Empty head accepted as complete")
