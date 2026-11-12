import json
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import load_file

from exllamav3 import ControlVector
from testlib.repo_scripts import load_repo_script

convert = load_repo_script("util/convert_control_vector.py").convert

pytestmark = pytest.mark.nogpu


@pytest.fixture
def gguf_file(tmp_path):
    gguf = pytest.importorskip("gguf", reason = "optional conversion dependency")
    def write(tensors, architecture = "controlvector"):
        path = tmp_path / "vector.gguf"
        writer = gguf.GGUFWriter(str(path), architecture)
        for name, value in tensors.items(): writer.add_tensor(name, value)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return path
    return write


@pytest.fixture
def source(tmp_path, gguf_file):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {"hidden_size": 64, "num_hidden_layers": 4, "hc_count": 4},
    }))
    tensors = {f"direction.{i}": np.random.default_rng(i).standard_normal(64).astype(np.float32)
               for i in range(1, 4)}
    return gguf_file(tensors), model, tensors


def test_conversion_preserves_values_and_refuses_overwrite(source, tmp_path):
    path, model, tensors = source
    destination = tmp_path / "vector.safetensors"
    assert convert(path, destination, model) == 3
    copied = load_file(str(destination))
    assert {k: v.tobytes() for k, v in copied.items()} == {k: v.tobytes() for k, v in tensors.items()}
    config = SimpleNamespace(architecture = "Qwen4ExpForConditionalGeneration", hidden_size = 64,
                             num_hidden_layers = 4, hc_mult = 4)
    vector = ControlVector.from_file(config, destination)
    assert [i for i, _, _ in vector.directions] == [1, 2, 3]
    original = destination.read_bytes()
    with pytest.raises(FileExistsError): convert(path, destination, model)
    assert destination.read_bytes() == original
    with pytest.raises(ValueError, match = "outside"):
        convert(path, model / "vector.safetensors", model)


def test_conversion_rejects_model_mismatch(source, tmp_path):
    path, model, _ = source
    config = json.loads((model / "config.json").read_text())
    config["text_config"]["hidden_size"] = 65
    (model / "config.json").write_text(json.dumps(config))
    destination = tmp_path / "bad.safetensors"
    with pytest.raises(ValueError, match = "width"):
        convert(path, destination, model)
    assert not destination.exists()


@pytest.mark.parametrize("case", ["architecture", "dtype", "zero", "nan", "name", "index", "gap", "empty"])
def test_conversion_rejects_invalid_directions(source, gguf_file, tmp_path, case):
    _, model, _ = source
    tensors = {"direction.0": np.ones(64, dtype = np.float32)}
    architecture = "controlvector"
    if case == "architecture": architecture = "llama"
    if case == "dtype": tensors["direction.0"] = np.ones(64, dtype = np.float16)
    if case == "zero": tensors["direction.0"].fill(0)
    if case == "nan": tensors["direction.0"][0] = np.nan
    if case == "name": tensors = {"direction.00": tensors["direction.0"]}
    if case == "index": tensors = {"direction.4": tensors["direction.0"]}
    if case == "gap": tensors["direction.2"] = tensors["direction.0"]
    if case == "empty": tensors = {}
    path = gguf_file(tensors, architecture)
    destination = tmp_path / "bad.safetensors"
    with pytest.raises(ValueError):
        convert(path, destination, model)
    assert not destination.exists()
