from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from exllamav3.modules.arch_specific.qwen4_exp import Qwen4ExpControlVector
from exllamav3.modules.hyperconnections import GatedResidual
from testlib.control_vector import write_control_vector


@pytest.fixture
def vector(tmp_path):
    config = SimpleNamespace(architecture = "Qwen4ExpForConditionalGeneration", hidden_size = 64,
                             num_hidden_layers = 4, hc_mult = 4, layer_map = None, layer_map_str = None)
    path = tmp_path / "vector.safetensors"
    tensors, metadata = write_control_vector(config, path)
    return config, path, tensors, metadata


def module(config, path, device = "cpu", scale = 0.3):
    spec = Qwen4ExpControlVector.from_file(config, path, scale = scale, layer_range = (1, 2))
    result = spec.modules(config, "model")[1]
    result.load(torch.device(device))
    return result


def reference(x, direction, scale):
    x, direction = x.double(), direction.double()
    return x - scale * (x * direction).sum(dim = -1, keepdim = True) * direction


@pytest.mark.nogpu
def test_normalization_and_independent_streams(vector):
    config, path, tensors, _ = vector
    spec = Qwen4ExpControlVector.from_file(config, path, scale = 0.7, layer_range = (1, 2))
    assert [i for i, _, _ in spec.directions] == [1, 2]
    for i, p in spec.modules(config, "model").items():
        weighted = tensors[f"direction.{i}"] * 0.7
        norm = weighted.double().norm()
        assert torch.equal(p.direction_cpu, (weighted.double() / norm).float())
        assert p.scale == norm.float().item()
        p.scale = 1.0
        p.load(torch.device("cpu"))
        x = torch.tensor([1., -2., 3., -4.]).reshape(1, 1, 4, 1) * p.direction
        assert p.forward(x, {}).abs().max() < 2e-6


@pytest.mark.nogpu
def test_cpu_reference_layout_and_handoff(vector):
    config, path, _, _ = vector
    p = module(config, path)
    x = torch.randn(2, 4, 9, config.hidden_size).transpose(1, 2)
    assert not x.is_contiguous()
    expected = reference(x, p.direction, p.scale)
    params = {"gr_weighted": object()}
    actual = p.forward(x, params)
    torch.testing.assert_close(actual.double(), expected, rtol = 3e-6, atol = 3e-6)
    assert "gr_weighted" not in params
    empty = torch.empty(1, 0, config.hc_mult, config.hidden_size)
    assert p.forward(empty, {}) is empty


@pytest.mark.nogpu
def test_model_lifetime_snapshot_and_identity(vector, tmp_path):
    config, path, _, _ = vector
    a = Qwen4ExpControlVector.from_file(config, path)
    copy = tmp_path / "moved.safetensors"
    copy.write_bytes(path.read_bytes())
    assert Qwen4ExpControlVector.from_file(config, copy).fingerprint == a.fingerprint
    assert Qwen4ExpControlVector.from_file(config, path, scale = 0.5).fingerprint != a.fingerprint
    assert Qwen4ExpControlVector.from_file(config, path, layer_range = (1, 2)).fingerprint != a.fingerprint
    p = a.modules(config, "model")[1]
    original = p.direction_cpu.clone()
    # Modules own separate tensor copies and the spec owns only immutable bytes.
    p.direction_cpu.zero_()
    assert torch.equal(a.modules(config, "model")[1].direction_cpu, original)
    write_control_vector(config, path, seed = 43)
    assert Qwen4ExpControlVector.from_file(config, path).fingerprint != a.fingerprint
    assert torch.equal(a.modules(config, "model")[1].direction_cpu, original)
    with pytest.raises(FrozenInstanceError):
        a.fingerprint = "changed"


@pytest.mark.nogpu
def test_direct_construction_owns_data_and_derives_identity(vector):
    config, path, _, _ = vector
    original = Qwen4ExpControlVector.from_file(config, path)
    mutable = [[i, bytearray(data), strength] for i, data, strength in original.directions]
    copy = Qwen4ExpControlVector(original.hidden_size, original.num_hidden_layers, original.hc_count, mutable)
    mutable[0][1][:] = bytes(len(mutable[0][1]))
    mutable.clear()
    assert copy == original
    with pytest.raises(TypeError):
        Qwen4ExpControlVector(
            original.hidden_size, original.num_hidden_layers, original.hc_count,
            original.directions, fingerprint = "unrelated",
        )
    changed = replace(original, directions = tuple((i, data, strength * 0.5)
                                                  for i, data, strength in original.directions))
    assert changed.fingerprint != original.fingerprint


@pytest.mark.nogpu
@pytest.mark.parametrize("fault", ["empty", "duplicate", "unordered", "outside", "bytes", "zero", "nan",
                                  "norm", "strength", "infinity", "dimension"])
def test_direct_construction_rejects_invalid_data(vector, fault):
    config, path, _, _ = vector
    spec = Qwen4ExpControlVector.from_file(config, path)
    entries = list(spec.directions)
    i, data, strength = entries[0]
    if fault == "empty": entries = []
    elif fault == "duplicate": entries[1] = entries[0]
    elif fault == "unordered": entries.reverse()
    elif fault == "outside": entries[0] = (-1, data, strength)
    elif fault == "bytes": entries[0] = (i, b"bad", strength)
    elif fault in ("zero", "nan", "norm"):
        value = torch.frombuffer(bytearray(data), dtype = torch.float32)
        if fault == "zero": value.zero_()
        elif fault == "nan": value[0] = float("nan")
        else: value.mul_(2)
        entries[0] = (i, value.numpy().tobytes(), strength)
    elif fault == "strength": entries[0] = (i, data, -1)
    elif fault == "infinity": entries[0] = (i, data, float("inf"))
    with pytest.raises(ValueError):
        Qwen4ExpControlVector(
            0 if fault == "dimension" else spec.hidden_size,
            spec.num_hidden_layers, spec.hc_count, entries,
        )


@pytest.mark.nogpu
@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -1.0])
def test_invalid_strength(vector, scale):
    config, path, _, _ = vector
    with pytest.raises(ValueError, match = "scale"):
        Qwen4ExpControlVector.from_file(config, path, scale = scale)


@pytest.mark.nogpu
@pytest.mark.parametrize("bounds", [(-1, 2), (1, 4), (3, 1), (True, 2)])
def test_invalid_layer_range(vector, bounds):
    config, path, _, _ = vector
    with pytest.raises(ValueError, match = "range"):
        Qwen4ExpControlVector.from_file(config, path, layer_range = bounds)


@pytest.mark.nogpu
@pytest.mark.parametrize("fault", ["metadata", "shape", "dtype", "nan", "zero", "name", "missing"])
def test_invalid_vector_file(vector, fault):
    config, path, tensors, metadata = vector
    if fault == "metadata": metadata["hc_count"] = "1"
    elif fault == "shape": tensors["direction.1"] = torch.ones(65)
    elif fault == "dtype": tensors["direction.1"] = tensors["direction.1"].half()
    elif fault == "nan": tensors["direction.1"][0] = float("nan")
    elif fault == "zero": tensors["direction.1"].zero_()
    elif fault == "name": tensors["wrong.1"] = tensors.pop("direction.1")
    elif fault == "missing": del tensors["direction.1"]
    save_file(tensors, str(path), metadata = metadata)
    with pytest.raises(ValueError):
        Qwen4ExpControlVector.from_file(config, path, layer_range = (0, 3))


@pytest.mark.nogpu
def test_invalid_runtime_and_unsupported_configuration(vector):
    config, path, _, _ = vector
    spec = Qwen4ExpControlVector.from_file(config, path)
    config.layer_map = [0, 1, 1, 3]
    with pytest.raises(ValueError, match = "remapping"):
        spec.modules(config, "model")
    config.layer_map = None
    p = module(config, path)
    x = torch.zeros(1, 2, 4, 64)
    for bad in [x.half(), x[..., 0], x[:, :, :3]]:
        with pytest.raises(ValueError, match = "FP32"):
            p.forward(bad, {})
    for params in [{"capture": True}, {"quant_preserve": True}]:
        with pytest.raises(ValueError, match = "inference-only"):
            p.forward(x, params)
    with pytest.raises(ValueError, match = "FP32"):
        p.forward(x, {}, out_dtype = torch.half)
    p.unload()
    with pytest.raises(RuntimeError, match = "not loaded"):
        p.forward(x, {})
    config.architecture = "LlamaForCausalLM"
    with pytest.raises(ValueError, match = "Qwen4Exp"):
        Qwen4ExpControlVector.from_file(config, path)


@pytest.mark.parametrize("width,streams,batch,tokens", [(64, 1, 1, 1), (257, 4, 2, 9),
                                                      (2560, 4, 2, 8), (2560, 4, 1, 2048)])
def test_cuda_reference(device, tmp_path, width, streams, batch, tokens):
    config = SimpleNamespace(architecture = "Qwen4ExpForConditionalGeneration", hidden_size = width,
                             num_hidden_layers = 4, hc_mult = streams)
    path = tmp_path / "vector.safetensors"
    write_control_vector(config, path)
    p = module(config, path, device, scale = 0.01)
    x = torch.randn(batch, tokens, streams, width)
    expected = reference(x, p.direction_cpu, p.scale)
    actual = p.forward(x.to(device), {}).cpu().double()
    torch.testing.assert_close(actual, expected, rtol = 3e-6, atol = 3e-6)


@pytest.mark.parametrize("weighted", [False, True])
def test_cuda_graph_replay(device, vector, weighted):
    config, path, _, _ = vector
    p = module(config, path, device)
    if weighted:
        p.next_site = GatedResidual(None, "next", 4, 64, 1e-6)
        p.next_site.w_h = torch.randn(4 * 64, device = device).half()
    x = torch.randn(1, 1, 4, 64, device = device)
    p.forward(x, {})
    graph = torch.cuda.CUDAGraph()
    params = {}
    with torch.cuda.graph(graph):
        result = p.forward(x, params)
    for _ in range(2):
        source = torch.randn_like(x)
        x.copy_(source)
        graph.replay()
        expected = reference(source.cpu(), p.direction_cpu, p.scale)
        torch.testing.assert_close(result.cpu().double(), expected, rtol = 3e-6, atol = 3e-6)
        if weighted:
            expected_copy = result.view(1, 4, 64) * p.next_site.w_h.float().view(4, 64)
            torch.testing.assert_close(params["gr_weighted"][2], expected_copy, rtol = 0, atol = 0)


@pytest.mark.parametrize("rows,weights", [(1, "gpu"), (2, "gpu"), (8, "gpu"),
                                        (9, "gpu"), (2, "cpu"), (2, "unloaded")])
def test_weighted_handoff_and_fallback(device, vector, rows, weights):
    config, path, _, _ = vector
    p = module(config, path, device)
    nxt = GatedResidual(None, "next", 4, 64, 1e-6)
    if weights != "unloaded":
        nxt.w_h = torch.randn(4 * 64, device = device if weights == "gpu" else "cpu").half()
    p.next_site = nxt
    x = torch.randn(rows, 1, 4, 64, device = device)
    expected = reference(x.cpu(), p.direction_cpu, p.scale)
    params = {"gr_weighted": object()}
    result = p.forward(x, params)
    torch.testing.assert_close(result.cpu().double(), expected, rtol = 3e-6, atol = 3e-6)
    if rows <= GatedResidual.FUSED_MAX_R and weights == "gpu":
        streams = result.view(rows, 4, 64)
        copy = nxt._weighted_copy(streams, params)
        torch.testing.assert_close(copy, streams * nxt.w_h.float().view(4, 64), rtol = 0, atol = 0)
        assert nxt._weighted_copy(streams, params) is None
    else:
        assert "gr_weighted" not in params


def test_pending_residual_is_applied_before_projection(device, vector):
    config, path, _, _ = vector
    p = module(config, path, device)
    x = torch.randn(1, 2, 4, 64, device = device)
    y = torch.randn(1, 2, 64, device = device)
    post = torch.randn(1, 2, 4, device = device)
    comb = torch.eye(4, device = device).expand(1, 2, 4, 4).contiguous()
    updated = x + post.unsqueeze(-1) * y.unsqueeze(-2)
    expected = reference(updated.cpu(), p.direction_cpu, p.scale)
    params = {"hc_pending": (x, y, post, comb), "gr_weighted": object()}
    result = p.forward(x, params)
    assert "hc_pending" not in params and "gr_weighted" not in params
    torch.testing.assert_close(result.cpu().double(), expected, rtol = 3e-6, atol = 3e-6)
