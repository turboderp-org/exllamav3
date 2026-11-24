import argparse
from types import SimpleNamespace

import pytest

from exllamav3 import Generator, model_init
from testlib.control_vector import write_control_vector

pytestmark = pytest.mark.nogpu


def args(*extra):
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, add_draft_model_args = True)
    return parser.parse_args(["-m", "unused-checkpoint", *extra])


@pytest.mark.parametrize("extra", [["--control_vector_scale", "1"], ["--control_vector_layer_range", "0", "1"]])
def test_options_require_file_before_config_loading(extra):
    with pytest.raises(ValueError, match = "require --control_vector"):
        model_init.init(args(*extra))


@pytest.mark.parametrize("extra", [["-tp"], ["-mtp"], ["-dm", "unused-draft"], ["-ngram", "2"], ["-hdt", "2"]])
def test_incompatible_options_fail_before_weights(tmp_path, monkeypatch, extra):
    config = SimpleNamespace(architecture = "Qwen4ExpForConditionalGeneration", hidden_size = 64,
                             num_hidden_layers = 4, hc_mult = 4, layer_map = None, layer_map_str = None)
    path = tmp_path / "vector.safetensors"
    write_control_vector(config, path)
    monkeypatch.setattr(model_init.Config, "from_directory", lambda *a, **k: config)
    with pytest.raises(ValueError, match = "does not support"):
        model_init.init(args("--control_vector", str(path), *extra))


@pytest.mark.parametrize("options", [{"draft_model": object()}, {"ngram_match_min": 2}, {"hybrid_draft_tokens": 2}])
def test_python_generator_rejects_speculation_before_cache_mutation(options):
    model = SimpleNamespace(control_vector_fingerprint = "active")
    cache = SimpleNamespace(owner_serial = 7)
    with pytest.raises(ValueError, match = "speculative"):
        Generator(model = model, cache = cache, tokenizer = None, **options)
    assert cache.owner_serial == 7


def test_python_generator_rejects_projected_draft():
    draft = SimpleNamespace(control_vector_fingerprint = "active")
    with pytest.raises(ValueError, match = "speculative"):
        Generator(model = SimpleNamespace(), cache = None, tokenizer = None, draft_model = draft)
