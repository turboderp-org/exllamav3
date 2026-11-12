import json
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file

from exllamav3 import Config, ControlVector, Model
from exllamav3.generator.disk_cache import cache_fingerprint
from exllamav3.modules.control_vector import ResidualProjection
from exllamav3.modules.transformer import TransformerBlock
from tests.testlib.control_vector import write_control_vector

pytestmark = pytest.mark.nogpu


@pytest.fixture
def tiny_config(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    text = dict(
        hidden_size = 256, num_hidden_layers = 4, vocab_size = 128,
        num_attention_heads = 4, num_key_value_heads = 2, head_dim = 64,
        full_attention_interval = 2, linear_num_key_heads = 2, linear_num_value_heads = 4,
        linear_key_head_dim = 64, linear_value_head_dim = 64,
        moe_intermediate_size = 128, shared_expert_intermediate_size = 128,
        num_experts = 4, num_experts_per_tok = 2, rms_norm_eps = 1e-6,
        hc_count = 4, indexer_n_heads = 2, indexer_head_dim = 64, indexer_budget = 16,
        ple_layer_ids = [2], ple_embed_dim = 8, ngram_size = 2, heads_per_ngram = 1,
        eos_token_id = 1, max_position_embeddings = 128,
    )
    (directory / "config.json").write_text(json.dumps({
        "architectures": ["Qwen4ExpForConditionalGeneration"], "text_config": text,
    }))
    save_file({}, str(directory / "model.safetensors"))
    return Config.from_directory(str(directory))


@pytest.mark.parametrize("bounds", [(0, 3), (1, 2)])
def test_graph_handoff_and_last_kv_index(tiny_config, tmp_path, bounds):
    config = tiny_config
    path = tmp_path / "vectors" / "vector.safetensors"
    write_control_vector(config, path)
    stock = Model.from_config(config)
    disabled = Model.from_config(config, control_vector = ControlVector.from_file(config, path, scale = 0))
    enabled = Model.from_config(config, control_vector = ControlVector.from_file(config, path, layer_range = bounds))
    signature = lambda model: [(type(m).__name__, m.key, m.layer_idx) for m in model.modules]
    assert signature(stock) == signature(disabled)
    assert signature(stock) == [(type(m).__name__, m.key, m.layer_idx) for m in enabled.modules
                                if not isinstance(m, ResidualProjection)]
    assert enabled.caps["supports_tp"] is False and disabled.caps["supports_tp"] is True
    assert disabled.control_vector_fingerprint is None
    projected = []
    for i, m in enumerate(enabled.modules):
        if isinstance(m, ResidualProjection):
            previous = enabled.modules[i - 1]
            assert isinstance(previous, TransformerBlock)
            assert previous.layer_idx == m.steered_layer_idx
            assert previous.mlp_hc.next_site is None
            projected.append(m.steered_layer_idx)
    assert projected == list(range(bounds[0], bounds[1] + 1))
    assert enabled.modules[enabled.last_kv_module_idx].layer_idx == config.num_hidden_layers - 1
    assert enabled.logit_layer_idx == len(enabled.modules) - 1
    assert len(enabled.fwd_modules) == len(enabled.modules)


def test_unsupported_model_rejects_option(tiny_config, tmp_path):
    config = tiny_config
    path = tmp_path / "vector.safetensors"
    write_control_vector(config, path)
    vector = ControlVector.from_file(config, path)
    config.model_classes["text"] = Model
    with pytest.raises(ValueError, match = "not supported"):
        Model.from_config(config, control_vector = vector)


def test_persistent_cache_identity(tmp_path):
    config = SimpleNamespace(architecture = "Qwen4ExpForConditionalGeneration", hidden_size = 64,
                             num_hidden_layers = 4, hc_mult = 4, config_dict = {}, directory = None)
    path = tmp_path / "vector.safetensors"
    write_control_vector(config, path)
    model = SimpleNamespace(config = config)
    cache = SimpleNamespace(model = model, layers = {})
    stock, stock_parts = cache_fingerprint([cache], [], 0)
    model.control_vector_fingerprint = None
    assert cache_fingerprint([cache], [], 0) == (stock, stock_parts)
    hashes = {stock}
    for scale, bounds in [(1.0, (0, 3)), (0.5, (0, 3)), (1.0, (1, 2))]:
        vector = ControlVector.from_file(config, path, scale = scale, layer_range = bounds)
        model.control_vector_fingerprint = vector.fingerprint
        identity, parts = cache_fingerprint([cache], [], 0)
        assert identity not in hashes
        assert parts["caches"][0]["control_vector"] == vector.fingerprint
        hashes.add(identity)
    write_control_vector(config, path, seed = 99)
    model.control_vector_fingerprint = ControlVector.from_file(config, path).fingerprint
    assert cache_fingerprint([cache], [], 0)[0] not in hashes
