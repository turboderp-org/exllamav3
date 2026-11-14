"""Projection through the real Qwen4Exp loader, generator and persistent cache; no downloaded weights."""
import pytest
import torch

from exllamav3 import Config
from testlib.control_vector import write_control_vector
from testlib.e2e import assert_logits_agree, forward_logits, greedy, load_model, noise_floor, prefill_decode_logits
from testlib.tiny_models import make_qwen4_checkpoint


@pytest.fixture(scope = "module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen4_projection")
    model = make_qwen4_checkpoint(root / "model")
    vector = root / "vector.safetensors"
    write_control_vector(Config.from_directory(model), vector)
    return model, vector


def load(checkpoint, device, *options):
    model, _ = checkpoint
    return load_model(model, device, "-ambs", "2", "-chunk_size", "256", "-rs32",
                      *options, cache_tokens = 2048)


def projection(checkpoint, scale = "0.03", bounds = (0, 3)):
    return ["--control_vector", str(checkpoint[1]), "--control_vector_scale", scale,
            "--control_vector_layer_range", str(bounds[0]), str(bounds[1])]


def ids(length, seed = 17):
    return torch.randint(3, 250, (1, length), generator = torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("bounds", [(0, 3), (1, 2)])
def test_chunked_decode_and_batch_match_forward(checkpoint, device, bounds):
    with load(checkpoint, device, *projection(checkpoint, bounds = bounds)) as lm:
        seqs = [ids(609), ids(609, 23)]
        seqs[1][:, :256] = seqs[0][:, :256]
        expected = [forward_logits(lm, seq)[0] for seq in seqs]
        floors = [noise_floor(lm, seq, 8, full) for seq, full in zip(seqs, expected)]
        actual = prefill_decode_logits(lm, seqs, 8, max_batch_size = 2, max_chunk_size = 256)
        for ref, got, floor in zip(expected, actual, floors):
            assert_logits_agree(ref[-9:], got, floor, f"projection {bounds}")


def test_disabled_and_zero_scale_match_stock(checkpoint, device):
    seq = ids(129)
    with load(checkpoint, device) as lm:
        reference = forward_logits(lm, seq)
        signature = [(type(m), m.key) for m in lm.model.modules]
    with load(checkpoint, device, *projection(checkpoint, scale = "0")) as lm:
        assert [(type(m), m.key) for m in lm.model.modules] == signature
        assert lm.model.control_vector_fingerprint is None
        actual = forward_logits(lm, seq)
    torch.testing.assert_close(actual, reference, rtol = 1e-4, atol = 1e-4)


def test_prefix_and_disk_restore_are_vector_specific(checkpoint, device, tmp_path):
    seq = ids(800)
    continuation = ids(8, 29)
    identities = []
    reference = None
    for scale in ("0.03", "0.03", "0.06"):
        with load(checkpoint, device, *projection(checkpoint, scale = scale)) as lm:
            gen = lm.generator(max_batch_size = 2, max_chunk_size = 256,
                               recurrent_checkpoint_interval = 256, recurrent_cache_size = 64 << 20,
                               cpu_cache_size = 64 << 20, disk_cache_dir = str(tmp_path / "cache"),
                               disk_cache_size = 128 << 20, disk_cache_checkpoint_interval = 256)
            try:
                identities.append(gen.disk_page_cache.fingerprint)
                _, logits, first = greedy(gen, seq, 8, return_logits = True, forced_tokens = continuation)
                _, reused_logits, second = greedy(gen, seq, 8, return_logits = True, forced_tokens = continuation)
                assert second["eos"]["cached_pages"] >= 3
                assert_logits_agree(logits, reused_logits, tag = "prefix reuse")
                if len(identities) == 1:
                    assert first["eos"]["cached_pages"] == 0
                    reference = logits
                elif len(identities) == 2:
                    assert first["eos"]["cached_pages"] >= 3
                    assert gen.disk_page_cache.metrics["page_reads"] > 0
                    assert_logits_agree(reference, logits, tag = "disk restore")
                else:
                    assert first["eos"]["cached_pages"] == 0
            finally:
                gen.close()
    assert identities[0] == identities[1] != identities[2]
