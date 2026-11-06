"""The long-input probe must not change the special-token mode of the actual encode."""

import pytest
from tokenizers import Tokenizer as HFTokenizer, models, pre_tokenizers

from exllamav3.tokenizer.tokenizer import Tokenizer, _CHUNK_MIN_CHARS

pytestmark = pytest.mark.nogpu
SPECIAL = "<|control|>"


@pytest.fixture(params = [False, True], ids = ["chunkable", "unchunkable"])
def tokenizer(request):
    # Prefix-space insertion makes splitting this tokenizer's text at newlines
    # change its IDs, exercising the probe's single-chunk fallback as well.
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = HFTokenizer(models.BPE(vocab = {c: i for i, c in enumerate(alphabet)}, merges = []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = request.param)
    backend.add_special_tokens([SPECIAL])

    tok = Tokenizer.__new__(Tokenizer)
    tok.tokenizer = backend
    tok.bos_token_id = tok.eos_token_id = None
    tok.pad_token_id = -1
    tok.unspecial_piece_to_id = {}
    tok.missing_special_piece_to_id = {}
    return tok, not request.param


@pytest.mark.parametrize("special", [False, True])
@pytest.mark.parametrize("newlines", [False, True])
def test_first_long_encode_preserves_special_token_mode(tokenizer, special, newlines):
    tokenizer, chunkable = tokenizer
    line = f"text {SPECIAL} text" + ("\n\n" if newlines else " ")
    text = line * (_CHUNK_MIN_CHARS // len(line) + 1)
    tokenizer.tokenizer.encode_special_tokens = not special
    expected = tokenizer.tokenizer.encode(text, add_special_tokens = False).ids
    control_id = tokenizer.tokenizer.token_to_id(SPECIAL)
    assert (control_id in expected) == special

    first = tokenizer.encode(text, encode_special_tokens = special)[0].tolist()
    second = tokenizer.encode(text, encode_special_tokens = special)[0].tolist()

    assert first == expected
    assert second == expected
    assert tokenizer.tokenizer.encode_special_tokens == (not special)
    assert tokenizer._chunking_verified == chunkable


@pytest.mark.parametrize("mode", [False, True])
@pytest.mark.parametrize("failing_probe", [1, 2])
def test_failed_probe_restores_special_token_mode(tokenizer, monkeypatch, mode, failing_probe):
    tokenizer, _ = tokenizer
    tokenizer.tokenizer.encode_special_tokens = mode
    encode_batch = tokenizer.tokenizer.encode_batch
    calls = 0

    def fail_probe(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failing_probe:
            raise RuntimeError("probe failed")
        return encode_batch(*args, **kwargs)

    monkeypatch.setattr(tokenizer.tokenizer, "encode_batch", fail_probe)
    with pytest.raises(RuntimeError, match = "probe failed"):
        tokenizer._chunking_ok()

    assert tokenizer.tokenizer.encode_special_tokens == mode
    assert not hasattr(tokenizer, "_chunking_verified")
