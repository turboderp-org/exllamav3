"""
Tokenizer.decode(decode_special_tokens = ...): an HF-special token outside exllamav3's extended-token map is
decoded through the HF decode call, which has to honor the flag (shown when True, dropped when False) while the
extended-map token that splits the sequence into segments is still emitted and the surrounding text survives.
Checked on the gemma4 tokenizer (role "swa"), where most HF-special tokens are outside the extended map.
"""

import pytest

from exllamav3 import Config, Tokenizer

pytestmark = pytest.mark.nogpu

SPECIAL = "<|tool>"         # HF-special in the gemma4 vocabulary, not in the extended map


@pytest.mark.model("swa")
def test_decode_special_tokens_flag(model_dir):
    tok = Tokenizer.from_config(Config.from_directory(model_dir))

    # The HF-special token sits in the segment before an extended-map token, which is the segment decoded
    # through the HF call under test
    ext_id, ext_piece = next(iter(tok.extended_id_to_piece.items()))
    sid = tok.tokenizer.token_to_id(SPECIAL)
    assert sid is not None, f"{SPECIAL!r} is not a token of this tokenizer"
    assert sid not in tok.extended_id_to_piece, f"{SPECIAL!r} must not be in the extended map for this test"

    ids = tok.encode(f"Hi{SPECIAL}there{ext_piece}x", encode_special_tokens = True)
    seq = ids[0].tolist()
    assert sid in seq
    assert ext_id in seq
    assert seq.index(sid) < seq.index(ext_id)

    with_specials = tok.decode(ids, decode_special_tokens = True)[0]
    without = tok.decode(ids, decode_special_tokens = False)[0]
    assert SPECIAL in with_specials
    assert ext_piece in with_specials
    assert SPECIAL not in without
    assert "there" in without
