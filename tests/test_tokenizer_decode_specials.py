import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Tokenizer

# gemma4: most HF-special tokens are outside exllamav3's extended-token map and go through the HF decode call
MODEL = os.environ.get("EXL3_TEST_MODEL", "/mnt/str/models/gemma4-12b-it/exl3/4.00bpw_mul1/")
SPECIAL = os.environ.get("EXL3_TEST_SPECIAL", "<|tool>")            # HF-special, not in the extended map
EXTENDED = os.environ.get("EXL3_TEST_EXTENDED")                     # in exllamav3's extended map (auto-picked if unset): splits the HF decode

@unittest.skipUnless(os.path.isdir(MODEL), "model dir not available")
class DecodeSpecialsTest(unittest.TestCase):
    def test_decode_special_tokens_flag(self):
        tok = Tokenizer.from_config(Config.from_directory(MODEL))
        # The HF-special token sits in the segment before an extended-map token, which is the segment
        # decoded through the HF call under test
        ext_piece = EXTENDED or next(iter(tok.extended_id_to_piece.values()))
        ext_id = next(i for i, pc in tok.extended_id_to_piece.items() if pc == ext_piece)
        sid = tok.tokenizer.token_to_id(SPECIAL)
        self.assertIsNotNone(sid, f"{SPECIAL!r} is not a token of this tokenizer")
        self.assertNotIn(sid, tok.extended_id_to_piece, f"{SPECIAL!r} must not be in the extended map for this test")
        ids = tok.encode(f"Hi{SPECIAL}there{ext_piece}x", encode_special_tokens = True)
        seq = ids[0].tolist()
        self.assertIn(sid, seq); self.assertIn(ext_id, seq); self.assertLess(seq.index(sid), seq.index(ext_id))
        with_specials = tok.decode(ids, decode_special_tokens = True)[0]
        without = tok.decode(ids, decode_special_tokens = False)[0]
        self.assertIn(SPECIAL, with_specials)
        self.assertIn(ext_piece, with_specials)
        self.assertNotIn(SPECIAL, without)
        self.assertIn("there", without)

if __name__ == "__main__": unittest.main()
