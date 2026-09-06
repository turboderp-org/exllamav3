import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.modules.attn import Attention

class FakeLinear:
    def __init__(self, key): self.key = key
    def optimizer_targets(self): return [self.key]

class AttnOptimizerTargetsTest(unittest.TestCase):
    def test_use_k_as_v_has_no_v_proj(self):
        m = Attention(config = None, key = "l.attn", layer_idx = 0, hidden_size = 64, head_dim = 16, num_q_heads = 4, num_kv_heads = 2,
                      rope_settings = None, q_proj = FakeLinear("q"), k_proj = FakeLinear("k"), o_proj = FakeLinear("o"), use_k_as_v = True)
        self.assertIsNone(m.v_proj)
        self.assertEqual(m.optimizer_targets(), [[["q"], ["k"], ["o"]]])

if __name__ == "__main__": unittest.main()
