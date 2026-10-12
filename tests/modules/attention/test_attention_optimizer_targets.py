"""
Attention.optimizer_targets with use_k_as_v (K doubles as V, no v_proj): the module must not build a v_proj and its
optimizer targets must list only the q, k and o projections. Stub projections, no weights or device.
"""

import pytest

from exllamav3.modules.attn import Attention

pytestmark = pytest.mark.nogpu


class FakeLinear:
    def __init__(self, key):
        self.key = key

    def optimizer_targets(self):
        return [self.key]


def test_use_k_as_v_has_no_v_proj():
    m = Attention(config = None, key = "l.attn", layer_idx = 0, hidden_size = 64, head_dim = 16, num_q_heads = 4,
                  num_kv_heads = 2, rope_settings = None, q_proj = FakeLinear("q"), k_proj = FakeLinear("k"),
                  o_proj = FakeLinear("o"), use_k_as_v = True)
    assert m.v_proj is None
    assert m.optimizer_targets() == [[["q"], ["k"], ["o"]]]
