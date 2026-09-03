"""
TP export/import must round-trip every forward-affecting constructor flag of the attention modules.
Builds bare modules (no config, no children, no weights) so the test runs without a model; loading is
stubbed out because it needs a config and weights, which the kwargs plumbing under test does not.
"""
import sys, os, unittest
from unittest.mock import patch
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.modules.attn import Attention
from exllamav3.modules.sliding_attn import SlidingAttention

HEAD_DIM, KV_HEADS, GQA = 64, 4, 2
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class FakeChild:
    """Stands in for a loaded Linear on both sides of the export/import."""
    def __init__(self, key = "fake"): self.key = key; self.device = DEVICE
    def tp_export(self, plan, producer): return {"cls": FakeChild}
    @staticmethod
    def tp_import_split(local_context, exported, plan, split): return FakeChild()


def _bare(cls, **flags):
    kw = dict(q_proj = FakeChild("q"), k_proj = FakeChild("k"), v_proj = FakeChild("v"), o_proj = FakeChild("o"))
    if cls is SlidingAttention: kw["sliding_window"] = 256
    m = cls(config = None, key = "model.layers.0.attn", layer_idx = 0, hidden_size = 512, head_dim = HEAD_DIM,
            num_q_heads = KV_HEADS * GQA, num_kv_heads = KV_HEADS, rope_settings = None, **kw, **flags)
    m.device = DEVICE
    return m


def _roundtrip(cls, first, last, **flags):
    m = _bare(cls, **flags)
    exported = m.tp_export(plan = {}, producer = None)
    plan = {m.key: (first, last, "heads")}
    with patch.object(cls, "load_local", lambda self, device, **kw: None), patch("torch.cuda.synchronize", lambda: None):
        imported = cls.tp_import({"device": DEVICE, "consumer": None}, exported, plan)
    return exported, imported


class TPExportAttentionTest(unittest.TestCase):

    def test_attention_flags_survive_export(self):
        for flags in ({"full_gate": True}, {"gate_softplus": True}, {"use_cu_seqlens": True}):
            exported, imported = _roundtrip(Attention, 0, 2, **flags)
            for k, v in flags.items():
                self.assertEqual(exported["kwargs"].get(k), v, f"Attention.tp_export drops {k}")
                self.assertEqual(getattr(imported, k), v, f"Attention.tp_import loses {k}")
            self.assertEqual(imported.num_kv_heads, 2)
            self.assertEqual(imported.num_q_heads, 2 * GQA)

    def test_sliding_attention_flags_survive_export(self):
        for flags in ({"full_gate": True}, {"gate_softplus": True}):
            exported, imported = _roundtrip(SlidingAttention, 1, 3, **flags)
            for k, v in flags.items():
                self.assertEqual(exported["kwargs"].get(k), v, f"SlidingAttention.tp_export drops {k}")
                self.assertEqual(getattr(imported, k), v, f"SlidingAttention.tp_import loses {k}")

    def test_attention_gate_split_follows_full_gate(self):
        # The g_proj split is applied through tp_import_split on the exported child; capture the split it receives
        seen = {}
        class FakeLinear:
            @staticmethod
            def tp_import_split(local_context, exported, plan, split):
                seen["split"] = split; return FakeChild()
        for full_gate, expect in ((False, (True, 2 * GQA, 4 * GQA)), (True, (True, 2 * GQA * HEAD_DIM, 4 * GQA * HEAD_DIM))):
            m = _bare(Attention, full_gate = full_gate)
            exported = m.tp_export(plan = {}, producer = None)
            exported["g_proj"] = {"cls": FakeLinear}
            with patch.object(Attention, "load_local", lambda self, device, **kw: None), patch("torch.cuda.synchronize", lambda: None):
                Attention.tp_import({"device": DEVICE, "consumer": None}, exported, {m.key: (2, 4, "heads")})
            self.assertEqual(seen["split"], expect, f"g_proj split wrong for full_gate={full_gate}")

    def test_attention_with_qsa_indexer_refuses_export(self):
        m = _bare(Attention)
        m.qsa_indexer = object()
        with self.assertRaises((AssertionError, NotImplementedError)):
            m.tp_export(plan = {}, producer = None)


if __name__ == "__main__":
    unittest.main()
