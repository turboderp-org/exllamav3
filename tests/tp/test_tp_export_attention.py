"""
TP export/import must round-trip every forward-affecting constructor flag of the attention modules (Attention,
SlidingAttention), split the gate projection by heads or by head channels according to full_gate, and place a
layer with a QSA indexer whole on one rank. Bare modules (no config, fake children from testlib.tp, loading
stubbed), checked against the constructor flags and the head range of the plan.
"""

import pytest

from exllamav3.modules.attn import Attention
from exllamav3.modules.sliding_attn import SlidingAttention
from testlib.tp import DEVICE, FakeChild, ctx, stub_loading

pytestmark = pytest.mark.nogpu

HEAD_DIM, KV_HEADS, GQA = 64, 4, 2


def bare(cls, **flags):
    kw = dict(q_proj = FakeChild("q"), k_proj = FakeChild("k"), v_proj = FakeChild("v"), o_proj = FakeChild("o"))
    if cls is SlidingAttention:
        kw["sliding_window"] = 256
    m = cls(config = None, key = "model.layers.0.attn", layer_idx = 0, hidden_size = 512, head_dim = HEAD_DIM,
            num_q_heads = KV_HEADS * GQA, num_kv_heads = KV_HEADS, rope_settings = None, **kw, **flags)
    m.device = DEVICE
    return m


def roundtrip(cls, first, last, **flags):
    m = bare(cls, **flags)
    exported = m.tp_export(plan = {}, producer = None)
    with stub_loading(cls):
        imported = cls.tp_import(ctx(), exported, {m.key: (first, last, "heads")})
    return exported, imported


@pytest.mark.parametrize("flags", [{"full_gate": True}, {"gate_softplus": True}, {"use_cu_seqlens": True}])
def test_attention_flags_survive_export(flags):
    exported, imported = roundtrip(Attention, 0, 2, **flags)
    for k, v in flags.items():
        assert exported["kwargs"].get(k) == v, f"Attention.tp_export drops {k}"
        assert getattr(imported, k) == v, f"Attention.tp_import loses {k}"
    assert imported.num_kv_heads == 2
    assert imported.num_q_heads == 2 * GQA


@pytest.mark.parametrize("flags", [{"full_gate": True}, {"gate_softplus": True}])
def test_sliding_attention_flags_survive_export(flags):
    exported, imported = roundtrip(SlidingAttention, 1, 3, **flags)
    for k, v in flags.items():
        assert exported["kwargs"].get(k) == v, f"SlidingAttention.tp_export drops {k}"
        assert getattr(imported, k) == v, f"SlidingAttention.tp_import loses {k}"


@pytest.mark.parametrize("full_gate, expect", [
    (False, (True, 2 * GQA, 4 * GQA)),
    (True, (True, 2 * GQA * HEAD_DIM, 4 * GQA * HEAD_DIM)),
])
def test_attention_gate_split_follows_full_gate(full_gate, expect):
    # The g_proj split is applied through tp_import_split on the exported child; capture the split it receives
    seen = {}

    class FakeLinear:
        @staticmethod
        def tp_import_split(local_context, exported, plan, split):
            seen["split"] = split
            return FakeChild()

    m = bare(Attention, full_gate = full_gate)
    exported = m.tp_export(plan = {}, producer = None)
    exported["g_proj"] = {"cls": FakeLinear}
    with stub_loading(Attention):
        Attention.tp_import(ctx(), exported, {m.key: (2, 4, "heads")})
    assert seen["split"] == expect, f"g_proj split wrong for full_gate={full_gate}"


def test_attention_with_qsa_indexer_is_placed_whole():
    # QSA layers run whole on one rank: single-channel allocation with a module-enforced device cap, the indexer
    # travels with the export, and the owner rank gets it back
    class FakeIndexer(FakeChild):
        head_dim = 32
        compress_ratio = 4

        def storage_size(self):
            return 0

        def tp_export(self, plan, producer):
            return {"cls": FakeIndexer, "key": self.key}

        @staticmethod
        def tp_import(local_context, exported, plan):
            return FakeIndexer(exported["key"])

    m = bare(Attention)
    m.qsa_indexer = FakeIndexer("idx")
    exported = m.tp_export(plan = {}, producer = None)
    assert exported.get("qsa_indexer") is not None
    with stub_loading(Attention):
        owner = Attention.tp_import(ctx(), exported, {m.key: (0, KV_HEADS, "heads")})
        stub = Attention.tp_import(ctx(), exported, {m.key: (KV_HEADS, KV_HEADS, "heads")})
    assert owner.qsa_indexer.key == "idx"
    assert stub.qsa_indexer is None
    with pytest.raises(AssertionError):
        Attention.tp_import(ctx(), exported, {m.key: (0, KV_HEADS // 2, "heads")})
