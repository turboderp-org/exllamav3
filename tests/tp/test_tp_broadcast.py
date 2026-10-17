"""
Single-owner TP modules must finish with a broadcast from the owner instead of an all-reduce:
Module.tp_single_owner reads the plan (one non-empty slice = owner; split or differently owned parts = None),
tp_collect routes accordingly, tp_import sets tp_owner for the stub and the owner alike, and a stub's forward
issues the broadcast with a zero buffer of the owner's output shape. Bare modules with fake children
(testlib.tp), checked against the collectives a recording backend sees.
"""

from types import SimpleNamespace

import pytest
import torch

from exllamav3.modules.attn import Attention
from exllamav3.modules.mla_attn import MLAttention
from exllamav3.modules.module import Module
from exllamav3.modules.ple import PLELayer
from testlib.tp import DEVICE, FakeChild, FakeConsumer, FakeProducer, RecordingBackend, ctx, stub_loading

pytestmark = pytest.mark.nogpu


def plan_ctx(plan, devices, device):
    return ctx(device = device, plan = plan, active_devices = devices)


def test_single_owner_from_plan():
    plan = {0: {"a": (0, 8, "heads"), "b": (0, 4, "ch"), "c": (0, 0, "ch")},
            1: {"a": (8, 8, "heads"), "b": (4, 8, "ch"), "c": (0, 8, "ch")},
            2: {"a": (8, 8, "heads"), "b": (8, 8, "ch"), "c": (8, 8, "ch")}}
    devs = [0, 1, 2]
    assert Module.tp_single_owner(plan_ctx(plan, devs, 0), "a") == 0
    assert Module.tp_single_owner(plan_ctx(plan, devs, 1), "c") == 1
    assert Module.tp_single_owner(plan_ctx(plan, devs, 0), "b") is None, "split module has no single owner"
    assert Module.tp_single_owner(plan_ctx(plan, devs, 0), "a", "c") is None, "parts on different ranks"
    assert Module.tp_single_owner(plan_ctx(plan, devs, 0), "a", "a") == 0
    assert Module.tp_single_owner({"device": 0}, "a") is None, "no plan (layer-split import paths)"


def test_collect_routes_by_owner():
    m = SimpleNamespace(); b = RecordingBackend(); t = torch.zeros(8)
    m.tp_owner = None; Module.tp_collect(m, b, t, False)
    m.tp_owner = 2; Module.tp_collect(m, b, t); Module.tp_collect(m, b, t, False)
    assert b.calls == [("all_reduce", False), ("broadcast", 2), ("broadcast", 2)]


def test_attention_import_sets_owner_on_stub_and_owner():
    m = Attention(config = None, key = "l.attn", layer_idx = 0, hidden_size = 64, head_dim = 16, num_q_heads = 8,
                  num_kv_heads = 4, rope_settings = None, q_proj = FakeChild("q"), k_proj = FakeChild("k"),
                  v_proj = FakeChild("v"), o_proj = FakeChild("o"))
    m.device = DEVICE
    exported = m.tp_export(plan = {}, producer = None)
    whole = {0: {"l.attn": (0, 4, "heads")}, 1: {"l.attn": (4, 4, "heads")}}
    split = {0: {"l.attn": (0, 2, "heads")}, 1: {"l.attn": (2, 4, "heads")}}
    with stub_loading(Attention):
        owner = Attention.tp_import(plan_ctx(whole, [0, 1], 0), exported, whole[0])
        stub = Attention.tp_import(plan_ctx(whole, [0, 1], 1), exported, whole[1])
        part = Attention.tp_import(plan_ctx(split, [0, 1], 1), exported, split[1])
    assert (owner.tp_owner, stub.tp_owner, part.tp_owner) == (0, 0, None)
    assert owner.tp_reduce and stub.tp_reduce and part.tp_reduce
    assert part.num_kv_heads == 2, "a split import keeps its slice of the heads"
    # The stub's forward broadcasts (from the owner) a zero buffer of the output shape
    b = RecordingBackend()
    y = stub.forward(torch.ones(1, 3, 64, dtype = torch.half, device = DEVICE), {"backend": b})
    assert b.calls == [("broadcast", 0)]
    assert tuple(y.shape) == (1, 3, 64)
    assert y.abs().sum().item() == 0.0


def test_mla_stub_broadcasts():
    m = MLAttention(config = None, key = "l.mla", layer_idx = 0, hidden_size = 64, num_q_heads = 4, kv_lora_rank = 32,
                    qk_nope_head_dim = 16, qk_rope_head_dim = 8, v_head_dim = 16, rope_settings = None, q_lora_rank = 16,
                    out_dtype = torch.float, submodules = {n: FakeChild(n) for n in
                    ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa", "kv_a_layernorm", "o_proj")})
    m.device = DEVICE
    exported = m.tp_export(plan = {}, producer = FakeProducer())
    plan = {0: {"l.mla": (0, 0, "heads")}, 1: {"l.mla": (0, 4, "heads")}}
    with stub_loading(MLAttention):
        stub = MLAttention.tp_import(plan_ctx(plan, [0, 1], 0), exported, plan[0])
    assert stub.tp_owner == 1
    b = RecordingBackend()
    stub.forward(torch.ones(2, 1, 64, dtype = torch.half, device = DEVICE), {"backend": b})
    assert b.calls == [("broadcast", 1)]


def test_ple_single_owner():
    # The PLE layer runs whole on one rank (max_devices = 1): the stub has no submodules, no recurrent-cache
    # cap (so the worker never registers it as a cache module) and its forward receives the owner's updated
    # stack via broadcast; the owner broadcasts
    kwargs = dict(key = "l.ple", layer_idx = -2, hidden_size = 64, hc_mult = 4, ple_embed_dim = 32, ngram_size = 3,
                  heads_per_ngram = 8, eos_token_id = 0, conv_kernel_size = 4, rms_norm_eps = 1e-6, out_dtype = None,
                  mm_token_id = None)
    exported = {"cls": PLELayer, "kwargs": kwargs, **{n: {"cls": FakeChild, "key": n} for n in PLELayer._tp_submodules},
                "conv_w": {"t": torch.ones(256, 1, 4, dtype = torch.half)}, "recurrent_layers": [], "device": DEVICE}
    plan = {0: {"l.ple": (0, 0, "layer")}, 1: {"l.ple": (0, 1, "layer")}}
    with stub_loading():
        stub = PLELayer.tp_import(ctx(device = 0, plan = plan, active_devices = [0, 1], consumer = FakeConsumer()),
                                  exported, plan[0])
        owner = PLELayer.tp_import(ctx(device = 1, plan = plan, active_devices = [0, 1], consumer = FakeConsumer()),
                                   exported, plan[1])
    assert stub.stub and not owner.stub
    assert (stub.tp_owner, owner.tp_owner) == (1, 1)
    assert not (stub.caps.get("recurrent_cache") or stub.caps.get("prefetch_ids"))
    assert owner.caps.get("recurrent_cache") and owner.caps.get("prefetch_ids")
    assert stub.all_recurrent_modules() == []
    assert owner.all_recurrent_modules() == [owner]
    x = torch.randn(2, 3, 4, 64, device = DEVICE)
    b = RecordingBackend()
    y = stub.forward(x, {"backend": b})
    assert b.calls == [("broadcast", 1)]
    # without a real backend the buffer keeps the input (warmup runs with a null backend)
    assert torch.equal(y, x) and y.data_ptr() != x.data_ptr()

    # The allocation is a single unsplittable channel
    class Sub:
        def storage_size(self): return 8

    class Stc:
        def get_tensor_sizes(self, key): return [4]

    owner.config = SimpleNamespace(stc = Stc()); owner.key_proj = Sub(); owner.value_proj = Sub()
    tpa = owner.make_tp_allocation({})[0]
    assert (tpa.max_devices, tpa.channels_to_split, tpa.channel_width, tpa.channel_unit) == (1, 1, 1, "layer")
    assert (tpa.storage_per_device, tpa.storage_to_split) == (0, 8 + 8 + 3 * 4 + 4)
