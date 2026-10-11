"""
Whole-layer tensor-parallel placement for MLA (and the KDA flavour of GatedDeltaNet): the allocator must keep a
module-enforced max_devices = 1 even when the user's per-type limit is wider (and pin DSA follower layers to
their leader's device), the MLA export/import must round-trip every forward-affecting constructor flag, the
rank that does not hold the layer must come back as a head-less stub that joins the all-reduce without
contributing, and the KDA export must carry the gate projections with per-channel (not per-head) dt_bias
slicing. GatedRMSNorm and HyperHead must keep their gate activation / mean flag across export. Bare modules
(no config, fake children from testlib.tp), checked against the plan's head ranges and the constructor flags;
HyperHead's mean against torch.mean.
"""

from unittest.mock import patch

import pytest
import torch

from exllamav3.model.model_tp_alloc import TPAllocation, TPAllocator
from exllamav3.modules.gated_delta_net import GatedDeltaNet
from exllamav3.modules.mla_attn import MLAttention
from testlib.tp import DEVICE, FakeChild, FakeConsumer, FakeProducer, RecordingBackend, ctx, stub_loading

HEADS, NOPE, PE, V, D_C, HID = 8, 64, 32, 64, 128, 256


def bare_mla(**flags):
    names = ["q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa", "kv_a_layernorm", "o_proj"]
    if flags.get("indexer_mode") == "full":
        names += ["idx_wq_b", "idx_wk", "idx_k_norm", "idx_weights"]
    m = MLAttention(config = None, key = "model.layers.3.self_attn", layer_idx = 3, hidden_size = HID,
                    num_q_heads = HEADS, kv_lora_rank = D_C, qk_nope_head_dim = NOPE, qk_rope_head_dim = PE,
                    v_head_dim = V, rope_settings = None, q_lora_rank = 96, out_dtype = torch.float,
                    submodules = {n: FakeChild(n) for n in names}, **flags)
    m.device = DEVICE
    m.w_uk_flat = torch.randn(D_C, HEADS * NOPE, dtype = torch.half, device = DEVICE)
    m.w_uv_flat = torch.randn(D_C, HEADS * V, dtype = torch.half, device = DEVICE)
    return m


def import_mla(m, first, last):
    exported = m.tp_export(plan = {}, producer = FakeProducer())
    with stub_loading(MLAttention):
        return exported, MLAttention.tp_import(ctx(consumer = FakeConsumer()), exported, {m.key: (first, last, "heads")})


@pytest.mark.nogpu
def test_allocator_honours_module_cap_over_user_limit():
    comps = [TPAllocation(key = f"l{i}", channel_width = HEADS, channel_unit = "heads",
                          storage_to_split = 1000, channels_to_split = 1, limit_key = "attn",
                          max_devices = 1) for i in range(4)]
    alloc = TPAllocator(comps, num_tokens = 16, output_num_tokens = 16, dev_limits = {"attn": 3})
    alloc.initial_split([10**6, 10**6, 10**6])
    for c in comps:
        assert sum(1 for s in c.current_split if s) == 1, f"{c.key} split across ranks: {c.current_split}"
        assert sum(c.current_split) == 1
    plan = alloc.compile_tp_plan()
    widths = sorted(plan[d]["l0"][1] - plan[d]["l0"][0] for d in range(3))
    assert widths == [0, 0, HEADS]


@pytest.mark.nogpu
def test_allocator_affinity_pins_followers_to_their_leader():
    # DSA "shared" indexer layers must land on the device of the "full" layer whose selection
    # they reuse, whatever the memory picture says; unrelated groups place freely
    def comp(i, aff):
        return TPAllocation(key = f"l{i}", channel_width = HEADS, channel_unit = "heads", storage_to_split = 1000,
                            channels_to_split = 1, limit_key = "attn", max_devices = 1, affinity_key = aff)
    comps = [comp(0, "g0"), comp(1, "g0"), comp(2, "g0"), comp(3, "g3"), comp(4, "g3"), comp(5, None)]
    alloc = TPAllocator(comps, num_tokens = 16, output_num_tokens = 16)
    # Device 1 has the most room, so leaders would otherwise alternate as devices fill
    alloc.initial_split([10**6, 3 * 10**6, 10**6])
    owner = lambda c: [i for i, s in enumerate(c.current_split) if s]
    assert owner(comps[0]) == owner(comps[1])
    assert owner(comps[0]) == owner(comps[2])
    assert owner(comps[3]) == owner(comps[4])
    for c in comps:
        assert sum(c.current_split) == 1, f"{c.key} must stay whole: {c.current_split}"


@pytest.mark.nogpu
def test_mla_flags_survive_export():
    flags = dict(indexer_mode = "full", index_n_heads = 4, index_head_dim = 32, index_topk = 512,
                 index_kpool = 4, index_kpool_tail = False, index_norm_eps = 1e-5, sm_scale = 0.123)
    m = bare_mla(**flags)
    m.idx_kpool_ape = torch.zeros(4, dtype = torch.float, device = DEVICE)
    m.idx_kpool_gate = torch.zeros(32, HID, dtype = torch.half, device = DEVICE)
    exported, imported = import_mla(m, 0, HEADS)
    for k, v in flags.items():
        assert exported["kwargs"].get(k) == v, f"MLAttention.tp_export drops {k}"
        assert getattr(imported, k) == v, f"MLAttention.tp_import loses {k}"
    assert imported.num_q_heads == HEADS
    assert imported.kv_lora_rank == D_C
    assert imported.idx_plane_dim == 64
    assert "q_proj" not in exported, "q_proj aliases q_b_proj and must be exported once"
    assert imported.q_proj is imported.q_b_proj
    for n in ("q_a_proj", "kv_a_proj_with_mqa", "o_proj", "idx_wq_b", "idx_wk", "idx_k_norm", "idx_weights"):
        assert getattr(imported, n).key == n
    assert tuple(imported.w_uk_flat.shape) == (D_C, HEADS * NOPE)
    assert tuple(imported.idx_kpool_gate.shape) == (32, HID)
    assert imported.tp_reduce
    assert imported.caps.get("kv_cache")
    assert len(imported.modules) == 10


@pytest.mark.nogpu
def test_mla_stub_on_other_rank():
    m = bare_mla()
    exported, stub = import_mla(m, HEADS, HEADS)
    assert stub.num_q_heads == 0
    assert not stub.caps.get("kv_cache"), "stub must not register as a cache module"
    assert stub.modules == []
    assert stub.q_proj is None
    assert stub.w_uk_flat is None
    x = torch.randn(2, 5, HID, dtype = torch.half, device = DEVICE)
    backend = RecordingBackend()
    y = stub.forward(x, {"backend": backend})
    assert tuple(y.shape) == (2, 5, HID)
    assert y.dtype == torch.float
    assert y.abs().sum().item() == 0.0
    assert backend.calls == [("all_reduce", False)]
    assert backend.tensors == [((2, 5, HID), torch.float)]


@pytest.mark.nogpu
def test_mla_rejects_partial_head_range():
    m = bare_mla()
    with pytest.raises(AssertionError):
        import_mla(m, 0, HEADS // 2)


@pytest.mark.nogpu
def test_mla_allocation_is_single_device():
    m = bare_mla()
    # storage_size/recons_size on the fake children and stc sizes are not available without a model;
    # only the placement parameters are under test
    class FakeStc:
        def get_tensor_sizes(self, prefix): return [0]
    class FakeCfg:
        stc = FakeStc()
    m.config = FakeCfg()
    tpa = m.make_tp_allocation({})[0]
    assert tpa.max_devices == 1
    assert tpa.channels_to_split == 1
    assert tpa.channel_width == HEADS
    assert tpa.limit_key == "attn"


@pytest.mark.nogpu
def test_kda_export_carries_gate_projections():
    hk = 32; nh = 4
    kw = dict(config = None, key = "model.layers.1.self_attn", layer_idx = 1, hidden_size = HID,
              k_head_dim = hk, v_head_dim = hk, num_k_heads = nh, num_v_heads = nh, rms_norm_eps = 1e-6,
              conv_kernel_size = 4, key_f_a = "f_a", key_f_b = "f_b", key_g_a = "g_a", key_g_b = "g_b",
              gate_lower_bound = -5.0, qkv_proj = FakeChild("qkv"), b_proj = FakeChild("b"),
              o_proj = FakeChild("o"), norm = FakeChild("norm"),
              f_a_proj = FakeChild("f_a"), f_b_proj = FakeChild("f_b"),
              g_a_proj = FakeChild("g_a"), g_b_proj = FakeChild("g_b"),
              a_log = torch.zeros(nh), dt_bias = torch.zeros(nh * hk),
              conv1d_weight = torch.zeros(3 * nh * hk, 1, 4), conv1d_bias = None)
    m = GatedDeltaNet(**kw)
    m.device = DEVICE
    assert m.kda
    exported = m.tp_export(plan = {}, producer = FakeProducer())
    for k in ("key_f_a", "key_f_b", "key_g_a", "key_g_b"):
        assert k in exported["kwargs"]
    assert exported["kwargs"]["gate_lower_bound"] == -5.0
    for n in ("f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj"):
        assert exported.get(n) is not None

    # Capture the splits the import applies: dt_bias must follow the k-channel range of the
    # local heads, f_b/g_b the same, f_a/g_a unsplit
    seen = {}
    class SplitSpy:
        @staticmethod
        def tp_import_split(local_context, exported, plan, split): seen[exported["key"]] = split; return FakeChild(exported["key"])
        @staticmethod
        def tp_import_split_3(local_context, exported, plan, s0, s1, s2): return FakeChild(exported["key"])
        @staticmethod
        def tp_import(local_context, exported, plan): seen[exported["key"]] = "whole"; return FakeChild(exported["key"])
    for n in ("qkv_proj", "b_proj", "o_proj", "norm", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj"):
        exported[n]["cls"] = SplitSpy
    for n in ("a_log", "dt_bias", "conv1d_weight"):
        exported[n] = {"cls": SplitSpy, "key": n}
    local_context = ctx(consumer = FakeConsumer())
    with stub_loading(GatedDeltaNet):
        imported = GatedDeltaNet.tp_import(local_context, exported, {m.key: (1, 3, "K-heads")})
    assert imported.kda
    assert imported.gate_lower_bound == -5.0
    assert imported.num_k_heads == 2
    assert seen["dt_bias"] == (True, 1 * hk, 3 * hk)
    assert seen["f_b"] == (True, 1 * hk, 3 * hk)
    assert seen["g_b"] == (True, 1 * hk, 3 * hk)
    assert seen["f_a"] == "whole"
    assert seen["g_a"] == "whole"
    assert seen["a_log"] == (True, 1, 3)
    assert seen["b"] == (True, 1, 3)

    # Rank without heads: a stub; nothing head-shaped is imported for it (the per-head-dim
    # norm weight is replicated on every rank, as for every GDN flavour)
    seen.clear()
    with stub_loading(GatedDeltaNet):
        stub = GatedDeltaNet.tp_import(local_context, exported, {m.key: (4, 4, "K-heads")})
    assert stub.num_k_heads == 0
    assert stub.kda
    assert seen == {"norm": "whole"}


def test_gated_rmsnorm_gate_activation_survives_export():
    # KDA gates the norm with a sigmoid; the imported graph object must be built with the same
    # activation as load() builds it (it used to default to silu on import, which broke every
    # KDA decode step under TP while the torch path stayed correct)
    from exllamav3.modules.gated_rmsnorm import GatedRMSNorm
    from exllamav3.ext import exllamav3_ext as ext
    built = []
    real = ext.BC_GatedRMSNorm
    def spy(*args): built.append(args); return real(*args)
    m = GatedRMSNorm(config = None, key = "n", rms_norm_eps = 1e-6, gate_activation = "sigmoid")
    m.device = DEVICE
    m.weight = torch.ones(64, dtype = torch.half, device = DEVICE)
    exported = m.tp_export(plan = {}, producer = FakeProducer())
    assert exported["kwargs"].get("gate_activation") == "sigmoid"
    with patch.object(ext, "BC_GatedRMSNorm", spy), stub_loading():
        imported = GatedRMSNorm.tp_import(ctx(consumer = FakeConsumer()), exported, {})
    assert imported.gate_activation == "sigmoid"
    assert built[-1][-1] == 1, "imported BC_GatedRMSNorm must carry the sigmoid gate flag"


@pytest.mark.nogpu
def test_hyper_head_mean_survives_export():
    # GLM5.3's stream collapse is a parameterless mean (no fn/base/scale); an import that
    # forgot the flag took the weighted path and dereferenced fn = None
    from exllamav3.modules.hyperconnections import HyperHead
    m = HyperHead(config = None, key = "hc_head", hc_mult = 4, rms_norm_eps = 1e-6, hc_eps = 1e-6, mean = True)
    m.device = DEVICE
    exported = m.tp_export(plan = {}, producer = FakeProducer())
    imported = HyperHead.tp_import(ctx(consumer = FakeConsumer()), exported, {})
    assert imported.mean
    x = torch.randn(2, 3, 4, 8, dtype = torch.float, device = DEVICE)
    torch.testing.assert_close(imported.forward(x, {}), x.mean(dim = 2))
