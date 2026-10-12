"""
CPU tests for streamed frozen MoE experts (``exllamav3/training/expert_stream.py``,
Qwen3.8-Flash-Next plan Phase C).

No GPU, no compiled extension, no real model: the training modules are loaded
under a synthetic package (as the other training suites do), EXL3 expert
linears are mocked by an inner whose frozen weight is DERIVED FROM ITS
``trellis`` TENSOR at call time (the fp16 weight's bytes, viewed as the int16
trellis), so a reconstruct that read a stale or wrong slot would produce a
wrong weight. On CPU host and device coincide, so the copies are plain
``copy_`` calls and the same residency / direction / eviction bookkeeping
runs as on CUDA. Checks:

  * the streamed MoE forward + backward (three ``_moe_out`` layers, shared
    expert with LoRA, the real ``NativeLlamaQLoRA._moe_out``) is bit-identical
    to the resident one, every reconstruct sees a VRAM-slot view, and at most
    ``slots`` layers are resident at any time;
  * a scripted forward / backward / forward access sequence produces exactly
    the expected copies (one synchronous miss on the first layer, the rest
    prefetched), evicts the right layers, rebinds evicted layers to their host
    views and resident ones to slot views holding the right bytes;
  * ``unpark_all`` restores the original tensors and removes the hooks;
  * ``park_block`` drops the inference fast-path references and refuses
    non-EXL3 experts;
  * ``incompatible_flags`` names each rejected trainer flag.

Run:  python tests/test_expert_stream.py
"""

from __future__ import annotations
import os
import sys
import types
import importlib.util
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.join(_ROOT, "exllamav3", "training")

_pkg = types.ModuleType("exl3train")
_pkg.__path__ = [_TRAIN_DIR]
sys.modules["exl3train"] = _pkg


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"exl3train.{name}", os.path.join(_TRAIN_DIR, f"{name}.py")
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules[f"exl3train.{name}"] = m
    spec.loader.exec_module(m)
    return m


_es = _load("expert_stream")
_nl = _load("native_llama")
DiffLinear = _nl.DiffLinear
NativeLlamaQLoRA = _nl.NativeLlamaQLoRA
ExpertStreamer = _es.ExpertStreamer


# ----------------------------------------------------------------------------
# Mock EXL3 linear whose weight lives in its trellis bytes.
# ----------------------------------------------------------------------------
class _TrellisInner:
    """Frozen [in, out] fp16 weight stored as a 3-D int16 ``trellis`` (the
    real layout is [in/16, out/16, 16K]; the streamer only needs a 3-D int16
    tensor). ``get_inner_weight_tensor`` reads ``self.trellis`` at call time,
    exactly the dependency the streamer relies on. No ``quant_type`` here, so
    DiffLinear takes the legacy closure path (``get_weight_tensor``), which on
    the real class also goes through ``get_inner_weight_tensor``."""

    def __init__(self, weight_f16: torch.Tensor):
        assert weight_f16.dtype == torch.float16
        self.in_features, self.out_features = weight_f16.shape
        self.trellis = weight_f16.contiguous().view(torch.int16).reshape(
            self.in_features, self.out_features, 1)
        self.bias = None
        self.check = None            # optional callable(trellis) run per reconstruct

    def get_inner_weight_tensor(self, n_offset=0, n_features=None, out_dtype=torch.half):
        if self.check is not None:
            self.check(self.trellis)
        w = self.trellis.reshape(self.in_features, self.out_features).view(torch.float16)
        return w.to(out_dtype)

    def get_weight_tensor(self):
        return self.get_inner_weight_tensor(out_dtype=torch.float64)

    def get_bias_tensor(self):
        return None


class MockLinear(nn.Module):
    def __init__(self, weight_f16: torch.Tensor, key: str, quant_type: str = "exl3"):
        super().__init__()
        self.in_features, self.out_features = weight_f16.shape
        self.key = key
        self.device = torch.device("cpu")
        self.quant_type = quant_type
        self.inner = _TrellisInner(weight_f16)
        self.lora_a_tensors = {}
        self.lora_b_tensors = {}


def _headless_net(dtype=torch.float64):
    n = NativeLlamaQLoRA.__new__(NativeLlamaQLoRA)
    n.compute_dtype = dtype
    n.attn_impl = "eager"
    n.use_liger = False
    return n


def _build_moe_layer(d, inter, E, k, seed, r=4):
    """One MoE block's MLP half as the trainer's entry/meta pair plus a fake
    ``BlockSparseMLP`` namespace (gates/ups/downs lists + inference objects)."""
    g = torch.Generator().manual_seed(seed)

    def w(i, o, scale=0.3):
        return (torch.randn(i, o, generator=g) * scale).to(torch.float16)

    f64 = torch.float64
    gates = [MockLinear(w(d, inter), f"L{seed}.mlp.experts.{e}.gate_proj") for e in range(E)]
    ups = [MockLinear(w(d, inter), f"L{seed}.mlp.experts.{e}.up_proj") for e in range(E)]
    downs = [MockLinear(w(inter, d), f"L{seed}.mlp.experts.{e}.down_proj") for e in range(E)]
    entry = types.SimpleNamespace()
    entry.router = DiffLinear(MockLinear(w(d, E, 1.0), f"L{seed}.mlp.gate"), r=0, compute_dtype=f64)
    entry.expert_gates = nn.ModuleList([DiffLinear(x, r=0, compute_dtype=f64) for x in gates])
    entry.expert_ups = nn.ModuleList([DiffLinear(x, r=0, compute_dtype=f64) for x in ups])
    entry.expert_downs = nn.ModuleList([DiffLinear(x, r=0, compute_dtype=f64) for x in downs])
    # Shared expert (Qwen3.5-MoE layout) with LoRA, so the backward has
    # adapter gradients to compare.
    entry.gates = [DiffLinear(MockLinear(w(d, inter), f"L{seed}.mlp.shared.gate_proj"),
                              r=r, compute_dtype=f64)]
    entry.ups = [DiffLinear(MockLinear(w(d, inter), f"L{seed}.mlp.shared.up_proj"),
                            r=r, compute_dtype=f64)]
    entry.downs = [DiffLinear(MockLinear(w(inter, d), f"L{seed}.mlp.shared.down_proj"),
                              r=r, compute_dtype=f64)]
    entry.shared_gate = DiffLinear(MockLinear(w(d, 1, 1.0), f"L{seed}.mlp.shared_gate"),
                                   r=0, compute_dtype=f64)
    for m in (entry.gates[0], entry.ups[0], entry.downs[0]):
        with torch.no_grad():
            m.lora_b.normal_(std=0.05)      # non-zero so the adapter matters
    meta = {"mlp_kind": "moe", "num_experts": E, "num_experts_per_tok": k,
            "activation": "silu"}
    mlp = types.SimpleNamespace(gates=gates, ups=ups, downs=downs,
                                bc=object(), multi_gate=object(), multi_up=object(),
                                multi_down=object(), batch_recon=object(),
                                fused_mode_buffers=object())
    return entry, meta, mlp


# ----------------------------------------------------------------------------
# 1. Streamed == resident through the real _moe_out, forward and backward.
# ----------------------------------------------------------------------------
def test_streamed_moe_matches_resident():
    torch.manual_seed(0)
    d, inter, E, k = 32, 48, 6, 2
    layers = [_build_moe_layer(d, inter, E, k, seed=s) for s in range(3)]
    net = _headless_net()
    x0 = torch.randn(2, 7, d, dtype=torch.float64)

    def run():
        x = x0.clone().requires_grad_(True)
        h = x
        for entry, meta, _ in layers:
            h = h + net._moe_out(meta, entry, h)
        (h.square().sum()).backward()
        grads = [x.grad.clone()]
        for entry, _, _ in layers:
            for m in (entry.gates[0], entry.ups[0], entry.downs[0]):
                grads += [m.lora_a.grad.clone(), m.lora_b.grad.clone()]
                m.lora_a.grad = None
                m.lora_b.grad = None
        return h.detach().clone(), grads

    ref_out, ref_grads = run()

    streamer = ExpertStreamer("cpu", slots=2, pin=False)
    for _, _, mlp in layers:
        _es.park_block(streamer, mlp)
    assert len(streamer.layers) == 3
    assert streamer.resident_layers() == []
    # Parked: every expert's trellis is a host view, values intact.
    for layer in streamer.layers:
        for inner in layer.inners:
            assert not streamer.is_resident_view(inner.trellis)

    seen = {"max_resident": 0, "calls": 0}

    def check(t):
        assert streamer.is_resident_view(t), "reconstruct read a non-resident trellis"
        seen["max_resident"] = max(seen["max_resident"], len(streamer.resident_layers()))
        seen["calls"] += 1
    for layer in streamer.layers:
        for inner in layer.inners:
            inner.check = check

    out, grads = run()
    assert torch.equal(out, ref_out), "streamed MoE forward differs from resident"
    assert len(grads) == len(ref_grads)
    for a, b in zip(grads, ref_grads):
        assert torch.equal(a, b), "streamed MoE backward differs from resident"
    assert seen["calls"] > 0
    assert seen["max_resident"] <= 2
    # Forward: layer 0 is the one synchronous miss, 1 and 2 are prefetched;
    # backward (2, 1, 0) re-reconstructs through the Functions' backward with
    # 2 and 1 resident and 0 prefetched while 1 runs.
    s = streamer.stats
    assert s["misses"] == 1, s
    assert s["copies"] == 4, s
    assert s["prefetches"] == 3, s
    assert s["hits"] > 0
    streamer.describe()
    streamer.stats_line()
    print("[expert_stream] streamed _moe_out == resident (fwd + bwd, bit-exact) PASSED")


# ----------------------------------------------------------------------------
# 2. Ring bookkeeping on a scripted access sequence.
# ----------------------------------------------------------------------------
class _Bare:
    def __init__(self, t):
        self.trellis = t

    def get_inner_weight_tensor(self, *a, **k):
        return self.trellis


def _bare_layers(n_layers=3, per=3):
    return [[_Bare(torch.full((4, 4, 2), i * 10 + j, dtype=torch.int16)) for j in range(per)]
            for i in range(n_layers)]


def _expected(i, j):
    return torch.full((4, 4, 2), i * 10 + j, dtype=torch.int16)


def _assert_bound(streamer, layers, resident):
    for i, L in enumerate(layers):
        for j, b in enumerate(L):
            assert torch.equal(b.trellis, _expected(i, j)), f"layer {i} expert {j} bytes"
            assert streamer.is_resident_view(b.trellis) == (i in resident), \
                f"layer {i}: resident={i in resident}, view says {streamer.is_resident_view(b.trellis)}"
    assert sorted(streamer.resident_layers()) == sorted(resident)


def test_ring_direction_and_eviction():
    layers = _bare_layers()
    s = ExpertStreamer("cpu", slots=2, pin=False)
    for L in layers:
        s.park(L)
    _assert_bound(s, layers, [])
    assert s.host_bytes == 3 * 3 * 128 * 2       # 32 elements aligned up to 128 each
    assert s.slot_bytes == 3 * 128 * 2

    # forward: the hooked method is what the DiffLinear closures call
    layers[0][0].get_inner_weight_tensor()
    _assert_bound(s, layers, [0, 1])                      # miss 0, prefetch 1
    assert s.stats == {"copies": 2, "misses": 1, "prefetches": 1, "hits": 0}
    layers[0][1].get_inner_weight_tensor()                # same layer: no change
    assert s.stats["copies"] == 2 and s.stats["hits"] == 1
    s.ensure(1)
    _assert_bound(s, layers, [1, 2])                      # prefetch 2 evicts 0
    s.ensure(1)
    s.ensure(2)                                           # nothing past the last layer
    _assert_bound(s, layers, [1, 2])
    assert s.stats["copies"] == 3
    # backward: the repeat of the last layer flips the direction
    s.ensure(2)
    _assert_bound(s, layers, [1, 2])                      # 1 already resident
    s.ensure(1)
    _assert_bound(s, layers, [0, 1])                      # prefetch 0 evicts 2
    s.ensure(1)
    s.ensure(0)
    _assert_bound(s, layers, [0, 1])                      # layer 0 => forward, 1 resident
    assert s.stats["copies"] == 4
    # next step's forward: no synchronous copy at all
    s.ensure(0)
    s.ensure(1)
    _assert_bound(s, layers, [1, 2])                      # prefetch 2 evicts 0
    s.ensure(2)
    assert s.stats == {"copies": 5, "misses": 1, "prefetches": 4, "hits": 11}, s.stats
    # park after the first ensure is refused; too few slots is refused
    try:
        s.park([_Bare(torch.zeros(4, 4, 2, dtype=torch.int16))])
        raise AssertionError("park after ensure should fail")
    except RuntimeError:
        pass
    try:
        ExpertStreamer("cpu", slots=1, pin=False)
        raise AssertionError("slots=1 should fail")
    except ValueError:
        pass
    print("[expert_stream] ring direction / eviction / rebinding PASSED")


def test_three_slots_prefetch_two_ahead():
    layers = _bare_layers(n_layers=5)
    s = ExpertStreamer("cpu", slots=3, pin=False)
    for L in layers:
        s.park(L)
    s.ensure(0)
    _assert_bound(s, layers, [0, 1, 2])
    s.ensure(1)
    _assert_bound(s, layers, [1, 2, 3])                   # evicts the previous layer
    s.ensure(2)
    s.ensure(3)
    s.ensure(4)
    _assert_bound(s, layers, [2, 3, 4])
    s.ensure(4)                                           # backward begins
    s.ensure(3)
    _assert_bound(s, layers, [1, 2, 3])                   # 1 prefetched, 4 evicted
    assert s.stats["misses"] == 1
    print("[expert_stream] three-slot ring PASSED")


# ----------------------------------------------------------------------------
# 3. unpark_all restores tensors and removes hooks.
# ----------------------------------------------------------------------------
def test_unpark_restores_everything():
    layers = _bare_layers()
    s = ExpertStreamer("cpu", slots=2, pin=False)
    for L in layers:
        s.park(L)
    for b in layers[0]:
        assert "get_inner_weight_tensor" in b.__dict__
    s.ensure(0)
    s.ensure(1)
    s.unpark_all()
    assert s.layers == [] and s.resident_layers() == []
    for i, L in enumerate(layers):
        for j, b in enumerate(L):
            assert torch.equal(b.trellis, _expected(i, j))
            assert not s.is_resident_view(b.trellis)
            assert "get_inner_weight_tensor" not in b.__dict__
            assert b.get_inner_weight_tensor() is b.trellis        # class method again
    print("[expert_stream] unpark_all PASSED")


# ----------------------------------------------------------------------------
# 4. park_block: inference references dropped, non-EXL3 refused.
# ----------------------------------------------------------------------------
def test_park_block_drops_inference_refs_and_rejects_fp16():
    _, _, mlp = _build_moe_layer(32, 48, 4, 2, seed=11)
    inners = [l.inner for l in mlp.gates + mlp.ups + mlp.downs]
    for inner in inners:
        inner.bc = object()
        inner._fused_reconstruct = object()
    s = ExpertStreamer("cpu", slots=2, pin=False)
    li = _es.park_block(s, mlp)
    assert li == 0 and len(s.layers[0].inners) == 12
    for name in ("bc", "multi_gate", "multi_up", "multi_down", "batch_recon",
                 "fused_mode_buffers"):
        assert getattr(mlp, name) is None, name
    for inner in inners:
        assert inner.bc is None and inner._fused_reconstruct is None

    _, _, mlp2 = _build_moe_layer(32, 48, 4, 2, seed=12)
    mlp2.ups[1].quant_type = "fp16"
    s2 = ExpertStreamer("cpu", slots=2, pin=False)
    try:
        _es.park_block(s2, mlp2)
        raise AssertionError("fp16 expert should be refused")
    except RuntimeError as e:
        assert "EXL3" in str(e)
    assert s2.layers == []
    print("[expert_stream] park_block PASSED")


# ----------------------------------------------------------------------------
# 5. Trainer flag compatibility.
# ----------------------------------------------------------------------------
def test_incompatible_flags():
    ok = _es.incompatible_flags("single", False, 0, ["q_proj", "k_proj", "gate_proj"], None)
    assert ok == []
    bad = _es.incompatible_flags("split", True, 10, ["q_proj", "expert_up_proj"], 4)
    text = "\n".join(bad)
    assert len(bad) == 5
    for needle in ("--parallel split", "--vram-spillover", "--sample-every",
                   "expert_up_proj", "--expert-r"):
        assert needle in text, needle
    assert _es.incompatible_flags("single", False, 0, None, None) == []
    print("[expert_stream] incompatible_flags PASSED")


if __name__ == "__main__":
    test_streamed_moe_matches_resident()
    test_ring_direction_and_eviction()
    test_three_slots_prefetch_two_ahead()
    test_unpark_restores_everything()
    test_park_block_drops_inference_refs_and_rejects_fp16()
    test_incompatible_flags()
    print("\nALL expert_stream TESTS PASSED")
