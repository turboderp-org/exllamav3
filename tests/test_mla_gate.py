"""Head-wise MLA output gating: CPU contracts and opt-in real GPU modules.

CPU (Torch only, no ExLlama import/JIT):
    python -m unittest discover -s tests -p test_mla_gate.py -v
GPU (parent schedules one visible device at a time; imports/builds the extension):
    EXL3_TEST_MLA_GATE_GPU=1 python -m unittest discover -s tests -p test_mla_gate.py -v

Math source: inclusionAI/Ling-3.0-flash, ef06d91fe382109ae82647da88ff99b0f11745b0,
modeling_bailing_moe_v3.py:623-631,709-718. No remote model code is executed.
The CPU source harness runs the actual Python classes with mocked native handles/kernels;
it tests wiring/lifecycle, not compiled attention. GPU tests import the real package lazily.
"""
from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import cached_property
import inspect
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "exllamav3" / "modules"


class TensorCollection:
    """In-memory original weights, with optional deferred destination population."""

    def __init__(self, tensors, defer = False):
        self.tensors = tensors
        self.defer = defer
        self.pending = []

    def has_tensor(self, key):
        return key in self.tensors

    def has_tensor_group(self, key, subkeys):
        return all(self.has_tensor(f"{key}.{s}") if isinstance(s, str)
                   else any(self.has_tensor(f"{key}.{k}") for k in s) for s in subkeys)

    def get_tensor(self, key, device = None, optional = False, allow_bf16 = False,
                   float2half = False, no_defer = False, transpose = False, pad_to = None):
        if key not in self.tensors:
            if optional:
                return None
            raise ValueError(f"Required tensor {key} not found")
        t = self.tensors[key].to(device or "cpu")
        if float2half and t.dtype in (torch.float32, torch.float64, torch.bfloat16):
            t = t.half()
        if transpose:
            t = t.T.contiguous()
        if pad_to:
            pad = []
            for i in reversed(range(len(pad_to))):
                pad.extend([0, pad_to[i] - t.shape[i]])
            t = torch.nn.functional.pad(t, pad)
        t = t.contiguous()
        if self.defer and not no_defer:
            dest = torch.empty_like(t)
            self.pending.append((dest, t))
            return dest
        return t

    def finish(self):
        for dest, source in self.pending:
            dest.copy_(source)
        self.pending.clear()

    def get_tensor_sizes(self, key):
        return [v.numel() * v.element_size() for k, v in self.tensors.items()
                if k.startswith(key + ".")]


def fixture_weights(H = 4, hidden = 128, rank = 64, nope = 16, rope = 8, v = 16,
                    q_lora = None, seed = 11):
    gen = torch.Generator().manual_seed(seed)
    key = "model.layers.5.attention"

    def rand(*shape):
        return (torch.randn(*shape, generator = gen) * 0.05).half()

    weights = {
        f"{key}.kv_a_proj_with_mqa.weight": rand(rank + rope, hidden),
        f"{key}.kv_a_layernorm.weight": torch.ones(rank, dtype = torch.half),
        f"{key}.kv_b_proj.weight": rand(H * (nope + v), rank),
        f"{key}.dense.weight": rand(hidden, H * v),
        f"{key}.g_proj.weight": rand(H, hidden),
    }
    if q_lora is None:
        weights[f"{key}.q_proj.weight"] = rand(H * (nope + rope), hidden)
    else:
        weights[f"{key}.q_a_proj.weight"] = rand(q_lora, hidden)
        weights[f"{key}.q_a_layernorm.weight"] = torch.ones(q_lora, dtype = torch.half)
        weights[f"{key}.q_b_proj.weight"] = rand(H * (nope + rope), q_lora)
    kwargs = dict(key = key, layer_idx = 5, hidden_size = hidden, num_q_heads = H,
                  kv_lora_rank = rank, qk_nope_head_dim = nope, qk_rope_head_dim = rope,
                  v_head_dim = v, rope_settings = None, q_lora_rank = q_lora,
                  key_o = "dense", qmap = "attn", key_gate = "g_proj")
    return weights, kwargs


def _load_class(filename, name, ns):
    tree = ast.parse(filename.read_text())
    body: list[ast.stmt] = [ast.ImportFrom(module = "__future__", names = [ast.alias(name = "annotations")], level = 0)]
    body.extend(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    assert len(body) == 2, (filename, name)
    exec(compile(ast.fix_missing_locations(ast.Module(body = body, type_ignores = [])),
                 str(filename), "exec"), ns)
    return ns[name]


@contextmanager
def cpu_source():
    # Private package names prevent contaminating a real ExLlama import in the GPU suite.
    name = "_mla_gate_cpu"
    native = SimpleNamespace(BC_LinearFP16 = Mock(return_value = object()))
    ns: dict = dict(__name__ = name + ".modules.mla_attn", __package__ = name + ".modules",
              torch = torch, nn = torch.nn, os = os, ABC = ABC, abstractmethod = abstractmethod,
              cached_property = cached_property, override = lambda f: f, ext = native,
              to2 = lambda x, dtype, default: x.to(dtype or default or x.dtype),
              to_device = lambda x, device: x.to(device),
              get_for_device = lambda p, k, d, *default: p.get(k, *default),
              TPAllocation = lambda **kw: SimpleNamespace(**kw),
              MAX_DECODE_QLEN = 16, _bc_max_bsz = 32, _bc_mla_enable = True,
              _prefill_mode = "mha", _score_tile = 32768,
              NoFittingConfig = type("NoFittingConfig", (Exception,), {}))
    for path, cls in (("module.py", "Module"), ("quant/fp16.py", "LinearFP16"),
                      ("linear.py", "Linear"), ("rmsnorm.py", "RMSNorm"),
                      ("mla_attn.py", "MLAttention")):
        _load_class(MODULES / path, cls, ns)
    # RMSNorm math is not under test here; use its existing Torch path rather than CUDA.
    ns["RMSNorm"].forward = ns["RMSNorm"].forward_torch
    triton = ModuleType(name + ".modules.attention_fn.mla_triton")
    triton._dbg_sync = lambda *a: None
    triton._debug_sync = False
    triton.mla_kv_append = Mock()
    cache = ModuleType(name + ".cache")
    cache.CacheLayer_MLA_quant = type("QuantCache", (), {})
    ns["QuantCache"] = cache.CacheLayer_MLA_quant
    config = ModuleType(name + ".model.config")
    config.NullConfig = SimpleNamespace
    constants = ModuleType(name + ".constants")
    constants.PAGE_SIZE = 256
    with patch.dict(sys.modules, {triton.__name__: triton, cache.__name__: cache,
                                  config.__name__: config, constants.__name__: constants}):
        yield ns, triton


def make_cpu(ns, **changes):
    weights, kwargs = fixture_weights()
    kwargs.update(changes)
    m = ns["MLAttention"](SimpleNamespace(stc = TensorCollection(weights)), **kwargs)
    m.load(torch.device("cpu"))
    return m, weights


def explicit_attention(q, c, uk, uv, qpe, kpe):
    """Expanded-K/V oracle independent of absorption or the production gate helper."""
    k = torch.einsum("bsc,hkc->bshk", c, uk)
    v = torch.einsum("bsc,hvc->bshv", c, uv)
    scores = torch.einsum("bthk,bshk->bhts", q, k)
    scores += torch.einsum("bthr,bsr->bhts", qpe, kpe)
    scores *= (q.shape[-1] + qpe.shape[-1]) ** -0.5
    length = q.shape[1]
    mask = torch.arange(length)[None, :] <= torch.arange(length)[:, None]
    p = scores.masked_fill(~mask.to(scores.device), -float("inf")).softmax(-1)
    return torch.einsum("bhts,bshv->bthv", p, v)


class TestMLAGateCPU(unittest.TestCase):
    def test_absorption_and_gate_order_independent_algebra(self):
        gen = torch.Generator().manual_seed(31)
        rnd = lambda *s: torch.randn(*s, generator = gen)
        B, S, H, C, K, V, R, D = 2, 7, 3, 5, 4, 2, 2, 6
        q, c, uk, uv = rnd(B, S, H, K), rnd(B, S, C), rnd(H, K, C), rnd(H, V, C)
        qpe, kpe, gate = rnd(B, S, H, R), rnd(B, S, R), rnd(B, S, H)
        wo = rnd(D, H * V)
        heads = explicit_attention(q, c, uk, uv, qpe, kpe)
        expected = (heads * (1 / (1 + torch.exp(-gate)))[..., None]).flatten(-2) @ wo.T
        absorbed = torch.einsum("bthk,hkc->bthc", q, uk)
        scores = torch.einsum("bthc,bsc->bhts", absorbed, c)
        scores += torch.einsum("bthr,bsr->bhts", qpe, kpe)
        scores *= (K + R) ** -0.5
        mask = torch.arange(S)[None, :] <= torch.arange(S)[:, None]
        p = scores.masked_fill(~mask, -float("inf")).softmax(-1)
        latent = torch.einsum("bhts,bsc->bthc", p, c)
        unfolded = torch.einsum("bthc,hvc->bthv", latent, uv)
        with cpu_source() as (ns, _):
            m = ns["MLAttention"].__new__(ns["MLAttention"])
            m.num_q_heads, m.v_head_dim = H, V
            m.o_proj = SimpleNamespace(forward = lambda x, params: x @ wo.T)
            actual = m._project_output(unfolded.flatten(-2).clone(), {}, gate)
        torch.testing.assert_close(actual, expected, rtol = 2e-5, atol = 2e-6)
        # A scalar averaged over heads, or gating after head mixing, is not this operation.
        wrong = (heads.flatten(-2) @ wo.T) * gate.sigmoid().mean(-1, keepdim = True)
        self.assertGreater((actual - wrong).abs().max().item(), 0.1)

    def test_gate_dtype_extremes_and_noncontiguous(self):
        with cpu_source() as (ns, _):
            m, _ = make_cpu(ns)
            m.o_proj = SimpleNamespace(forward = lambda x, p: x)
            for dtype in (torch.float, torch.half, torch.bfloat16):
                with self.subTest(dtype = dtype):
                    o = torch.arange(2 * 3 * 128).reshape(2, 3, 128).to(dtype)[..., ::2]
                    self.assertFalse(o.is_contiguous())
                    gate = torch.tensor([-1000., -1., 0., 1000.]).expand(2, 3, 4)
                    expected = o.clone().unflatten(-1, (4, 16))
                    expected = expected * (1 / (1 + torch.exp(-gate.double()))).to(dtype)[..., None]
                    actual = m._project_output(o, {}, gate)
                    torch.testing.assert_close(actual, expected.flatten(-2), rtol = 0, atol = 0)
                    self.assertEqual(actual.dtype, dtype)

    def test_native_gate_dimensions_targets_defaults(self):
        with cpu_source() as (ns, _):
            weights, kwargs = fixture_weights()
            kwargs.update(hidden_size = 2560, num_q_heads = 32, kv_lora_rank = 512,
                          qk_nope_head_dim = 128, qk_rope_head_dim = 64, v_head_dim = 128)
            m = ns["MLAttention"](SimpleNamespace(stc = TensorCollection(weights)), **kwargs)
            self.assertIsInstance(m.g_proj, ns["Linear"])
            self.assertEqual((m.g_proj.in_features, m.g_proj.out_features), (2560, 32))
            self.assertIsNone(m.g_proj.qmap)
            self.assertIn(m.g_proj, m.modules)
            self.assertNotIn(m.g_proj.key, str(m.optimizer_targets()))
            self.assertEqual(m.o_proj.key, m.key + ".dense")
            del kwargs["key_gate"]
            plain = ns["MLAttention"](m.config, **kwargs)
            explicit = ns["MLAttention"](m.config, **kwargs, key_gate = None)
            self.assertIsNone(plain.g_proj)
            self.assertEqual(plain.optimizer_targets(), explicit.optimizer_targets())
            self.assertEqual([x.key for x in plain.modules], [x.key for x in explicit.modules])
            signature = inspect.signature(ns["MLAttention"])
            self.assertIsNone(signature.parameters["key_gate"].default)
            self.assertEqual(list(signature.parameters)[-1], "key_gate")
            o = torch.randn(2, 3, 4096)
            spy = Mock(side_effect = lambda x, p: x)
            plain.o_proj = SimpleNamespace(forward = spy)
            self.assertIs(plain._project_output(o, {}), o)
            self.assertIs(spy.call_args.args[0], o)

    def test_deferred_gate_loader_conversion_unload_reload(self):
        with cpu_source() as (ns, _):
            weights, kwargs = fixture_weights()
            stc = TensorCollection(weights, defer = True)
            m = ns["MLAttention"](SimpleNamespace(stc = stc), **kwargs)
            m.load(torch.device("cpu"))
            gate_dest = m.g_proj.inner.weight
            self.assertTrue(any(dest is gate_dest for dest, _ in stc.pending))
            stc.finish()
            self.assertIs(m.g_proj.inner.weight, gate_dest)
            torch.testing.assert_close(gate_dest.T, weights[m.key + ".g_proj.weight"], rtol = 0, atol = 0)
            saved = {k: t for sub in m for k, t in sub.get_tensors().items()}
            self.assertIn(m.key + ".g_proj.weight", saved)
            torch.testing.assert_close(saved[m.key + ".g_proj.weight"], gate_dest.T, rtol = 0, atol = 0)
            m.unload()
            self.assertIsNone(m.g_proj.inner)
            self.assertIsNone(m.g_proj.device)
            m.load(torch.device("cpu"))
            stc.finish()
            torch.testing.assert_close(m.g_proj.inner.weight.T, saved[m.key + ".g_proj.weight"], rtol = 0, atol = 0)

    def test_tp_allocation_accounts_for_native_gate(self):
        with cpu_source() as (ns, _):
            m, _ = make_cpu(ns)
            plain, _ = make_cpu(ns, key_gate = None)
            gated_alloc = m.make_tp_allocation({})[0]
            plain_alloc = plain.make_tp_allocation({})[0]
            self.assertEqual(gated_alloc.max_devices, 1)
            self.assertEqual(gated_alloc.storage_to_split - plain_alloc.storage_to_split,
                             m.hidden_size * m.num_q_heads * torch.half.itemsize)
            self.assertGreaterEqual(gated_alloc.overhead_to_split - plain_alloc.overhead_to_split,
                                    m.num_q_heads * (torch.half.itemsize + 2 * torch.float.itemsize))

    def test_all_dispatch_exits_gate_before_output(self):
        # Exercise actual _attend and cached/cache-less wrappers; only attention kernels are
        # stubs. Distinct values per head catch skipped gates, wrong axes and double gating.
        for path, S, mode in (("decode", 3, "mha"), ("absorbed", 33, "absorbed"),
                              ("mha", 33, "mha"), ("fallback", 3, "mha"),
                              ("sparse", 7, "mha")):
            for cached in (False, "fp16", "quant"):
                with self.subTest(path = path, cached = cached), cpu_source() as (ns, triton):
                    m, weights = make_cpu(ns)
                    B, H, V = 2, m.num_q_heads, m.v_head_dim
                    x = torch.randn(B, S, m.hidden_size, generator = torch.Generator().manual_seed(17)).half()
                    gate = x @ weights[m.key + ".g_proj.weight"].T
                    heads = (torch.arange(B * S * H * V).reshape(B, S, H, V) / 1000 + 0.2).half()
                    gated = heads * gate.float().sigmoid().half()[..., None]
                    # Keep native Linear's padding/accumulation identical here. The independent
                    # algebra test above covers head mixing; this test isolates dispatch wiring.
                    expected = m.o_proj.forward(gated.flatten(-2), {})
                    project_output = Mock(wraps = m.o_proj.forward)
                    m.o_proj.forward = project_output
                    ns["_prefill_mode"] = mode
                    ns["mla_absorb"] = Mock(return_value = torch.zeros(H, B * S, m.kv_lora_rank))
                    latent = torch.zeros(H, B * S, m.kv_lora_rank)
                    decode = Mock(return_value = latent)
                    prefill = Mock(return_value = latent)
                    mha = Mock(side_effect = lambda *a, **k: heads.clone().reshape(B * S, H, V))
                    ns.update(mla_attn_triton_decode = decode, mla_attn_triton_prefill = prefill,
                              mla_attn_triton_prefill_mha = mha,
                              mla_unfold = Mock(side_effect = lambda *a: heads.clone().reshape(B * S, H, V)))
                    if path == "fallback":
                        decode.side_effect = ns["NoFittingConfig"]()
                    if path == "sparse":
                        m.indexer_mode, m.index_topk = "shared", 1
                        m._attend_sparse = Mock(return_value = latent)
                    params = {"dsa_topk_indices": torch.zeros(B * S, 32, dtype = torch.int32)}
                    # Host lengths only: no CUDA device reads in this CPU harness.
                    ns["_host_seqlens"] = lambda *a: [0] * B
                    original_project = m.project_q

                    def reuse_input(x, params, return_resid):
                        q = original_project(x, params, return_resid)
                        x.zero_()   # gate must already have been projected, independently of x
                        return q

                    m.project_q = reuse_input
                    gate_spy = Mock(wraps = m.g_proj.forward)
                    m.g_proj.forward = gate_spy
                    if cached:
                        layer = ns["QuantCache"]() if cached == "quant" else SimpleNamespace()
                        layer.get_kv = Mock(return_value = (None, None))
                        layer.get_qc = Mock(return_value = (None, "scales", None, 4))
                        layer.update_kv_direct = Mock()
                        params.update(attn_mode = "flash_attn", cache = layer,
                                      block_table = torch.zeros(B, 1, dtype = torch.int32),
                                      cache_seqlens = torch.zeros(B, dtype = torch.int32))
                    else:
                        params["attn_mode"] = "flash_attn_nc"
                    actual = m.forward(x, params)
                    torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)
                    project_output.assert_called_once()
                    torch.testing.assert_close(project_output.call_args.args[0], gated.flatten(-2), rtol = 0, atol = 0)
                    gate_spy.assert_called_once()
                    if path in ("mha", "fallback"):
                        mha.assert_called_once()
                    elif path == "sparse":
                        m._attend_sparse.assert_called_once()
                    elif path == "decode":
                        decode.assert_called_once()
                    else:
                        prefill.assert_called_once()
                    if cached:
                        layer.update_kv_direct.assert_called_once()
                        if cached == "quant":
                            layer.get_qc.assert_called_once()
                    else:
                        triton.mla_kv_append.assert_called_once()

    def test_bc_declines_only_gated_modules_and_autosplit(self):
        with cpu_source() as (ns, _):
            m, _ = make_cpu(ns)
            layer = object()
            graph = SimpleNamespace(step = Mock(return_value = "ungated graph"))
            m.dispatch_cache[("bcm", id(layer))] = graph
            self.assertIsNone(m.bc_mla_step(None, {}, layer, None, None))
            m._autosplit_layer = Mock(side_effect = AssertionError("should not allocate BC statics"))
            m.autosplit_prepare({})
            m._autosplit_layer.assert_not_called()
            m.g_proj = None
            self.assertEqual(m.bc_mla_step(torch.zeros(1), {}, layer, None, None), "ungated graph")
            graph.step.assert_called_once()

    def test_tp_gate_export_import_owner_and_stub(self):
        with cpu_source() as (ns, _):
            m, weights = make_cpu(ns)
            # Export/import only the real gate Linear; other child transport is orthogonal.
            for n in m._tp_submodules:
                child = getattr(m, n)
                if child is not None and n != "g_proj":
                    child.tp_export = lambda plan, prod, child = child: {
                        "cls": SimpleNamespace(tp_import = lambda *args, child = child: child)}
            producer = SimpleNamespace(send = lambda t: t)
            consumer = SimpleNamespace(recv = lambda t, **kw: t)
            exported = m.tp_export({}, producer)
            self.assertEqual(exported["kwargs"]["key_gate"], "g_proj")
            self.assertIn("g_proj", exported)
            context = {"device": torch.device("cpu"), "consumer": consumer}
            with patch.object(torch.cuda, "synchronize", return_value = None):
                owner = ns["MLAttention"].tp_import(context, exported, {m.key: (0, 4, "heads")}, skip_reduction = True)
                stub = ns["MLAttention"].tp_import(context, exported, {m.key: (0, 0, "heads")}, skip_reduction = True)
            self.assertEqual(owner.g_proj.out_features, 4)   # no TP re-padding to 128
            self.assertEqual(owner.g_proj.in_features, 128)
            x = torch.randn(2, 3, 128).half()
            torch.testing.assert_close(owner.g_proj.forward(x, {}),
                                       x @ weights[m.key + ".g_proj.weight"].T, rtol = 0, atol = 0)
            self.assertIsNone(stub.g_proj)
            self.assertEqual(stub.modules, [])
            torch.testing.assert_close(stub.forward(x, {}), torch.zeros_like(x), rtol = 0, atol = 0)


@unittest.skipUnless(os.environ.get("EXL3_TEST_MLA_GATE_GPU") == "1",
                     "real GPU module tests require explicit EXL3_TEST_MLA_GATE_GPU=1")
class TestMLAGateGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import only after opt-in. A default CPU run cannot trigger extension JIT.
        sys.path.insert(0, str(ROOT))
        from exllamav3.modules import MLAttention
        from exllamav3.cache import CacheLayer_MLA_fp16
        from exllamav3.util.rope import RopeSettings, RopeStyle
        from exllamav3.model.config import InferParams
        import exllamav3.modules.mla_attn as impl
        cls.MLA, cls.Cache = MLAttention, CacheLayer_MLA_fp16
        cls.RopeSettings, cls.RopeStyle = RopeSettings, RopeStyle
        cls.InferParams, cls.impl = InferParams, impl
        cls.device = torch.device("cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU tests explicitly requested but CUDA is unavailable")

    def build(self, q_lora = None, gated = True, sparse = False, production = False):
        weights, kwargs = fixture_weights(H = 32 if production else 8,
                                          hidden = 2560 if production else 512,
                                          rank = 512, nope = 128, rope = 64, v = 128,
                                          q_lora = q_lora)
        kwargs["rope_settings"] = self.RopeSettings(head_dim = 64, rope_theta = 6000000.0,
                                                     rope_style = self.RopeStyle.GPTJ)
        if not gated:
            kwargs["key_gate"] = None
        if sparse:
            kwargs.update(indexer_mode = "shared", index_topk = 2)
        config = SimpleNamespace(stc = TensorCollection(weights), infer_params = self.InferParams())
        m = self.MLA(config, **kwargs)
        m.load(self.device)
        self.addCleanup(m.unload)
        return m, {k: v.to(self.device) for k, v in weights.items()}

    def oracle(self, m, w, x, indices = None, gated = True):
        """Expanded fp32 reference, including independent adjacent-pair RoPE and gate."""
        B, S, _ = x.shape
        H, R, N, V = m.num_q_heads, m.qk_rope_head_dim, m.qk_nope_head_dim, m.v_head_dim
        xf = x.float()

        def linear(x, name):
            return x @ w[m.key + "." + name + ".weight"].float().T

        def norm(x, name):
            return x * torch.rsqrt(x.square().mean(-1, keepdim = True) + m.norm_eps) * w[m.key + "." + name + ".weight"].float()

        q = linear(xf, "q_proj") if m.q_lora_rank is None else linear(norm(linear(xf, "q_a_proj"), "q_a_layernorm"), "q_b_proj")
        q = q.view(B, S, H, N + R)
        cpe = linear(xf, "kv_a_proj_with_mqa")
        c = norm(cpe[..., :m.kv_lora_rank], "kv_a_layernorm")
        qpe, kpe = q[..., N:], cpe[..., m.kv_lora_rank:].view(B, S, 1, R)
        freq = 6000000.0 ** (-torch.arange(0, R, 2, device = x.device).float() / R)
        angle = torch.arange(S, device = x.device).float()[:, None] * freq
        cos, sin = angle.cos()[None, :, None, :], angle.sin()[None, :, None, :]

        def rotate(t):
            a, b = t[..., ::2], t[..., 1::2]
            return torch.stack((a * cos - b * sin, b * cos + a * sin), dim = -1).flatten(-2)

        qpe, kpe = rotate(qpe), rotate(kpe)
        kv = linear(c, "kv_b_proj").view(B, S, H, N + V)
        qfull = torch.cat((q[..., :N], qpe), -1)
        kfull = torch.cat((kv[..., :N], kpe.expand(B, S, H, R)), -1)
        scores = torch.einsum("bthd,bshd->bhts", qfull, kfull) * m.sm_scale
        mask = torch.arange(S, device = x.device)[None, :] <= torch.arange(S, device = x.device)[:, None]
        scores = scores.masked_fill(~mask, -float("inf"))
        if indices is not None:
            selection = torch.zeros(B, S, S, dtype = torch.bool, device = x.device)
            idx = indices.view(B, S, -1).long()
            selection.scatter_(-1, idx.clamp_min(0), True)
            scores = scores.masked_fill(~selection[:, None], -float("inf"))
        heads = torch.einsum("bhts,bshv->bthv", scores.softmax(-1), kv[..., N:])
        if gated:
            heads *= (1 / (1 + torch.exp(-linear(xf, "g_proj"))))[..., None]
        return linear(heads.flatten(-2), "dense")

    def assert_accuracy(self, actual, expected, label):
        self.assertTrue(torch.isfinite(actual).all().item(), label)
        delta = actual.float() - expected.float()
        relmax = delta.abs().max().item() / max(expected.abs().max().item(), 1e-6)
        relrms = delta.square().mean().sqrt().item() / max(expected.square().mean().sqrt().item(), 1e-6)
        print(f"{label}: relative_max={relmax:.6g} relative_rms={relrms:.6g}")
        # Predeclared fp16 module vs expanded fp32 oracle thresholds; do not retune on failure.
        self.assertLess(relmax, 1e-2, label)
        self.assertLess(relrms, 5e-3, label)

    def cached(self, m, x, chunks):
        B, S, _ = x.shape
        pages = (S + 255) // 256
        layer = self.Cache(None, m, 0, B * pages * 256)
        layer.alloc(self.device)
        try:
            self.assertEqual(layer.k.shape, (B * pages, 256, 1, 512))
            self.assertEqual(layer.v.shape, (B * pages, 256, 1, 64))
            bt = torch.arange(B * pages, dtype = torch.int32).flip(0).reshape(B, pages)
            outputs = []
            for start in range(0, S, chunks):
                length = min(chunks, S - start)
                seq = torch.full((B,), start, dtype = torch.int32)
                outputs.append(m.forward(x[:, start:start + length].contiguous(), {
                    "attn_mode": "flash_attn", "cache": layer, "block_table": bt,
                    "cache_seqlens": seq, "positions": seq.clone(),
                }))
            return torch.cat(outputs, 1)
        finally:
            layer.free()

    def test_actual_nocache_calibration_prefill_decode(self):
        for q_lora in (None, 256):
            m, w = self.build(q_lora)
            for S in (1, 2, 3, 4, 7, 31, 32, 33, 127, 128, 129, 257):
                gen = torch.Generator(device = self.device).manual_seed(91 + S)
                x = torch.randn(2, S, m.hidden_size, device = self.device, generator = gen).half() * 0.5
                ref = self.oracle(m, w, x)
                with torch.inference_mode():
                    for mode in ("mha", "absorbed"):
                        with patch.object(self.impl, "_prefill_mode", mode):
                            out = m.forward(x, {"attn_mode": "flash_attn_nc"})
                            self.assert_accuracy(out, ref, f"nc/{mode}/q={q_lora}/S={S}")
                    if S in (3, 33, 129, 257):
                        for chunk in (1, 17, S):
                            out = self.cached(m, x, chunk)
                            self.assert_accuracy(out, ref, f"cache/chunk={chunk}/q={q_lora}/S={S}")
                self.assertFalse(any(m.dispatch_cache.values()), "gated model must not build BC graph")
            # Real calibration capture must see the gated o_proj input, not gate weights.
            with torch.inference_mode():
                captured = {}
                m.forward(x, {"attn_mode": "flash_attn_nc", "capture": captured})
                self.assertIn("attn.o", captured)
                self.assertNotIn(None, captured)

    def test_actual_sparse_output_gate(self):
        m, w = self.build(sparse = True)
        x = torch.randn(2, 33, m.hidden_size, device = self.device).half() * 0.5
        # Deterministic shared selection: first and current token, padded with -1.
        indices = torch.full((2, 33, 32), -1, dtype = torch.int32, device = self.device)
        indices[..., 0] = 0
        indices[:, :, 1] = torch.arange(33, device = self.device)
        indices[:, 0, 1] = -1   # no duplicate first-token entry
        with torch.inference_mode():
            out = m.forward(x, {"attn_mode": "flash_attn_nc", "dsa_topk_indices": indices.flatten(0, 1)})
            ref = self.oracle(m, w, x, indices)
        self.assert_accuracy(out, ref, "sparse-gate")

    def test_actual_production_geometry_default_and_reload(self):
        m, w = self.build(production = True)
        x = torch.randn(2, 33, 2560, device = self.device).half() * 0.5
        with torch.inference_mode():
            actual = m.forward(x, {})
            self.assert_accuracy(actual, self.oracle(m, w, x), "Ling geometry")
            saved = {k: t.clone() for sub in m for k, t in sub.get_tensors().items()}
            self.assertIn(m.key + ".g_proj.weight", saved)
            m.unload()
            self.assertIsNone(m.g_proj.inner)
            m.load(self.device)
            torch.testing.assert_close(m.forward(x, {}), actual, rtol = 0, atol = 0)
        plain, pw = self.build(gated = False)
        x = x[..., :plain.hidden_size].contiguous()
        with torch.inference_mode():
            ungated = plain.forward(x, {})
            self.assert_accuracy(ungated, self.oracle(plain, pw, x, gated = False), "default unchanged")

    def test_actual_tp_recreation_and_stub(self):
        # Same-device transport exercises real native modules/TP serialization, not IPC or
        # cross-GPU collectives; those still require the parent's full-model TP test.
        m, _ = self.build()
        x = torch.randn(2, 7, m.hidden_size, device = self.device).half() * 0.5
        producer = SimpleNamespace(send = lambda t: t)
        consumer = SimpleNamespace(recv = lambda t, **kw: t)
        exported = m.tp_export({}, producer)
        context = {"device": self.device, "consumer": consumer}
        owner = self.MLA.tp_import(context, exported, {m.key: (0, m.num_q_heads, "heads")},
                                   skip_reduction = True)
        stub = self.MLA.tp_import(context, exported, {m.key: (0, 0, "heads")}, skip_reduction = True)
        self.addCleanup(owner.unload)
        self.addCleanup(stub.unload)
        self.assertEqual(owner.g_proj.out_features, m.num_q_heads)
        self.assertIsNone(stub.g_proj)
        with torch.inference_mode():
            torch.testing.assert_close(owner.forward(x, {}), m.forward(x, {}), rtol = 0, atol = 0)
            torch.testing.assert_close(stub.forward(x, {}), torch.zeros_like(x), rtol = 0, atol = 0)


if __name__ == "__main__":
    unittest.main()
