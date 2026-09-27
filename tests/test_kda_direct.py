"""Direct KDA gates: CPU contract tests and opt-in compiled GPU parity.

CPU: python -B tests/test_kda_direct.py -v
GPU (scheduled separately): EXL3_TEST_KDA_GPU=1 python -B tests/test_kda_direct.py -v
Select one physical GPU with CUDA_VISIBLE_DEVICES; GPU tests use cuda:0.

The CPU harness extracts local production Python classes without importing the
CUDA/JIT package. Linear/loading/forward/TP methods are exercised, but extension
calls and tensor transport are CPU test adapters, NOT compiled-kernel evidence.
The independent oracle uses explicit convolution windows and the matrix-form
KDA transition, not the implementation's delta-update loop. GPU tests import the
real package only after explicit opt-in. They may build/JIT and are unrun by the
CPU command. No downloaded model Python or model weights are executed.
"""
from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from functools import cached_property
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
KEY = "model.layers.0.attention"
LENGTHS = (1, 2, 3, 4, 7, 31, 32, 33, 127, 128, 129, 257)


def _extract(path, names, namespace: dict[str, Any]):
    """Execute only selected local definitions; no package imports or JIT."""
    tree = ast.parse((ROOT / path).read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), namespace)


def _log_decay(f, a_log, dt):
    # Independent double-precision expression; no production gate helper used.
    b, s, h, d = f.shape
    scaled = (f.double() + dt.double().reshape(1, 1, h, d)) * a_log.double().exp().reshape(1, 1, h, 1)
    return -5.0 / (1.0 + torch.exp(-scaled))


def _recurrence_oracle(q, k, v, log_decay, beta, initial):
    """S_t = (I - beta kk^T) diag(exp(ell)) S_(t-1) + beta kv^T."""
    q, k, v, log_decay, beta = [t.double() for t in (q, k, v, log_decay, beta)]
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    state = initial.double().clone()
    eye = torch.eye(k.shape[-1], dtype=torch.float64, device=k.device)
    outputs, states = [], []
    for t in range(q.shape[1]):
        kt = k[:, t].unsqueeze(-1)
        bt = beta[:, t, :, None, None]
        transition = (eye - bt * (kt @ kt.transpose(-1, -2))) @ torch.diag_embed(log_decay[:, t].exp())
        state = transition @ state + bt * (kt @ v[:, t].unsqueeze(-2))
        outputs.append((q[:, t].unsqueeze(-2) @ state).squeeze(-2) / math.sqrt(q.shape[-1]))
        states.append(state.clone())
    return torch.stack(outputs, 1), torch.stack(states, 1)


def _conv_oracle(raw, weight, history):
    """Explicit oldest-to-newest window, retaining raw (not SiLU) history."""
    width = weight.shape[-1]
    window = history.double().clone()
    outputs = []
    for t in range(raw.shape[1]):
        window = torch.cat((window[..., 1:], raw[:, t].double().unsqueeze(-1)), -1)
        summed = sum(window[..., j] * weight[:, j].double() for j in range(width))
        outputs.append(summed / (1.0 + torch.exp(-summed)))
    return torch.stack(outputs, 1), window


class _TensorSource:
    """In-memory loader implementing the flags exercised by real Linear.load_fp16.

    Deferred destinations are NaN-poisoned and filled in place after load_local,
    so a premature copy/rebind cannot accidentally pass by using zeros.
    """
    def __init__(self, tensors, defer=False):
        self.tensors = tensors
        self.defer = defer
        self.pending = []
        self.requests = []

    def has_tensor(self, key):
        return key in self.tensors

    def has_tensor_group(self, key, suffixes):
        if isinstance(key, list):
            return all(self.has_tensor_group(k, suffixes) for k in key)
        return all(any(f"{key}.{s}" in self.tensors for s in suffix) if isinstance(suffix, list)
                   else f"{key}.{suffix}" in self.tensors for suffix in suffixes)

    def get_tensor(self, key, device, optional=False, allow_bf16=False, float2half=False,
                   transpose=False, pad_to=None, no_defer=False, **kwargs):
        self.requests.append((key, no_defer))
        if key not in self.tensors:
            if optional:
                return None
            raise KeyError(key)
        source = self.tensors[key]
        value = source.T if transpose else source
        if float2half or (value.dtype == torch.bfloat16 and not allow_bf16):
            value = value.half()
        value = value.to(device).contiguous()
        if pad_to is not None and tuple(value.shape) != tuple(pad_to):
            padded = torch.zeros(pad_to, dtype=value.dtype, device=device)
            padded[tuple(slice(0, n) for n in value.shape)] = value
            value = padded
        if self.defer and not no_defer:
            target = torch.full_like(value, float("nan"))
            self.pending.append((key, target, value))
            return target
        return value.clone()

    def finish(self):
        for _, target, value in self.pending:
            target.copy_(value)

    def get_tensor_sizes(self, prefix):
        return [t.numel() * t.element_size() for k, t in self.tensors.items() if k.startswith(prefix + ".")]


class _Wire:
    """Only transport is mocked; production Linear TP slicing remains in use."""
    def send(self, tensor):
        return tensor

    def recv(self, tensor, cuda=True, slice_dim=None, first=None, last=None):
        if tensor is None:
            return None
        if slice_dim is not None:
            assert first is not None and last is not None
            tensor = tensor.narrow(slice_dim, first, last - first)
        return tensor.clone()

    @staticmethod
    def tp_export(tensor, plan, producer):
        return {"cls": _Wire, "tensor": tensor}

    @staticmethod
    def tp_import_split(context, exported, plan, split):
        _, first, last = split
        return exported["tensor"][first:last].clone()

    @staticmethod
    def tp_import_split_3(context, exported, plan, *splits):
        return torch.cat([exported["tensor"][first:last] for _, first, last in splits]).clone()


class _CPUExt:
    @staticmethod
    def BC_LinearFP16(*args):
        return SimpleNamespace()

    @staticmethod
    def BC_GatedRMSNorm(*args):
        return SimpleNamespace()

    @staticmethod
    def hgemm(x, weight, out):
        out.copy_(x.float() @ weight.float())

    @staticmethod
    def gated_rms_norm(x, weight, out, gate, eps, bias, groups, first, activation):
        assert activation == 1 and not first and groups == 1 and bias == 0
        h = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        out.copy_(h * weight.float() * torch.sigmoid(gate.float()))

    def __getattr__(self, name):
        raise AssertionError(f"Unexpected extension call in CPU test: {name}")


def _cpu_namespace():
    ns: dict[str, Any] = dict(torch=torch, F=F, nn=torch.nn, np=np, ABC=ABC, abstractmethod=abstractmethod,
              cached_property=cached_property, override=lambda f: f, ext=_CPUExt(),
              TPAllocation=lambda **kw: SimpleNamespace(**kw), TPTensorWrapper=_Wire,
              _proj_dtype=torch.half, _gate_dtype=torch.float, _conv_token_major=True,
              _bc_gdn_enable=True, _qkv_slice_enable=False, _BC_MAX_BSZ=8, _BC_MAX_QLEN=32,
              to2=lambda x, *dtypes: x.to(next((d for d in dtypes if d is not None), x.dtype)),
              get_for_device=lambda params, key, device, default=None: params.get(key, default))
    _extract("exllamav3/modules/module.py", ["Module"], ns)
    actual_module = ns["Module"]

    class CPUBaseModule(actual_module):
        def __init__(self, config, key, qmap):
            # Avoid the package-relative NullConfig import, which initializes the engine.
            super().__init__(config or SimpleNamespace(), key, qmap)

    ns["Module"] = CPUBaseModule
    _extract("exllamav3/modules/quant/fp16.py", ["LinearFP16"], ns)
    _extract("exllamav3/modules/linear.py", ["Linear"], ns)
    _extract("exllamav3/modules/gated_rmsnorm.py", ["GatedRMSNorm"], ns)
    _extract("exllamav3/modules/gated_delta_net.py", ["GatedDeltaNet", "GDNLayerState"], ns)
    _extract("exllamav3/modules/gated_delta_net_fn/gated_delta_rule.py", ["torch_recurrent_kda"], ns)

    def conv_adapter(mixed_qkv, conv_state, recurrent_slots, conv1d_weight, conv1d_bias,
                     history, params, token_major):
        x = mixed_qkv.transpose(1, 2) if token_major else mixed_qkv
        width = conv1d_weight.shape[-1]
        slots = list(range(x.shape[0])) if recurrent_slots is None else recurrent_slots.tolist()
        prior = torch.zeros(x.shape[0], x.shape[1], width) if conv_state is None else conv_state[slots, :, :width].float()
        full = torch.cat((prior, x.float()), -1)
        y = F.silu(F.conv1d(full, conv1d_weight.float().unsqueeze(1), conv1d_bias, groups=x.shape[1]))[..., -x.shape[-1]:]
        if conv_state is not None:
            for i, slot in enumerate(slots):
                if history:
                    count = min(conv_state.shape[-1], full.shape[-1])
                    conv_state[slot, :, -count:] = full[i, :, -count:]
                else:
                    conv_state[slot, :, :width] = full[i, :, -width:]
        return y.transpose(1, 2).contiguous().bfloat16()

    def recurrence_adapter(mixed_qkv, beta, g, recurrent_state, recurrent_slots, history,
                           save_state, num_k_heads, num_v_heads, k_dim, v_dim, k_head_dim,
                           v_head_dim, params, channelwise_g):
        assert channelwise_g and num_k_heads == num_v_heads
        b, s, _ = mixed_qkv.shape
        q, k, v = [t.reshape(b, s, num_k_heads, k_head_dim) for t in mixed_qkv.split((k_dim, k_dim, v_dim), -1)]
        params["probe"] = dict(g=g.clone(), beta=beta.clone(), q=q.clone(), k=k.clone(), v=v.clone())
        slots = list(range(b)) if recurrent_slots is None else recurrent_slots.tolist()
        outputs = []
        for bi, slot in enumerate(slots):
            state = torch.zeros(1, num_k_heads, k_head_dim, v_head_dim) if recurrent_state is None else recurrent_state[slot:slot+1, 0].clone()
            per_token = []
            for t in range(s):
                per_token.append(ns["torch_recurrent_kda"](q[bi:bi+1, t:t+1], k[bi:bi+1, t:t+1], v[bi:bi+1, t:t+1],
                                                          g[bi:bi+1, t:t+1], beta[bi:bi+1, t:t+1], state))
                if history and t < s - 1:
                    assert recurrent_state is not None
                    recurrent_state[slot, t+1] = state[0]
            if save_state:
                assert recurrent_state is not None
                recurrent_state[slot, 0] = state[0]
            outputs.append(torch.cat(per_token, 1))
        return torch.cat(outputs)

    ns.update(causal_conv1d_update=conv_adapter, gated_delta_rule_fn=recurrence_adapter)
    return ns


def _kwargs(mode="direct", hidden=128, heads=2, dim=64):
    kw = dict(config=SimpleNamespace(), key=KEY, layer_idx=0, hidden_size=hidden,
              num_k_heads=heads, num_v_heads=heads, k_head_dim=dim, v_head_dim=dim,
              rms_norm_eps=1e-6, conv_kernel_size=4, qmap=KEY, key_qkv="qkv_proj",
              key_qkv_alt=["q_proj", "k_proj", "v_proj"], key_b="b_proj",
              key_a_log="A_log", key_dt_bias="dt_bias", key_conv1d_q="q_conv1d",
              key_conv1d_k="k_conv1d", key_conv1d_v="v_conv1d", key_norm="o_norm",
              key_o="o_proj", out_dtype=torch.float)
    if mode == "direct":
        kw.update(key_f="f_proj", key_g="g_proj", gate_lower_bound=-5.0)
    elif mode == "low_rank":
        kw.update(key_f_a="f_a_proj", key_f_b="f_b_proj", key_g_a="g_a_proj", key_g_b="g_b_proj", gate_lower_bound=-5.0)
    elif mode == "gdn":
        kw.update(key_a="a_proj", key_z="z_proj")
    return kw


def _weights(kw):
    rng = torch.Generator().manual_seed(8741)
    hidden, heads, d = kw["hidden_size"], kw["num_k_heads"], kw["k_head_dim"]
    channels = heads * d
    tensors = {}
    shapes = {"q_proj": (channels, hidden), "k_proj": (channels, hidden), "v_proj": (channels, hidden),
              "b_proj": (heads, hidden), "o_proj": (hidden, channels)}
    if kw.get("key_f"):
        shapes.update(f_proj=(channels, hidden), g_proj=(channels, hidden))
    elif kw.get("key_f_a"):
        shapes.update(f_a_proj=(d, hidden), f_b_proj=(channels, d), g_a_proj=(d, hidden), g_b_proj=(channels, d))
    else:
        shapes.update(a_proj=(heads, hidden), z_proj=(channels, hidden))
    for name, shape in shapes.items():
        tensors[f"{KEY}.{name}.weight"] = (torch.randn(shape, generator=rng) / math.sqrt(shape[1])).bfloat16()
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        tensors[f"{KEY}.{name}.weight"] = (torch.randn(channels, 1, 4, generator=rng) * 0.4).bfloat16()
    tensors[f"{KEY}.o_norm.weight"] = torch.linspace(0.6, 1.4, d).bfloat16()
    tensors[f"{KEY}.A_log"] = torch.linspace(-1.5, 1.0, heads).bfloat16()
    dt_size = channels if kw.get("key_f") or kw.get("key_f_a") else heads
    tensors[f"{KEY}.dt_bias"] = torch.linspace(-3.0, 2.0, dt_size).bfloat16()
    return tensors


def _loaded(ns, mode="direct", defer=False, **shape):
    kw = _kwargs(mode, **shape)
    stc = _TensorSource(_weights(kw), defer)
    kw["config"] = SimpleNamespace(stc=stc)
    module = ns["GatedDeltaNet"](**kw)
    module.load(torch.device("cpu"))
    return module, stc


def _cache_params(module, layer_cls, bsz=2, max_history=0, device="cpu"):
    layer = layer_cls(module, bsz + 1, max_history, 123)
    layer.alloc(torch.device(device))
    cache = SimpleNamespace(get_recurrent_layer=lambda instance: layer)
    params = {"recurrent_states": [SimpleNamespace(exported=False, cache=cache)],
              "recurrent_slots": torch.tensor([bsz, 0], dtype=torch.int32, device=device)[:bsz]}
    return layer, params


def _forward_oracle(module, x, conv_state=None, state=None):
    """Independent full direct layer, with documented runtime rounding points."""
    b, s, _ = x.shape
    h, d = module.num_k_heads, module.k_head_dim
    def linear(p, value, dtype):
        return (value.double() @ p.inner.get_weight_tensor().double()).to(dtype)
    raw = linear(module.qkv_proj, x, torch.half)
    conv_weight = module.conv1d_weight_flat
    if conv_state is None:
        conv_state = torch.zeros(b, raw.shape[-1], module.conv_kernel_size, device=x.device)
    conv, final_conv = _conv_oracle(raw, conv_weight, conv_state)
    q, k, v = [t.reshape(b, s, h, d) for t in conv.bfloat16().chunk(3, -1)]
    f = linear(module.f_proj, x, torch.float).view(b, s, h, d)
    decay = _log_decay(f, module.a_log, module.dt_bias).float()
    beta = torch.sigmoid(linear(module.b_proj, x, torch.float)).bfloat16()
    if state is None:
        state = torch.zeros(b, h, d, d, device=x.device)
    out, states = _recurrence_oracle(q, k, v, decay, beta, state)
    out = out.bfloat16().double()
    out = out / torch.sqrt(out.square().mean(-1, keepdim=True) + module.rms_norm_eps)
    out *= module.norm.weight.double()
    gate = linear(module.g_proj, x, torch.float).reshape(b, s, h, d).double()
    out *= 1.0 / (1.0 + torch.exp(-gate))
    result = linear(module.o_proj, out.half().reshape(b, s, -1), torch.float)
    return result, states[:, -1].float(), final_conv.bfloat16()


class DirectKDACPUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.ns = _cpu_namespace()
        self.cls = self.ns["GatedDeltaNet"]

    def test_production_dimensions_native_gates_and_optimizer_targets(self):
        m = self.cls(**_kwargs(hidden=2560, heads=32, dim=128))
        self.assertTrue(m.kda and m.kda_direct)
        for p in (m.f_proj, m.g_proj):
            self.assertEqual((p.in_features, p.out_features), (2560, 4096))
            self.assertIsNone(p.qmap)
            self.assertEqual(p.out_dtype, torch.float)
            self.assertIn(p, m.modules)
        self.assertIsNone(m.f_a_proj)
        self.assertIsNone(m.g_b_proj)
        self.assertEqual(m.norm.gate_activation, "sigmoid")
        self.assertEqual(m.optimizer_targets(), [[[f"{KEY}.qkv_proj"], [f"{KEY}.o_proj"]]])
        quantizable = [p.key for p in m if isinstance(p, self.ns["Linear"]) and p.qmap is not None]
        self.assertEqual(quantizable, [f"{KEY}.qkv_proj", f"{KEY}.o_proj"])
        self.assertEqual(len({p.key for p in m}), len(list(m)))

    def test_invalid_direct_layouts_fail_early(self):
        invalid = [dict(key_f=None), dict(key_g=None), dict(key_f=""), dict(key_f_a="fa"),
                   dict(key_f_b="fb"), dict(key_g_a="ga"), dict(key_g_b="gb"),
                   dict(f_a_proj=object()), dict(f_b_proj=object()), dict(g_a_proj=object()), dict(g_b_proj=object()),
                   dict(f_proj=object()), dict(g_proj=object()), dict(key_fused_qkvz="qkvz"),
                   dict(key_fused_ba="ba"), dict(key_z="z"), dict(key_a="a"),
                   dict(z_proj=object()), dict(a_proj=object()), dict(key_qkv=None), dict(key_b=None),
                   dict(num_v_heads=4), dict(v_head_dim=128)]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(AssertionError):
                self.cls(**(_kwargs() | overrides))

    def test_default_low_rank_and_gdn_topologies_unchanged(self):
        low = self.cls(**_kwargs("low_rank"))
        self.assertTrue(low.kda)
        self.assertFalse(low.kda_direct)
        self.assertIsNone(low.f_proj)
        self.assertEqual((low.f_a_proj.in_features, low.f_a_proj.out_features), (128, 64))
        self.assertEqual((low.f_b_proj.in_features, low.f_b_proj.out_features), (64, 128))
        self.assertEqual(low.norm.gate_activation, "sigmoid")
        gdn = self.cls(**_kwargs("gdn"))
        self.assertFalse(gdn.kda or gdn.kda_direct)
        self.assertEqual(gdn.norm.gate_activation, "silu")
        self.assertEqual(gdn.optimizer_targets(), [[[f"{KEY}.qkv_proj"], [f"{KEY}.z_proj"], [f"{KEY}.o_proj"]]])

    def test_direct_loader_deferred_ownership_roundtrip_and_unload(self):
        m, stc = _loaded(self.ns, defer=True)
        for p in (m.f_proj, m.g_proj):
            pending = next(t for key, t, value in stc.pending if key == p.key + ".weight")
            self.assertEqual(p.inner.weight.data_ptr(), pending.data_ptr())
            self.assertTrue(torch.isnan(p.inner.weight).all())
        self.assertTrue(all(no_defer for key, no_defer in stc.requests if "conv1d" in key))
        stc.finish()
        preserved = {key: value for child in m for key, value in child.get_tensors().items()}
        for name in ("f_proj", "g_proj"):
            key = f"{KEY}.{name}.weight"
            torch.testing.assert_close(preserved[key], stc.tensors[key].half(), rtol=0, atol=0)
        # A real native-tensor serialization, not a claim of EXL3 quantization.
        from safetensors.torch import load, save
        serialized = load(save({k: t.contiguous() for k, t in preserved.items()}))
        reloaded = self.cls(**(_kwargs() | {"config": SimpleNamespace(stc=_TensorSource(serialized))}))
        reloaded.load(torch.device("cpu"))
        x = torch.randn(2, 7, 128, generator=torch.Generator().manual_seed(81)).half()
        torch.testing.assert_close(m.forward(x, {}), reloaded.forward(x, {}), rtol=0, atol=0)
        old_weight = m.f_proj.inner.weight
        m.unload()
        self.assertIsNone(m.f_proj.inner)
        self.assertIsNone(m.g_proj.inner)
        self.assertIsNone(m.conv1d_weight)
        m.load(torch.device("cpu"))
        stc.finish()
        self.assertIsNot(m.f_proj.inner.weight, old_weight)
        torch.testing.assert_close(m.forward(x, {}), reloaded.forward(x, {}), rtol=0, atol=0)

    def test_direct_declines_low_rank_batch_compiled_path(self):
        m, _ = _loaded(self.ns)
        m.qkv_proj.quant_type = m.o_proj.quant_type = "exl3"
        # Only a device label is passed; no GPU allocation or query. An accidental BC
        # construction calls the fail-closed _CPUExt and fails this test.
        m.load_local(torch.device("cuda:0"))
        self.assertIsNone(m.bc)
        self.assertFalse(m.bc_split)
        self.assertFalse(hasattr(m, "kda_fa_t"))

    def test_safe_gate_formula_beta_and_raw_output_gate(self):
        m, _ = _loaded(self.ns)
        with torch.no_grad():
            m.f_proj.inner.weight.zero_()
            m.f_proj.inner.weight[0] = torch.linspace(-100, 100, m.k_dim)
        x = torch.zeros(2, 3, 128, dtype=torch.half)
        x[..., 0] = 1
        params = {}
        seen = []
        original = m.norm.forward
        def norm_probe(y, params, **kw):
            seen.append(kw["gate"].clone())
            return original(y, params, **kw)
        with patch.object(m.norm, "forward", norm_probe):
            m.forward(x, params)
        f = (x.float() @ m.f_proj.inner.weight.float()).reshape(2, 3, 2, 64)
        expected = _log_decay(f, m.a_log, m.dt_bias)
        torch.testing.assert_close(params["probe"]["g"].double(), expected, rtol=3e-6, atol=1e-6)
        self.assertTrue(torch.isfinite(params["probe"]["g"]).all())
        self.assertTrue(((params["probe"]["g"] >= -5) & (params["probe"]["g"] <= 0)).all())
        wrong = (-m.a_log.float().exp().view(1, 1, 2, 1) * F.softplus(f + m.dt_bias.reshape(1, 1, 2, 64))).clamp(min=-5)
        self.assertGreater((wrong.double() - expected).abs().max().item(), 0.1)
        torch.testing.assert_close(params["probe"]["beta"], torch.sigmoid(x.float() @ m.b_proj.inner.weight.float()).bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(seen[0], (x.float() @ m.g_proj.inner.weight.float()).reshape(2, 3, 2, 64), rtol=0, atol=0)

    def test_low_rank_forward_preserves_half_latents_and_safe_gate(self):
        m, _ = _loaded(self.ns, "low_rank")
        x = torch.randn(2, 4, 128, generator=torch.Generator().manual_seed(82)).half()
        params = {}
        m.forward(x, params)
        latent = (x.float() @ m.f_a_proj.inner.weight.float()).half()
        f = (latent.float() @ m.f_b_proj.inner.weight.float()).reshape(2, 4, 2, 64)
        torch.testing.assert_close(params["probe"]["g"].double(), _log_decay(f, m.a_log, m.dt_bias), rtol=3e-6, atol=1e-6)

    def test_direct_unbounded_gate_and_half_projection_override(self):
        m, _ = _loaded(self.ns)
        m.gate_lower_bound = None
        self.ns["_gate_dtype"] = torch.half
        x = torch.randn(2, 3, 128, generator=torch.Generator().manual_seed(83)).half()
        params = {}
        m.forward(x, params)
        f = (x.float() @ m.f_proj.inner.weight.float()).half().double().reshape(2, 3, 2, 64)
        expected = -m.a_log.double().exp().reshape(1, 1, 2, 1) * torch.logaddexp(
            torch.zeros_like(f), f + m.dt_bias.double().reshape(1, 1, 2, 64))
        torch.testing.assert_close(params["probe"]["g"].double(), expected, rtol=3e-6, atol=1e-6)

    def test_recurrence_against_independent_matrix_oracle(self):
        rng = torch.Generator().manual_seed(830)
        for length in LENGTHS:
            with self.subTest(length=length):
                q, k, v = [torch.randn(2, length, 3, 4, generator=rng) for _ in range(3)]
                # Zero/tiny q/k and non-symmetric nonzero state expose norm/axis errors.
                q[0, 0] = 0
                k[1, 0] *= 1e-8
                g = -5 * torch.rand(2, length, 3, 4, generator=rng)
                beta = torch.rand(2, length, 3, generator=rng)
                initial = torch.randn(2, 3, 4, 4, generator=rng)
                expected, states = _recurrence_oracle(q, k, v, g, beta, initial)
                state = initial.clone()
                actual = self.ns["torch_recurrent_kda"](q, k, v, g, beta, state)
                torch.testing.assert_close(actual.float(), expected.bfloat16().float(), rtol=8e-3, atol=1e-4)
                torch.testing.assert_close(state.double(), states[:, -1], rtol=2e-5, atol=2e-6)
        # Explicit wrong-order negative control: decay on the right of correction is wrong.
        k = torch.tensor([0.6, 0.8], dtype=torch.float64)
        decay = torch.diag(torch.tensor([0.1, 0.9], dtype=torch.float64))
        correction = torch.eye(2, dtype=torch.float64) - 0.7 * torch.outer(k, k)
        self.assertGreater((correction @ decay - decay @ correction).abs().max().item(), 0.1)

    def test_conv_orientation_impulse_and_raw_history(self):
        weight = torch.tensor([[1., 2., 3., 4.], [-3., 1., 2., 5.]])
        raw = torch.zeros(1, 7, 2)
        raw[:, 0] = torch.tensor([1., -1.])
        expected, window = _conv_oracle(raw, weight, torch.zeros(1, 2, 4))
        params = {}
        actual = self.ns["causal_conv1d_update"](raw, None, None, weight, None, False, params, True)
        torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)
        self.assertEqual(expected[0, 0, 0].item(), F.silu(torch.tensor(4., dtype=torch.float64)).item())
        for split in (1, 2, 3, 4):
            state = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
            left = self.ns["causal_conv1d_update"](raw[:, :split], state, torch.tensor([0]), weight, None, False, {}, True)
            right = self.ns["causal_conv1d_update"](raw[:, split:], state, torch.tensor([0]), weight, None, False, {}, True)
            torch.testing.assert_close(torch.cat((left, right), 1), expected.bfloat16(), rtol=0, atol=0)
            torch.testing.assert_close(state, window.bfloat16(), rtol=0, atol=0)

    def test_direct_forward_matches_oracle_with_noncontiguous_input(self):
        m, _ = _loaded(self.ns)
        for length in (1, 7, 33, 129):
            with self.subTest(length=length):
                x = torch.randn(2, length, 256, generator=torch.Generator().manual_seed(length)).half()[..., ::2]
                self.assertFalse(x.is_contiguous())
                expected, _, _ = _forward_oracle(m, x)
                actual = m.forward(x, {})
                # BF16 recurrent readout then FP16 normalization output are quantization
                # boundaries; state and gate formulas have stricter tests above.
                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)

    def test_cache_nonzero_state_slots_history_and_rewind(self):
        m, _ = _loaded(self.ns)
        layer, params = _cache_params(m, self.ns["GDNLayerState"], max_history=7)
        rng = torch.Generator().manual_seed(875)
        layer.conv_state.copy_(torch.randn(layer.conv_state.shape, generator=rng))
        layer.recurrent_state.copy_(torch.randn(layer.recurrent_state.shape, generator=rng) * 0.1)
        self.assertEqual(layer.recurrent_state.dtype, torch.float32)
        original_conv = layer.conv_state.clone()
        original_state = layer.recurrent_state.clone()
        x = torch.randn(2, 7, 128, generator=rng).half()
        slots = params["recurrent_slots"].long()
        expected, final_state, final_conv = _forward_oracle(m, x, original_conv[slots, :, :4], original_state[slots, 0])
        params["recurrent_history"] = True
        actual = m.forward(x, params)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
        torch.testing.assert_close(layer.recurrent_state[slots, 0], final_state, rtol=3e-4, atol=1e-5)
        torch.testing.assert_close(layer.recurrent_state[1], original_state[1], rtol=0, atol=0)
        torch.testing.assert_close(layer.conv_state[1], original_conv[1], rtol=0, atol=0)
        for slot in slots.tolist():
            layer.rewind(slot, 6, 3)  # prepare_for_recurrence records seqlen - 1
        _, prefix_state, prefix_conv = _forward_oracle(m, x[:, :4], original_conv[slots, :, :4], original_state[slots, 0])
        torch.testing.assert_close(layer.recurrent_state[slots, 0], prefix_state, rtol=3e-4, atol=1e-5)
        torch.testing.assert_close(layer.conv_state[slots, :, :4], prefix_conv, rtol=0, atol=0)

    def test_tp_direct_and_low_rank_roundtrip_allocation_and_empty_rank(self):
        for mode in ("direct", "low_rank"):
            with self.subTest(mode=mode):
                m, _ = _loaded(self.ns, mode, heads=4)
                allocation = m.make_tp_allocation({})[0]
                split_names = ("qkv_proj", "b_proj", "f_proj", "g_proj", "o_proj") if mode == "direct" else ("qkv_proj", "b_proj", "f_b_proj", "g_b_proj", "o_proj")
                self.assertEqual(allocation.storage_to_split, sum(getattr(m, name).storage_size() for name in split_names))
                self.assertEqual(allocation.storage_per_device, 0 if mode == "direct" else m.f_a_proj.storage_size() + m.g_a_proj.storage_size())
                wire = _Wire()
                exported = m.tp_export({}, wire)
                context = {"device": torch.device("cpu"), "consumer": wire}
                with patch.object(torch.cuda, "synchronize", lambda: None):
                    imported = self.cls.tp_import(context, exported, {KEY: (1, 3, "K-heads")}, skip_reduction=True)
                    empty = self.cls.tp_import(context, exported, {KEY: (4, 4, "K-heads")}, skip_reduction=True)
                    ranks = [self.cls.tp_import(context, exported, {KEY: (first, first+2, "K-heads")}, skip_reduction=True)
                             for first in (0, 2)]
                self.assertEqual(imported.kda_direct, mode == "direct")
                self.assertEqual(imported.gate_lower_bound, -5)
                torch.testing.assert_close(imported.dt_bias, m.dt_bias[64:192], rtol=0, atol=0)
                torch.testing.assert_close(imported.a_log, m.a_log[1:3], rtol=0, atol=0)
                for name in (("f_proj", "g_proj") if mode == "direct" else ("f_b_proj", "g_b_proj")):
                    torch.testing.assert_close(getattr(imported, name).inner.weight, getattr(m, name).inner.weight[:, 64:192], rtol=0, atol=0)
                    self.assertIsNone(getattr(imported, name).qmap)
                if mode == "low_rank":
                    torch.testing.assert_close(imported.f_a_proj.inner.weight, m.f_a_proj.inner.weight, rtol=0, atol=0)
                x = torch.randn(2, 7, 128, generator=torch.Generator().manual_seed(877)).half()
                actual = sum(rank.forward(x, {}) for rank in ranks)
                torch.testing.assert_close(actual, m.forward(x, {}), rtol=1e-4, atol=2e-5)
                self.assertEqual(empty.num_k_heads, 0)
                torch.testing.assert_close(empty.forward(torch.ones(2, 1, 128), {}), torch.zeros(2, 1, 128), rtol=0, atol=0)
                empty.unload()
                self.assertIsNone(empty.device)


@unittest.skipUnless(os.environ.get("EXL3_TEST_KDA_GPU") == "1", "UNRUN: set EXL3_TEST_KDA_GPU=1 only in a scheduled GPU/JIT job")
class DirectKDAGPUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("UNRUN: CUDA unavailable")
        # Imports are intentionally below the opt-in guard.
        import sys
        sys.path.insert(0, str(ROOT))
        from exllamav3.ext import exllamav3_ext
        from exllamav3.modules.gated_delta_net import GatedDeltaNet, GDNLayerState
        cls.ext = exllamav3_ext
        cls.module_cls = GatedDeltaNet
        cls.layer_cls = GDNLayerState
        cls.device = torch.device("cuda:0")
        torch.cuda.set_device(cls.device)

    def _assert_error(self, actual, expected, label, rtol, atol):
        actual, expected = actual.float().cpu(), expected.float().cpu()
        error = actual - expected
        print(f"{label}: max_abs={error.abs().max().item():.8g} rms={error.square().mean().sqrt().item():.8g}")
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    def test_compiled_recurrence_channelwise_state_and_history(self):
        rng = torch.Generator().manual_seed(921)
        heads, d, bsz = 32, 128, 2
        for length in LENGTHS:
            for history in (False, True):
                with self.subTest(length=length, history=history):
                    mixed = (torch.randn(bsz, length, heads * d * 3, generator=rng) * 0.3).bfloat16()
                    q, k, v = [t.reshape(bsz, length, heads, d) for t in mixed.chunk(3, -1)]
                    f = torch.randn(bsz, length, heads, d, generator=rng) * 3
                    g = _log_decay(f, torch.linspace(-2, 2, heads), torch.linspace(-3, 3, heads*d)).float()
                    beta = torch.sigmoid(torch.randn(bsz, length, heads, generator=rng)).bfloat16()
                    slots = torch.tensor([2, 0], dtype=torch.int32)
                    state = torch.randn(3, length if history else 1, heads, d, d, generator=rng) * 0.1
                    expected, history_ref = _recurrence_oracle(q, k, v, g, beta, state[slots.long(), 0])
                    gpu_state = state.to(self.device)
                    output = torch.empty(bsz, length, heads, d, dtype=torch.bfloat16, device=self.device)
                    self.ext.cuda_recurrent_gated_delta_rule(mixed.to(self.device), g.to(self.device), beta.to(self.device),
                                                           gpu_state, output, heads, heads, d, d, slots.to(self.device), history)
                    torch.cuda.synchronize()
                    self._assert_error(output, expected, f"recurrence T={length} history={history}", 1e-2, 1e-3)
                    self._assert_error(gpu_state[slots.long().to(self.device), 0], history_ref[:, -1], "final state", 3e-4, 2e-5)
                    torch.testing.assert_close(gpu_state[1].cpu(), state[1], rtol=0, atol=0)
                    if history and length > 1:
                        self._assert_error(gpu_state[slots.long().to(self.device), 1:], history_ref[:, :-1], "history", 3e-4, 2e-5)

    def _forward_capture_qkv(self, module, x, params):
        # Keep the independent end-to-end oracle, but check cache bookkeeping against
        # the exact projected input consumed by the conv kernel. A separately evaluated
        # FP64 GEMM can round to a neighboring FP16 value, then a different BF16 bin.
        captured = []
        original = module.qkv_proj.forward
        def capture(*args, **kwargs):
            raw = original(*args, **kwargs)
            captured.append(raw.clone())
            return raw
        with patch.object(module.qkv_proj, "forward", side_effect=capture):
            output = module.forward(x, params)
        self.assertEqual(len(captured), 1)
        raw = captured[0]
        weight = module.qkv_proj.inner.get_weight_tensor().double()
        reference = x.double() @ weight
        # FP16 products are exact in FP32. Bound FP32 accumulation with gamma_n,
        # plus one local FP16 spacing for rounding either side of a bin boundary.
        u = torch.finfo(torch.float).eps / 2
        n = x.shape[-1]
        gamma = n * u / (1 - n * u)
        accum_bound = gamma * (x.double().abs() @ weight.abs())
        rounded = reference.half()
        spacing = (torch.nextafter(rounded, torch.full_like(rounded, float("inf"))) - rounded).double().abs()
        error = (raw.double() - reference).abs()
        self.assertTrue(torch.isfinite(raw).all())
        self.assertTrue((error <= accum_bound + spacing).all(), "native QKV projection exceeds rounding bound")
        return output, raw

    def test_direct_full_forward_prefill_decode_and_rewind(self):
        kw = _kwargs(hidden=128, heads=32, dim=128)
        kw["config"] = SimpleNamespace(stc=_TensorSource(_weights(kw)))
        m = self.module_cls(**kw)
        m.load(self.device)
        self.assertIsNone(m.bc)
        rng = torch.Generator().manual_seed(922)
        try:
            for length in LENGTHS:
                with self.subTest(length=length):
                    x = torch.randn(2, length, 128, generator=rng).half().to(self.device)
                    layer, params = _cache_params(m, self.layer_cls, max_history=length, device=self.device)
                    layer.conv_state.copy_(torch.randn(layer.conv_state.shape, generator=rng).to(self.device) * 0.1)
                    layer.recurrent_state.copy_(torch.randn(layer.recurrent_state.shape, generator=rng).to(self.device) * 0.1)
                    slots = params["recurrent_slots"].long()
                    conv0, state0 = layer.conv_state.clone(), layer.recurrent_state.clone()
                    expected, expected_state, expected_conv = _forward_oracle(m, x, conv0[slots, :, :4], state0[slots, 0])
                    actual, raw = self._forward_capture_qkv(m, x, params)  # native recurrent/chunk dispatch
                    _, exact_conv = _conv_oracle(raw, m.conv1d_weight_flat, conv0[slots, :, :4])
                    torch.cuda.synchronize()
                    self._assert_error(actual, expected, f"full layer T={length}", 3e-2, 1e-2)
                    self._assert_error(layer.recurrent_state[slots, 0], expected_state, "full-layer state", 2e-2, 2e-3)
                    torch.testing.assert_close(layer.conv_state[slots, :, :4], exact_conv.bfloat16(), rtol=0, atol=0)
                    # Independent token-by-token execution carries BF16 raw conv history;
                    # it need not be bit-identical to a token-major FP16 prefill.
                    layer.conv_state.copy_(conv0)
                    layer.recurrent_state.copy_(state0)
                    decoded = torch.cat([m.forward(x[:, t:t+1].contiguous(), params) for t in range(length)], 1)
                    self._assert_error(decoded, expected, f"decode T={length}", 3e-2, 2e-2)
                    if length >= 3:
                        layer.conv_state.copy_(conv0)
                        layer.recurrent_state.copy_(state0)
                        params["recurrent_history"] = True
                        _, history_raw = self._forward_capture_qkv(m, x, params)
                        for slot in slots.tolist():
                            layer.rewind(slot, length - 1, 2)
                        _, prefix_state, prefix_conv = _forward_oracle(m, x[:, :-2].contiguous(), conv0[slots, :, :4], state0[slots, 0])
                        self._assert_error(layer.recurrent_state[slots, 0], prefix_state, "rewound state", 2e-2, 2e-3)
                        _, exact_prefix_conv = _conv_oracle(history_raw[:, :-2], m.conv1d_weight_flat, conv0[slots, :, :4])
                        torch.testing.assert_close(layer.conv_state[slots, :, :4], exact_prefix_conv.bfloat16(), rtol=0, atol=0)
                    layer.free()
        finally:
            m.unload()
        self.assertIsNone(m.f_proj.inner)
        self.assertIsNone(m.g_proj.inner)


if __name__ == "__main__":
    unittest.main()
