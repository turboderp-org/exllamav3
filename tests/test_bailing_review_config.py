"""CPU-only regressions for Ling review R1/R2/R5/R6.

Extract local production definitions without importing the CUDA/JIT package.
Config parsing, Ling assembly, BlockSparseMLP construction/loading and TP metadata
run unchanged; module IO, kernels and TP transport are explicit CPU adapters.
The pinned JSON/Jinja are data, not downloaded Python. These are not native
kernel, real-weight, conversion or full-model/MTP generation qualification.
Set LING_REVIEW_HF to a copy of the pinned ef06d91f HF research directory if needed.
"""
from __future__ import annotations

import ast
from abc import ABC
from typing import Any, cast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import mock_open, patch

import torch
import torch.nn.functional as F
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[1]
HF = Path(os.environ.get("LING_REVIEW_HF", str(
    ROOT.parents[1] / "experiments/ling3-flash/research/official/hf")))
POSITIVE_INTS = (
    "hidden_size", "num_hidden_layers", "num_attention_heads", "head_dim",
    "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
    "intermediate_size", "moe_intermediate_size", "num_experts",
    "num_experts_per_tok", "n_group", "topk_group", "layer_group_size",
    "short_conv_kernel_size", "moe_shared_expert_intermediate_size",
    "max_position_embeddings", "vocab_size",
)
COUNTS = ("num_shared_experts", "first_k_dense_replace", "num_nextn_predict_layers")
ASSERTED_INTS = ("group_norm_size", "num_kv_heads_for_linear_attn",
                 "num_key_value_heads", "qk_head_dim", "rotary_dim", "q_lora_rank")
LIMITS = ("expert_swiglu_limit_list", "share_expert_swiglu_limit_list")
KEY = "model.layers.2.mlp"
PRIMARY = KEY + ".gate.expert_bias"
FALLBACK = KEY + ".gate.e_score_correction_bias"


def _extract(path, names, ns, adapt_mtp_import=False):
    tree = ast.parse((ROOT / path).read_text())
    selected = [n for n in tree.body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    if adapt_mtp_import:
        # The only nested import reached by config init would import the package.
        # Bind the real AST-extracted MTP class below instead; no other edits.
        for node in selected:
            if isinstance(node, ast.ClassDef) and node.name == "BailingMoeV3Config":
                init = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
                imports = [n for n in init.body if isinstance(n, ast.ImportFrom)]
                assert len(imports) == 1 and imports[0].module == "bailing_moe_v3_mtp"
                init.body.remove(imports[0])
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])),
                 str(ROOT / path), "exec"), ns)


def _forbidden(*args, **kwargs):
    raise AssertionError("Native/router/kernel work must not run in this CPU contract test")


class _Module:
    def __init__(self, config=None, key=None, *args, **kwargs):
        self.config, self.key = config, key
        self.modules, self.device = [], None
        self.__dict__.update(kwargs)

    def register_submodule(self, child):
        if child is not None:
            self.modules.append(child)

    def load(self, device, **kwargs):
        self.device = torch.device(device or "cpu")
        self.config.stc.events.append("children")

    def unload(self):
        self.device = None

    def get_tensors(self):
        return {}

    def tp_export(self, plan, producer):
        return {"cls": type(self), "child": self}

    @staticmethod
    def tp_import(context, exported, plan, **kwargs):
        return exported["child"]

    @staticmethod
    def tp_import_split(context, exported, plan, split):
        return exported["child"]


class _CPUOffload:
    def _cpu_init_state(self):
        self.cpu_offload = False

    def cpu_maybe_offload_load(self, device, **kwargs):
        self.config.stc.events.append("offload")
        return False

    def cpu_maybe_split_load(self, device, **kwargs):
        self.config.stc.events.append("split")

    def cpu_unload(self):
        pass


class _Model:
    def __init__(self, config, **kwargs):
        self.config, self.caps = config, {}


class _TensorSource:
    """Immediate/deferred loader contract, not a safetensors implementation."""
    def __init__(self, tensors, defer=True):
        self.tensors, self.defer = tensors, defer
        self.requests, self.events, self.destinations, self.pending = [], [], {}, []

    def get_tensor(self, key, device=None, optional=False, allow_bf16=False, no_defer=False, **kwargs):
        self.requests.append((key, optional, allow_bf16, no_defer))
        self.events.append(key)
        if key not in self.tensors:
            if optional:
                return None
            raise ValueError("Required tensor " + key)
        value = self.tensors[key].clone()
        if self.defer and not no_defer:
            dest = torch.full_like(value, float("nan"))
            self.pending.append((dest, value))
        else:
            dest = value
        self.destinations[key] = dest
        return dest

    def finish(self):
        for dest, value in self.pending:
            dest.copy_(value)


class _Wire:
    @staticmethod
    def send(tensor):
        return tensor

    @staticmethod
    def recv(tensor, **kwargs):
        return tensor


def _namespace():
    ns: dict[str, Any] = dict(__name__=__name__, ABC=ABC, torch=torch, F=F, math=math, cast=cast, T=Any,
              override=lambda f: f, no_default=object(), no_value=object(),
              Config=None, Module=_Module, BlockSparseMLP_CPU=_CPUOffload,
              Linear=_Module, MLP=_Module, GatedMLP=_Module, RMSNorm=_Module,
              Model=_Model, Embedding=_Module, MLAttention=_Module,
              GatedDeltaNet=_Module, TransformerBlock=_Module, GDNState=object,
              Qwen3_5MTPInputLayer=_Module, RopeSettings=SimpleNamespace,
              RopeStyle=SimpleNamespace(GPTJ="GPTJ"), TEMP_ROWS_GRAPH=32,
              TEMP_ROWS_FUSED=128, _replicated_router_types=("std", "std_bias", "dots", "sqrtsp", "sqrtsp_hash"),
              _routing_check=False, ext=SimpleNamespace(silu_mul=_forbidden),
              os=os, json=json, uuid=SimpleNamespace(uuid4=lambda: "cpu-test"),
              SafetensorsCollection=lambda *args: SimpleNamespace(),
              InferParams=lambda: SimpleNamespace(),
              routing_std=_forbidden, routing_std_bias=_forbidden, routing_ds3=_forbidden,
              routing_ds3_fp32=_forbidden, routing_dots=_forbidden, routing_sqrtsp=_forbidden,
              routing_sqrtsp_hash=_forbidden)
    _extract("exllamav3/util/file.py", {"read_dict"}, ns)
    _extract("exllamav3/model/config.py", {"Config"}, ns)
    _extract("exllamav3/modules/block_sparse_mlp.py", {"BlockSparseMLP"}, ns)
    _extract("exllamav3/architecture/bailing_moe_v3.py",
             {"BailingMoeV3Config", "BailingMoeV3Model", "bailing_mla", "bailing_mlp"}, ns, True)
    # R4 may replace the Ling block with a narrowly scoped subclass. Extract it
    # too if present; this test still adapts only its shared TransformerBlock base.
    mtp_path = "exllamav3/architecture/bailing_moe_v3_mtp.py"
    mtp_classes = {n.name for n in ast.parse((ROOT / mtp_path).read_text()).body
                   if isinstance(n, ast.ClassDef)}
    _extract(mtp_path, mtp_classes, ns)
    return ns


class ReviewConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _namespace()
        cls.pinned = json.loads((HF / "config.json").read_text())

    def config(self, updates=None, remove=()):
        raw = dict(self.pinned)
        raw.update(updates or {})
        for key in remove:
            raw.pop(key, None)
        # Run the real base initializer/read_dict too. Only disk IO is adapted.
        with patch("builtins.open", mock_open(read_data=json.dumps(raw))):
            return self.ns["BailingMoeV3Config"]("cpu-adapter")

    def test_pinned_valid(self):
        cfg = self.config()
        self.assertEqual([i for i, t in enumerate(cfg.layer_types) if t == "full_attention"],
                         [5, 11, 17, 23, 29, 35, 41])
        self.assertEqual(cfg.vocab_size, 157184)
        self.assertEqual(cfg.rope_settings.rope_theta, 6000000)
        self.assertEqual(cfg.rope_settings.max_position_embeddings, 262144)
        self.assertEqual(cfg.expert_swiglu_limits, self.pinned[LIMITS[0]])
        self.assertEqual(cfg.shared_swiglu_limits, self.pinned[LIMITS[1]])

    def test_invalid_positive_integer_matrix(self):
        for name in POSITIVE_INTS:
            for value in (True, False, 0, -1, 1.0, 1.5, float("nan"), float("inf"), -float("inf"), "4"):
                with self.subTest(name=name, value=value):
                    with self.assertRaises((ValueError, TypeError)):
                        self.config({name: value})

    def test_invalid_count_matrix(self):
        for name in COUNTS:
            for value in (True, False, -1, 0.0, 1.5, float("nan"), float("inf"), -float("inf"), "1"):
                with self.subTest(name=name, value=value):
                    with self.assertRaises((ValueError, TypeError)):
                        self.config({name: value})

    def test_required_dimensions_missing_or_null(self):
        for name in POSITIVE_INTS:
            with self.subTest(missing=name), self.assertRaises(ValueError):
                self.config(remove=(name,))
            with self.subTest(null=name), self.assertRaises(ValueError):
                self.config({name: None})

    def test_strict_asserted_integers(self):
        for name in ASSERTED_INTS:
            value = self.pinned[name]
            invalid = [True, False, float("nan"), float("inf"), "1"]
            if value is not None:
                invalid.append(float(value))
            for bad in invalid:
                with self.subTest(name=name, value=bad):
                    with self.assertRaises((ValueError, TypeError)):
                        self.config({name: bad})

    def test_invalid_scalar_matrix(self):
        for name in ("rope_theta", "rms_norm_eps", "routed_scaling_factor", "kda_lower_bound"):
            invalid = [True, False, 0, float("nan"), float("inf"), -float("inf"), "1"]
            invalid.append(1 if name == "kda_lower_bound" else -1)
            for bad in invalid:
                with self.subTest(name=name, value=bad):
                    with self.assertRaises((ValueError, TypeError)):
                        self.config({name: bad})

    def test_required_clamp_schedules(self):
        for remove in ((LIMITS[0],), (LIMITS[1],), LIMITS):
            with self.subTest(remove=remove), self.assertRaises(ValueError):
                self.config(remove=remove)
        for name in LIMITS:
            for bad in (None, [], [0] * 41, [0] * 43, [False] * 42,
                        [-1] * 42, [float("nan")] * 42, [float("inf")] * 42, ["0"] * 42):
                with self.subTest(name=name, value=bad), self.assertRaises((ValueError, TypeError)):
                    self.config({name: bad})

    def test_valid_mutation_matrix(self):
        mutations = [
            {"hidden_size": 128}, {"num_attention_heads": 16, "num_key_value_heads": 16},
            {"head_dim": 64}, {"kv_lora_rank": 256},
            {"qk_nope_head_dim": 64, "qk_head_dim": 128},
            {"qk_rope_head_dim": 32, "qk_head_dim": 160, "rotary_dim": 32},
            {"v_head_dim": 64}, {"intermediate_size": 128},
            {"moe_intermediate_size": 128}, {"moe_shared_expert_intermediate_size": 128},
            {"num_experts": 16, "n_group": 4, "topk_group": 2, "num_experts_per_tok": 2},
            {"layer_group_size": 7}, {"num_shared_experts": 0}, {"num_shared_experts": 2},
            {"first_k_dense_replace": 0}, {"first_k_dense_replace": 42},
            {"num_nextn_predict_layers": 0}, {"max_position_embeddings": 1},
            {"vocab_size": 1000}, {"rope_theta": 1}, {"rope_theta": 10000.5},
            {"rms_norm_eps": 1e-5}, {"kda_lower_bound": -1}, {"routed_scaling_factor": 1},
            {LIMITS[0]: [0] * 42, LIMITS[1]: [0.5] * 42},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.config(mutation)

    def test_convolution_bounds(self):
        for width in (1, 2, 4, 5, 16):
            with self.subTest(valid=width):
                self.assertEqual(self.config({"short_conv_kernel_size": width}).linear_conv_kernel_size, width)
        for width in (0, -1, 17, 32):
            with self.subTest(invalid=width), self.assertRaises(ValueError):
                self.config({"short_conv_kernel_size": width})

    def test_invalid_relations(self):
        for mutation in ({"num_nextn_predict_layers": 2}, {"first_k_dense_replace": 43},
                         {"num_experts": 513}, {"n_group": 512}, {"topk_group": 9},
                         {"num_experts_per_tok": 257}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.config(mutation)

    def test_reference_tail_schedule_and_mtp_identity(self):
        for group in (1, 2, 6, 7):
            for count in (1, 5, 6, 7, 41, 42, 43, 47):
                with self.subTest(group=group, count=count):
                    cfg = self.config({"num_hidden_layers": count, "layer_group_size": group,
                                       "first_k_dense_replace": min(2, count),
                                       LIMITS[0]: [4] * count, LIMITS[1]: [5] * count})
                    # Independent group construction: each full group's last
                    # layer is MLA; every member of a partial final group is MLA.
                    expected = []
                    for start in range(0, count, group):
                        size = min(group, count - start)
                        expected.extend(["full_attention"] * size if size < group else
                                        ["linear_attention"] * (size - 1) + ["full_attention"])
                    self.assertEqual(cfg.layer_types, expected)
                    mtp = self.ns["BailingMoeV3MTPModel"](cfg)
                    block = mtp.modules[mtp.first_block_idx]
                    self.assertEqual(block.key, f"model.layers.{count}")
                    self.assertEqual(block.layer_idx, 0)
                    self.assertEqual(block.attn.layer_idx, 0)
                    self.assertEqual(block.attn.key, f"model.layers.{count}.attention")
                    self.assertEqual(block.mlp.act_limit, 0)
                    self.assertEqual(block.mlp.shared_experts.act_limit, 0)

    def test_target_and_mtp_require_correction(self):
        cfg = self.config({"num_experts": 16, "n_group": 4, "topk_group": 2})
        target = self.ns["BailingMoeV3Model"](cfg)
        mtp = self.ns["BailingMoeV3MTPModel"](cfg)
        for block in target.modules[target.first_block_idx:target.last_kv_module_idx + 1] + [mtp.modules[1]]:
            if isinstance(block.mlp, self.ns["BlockSparseMLP"]):
                self.assertTrue(block.mlp.require_e_score_bias)
                self.assertEqual(block.mlp.e_score_correction_bias_key, "gate.expert_bias")
                cfg.stc = _TensorSource({})
                with self.assertRaisesRegex(ValueError, block.mlp.key):
                    block.mlp.load("cpu")
                self.assertNotIn("children", cfg.stc.events)


class PromptTests(unittest.TestCase):
    def test_pinned_template_bytes(self):
        ns = _namespace()
        source = (HF / "chat_template.jinja").read_bytes()
        # Pin the exact independent rendering oracle, not merely a model name.
        self.assertEqual(hashlib.sha256(source).hexdigest(), "b4ab3a1c8f748e6f874d9aea102333efe7ab82528e8ab81c2eb155851e8705c6")
        template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(source.decode())
        for system in (None, "", "Be concise.", "日本語\nPortuguês 👋", "detailed thinking on",
                       "detailed thinking off", "before detailed thinking off\nafter", " trailing\n"):
            for prompt in ("Hello", "", "  中文\nOlá 👋  "):
                with self.subTest(system=system, prompt=prompt):
                    messages = [] if system is None else [{"role": "system", "content": system}]
                    messages.append({"role": "user", "content": prompt})
                    expected = template.render(messages=messages, add_generation_prompt=True)
                    actual = ns["BailingMoeV3Model"].default_chat_prompt(None, prompt, system)
                    self.assertEqual(actual.encode("utf-8"), expected.encode("utf-8"))
                    self.assertTrue(actual.endswith("<role>ASSISTANT</role>\n<think>"))


class CorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _namespace()

    def module(self, tensors=None, required=True, **kwargs):
        source = _TensorSource(tensors or {})
        module = self.ns["BlockSparseMLP"](
            config=SimpleNamespace(stc=source), key=KEY, hidden_size=8,
            intermediate_size=16, num_experts=4, num_experts_per_tok=2,
            key_up="experts.{expert_idx}.up_proj", key_gate="experts.{expert_idx}.gate_proj",
            key_down="experts.{expert_idx}.down_proj", key_routing_gate="gate",
            key_e_score_bias="gate.expert_bias", **({"require_e_score_bias": True} if required else {}), **kwargs)
        module.load_local = _forbidden
        module.load_routing = _forbidden
        return module, source

    def test_missing_required_before_child_or_offload_load(self):
        m, source = self.module()
        with self.assertRaisesRegex(ValueError, r"gate\.expert_bias"):
            m.load("cpu")
        self.assertFalse(set(source.events) & {"children", "offload", "split"})
        self.assertIsNone(m.e_score_correction_bias)

    def test_malformed_primary_never_hidden_by_fallback(self):
        for tensor in (torch.zeros(3), torch.zeros(5), torch.zeros(1, 4), torch.zeros(4, 1),
                       torch.tensor(0.), torch.zeros(4, dtype=torch.int64),
                       torch.full((4,), float("nan")), torch.full((4,), float("inf"))):
            with self.subTest(shape=tuple(tensor.shape), dtype=tensor.dtype):
                m, source = self.module({PRIMARY: tensor, FALLBACK: torch.zeros(4)})
                with self.assertRaisesRegex(ValueError, r"gate\.expert_bias"):
                    m.load("cpu")
                self.assertNotIn(FALLBACK, source.events)
                self.assertFalse(set(source.events) & {"children", "offload", "split"})

    def test_fallback_only_validated_and_serialized_as_primary(self):
        m, source = self.module({FALLBACK: torch.tensor([-2., 0., 1., 3.])})
        m.load("cpu")
        self.assertEqual([r[0] for r in source.requests], [PRIMARY, FALLBACK])
        torch.testing.assert_close(m.get_tensors()[PRIMARY], source.tensors[FALLBACK])
        self.assertNotIn(FALLBACK, m.get_tensors())
        for bad in (torch.zeros(3), torch.zeros(1, 4), torch.full((4,), float("nan"))):
            m, source = self.module({FALLBACK: bad})
            with self.assertRaisesRegex(ValueError, r"gate\.e_score_correction_bias"):
                m.load("cpu")
            self.assertNotIn("children", source.events)

    def test_primary_wins_when_both_exist(self):
        m, source = self.module({PRIMARY: torch.zeros(4), FALLBACK: torch.ones(4)})
        m.load("cpu")
        self.assertEqual([r[0] for r in source.requests], [PRIMARY])
        torch.testing.assert_close(m.e_score_correction_bias, torch.zeros(4))

    def test_valid_buffers_no_defer_and_reload(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            for values in ([0., 0., 0., 0.], [-2., -0.5, 0.25, 3.]):
                with self.subTest(dtype=dtype, values=values):
                    m, source = self.module({PRIMARY: torch.tensor(values, dtype=dtype)})
                    m.load("cpu")
                    self.assertTrue(all(r[2:] == (True, True) for r in source.requests))
                    self.assertEqual(source.pending, [])
                    if dtype in (torch.float16, torch.float32):
                        self.assertIs(m.e_score_correction_bias, source.destinations[PRIMARY])
                    source.finish()
                    torch.testing.assert_close(m.e_score_correction_bias.float(), torch.tensor(values))
                    old = m.e_score_correction_bias
                    m.unload()
                    self.assertIsNone(m.e_score_correction_bias)
                    source.tensors[PRIMARY] = torch.ones(4, dtype=dtype)
                    m.load("cpu")
                    self.assertIsNot(m.e_score_correction_bias, old)
                    torch.testing.assert_close(m.e_score_correction_bias.float(), torch.ones(4))

    def test_optional_default_preserves_missing_and_legacy_shapes(self):
        for tensors in ({}, {PRIMARY: torch.zeros(1, 4)}, {FALLBACK: torch.ones(4)}):
            with self.subTest(tensors=tensors):
                m, source = self.module(tensors, required=False)
                self.assertFalse(m.require_e_score_bias)
                m.load("cpu")
                self.assertEqual(source.events[:3], ["offload", "split", "children"])
                self.assertTrue(all(r[1:] == (True, True, True) for r in source.requests))
                if not tensors:
                    self.assertIsNone(m.e_score_correction_bias)

    def test_required_without_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "key_e_score_bias"):
            self.ns["BlockSparseMLP"](None, KEY, 8, 16, 4, 2, key_e_score_bias=None,
                                      require_e_score_bias=True)

    def test_bias_free_default_does_not_probe(self):
        source = _TensorSource({})
        m = self.ns["BlockSparseMLP"](
            SimpleNamespace(stc=source), KEY, 8, 16, 4, 2, key_e_score_bias=None,
            key_up="up_proj", key_down="down_proj")
        m.load("cpu")
        self.assertFalse(m.require_e_score_bias)
        self.assertEqual(source.requests, [])
        self.assertIsNone(m.e_score_correction_bias)

    def test_required_checked_before_offload_claim(self):
        for tensor in (None, torch.zeros(3), torch.zeros(4)):
            with self.subTest(tensor=tensor):
                m, source = self.module({} if tensor is None else {PRIMARY: tensor})
                with patch.object(m, "cpu_maybe_offload_load", return_value=True) as claim:
                    if tensor is None or tensor.shape != (4,):
                        with self.assertRaises(ValueError):
                            m.load("cpu")
                        claim.assert_not_called()
                    else:
                        m.load("cpu")
                        claim.assert_called_once()
                        torch.testing.assert_close(m.e_score_correction_bias, tensor)
                self.assertNotIn("children", source.events)

    def test_tp_required_malformed_fails_before_local_or_routing(self):
        cls = self.ns["BlockSparseMLP"]
        for bad in (None, torch.zeros(3), torch.zeros(1, 4), torch.full((4,), float("nan"))):
            with self.subTest(bad=bad):
                m, _ = self.module({PRIMARY: torch.zeros(4)})
                m.load("cpu")
                exported = m.tp_export({}, _Wire())
                exported["e_score_correction_bias"] = bad
                context = {"device": "cpu", "output_device": "cpu", "consumer": _Wire()}
                with patch.object(cls, "load_local", _forbidden), patch.object(cls, "load_routing", _forbidden):
                    with self.assertRaisesRegex(ValueError, r"gate\.expert_bias"):
                        cls.tp_import(context, exported, {KEY: (0, 4, "experts")}, skip_reduction=True)

    def test_legacy_tp_metadata_keeps_optional_default(self):
        m, _ = self.module(required=False)
        m.load("cpu")
        exported = m.tp_export({}, _Wire())
        exported["kwargs"].pop("key_e_score_bias")
        exported["kwargs"].pop("require_e_score_bias")
        cls = self.ns["BlockSparseMLP"]
        context = {"device": "cpu", "output_device": "cpu", "consumer": _Wire()}
        with patch.object(cls, "load_local", lambda self: None), patch.object(cls, "load_routing", lambda self: None):
            restored = cls.tp_import(context, exported, {KEY: (0, 4, "experts")}, skip_reduction=True)
        self.assertFalse(restored.require_e_score_bias)
        self.assertIsNone(restored.e_score_correction_bias)
        self.assertEqual(restored.e_score_correction_bias_key, "gate.e_score_correction_bias")

    def test_tp_metadata_and_recreation(self):
        for required in (False, True):
            for unit, end in (("experts", 4), ("channels", 16)):
                with self.subTest(required=required, unit=unit):
                    m, _ = self.module({PRIMARY: torch.zeros(4)}, required=required)
                    m.load("cpu")
                    exported = m.tp_export({}, _Wire())
                    self.assertEqual(exported["kwargs"]["require_e_score_bias"], required)
                    self.assertEqual(exported["kwargs"]["key_e_score_bias"], "gate.expert_bias")
                    cls = self.ns["BlockSparseMLP"]
                    context = {"device": "cpu", "output_device": "cpu", "consumer": _Wire()}
                    with patch.object(cls, "load_local", lambda self: None), \
                         patch.object(cls, "load_routing", lambda self: None):
                        restored = cls.tp_import(context, exported, {KEY: (0, end, unit)}, skip_reduction=True)
                    self.assertEqual(restored.require_e_score_bias, required)
                    self.assertEqual(restored.e_score_correction_bias_key, m.e_score_correction_bias_key)
                    self.assertIs(restored.e_score_correction_bias, m.e_score_correction_bias)


class IsolationTests(unittest.TestCase):
    def test_no_engine_import_or_cuda_initialization(self):
        self.assertFalse(any(n == "exllamav3" or n.startswith("exllamav3.") for n in sys.modules))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
