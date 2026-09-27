"""CPU-only Ling MTP output-domain and compile-ownership regressions.

Run with the existing Torch/safetensors environment, CUDA_VISIBLE_DEVICES='',
python -B tests/test_bailing_mtp_contracts.py -v. On instrumented hosts use -S
and explicitly add that environment's site-packages before running this file.

Production definitions are AST-loaded to avoid importing the CUDA/JIT package.
Constructors, prefix collection, compile_model, safetensors IO, and the ordinary
sampler/ MTP generator loop execute on CPU. Native handles/transport and metadata
model discovery are explicit adapters; this is NOT GPU, full-model, or numerical
EXL3 parity. No quantization runs here.

The optional real-EXL3 tests require EXL3_TEST_MTP_EXL3_SOURCE (a local compiled
pack directory). They copy a bounded 256x128 tile/scale slice from its first
suitable EXL3 linear, preserving codebook metadata, then compile/reload under
MTP keys. This is actual previously quantized storage, not native weights with
an EXL3 label, and not a quantization of Ling weights. Without a source these
tests skip rather than substituting fabricated quantized data.
"""
from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
import gc
import glob
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import weakref

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
KEY = "model.layers.42"


def extract(path, names, ns, methods=None):
    """Execute unmodified local definitions, not their package-level imports."""
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    if methods is not None:
        assert len(nodes) == 1 and isinstance(nodes[0], ast.ClassDef)
        nodes = [n for n in nodes[0].body if isinstance(n, ast.FunctionDef) and n.name in methods]
        assert {n.name for n in nodes} == set(methods)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(ROOT / path), "exec"), ns)


class CPUHandles:
    """Handles only: no EXL3 arithmetic or quantization is emulated."""
    @staticmethod
    def BC_LinearEXL3(*args):
        return SimpleNamespace()

    def __getattr__(self, name):
        # Constructors retain kernel references without executing them.
        def unsupported(*args, **kwargs):
            raise AssertionError(f"Unexpected native call in CPU contracts: {name}")
        return unsupported


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.interval = time.perf_counter() - self.start


class CPUOffload:
    def _cpu_init_state(self):
        pass


def namespace():
    ns = dict(globals(), override=lambda f: f, nn=torch.nn, F=torch.nn.functional,
              ext=CPUHandles(), Model_TPMixin=type("TPUnused", (), {}),
              Model_LSMixin=type("LSUnused", (), {}), BlockSparseMLP_CPU=CPUOffload,
              MAX_MLP_INTERMEDIATE=55296, TEMP_ROWS_GRAPH=32, TEMP_ROWS_FUSED=128,
              MAX_HEADER_SIZE=100 * 1024**2, MAX_DEFERRED_LOAD_CHUNK=4 * 1024**2,
              routing_ds3_fp32=lambda *a: (_ for _ in ()).throw(AssertionError("No routing in compile tests")),
              to_device=lambda x, d: x.to(d),
              get_for_device=lambda p, k, d: p[k].to(d),
              to2=lambda x, *ds: x.to(next((d for d in ds if d is not None), x.dtype)),
              free_mem=gc.collect,
              g_tensor_cache=SimpleNamespace(get=lambda device, shape, dtype: torch.empty(shape, dtype=dtype)),
              Sampler=object)
    for path, names in (
        ("modules/module.py", ["Module"]),
        ("modules/linear.py", ["Linear"]),
        ("modules/rmsnorm.py", ["RMSNorm"]),
        ("modules/embedding.py", ["Embedding"]),
        ("modules/mlp.py", ["MLP", "GatedMLP"]),
        ("modules/block_sparse_mlp.py", ["BlockSparseMLP"]),
        ("modules/mla_attn.py", ["MLAttention"]),
        ("modules/transformer.py", ["TransformerBlock"]),
        ("modules/arch_specific/qwen3_5_mtp.py", ["Qwen3_5MTPInputLayer"]),
        ("model/model.py", ["Model"]),
        ("architecture/bailing_moe_v3.py", ["bailing_mla", "bailing_mlp"]),
        ("architecture/bailing_moe_v3_mtp.py", ["BailingMoeV3MTPBlock", "BailingMoeV3MTPModel"]),
        ("loader/safetensors.py", ["convert_dtype", "validate_header", "read_header", "STCMetrics", "SafetensorsCollection"]),
        ("modules/quant/exl3_lib/quantize.py", ["frac_k"]),
        ("modules/quant/exl3.py", ["LinearEXL3"]),
        ("conversion/quant_config.py", ["update_config", "create_quantization_config_json"]),
        ("conversion/compile.py", ["tsize", "dsize", "compile_model"]),
        ("generator/sampler/custom.py", ["SS", "SamplingState", "SS_Base", "SS_Argmax"]),
    ):
        extract("exllamav3/" + path, names, ns)
    # This IO module has no engine or extension imports; execute it unchanged.
    io_ns = dict(__name__=__name__)
    exec(compile((ROOT / "exllamav3/loader/safetensors_alt.py").read_text(),
                 str(ROOT / "exllamav3/loader/safetensors_alt.py"), "exec"), io_ns)
    ns["save_file"] = io_ns["save_file"]
    ns["__version__"] = "cpu-contract-fixture"
    extract("exllamav3/generator/generator.py", ["Generator"], ns,
            methods=["iterate_draftmodel_mtp_gen"])
    extract("exllamav3/generator/sampler/custom.py", ["CustomSampler"], ns, methods=["forward"])
    ns["ordinary_sample"] = ns.pop("forward")
    return ns


def config(**kwargs):
    values = dict(num_mtp_layers=1, num_hidden_layers=42, hidden_size=128, rms_norm_eps=1e-6,
                  vocab_size=157184, num_q_heads=2, kv_lora_rank=128, q_lora_rank=None,
                  qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=64,
                  rope_settings=None, sm_scale=128 ** -0.5, first_k_dense_replace=2,
                  expert_swiglu_limits=[4.0] * 42, shared_swiglu_limits=[5.0] * 42,
                  intermediate_size=128, moe_shared_expert_intermediate_size=128,
                  num_shared_experts=1, moe_intermediate_size=128, num_experts=2,
                  num_experts_per_tok=1, n_group=1, topk_group=1, routed_scaling_factor=2.5)
    values.update(kwargs)
    return SimpleNamespace(**values)


def make_target(ns, cfg, head=None):
    target = ns["Model"](cfg)
    target.modules = [ns["Embedding"](cfg, "model.word_embeddings", cfg.vocab_size, cfg.hidden_size),
                      ns["RMSNorm"](cfg, "model.norm", cfg.rms_norm_eps),
                      head or ns["Linear"](cfg, "lm_head", cfg.hidden_size, cfg.vocab_size)]
    target.logit_layer_idx = 2
    target.loaded_tp = False
    return target


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = namespace()

    def head_fixture(self, shape, domain=157153, physical=157184, padded=157184):
        ns = self.ns
        class Head(ns["Linear"]):
            def __init__(self, cfg):
                super().__init__(cfg, "lm_head", cfg.hidden_size, physical)
                self.logits = torch.full((*shape, padded), -100.0)
                self.calls = 0
            def prepare_for_device(self, state, params):
                return state
            def forward(self, state, params):
                self.calls += 1
                return self.logits
        cfg = config(vocab_size=physical)
        head = Head(cfg)
        target = make_target(ns, cfg, head)
        mtp = ns["BailingMoeV3MTPModel"](cfg)
        mtp.attach_to(target)
        return mtp, target, head, SimpleNamespace(actual_vocab_size=domain)

    def normal_ids(self, logits, tokenizer):
        sampler = SimpleNamespace(reqs_torch_seed=False, fused_only=False,
                                  steps=[self.ns["SS_Argmax"]()])
        return self.ns["ordinary_sample"](sampler, logits.clone(), rand_u32=0, tokenizer=tokenizer)

    def test_all_31_unmapped_and_every_padded_row(self):
        # Physical checkpoint width plus a deliberately nonzero extra padding block.
        mtp, target, head, tok = self.head_fixture((1, 1), padded=157184 + 128)
        state = torch.zeros(1, 1, 128)
        original_shapes = (target.modules[0].vocab_size, head.out_features_unpadded, head.out_features)
        head.logits[..., 7] = -1.25  # negative valid max catches unmasked zero padding too
        for winner in range(tok.actual_vocab_size, head.logits.shape[-1]):
            for export in (False, True):
                with self.subTest(winner=winner, export=export):
                    head.logits[..., winner] = 100.0
                    params = {"output_vocab_size": tok.actual_vocab_size, "export_draft_conf": export}
                    ids = mtp.sample_from_state(state, params)
                    self.assertEqual(ids.tolist(), [[7]])
                    self.assertEqual(ids.dtype, torch.long)
                    self.assertEqual(ids.shape, state.shape[:-1])
                    self.assertTrue(torch.equal(ids, self.normal_ids(head.logits, tok)))
                    if export:
                        self.assertEqual(params["draft_conf"].tolist(), [[-1.25]])
                        self.assertEqual(params["draft_conf"].shape, ids.shape)
                    else:
                        self.assertNotIn("draft_conf", params)
                    self.assertEqual(head.logits[..., winner].item(), 100.0)
                    head.logits[..., winner] = -100.0
        self.assertEqual(original_shapes, (target.modules[0].vocab_size,
                                          head.out_features_unpadded, head.out_features))
        self.assertIs(mtp.target_embed(), target.modules[0])
        self.assertIs(mtp.target_lm_head(), head)

    def test_domain_is_dynamic_batched_and_not_physical_resize(self):
        mtp, target, head, tok = self.head_fixture((2, 3), domain=11, physical=42, padded=128)
        expected = torch.tensor([[0, 10, 3], [9, 1, 8]])
        expected_conf = torch.arange(6, dtype=torch.float).reshape(2, 3) - 3
        head.logits.scatter_(-1, expected.unsqueeze(-1), expected_conf.unsqueeze(-1))
        head.logits[..., 11:] = 100
        params = {"output_vocab_size": 11, "export_draft_conf": True}
        ids = mtp.sample_from_state(torch.zeros(2, 3, 128), params)
        self.assertTrue(torch.equal(ids, expected))
        self.assertTrue(torch.equal(params["draft_conf"], expected_conf))
        self.assertTrue(torch.equal(ids, self.normal_ids(head.logits, tok)))
        # The last physical row becomes admissible for a different tokenizer domain.
        head.logits[..., 41] = 101
        self.assertTrue((mtp.sample_from_state(torch.zeros(2, 3, 128), {"output_vocab_size": 42}) == 41).all())
        self.assertEqual(target.modules[0].vocab_size, 42)
        self.assertEqual((head.out_features_unpadded, head.out_features), (42, 128))

    def test_uninitialized_and_invalid_domains_fail_before_head(self):
        mtp, target, head, _ = self.head_fixture((1, 1), domain=11, physical=42, padded=128)
        for params in ({}, *({"output_vocab_size": v} for v in (None, 0, -1, True, 1.5, "11", 43))):
            with self.subTest(params=params), self.assertRaisesRegex(ValueError, "output_vocab_size"):
                mtp.sample_from_state(torch.zeros(1, 1, 128), params)
        self.assertEqual(head.calls, 0)
        head.logits = torch.zeros(1, 1, 10)
        with self.assertRaisesRegex(ValueError, "head is smaller"):
            mtp.sample_from_state(torch.zeros(1, 1, 128), {"output_vocab_size": 11})
        mtp.target_lm_head = None
        with self.assertRaisesRegex(RuntimeError, "Attach"):
            mtp.sample_from_state(torch.zeros(1, 1, 128), {"output_vocab_size": 11})

    def test_attach_postnorm_and_embedding_first_contracts(self):
        ns = self.ns
        cfg = config(vocab_size=42)
        mtp = ns["BailingMoeV3MTPModel"](cfg)
        target = make_target(ns, cfg)
        with self.assertRaisesRegex(ValueError, "same Config"):
            mtp.attach_to(make_target(ns, config()))
        mtp.attach_to(target)
        self.assertEqual(mtp.draft_verifier_params, {"export_state_norm_keys": {"model.norm"}})
        self.assertEqual((mtp.modules[1].layer_idx, mtp.modules[1].attn.layer_idx), (0, 0))
        self.assertEqual(mtp.modules[1].key, KEY)
        self.assertEqual(mtp.modules[1].mlp.act_limit, 0)
        self.assertEqual(mtp.modules[1].mlp.shared_experts.act_limit, 0)
        self.assertEqual(mtp.final_norm.out_dtype, torch.half)
        layer = mtp.input_layer
        self.assertEqual(layer.fc.qbits_key, "mtp_bits")
        self.assertEqual(layer.pre_fc_norm_hidden.constant_bias, 0)
        self.assertEqual(layer.pre_fc_norm_embedding.constant_bias, 0)
        # Execute the actual input forward with asymmetric branch adapters.
        layer.device = torch.device("cpu")
        target.modules[0].forward = lambda ids, params, out_dtype: torch.full((*ids.shape, 128), 3.0)
        layer.pre_fc_norm_hidden.forward = lambda x, params: x * 5
        layer.pre_fc_norm_embedding.forward = lambda x, params: x * 7
        layer.fc.forward = lambda x, params: x
        merged = layer.forward(torch.ones(2, 3, dtype=torch.long), {"target_hidden": torch.full((2, 3, 128), 2.0)})
        self.assertTrue((merged[..., :128] == 21).all())
        self.assertTrue((merged[..., 128:] == 10).all())

    def test_generator_passes_tokenizer_domain_each_depth_and_confidence(self):
        ns = self.ns
        for calibrated in (False, True):
            with self.subTest(calibrated=calibrated):
                mtp, target, head, tok = self.head_fixture((2, 1), domain=11, physical=42, padded=128)
                head.logits[0, 0, 7], head.logits[1, 0, 10] = 9, 8
                head.logits[..., 11:] = 100
                seen = []
                def forward(ids, params):
                    seen.append(dict(params))
                    return params["target_hidden"] + 1
                mtp.forward = forward
                jobs = [SimpleNamespace(is_prefill_done=lambda: True, get_max_seq_len=lambda: 3,
                                        sequences=[SimpleNamespace(block_index_tensor=torch.tensor([[4]]), kv_position=3)],
                                        mtp_last_hidden=torch.zeros(1, 1, 128), time_first_token=1,
                                        get_input_ids_list=lambda: [torch.tensor([[2]])]) for _ in range(2)]
                gen = SimpleNamespace(active_jobs=jobs, num_draft_tokens=3, draft_model=mtp, model=target,
                                      tokenizer=tok, draft_cache=object(),
                                      draft_input_ids_pinned=torch.zeros(2, 1, dtype=torch.long),
                                      draft_ids_pinned=torch.zeros(2, 3, dtype=torch.long),
                                      _staging=lambda name, *shape: torch.zeros(shape, dtype=torch.int32),
                                      draft_calibrator=SimpleNamespace(estimate=lambda v: 1.0, confidence=0.5) if calibrated else None)
                ns["PAGE_SIZE"] = 256
                ids = ns["iterate_draftmodel_mtp_gen"](gen, [])
                self.assertEqual(ids.tolist(), [[7, 7, 7], [10, 10, 10]])
                self.assertEqual([p["output_vocab_size"] for p in seen], [11] * 3)
                self.assertEqual([p["draft_step"] for p in seen], [0, 1, 2])
                self.assertEqual(head.calls, 3)
                if calibrated:
                    self.assertEqual(gen._draft_conf_round["conf"].tolist(), [[9] * 3, [8] * 3])
                else:
                    self.assertIsNone(gen._draft_conf_round)

    def test_other_mtp_sampler_ignores_new_optional_parameter(self):
        # The shared generator adds information, not a new global masking policy.
        ns = dict(self.ns)
        extract("exllamav3/architecture/qwen3_5_mtp.py", ["Qwen3_5MTPModel"], ns,
                methods=["sample_from_state"])
        mtp, target, head, _ = self.head_fixture((2, 1), domain=11, physical=42, padded=128)
        head.logits[..., 41] = 10
        head.logits[..., 100] = 20
        for export in (False, True):
            before = {"export_draft_conf": export}
            after = dict(before, output_vocab_size=11)
            state = torch.zeros(2, 1, 128)
            expected = ns["sample_from_state"](mtp, state, before)
            actual = ns["sample_from_state"](mtp, state, after)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue((actual == (41 if export else 100)).all())
            if export:
                self.assertTrue(torch.equal(before["draft_conf"], after["draft_conf"]))


class CompileContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = namespace()

    def fixture(self, key_prefix="model"):
        ns = self.ns
        cfg = config(vocab_size=128)
        target = make_target(ns, cfg)
        mtp = ns["BailingMoeV3MTPModel"](cfg, key_prefix=key_prefix)
        cfg.model_classes = {"text": SimpleNamespace(get_additional_compiled_tensors=lambda c: {})}
        cfg.get_tensor_name_fixes = lambda: {}
        tensors = {}
        # Real production leaf constructors, with small native CPU weights.
        for model in (target, mtp):
            for m in model:
                if isinstance(m, ns["Linear"]):
                    tensors[m.key + ".weight"] = torch.arange(m.out_features_unpadded * m.in_features_unpadded,
                                                               dtype=torch.float).reshape(m.out_features_unpadded, m.in_features_unpadded).remainder(17).half()
                elif isinstance(m, ns["RMSNorm"]):
                    width = cfg.kv_lora_rank if m.key.endswith("kv_a_layernorm") else cfg.hidden_size
                    tensors[m.tensor_key] = torch.ones(width, dtype=torch.bfloat16)
                elif isinstance(m, ns["Embedding"]):
                    tensors[m.key + ".weight"] = torch.zeros(cfg.vocab_size, cfg.hidden_size, dtype=torch.half)
        key = mtp.modules[1].key
        # MLA's native absorbed KV-B matrix is not a Linear child.
        tensors[key + ".attention.kv_b_proj.weight"] = torch.ones(cfg.num_q_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank, dtype=torch.bfloat16)
        tensors[key + ".mlp.gate.expert_bias"] = torch.tensor([0.125, -0.25], dtype=torch.float)
        return cfg, target, mtp, tensors

    def source_slice(self):
        directory = os.environ.get("EXL3_TEST_MTP_EXL3_SOURCE")
        if not directory:
            self.skipTest("Set EXL3_TEST_MTP_EXL3_SOURCE for real previously quantized EXL3 storage")
        metadata = json.loads((Path(directory) / "config.json").read_text())
        self.assertEqual(metadata["quantization_config"]["quant_method"], "exl3")
        # Read headers and bounded slices, never entire source weights or model code.
        for path in sorted(Path(directory).glob("*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as f:
                keys = set(f.keys())
                for name in sorted(k for k in keys if k.endswith(".trellis")):
                    prefix = name[:-len(".trellis")]
                    if not all(prefix + "." + k in keys for k in ("suh", "svh")):
                        continue
                    shape = f.get_slice(name).get_shape()
                    if shape[0] < 16 or shape[1] < 8:
                        continue
                    # Quantizer storage is [input/16, output/16, 16*bpw].
                    result = {"trellis": f.get_slice(name)[:16, :8, :].clone().contiguous(),
                              "suh": f.get_slice(prefix + ".suh")[:256].clone(),
                              "svh": f.get_slice(prefix + ".svh")[:128].clone()}
                    for suffix in ("mul1", "mcg"):
                        if prefix + "." + suffix in keys:
                            result[suffix] = f.get_tensor(prefix + "." + suffix).clone()
                    self.assertEqual(tuple(result["suh"].shape), (256,))
                    self.assertEqual(tuple(result["svh"].shape), (128,))
                    self.assertEqual(result["trellis"].dtype, torch.int16)
                    self.assertGreater(result["trellis"].count_nonzero().item(), 0)
                    digest = hashlib.sha256(result["trellis"].numpy().tobytes()).hexdigest()
                    print(f"REAL_EXL3_SLICE {path}:{prefix} shape={tuple(result['trellis'].shape)} sha256={digest}")
                    return result
        self.fail("Source has no suitable EXL3 trellis/suh/svh group; no synthetic fallback")

    def check_ownership(self, modules, stc, tensors):
        owners = Counter()
        sizes = []
        for m in modules:
            emitted = m.get_compile_tensors(stc)
            declared = m.get_compile_sizes(stc)
            self.assertEqual(sorted(declared), sorted(t.numel() * t.element_size() for t in emitted.values()))
            owners.update(emitted.keys())
            sizes.extend(declared)
        self.assertEqual(owners, Counter({k: 1 for k in tensors}))
        self.assertEqual(sum(sizes), sum(t.numel() * t.element_size() for t in tensors.values()))

    def compile_and_reload(self, quantized=False, multi=False, boundary_delta=0):
        ns = self.ns
        cfg, target, mtp, tensors = self.fixture()
        source = self.source_slice() if quantized else None
        if source is not None:
            del tensors[KEY + ".eh_proj.weight"]
            tensors.update({KEY + ".eh_proj." + k: v for k, v in source.items()})
            # Also exercise packed tensors inside the block's actual attention prefix.
            # Crop a square 128 -> 128 for dense (same block owner).
            del tensors[KEY + ".attention.dense.weight"]
            tensors.update({KEY + ".attention.dense." + k:
                            (v[:8, :8].contiguous() if k == "trellis" else v[:128].clone() if k == "suh" else v.clone())
                            for k, v in source.items()})
        with tempfile.TemporaryDirectory(prefix="ling-mtp-contract-", dir=os.environ.get("TMPDIR")) as tmp:
            root = Path(tmp)
            input_dir, work, output = root / "input", root / "work", root / "output"
            input_dir.mkdir()
            (work / "qtensors").mkdir(parents=True)
            (input_dir / "config.json").write_text(json.dumps({"architectures": ["CPUContractFixture"]}))
            ns["save_file"](tensors, str(work / "qtensors" / "fixture.safetensors"))
            STC = ns["SafetensorsCollection"]
            cfg.stc = STC(str(work / "qtensors"), load_method="python")
            self.check_ownership(target.modules + mtp.modules, cfg.stc, tensors)
            # Negative control: the previous broad block owner duplicates all input/final siblings.
            broad = ns["TransformerBlock"](cfg, KEY)
            old_counts = Counter()
            for m in [mtp.input_layer, broad, mtp.final_norm]:
                old_counts.update(m.get_compile_tensors(cfg.stc).keys())
            for suffix in ("hnorm.weight", "enorm.weight", "final_layernorm.weight"):
                self.assertEqual(old_counts[KEY + "." + suffix], 2)
            self.assertEqual(old_counts[KEY + (".eh_proj.trellis" if source else ".eh_proj.weight")], 2)
            first_group_bytes = sum(sum(m.get_compile_sizes(cfg.stc)) for m in target.modules + [mtp.input_layer])
            shard_bytes = first_group_bytes + boundary_delta if multi else sum(t.numel() * t.element_size() for t in tensors.values())
            # Use production Python IO; intercept discovery only to select CPU transport
            # and construct this tiny fixture instead of importing the model registry.
            def collection(directory, **kwargs):
                return STC(directory, load_method="python", **kwargs)
            def read_config(directory):
                return SimpleNamespace(stc=collection(directory))
            def read_model(config):
                # Quantization metadata's production API describes the primary model.
                return target
            saved = {name: ns.get(name) for name in ("SafetensorsCollection", "Config", "Model")}
            ns.update(SafetensorsCollection=collection, Config=SimpleNamespace(from_directory=read_config),
                      Model=SimpleNamespace(from_config=read_model))
            args = dict(in_dir=str(input_dir), work_dir=str(work), out_dir=str(output),
                        shard_size=shard_bytes / 1024**2, final_bits=16,
                        head_bits=16, mtp_bits=source["trellis"].shape[-1] / 16 if source else 16)
            log = io.StringIO()
            try:
                with redirect_stdout(log):
                    ns["compile_model"](args, target, cfg, None, mtp_model=mtp)
                    reload_stc = collection(str(output))
            finally:
                ns.update(saved)
            self.assertNotIn("Overriding", log.getvalue())
            self.assertNotIn("Replaced", log.getvalue())
            self.assertNotIn("dropped", log.getvalue())
            physical_map, owners, payload_bytes = {}, Counter(), 0
            shards = sorted(output.glob("*.safetensors"))
            if multi:
                # One byte below the target+input boundary splits off the input.
                # The indivisible block and the final norm each require a shard.
                self.assertEqual(len(shards), 4 if boundary_delta < 0 else 3)
            else:
                self.assertEqual(len(shards), 1)
            for path in shards:
                header = ns["read_header"](str(path), None)
                for name, meta in header.items():
                    if name in ("_header_offset", "__metadata__"):
                        continue
                    owners[name] += 1
                    physical_map[name] = path.name
                    payload_bytes += meta["data_offsets"][1] - meta["data_offsets"][0]
            self.assertEqual(owners, Counter({k: 1 for k in tensors}))
            self.assertEqual(payload_bytes, sum(t.numel() * t.element_size() for t in tensors.values()))
            if multi:
                index = json.loads((output / "model.safetensors.index.json").read_text())
                self.assertEqual(index["weight_map"], physical_map)
                self.assertEqual(index["metadata"]["total_size"], payload_bytes)
                input_key = KEY + (".eh_proj.trellis" if source else ".eh_proj.weight")
                self.assertNotEqual(physical_map[input_key], physical_map[KEY + ".input_layernorm.weight"])
                self.assertNotEqual(physical_map[input_key], physical_map[KEY + ".final_layernorm.weight"])
            for name, expected in tensors.items():
                actual = reload_stc.get_tensor(name, allow_bf16=True)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.shape, expected.shape)
                self.assertTrue(torch.equal(actual, expected), name)
            metadata = json.loads((output / "quantization_config.json").read_text())
            self.assertEqual(metadata["mtp_bits"], args["mtp_bits"])
            self.assertEqual(metadata["head_bits"], 16)
            if source is not None:
                # Production format detection + data loading + LinearEXL3 constructor.
                # Only BC native handles are adapted, not the stored tensors or loader.
                cfg.stc = reload_stc
                for key, k, n in ((KEY + ".eh_proj", 256, 128), (KEY + ".attention.dense", 128, 128)):
                    linear = ns["Linear"](cfg, key, k, n, out_dtype=torch.float)
                    linear.load(torch.device("cpu"))
                    self.assertEqual(linear.quant_type, "exl3")
                    self.assertIsInstance(linear.inner, ns["LinearEXL3"])
                    for name, actual in linear.inner.get_tensors(key).items():
                        self.assertTrue(torch.equal(actual, tensors[name]), name)
            self.check_ownership(target.modules + mtp.modules, reload_stc, tensors)
            cfg.stc.close()
            reload_stc.close()
            index_status = "PASS" if multi else "not-written(single-shard)"
            print(f"COMPILE_{'EXL3_SLICE' if quantized else 'NATIVE'} shards={len(shards)} keys={len(owners)} bytes={payload_bytes} boundary_delta={boundary_delta} unique/reload=PASS index={index_status}")

    def test_native_single_shard(self):
        self.compile_and_reload()

    def test_native_small_shard_boundaries(self):
        for delta in (-1, 0, 1):
            with self.subTest(delta=delta):
                self.compile_and_reload(multi=True, boundary_delta=delta)

    def test_real_exl3_single_shard(self):
        self.compile_and_reload(quantized=True)

    def test_real_exl3_small_shard_boundaries(self):
        if not os.environ.get("EXL3_TEST_MTP_EXL3_SOURCE"):
            self.skipTest("Set EXL3_TEST_MTP_EXL3_SOURCE for real previously quantized EXL3 storage")
        for delta in (-1, 0, 1):
            with self.subTest(delta=delta):
                self.compile_and_reload(quantized=True, multi=True, boundary_delta=delta)

    def test_child_prefixes_not_hardcoded(self):
        cfg, target, mtp, tensors = self.fixture(key_prefix="renamed")
        with tempfile.TemporaryDirectory(prefix="ling-mtp-prefix-", dir=os.environ.get("TMPDIR")) as tmp:
            self.ns["save_file"](tensors, str(Path(tmp) / "fixture.safetensors"))
            stc = self.ns["SafetensorsCollection"](tmp, load_method="python")
            self.check_ownership(target.modules + mtp.modules, stc, tensors)
            self.assertIs(self.ns["TransformerBlock"].get_compile_tensors, self.ns["Module"].get_compile_tensors)
            stc.close()


def tearDownModule():
    assert not torch.cuda.is_initialized(), "CPU tests initialized CUDA"
    assert not any(n == "exllamav3" or n.startswith("exllamav3.") for n in sys.modules), "CPU tests imported engine package"


if __name__ == "__main__":
    unittest.main()
