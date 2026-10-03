"""
CPU tests for exllamav3/training/realtime.py -- the real-time
(inference-time) training coordinator:

  * RWLock semantics: concurrent readers, writer exclusion, writer preference
    (a waiting ingest blocks NEW inference readers while in-flight ones
    drain);
  * sample encoding: prompt/response exact mask boundary, text supervision,
    pre-tokenized passthrough + validation, messages via a render_segments
    callable, BOS normalization (auto-prepending tokenizer), seq_len
    truncation, eot_text;
  * the ingest loop: in-order consumption, batch/grad-accum windowing, step
    counting across ingests, token-weighted mean loss, adapter sync + update
    callbacks (cache invalidation) firing once per ingest, generator drain;
  * preference samples: TRL explicit-prompt DPO pairs and KTO rows (string
    and conversational prompts via render_prompt, pre-tokenized forms, label
    coercion, completion-tail fit / prompt-overflow skip), the DPO batch
    layout (chosen block then rejected block, policy + adapters-disabled
    reference forward), the ln-2 step-0 anchor and reward direction, KTO's
    mismatched-pair KL rows (batch >= 2 only) and row weights, and mixed
    ingests where a kind change closes the accumulation window;
  * checkpoint policy: timestamped names, cadence, pruning to
    keep_checkpoints, optimizer-state save/resume via load();
  * the externally settable constant lr;
  * aux component parking (aux_offload.ModelParker + attach_aux_models):
    device-resident modules unloaded for the ingest and reloaded to their
    recorded devices afterwards (deferred-load bracketing, stc handle close),
    CPU/unloaded modules left alone, restore guaranteed on ingest failure,
    the offload_aux_when_training switch, and ordering against the idle
    offload (park before restore, unpark after re-park).

The net/tokenizer are stubs implementing only the narrow surface the
coordinator uses. No GPU / compiled extension / real model needed. Run:
    python tests/test_realtime.py
"""

from __future__ import annotations
import contextlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import threading
import time
import types
from unittest.mock import patch

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


_load("fused_ce")
_load("preference")
aux_mod = _load("aux_offload")
rt_mod = _load("realtime")

RealtimeQLoRA = rt_mod.RealtimeQLoRA
RealtimeConfig = rt_mod.RealtimeConfig
RWLock = rt_mod.RWLock
ModelParker = aux_mod.ModelParker


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _Ids:
    def __init__(self, ids): self._ids = ids
    def tolist(self): return list(self._ids)


class StubTokenizer:
    """Char-level tokenizer that auto-prepends BOS on EVERY encode() call even
    with add_bos=False (the HF add_bos_token=true behavior the encoding must
    normalize away). No pad token, so the coordinator falls back to EOS."""
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None

    def encode(self, text, add_bos=False, encode_special_tokens=True):
        return [_Ids([self.bos_token_id] + [ord(c) for c in text])]


class StubNet(nn.Module):
    """The narrow NativeLlamaQLoRA surface RealtimeQLoRA drives. One real
    parameter so AdamW/clipping run for real; compute_loss returns queued
    constants (rides the parameter with a zero coefficient so backward works)
    and records every micro-batch for order/shape assertions."""

    def __init__(self, losses=None):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.losses = list(losses or [])
        self.seen = []            # (input_ids, labels, attn) tensors per call
        self.logps_calls = []     # (input_ids, labels, attn, adapters_off)
        self.adapters_off = False
        self.applied = 0
        self.removed = 0
        self.saved = []           # directories save_adapter wrote to
        self.loaded = []          # directories load_adapter read from
        self.target_modules = ["q_proj"]

    def param_groups(self, weight_decay):
        return [{"params": [self.p], "weight_decay": weight_decay}]

    def trainable_parameters(self):
        return [self.p]

    def num_trainable(self):
        return 1

    def compute_loss(self, input_ids, labels, attention_mask=None, chunk=0,
                     position_ids=None, seg_ids=None):
        self.seen.append((input_ids, labels, attention_mask))
        c = self.losses.pop(0) if self.losses else 1.0
        return self.p.sum() * 0.0 + c

    # -- preference surface (DPO / KTO) --
    # The "model" scores every completion token at logp -1 as the frozen
    # base, and at -1 + p as the adapted policy: reference == policy at p = 0
    # (DPO's ln-2 anchor holds exactly), and a longer chosen completion makes
    # the DPO gradient push p UP.
    def compute_logps(self, input_ids, labels, attention_mask=None, chunk=0):
        self.logps_calls.append((input_ids, labels, attention_mask,
                                 self.adapters_off))
        counts = (labels[:, 1:] != -100).sum(dim=-1)
        base = -counts.float()
        if self.adapters_off:
            return base, counts
        return base + self.p * counts.float(), counts

    @contextlib.contextmanager
    def adapters_disabled(self):
        prev = self.adapters_off
        self.adapters_off = True
        try:
            yield self
        finally:
            self.adapters_off = prev

    def apply_to_native(self, scaling=1.0):
        self.applied += 1

    def remove_from_native(self):
        self.removed += 1

    def save_adapter(self, directory, base_model_name_or_path=None):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "adapter_model.safetensors"), "w") as f:
            f.write("stub")

    def load_adapter(self, directory):
        self.loaded.append(directory)
        return 1


class StubGenerator:
    """num_remaining_jobs counts down per poll (simulating in-flight jobs
    draining); pagetable records resets."""

    def __init__(self, jobs=0):
        self.jobs = jobs
        self.pagetable = types.SimpleNamespace(resets=0)
        self.pagetable.reset_page_table = self._reset

    def _reset(self):
        self.pagetable.resets += 1

    def num_remaining_jobs(self):
        n = self.jobs
        if n > 0:
            self.jobs -= 1
        return n


def make_rt(config=None, net=None, **kw):
    return RealtimeQLoRA(None, StubTokenizer(), config or RealtimeConfig(),
                         net=net or StubNet(), **kw)


# ---------------------------------------------------------------------------
# RWLock
# ---------------------------------------------------------------------------

def test_rwlock():
    lock = RWLock()

    # Concurrent readers.
    lock.acquire_read()
    lock.acquire_read()
    lock.release_read()

    # A waiting writer blocks NEW readers (writer preference) but enters only
    # once the in-flight reader drains.
    state = {"writer_in": False, "late_reader_in": False}
    writer_started = threading.Event()

    def writer():
        writer_started.set()
        with lock.write():
            state["writer_in"] = True

    def late_reader():
        with lock.read():
            state["late_reader_in"] = True

    wt = threading.Thread(target=writer)
    wt.start()
    writer_started.wait()
    time.sleep(0.05)                       # writer now queued on the held read
    assert not state["writer_in"]
    rt_thread = threading.Thread(target=late_reader)
    rt_thread.start()
    time.sleep(0.05)
    assert not state["late_reader_in"]     # blocked behind the waiting writer
    lock.release_read()                    # drain the in-flight reader
    wt.join(2)
    rt_thread.join(2)
    assert state["writer_in"] and state["late_reader_in"]

    # Timeout raises instead of hanging.
    lock.acquire_write()
    try:
        try:
            lock.acquire_read(timeout=0.05)
            assert False, "expected TimeoutError"
        except TimeoutError:
            pass
    finally:
        lock.release_write()
    print("rwlock: OK")


# ---------------------------------------------------------------------------
# Sample encoding
# ---------------------------------------------------------------------------

def test_encode_prompt_response():
    rt = make_rt(RealtimeConfig(eot_text="!"))
    ex = rt._encode({"prompt": "ab", "response": "cd"})
    # one BOS survives (auto-prepended twice, deduped once, then add_bos is a
    # no-op); prompt masked, response + eot supervised, boundary exact
    assert ex["input_ids"] == [1, ord("a"), ord("b"), ord("c"), ord("d"), ord("!")]
    assert ex["labels"] == [-100, -100, -100, ord("c"), ord("d"), ord("!")]
    print("encode prompt/response: OK")


def test_encode_text_and_pretokenized():
    rt = make_rt()
    ex = rt._encode({"text": "xy"})
    assert ex["input_ids"] == [1, ord("x"), ord("y")]
    assert ex["labels"] == [-100, ord("x"), ord("y")]       # BOS never supervised

    ex = rt._encode({"input_ids": [7, 8, 9]})
    assert ex["labels"] == [7, 8, 9]                        # default: supervise all
    ex = rt._encode({"input_ids": [7, 8, 9], "labels": [-100, 8, 9]})
    assert ex["labels"] == [-100, 8, 9]
    try:
        rt._encode({"input_ids": [7], "labels": [1, 2]})
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        rt._encode({"bogus": 1})
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("encode text/pretokenized: OK")


def test_encode_truncation():
    rt = make_rt(RealtimeConfig(seq_len=3))
    ex = rt._encode({"text": "abcdefg"})
    assert len(ex["input_ids"]) == 3 and len(ex["labels"]) == 3
    ex = rt._encode({"input_ids": list(range(10, 20))})
    assert ex["input_ids"] == [10, 11, 12]
    print("encode truncation: OK")


def test_encode_messages():
    def render(sample):
        # a fake template: user turn masked, assistant turn supervised
        m = sample["messages"]
        return [(m[0]["content"], False), (m[1]["content"], True)]

    rt = make_rt(render_segments=render)
    ex = rt._encode({"messages": [{"role": "user", "content": "ab"},
                                  {"role": "assistant", "content": "c"}]})
    assert ex["input_ids"] == [1, ord("a"), ord("b"), ord("c")]
    assert ex["labels"] == [-100, -100, -100, ord("c")]

    rt = make_rt()   # no renderer
    try:
        rt._encode({"messages": []})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "render_segments" in str(e)
    print("encode messages: OK")


# ---------------------------------------------------------------------------
# Ingest loop
# ---------------------------------------------------------------------------

def test_ingest_batching_and_callbacks():
    net = StubNet()
    rt = make_rt(RealtimeConfig(batch_size=2, grad_accum=2, lr=1e-3), net=net)
    fired = []
    rt.add_update_callback(lambda: fired.append(1))
    gen = StubGenerator(jobs=3)
    rt.attach_generator(gen)

    # 7 samples @ batch 2 -> 4 micro-batches -> 2 accumulation windows.
    samples = [{"text": chr(ord("a") + i)} for i in range(7)]
    stats = rt.ingest(samples)
    assert stats["steps"] == 2 and rt.step == 2
    assert stats["samples"] == 7 and rt.samples_seen == 7
    assert len(net.seen) == 4
    # in-order consumption: first micro-batch holds samples 0..1
    assert net.seen[0][0].shape[0] == 2
    assert net.seen[0][0][0].tolist()[:2] == [1, ord("a")]
    assert net.seen[3][0].shape[0] == 1                     # 7th sample alone
    # one adapter push + one cache nuke + generator drained first
    assert net.applied == 1
    assert gen.pagetable.resets == 1
    assert len(fired) == 1
    assert gen.jobs == 0

    # a second ingest continues the same run
    stats = rt.ingest([{"text": "z"}])
    assert stats["steps"] == 1 and rt.step == 3 and net.applied == 2

    # empty ingest is a no-op (no sync, no callbacks)
    stats = rt.ingest([])
    assert stats["steps"] == 0 and net.applied == 2
    print("ingest batching/callbacks: OK")


def test_ingest_token_weighted_loss():
    # Two micro-batches in one window with 1 and 3 supervised (shifted)
    # tokens and losses 2.0 / 4.0: the token-weighted mean is
    # (2*1 + 4*3) / 4 = 3.5, not the plain mean 3.0.
    net = StubNet(losses=[2.0, 4.0])
    rt = make_rt(RealtimeConfig(batch_size=1, grad_accum=2), net=net)
    stats = rt.ingest([
        {"input_ids": [5, 6], "labels": [5, 6]},              # 1 shifted sup
        {"input_ids": [5, 6, 7, 8], "labels": [5, 6, 7, 8]},  # 3 shifted sup
    ])
    assert stats["steps"] == 1
    assert stats["supervised_tokens"] == 4
    assert abs(stats["mean_loss"] - 3.5) < 1e-6
    print("ingest token-weighted loss: OK")


def test_lr_control():
    rt = make_rt(RealtimeConfig(lr=1e-4))
    assert abs(rt.lr - 1e-4) < 1e-12
    rt.lr = 5e-5
    assert all(abs(g["lr"] - 5e-5) < 1e-12 for g in rt.opt.param_groups)
    rt.ingest([{"text": "a"}], lr=3e-5)
    assert abs(rt.lr - 3e-5) < 1e-12       # per-call override sticks
    print("lr control: OK")


def test_ingest_blocks_inference():
    """While an ingest runs, rt.inference() must wait; after it returns the
    adapter has been applied."""
    net = StubNet()
    rt = make_rt(RealtimeConfig(batch_size=1, grad_accum=1), net=net)
    in_ingest = threading.Event()
    release = threading.Event()
    orig_apply = net.apply_to_native

    def slow_apply(scaling=1.0):
        in_ingest.set()
        release.wait(2)
        orig_apply(scaling)

    net.apply_to_native = slow_apply
    t = threading.Thread(target=rt.ingest, args=([{"text": "a"}],))
    t.start()
    in_ingest.wait(2)
    got_in = {"v": False}

    def reader():
        with rt.inference():
            got_in["v"] = True

    r = threading.Thread(target=reader)
    r.start()
    time.sleep(0.05)
    assert not got_in["v"]                 # blocked while ingest holds write
    release.set()
    t.join(2)
    r.join(2)
    assert got_in["v"] and net.applied == 1
    print("ingest blocks inference: OK")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def test_checkpoint_naming_and_listing():
    now = time.localtime(1700000000)
    name = rt_mod.checkpoint_name(42, now)
    ts = time.strftime("%Y%m%d-%H%M%S", now)
    assert name == f"ckpt-{ts}-step42"
    print("checkpoint naming: OK")


def test_checkpoint_cadence_prune_resume():
    with tempfile.TemporaryDirectory() as d:
        net = StubNet()
        cfg = RealtimeConfig(batch_size=1, grad_accum=1, checkpoint_dir=d,
                             checkpoint_every=2, keep_checkpoints=2, lr=1e-3)
        rt = make_rt(cfg, net=net)
        stats = rt.ingest([{"text": chr(ord("a") + i)} for i in range(6)])
        # 6 steps @ every-2 -> checkpoints at steps 2, 4, 6; pruned to last 2
        assert len(stats["checkpoints"]) == 3
        names = rt_mod.list_realtime_checkpoints(d)
        assert len(names) == 2
        assert names[-1].endswith("-step6")
        last = os.path.join(d, names[-1])
        assert os.path.exists(os.path.join(last, "realtime_trainer_state.pt"))
        with open(os.path.join(last, "realtime_meta.json")) as f:
            meta = json.load(f)
        assert meta["step"] == 6 and meta["config"]["r"] == cfg.r

        # Resume into a fresh coordinator: step counter and optimizer state
        # come back; the adapter is pushed to inference immediately.
        net2 = StubNet()
        rt2 = make_rt(RealtimeConfig(lr=1e-3), net=net2)
        rt2.load(last)
        assert rt2.step == 6 and rt2.samples_seen == 6
        assert net2.loaded == [last]
        assert net2.applied == 1
        # optimizer state actually restored (AdamW step count is per-param)
        st = rt2.opt.state_dict()["state"]
        assert st and all(int(s["step"]) == 6 for s in st.values())
        print("checkpoint cadence/prune/resume: OK")


def test_manual_checkpoint_requires_dir():
    rt = make_rt()
    try:
        rt.checkpoint()
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("manual checkpoint guard: OK")


class OffloadStubNet(StubNet):
    """StubNet that also implements the idle-offload surface, recording the
    interleaving so tests can assert restore-before-train / offload-after-sync
    ordering, and that training never runs while offloaded."""

    def __init__(self, losses=None):
        super().__init__(losses)
        self.offloaded = False
        self.events = []

    def offload_training_state(self):
        assert not self.offloaded
        self.offloaded = True
        self.events.append("offload")
        return True

    def restore_training_state(self):
        self.offloaded = False
        self.events.append("restore")

    def compute_loss(self, *args, **kwargs):
        assert not self.offloaded, "trained while training state was offloaded"
        self.events.append("loss")
        return super().compute_loss(*args, **kwargs)

    def apply_to_native(self, scaling=1.0):
        self.events.append("apply")
        super().apply_to_native(scaling)

    def load_adapter(self, directory):
        assert not self.offloaded, "load_adapter while training state offloaded"
        self.events.append("load")
        return super().load_adapter(directory)


def test_idle_offload_lifecycle():
    net = OffloadStubNet()
    rt = make_rt(net=net)                   # offload_when_idle defaults on
    # Parked immediately at construction (serving footprint from the start).
    assert net.events == ["offload"] and rt._idle_offloaded

    rt.ingest([{"text": "ab"}])
    # Restored before training, trained, pushed to inference, re-parked.
    assert net.events == ["offload", "restore", "loss", "apply", "offload"]
    assert net.offloaded and rt._idle_offloaded

    # load() un-parks around the state surgery and re-parks after the sync.
    with tempfile.TemporaryDirectory() as d:
        cfg_dir = os.path.join(d, "ck")
        net.save_adapter(cfg_dir)
        net.events.clear()
        rt.load(cfg_dir)
        assert net.events == ["restore", "load", "apply", "offload"]
    print("idle offload lifecycle: OK")


def test_idle_offload_disabled():
    net = OffloadStubNet()
    rt = make_rt(RealtimeConfig(offload_when_idle=False), net=net)
    rt.ingest([{"text": "ab"}])
    assert not rt._idle_offloaded
    assert "offload" not in net.events and "restore" not in net.events
    print("idle offload disabled: OK")


def test_idle_offload_plain_stub():
    # A net without the offload surface (the DI contract) must still work:
    # the coordinator only moves optimizer state, which is a CPU no-op here.
    net = StubNet()
    rt = make_rt(net=net)
    assert rt._idle_offloaded
    stats = rt.ingest([{"text": "ab"}])
    assert stats["steps"] == 1 and rt._idle_offloaded
    assert rt._opt_state_homes == []        # nothing was on an accelerator
    print("idle offload without net surface: OK")


class StubAuxModule:
    """One top-level module of a stub component model: the narrow surface
    ModelParker drives (device attribute, unload/load, can_defer_load)."""

    def __init__(self, device, log, name, defer=False):
        self.device = device
        self._log = log
        self._name = name
        self._defer = defer

    def can_defer_load(self):
        return self._defer

    def unload(self):
        self._log.append(f"unload:{self._name}")
        self.device = None

    def load(self, device):
        self._log.append(f"load:{self._name}:{device}")
        self.device = device


class StubSTC:
    def __init__(self, log):
        self._log = log
        self.deferred = 0

    def begin_deferred_load(self):
        self.deferred += 1
        self._log.append("defer+")

    def end_deferred_load(self):
        assert self.deferred > 0
        self.deferred -= 1
        self._log.append("defer-")

    def abort_deferred_load(self):
        assert self.deferred > 0
        self.deferred -= 1
        self._log.append("defer!")

    def close(self):
        self._log.append("stc_close")


class StubAuxModel:
    """Narrow loaded-component-Model surface: .modules, .config.stc,
    loaded_tp. devices[i] is module i's device (None = never loaded)."""

    def __init__(self, devices, log=None, defer_idx=()):
        self.log = log if log is not None else []
        self.loaded_tp = False
        self.config = types.SimpleNamespace(
            stc=StubSTC(self.log),
            infer_params=types.SimpleNamespace(vision_pinned=False),
        )
        self.modules = [
            StubAuxModule(d, self.log, str(i), defer=(i in defer_idx))
            for i, d in enumerate(devices)
        ]


def test_model_parker():
    m = StubAuxModel(["cuda:0", "cpu", None, "cuda:1"], defer_idx={0})
    p = ModelParker(m)
    assert not p.parked

    p.park()
    assert p.parked
    # Only device-resident modules are unloaded; cpu/never-loaded stay put.
    assert m.log == ["unload:0", "unload:3"]
    assert m.modules[1].device == "cpu" and m.modules[2].device is None
    p.park()                                  # idempotent
    assert m.log == ["unload:0", "unload:3"]

    m.log.clear()
    p.unpark()
    assert not p.parked
    # Reloaded to the recorded devices in module order; deferred-load
    # bracketing only where the module supports it; handles closed after.
    assert m.log == ["defer+", "load:0:cuda:0", "defer-",
                     "load:3:cuda:1", "stc_close"]
    assert m.config.stc.deferred == 0
    p.unpark()                                # idempotent
    assert m.log[-1] == "stc_close" and len(m.log) == 5

    # Tensor-parallel models are refused.
    tp = StubAuxModel(["cuda:0"])
    tp.loaded_tp = True
    try:
        ModelParker(tp).park()
        assert False, "expected AssertionError"
    except AssertionError:
        pass
    print("ModelParker park/unpark: OK")


def test_aux_offload_in_ingest():
    events = []
    aux = StubAuxModel(["cuda:0"], log=events)

    class Net(OffloadStubNet):
        def restore_training_state(self):
            # Aux models leave VRAM BEFORE the training state comes back.
            assert aux.modules[0].device is None, \
                "training state restored before aux model was parked"
            super().restore_training_state()

        def compute_loss(self, *args, **kwargs):
            assert aux.modules[0].device is None, \
                "aux model resident in VRAM during training"
            return super().compute_loss(*args, **kwargs)

        def offload_training_state(self):
            # On the way out of an ingest the training state re-parks BEFORE
            # the aux models return. (The construction-time park runs outside
            # any ingest, before aux models are even attached -- skip it.)
            r = super().offload_training_state()
            if "restore" in self.events:
                assert aux.modules[0].device is None, \
                    "aux model returned before training state was re-parked"
            return r

    net = Net()
    rt = make_rt(net=net)
    rt.attach_aux_models(aux, None)           # None entries are ignored
    assert len(rt._aux_parkers) == 1

    rt.ingest([{"text": "ab"}])
    assert aux.modules[0].device == "cuda:0"  # restored for serving
    assert net.events == ["offload", "restore", "loss", "apply", "offload"]
    assert events == ["unload:0", "load:0:cuda:0", "stc_close"]
    print("aux offload in ingest: OK")


def test_aux_offload_restored_on_error():
    aux = StubAuxModel(["cuda:0"])

    class BoomNet(StubNet):
        def compute_loss(self, *args, **kwargs):
            raise RuntimeError("boom")

    rt = make_rt(net=BoomNet())
    rt.attach_aux_models(aux)
    try:
        rt.ingest([{"text": "ab"}])
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    # Restored despite the failure, and the write lock was released.
    assert aux.modules[0].device == "cuda:0"
    with rt._lock.read(timeout=1.0):
        pass
    print("aux offload restored on ingest failure: OK")


def test_aux_offload_disabled():
    aux = StubAuxModel(["cuda:0"])
    rt = make_rt(RealtimeConfig(offload_aux_when_training=False))
    rt.attach_aux_models(aux)
    rt.ingest([{"text": "ab"}])
    assert aux.log == [] and aux.modules[0].device == "cuda:0"
    print("aux offload disabled: OK")


def test_internal_net_build_passes_head_vocab_chunk():
    """The coordinator's own net build must hand RealtimeConfig.head_vocab_chunk
    to NativeLlamaQLoRA: the single-shot fused head reconstructs the whole
    [hidden, vocab] weight plus two fp32 copies (~12 GB on a 248k-vocab 27B),
    which OOMs a card that is also serving -- seen as CUDA driver errors under
    expandable segments. Chunked by default; 0 opts out."""
    captured = []

    class FakeNative(StubNet):
        def __init__(self, model, **kw):
            super().__init__()
            captured.append(kw)

    fake = types.ModuleType("exl3train.native_llama")
    fake.NativeLlamaQLoRA = FakeNative
    with patch.dict(sys.modules, {"exl3train.native_llama": fake}):
        RealtimeQLoRA(None, StubTokenizer(), RealtimeConfig())
        assert captured[-1]["head_vocab_chunk"] == 32768
        RealtimeQLoRA(None, StubTokenizer(), RealtimeConfig(head_vocab_chunk=0))
        assert captured[-1]["head_vocab_chunk"] == 0
        config = RealtimeConfig.from_dict({"head_vocab_chunk": 8192})
        RealtimeQLoRA(None, StubTokenizer(), config)
        assert captured[-1]["head_vocab_chunk"] == 8192
    print("internal net build passes head_vocab_chunk: OK")


def test_unload_reload():
    net = StubNet()
    rt = make_rt(net=net)
    fired = []
    rt.add_update_callback(lambda: fired.append(1))
    rt.unload_from_inference()             # never synced -> no-op
    assert net.removed == 0 and not fired
    rt.ingest([{"text": "a"}])
    assert net.applied == 1 and len(fired) == 1
    rt.unload_from_inference()
    assert net.removed == 1 and len(fired) == 2
    rt.sync_to_inference()
    assert net.applied == 2 and len(fired) == 3
    print("unload/reload: OK")


# ---------------------------------------------------------------------------
# Preference samples (DPO / KTO)
# ---------------------------------------------------------------------------

def _render_prompt(sample):
    # a fake template: every prompt message as a segment, then the
    # assistant-turn opener (the generation prompt)
    return [(m["content"], False) for m in sample["prompt"]] + [("<a>", False)]


def test_encode_preference():
    rt = make_rt(RealtimeConfig(eot_text="!"), render_prompt=_render_prompt)

    # DPO, rendered-string prompt: prompt keeps one BOS (auto-prepended,
    # deduped), completions lose theirs and gain the eot; nothing is
    # supervised in the prompt (the kind tag is the only extra key)
    ex = rt._encode({"prompt": "ab", "chosen": "c", "rejected": "de"})
    assert ex["kind"] == "dpo"
    assert ex["prompt_ids"] == [1, ord("a"), ord("b")]
    assert ex["chosen_ids"] == [ord("c"), ord("!")]
    assert ex["rejected_ids"] == [ord("d"), ord("e"), ord("!")]

    # conversational prompt through render_prompt (+ generation prompt);
    # completions as assistant message lists
    ex = rt._encode({"prompt": [{"role": "user", "content": "a"}],
                     "chosen": [{"role": "assistant", "content": "c"}],
                     "rejected": [{"role": "assistant", "content": "d"}]})
    assert ex["prompt_ids"] == [1, ord("a")] + [ord(ch) for ch in "<a>"]
    assert ex["chosen_ids"] == [ord("c"), ord("!")]

    # KTO: label coercion (bool / int / string spellings), kind tag
    ex = rt._encode({"prompt": "ab", "completion": "c", "label": "false"})
    assert ex["kind"] == "kto" and ex["label"] is False
    assert ex["completion_ids"] == [ord("c"), ord("!")]
    assert rt._encode({"prompt": "a", "completion": "c", "label": 1})["label"] is True
    assert rt._encode({"prompt": "a", "completion": "c", "label": True})["label"] is True

    # pre-tokenized forms pass through untouched (no BOS / eot handling)
    ex = rt._encode({"prompt_ids": [5, 6], "chosen_ids": [7],
                     "rejected_ids": [8, 9]})
    assert ex == {"kind": "dpo", "prompt_ids": [5, 6], "chosen_ids": [7],
                  "rejected_ids": [8, 9]}
    ex = rt._encode({"prompt_ids": [5], "completion_ids": [7, 7], "label": 0})
    assert ex == {"kind": "kto", "prompt_ids": [5], "completion_ids": [7, 7],
                  "label": False}

    # malformed rows raise (never silently mistrain)
    for bad in ({"prompt": "a", "completion": "c", "label": "maybe"},
                {"prompt": "a", "chosen": "", "rejected": "d"},
                {"prompt": "a", "completion": "  ", "label": True},
                {"prompt": 7, "chosen": "c", "rejected": "d"}):
        try:
            rt._encode(bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass
    try:
        make_rt()._encode({"prompt": [{"role": "user", "content": "a"}],
                           "chosen": "c", "rejected": "d"})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "render_prompt" in str(e)

    # seq_len: completion tail truncated first; prompt-only overflow -> None
    rt = make_rt(RealtimeConfig(seq_len=4, eot_text="!"))
    ex = rt._encode({"prompt": "ab", "chosen": "cd", "rejected": "e"})
    assert ex["chosen_ids"] == [ord("c")] and ex["rejected_ids"] == [ord("e")]
    assert rt._encode({"prompt": "abc", "chosen": "d", "rejected": "e"}) is None
    assert rt._encode({"prompt": "abc", "completion": "d", "label": True}) is None

    # config validation of the loss variants
    for kw in ({"dpo_loss": "bogus"}, {"kto_loss": "bogus"}):
        try:
            RealtimeConfig(**kw)
            assert False, "expected ValueError"
        except ValueError:
            pass
    print("encode preference: OK")


def _pair(prompt, chosen_len, rejected_len):
    return {"prompt_ids": prompt, "chosen_ids": [7] * chosen_len,
            "rejected_ids": [8] * rejected_len}


def test_ingest_dpo():
    net = StubNet()
    rt = make_rt(RealtimeConfig(batch_size=2, grad_accum=1, lr=1e-2,
                                weight_decay=0.0, beta=0.1), net=net)
    fired = []
    rt.add_update_callback(lambda: fired.append(1))
    # 3 pairs @ batch 2 -> micro-batches of 2 and 1 -> 2 steps. Chosen
    # completions are longer (3 tokens vs 1), so the policy's p lifts the
    # chosen logratio more than the rejected one: the gradient pushes p up.
    pairs = [_pair([1, 2], 3, 1), _pair([1, 3], 3, 1), _pair([1, 4], 3, 1)]
    stats = rt.ingest(pairs)

    assert stats["steps"] == 2 and rt.step == 2
    assert stats["samples"] == 3 and rt.samples_seen == 3
    assert stats["skipped"] == 0
    assert stats["sft_samples"] == 0 and stats["mean_loss"] is None
    assert stats["kto"] is None
    assert not net.seen                        # no SFT forward at all
    # first micro-batch: one policy + one reference (adapters off) forward
    # over 2*b rows, chosen block first
    ids0, labels0, attn0, off0 = net.logps_calls[0]
    _, _, _, off1 = net.logps_calls[1]
    assert (off0, off1) == (False, True)
    assert ids0.shape[0] == 4
    assert ids0[0].tolist()[:2] == [1, 2] and ids0[2].tolist()[:2] == [1, 2]
    counts = (labels0[:, 1:] != -100).sum(dim=-1).tolist()
    assert counts == [3, 3, 1, 1]              # chosen ×2, then rejected ×2
    assert labels0[0].tolist()[:2] == [-100, -100]   # prompt masked
    assert len(net.logps_calls) == 4           # 2 micro-batches × (pol, ref)
    assert net.adapters_off is False           # context restored
    # step-0 anchor: policy == reference -> loss ln 2, zero rewards; the
    # second micro-batch ran after one step, so the pair-weighted mean sits
    # at or below ln 2 and p moved up
    d = stats["dpo"]
    assert d["pairs"] == 3
    assert d["loss"] <= math.log(2) + 1e-6
    assert net.p.item() > 0
    assert stats["supervised_tokens"] == 3 * (3 + 1)
    assert stats["total_tokens"] == sum(int(a.sum()) for _, _, a, off
                                        in net.logps_calls if not off)
    assert net.applied == 1 and len(fired) == 1

    # with p > 0 the chosen reward now exceeds the rejected one on every pair
    stats = rt.ingest(pairs)
    d = stats["dpo"]
    assert d["acc"] == 1.0 and d["margin"] > 0
    assert d["loss"] < math.log(2)
    assert rt.step == 4 and net.applied == 2

    # the hinge / ipo variants run through the same path. hinge: relu(1 -
    # beta*delta) still pushes p up; ipo length-normalizes, and the stub's
    # per-token logratio is p on both sides, so its loss is the constant
    # (0 - 1/(2 beta))^2 = 25 at beta 0.1 with a zero gradient.
    net = StubNet()
    rt = make_rt(RealtimeConfig(batch_size=1, grad_accum=3, lr=1e-2,
                                dpo_loss="hinge"), net=net)
    stats = rt.ingest(pairs)
    assert stats["steps"] == 1 and stats["dpo"]["pairs"] == 3
    assert net.p.item() > 0
    net = StubNet()
    rt = make_rt(RealtimeConfig(batch_size=1, grad_accum=3, lr=1e-2,
                                dpo_loss="ipo"), net=net)
    stats = rt.ingest(pairs)
    assert stats["steps"] == 1 and abs(stats["dpo"]["loss"] - 25.0) < 1e-4
    assert net.p.item() == 0.0
    print("ingest dpo: OK")


def _row(prompt, comp_len, label):
    return {"prompt_ids": prompt, "completion_ids": [7] * comp_len,
            "label": label}


def test_ingest_kto():
    # batch 2: KL rows from mismatched pairs (prompt i + completion i-1) ->
    # 4 forwards per micro-batch: policy, reference, KL policy, KL reference
    net = StubNet()
    rt = make_rt(RealtimeConfig(batch_size=2, grad_accum=1, lr=1e-2,
                                weight_decay=0.0), net=net)
    rows = [_row([1, 2], 3, True), _row([1, 3], 1, False)]
    stats = rt.ingest(rows)
    assert stats["steps"] == 1 and stats["samples"] == 2
    assert stats["dpo"] is None and stats["mean_loss"] is None
    assert len(net.logps_calls) == 4
    offs = [c[3] for c in net.logps_calls]
    assert offs == [False, True, False, True]
    kl_ids, kl_labels, _, _ = net.logps_calls[2]
    assert kl_ids[0].tolist()[:2] == [1, 2] and kl_ids[1].tolist()[:2] == [1, 3]
    kl_counts = (kl_labels[:, 1:] != -100).sum(dim=-1).tolist()
    assert kl_counts == [1, 3]                 # completions swapped
    # step 0: policy == reference -> KL 0, every row at 1 - sigmoid(0) = 0.5
    k = stats["kto"]
    assert k["samples"] == 2
    assert abs(k["loss"] - 0.5) < 1e-6 and abs(k["kl"]) < 1e-6
    assert abs(k["reward_d"]) < 1e-6 and abs(k["reward_u"]) < 1e-6
    assert stats["supervised_tokens"] == 4
    # the desirable row is longer, so lifting p raises its reward more than
    # the undesirable row's: p moves up
    assert net.p.item() > 0

    # singleton micro-batches: no KL rows (2 forwards each), loss reduces to
    # apo_zero_unpaired; row weights scale the per-row losses
    net = StubNet()
    rt = make_rt(RealtimeConfig(batch_size=1, grad_accum=2, lr=1e-2,
                                desirable_weight=2.0, undesirable_weight=1.0),
                 net=net)
    stats = rt.ingest(rows)
    assert stats["steps"] == 1
    assert len(net.logps_calls) == 4 and [c[3] for c in net.logps_calls] == \
        [False, True, False, True]
    assert all(c[0].shape[0] == 1 for c in net.logps_calls)
    assert abs(stats["kto"]["loss"] - (2.0 * 0.5 + 1.0 * 0.5) / 2) < 1e-6
    assert stats["kto"]["reward_u"] is not None

    # all-desirable batch: reward_u is None, not NaN
    net = StubNet()
    rt = make_rt(RealtimeConfig(batch_size=2, kto_loss="apo_zero_unpaired"),
                 net=net)
    stats = rt.ingest([_row([1, 2], 2, True), _row([1, 3], 2, True)])
    assert stats["kto"]["reward_u"] is None and stats["kto"]["reward_d"] == 0.0
    assert len(net.logps_calls) == 2           # apo_zero: no KL forwards
    print("ingest kto: OK")


def test_ingest_mixed_kinds():
    # A kind change closes the accumulation window: [sft, sft, dpo, sft] at
    # grad_accum 4 is three runs -> three steps, in order, and the SFT
    # mean loss only counts the SFT rows.
    net = StubNet(losses=[2.0, 2.0, 6.0])
    rt = make_rt(RealtimeConfig(batch_size=1, grad_accum=4), net=net)
    stats = rt.ingest([
        {"input_ids": [5, 6], "labels": [5, 6]},
        {"input_ids": [5, 6], "labels": [5, 6]},
        _pair([1, 2], 2, 1),
        {"input_ids": [5, 6], "labels": [5, 6]},
    ])
    assert stats["steps"] == 3 and rt.step == 3
    assert stats["samples"] == 4 and rt.samples_seen == 4
    assert stats["sft_samples"] == 3 and stats["dpo"]["pairs"] == 1
    assert abs(stats["mean_loss"] - (2.0 + 2.0 + 6.0) / 3) < 1e-6
    assert len(net.seen) == 3 and len(net.logps_calls) == 2
    assert stats["supervised_tokens"] == 3 * 1 + (2 + 1)
    assert net.applied == 1

    # skipped rows (prompt fills seq_len) are counted, the rest still train;
    # an ingest of only skipped rows is a no-op that reports the count
    rt = make_rt(RealtimeConfig(seq_len=3), net=StubNet())
    stats = rt.ingest([_pair([1, 2, 3], 1, 1), {"text": "ab"}])
    assert stats["skipped"] == 1 and stats["steps"] == 1
    stats = rt.ingest([_pair([1, 2, 3], 1, 1)])
    assert stats["skipped"] == 1 and stats["steps"] == 0
    assert stats["dpo"] is None and stats["kto"] is None
    print("ingest mixed kinds: OK")


if __name__ == "__main__":
    test_rwlock()
    test_encode_prompt_response()
    test_encode_text_and_pretokenized()
    test_encode_truncation()
    test_encode_messages()
    test_ingest_batching_and_callbacks()
    test_ingest_token_weighted_loss()
    test_encode_preference()
    test_ingest_dpo()
    test_ingest_kto()
    test_ingest_mixed_kinds()
    test_lr_control()
    test_ingest_blocks_inference()
    test_checkpoint_naming_and_listing()
    test_checkpoint_cadence_prune_resume()
    test_manual_checkpoint_requires_dir()
    test_idle_offload_lifecycle()
    test_idle_offload_disabled()
    test_idle_offload_plain_stub()
    test_model_parker()
    test_aux_offload_in_ingest()
    test_aux_offload_restored_on_error()
    test_aux_offload_disabled()
    test_internal_net_build_passes_head_vocab_chunk()
    test_unload_reload()
    print("\nALL OK")
