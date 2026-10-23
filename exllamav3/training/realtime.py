"""
Real-time (inference-time) QLoRA training over a loaded exllamav3 ``Model``.

The idea: one loaded EXL3 model serves generation AND trains its LoRA adapter,
alternating between the two. This works because the two forwards already share
everything: the differentiable training forward (:class:`NativeLlamaQLoRA`)
reconstructs the decoder on top of the *same* loaded modules the native
inference forward uses, and the native ``Linear`` modules carry runtime LoRA
slots that ``NativeLlamaQLoRA.apply_to_native()`` can fill in memory -- no
save/load round-trip, no second model copy. This module adds the coordination
layer that makes the alternation safe and useful:

  * :class:`RealtimeQLoRA` -- owns the trainable net, the AdamW optimizer and
    a global step counter, all kept alive between calls (the adapter trains
    across many small ingests like one long run).
  * ``ingest(samples)`` -- trains through an array of samples at the
    configured batch size / gradient accumulation until they are depleted,
    then pushes the updated adapter into the inference forward and fires the
    registered update callbacks (KV-cache invalidation), then returns to
    serving.
  * a readers/writer lock -- inference requests are readers
    (``with rt.inference(): ...``), an ingest is the writer. A waiting ingest
    blocks NEW inference readers (writer preference, so a busy server cannot
    starve training) but always lets in-flight ones drain first.
  * timestamped checkpoints -- every ``checkpoint_every`` optimizer steps the
    adapter (PEFT format, loadable by ``LoRA.from_directory`` / PEFT / the
    offline trainers) plus optimizer state is written to
    ``checkpoint_dir/ckpt-YYYYmmdd-HHMMSS-stepN``, giving a rollback point if
    a bad batch damages the adapter. ``keep_checkpoints`` prunes old ones.
  * a constant, externally settable learning rate (``rt.lr = 5e-5`` between
    ingests, or ``ingest(..., lr=...)`` per call) -- an endless sample stream
    has no epochs, so there is no schedule to run.

KV-cache staleness: after an adapter update, every cached KV entry computed
THROUGH an adapted k_proj/v_proj is stale. Two supported answers:

  1. (default) nuke the cache on every update -- ``attach_generator(gen)``
     registers ``invalidate_generator_cache(gen)`` as an update callback: the
     VRAM page table is reset, and the CPU page tier and the recurrent-state
     checkpoints behind it are cleared too (pages restored from those by hash
     would otherwise carry KV computed through the previous adapter), so prefix
     reuse restarts from nothing and the next requests re-prefill. A generator
     with a disk cache tier is refused: its pages persist across processes and
     its fingerprint knows nothing about adapters.
  2. adapt only projections that never enter the cache --
     ``target_modules=["q_proj", "o_proj", "gate_proj", "up_proj",
     "down_proj"]`` -- and register no cache callback. Zero invalidation cost,
     slightly weaker adapter.

Serving integration (e.g. tabbyAPI's ``backends/exllamav3/model.py``): build a
``RealtimeQLoRA`` next to the loaded ``Model``/``Generator``, call
``attach_generator(generator)``, wrap each generator drive (the
``generator.iterate()`` loop or a blocking ``generator.generate``) in
``with rt.inference():``, and hook ``rt.ingest(samples)`` (run it in a worker
thread/executor -- it blocks until the samples are depleted) to whatever
endpoint or UI feeds training examples. Everything else -- what the model
serves, how requests are queued -- is untouched; the wrapper deliberately does
NOT re-implement any of ``Model``'s API.

Sample forms accepted by ``ingest`` (mixable in one call):

  * ``{"input_ids": [...], "labels": [...]}`` -- pre-tokenized (labels
    optional; ``-100`` masks positions; defaults to supervising everything).
  * ``{"text": "..."}`` -- plain text, fully supervised.
  * ``{"prompt": "...", "response": "..."}`` -- tokenized separately and
    concatenated so the mask boundary is exact; the prompt is masked, the
    response (plus ``config.eot_text``, if set) is supervised. The caller
    renders the chat template into these strings -- exactly what an inference
    server already does.
  * ``{"messages": [...]}`` -- OpenAI-style conversations, IF a
    ``render_segments`` callable was provided (sample -> ordered
    ``[(text, supervised)]`` segments; ``training/chat_jinja.py``'s
    template-driven builders produce this contract -- see
    ``training/realtime_chat.py`` for the wiring).

Image+text samples (VLM bases -- Qwen-VL, Gemma3/4, Mistral3): a
``messages`` sample (or a preference sample's conversational ``prompt``)
whose turn contents are OpenAI content-parts lists carrying image parts
(``{"type": "image", "path"|"url"|"base64"|"image"|"image_url": ...}``)
trains WITH the pixels when a ``vision`` service is wired in
(``RealtimeQLoRA(..., vision=...)``; ``training/vision_data.py``'s
``VisionData`` is the reference implementation, built on the frozen vision
component the generator serves images through). The exact offline
``--vision`` recipe applies: the tower runs frozen, its features replace the
token embeddings at the arch's own image token layout (``<|vision_start|>``
+ N slots + ``<|vision_end|>`` on Qwen-VL, ...), image tokens are masked,
3-D mRoPE positions / deepstack maps / bidirectional image spans are
reproduced in the differentiable forward, and a sample whose image layout
would be cut by ``seq_len`` is skipped rather than truncated through the
image. Without a ``vision`` service an image-bearing sample is REFUSED
(``ValueError``): the template would still render its placeholder tokens,
and an adapter trained on those empty slots learns answers conditioned on
nothing -- the loss falls, the served behaviour (which sees real features
in those slots) does not move. Text-only samples train as before on a VLM
base. The ``vision`` object's surface: ``mrope`` (bool), ``flatten(messages,
base_dir=None) -> (messages_with_sentinels, refs)`` and
``encode_segments(segments, refs) -> (input_ids, labels, [(start, k)],
{k: ImageFeatures})``.

Preference samples (DPO / KTO) mix into the same ``ingest`` call. They use
TRL's *explicit prompt* format -- the prompt is its own field, never
repeated inside the completions -- and train the adapter with the same
objectives as the offline ``training/qlora_train_pref.py`` (the frozen base
is the reference model, obtained by running the same net with its adapters
disabled; no second model copy):

  * ``{"prompt": ..., "chosen": ..., "rejected": ...}`` -- one DPO pair.
    ``prompt`` is either a rendered prompt string (ending with the
    assistant-turn opener, as for ``prompt``/``response``) or a list of
    messages, which needs a ``render_prompt`` callable (sample -> masked
    ``[(text, supervised)]`` prompt segments rendered WITH the generation
    prompt; ``realtime_chat.py`` wires the chat template). ``chosen`` /
    ``rejected`` are completion strings or single-turn assistant message
    lists; ``config.eot_text`` is appended to each. Loss: ``config.dpo_loss``
    (sigmoid / hinge / ipo) at ``config.beta``, cDPO ``label_smoothing`` --
    or, with ``config.pair_objective = "orpo"``, the reference-free ORPO
    objective (NLL on chosen + odds-ratio term at ``config.beta`` = the
    paper's lambda; ``config.orpo_nll`` full / completion picks what the NLL
    supervises, TRL/axolotl's ``full`` by default). Pairs are then half the
    cost of DPO (no reference forward) and the stats still land under
    ``"dpo"`` (they are pairs either way).
  * ``{"prompt": ..., "completion": ..., "label": True/False}`` -- one
    UNPAIRED KTO row (``label`` truthy = desirable). Loss: ``config.kto_loss``
    (kto / apo_zero_unpaired) at ``config.beta`` with
    ``desirable_weight`` / ``undesirable_weight``. The ``kto`` variant's KL
    reference point is estimated per micro-batch from mismatched
    prompt/completion pairs, which needs ``batch_size >= 2`` (TRL's rule);
    a singleton KTO micro-batch trains with ``KL = 0``, which is exactly the
    ``apo_zero_unpaired`` loss.
  * pre-tokenized: ``{"prompt_ids", "chosen_ids", "rejected_ids"}`` and
    ``{"prompt_ids", "completion_ids", "label"}`` (no BOS/eot handling --
    the ids are used as given).

SFT rows, DPO pairs and KTO rows need different forwards, so consecutive
same-kind samples form the micro-batches and a kind change closes the
current accumulation window (order is still preserved). Preference windows
accumulate example-weighted (the step gradient equals one big batch mean,
TRL's reduction); a prompt+completion that does not fit ``seq_len`` gets its
completion tail truncated, and a row whose prompt alone fills ``seq_len`` is
skipped and counted in the returned stats. Note that a preference step costs
one extra no-grad reference forward (two more for KTO's KL rows) on top of
the policy forward, and that a DPO pair is two sequences per forward.

VRAM note: training shares the GPU with the serving cache. The transient
training footprint (activations under gradient checkpointing + fp32 Adam
moments for the LoRA only + per-layer trellis dequant scratch) rides on top of
the allocated ``Cache``; size ``seq_len`` x ``batch_size`` so both fit --
on a 16 GB card with a small model and a modest cache, seq_len 1-2k at batch 1
with grad accumulation is the sane starting point.

Idle offload (``offload_when_idle``, default on): between ingests the model
only SERVES, but the persistent training state -- fp32 LoRA masters, fp32 Adam
moments, PiSSA offset copies -- would otherwise sit idle in VRAM. So after
every ingest (and at construction) that state is moved to system memory and
the CUDA caching allocator is flushed; the next ingest moves it back before
training. Value-exact both ways (device moves, not casts), and generation is
unaffected while offloaded -- inference reads the fp16 runtime LoRA slots
pushed by ``sync_to_inference``, which stay on the GPU. The cost is one
HtoD/DtoH round-trip of the training state per ingest, noise next to the
training itself. Disable it (``offload_when_idle=False``) only if ingests are
so tiny and frequent that the copies ever show up in a profile.

Aux component parking (``offload_aux_when_training``, default on): a serving
stack for a multimodal / MTP-capable checkpoint typically loads extra
component models next to the text trunk -- a vision tower for image input,
an MTP head as speculative-decoding draft model. The backward never touches
them (the LoRA targets live in the text trunk; the vision tower only
produces the frozen features an image sample trains against, and that
happens while the samples are encoded, BEFORE the training forwards), so
while the training forwards run they are dead weight in VRAM. Register them
with ``attach_aux_models(vision_model, draft_model)`` and every ingest parks
them out of VRAM for its duration -- unloaded after the samples are encoded
(KV cache layers of a draft model included), reloaded value-exact to the
same devices on the way out (through the safetensors collection; the OS
page cache makes the reload a system-RAM read in practice). Restoration is
guaranteed even when the ingest itself fails. See ``aux_offload.py`` for
the mechanism and its limits (no TP-loaded models).

CPU-testable: the ingest loop, lock, encoding and checkpoint policy take the
net/tokenizer through narrow interfaces, so ``tests/test_realtime.py``
exercises them with stubs -- no CUDA, no model.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Sequence

import torch

from .aux_offload import ModelParker
from .fused_ce import DEFAULT_CHUNK, DEFAULT_VOCAB_CHUNK
from .vision import build_mm_batch, mrope_position_ids
from .preference import (dpo_loss, kto_loss, orpo_loss, mismatched_kl_shift,
                         DPO_LOSS_TYPES, KTO_LOSS_TYPES, ORPO_NLL_MODES)

IGNORE_INDEX = -100


class RWLock:
    """
    Readers/writer lock with writer preference. Inference requests hold read;
    ``ingest`` holds write. A waiting writer blocks NEW readers (so a stream of
    requests can't starve training) while in-flight readers drain. Plain
    condition-variable implementation, not reentrant.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    def acquire_read(self, timeout: Optional[float] = None) -> None:
        with self._cond:
            if not self._cond.wait_for(
                    lambda: not self._writer and self._writers_waiting == 0,
                    timeout):
                raise TimeoutError("timed out waiting for read lock")
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            assert self._readers > 0
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self, timeout: Optional[float] = None) -> None:
        with self._cond:
            self._writers_waiting += 1
            try:
                if not self._cond.wait_for(
                        lambda: self._readers == 0 and not self._writer,
                        timeout):
                    raise TimeoutError("timed out waiting for write lock")
            finally:
                self._writers_waiting -= 1
            self._writer = True

    def release_write(self) -> None:
        with self._cond:
            assert self._writer
            self._writer = False
            self._cond.notify_all()

    @contextlib.contextmanager
    def read(self, timeout: Optional[float] = None):
        self.acquire_read(timeout)
        try:
            yield
        finally:
            self.release_read()

    @contextlib.contextmanager
    def write(self, timeout: Optional[float] = None):
        self.acquire_write(timeout)
        try:
            yield
        finally:
            self.release_write()


@dataclass
class RealtimeConfig:
    """
    The "small extra config" for real-time training: LoRA hyperparameters,
    optimization settings and the checkpoint directory. Everything else about
    the model (which model, cache size, devices) belongs to the host
    application's own loading config.
    """

    # -- adapter hyperparameters (fixed at construction) --
    r: int = 16
    alpha: float = 32.0
    target_modules: Optional[list] = None       # None = DEFAULT_TARGET_MODULES.
                                                # Exclude k_proj/v_proj to make
                                                # the KV cache adapter-free (no
                                                # invalidation needed).
    use_rslora: bool = False
    lora_dropout: float = 0.0
    compute_dtype: str = "bfloat16"             # "bfloat16" | "float16" | "float32"
    gradient_checkpointing: bool = True
    head_vocab_chunk: int = DEFAULT_VOCAB_CHUNK # LM-head loss in vocab-column
                                                # tiles of this many columns
                                                # (0 = single-shot head). The
                                                # one-shot path reconstructs
                                                # the full [hidden, vocab]
                                                # head plus two fp32 copies of
                                                # it -- ~12 GB for a 248k-vocab
                                                # 27B, on a card that is ALSO
                                                # holding the served model and
                                                # its KV cache. Tiled, the peak
                                                # is a [hidden, chunk] slice;
                                                # same total dequant work, same
                                                # loss to within fp16 rounding.

    # -- optimization (lr is live-settable afterwards via ``rt.lr``) --
    lr: float = 1e-4                            # constant; no schedule (streams
                                                # have no epochs)
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    batch_size: int = 1
    grad_accum: int = 4
    seq_len: int = 2048                         # per-sample truncation length

    # -- checkpointing --
    checkpoint_dir: Optional[str] = None
    checkpoint_every: int = 0                   # optimizer steps; 0 = manual only
    keep_checkpoints: int = 0                   # prune to newest N; 0 = keep all

    # -- idle VRAM --
    offload_when_idle: bool = True              # park training state (fp32
                                                # masters + Adam moments) in
                                                # system RAM between ingests;
                                                # restored before each ingest.
                                                # Value-exact either way.
    offload_aux_when_training: bool = True      # park attached aux component
                                                # models (vision tower, MTP/
                                                # draft head -- see
                                                # attach_aux_models) out of
                                                # VRAM for the duration of
                                                # each ingest; restored
                                                # value-exact afterwards.

    # -- ingestion --
    add_bos: bool = True                        # ensure exactly one leading BOS
    eot_text: str = ""                          # appended to prompt/response
                                                # samples' response and to every
                                                # preference completion (the
                                                # turn-end token string, if the
                                                # caller's rendering doesn't
                                                # include it)

    # -- preference objectives (DPO pairs / KTO rows -- see ingest) --
    beta: float = 0.1                           # inverse temperature (dpo + kto;
                                                # ORPO's lambda)
    pair_objective: str = "dpo"                 # dpo | orpo: the loss a
                                                # prompt+chosen+rejected pair
                                                # trains (orpo: reference-free)
    dpo_loss: str = "sigmoid"                   # sigmoid | hinge | ipo
    label_smoothing: float = 0.0                # cDPO label-flip prob (sigmoid)
    orpo_nll: str = "full"                      # full | completion: tokens the
                                                # ORPO NLL term supervises
    kto_loss: str = "kto"                       # kto | apo_zero_unpaired
    desirable_weight: float = 1.0               # KTO lambda_D (desirable rows)
    undesirable_weight: float = 1.0             # KTO lambda_U (undesirable rows)

    def __post_init__(self):
        if self.dpo_loss not in DPO_LOSS_TYPES:
            raise ValueError(f"dpo_loss must be one of {DPO_LOSS_TYPES}, "
                             f"got {self.dpo_loss!r}")
        if self.pair_objective not in ("dpo", "orpo"):
            raise ValueError(f"pair_objective must be 'dpo' or 'orpo', "
                             f"got {self.pair_objective!r}")
        if self.orpo_nll not in ORPO_NLL_MODES:
            raise ValueError(f"orpo_nll must be one of {ORPO_NLL_MODES}, "
                             f"got {self.orpo_nll!r}")
        if self.kto_loss not in KTO_LOSS_TYPES:
            raise ValueError(f"kto_loss must be one of {KTO_LOSS_TYPES}, "
                             f"got {self.kto_loss!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "RealtimeConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown RealtimeConfig keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    def torch_compute_dtype(self) -> torch.dtype:
        m = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}
        if self.compute_dtype not in m:
            raise ValueError(f"compute_dtype must be one of {sorted(m)}, "
                             f"got {self.compute_dtype!r}")
        return m[self.compute_dtype]


def _dedup_leading_bos(ids: list, bos: Optional[int]) -> list:
    """Collapse doubled leading BOS (templates that embed a literal BOS on top
    of a tokenizer that auto-prepends one -- same normalization as the offline
    trainer's encode_prompt_response)."""
    if bos is None:
        return ids
    while len(ids) >= 2 and ids[0] == bos and ids[1] == bos:
        ids = ids[1:]
    return ids


def encode_sample_segments(tokenizer, segments: Sequence[tuple], seq_len: int,
                    add_bos: bool) -> dict:
    """
    Tokenize ordered ``(text, supervised)`` segments SEPARATELY and concatenate
    -- the exact-mask-boundary contract shared with the offline trainers
    (``training/chat_turns.py`` / ``chat_jinja.py`` emit this shape). Masked
    segments' labels are ``-100``; supervised segments supervise every token.
    Truncates to ``seq_len``. Returns ``{"input_ids", "labels"}`` (int lists).
    """
    bos = getattr(tokenizer, "bos_token_id", None)
    input_ids: list = []
    labels: list = []
    for text, supervised in segments:
        if not text:
            continue
        ids = tokenizer.encode(
            text, add_bos=False, encode_special_tokens=True)[0].tolist()
        if not input_ids:
            ids = _dedup_leading_bos(ids, bos)
        elif bos is not None and ids and ids[0] == bos:
            # A non-first segment must not re-introduce an auto-prepended BOS.
            ids = ids[1:]
        input_ids += ids
        labels += ids if supervised else [IGNORE_INDEX] * len(ids)
    if add_bos and bos is not None and (not input_ids or input_ids[0] != bos):
        input_ids = [bos] + input_ids
        labels = [IGNORE_INDEX] + labels
    if bos is not None and input_ids and input_ids[0] == bos:
        # A leading BOS is never a training target, even when the tokenizer
        # auto-prepended it inside a supervised segment. (The shifted CE
        # ignores position 0's label anyway; this keeps the mask honest.)
        labels[0] = IGNORE_INDEX
    return {"input_ids": input_ids[:seq_len], "labels": labels[:seq_len]}


def collate(batch: Sequence[dict], pad_id: int):
    """Right-pad a batch of ``{"input_ids", "labels"}`` examples. Returns
    ``(input_ids, labels, attention_mask)`` long tensors."""
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(list(b["input_ids"]) + [pad_id] * pad)
        labels.append(list(b["labels"]) + [IGNORE_INDEX] * pad)
        attn.append([1] * n + [0] * pad)
    return (torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long))


# -- image samples --------------------------------------------------------------

IMAGE_PART_TYPES = ("image", "image_url", "input_image")


def has_image_parts(messages) -> bool:
    """True when any message's content is a content-parts list holding an
    image part (the OpenAI multimodal layout). Mirrors
    ``training/vision_data.py``'s predicate so the coordinator can tell an
    image sample from a text one without importing the script-side module."""
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, (list, tuple)):
            for p in c:
                if isinstance(p, dict) and (p.get("type") or "text") in IMAGE_PART_TYPES:
                    return True
    return False


def image_layout_end(images: Sequence[tuple]) -> int:
    """One past the last token of the last image layout in ``images``
    (``[(start, ImageFeatures)]``); 0 without images."""
    return max((int(s) + len(f.token_ids) for s, f in images), default=0)


# -- preference (DPO / KTO) sample helpers ------------------------------------

def completion_text(value) -> str:
    """A preference completion field -> text. Accepts a plain string or a
    TRL-conversational list of messages (assistant contents joined) -- the
    offline preference trainer's contract."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [(m.get("content") or "").strip() for m in value
                 if (m.get("role") or "").lower() == "assistant"]
        return "\n".join(p for p in parts if p).strip()
    return ""


def coerce_label(value) -> bool:
    """A KTO ``label`` field -> bool. Bools/ints as-is (TRL's format); the
    obvious string spellings are accepted for hand-written JSONL. Anything
    else is an error rather than a silent ``bool(...)`` (``"false"`` would be
    truthy)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "desirable", "good"):
            return True
        if v in ("false", "0", "no", "undesirable", "bad"):
            return False
    raise ValueError(f"KTO label must be a bool (desirable = True), got {value!r}")


def encode_completion(tokenizer, text: str, eot_text: str) -> list:
    """Tokenize a bare completion (+ turn-end string) that will follow a
    separately encoded prompt: no BOS, any auto-prepended one stripped (the
    same normalization ``encode_sample_segments`` applies to a non-first
    segment)."""
    ids = tokenizer.encode(
        text + eot_text, add_bos=False, encode_special_tokens=True)[0].tolist()
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and ids and ids[0] == bos:
        ids = ids[1:]
    return ids


def fit_completion(prompt_ids: list, comp_ids: list, seq_len: int) -> Optional[list]:
    """Truncate a (prompt, completion) pair to ``seq_len``, completion tail
    first (the offline trainers' rule). Returns the truncated completion, or
    ``None`` when nothing scorable survives (prompt alone fills the window)."""
    room = seq_len - len(prompt_ids)
    if room <= 0:
        return None
    out = list(comp_ids[:room])
    return out if out else None


def preference_rows(rows: Sequence[tuple]) -> list:
    """``(prompt_ids, completion_ids)`` pairs -> ``{"input_ids", "labels"}``
    examples with the prompt masked (completion-only scoring), ready for
    :func:`collate`."""
    return [{"input_ids": list(p) + list(c),
             "labels": [IGNORE_INDEX] * len(p) + list(c)} for p, c in rows]


def checkpoint_name(step: int, now: Optional[time.struct_time] = None) -> str:
    """Timestamped checkpoint directory name. The timestamp leads (sorts
    chronologically, which is also step order); the step disambiguates two
    checkpoints inside one second and makes the name self-describing."""
    ts = time.strftime("%Y%m%d-%H%M%S", now if now is not None else time.localtime())
    return f"ckpt-{ts}-step{step}"


def list_realtime_checkpoints(directory: str) -> list:
    """Checkpoint subdirectories of ``directory``, oldest first (name order ==
    time order by construction)."""
    if not os.path.isdir(directory):
        return []
    return sorted(d for d in os.listdir(directory)
                  if d.startswith("ckpt-")
                  and os.path.isdir(os.path.join(directory, d)))


class RealtimeQLoRA:
    """
    Coordinator for serve-and-train on one loaded exllamav3 model.

    Construct next to a loaded ``Model`` (+ its ``Tokenizer``)::

        rt = RealtimeQLoRA(model, tokenizer, RealtimeConfig(
            r=16, alpha=32, lr=1e-4,
            checkpoint_dir="out/realtime", checkpoint_every=50))
        rt.attach_generator(generator)      # cache nuked on adapter updates

    Serve inference under the read lock::

        with rt.inference():
            out = generator.generate(...)

    Train whenever samples arrive (blocks until depleted; run in a worker
    thread on a server)::

        stats = rt.ingest([
            {"prompt": "<rendered chat prompt>", "response": "the reply"},
            {"text": "raw document to absorb"},
            {"prompt": "<rendered chat prompt>",            # one DPO pair
             "chosen": "the better reply", "rejected": "the worse reply"},
            {"prompt": "<rendered chat prompt>",            # one KTO row
             "completion": "a reply", "label": False},
        ])

    The trained adapter is live in generation the moment ``ingest`` returns.
    Adapter/optimizer state persists across ``ingest`` calls; ``checkpoint()``
    / automatic ``checkpoint_every`` snapshots are ordinary PEFT adapter dirs,
    so any of them also loads in the offline trainers, ``LoRA.from_directory``
    or PEFT itself, and ``RealtimeQLoRA(..., adapter_dir=...)`` resumes from
    one (including optimizer state).

    ``net`` / dependency injection: pass a prebuilt ``NativeLlamaQLoRA`` (or a
    stub implementing its small training surface -- see tests) to skip the
    internal build; ``model`` may then be ``None``.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[RealtimeConfig] = None,
        adapter_dir: Optional[str] = None,
        net=None,
        render_segments: Optional[Callable[[dict], Sequence[tuple]]] = None,
        base_model_name_or_path: Optional[str] = None,
        render_prompt: Optional[Callable[[dict], Sequence[tuple]]] = None,
        vision=None,
    ):
        self.config = config or RealtimeConfig()
        self.tokenizer = tokenizer
        self.render_segments = render_segments
        # Image service for image+text samples (see the module docstring for
        # its surface; training/vision_data.py's VisionData implements it).
        # None = text-only: image-bearing samples are refused, not silently
        # trained against empty image slots.
        self.vision = vision
        # Preference samples whose ``prompt`` is a message list: sample ->
        # ``[(text, supervised)]`` segments of the prompt rendered WITH the
        # generation prompt (supervision flags are ignored -- a prompt is
        # always masked). See training/realtime_chat.py for the wiring.
        self.render_prompt = render_prompt
        self.base_model_name_or_path = base_model_name_or_path

        if net is None:
            from .native_llama import NativeLlamaQLoRA
            net = NativeLlamaQLoRA(
                model,
                r=self.config.r,
                alpha=self.config.alpha,
                target_modules=self.config.target_modules,
                use_rslora=self.config.use_rslora,
                lora_dropout=self.config.lora_dropout,
                compute_dtype=self.config.torch_compute_dtype(),
                gradient_checkpointing=self.config.gradient_checkpointing,
                head_vocab_chunk=self.config.head_vocab_chunk,
            )
        self.net = net
        self.net.train()
        self.opt = torch.optim.AdamW(
            self.net.param_groups(self.config.weight_decay), lr=self.config.lr)

        self.step = 0                       # optimizer steps, across all ingests
        self.samples_seen = 0
        self._lock = RWLock()
        self._update_callbacks: list = []
        self._generators: list = []
        self._synced = False                # adapter pushed to inference slots yet?
        self._idle_offloaded = False        # training state parked in sysmem?
        self._opt_state_homes: list = []    # (state dict, key, device) to undo
        self._aux_parkers: list = []        # ModelParkers for aux component
                                            # models parked during ingests

        pad = getattr(tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(tokenizer, "eos_token_id", None)
        self.pad_id = pad if pad is not None else 0

        if adapter_dir:
            self.load(adapter_dir)

        # Nothing trains until the first ingest -- park the training state now
        # so a freshly loaded server starts at its serving footprint. (No-op if
        # load() above already parked it.)
        self._offload_idle()

    # -- learning rate (constant, externally controllable) -------------------

    @property
    def lr(self) -> float:
        return self.opt.param_groups[0]["lr"]

    @lr.setter
    def lr(self, value: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = float(value)

    # -- inference / training coordination -----------------------------------

    def inference(self, timeout: Optional[float] = None):
        """Read-lock context manager -- wrap every generator drive in this."""
        return self._lock.read(timeout)

    def add_update_callback(self, fn: Callable[[], None]) -> None:
        """Register a zero-arg callable fired after every adapter push to the
        inference forward (i.e. when cached state derived from the old adapter
        went stale)."""
        self._update_callbacks.append(fn)

    @staticmethod
    def invalidate_generator_cache(generator) -> None:
        """Drop every cached state the generator could hand a new request:
        the VRAM page table, the CPU page tier behind it and the recurrent-state
        checkpoints. All of it was computed through the previous adapter."""
        pagetable = generator.pagetable
        pagetable.reset_page_table()
        recurrent = getattr(generator, "recurrent_cache", None)
        if recurrent is not None:
            recurrent.clear()
        tier = getattr(pagetable, "cpu_tier", None)
        if tier is not None:
            tier.clear()

    def attach_generator(self, generator) -> None:
        """Wire a generator in: its cached state (page table, CPU page tier,
        recurrent checkpoints) is invalidated on every adapter update (the
        default cache-staleness answer), and ``ingest`` refuses to start while
        it still has jobs in flight (belt and suspenders under the write lock).
        A generator with a disk cache tier is refused: pages written there
        outlive the process and are keyed by model, not adapter, so a later
        run would restore KV computed through a stale adapter."""
        if getattr(generator, "disk_page_cache", None) is not None:
            raise ValueError(
                "RealtimeQLoRA cannot serve from a generator with a disk cache tier: "
                "cached pages persist across processes and are not keyed by adapter, "
                "so an adapter update cannot invalidate them. Build the generator "
                "without disk_cache_dir."
            )
        self._generators.append(generator)
        self.add_update_callback(lambda: self.invalidate_generator_cache(generator))

    def attach_aux_models(self, *models) -> None:
        """Register serving-only component models -- a vision tower, an MTP/
        draft head -- to be parked out of VRAM for the duration of every
        ingest (``offload_aux_when_training``). Training never touches these
        (the adapter targets live in the text trunk), so during a training
        burst their VRAM is better spent on activations and optimizer state.
        ``None`` entries are ignored, so optional models can be passed
        unconditionally::

            rt.attach_aux_models(vision_model, draft_model)

        Only register models the adapter does not touch, and never the
        trained model itself. Parked models are restored value-exact (same
        devices, same weights) before ``ingest`` returns, including when the
        ingest fails. A draft model's KV cache is freed and reallocated
        empty across the cycle -- benign, since draft tokens are always
        verified by the main model and the default flow resets the page
        table after every ingest anyway."""
        for model in models:
            if model is None:
                continue
            self._aux_parkers.append(ModelParker(model))

    def _wait_generators_idle(self, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        for gen in self._generators:
            while gen.num_remaining_jobs():
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "ingest: generator still has jobs in flight -- drive "
                        "generation inside `with rt.inference():` so the write "
                        "lock can drain it")
                time.sleep(0.01)

    def sync_to_inference(self) -> None:
        """Push current adapter weights into the native runtime LoRA slots and
        fire the update callbacks (cache invalidation). Called automatically at
        the end of every ``ingest``; public for manual use after ``load``-like
        surgery."""
        self.net.apply_to_native()
        self._synced = True
        for fn in self._update_callbacks:
            fn()

    def unload_from_inference(self) -> None:
        """Remove the adapter from generation (revert to the base model). The
        training state is untouched; the next ``ingest`` re-applies."""
        if self._synced:
            self.net.remove_from_native()
            self._synced = False
            for fn in self._update_callbacks:
                fn()

    # -- idle offload ---------------------------------------------------------

    def _offload_idle(self) -> None:
        """Park the persistent training state in system memory while the model
        only serves: the net's fp32 masters/PiSSA offsets (via its
        ``offload_training_state``, if it has one -- stubs need not) plus the
        optimizer's device state (Adam moments), then flush the CUDA caching
        allocator so the freed VRAM is actually returned. Value-exact."""
        if not self.config.offload_when_idle or self._idle_offloaded:
            return
        fn = getattr(self.net, "offload_training_state", None)
        moved = bool(fn()) if fn is not None else False
        homes: list = []
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v) and v.device.type != "cpu":
                    homes.append((state, k, v.device))
                    state[k] = v.to("cpu")
        self._opt_state_homes = homes
        self._idle_offloaded = True
        if (moved or homes) and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _restore_idle(self) -> None:
        """Undo :meth:`_offload_idle` before training touches the state. The
        net restores first so optimizer state lands back beside its (now
        on-device) parameters."""
        if not self._idle_offloaded:
            return
        fn = getattr(self.net, "restore_training_state", None)
        if fn is not None:
            fn()
        for state, k, dev in self._opt_state_homes:
            state[k] = state[k].to(dev)
        self._opt_state_homes = []
        self._idle_offloaded = False

    def _park_aux(self) -> None:
        """Park attached aux component models (vision/MTP) out of VRAM for
        the duration of an ingest. No-op when disabled or nothing attached."""
        if not self.config.offload_aux_when_training:
            return
        for parker in self._aux_parkers:
            parker.park()

    def _unpark_aux(self) -> None:
        """Restore any parked aux component models. Idempotent; called on
        both the success and failure paths out of an ingest."""
        for parker in self._aux_parkers:
            parker.unpark()

    # -- ingestion ------------------------------------------------------------

    def ingest(
        self,
        samples: Sequence[dict],
        lr: Optional[float] = None,
        lock_timeout: Optional[float] = None,
        image_base_dir: Optional[str] = None,
    ) -> dict:
        """
        Train through ``samples`` until depleted, then push the adapter into
        the inference forward and return. Blocks while running (the writer
        side of the lock); in-flight inference drains first, new inference
        waits.

        Samples are consumed IN ORDER: consecutive groups of ``batch_size``
        same-kind samples (SFT rows / DPO pairs / KTO rows) form
        micro-batches, ``grad_accum`` micro-batches form one optimizer step
        (SFT windows token-weighted, preference windows example-weighted, so
        the step gradient equals one big batch); a kind change or a final
        partial window still steps. ``lr`` overrides the constant learning
        rate for this call onward (it sticks -- "externally controllable"
        means the last setting wins). ``image_base_dir`` resolves relative
        image ``path`` parts (e.g. the directory of the JSONL file the
        samples were read from).

        Samples are encoded under the write lock, before the aux models are
        parked: an image sample runs the (resident) vision tower there, so
        its features are in hand before the tower leaves VRAM for the
        training forwards. Image samples are counted in ``image_samples`` /
        ``image_tokens``; one whose image layout does not fit ``seq_len`` is
        skipped (never cut through an image).

        Returns a stats dict: steps taken, samples/tokens consumed, skipped
        rows, mean SFT loss (token-weighted; ``None`` without SFT samples),
        per-objective preference metrics (``"dpo"``: pairs / loss / reward
        accuracy / reward margin, ``"kto"``: samples / loss / kl / mean
        desirable and undesirable rewards; ``None`` when that kind was not
        ingested), duration, lr, checkpoints written.
        """
        cfg = self.config
        if lr is not None:
            self.lr = lr

        t0 = time.time()
        stats_steps = 0
        sup_total = 0                       # supervised tokens, all kinds
        tok_total = 0
        sft_n = 0
        sft_sup = 0                         # SFT supervised tokens (loss denom)
        sft_loss_weighted = 0.0
        pref = {"dpo": {"n": 0, "loss": 0.0, "acc": 0.0, "margin": 0.0},
                "kto": {"n": 0, "loss": 0.0, "kl": 0.0,
                        "n_d": 0, "reward_d": 0.0, "n_u": 0, "reward_u": 0.0}}
        ckpts: list = []

        with self._lock.write(lock_timeout):
            self._wait_generators_idle()
            # Encode under the lock, before the aux models are parked: an
            # image sample runs the vision tower here (serving has drained,
            # so the tower is idle and resident), and its features ride on
            # the example from now on -- the tower can leave VRAM for the
            # training forwards without anything needing it again.
            encoded = [self._encode(s, image_base_dir) for s in samples]
            examples = [e for e in encoded
                        if e is not None and (e["kind"] != "sft" or e["input_ids"])]
            skipped = len(encoded) - len(examples)
            img_n = sum(1 for e in examples if e.get("images"))
            img_tok = sum(f.n_tokens for e in examples
                          for _, f in e.get("images", ()))
            if not examples:
                return {"steps": 0, "samples": 0, "skipped": skipped,
                        "sft_samples": 0, "supervised_tokens": 0,
                        "total_tokens": 0, "mean_loss": None, "dpo": None,
                        "kto": None, "image_samples": 0, "image_tokens": 0,
                        "duration_s": time.time() - t0, "lr": self.lr,
                        "checkpoints": []}

            # Runs of consecutive same-kind samples: each kind takes a
            # different forward and loss, so a kind change closes the current
            # micro-batch AND accumulation window. Order is still preserved.
            runs: list = []
            for e in examples:
                if runs and runs[-1][0] == e["kind"]:
                    runs[-1][1].append(e)
                else:
                    runs.append((e["kind"], [e]))

            # Aux component models (vision tower, MTP/draft head) leave VRAM
            # first, so the training state and activations land in the space
            # they vacate; they come back last, into the space the re-parked
            # training state frees -- and they come back even when the ingest
            # fails, so serving resumes whole.
            self._park_aux()
            try:
                self._restore_idle()
                self.net.train()

                for kind, run in runs:
                    micro = [run[i:i + cfg.batch_size]
                             for i in range(0, len(run), cfg.batch_size)]
                    windows = [micro[i:i + cfg.grad_accum]
                               for i in range(0, len(micro), cfg.grad_accum)]

                    for window in windows:
                        self.opt.zero_grad(set_to_none=True)
                        n_win = sum(len(mb) for mb in window)
                        if kind == "sft":
                            w_loss, w_sup, w_tok = self._train_window_sft(window)
                            sft_n += n_win
                            sft_sup += w_sup
                            sft_loss_weighted += w_loss * w_sup
                        else:
                            w_sup, w_tok = self._train_window_pref(
                                kind, window, pref[kind])
                        sup_total += w_sup
                        tok_total += w_tok
                        torch.nn.utils.clip_grad_norm_(
                            self.net.trainable_parameters(),
                            cfg.max_grad_norm or float("inf"))
                        self.opt.step()
                        self.step += 1
                        stats_steps += 1
                        self.samples_seen += n_win

                        if (cfg.checkpoint_every and cfg.checkpoint_dir
                                and self.step % cfg.checkpoint_every == 0):
                            ckpts.append(self.checkpoint())

                self.opt.zero_grad(set_to_none=True)
                self.sync_to_inference()
                self._offload_idle()
            finally:
                self._unpark_aux()

        d, k = pref["dpo"], pref["kto"]
        return {
            "steps": stats_steps,
            "samples": len(examples),
            "skipped": skipped,
            "sft_samples": sft_n,
            "supervised_tokens": sup_total,
            "total_tokens": tok_total,
            "mean_loss": (sft_loss_weighted / sft_sup) if sft_sup else None,
            "dpo": None if not d["n"] else {
                "pairs": d["n"],
                "loss": d["loss"] / d["n"],
                "acc": d["acc"] / d["n"],
                "margin": d["margin"] / d["n"],
            },
            "kto": None if not k["n"] else {
                "samples": k["n"],
                "loss": k["loss"] / k["n"],
                "kl": k["kl"] / k["n"],
                "reward_d": (k["reward_d"] / k["n_d"]) if k["n_d"] else None,
                "reward_u": (k["reward_u"] / k["n_u"]) if k["n_u"] else None,
            },
            "image_samples": img_n,
            "image_tokens": img_tok,
            "duration_s": time.time() - t0,
            "lr": self.lr,
            "checkpoints": ckpts,
        }

    # -- per-window training (gradients accumulate; the caller steps) --------

    def _train_window_sft(self, window: Sequence[list]) -> tuple:
        """Backward through one accumulation window of SFT micro-batches.
        Returns ``(window_loss, supervised_tokens, total_tokens)``."""
        batches = [collate(mb, self.pad_id) for mb in window]
        extras = [self._mm_kwargs([e.get("images", ()) for e in mb],
                                  [len(e["input_ids"]) for e in mb],
                                  ids.shape[1])
                  for mb, (ids, _, _) in zip(window, batches)]
        # Token-weighted accumulation (the offline trainer's --ga-loss token):
        # weight each micro-batch's mean loss by its share of the window's
        # supervised tokens, counted on the SHIFTED labels to match the CE
        # denominator.
        n_sups = [int((lb[:, 1:] != IGNORE_INDEX).sum()) for _, lb, _ in batches]
        total_sup = max(sum(n_sups), 1)
        window_loss = 0.0
        tok = 0
        for (input_ids, labels, attn), n_sup, extra in zip(batches, n_sups, extras):
            loss = self.net.compute_loss(
                input_ids, labels, attention_mask=attn, chunk=DEFAULT_CHUNK,
                **extra)
            w_i = n_sup / total_sup
            (loss * w_i).backward()
            window_loss += loss.item() * w_i
            tok += int(attn.sum())
        return window_loss, total_sup, tok

    def _train_window_pref(self, kind: str, window: Sequence[list],
                           acc: dict) -> tuple:
        """Backward through one accumulation window of DPO pairs or KTO rows.
        Example-weighted: each micro-batch's mean loss is scaled by its share
        of the window's examples (TRL's batch-mean reduction over the
        effective batch). Metric SUMS land in ``acc`` (the caller divides).
        Returns ``(supervised_tokens, total_tokens)``."""
        n_win = sum(len(mb) for mb in window)
        sup = tok = 0
        for mb in window:
            if kind == "dpo" and self.config.pair_objective == "orpo":
                loss, m = self._orpo_batch(mb)
            elif kind == "dpo":
                loss, m = self._dpo_batch(mb)
            else:
                loss, m = self._kto_batch(mb)
            w_i = len(mb) / n_win
            (loss * w_i).backward()
            acc["n"] += len(mb)
            acc["loss"] += loss.item() * len(mb)
            for key, val in m.items():
                if key in acc:
                    acc[key] += val
            sup += m["sup"]
            tok += m["tot"]
        return sup, tok

    def _mm_kwargs(self, images: Sequence, lengths: Sequence[int],
                   padded_len: int) -> dict:
        """The image splice for one collated micro-batch: ``{}`` on a
        text-only batch, else ``{"mm": ...}`` (``vision.build_mm_batch`` over
        each row's ``[(start, ImageFeatures)]``) plus, on an mRoPE tower,
        ``{"position_ids": [3, b, t]}`` -- the 3-D positions of every row
        computed at the row's own length (a preference row's completion
        follows its prompt's images, so the positions are built per
        assembled row, not per sample), text-only rows in the batch getting
        the sequential positions 1-D RoPE reduces to."""
        if not any(images):
            return {}
        rows = [[(int(s), f) for s, f in imgs] for imgs in images]
        kw = {"mm": build_mm_batch(rows, padded_len,
                                   self.config.torch_compute_dtype())}
        if getattr(self.vision, "mrope", False):
            pos = []
            for imgs, n in zip(rows, lengths):
                spans = [(s + f.slot_offsets[0], f.n_tokens, f.grid_thw)
                         for s, f in imgs]
                merge = imgs[0][1].merge_size if imgs else 1
                p = mrope_position_ids(n, spans, merge or 1)        # [3, n]
                if padded_len > n:
                    p = torch.cat([p, torch.zeros(3, padded_len - n,
                                                  dtype=p.dtype)], 1)
                pos.append(p)
            kw["position_ids"] = torch.stack(pos, 1).contiguous()  # [3, b, t]
        return kw

    def _logps(self, input_ids, labels, attn, reference: bool = False,
               per_token: bool = False, extra: Optional[dict] = None):
        """Per-sequence summed completion logps through the net -- as the
        adapted policy, or (``reference=True``, no grad) as the frozen base
        with the adapters disabled: the reference model for DPO/KTO without
        a second model copy. ``per_token=True`` asks for the ``(token_logps,
        mask)`` form instead (ORPO). ``extra`` carries the image splice /
        3-D positions of an image batch (:meth:`_mm_kwargs`)."""
        kw = dict(attention_mask=attn, chunk=DEFAULT_CHUNK, **(extra or {}))
        if per_token:
            kw["return_per_token"] = True
        if not reference:
            return self.net.compute_logps(input_ids, labels, **kw)
        with torch.no_grad(), self.net.adapters_disabled():
            return self.net.compute_logps(input_ids, labels, **kw)

    def _dpo_batch(self, batch: Sequence[dict]) -> tuple:
        """Loss + metric sums for one micro-batch of DPO pairs: the 2*b rows
        (chosen block, then rejected block) share one policy forward and one
        no-grad reference forward."""
        cfg = self.config
        rows = ([(e["prompt_ids"], e["chosen_ids"]) for e in batch]
                + [(e["prompt_ids"], e["rejected_ids"]) for e in batch])
        input_ids, labels, attn = collate(preference_rows(rows), self.pad_id)
        extra = self._pref_mm_kwargs(batch + batch, rows, input_ids.shape[1])
        pol, counts = self._logps(input_ids, labels, attn, extra=extra)
        ref, _ = self._logps(input_ids, labels, attn, reference=True, extra=extra)
        b = len(batch)
        losses, cr, rr = dpo_loss(
            pol[:b], pol[b:], ref[:b], ref[b:],
            beta=cfg.beta, label_smoothing=cfg.label_smoothing,
            loss_type=cfg.dpo_loss,
            chosen_counts=counts[:b], rejected_counts=counts[b:])
        metrics = {
            "margin": (cr - rr).sum().item(),
            "acc": (cr > rr).float().sum().item(),
            "sup": int((labels[:, 1:] != IGNORE_INDEX).sum()),
            "tot": int(attn.sum()),
        }
        return losses.mean(), metrics

    def _orpo_batch(self, batch: Sequence[dict]) -> tuple:
        """Loss + metric sums for one micro-batch of pairs under ORPO: one
        policy forward over the 2*b rows, no reference. Loss = NLL(chosen) +
        odds-ratio term; ``config.orpo_nll`` decides whether the NLL spans the
        prompt (``full``: one forward with every non-pad token supervised,
        per-token logps re-masked for the completion-only odds) or only the
        chosen completion tokens."""
        cfg = self.config
        rows = ([(e["prompt_ids"], e["chosen_ids"]) for e in batch]
                + [(e["prompt_ids"], e["rejected_ids"]) for e in batch])
        input_ids, labels, attn = collate(preference_rows(rows), self.pad_id)
        extra = self._pref_mm_kwargs(batch + batch, rows, input_ids.shape[1])
        b = len(batch)
        cmask = labels[:, 1:] != IGNORE_INDEX
        if cfg.orpo_nll == "full":
            # Image tokens are never supervised: the full-sequence NLL skips
            # the slot positions the way the SFT mask does.
            full_labels = input_ids.masked_fill(attn == 0, IGNORE_INDEX)
            if extra:
                full_labels = full_labels.masked_fill(extra["mm"]["index"] >= 0,
                                                      IGNORE_INDEX)
            tok_logps, fmask = self._logps(input_ids, full_labels, attn,
                                           per_token=True, extra=extra)
            cmask = cmask.to(tok_logps.device)
            pol = (tok_logps * cmask).sum(dim=-1)
            counts = cmask.sum(dim=-1)
            fm = fmask[:b]
            nll = -(tok_logps[:b] * fm).sum() / fm.sum().clamp(min=1)
        else:
            pol, counts = self._logps(input_ids, labels, attn, extra=extra)
            nll = -pol[:b].sum() / counts[:b].sum().clamp(min=1)
        losses, cr, rr, _ = orpo_loss(pol[:b], pol[b:], counts[:b], counts[b:],
                                      beta=cfg.beta)
        metrics = {
            "margin": (cr - rr).sum().item(),
            "acc": (cr > rr).float().sum().item(),
            "sup": int(cmask.sum()),
            "tot": int(attn.sum()),
        }
        return nll + losses.mean(), metrics

    def _kto_batch(self, batch: Sequence[dict]) -> tuple:
        """Loss + metric sums for one micro-batch of KTO rows. For the ``kto``
        loss with ``len(batch) >= 2``, two extra no-grad forwards estimate the
        KL reference point on mismatched prompt/completion pairs (TRL's
        +1-offset rotation); a singleton batch trains with KL = 0."""
        cfg = self.config
        rows = [(e["prompt_ids"], e["completion_ids"]) for e in batch]
        input_ids, labels, attn = collate(preference_rows(rows), self.pad_id)
        extra = self._pref_mm_kwargs(batch, rows, input_ids.shape[1])
        pol, _ = self._logps(input_ids, labels, attn, extra=extra)
        ref, _ = self._logps(input_ids, labels, attn, reference=True, extra=extra)

        pol_kl = ref_kl = None
        if cfg.kto_loss == "kto" and len(batch) > 1:
            kl_rows, kl_ex = [], []
            for i, j in enumerate(mismatched_kl_shift(len(batch))):
                comp = fit_completion(batch[i]["prompt_ids"],
                                      batch[j]["completion_ids"], cfg.seq_len)
                if comp:
                    kl_rows.append((batch[i]["prompt_ids"], comp))
                    kl_ex.append(batch[i])        # the prompt's images ride along
            if kl_rows:
                kl_ids, kl_labels, kl_attn = collate(preference_rows(kl_rows),
                                                     self.pad_id)
                kl_extra = self._pref_mm_kwargs(kl_ex, kl_rows, kl_ids.shape[1])
                with torch.no_grad():
                    pol_kl, _ = self._logps(kl_ids, kl_labels, kl_attn,
                                            extra=kl_extra)
                ref_kl, _ = self._logps(kl_ids, kl_labels, kl_attn,
                                        reference=True, extra=kl_extra)

        des = torch.tensor([e["label"] for e in batch], dtype=torch.bool,
                           device=pol.device)
        losses, cr, rr, kl = kto_loss(
            pol[des], pol[~des], pol_kl, ref[des], ref[~des], ref_kl,
            beta=cfg.beta, desirable_weight=cfg.desirable_weight,
            undesirable_weight=cfg.undesirable_weight, loss_type=cfg.kto_loss)
        metrics = {
            "kl": kl.item() * len(batch),
            "n_d": int(cr.numel()),
            "reward_d": cr.sum().item(),
            "n_u": int(rr.numel()),
            "reward_u": rr.sum().item(),
            "sup": int((labels[:, 1:] != IGNORE_INDEX).sum()),
            "tot": int(attn.sum()),
        }
        return losses.mean(), metrics

    # -- sample encoding -------------------------------------------------------

    def _pref_mm_kwargs(self, examples: Sequence[dict], rows: Sequence[tuple],
                        padded_len: int) -> dict:
        """:meth:`_mm_kwargs` for assembled preference rows: row ``i`` is
        ``examples[i]``'s prompt (whose images it carries) followed by
        ``rows[i]``'s completion."""
        return self._mm_kwargs([e.get("images", ()) for e in examples],
                               [len(p) + len(c) for p, c in rows], padded_len)

    def _encode_image_segments(self, sample: dict, messages: list,
                               base_dir: Optional[str], render: Callable):
        """Encode a conversation that carries image parts: flatten the parts
        to sentinels (``vision.flatten``), render through the chat template
        exactly like a text conversation (the sentinels pass through as
        text), then tokenize around the sentinels with the arch's image
        token layout in their place and the frozen features attached
        (``vision.encode_segments``). Returns ``(input_ids, labels, images)``
        with ``images = [(start, ImageFeatures)]`` -- BOS normalized as
        :func:`encode_sample_segments` does, so image starts shift with it."""
        if self.vision is None:
            raise ValueError(
                "sample carries image content parts but no vision service is "
                "wired (RealtimeQLoRA(vision=...)); refusing to train it "
                "text-only -- the template would render empty image slots and "
                "the adapter would learn answers conditioned on nothing")
        flat, refs = self.vision.flatten(messages, base_dir=base_dir)
        segments = render(flat)
        ids, labels, images, feats = self.vision.encode_segments(segments, refs)
        ids, labels = list(ids), list(labels)
        images = [(int(s), feats[k]) for s, k in images]
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if self.config.add_bos and bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
            labels = [IGNORE_INDEX] + labels
            images = [(s + 1, f) for s, f in images]
        if bos is not None and ids and ids[0] == bos:
            labels[0] = IGNORE_INDEX
        return ids, labels, images

    def _encode(self, sample: dict, base_dir: Optional[str] = None) -> Optional[dict]:
        """One ingest sample -> a kind-tagged example: ``{"kind": "sft",
        input_ids, labels}``, ``{"kind": "dpo", prompt_ids, chosen_ids,
        rejected_ids}`` or ``{"kind": "kto", prompt_ids, completion_ids,
        label}``, each plus ``images = [(start, ImageFeatures)]`` when the
        sample (its prompt, for a preference sample) carried images.
        ``None`` for a preference row whose prompt alone fills ``seq_len``
        or an image sample whose image layout does not fit it (skipped,
        counted by ``ingest``); malformed samples raise."""
        cfg = self.config
        if "input_ids" in sample:
            ids = list(sample["input_ids"])[:cfg.seq_len]
            labels = list(sample.get("labels", ids))[:cfg.seq_len]
            if len(labels) != len(ids):
                raise ValueError("labels length must match input_ids")
            return {"kind": "sft", "input_ids": ids, "labels": labels}
        if "prompt_ids" in sample and "chosen_ids" in sample and "rejected_ids" in sample:
            return self._fit_pair(list(sample["prompt_ids"]),
                                  list(sample["chosen_ids"]),
                                  list(sample["rejected_ids"]))
        if "prompt_ids" in sample and "completion_ids" in sample and "label" in sample:
            return self._fit_kto(list(sample["prompt_ids"]),
                                 list(sample["completion_ids"]),
                                 coerce_label(sample["label"]))
        if "messages" in sample:
            if self.render_segments is None:
                raise ValueError(
                    "messages samples need a render_segments callable (see "
                    "training/realtime_chat.py for the chat_jinja wiring); or "
                    "render the template yourself and send prompt/response")
            if has_image_parts(sample["messages"]):
                ids, labels, images = self._encode_image_segments(
                    sample, sample["messages"], base_dir,
                    lambda flat: self.render_segments({**sample, "messages": flat}))
                if len(ids) > cfg.seq_len:
                    if image_layout_end(images) > cfg.seq_len:
                        return None             # never cut through an image
                    ids, labels = ids[:cfg.seq_len], labels[:cfg.seq_len]
                return {"kind": "sft", "input_ids": ids, "labels": labels,
                        "images": images}
            segments = self.render_segments(sample)
            ex = encode_sample_segments(self.tokenizer, segments, cfg.seq_len,
                                        cfg.add_bos)
            return {"kind": "sft", **ex}
        if "prompt" in sample and "chosen" in sample and "rejected" in sample:
            chosen = completion_text(sample["chosen"])
            rejected = completion_text(sample["rejected"])
            if not chosen or not rejected:
                raise ValueError("a DPO sample needs non-empty chosen and "
                                 "rejected completions")
            prompt_ids, images = self._encode_prompt_ids(sample, base_dir)
            return self._fit_pair(
                prompt_ids,
                encode_completion(self.tokenizer, chosen, cfg.eot_text),
                encode_completion(self.tokenizer, rejected, cfg.eot_text),
                images)
        if "prompt" in sample and "completion" in sample and "label" in sample:
            completion = completion_text(sample["completion"])
            if not completion:
                raise ValueError("a KTO sample needs a non-empty completion")
            label = coerce_label(sample["label"])
            prompt_ids, images = self._encode_prompt_ids(sample, base_dir)
            return self._fit_kto(
                prompt_ids,
                encode_completion(self.tokenizer, completion, cfg.eot_text),
                label, images)
        if "prompt" in sample and "response" in sample:
            segments = [(sample["prompt"], False),
                        (sample["response"] + cfg.eot_text, True)]
            ex = encode_sample_segments(self.tokenizer, segments, cfg.seq_len,
                                        cfg.add_bos)
            return {"kind": "sft", **ex}
        if "text" in sample:
            ex = encode_sample_segments(self.tokenizer, [(sample["text"], True)],
                                        cfg.seq_len, cfg.add_bos)
            return {"kind": "sft", **ex}
        raise ValueError(
            f"unrecognized sample shape {sorted(sample)} -- expected "
            f"input_ids[/labels], messages, prompt+response, text, "
            f"prompt+chosen+rejected (DPO), prompt+completion+label (KTO), or "
            f"their pre-tokenized prompt_ids forms")

    def _encode_prompt_ids(self, sample: dict,
                           base_dir: Optional[str] = None) -> tuple:
        """A preference sample's ``prompt`` -> ``(masked prompt ids, images)``.
        A string is an already-rendered prompt (ending with the
        assistant-turn opener); a message list goes through
        ``render_prompt`` -- with its image parts encoded through the vision
        service when it carries any (``images = [(start, ImageFeatures)]``,
        empty for a text prompt)."""
        prompt = sample["prompt"]
        if isinstance(prompt, str):
            segments = [(prompt, False)]
        elif isinstance(prompt, list):
            if self.render_prompt is None:
                raise ValueError(
                    "a conversational preference prompt (message list) needs "
                    "a render_prompt callable (see training/realtime_chat.py "
                    "for the chat_jinja wiring); or render the template "
                    "yourself and send the prompt as a string")
            if has_image_parts(prompt):
                ids, _, images = self._encode_image_segments(
                    sample, prompt, base_dir,
                    lambda flat: [(t, False) for t, _ in
                                  self.render_prompt({**sample, "prompt": flat})])
                # A prompt longer than seq_len leaves no room for a
                # completion: fit_completion skips the row (so an image is
                # never cut -- the truncated ids are never trained on).
                return ids[:self.config.seq_len], images
            segments = [(t, False) for t, _ in self.render_prompt(sample)]
        else:
            raise ValueError("preference prompt must be a rendered string or "
                             "a list of messages")
        ex = encode_sample_segments(self.tokenizer, segments,
                                    self.config.seq_len, self.config.add_bos)
        return ex["input_ids"], []

    def _fit_pair(self, prompt_ids: list, chosen_ids: list,
                  rejected_ids: list, images: Sequence = ()) -> Optional[dict]:
        seq_len = self.config.seq_len
        chosen_ids = fit_completion(prompt_ids, chosen_ids, seq_len)
        rejected_ids = fit_completion(prompt_ids, rejected_ids, seq_len)
        if chosen_ids is None or rejected_ids is None:
            return None
        ex = {"kind": "dpo", "prompt_ids": prompt_ids,
              "chosen_ids": chosen_ids, "rejected_ids": rejected_ids}
        if images:
            ex["images"] = list(images)
        return ex

    def _fit_kto(self, prompt_ids: list, completion_ids: list,
                 label: bool, images: Sequence = ()) -> Optional[dict]:
        completion_ids = fit_completion(prompt_ids, completion_ids,
                                        self.config.seq_len)
        if completion_ids is None:
            return None
        ex = {"kind": "kto", "prompt_ids": prompt_ids,
              "completion_ids": completion_ids, "label": label}
        if images:
            ex["images"] = list(images)
        return ex

    # -- checkpointing ---------------------------------------------------------

    def checkpoint(self, directory: Optional[str] = None) -> str:
        """Write a timestamped adapter checkpoint (PEFT dir + optimizer state
        + meta) under ``directory`` (default ``config.checkpoint_dir``) and
        prune to ``keep_checkpoints``. Returns the checkpoint path."""
        root = directory or self.config.checkpoint_dir
        if not root:
            raise ValueError("no checkpoint_dir configured")
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, checkpoint_name(self.step))
        self.net.save_adapter(
            path, base_model_name_or_path=self.base_model_name_or_path)
        torch.save({"step": self.step, "samples_seen": self.samples_seen,
                    "optimizer": self.opt.state_dict()},
                   os.path.join(path, "realtime_trainer_state.pt"))
        with open(os.path.join(path, "realtime_meta.json"), "w",
                  encoding="utf8") as f:
            json.dump({"step": self.step, "samples_seen": self.samples_seen,
                       "lr": self.lr, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "config": self.config.to_dict()}, f, indent=2)
        if self.config.keep_checkpoints > 0:
            names = list_realtime_checkpoints(root)
            for stale in names[:-self.config.keep_checkpoints]:
                shutil.rmtree(os.path.join(root, stale))
        return path

    def load(self, directory: str) -> None:
        """Resume adapter weights (and, when present, optimizer state / step
        counter) from a checkpoint written by :meth:`checkpoint` -- or any
        PEFT adapter dir saved by the offline trainers (weights only). The
        resumed adapter is pushed to inference immediately."""
        # Un-park first: load_adapter writes into the masters at their home
        # devices, and load_state_dict maps optimizer state to the params'
        # CURRENT devices -- both want the training state on-device.
        self._restore_idle()
        self.net.load_adapter(directory)
        state_path = os.path.join(directory, "realtime_trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu",
                               weights_only=True)
            self.opt.load_state_dict(state["optimizer"])
            self.step = int(state["step"])
            self.samples_seen = int(state.get("samples_seen", 0))
            # load_state_dict restores the checkpoint's lr; the configured /
            # externally set constant lr wins.
            self.lr = self.config.lr
        self.sync_to_inference()
        self._offload_idle()
