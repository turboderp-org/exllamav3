"""
QLoRA fine-tuning of an EXL3 model with NO HuggingFace Transformers in the loop.

This trains low-rank adapters on a frozen EXL3 model using exllamav3's own
weights and a transformers-free differentiable forward
(:class:`exllamav3.training.native_llama.NativeLlamaQLoRA`). It exists because
the Transformers-based path couples training to a specific transformers version
(the EXL3 Llama-3.2 weights were calibrated against 4.45 and 5.x mis-handles the
llama3 RoPE); the native path reuses the exact RoPE/norms/scale that
exllamav3's correct inference forward uses, so it can't be broken upstream.

Requirements (CUDA box with the exllamav3 extension built):
    pip install datasets            # note: NO transformers / accelerate needed

Usage:
    python training/qlora_train_native.py \
        --model /path/to/exl3_model \
        --out   out/exl3_qlora_adapter

Defaults fine-tune on superdrew100/UwU_Alpaca_data: the Alpaca-cleaned
instruction set with every answer rewritten in over-the-top "UwU" furry speak
(caps, emoji, "OwO", "*twitches whiskers*"). Because it keeps Alpaca's clean
question->on-topic-answer structure, the model stays coherent while the style
is unmistakable at scale 1.0 -- unlike play-script style sets, whose responses
are tangential monologues that teach the model to ramble. (Note: the persona
has mild PG-13 innuendo in places.)

The data loader is dataset-agnostic: it reads instruction / context / response
columns whose names are configurable via --instruction-key / --context-key /
--response-key, so swapping in another instruction set (e.g. Dolly-schema
TeeZee/dolly-15k-pirate-speech via --instruction-key instruction --context-key
context --response-key response) needs no code change. Validate first with
training/qlora_validate_native.py, then check the trained adapter with
training/qlora_infer_native.py -- both are also transformers-free.

The adapter is saved in PEFT format, loadable by exllamav3.model.lora.LoRA
(and by PEFT).
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import deque
import torch

from exllamav3 import Config, Model, Tokenizer
from exllamav3.training.native_llama import NativeLlamaQLoRA

# Local run logger + self-contained HTML report -- the default logging path
# (replaces wandb for shareable dashboards). Same dir on sys.path when this file
# is run as a script or imported by the other trainers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_report import (RunLogger, decode_example_docs,  # noqa: E402
                        start_live_monitor)
from chat_turns import (extract_turns, single_turn_shape,  # noqa: E402
                        trim_trailing_context, make_segment_builder,
                        encode_segments, encode_completion,
                        turn_text, AUTO_SINGLE_TURN_HINT)


class ThroughputMeter:
    """Rolling tok/s over a sliding window of recent steps, for a live readout.

    Tracks supervised (loss-bearing, labels != -100) and total (non-pad) tokens
    separately, so the per-step line can show real throughput rather than the
    run-average the final ``[PERF]`` line reports. Window-based (not cumulative)
    so the number reflects current steady state, not warmup. Time fed in should
    be the train-step compute only (exclude eval/sample/save) for a clean rate.
    """

    def __init__(self, window=20):
        self.buf = deque(maxlen=window)   # (dt, supervised_tokens, total_tokens)

    def update(self, dt, supervised, total):
        self.buf.append((float(dt), int(supervised), int(total)))

    def rates(self):
        """Return (supervised_tok_per_s, total_tok_per_s) over the window."""
        tt = sum(b[0] for b in self.buf)
        if tt <= 0:
            return 0.0, 0.0
        return sum(b[1] for b in self.buf) / tt, sum(b[2] for b in self.buf) / tt


class StepTimer:
    """Wall-clock breakdown of every training step into sections: ``data``
    (batch build/collate), ``fwd`` (loss forward), ``bwd`` (backward), ``opt``
    (grad clip + optimizer + scheduler step).

    ``mark(section)`` charges the time since the previous mark to ``section``;
    sections repeat within a step under grad accumulation and accumulate. On
    CUDA each mark synchronizes the active devices first, so async GPU work is
    charged to the section that launched it (a few extra syncs per step, ~µs
    each -- the loop already syncs at ``loss.item()`` / ``gnorm.item()``).

    Keeps cumulative totals (for the ``[PERF]`` summary and the run-log CSV)
    plus a rolling window of recent steps (for the live per-step line), so a
    run answers "where does the time go" without a separate profiling run.
    """

    SECTIONS = ("data", "fwd", "bwd", "opt")

    def __init__(self, devices=None, window=20):
        # devices: CUDA device indices to synchronize at each mark (the split
        # load spans several); None -> current device only.
        self.devices = devices
        self.total = {s: 0.0 for s in self.SECTIONS}
        self.steps = 0
        self.win = deque(maxlen=window)
        self._cur = None
        self._t = None

    def _now(self):
        if torch.cuda.is_available():
            for d in (self.devices or [None]):
                torch.cuda.synchronize(d)
        return time.perf_counter()

    def begin_step(self):
        self._cur = {s: 0.0 for s in self.SECTIONS}
        self._t = self._now()

    def mark(self, section):
        t = self._now()
        self._cur[section] += t - self._t
        self._t = t

    def end_step(self):
        for s in self.SECTIONS:
            self.total[s] += self._cur[s]
        self.steps += 1
        self.win.append(self._cur)
        self._cur = None

    def step_line(self):
        """Compact rolling mean for the per-step line, e.g.
        ``1.84s: f 52% b 39% o 8%`` (data shown only when it reaches 1%)."""
        if not self.win:
            return ""
        n = len(self.win)
        avg = {s: sum(w[s] for w in self.win) / n for s in self.SECTIONS}
        tot = sum(avg.values())
        if tot <= 0:
            return ""
        parts = []
        for key, label in (("data", "d"), ("fwd", "f"), ("bwd", "b"), ("opt", "o")):
            pct = 100.0 * avg[key] / tot
            if key != "data" or pct >= 1.0:
                parts.append(f"{label} {pct:.0f}%")
        return f"{tot:.2f}s: " + " ".join(parts)

    def summary(self):
        """Run-total split for the [PERF] line, e.g. ``data 1% fwd 51% bwd 40% opt 8%``."""
        tot = sum(self.total.values())
        if tot <= 0:
            return "n/a"
        return " ".join(f"{s} {100.0 * self.total[s] / tot:.0f}%" for s in self.SECTIONS)


# Canonical schema for the per-run CSV log. Fixed order so the "mega CSV" stays
# consistent across runs/arms; the BNB arm inlines an identical copy (separate
# venv) and the DDP script imports these. Unknown keys are ignored and missing
# fields written blank, so adding a column later only needs an entry here.
RUN_LOG_FIELDS = [
    "timestamp", "arm", "status", "model", "arch", "out",
    "dataset", "eval_split", "eval_dataset", "eval2_dataset",
    "r", "alpha", "expert_r", "use_rslora", "init_lora", "quant_aware", "quant_aware_scale",
    "lr", "scheduler", "warmup_steps", "weight_decay",
    "batch", "grad_accum", "world_size", "eff_batch",
    "epochs", "steps_planned", "steps_done", "seq_len",
    "targets", "mtp_targets", "freeze_trunk",
    "compute_dtype", "attn_impl", "parallel", "shuffle", "pack", "pack_algo", "ga_loss",
    "max_samples", "train_embeddings", "train_head",
    "lora_embed", "lora_head", "module_lora_lr_mul",
    "prompt_format",
    "trainable_params", "n_train", "n_val", "n_eval2",
    "start_loss", "end_loss", "best_val", "best_val_step",
    "start_val", "start_eval2", "final_val", "final_eval2",
    "total_s", "s_per_step", "sup_tok_s", "tot_tok_s", "peak_vram_gb",
    # Wall-clock section totals (seconds) from StepTimer, and the measured
    # dequant (trellis reconstruction) time per step when --profile-dequant ran.
    "t_data_s", "t_fwd_s", "t_bwd_s", "t_opt_s", "dequant_s_per_step",
    # Failure forensics: where the run died and the exception summary. Blank
    # for completed runs; status=failed rows carry them, so the CSV doubles as
    # a lab notebook of what was tried and why it fell over.
    "phase", "error",
    "notes",
]


def append_run_log(path, record):
    """Append one run's metadata as a row to a CSV (header written once on
    create). Pure stdlib so the DDP script imports it and the BNB arm inlines a
    copy. Keys outside RUN_LOG_FIELDS are ignored and missing fields left blank.
    If an existing file's header doesn't match the current schema (columns were
    added/removed), the old file is moved aside to ``<path>.bak`` and a fresh one
    started, so the CSV never ends up with misaligned rows."""
    if not path:
        return
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header is not None and header != RUN_LOG_FIELDS:
            # Timestamped .bak: a fixed name loses the PREVIOUS backup when two
            # arms with different schemas alternate (2026-07-14 incident).
            bak = f"{path}.{datetime.datetime.now():%Y%m%d-%H%M%S}.bak"
            os.replace(path, bak)
            print(f"[run-log] schema changed; moved old log to {bak}")
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_LOG_FIELDS, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow({k: record.get(k, "") for k in RUN_LOG_FIELDS})
    print(f"[run-log] appended 1 row to {path}")


# Mutable context for the failure logger. _run_main() fills this in as the run
# progresses (args first, then per-milestone phase updates, then per-step
# progress), so a crash ANYWHERE -- dataset typo, OOM at step 3, a guard's
# SystemExit -- still writes a meaningful run-log row: the CSV records failed
# experiments and why, not just the ones that finished. A run that completes
# (or is Ctrl-C'd through the normal path) sets ``logged`` and the failure
# logger stays silent. Note a hard process kill (segfault, OOM-killer, SLURM
# preemption) can't be caught -- those runs leave no row.
_FAIL_CTX = {"run_log": None, "record": {}, "phase": "startup", "logged": False}

# The live wandb run (when --wandb-project is set), kept module-level so the
# failure logger can close it with a failure exit code. _finish_wandb is
# idempotent: the first caller (normal finish, Ctrl-C, or _log_failure) wins.
_WANDB_RUN = {"run": None}

# The local run report (default logging path), kept module-level for the same
# reason: the failure logger renders it with a failure exit code on a crash so a
# died run still gets its report.html. _finish_report is idempotent.
_REPORT = {"rep": None}


def _finish_wandb(exit_code=0):
    run, _WANDB_RUN["run"] = _WANDB_RUN["run"], None
    if run is not None:
        try:
            run.finish(exit_code=exit_code)
        except Exception as exc:  # never let wandb teardown mask the real exit
            print(f"[wandb] finish failed: {exc}")


def _finish_report(exit_code=0, status=None):
    rep, _REPORT["rep"] = _REPORT["rep"], None
    if rep is not None:
        try:
            rep.finish(exit_code=exit_code, status=status)
        except Exception as exc:  # never let report render mask the real exit
            print(f"[report] finish failed: {exc}")


def _log_failure(status, exc):
    """Append a run-log row for a run that died, plus the full traceback to a
    sidecar ``<run_log>.errors.log`` (tracebacks don't fit a CSV cell)."""
    _finish_wandb(exit_code=1)
    _finish_report(exit_code=1)
    if _FAIL_CTX["logged"] or not _FAIL_CTX["run_log"]:
        return
    _FAIL_CTX["logged"] = True
    import traceback
    err = f"{type(exc).__name__}: {exc}".strip()[:400]
    rec = dict(_FAIL_CTX["record"])
    rec.update(status=status, phase=_FAIL_CTX["phase"], error=err)
    try:
        append_run_log(_FAIL_CTX["run_log"], rec)
        elog = os.path.abspath(_FAIL_CTX["run_log"]) + ".errors.log"
        with open(elog, "a", encoding="utf-8") as f:
            f.write(f"\n===== {datetime.datetime.now().isoformat(timespec='seconds')}"
                    f" | phase: {_FAIL_CTX['phase']} | {err}\n")
            f.write(traceback.format_exc())
        print(f"[run-log] failure recorded (phase: {_FAIL_CTX['phase']}); "
              f"traceback appended to {elog}")
    except Exception as log_exc:  # never mask the original error with a log error
        print(f"[run-log] could not record failure: {log_exc}")


def checkpoint_dir(out, step):
    """Path of the retained per-step checkpoint under ``out``. Zero-padded so the
    lexicographic order matches the numeric order (handy for ``ls`` and for the
    prune-by-age logic below). Distinct from ``out`` itself (which --save-every /
    --save-best overwrite); these accumulate a history you can roll back to."""
    return os.path.join(out, f"checkpoint-{step:08d}")


def final_dir(out):
    """Where the last-step (or interrupted) adapter goes when --save-best owns
    ``out`` itself. Separate path so the final weights are always preserved
    without clobbering the best-val adapter -- before this existed, --save-best
    silently discarded both the final step and everything after the last
    --checkpoint-every boundary on Ctrl-C. Not a ``checkpoint-<step>`` dir, so
    prune_checkpoints never touches it."""
    return os.path.join(out, "final")


def list_checkpoints(out):
    """Existing ``checkpoint-<step>`` dirs under ``out``, oldest-step first."""
    if not os.path.isdir(out):
        return []
    found = []
    for name in os.listdir(out):
        if name.startswith("checkpoint-") and os.path.isdir(os.path.join(out, name)):
            tail = name[len("checkpoint-"):]
            if tail.isdigit():
                found.append((int(tail), os.path.join(out, name)))
    return [p for _, p in sorted(found)]


def prune_checkpoints(out, keep):
    """Keep only the ``keep`` most-recent checkpoint dirs under ``out`` (delete the
    oldest). ``keep <= 0`` means keep everything. Adapters are small, but
    --train-embeddings/--train-head make a checkpoint large, so capping matters."""
    if keep is None or keep <= 0:
        return
    existing = list_checkpoints(out)
    for path in existing[:max(0, len(existing) - keep)]:
        shutil.rmtree(path, ignore_errors=True)
        print(f"  [checkpoint] pruned old {path}")


TRAINER_STATE_FILE = "trainer_state.pt"


def save_trainer_state(directory, *, step, opt, sched, best_val, best_val_step, ema,
                       offload_opt=None):
    """Persist resumable training state next to the adapter so --resume continues
    the optimizer + LR schedule instead of cold-restarting them (which would
    wrongly replay warmup/cosine from step 0). Small for LoRA (AdamW moments are a
    few MB at low rank). Written into every save target, so any checkpoint dir --
    the best at --out, a --save-every copy, or a --checkpoint-every history dir --
    is a complete resume point. ``offload_opt`` is the optional CPU-offload optimizer
    for the embed/head group (its state is large -- the embed/head Adam moments --
    but lives on CPU); stored under a separate key so a run without it still loads."""
    state = {
        "step": int(step),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "best_val": best_val,
        "best_val_step": best_val_step,
        "ema": ema,
    }
    if offload_opt is not None:
        state["offload_optimizer"] = offload_opt.state_dict()
    torch.save(state, os.path.join(directory, TRAINER_STATE_FILE))


def load_trainer_state(directory):
    """Load the trainer-state dict written by ``save_trainer_state`` (to CPU; the
    caller moves optimizer tensors onto each param's device). Returns ``None`` when
    the dir has only adapter weights (e.g. a checkpoint from before this existed,
    or a foreign PEFT adapter) so resume falls back to weights-only."""
    path = os.path.join(directory, TRAINER_STATE_FILE)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_optimizer_state(opt, opt_state):
    """Load an optimizer state_dict and move each state tensor onto its param's
    current device -- params can be split across GPUs under --parallel split, so a
    single map_location won't do. Matches saved state to params by order, so the
    param_groups must be built identically (same --r/--targets/--train-*)."""
    opt.load_state_dict(opt_state)
    for p, st in opt.state.items():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(p.device)


def turn_end_token(tokenizer):
    """End-of-assistant-turn marker for completion-only SFT, per chat format.

    The model must learn to emit a stop token after the response or generation
    never terminates. The right token is architecture-specific: the Llama-3
    family ends a turn with ``<|eot_id|>``; Mistral/Tekken and most others use
    their EOS (``</s>``). We pick ``<|eot_id|>`` only when it actually exists as
    a special token (preserving the proven Llama path), otherwise the tokenizer's
    EOS. Exception: a ChatML EOS (``<|im_end|>``) wins over a merely-registered
    ``<|eot_id|>`` -- AFMoE/Trinity ships a Llama-derived tokenizer that still
    REGISTERS ``<|eot_id|>`` but ends turns with ``<|im_end|>`` (its EOS and
    what its own chat template emits). Encoded with
    ``encode_special_tokens=True`` it maps to the single special id, matching
    the generator's stop condition.

    Second exception, same shape: a Harmony-style base whose EOS is NOT the
    turn-end marker. MuseGlimmer ends every turn with ``<|eot|>`` (its
    ``default_chat_prompt``, its chat template and the inference stop
    conditions all agree), while its EOS is a different special token
    entirely -- so the EOS fallback below would train a stop token the
    generator doesn't stop on. Prefer a registered bare ``<|eot|>``.
    """
    if tokenizer.eos_token == "<|im_end|>":
        return tokenizer.eos_token
    if "<|eot_id|>" in tokenizer.extended_piece_to_id:
        return "<|eot_id|>"
    if "<|eot|>" in tokenizer.extended_piece_to_id:
        return "<|eot|>"
    if tokenizer.eos_token:
        return tokenizer.eos_token
    return ""


def format_prompt_and_eot(model, tokenizer, prompt_format,
                          chat_template_file=None, template_vars=None):
    """Return ``(build_prompt(user, system=None) -> str, eot_str)`` for the
    chosen chat format. ``system`` is optional and folded into the template's
    system turn when given (falsy/None omits it entirely -- identical output
    to before system support existed).

    - ``jinja``: the model directory's own Jinja chat template
      (chat_template.jinja / chat_template.json / tokenizer_config.json's
      chat_template key; ``chat_template_file`` overrides), rendered with
      ``add_generation_prompt=True`` and ``template_vars`` in the context --
      see training/chat_jinja.py. The eot is derived from the template itself
      (the text it appends after assistant content), falling back to
      :func:`turn_end_token` for templates the probe can't render.

    - ``auto`` (default): the model's own template (``default_chat_prompt`` --
      Llama-3, Mistral ``[INST]``, mistral3 ``[SYSTEM_PROMPT]``/``[INST]``, etc.)
      and the architecture-correct turn-end token (:func:`turn_end_token`).
      Unchanged from prior behavior.
    - ``mistral``: the explicit Mistral V7+/V13 instruct format
      ``<s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{user}[/INST]{response}</s>``
      (no spaces; ``[INST]``/``[/INST]``/``[SYSTEM_PROMPT]`` are control tokens;
      the system block is omitted when there's no system text). This is what
      ``auto`` already emits for the ``mistral3`` arch (Mistral Small/Medium 3.x,
      incl. Mistral-Medium-3.5-128B) -- the explicit option just doesn't depend
      on arch detection. EOS ends the turn.
    - ``metharme``: the Pygmalion/Metharme format
      ``<s><|system|>{system}<|user|>{user}<|model|>{response}</s>`` (the
      ``<|system|>`` block omitted when there's no system text). The
      ``<|system|>``/``<|user|>``/``<|model|>`` markers are plain text on a base
      model (not registered special tokens) -- the model learns them as a
      literal pattern, which is the standard way these tunes are trained. EOS
      ends the turn.
    - ``gemma4-nothink``: the Gemma4 turn format with the thought channel
      pre-closed empty (``<|turn>system\\n{system}<turn|>\\n<|turn>user\\n{user}
      <turn|>\\n<|turn>model\\n<|channel>thought\\n<channel|>{response}``, system
      turn omitted when there's no system text), so the model is trained to
      answer directly instead of emitting a reasoning span. Matches the
      ``"gemma4"`` case in ``examples/common.py`` / ``PromptFormat_gemma4`` in
      ``examples/chat_templates.py`` used for inference. ``<turn|>`` (a
      registered special token) ends the turn, not EOS.
    - ``llama3``: the explicit Llama-3 header format
      ``<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n{system}
      <|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\n{user}<|eot_id|>
      <|start_header_id|>assistant<|end_header_id|>\\n\\n{response}<|eot_id|>``
      (system turn omitted when there's no system text). Identical to ``auto``
      on the llama arch -- the explicit option trains the Llama-3 pattern onto
      any base without depending on arch detection (on non-Llama tokenizers the
      header markers are plain text the model learns, like ``metharme``). The
      model's own BOS is used; ``<|eot_id|>`` ends the turn.
    - ``qwen3.5``: plain ChatML, ``<|im_start|>system\\n{system}<|im_end|>\\n
      <|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n{response}
      <|im_end|>`` (system turn omitted when there's no system text). Identical
      to ``auto`` on the qwen3/qwen3.5 archs. Use this when the responses carry
      their own ``<think>...</think>`` spans (reasoning SFT) or the base is a
      non-reasoning ChatML model. No BOS (Qwen tokenizers define none);
      ``<|im_end|>`` ends the turn.
    - ``qwen3.5-nothink``: ChatML with the reasoning block pre-closed empty --
      the assistant turn opens with ``<think>\\n\\n</think>\\n\\n`` inside the
      (masked) prompt, so the model is trained to answer directly. This matches
      exactly what ``PromptFormat_qwen35`` in ``examples/chat_templates.py``
      prefills at inference when thinking is disabled, so train and serve see
      the same context (the gemma4-nothink of the Qwen3.5/3.6 family).
      ``<|im_end|>`` ends the turn.

    For ``mistral``/``metharme``/``gemma4-nothink``/``llama3`` a literal BOS is
    prepended so the sequence starts with one; the caller's BOS-normalization
    then collapses any duplicate the tokenizer auto-adds. The ChatML formats
    prepend no BOS.
    """
    if prompt_format == "mistral":
        bos = tokenizer.bos_token or ""
        eos = tokenizer.eos_token or ""
        def build(user, system=None):
            sys_part = f"[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]" if system else ""
            return f"{bos}{sys_part}[INST]{user}[/INST]"
        return build, eos
    if prompt_format == "metharme":
        bos = tokenizer.bos_token or ""
        eos = tokenizer.eos_token or ""
        def build(user, system=None):
            sys_part = f"<|system|>{system}" if system else ""
            return f"{bos}{sys_part}<|user|>{user}<|model|>"
        return build, eos
    if prompt_format == "gemma4-nothink":
        bos = tokenizer.bos_token or ""
        def build(user, system=None):
            sys_part = f"<|turn>system\n{system}<turn|>\n" if system else ""
            return (f"{bos}{sys_part}<|turn>user\n{user}<turn|>\n<|turn>model\n"
                     f"<|channel>thought\n<channel|>")
        return build, "<turn|>"
    if prompt_format == "llama3":
        bos = tokenizer.bos_token or ""
        def build(user, system=None):
            sys_part = (f"<|start_header_id|>system<|end_header_id|>\n\n"
                        f"{system}<|eot_id|>") if system else ""
            return (f"{bos}{sys_part}"
                    f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")
        return build, "<|eot_id|>"
    if prompt_format in ("qwen3.5", "qwen3.5-nothink", "chatml"):
        # "chatml" is an alias for the plain (non-nothink) ChatML template --
        # the same format Qwen uses, and what AFMoE / Trinity checkpoints ship
        # as their Jinja template (no think block).
        nothink = "<think>\n\n</think>\n\n" if prompt_format.endswith("-nothink") else ""
        def build(user, system=None):
            sys_part = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
            return (f"{sys_part}<|im_start|>user\n{user}<|im_end|>\n"
                    f"<|im_start|>assistant\n{nothink}")
        return build, "<|im_end|>"
    if prompt_format == "jinja":
        from chat_jinja import jinja_renderers, tokenizer_special_tokens
        _, build, eot = jinja_renderers(
            tokenizer.config.directory,
            special_tokens=tokenizer_special_tokens(tokenizer),
            template_file=chat_template_file, default_vars=template_vars)
        return build, (eot or turn_end_token(tokenizer))
    if prompt_format == "auto":
        return ((lambda user, system=None: model.default_chat_prompt(user, system_prompt=system)),
                turn_end_token(tokenizer))
    raise ValueError(f"unknown prompt-format '{prompt_format}' "
                      f"(expected auto/mistral/metharme/gemma4-nothink/llama3/"
                      f"qwen3.5/qwen3.5-nothink/chatml/jinja)")


# Stage directions / inline actions, e.g. "[as CAMBIO]", "[TRINCULO grabs ...]",
# "*stares at the ceiling*". Style datasets built from play scripts carry these,
# and the model happily learns to emit them, producing disjoint non-answers.
_STAGE_DIR = re.compile(r"\[[^\]]*\]|\*[^*]*\*")
_WHITESPACE = re.compile(r"\s+")


def clean_style_text(s):
    """Strip stage directions and collapse runaway whitespace/newlines."""
    s = _STAGE_DIR.sub(" ", s)
    s = _WHITESPACE.sub(" ", s)
    return s.strip()


def extract_single_turn(messages):
    """Pull (system_text, user_text, assistant_text) from an OpenAI-style
    ``messages`` list. LEGACY: superseded by chat_turns.extract_turns /
    single_turn_shape -- kept for the comparison arms' identical copies and
    any external callers.

    For single-turn rows (e.g. UnstableLlama/semancy: one user, one assistant,
    no system message) this is exact. On multi-turn input it silently keeps
    only the first exchange (it breaks at the FIRST assistant turn, so it can
    also pick a user turn from the middle of a history) -- which is why the
    trainers no longer call it.
    """
    sys_text, user_text, asst_text = "", "", ""
    for m in messages or []:
        role = (m.get("role") or "").lower()
        content = (m.get("content") or "").strip()
        if role == "system":
            if not sys_text:
                sys_text = content    # keep the first system turn only
        elif role == "user":
            user_text = content       # remember the most recent user turn
        elif role == "assistant":
            asst_text = content
            break                     # first assistant reply is the target
    return sys_text, user_text, asst_text


def encode_prompt_response(tokenizer, prompt_text, response_text, eot):
    """Tokenize a (prompt, response) pair for completion-only supervision.

    The prompt and response are encoded SEPARATELY and concatenated by the
    caller, so the prompt/response mask boundary is exact (masking by prompt
    string length is vulnerable to tokenizer boundary merges). ``eot`` (the
    architecture-correct turn-end token) is appended to the response text.

    BOS is normalized to exactly one leading token: with
    ``encode_special_tokens=True`` the underlying HF tokenizer may auto-prepend
    ``<|begin_of_text|>`` (Llama-3 has ``add_bos_token=true``) *in addition to*
    the literal one the chat template embeds, and *again* on the separately
    encoded response -- so the prompt would start with two BOS and the response
    with a spurious one. Drop the duplicates; a no-op for tokenizers that don't
    auto-prepend. Returns ``(prompt_ids, response_ids)`` as python int lists.
    """
    prompt_ids = tokenizer.encode(
        prompt_text, add_bos=False, encode_special_tokens=True
    )[0].tolist()
    resp_ids = tokenizer.encode(
        response_text + eot, add_bos=False, encode_special_tokens=True
    )[0].tolist()
    bos = tokenizer.bos_token_id
    if bos is not None:
        while len(prompt_ids) >= 2 and prompt_ids[0] == bos and prompt_ids[1] == bos:
            prompt_ids = prompt_ids[1:]
        if resp_ids and resp_ids[0] == bos:
            resp_ids = resp_ids[1:]
    return prompt_ids, resp_ids


def build_optimizer(param_groups, lr, optim="adamw"):
    """Build the optimizer over the trainable param groups.

    ``adamw`` is torch's AdamW: ``m``/``v`` in fp32 = 8 bytes per trainable param.
    For a 262M-param r=64 adapter that is ~2.1 GB of optimizer state (split across
    devices under ``--parallel split``), allocated lazily on the first
    ``optimizer.step()`` -- which is why a run can pass step 0 / the first few
    steps and then OOM once the moments materialize.

    ``adamw8bit`` / ``paged_adamw8bit`` are bitsandbytes 8-bit AdamW: the moments
    are quantized to ~2 bytes/param (~4x less state, ~1.6 GB freed at r=64), with
    negligible quality cost (the QLoRA paper trains with paged 8-bit Adam). The
    ``paged_`` variant additionally offloads optimizer state to host memory on a
    spike, smoothing transient peaks. Both need ``bitsandbytes`` importable.
    """
    if optim == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr)
    try:
        import bitsandbytes as bnb
    except Exception as e:
        raise SystemExit(
            f"--optim {optim} needs bitsandbytes, which is not importable "
            f"({e}). Install it in this venv (pip install bitsandbytes) or use "
            f"--optim adamw."
        )
    cls = bnb.optim.PagedAdamW8bit if optim == "paged_adamw8bit" else bnb.optim.AdamW8bit
    return cls(param_groups, lr=lr)


def build_cpu_offload_optimizer(params, lr):
    """A torchao ZeRO-Offload optimizer for the fully-trained embedding / LM head.

    Keeps the optimizer state (and the bf16 master weights) on CPU and runs the
    AdamW step there, so the ~12 bytes/param of fp32 Adam state for the (huge,
    untied ~0.8B-each) embed/head matrices never sits on the GPU -- only the bf16
    parameter and its transient grad do. The base optimizer is torchao's AdamW with
    ``bf16_stochastic_round=True`` (bound via ``partial`` so it applies regardless of
    whether CPUOffloadOptimizer forwards kwargs): bf16 master updates stay an
    unbiased estimate of fp32, so small embedding updates aren't rounded away. The
    embed/head params must already be bf16 (NativeLlamaQLoRA(modules_to_save_dtype=
    bfloat16)) for the rounding to apply.

    State-only offload (NOT offload_gradients) so gradient accumulation still works.
    CPUOffloadOptimizer is a wrapper, not a real optimizer: it has no LR-scheduler
    support and forbids gradient clipping on its params, so the caller mirrors the
    schedule's LR via set_offload_lr() each step and excludes these params from the
    clip. Single-process only (CUDA); not for the DDP arm.
    """
    try:
        import functools
        import torchao.optim as aoopt
        from torchao.optim import CPUOffloadOptimizer
    except Exception as e:
        raise SystemExit(
            f"--offload-embed-head-optim needs torchao, which is not importable "
            f"({e}). Install it in this venv (pip install torchao) or drop the flag "
            f"(the embed/head optimizer then stays on GPU; use --optim adamw8bit to "
            f"shrink it instead)."
        )
    # The fp32 AdamW clone that supports bf16 stochastic rounding is `_AdamW` in
    # current torchao (README: `_AdamW(..., bf16_stochastic_round=True)`); older/newer
    # layouts may call it `AdamW`. The 8-bit variants (AdamW8bit/4bit) use CUDA-only
    # quant kernels and can't run on the CPU-offloaded step, so they're not used here.
    AOAdamW = getattr(aoopt, "_AdamW", None) or getattr(aoopt, "AdamW", None)
    if AOAdamW is None:
        raise SystemExit(
            "torchao.optim exposes no _AdamW/AdamW for bf16 stochastic rounding; "
            "available: " + ", ".join(n for n in dir(aoopt) if "dam" in n.lower())
            + ". Tell me which to use.")
    base = functools.partial(AOAdamW, bf16_stochastic_round=True)
    opt = CPUOffloadOptimizer(params, base, lr=lr)
    set_offload_lr(opt, lr)   # don't rely on lr= forwarding to the base optimizer
    return opt


def set_offload_lr(opt, lr):
    """Set the LR on every param group of a CPUOffloadOptimizer (which is not
    compatible with torch's LR schedulers). Handles the tensor-LR case torchao uses
    for its fused/compiled path."""
    for g in opt.param_groups:
        cur = g.get("lr")
        if isinstance(cur, torch.Tensor):
            cur.fill_(lr)
        else:
            g["lr"] = lr


def make_lr_scheduler(optimizer, name, total_steps, warmup_steps):
    """A transformers-free LR scheduler (none/linear/cosine) with linear warmup.

    Matches HuggingFace's ``get_{linear,cosine}_schedule_with_warmup`` exactly so
    behavior is well understood: LR ramps 0->1 over ``warmup_steps``, then decays
    to 0 (linear) or follows a half-cosine to 0 (cosine) over the remaining
    ``total_steps - warmup_steps``. ``none``/``constant`` holds the base LR after
    warmup. Driven by one ``scheduler.step()`` per optimizer step.
    """
    name = (name or "none").lower()
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if name in ("none", "constant"):
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        if name == "linear":
            return 1.0 - progress
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"unknown scheduler '{name}' (expected none/linear/cosine)")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_steps_and_warmup(args, num_train_examples, effective_batch):
    """Finalize args.steps (from --epochs if given) and compute warmup steps.

    ``--epochs`` (when > 0) overrides ``--steps`` so the schedule length matches
    the requested number of passes over the data: one optimizer step consumes
    ``effective_batch`` examples, so an epoch is ``ceil(N / effective_batch)``
    steps. ``--warmup-steps`` (when > 0) wins over ``--warmup-ratio``.

    Also stashes ``args.steps_per_epoch`` (computed even in --steps mode) so the
    training loop can show per-step epoch progress.
    """
    eff = max(1, int(effective_batch))
    steps_per_epoch = max(1, math.ceil(num_train_examples / eff))
    args.steps_per_epoch = steps_per_epoch
    if getattr(args, "epochs", 0) and args.epochs > 0:
        args.steps = max(1, math.ceil(args.epochs * steps_per_epoch))
    warmup = (args.warmup_steps if getattr(args, "warmup_steps", 0) and args.warmup_steps > 0
              else int(round(getattr(args, "warmup_ratio", 0.0) * args.steps)))
    return args.steps, max(0, warmup)


_LOCAL_DATA_BUILDERS = {".json": "json", ".jsonl": "json",
                        ".parquet": "parquet", ".csv": "csv"}


def load_dataset_split(dataset_name, split, config_name=None):
    """Resolve a --dataset-style argument to a ``datasets`` split.

    Accepts a HF Hub id or a local data file (.json / .jsonl / .parquet /
    .csv; ``~`` expands). load_dataset() can't sniff a bare local path, so
    the builder is picked from the extension. An argument that has a local
    data-file extension but doesn't exist on disk is almost certainly a
    typo'd path -- fail with that, not the Hub lookup's confusing 404.
    """
    from datasets import load_dataset
    path = os.path.expanduser(dataset_name)
    ext = os.path.splitext(path)[1].lower()
    if os.path.exists(path):
        builder = _LOCAL_DATA_BUILDERS.get(ext, "json")
        return load_dataset(builder, data_files=path, split=split)
    if ext in _LOCAL_DATA_BUILDERS:
        raise FileNotFoundError(
            f"dataset {dataset_name!r} looks like a local {ext} file but "
            f"does not exist (cwd: {os.getcwd()})")
    if config_name:
        return load_dataset(dataset_name, config_name, split=split)
    return load_dataset(dataset_name, split=split)


DATASET_SNAPSHOT_DIR = "dataset"
DATASET_SNAPSHOT_META = "meta.json"


def _dataset_is_local_file(name):
    """True when a --dataset-style argument points at a local data file (as
    opposed to a HF Hub id), mirroring load_dataset_split's resolution."""
    path = os.path.expanduser(name)
    return (os.path.splitext(path)[1].lower() in _LOCAL_DATA_BUILDERS
            and os.path.isfile(path))


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_datasets(out_dir, roles):
    """Preserve the run's data mix inside the run dir: copy each local dataset
    file into <out>/dataset/ and record original path + sha256 in meta.json,
    so a later --resume can verify -- and if the source was edited or
    regenerated since, reload -- the EXACT data this run started on.

    ``roles`` maps a role name ("dataset", "eval_dataset", ...) to the
    --dataset-style argument; falsy entries are skipped. HF Hub ids get a
    metadata row only (no copy -- content pinning for Hub sets would need a
    revision hash, and every run here trains on local jsonl anyway). Called on
    fresh runs only; an existing snapshot is overwritten, since a non-resume
    launch into the same --out IS a new run. Resumes go through
    resolve_resumed_datasets() instead and never rewrite the record.
    """
    snap_dir = os.path.join(out_dir, DATASET_SNAPSHOT_DIR)
    os.makedirs(snap_dir, exist_ok=True)
    meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
            "roles": {}}
    for role, name in roles.items():
        if not name:
            continue
        if _dataset_is_local_file(name):
            src = os.path.expanduser(name)
            copy_name = role + os.path.splitext(src)[1].lower()
            shutil.copy2(src, os.path.join(snap_dir, copy_name))
            meta["roles"][role] = {
                "original": os.path.abspath(src), "copy": copy_name,
                "sha256": _sha256_file(src), "bytes": os.path.getsize(src)}
        else:
            meta["roles"][role] = {"original": name, "copy": None,
                                   "sha256": None}
    with open(os.path.join(snap_dir, DATASET_SNAPSHOT_META), "w") as f:
        json.dump(meta, f, indent=2)
    copied = [r for r, m in meta["roles"].items() if m["copy"]]
    if copied:
        print(f" -- dataset mix snapshotted to {snap_dir}/ "
              f"({', '.join(copied)})")


def _find_dataset_snapshot(out_dir, resume_dir):
    """Locate the snapshot for a resumed run. --resume usually points INSIDE
    the run root (a checkpoint-NNNNNNNN dir) or AT it, and --out is normally
    that same root -- check all three spots and take the first hit."""
    resume_dir = os.path.normpath(resume_dir)
    for root in (resume_dir, os.path.dirname(resume_dir), out_dir):
        snap_dir = os.path.join(root, DATASET_SNAPSHOT_DIR)
        if os.path.exists(os.path.join(snap_dir, DATASET_SNAPSHOT_META)):
            return snap_dir
    return None


def resolve_resumed_datasets(out_dir, resume_dir, roles, interactive=True,
                             verbose=True):
    """--resume counterpart of snapshot_datasets: verify each role's file
    against the sha256 recorded at run start and return ``{role: path}`` with
    any substitutions applied.

    On a hash mismatch (the source file was edited/regenerated since the run
    started) the choice matters too much to guess: prompt for snapshot /
    current / abort. Continuing on the snapshot preserves exact data
    continuity (including the fast-forwarded batch order); continuing on the
    current file makes this a different mix from the step counter's point of
    view. Non-interactive callers (DDP ranks -- a prompt would desync them --
    or no tty) abort instead. A missing current file falls back to the
    snapshot automatically; a run predating snapshots proceeds unverified
    with a warning.
    """
    resolved = dict(roles)
    snap_dir = _find_dataset_snapshot(out_dir, resume_dir)
    if snap_dir is None:
        if verbose:
            print(" -- no dataset snapshot found (run predates snapshots); "
                  "resuming on the config's dataset paths UNVERIFIED")
        return resolved
    with open(os.path.join(snap_dir, DATASET_SNAPSHOT_META)) as f:
        meta = json.load(f)
    for role, name in roles.items():
        if not name:
            continue
        rec = meta["roles"].get(role)
        if rec is None or rec["copy"] is None:
            continue  # hub id, or a role this run didn't record
        snap_path = os.path.join(snap_dir, rec["copy"])
        cur = os.path.expanduser(name)
        if not os.path.isfile(cur):
            resolved[role] = snap_path
            if verbose:
                print(f" -- dataset '{role}': {name} no longer exists; using "
                      f"the run's snapshot {snap_path}")
            continue
        if _sha256_file(cur) == rec["sha256"]:
            if verbose:
                print(f" -- dataset '{role}' verified against the run "
                      f"snapshot (sha256 match)")
            continue
        if verbose:
            print(f"\n !! dataset '{role}' has CHANGED since this run started:")
            print(f"      current:  {cur}")
            print(f"      snapshot: {snap_path}  (the data the run began on)")
        if not (interactive and sys.stdin.isatty()):
            raise SystemExit(
                f"dataset '{role}' changed since run start; re-run pointing "
                f"--{role.replace('_', '-')} at the snapshot or the intended "
                f"file explicitly (interactive prompt unavailable here)")
        while True:
            ans = input("    continue on [s]napshot (exact continuation), "
                        "[c]urrent file (new mix mid-run), or [a]bort? "
                        ).strip().lower()
            if ans in ("s", "snapshot"):
                resolved[role] = snap_path
                break
            if ans in ("c", "current"):
                break
            if ans in ("a", "abort", "q", "quit"):
                raise SystemExit("aborted: dataset changed since run start")
    return resolved


def _build_multi_turn_example(tokenizer, seg_builder, turns, seq_len,
                              clean_text, min_response_words,
                              uppercase_response, vision=None, refs=None):
    """Segment-render one multi-turn conversation into an (input_ids, labels)
    example with only the assistant turns supervised. Returns
    ``(example, None)`` or ``(None, reason)`` where reason is a short skip
    label for the caller's counter (``malformed`` / ``short`` /
    ``unrenderable`` / ``truncated``).

    Rich turns (the jinja path: content-parts lists, tool_calls,
    reasoning_content) are handled conservatively: cleaning/uppercasing
    touch STRING contents only, word counts use the text view
    (chat_turns.turn_text), and a turn with tool_calls/reasoning survives an
    empty content."""
    def keep(t):
        return (turn_text(t).strip() or t.get("tool_calls")
                or t.get("reasoning_content"))
    if clean_text:
        turns = [dict(t, content=clean_style_text(t["content"]))
                 if t["role"] != "system" and isinstance(t.get("content"), str)
                 else t for t in turns]
        turns = [t for t in turns if keep(t)]
    turns = trim_trailing_context(turns)
    roles = [t["role"] for t in turns]
    if "user" not in roles or "assistant" not in roles:
        return None, "malformed"
    asst = [t for t in turns if t["role"] == "assistant"]
    asst_words = sum(len(turn_text(t).split()) for t in asst)
    if asst_words < min_response_words and not any(
            t.get("tool_calls") or t.get("reasoning_content") for t in asst):
        return None, "short"
    if uppercase_response:
        turns = [dict(t, content=t["content"].upper())
                 if t["role"] == "assistant" and isinstance(t.get("content"), str)
                 else t for t in turns]
    try:
        segments = seg_builder(turns)
    except ValueError:
        return None, "unrenderable"
    if refs:
        # Image row (--vision): the segment texts carry image sentinels; the
        # vision-aware encoder splices the arch's image token layout in and
        # attaches the per-image bookkeeping (see training/vision_data.py).
        ids, labels, images, feats = vision.encode_segments(segments, refs)
        return vision.finalize(ids, labels, images, feats, refs, seq_len)
    input_ids, labels = encode_segments(tokenizer, segments)
    input_ids, labels = input_ids[:seq_len], labels[:seq_len]
    if all(l == -100 for l in labels):
        return None, "truncated"
    return {"input_ids": input_ids, "labels": labels}, None


def build_sft_examples(model, tokenizer, dataset_name, max_samples, seq_len,
                       instruction_key="instruction", context_key="context",
                       response_key="response", split="train",
                       clean_text=True, min_response_words=3,
                       uppercase_response=False, messages_key=None,
                       prompt_format="auto", shuffle=False, shuffle_seed=0,
                       config_name=None, chat_template_file=None,
                       template_vars=None, vision=None):
    """
    Load an instruction dataset and tokenize for completion-only SFT using the
    model's native chat template (Llama-3, Mistral, etc. -- whatever
    ``model.default_chat_prompt`` emits for this architecture). Prompt tokens are
    masked with -100 so loss is computed only over the (styled) response, which
    is terminated with the architecture-correct turn-end token (see
    :func:`turn_end_token`) so the model learns to stop.

    Two input layouts are supported:
      * flat columns -- instruction_key / context_key / response_key (Alpaca,
        Dolly, ...); context_key may be absent in the dataset (treated as empty).
      * OpenAI ``messages`` -- pass ``messages_key`` (e.g. "messages"). When
        set it takes precedence over the flat-column keys. Single-turn rows
        ([system?] user assistant, e.g. UnstableLlama/semancy) tokenize
        exactly as before: user turn -> prompt, assistant turn -> supervised
        response. Multi-turn rows render EVERY turn via the per-format segment
        builder (chat_turns.make_segment_builder) and supervise ONLY the
        assistant turns -- system/user turns and assistant headers are masked
        to -100, with exact boundaries because each segment is tokenized
        separately. Multi-turn data needs an explicit --prompt-format (auto's
        per-arch default_chat_prompt is single-turn only; such rows fail fast
        with a pointer instead of being silently truncated). Rows with roles
        the format can't render (e.g. ``tool``) or with no user turn are
        skipped and counted.

    ``--prompt-format jinja`` renders EVERY messages row (single- and
    multi-turn) through the model's own Jinja chat template via the segment
    path (training/chat_jinja.py) -- there is no single-turn shortcut, the
    template is the single source of truth. Rich message keys
    (reasoning_content / tool_calls / tool roles / content-parts lists) are
    preserved and rendered; per-row ``tools`` and ``template_vars`` /
    ``chat_template_kwargs`` columns join the render context on top of the
    CLI-level ``template_vars``. Rows the template can't segment-render
    (non-prefix-monotonic history, see chat_jinja) are skipped and counted.
    Rows with non-text content parts (images etc.) train on the rendered
    text only -- placeholder tokens, no pixel features -- and are counted,
    UNLESS ``vision`` (a training/vision_data.VisionData, the ``--vision``
    path) is given: then every messages row whose content parts carry
    images -- on any prompt format, single- or multi-turn -- has its images
    encoded by the frozen vision tower and spliced into the token stream at
    the parts' positions (the arch's own image token layout, image tokens
    masked), with the per-image bookkeeping the forward needs attached to
    the example (``images``, and ``mrope_position_ids`` on a Qwen-VL tower).
    Rows with video/audio parts are skipped and counted. Text-only rows are
    unaffected.

    clean_text strips stage directions / inline actions and normalizes
    whitespace (helps play-script style sets like the Shakespeare default, whose
    raw rows otherwise teach the model to emit "[stage directions]"). Rows whose
    cleaned response has fewer than min_response_words tokens are dropped.

    shuffle (with shuffle_seed) permutes the rows once after loading, BEFORE any
    --val-frac carve and before training, so the held-out split is a random
    sample rather than the first N rows and training order is randomized. It is
    deterministic given the seed, so the EXL3 and BNB arms (which call the same
    HF datasets shuffle) stay matched. Default off preserves the original order;
    the existing shuffle-on-cap (random subset when capping) is unchanged.

    Returns a list of dicts with python int lists: input_ids / labels.
    """
    # Accept either a Hub dataset id or a local file (e.g. a styled set produced
    # by training/experiments/make_style_dataset.py).
    ds = load_dataset_split(dataset_name, split, config_name)
    # Shuffle the full set when asked, or (as before) when capping rows so the
    # subset is random rather than the first max_samples. shuffle_seed defaults to
    # 0, matching the prior cap behavior exactly when --shuffle is off.
    if shuffle or (max_samples and max_samples < len(ds)):
        ds = ds.shuffle(seed=shuffle_seed)
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    jinja = prompt_format == "jinja"
    if jinja:
        from chat_jinja import (jinja_renderers, tokenizer_special_tokens,
                                extract_rich_turns, row_template_extras,
                                has_nontext_parts)
        seg_builder, build_prompt, eot = jinja_renderers(
            tokenizer.config.directory,
            special_tokens=tokenizer_special_tokens(tokenizer),
            template_file=chat_template_file, default_vars=template_vars)
        eot = eot or turn_end_token(tokenizer)
    else:
        build_prompt, eot = format_prompt_and_eot(model, tokenizer, prompt_format)
        seg_builder = make_segment_builder(prompt_format,
                                           bos_token=tokenizer.bos_token,
                                           eos_token=tokenizer.eos_token)

    examples = []
    n_multi, n_mm, n_img, skipped = 0, 0, 0, {}
    resolver = (vision.resolver(dataset_name, ds, messages_key)
                if vision is not None and messages_key else None)
    for row_idx, ex in enumerate(ds):
        # --vision: flatten image content parts to sentinels (one per image,
        # the pixels resolved lazily) so the chat renderers below see plain
        # strings; the encode step re-expands each sentinel into the arch's
        # image token layout. refs is empty on a text-only row.
        refs = []
        if resolver is not None:
            try:
                msgs, refs = vision.prepare(resolver, ex, row_idx, ex.get(messages_key))
            except ValueError as e:
                skipped["unsupported_media"] = skipped.get("unsupported_media", 0) + 1
                if skipped["unsupported_media"] <= 3:
                    print(f" -- skipping row {row_idx}: {e}")
                continue
            if refs:
                ex = dict(ex)
                ex[messages_key] = msgs
                n_img += 1
        if messages_key and jinja:
            # The template is the single source of truth: every messages row
            # (single- or multi-turn, tool calls, reasoning) takes the
            # segment path, with the row's tools/template_vars in context.
            turns = trim_trailing_context(
                extract_rich_turns(ex.get(messages_key)))
            if not any(t["role"] == "assistant" for t in turns):
                continue  # nothing to supervise
            if has_nontext_parts(turns):
                n_mm += 1
            extras = row_template_extras(ex)
            row_builder = (lambda t, _x=extras, **kw:
                           seg_builder(t, **kw, **_x))
            built, reason = _build_multi_turn_example(
                tokenizer, row_builder, turns, seq_len, clean_text,
                min_response_words, uppercase_response,
                vision=vision, refs=refs)
            if built is None:
                skipped[reason] = skipped.get(reason, 0) + 1
            else:
                n_multi += 1
                examples.append(built)
            continue
        if messages_key:
            # Trailing non-assistant turns supervise nothing; trimming them
            # first also lets a [user, assistant, user] row keep the exact
            # single-turn path it always took.
            turns = trim_trailing_context(extract_turns(ex.get(messages_key)))
            if not any(t["role"] == "assistant" for t in turns):
                continue  # nothing to supervise (same skip as before)
            single = single_turn_shape(turns)
            if single is None:
                # Genuine multi-turn (or oddly-shaped) row -> segment renderer.
                if seg_builder is None:
                    raise SystemExit(AUTO_SINGLE_TURN_HINT)
                built, reason = _build_multi_turn_example(
                    tokenizer, seg_builder, turns, seq_len, clean_text,
                    min_response_words, uppercase_response,
                    vision=vision, refs=refs)
                if built is None:
                    skipped[reason] = skipped.get(reason, 0) + 1
                else:
                    n_multi += 1
                    examples.append(built)
                continue
            sys_text, instr, resp = single
            ctx = ""
        else:
            sys_text = ""
            instr = (ex.get(instruction_key) or "").strip()
            ctx = (ex.get(context_key) or "").strip()
            resp = (ex.get(response_key) or "").strip()
        if clean_text:
            instr, ctx, resp = (clean_style_text(instr), clean_style_text(ctx),
                                clean_style_text(resp))
        if not resp or len(resp.split()) < min_response_words:
            continue
        # Smoke test: a maximally dense+consistent transform (every token of every
        # response changes), so there's no low-loss path that ISN'T uppercased and
        # it must surface in generation. Only the response is transformed, so it
        # proves a learned *behavior*, not input echoing.
        if uppercase_response:
            resp = resp.upper()
        user = instr if not ctx else f"{instr}\n\n{ctx}"

        # default_chat_prompt() already includes <|begin_of_text|> and ends with
        # the assistant header, so encode specials and don't add another BOS.
        # encode_prompt_response tokenizes the prompt and the response SEPARATELY
        # (exact mask boundary) and normalizes to exactly one leading BOS.
        prompt_text = build_prompt(user, system=sys_text or None)
        if refs:
            # Single-turn image row: the same (prompt, response) split, encoded
            # through the sentinel-aware segment encoder (prompt masked,
            # response + eot supervised, image tokens masked).
            ids, labels, images, feats = vision.encode_segments(
                [(prompt_text, False), (resp + eot, True)], refs)
            built, reason = vision.finalize(ids, labels, images, feats, refs, seq_len)
            if built is None:
                skipped[reason] = skipped.get(reason, 0) + 1
            else:
                examples.append(built)
            continue
        prompt_ids, resp_ids = encode_prompt_response(
            tokenizer, prompt_text, resp, eot)

        input_ids = (prompt_ids + resp_ids)[:seq_len]
        labels = [-100] * len(prompt_ids) + list(resp_ids)
        labels = labels[:seq_len]
        if all(l == -100 for l in labels):
            continue  # response got truncated away; skip
        examples.append({"input_ids": input_ids, "labels": labels})

    if n_img and int(os.environ.get("RANK", "0") or 0) == 0:
        print(f" -- vision: {n_img} rows carried images; {vision.describe()}")
    if (n_multi or skipped) and int(os.environ.get("RANK", "0") or 0) == 0:
        note = ", ".join(f"{v} {k}" for k, v in sorted(skipped.items()))
        what = ("rows rendered via the model's Jinja chat template" if jinja
                else "multi-turn rows rendered")
        print(f" -- messages: {n_multi} {what} "
              f"(assistant turns supervised, user/system masked)"
              + (f"; skipped {sum(skipped.values())} ({note})" if skipped else ""))
        if n_mm:
            print(f" -- NOTE: {n_mm} rows carry non-text content parts "
                  f"(image/video/audio). They render as the template's "
                  f"placeholder tokens WITHOUT pixel features -- this trainer "
                  f"is text-only, so only the text is trained.")
    return examples


def build_lm_examples(tokenizer, dataset_name, split, seq_len,
                      text_key="text", max_samples=0, config_name=None,
                      max_blocks=0):
    """Plain-text language-modeling eval set (e.g. wikitext) for a second,
    task-independent held-out loss.

    Concatenates the dataset's text column and packs it into non-overlapping
    ``seq_len`` blocks with every token supervised (no completion mask), so the
    resulting loss is a straight nats/token cross-entropy -- on the same scale as
    the SFT eval loss, which lets you watch the two move together (does the task
    fit track or diverge from general LM ability?). Tokenization matches the SFT
    path's underlying tokenizer, so the EXL3 and BNB arms produce identical
    blocks and hence a comparable number.

    Returns a list of dicts (input_ids / labels), same shape as
    :func:`build_sft_examples`, so the same eval loop consumes it.
    """
    # Many text corpora need a config (e.g. wikitext -> "wikitext-2-raw-v1").
    ds = load_dataset_split(dataset_name, split, config_name)
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    bos = tokenizer.bos_token_id
    # Each packed block is scored as an independent sequence (batch-1, no KV
    # carryover), so it should begin like a real sequence does. Match how the SFT
    # path / the model expects input: exactly one leading BOS -- but only for
    # models that actually use one (bos_token_id is None, e.g. Qwen -> none). The
    # block stays seq_len long: one BOS + (seq_len-1) content tokens.
    add_block_bos = bos is not None
    content_len = seq_len - 1 if add_block_bos else seq_len
    buf, examples = [], []
    for row in ds:
        text = row.get(text_key) or ""
        if not text.strip():
            continue
        ids = tokenizer.encode(text, add_bos=False,
                               encode_special_tokens=False)[0].tolist()
        # Drop any BOS the tokenizer auto-prepended; we re-add exactly one per
        # block below, never mid-stream.
        if bos is not None and ids and ids[0] == bos:
            ids = ids[1:]
        buf.extend(ids)
        while len(buf) >= content_len:
            block = buf[:content_len]
            buf = buf[content_len:]
            if add_block_bos:
                block = [bos] + block
            examples.append({"input_ids": block, "labels": list(block)})
            # Cap the number of packed blocks directly (independent of seq_len),
            # so eval2 can be sized to roughly match the primary eval set rather
            # than ballooning -- max_samples only caps source rows, which is
            # unpredictable after packing.
            if max_blocks and len(examples) >= max_blocks:
                return examples
    return examples


def pack_examples(examples, seq_len, pad_id, algo="bfd"):
    """Pack tokenized SFT examples into ``seq_len`` blocks (sample packing).

    Each input example (from :func:`build_sft_examples`) is one *document*;
    documents are concatenated into blocks of at most ``seq_len`` tokens, so a
    short-answer dataset stops wasting most of every forward on pad tokens -- the
    same real tokens are processed in far fewer, fuller sequences.

    ``algo`` picks the bin-packing strategy:
      * ``"bfd"`` (default) -- best-fit decreasing: documents sorted longest-first,
        each placed into the block with the least remaining room that still fits
        (found by bisect over remaining capacities, so it's O(n log n)-ish and
        deterministic). This is the multipack approach of Axolotl (FFD) /
        Chronicals (BFD) and lifts fill from ~80-85% (next-fit) to typically 97%+
        -- directly ~1.15-1.2x more real tokens per step. Reordering documents
        across blocks is harmless: attention is document-isolated, positions
        reset per document, and the training loop shuffles blocks anyway.
      * ``"nextfit"`` -- the pre-Session-11 behavior (arrival order, seal a block
        when the next document doesn't fit), kept for A/B comparison.

    Correctness is preserved by the native forward, NOT here: each block carries
      * ``seg_ids``      -- per-token document index, so attention is restricted to
                            the same document (block-diagonal / flash-varlen). Pad
                            positions inherit the LAST document's seg id, so a pad
                            query still attends back into a real doc and is never
                            fully masked (no softmax NaN); pads are still blocked as
                            keys by the attention mask.
      * ``position_ids`` -- reset to 0..len-1 PER document, so RoPE sees each
                            document at its true positions, not its block offset.
    The completion-only ``-100`` prompt masks are already in each document's
    labels; at a document join the shifted CE predicts the next document's first
    (masked) prompt token, so boundaries contribute no loss and need no fixup.

    Every block is padded to ``seq_len`` so blocks are uniform. Deterministic for
    a given input order (BFD ties keep dataset order), so DDP ranks packing the
    same list get identical blocks. Returns a list of dicts (input_ids / labels /
    seg_ids / position_ids), consumed by :func:`collate`.
    """
    docs = []
    for ex in examples:
        ids, labs = ex["input_ids"], ex["labels"]
        if len(ids) > seq_len:                         # shouldn't happen (build_sft
            ids, labs = ids[:seq_len], labs[:seq_len]  # truncates), but guard
        docs.append((ids, labs))

    # Phase 1: assign document indices to blocks.
    if algo == "nextfit":
        assignments, cur, cur_len = [], [], 0
        for i, (ids, _) in enumerate(docs):
            if cur and cur_len + len(ids) > seq_len:
                assignments.append(cur)
                cur, cur_len = [], 0
            cur.append(i)
            cur_len += len(ids)
        if cur:
            assignments.append(cur)
    elif algo == "bfd":
        import bisect
        order = sorted(range(len(docs)), key=lambda i: (-len(docs[i][0]), i))
        assignments = []
        rems, block_by_rem = [], []       # remaining capacity (sorted) -> block idx
        for i in order:
            n = len(docs[i][0])
            j = bisect.bisect_left(rems, n)   # smallest remaining that fits (best fit)
            if j == len(rems):
                assignments.append([i])
                b, rem = len(assignments) - 1, seq_len - n
            else:
                rem = rems.pop(j) - n
                b = block_by_rem.pop(j)
                assignments[b].append(i)
            k = bisect.bisect_left(rems, rem)
            rems.insert(k, rem)
            block_by_rem.insert(k, b)
        for a in assignments:
            a.sort()                      # within-block docs in dataset order
    else:
        raise ValueError(f"unknown packing algo '{algo}' (expected bfd/nextfit)")

    # Phase 2: materialize blocks (identical layout for both algorithms).
    blocks = []
    for doc_idxs in assignments:
        cur_ids, cur_labels, cur_seg, cur_pos = [], [], [], []
        for seg, i in enumerate(doc_idxs):
            ids, labs = docs[i]
            cur_ids += ids
            cur_labels += labs
            cur_seg += [seg] * len(ids)
            cur_pos += list(range(len(ids)))
        pad = seq_len - len(cur_ids)
        last_seg = cur_seg[-1] if cur_seg else 0
        blocks.append({
            "input_ids": cur_ids + [pad_id] * pad,
            "labels": cur_labels + [-100] * pad,
            "seg_ids": cur_seg + [last_seg] * pad,
            "position_ids": cur_pos + [0] * pad,
        })
    return blocks


def collate(batch, pad_id):
    """Right-pad a batch; pad input_ids with pad_id, labels with -100.

    Returns ``(input_ids, labels, attention_mask, position_ids, seg_ids)``. For
    plain (unpacked) examples ``position_ids`` and ``seg_ids`` are ``None`` (the
    forward derives positions from the mask, with no block-diagonal constraint).
    For packed blocks (carrying ``seg_ids`` from :func:`pack_examples`) they are
    returned so the forward resets RoPE per document and isolates documents.
    """
    maxlen = max(len(b["input_ids"]) for b in batch)
    packed = "seg_ids" in batch[0]
    # Image rows on an mRoPE (Qwen-VL) tower carry 3-D positions
    # ([3, t] per example, from vision.mrope_position_ids); the batch's
    # position_ids is then [3, b, t]. A text-only row in the same batch gets
    # plain sequential positions on all three axes (== 1-D RoPE for it).
    mrope = any("mrope_position_ids" in b for b in batch)
    assert not (packed and mrope), "image rows cannot be sample-packed"
    input_ids, labels, attn = [], [], []
    seg_ids = [] if packed else None
    pos_ids = [] if packed else None
    pos3 = [] if mrope else None
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
        if packed:
            last_seg = b["seg_ids"][-1] if b["seg_ids"] else 0
            seg_ids.append(b["seg_ids"] + [last_seg] * pad)
            pos_ids.append(b["position_ids"] + [0] * pad)
        if mrope:
            p = b.get("mrope_position_ids") or [list(range(n))] * 3
            pos3.append([axis + [0] * pad for axis in p])
    if mrope:
        position_ids = torch.tensor(pos3, dtype=torch.long).permute(1, 0, 2).contiguous()
    else:
        position_ids = torch.tensor(pos_ids, dtype=torch.long) if packed else None
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
        position_ids,
        torch.tensor(seg_ids, dtype=torch.long) if packed else None,
    )


def sample(model, cache, tokenizer, generator, build_prompt, prompt, max_new_tokens=48):
    """Quick native generation for live progress feedback (uses the same chat
    format as training, so a metharme-trained adapter previews meaningfully)."""
    text = build_prompt(prompt)
    resp = generator.generate(
        prompt=text, max_new_tokens=max_new_tokens,
        add_bos=False, completion_only=True,
    )
    return resp.strip().replace("\n", " ")


def compute_even_split_budgets(model, config, margin_gb):
    """Per-device ``use_per_device`` budgets (GB) that steer the autosplit
    loader into an approximately EVEN split of the weights across all visible
    CUDA devices (--split-even).

    The stock autosplit is greedy: it packs device 0 until an allocation
    fails, then advances -- so device 0 ends up wall-to-wall weights with no
    training headroom while the last device sits half empty. Rather than
    patching the loader, this measures each module's weight bytes from the
    safetensors headers (``config.stc``; modules that load to CPU, i.e. the
    embedding, are excluded) and solves the contiguous balanced-partition
    problem over the module sequence (binary search on capacity + greedy
    feasibility -- the module order is fixed by the forward, only the
    boundaries move). Each device's byte share plus ``margin_gb`` of
    load-time slack (dummy-forward activations, KV cache slices, dequant
    temporaries) becomes its ``use_per_device`` cap, which the loader
    enforces via per-device memory fractions and lifts after loading.

    Approximate by design: the slack lets a realized boundary drift a layer
    or two past the ideal one. A tied-embeddings LM head has no tensors of
    its own in the file, so it's estimated at the embedding's size.

    Caches attached before ``model.load()`` (the live-sample generator's KV
    cache here, the rollout sampler's in the EBFT trainer) allocate their
    tensors on each layer's device DURING load, so they compete with the
    weights for the per-device budget. Each module is therefore weighed as
    weight bytes + its attached cache/state bytes, via the cache classes'
    own ``storage_size()`` (exact -- the tensors exist on the meta device
    before load). This covers paged KV layers AND recurrent per-slot states
    (SWA ring buffers, GDN states): on a sliding-window model the SWA
    states dominate -- gemma12b with a 32k-token cache carries ~190 MB per
    sliding layer (16 batch slots x window), ~7.5 GB total, dwarfing both
    the 0.5 GB of paged KV and the weight-only budgets that used to be
    computed here ("Insufficient VRAM in split").
    """
    n = torch.cuda.device_count()
    if n < 2:
        raise SystemExit("--split-even needs >= 2 visible CUDA devices")

    def attached_cache_bytes(module):
        b = 0
        for sub in module:
            layers = list(getattr(sub, "cache_layers", ())) \
                   + list(getattr(sub, "recurrent_layers", ()))
            for cl in layers:
                if hasattr(cl, "storage_size"):
                    b += cl.storage_size()
                if hasattr(cl, "overhead_size"):
                    b += cl.overhead_size()
        return b

    cpu_bytes = [sum(config.stc.get_tensor_sizes(m.key)) for m in model.modules
                 if m.caps.get("prefer_cpu")]
    sizes = []
    head_bytes = 0
    for m in model.modules:
        if m.caps.get("prefer_cpu"):
            continue
        b = sum(config.stc.get_tensor_sizes(m.key))
        if b == 0 and m.caps.get("logits_output") and cpu_bytes:
            b = max(cpu_bytes)   # tied head: materialized from the embedding
        if m.caps.get("logits_output"):
            head_bytes = b
        b += attached_cache_bytes(m)
        sizes.append(b)

    # Smallest capacity that fits the sequence in <= n contiguous chunks.
    def chunks_at(cap):
        parts, acc = 1, 0
        for s in sizes:
            if acc + s > cap and acc > 0:
                parts += 1
                acc = 0
            acc += s
        return parts

    lo, hi = max(sizes), sum(sizes)
    while lo < hi:
        mid = (lo + hi) // 2
        if chunks_at(mid) <= n:
            hi = mid
        else:
            lo = mid + 1

    budgets, acc = [], 0
    for s in sizes:
        if acc + s > lo and acc > 0:
            budgets.append(acc)
            acc = 0
        acc += s
    budgets.append(acc)
    # The margin is really a transient estimate: the loader frees each
    # module's load/dummy-forward temporaries before packing the next one, so
    # a device stops accepting layers at (cap - transient) -- an oversized
    # margin makes earlier devices steal layers past their even share. The
    # LAST device is exempt from that failure mode (it only ever gets the
    # remainder), and it hosts the largest transient: materializing the
    # logits head (a tied head is copied out of the CPU embedding at ~2 GB
    # for a 256k vocab) roughly doubles the head's own footprint while
    # loading. Give the last device that much extra instead of inflating the
    # shared margin.
    budgets = [float(b) / 1024**3 + margin_gb if b > 0 else 0.0 for b in budgets]
    for i in reversed(range(len(budgets))):
        if budgets[i] > 0:
            budgets[i] += float(head_bytes) / 1024**3
            break
    return budgets


def main():
    """Run the trainer with failure capture: any exception or non-zero
    SystemExit appends a ``status=failed`` row (with phase + error summary) to
    the run-log CSV and the full traceback to ``<run_log>.errors.log`` before
    re-raising, so failed experiments are documented automatically. Ctrl-C
    outside the training loop's own handler is recorded as ``interrupted``."""
    try:
        _run_main()
    except KeyboardInterrupt as e:
        _log_failure("interrupted", e)
        raise SystemExit(130)
    except SystemExit as e:
        # SystemExit(0) is the normal Ctrl-C exit path (already logged);
        # a message/non-zero code is a guard-rail abort worth recording.
        if e.code not in (0, None):
            _log_failure("failed", e)
        raise
    except BaseException as e:
        _log_failure("failed", e)
        raise


def _run_main():
    # Line-buffer stdout/stderr so the per-step progress lines (and interleaved
    # eval/sample/checkpoint lines) flush on each newline. Python block-buffers
    # stdout when it isn't a TTY -- i.e. exactly when the run is redirected to a
    # file or piped through tee -- which otherwise holds every step line in an
    # ~8KB buffer and dumps them all at once when the process exits.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (already line-buffered, or wrapped)

    # This is the single-process trainer (--parallel single|split). Launched under
    # torchrun (RANK/WORLD_SIZE in env) it would silently run N independent copies,
    # so redirect to the DDP entry point with a clear one-liner instead of the
    # confusing argparse error you'd get from a stray --parallel ddp.
    if os.environ.get("RANK") is not None or os.environ.get("WORLD_SIZE") is not None:
        raise SystemExit(
            "qlora_train_native.py is single-process (--parallel single|split). "
            "For multi-GPU DDP under torchrun use training/qlora_train_native_ddp.py "
            "(note: --lora-r not --r; no --parallel / --sample-every)."
        )
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to a local EXL3 model dir")
    ap.add_argument("--out", default="out/exl3_qlora_adapter")
    ap.add_argument("--device", default="cuda:0",
                    help="single-device load target (ignored when --parallel split)")
    ap.add_argument("--parallel", choices=["single", "split"], default="single",
                    help="single: one GPU; split: layer-autosplit the frozen base "
                         "across visible GPUs (memory, for models too big for one card)")
    ap.add_argument("--reserve-per-device", nargs="*", type=float, default=None, metavar="GB",
                    help="(split) GB to reserve per device; negative excludes a device")
    ap.add_argument("--split-even", nargs="?", type=float, const=2.0, default=None,
                    metavar="MARGIN_GB",
                    help="(split) Balance the layer split across GPUs instead of the "
                         "default greedy fill-device-0-first: each device is capped to "
                         "its even share of the model's weight bytes (measured from the "
                         "safetensors headers) plus any pre-attached KV-cache pages, "
                         "plus MARGIN_GB of load-time slack "
                         "(default 2.0), leaving every card similar training headroom. "
                         "Raise the margin if loading fails with 'Insufficient VRAM in "
                         "split'; shrink it if the split comes out lopsided. Mutually "
                         "exclusive with --reserve-per-device/--use-per-device.")
    ap.add_argument("--use-per-device", nargs="*", type=float, default=None, metavar="GB",
                    help="(split) GB budget per device; caps a card to force/tune the split")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0,
                    help="PEFT-style dropout on each per-linear LoRA branch's "
                         "input (frozen base path never dropped; train-time "
                         "only, off in eval/inference). Typical 0.05-0.1; mild "
                         "regularizer for small datasets / many epochs. Not "
                         "applied to embed/head adapters. Incompatible with "
                         "--quant-aware ste.")
    ap.add_argument("--use-rslora", action="store_true",
                    help="Rank-stabilized LoRA scaling: scale = alpha/sqrt(r) "
                         "instead of alpha/r. At a FIXED rank this is just an "
                         "alpha rescale (r=64: alpha/8 -> same scale), but it "
                         "keeps the effective scale stable across rank sweeps.")
    ap.add_argument("--init-lora", choices=["default", "pissa", "qerr", "eva"],
                    default="default",
                    help="Adapter initialization. default: kaiming A / zero B. "
                         "pissa: top-r principal components of the frozen base "
                         "(trained against a frozen-offset residual; adapter "
                         "exports as a converted rank-2r standard LoRA). "
                         "qerr: top-r SVD of the quantization error vs the "
                         "ORIGINAL model (needs --init-ref-model); training "
                         "starts from the closest rank-r repair of the bf16 "
                         "model. eva: A = top-r right-singular vectors of each "
                         "target's input activations, streamed from a short "
                         "pre-pass of the training data through the quantized "
                         "forward (B stays 0, so step 0 is exactly the base). "
                         "All need a validated step-0 gate: run "
                         "qlora_validate_native.py --init-lora first.")
    ap.add_argument("--init-svd-niter", type=int, default=16,
                    help="Randomized-SVD subspace iterations for --init-lora "
                         "(PiSSA's fast-SVD recipe; 0 = exact full SVD, much "
                         "slower). eva caps this at 8 for its incremental "
                         "sketch updates. Default 16.")
    ap.add_argument("--init-ref-model", default=None,
                    help="Path to the ORIGINAL (unquantized) HF model dir; "
                         "required by --init-lora qerr to form the "
                         "quantization error.")
    ap.add_argument("--init-eva-tokens", type=int, default=65536,
                    help="Token budget for the --init-lora eva activation "
                         "pre-pass (drawn in order from the training set; "
                         "no gradients). Default 65536.")
    ap.add_argument("--quant-aware", choices=["none", "noise", "ste"],
                    default="none",
                    help="Quantization-aware LoRA training, so the trained "
                         "delta survives the merge-and-requantize deploy path "
                         "by construction. noise: fresh per-micro-batch "
                         "pseudo-quantization noise on the adapted frozen "
                         "weights (differentiable, the NIPQ proxy for the "
                         "requantize the merged model will undergo). ste: the "
                         "effective adapter delta is snapped to a quant-floor "
                         "grid in the forward with a straight-through "
                         "gradient (sub-floor delta components contribute "
                         "nothing, exactly as after a requantize). Training "
                         "only -- eval/saves always use exact weights.")
    ap.add_argument("--quant-aware-scale", type=float, default=1.0,
                    help="Multiplier on the per-layer quantization-error "
                         "scale used by --quant-aware (default 1.0).")
    ap.add_argument("--quant-aware-ref-model", default=None,
                    help="Path to the ORIGINAL (unquantized) HF model dir to "
                         "MEASURE the per-channel quantization error for "
                         "--quant-aware (exact). Defaults to --init-ref-model "
                         "when that is set; otherwise the error scale is "
                         "estimated from the trellis bitrate (std·2^-K).")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="AdamW weight decay on the LoRA params (default 0.01).")
    ap.add_argument("--optim", choices=["adamw", "adamw8bit", "paged_adamw8bit"],
                    default="adamw",
                    help="Optimizer. 'adamw' = torch AdamW (fp32 moments, 8 "
                         "bytes/param). 'adamw8bit' / 'paged_adamw8bit' = "
                         "bitsandbytes 8-bit AdamW (~2 bytes/param) -- cuts "
                         "optimizer state ~4x, the lever for fitting bigger r / "
                         "longer context on tight VRAM. 'paged_' offloads optimizer "
                         "state to host on spikes (needs bitsandbytes installed).")
    ap.add_argument("--scheduler", choices=["none", "linear", "cosine"],
                    default="none",
                    help="LR schedule after warmup: none (constant), linear "
                         "decay to 0, or cosine decay to 0.")
    ap.add_argument("--warmup-ratio", type=float, default=0.0,
                    help="Fraction of total steps spent linearly warming up the "
                         "LR from 0 (e.g. 0.05-0.1). Ignored if --warmup-steps>0.")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="Absolute warmup steps; overrides --warmup-ratio when >0.")
    ap.add_argument("--epochs", type=float, default=0.0,
                    help="If >0, set --steps to cover this many passes over the "
                         "training data (one step = batch*grad-accum examples), "
                         "so the schedule length matches the epoch count.")
    ap.add_argument("--steps", type=int, default=1000,
                    help="Training steps (ignored when --epochs>0). ~steps*batch "
                         "examples seen; aim for >=1 epoch to pick up a style.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument(
        "--dataset",
        default="superdrew100/UwU_Alpaca_data",
        help="HF dataset id. Default is the UwU-furry Alpaca style set.",
    )
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--instruction-key", default="instruction",
                    help="Column holding the prompt/instruction")
    ap.add_argument("--context-key", default="input",
                    help="Optional extra-context column; absent columns are ignored "
                         "(Alpaca uses 'input', Dolly uses 'context')")
    ap.add_argument("--response-key", default="output",
                    help="Column holding the target response (Alpaca: 'output', "
                         "Dolly: 'response')")
    ap.add_argument("--messages-key", default=None,
                    help="Column holding OpenAI-style messages (e.g. 'messages' "
                         "for UnstableLlama/semancy). Single-turn rows: user turn "
                         "-> prompt, assistant turn -> supervised response, as "
                         "before. Multi-turn rows: every turn is rendered and "
                         "ONLY assistant turns are supervised (user/system masked "
                         "to -100); needs an explicit --prompt-format (auto is "
                         "single-turn only). With --prompt-format jinja, rows "
                         "may also carry tools and template_vars/"
                         "chat_template_kwargs columns, and messages may carry "
                         "reasoning_content/tool_calls. "
                         "--instruction/context/response-key are ignored.")
    ap.add_argument("--prompt-format",
                    choices=["auto", "mistral", "metharme", "gemma4-nothink",
                             "llama3", "qwen3.5", "qwen3.5-nothink", "chatml",
                             "jinja"],
                    default="auto",
                    help="Chat format. auto: the model's native template "
                         "(Llama-3, Mistral [INST], mistral3 [SYSTEM_PROMPT]/[INST]). "
                         "mistral: explicit <s>[INST]{q}[/INST]{a}</s> (= auto for "
                         "the mistral3 arch, e.g. Mistral-Medium-3.5). metharme: "
                         "Pygmalion <|user|>{q}<|model|>{a}</s>. gemma4-nothink: "
                         "<|turn>user\\n{q}<turn|>\\n<|turn>model\\n<|channel>thought\\n"
                         "<channel|>{a} with the thought span pre-closed empty (no "
                         "reasoning trained). llama3: explicit Llama-3 headers "
                         "(= auto for the llama arch). qwen3.5: plain ChatML "
                         "(= auto for qwen3/3.5; use when responses carry their own "
                         "<think> spans). qwen3.5-nothink: ChatML with an empty "
                         "<think>\\n\\n</think>\\n\\n pre-closed in the masked prompt, "
                         "matching the inference-side no-think prefill. "
                         "jinja: the model directory's own Jinja chat template "
                         "(tokenizer_config.json / chat_template.jinja), with "
                         "tool roles, tool_calls, reasoning_content, per-row "
                         "tools + template_vars/chat_template_kwargs columns, "
                         "and exact per-turn masks via incremental rendering "
                         "(see training/chat_jinja.py; --chat-template-file / "
                         "--template-vars below). "
                         "EOS ends the turn for mistral/metharme; <turn|> for "
                         "gemma4-nothink; <|eot_id|> for llama3; <|im_end|> for "
                         "qwen3.5/qwen3.5-nothink; whatever the template "
                         "appends after assistant content for jinja.")
    ap.add_argument("--chat-template-file", default=None,
                    help="(jinja) Path to a Jinja template file to use instead "
                         "of the one in the model directory.")
    ap.add_argument("--template-vars", default=None,
                    help="(jinja) JSON object of extra template variables in "
                         "every render, e.g. '{\"enable_thinking\": false}'. "
                         "Per-row template_vars/chat_template_kwargs columns "
                         "override these per key.")
    ap.add_argument("--strip-sys-prompt-extras", action="store_true",
                    help="(jinja) Delete template-injected 'Reasoning "
                         "strength:', 'Knowledge cutoff:' and 'Current date:' "
                         "lines from every rendered prompt (OFF by default). "
                         "Harmony-style templates (Muse-Glimmer, gpt-oss) "
                         "synthesize these into the system block whenever a row "
                         "has no system message of its own; none of it is in "
                         "your data. The date line in particular comes from "
                         "strftime_now(), so without this flag the rendered "
                         "corpus changes from one day to the next.")
    ap.add_argument("--clean-text", action="store_true",
                    help="Strip [stage directions]/*actions* and normalize "
                         "whitespace before training (OFF by default). Helps "
                         "play-script style sets; leave off for reasoning / code / "
                         "markdown data, where brackets and structure are content.")
    ap.add_argument("--no-clean-text", action="store_true",
                    help=argparse.SUPPRESS)  # deprecated: cleaning is now opt-in
    ap.add_argument("--min-response-words", type=int, default=3,
                    help="Drop rows whose cleaned response is shorter than this")
    ap.add_argument("--uppercase-response", action="store_true",
                    help="Smoke test: train the model to RESPOND IN ALL CAPS. A "
                         "maximally dense/consistent transform that must show in "
                         "generation if the training path works at all.")
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle the training rows once (deterministically) "
                         "before the --val-frac carve and before training, so the "
                         "held-out split is a random sample and training order is "
                         "randomized. Matched across arms given the same seed.")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="Seed for --shuffle (also the random-subset seed when "
                         "--max-samples caps the rows). Default 0.")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--pack", action="store_true",
                    help="Sample packing: concatenate multiple training documents "
                         "into each --seq-len sequence instead of padding each to "
                         "--seq-len, so short-answer data stops wasting most of the "
                         "forward on pad tokens. Documents stay isolated (per-doc "
                         "RoPE reset + block-diagonal attention; flash-varlen on "
                         "CUDA fp16/bf16). Only the training set is packed; the "
                         "held-out eval stays per-example for comparable losses.")
    ap.add_argument("--pack-algo", choices=["bfd", "nextfit"], default="bfd",
                    help="Bin-packing strategy for --pack. bfd (default): best-fit "
                         "decreasing -- typically 97%%+ fill vs ~80-85%% for the old "
                         "next-fit, i.e. ~1.15-1.2x more real tokens per step. "
                         "nextfit: the pre-Session-11 arrival-order behavior, kept "
                         "for A/B comparison.")
    ap.add_argument("--vision", action="store_true",
                    help="Image+text SFT on a vision-language EXL3 model (Qwen2.5/3/"
                         "3.5-VL, Gemma3/4, Mistral3, ...). Loads the model's own "
                         "vision component (frozen, as at inference) and, for every "
                         "--messages-key row whose content parts carry images "
                         "(Axolotl/OpenAI layout: {type: image, path|url|base64|"
                         "image}, or bare {type: image} parts drawn from the row's "
                         "--images-key column), splices the image features into the "
                         "token stream at the parts' positions. Only the language "
                         "model's adapters train. Text-only rows in the same set "
                         "train as usual. Not combinable with --pack. Gate a new base "
                         "with qlora_validate_native.py --image first.")
    ap.add_argument("--images-key", default="images",
                    help="(--vision) row column holding the images that bare "
                         "{type: image} parts refer to, in order (default: images).")
    ap.add_argument("--image-max-pixels", type=int, default=0,
                    help="(--vision) downscale images to at most this many pixels "
                         "(aspect kept) BEFORE the arch's own preprocessing -- the "
                         "lever on image token count / VRAM (e.g. 1000000 -> ~1.3k "
                         "tokens on Qwen-VL). 0 = the model's own bounds.")
    ap.add_argument("--vision-cache-gb", type=float, default=8.0,
                    help="(--vision) CPU RAM budget for cached image features "
                         "(fp16). Images past the budget are re-encoded each time "
                         "they are batched (the vision tower then stays loaded); "
                         "when everything fits, the tower is unloaded after the "
                         "dataset build to free its VRAM.")
    ap.add_argument("--vision-device", default=None,
                    help="(--vision) device for the vision tower (default: the "
                         "training device / first split device).")
    ap.add_argument("--inspect", type=int, default=0, metavar="N",
                    help="Tokenization check: decode the first N built examples "
                         "(prompt span vs supervised response span + whether the "
                         "response was truncated by --seq-len), then exit without "
                         "training. Run this once to verify a new dataset/schema.")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="Target module leaf names (default: attn+mlp projections). "
                         "On a MoE model the plain gate/up/down_proj names adapt "
                         "the dense/shared-expert paths only; add expert_gate_proj/"
                         "expert_up_proj/expert_down_proj to also adapt the routed "
                         "experts (one adapter pair per expert per layer -- see "
                         "--expert-r). The router is always frozen.")
    ap.add_argument("--mtp-targets", nargs="*", default=None, metavar="LEAF",
                    help="ALSO load and train the model's MTP (multi-token "
                         "prediction / self-speculative draft) head -- Qwen3.5/"
                         "3.6 family -- adapting these leaf names INSIDE it "
                         "(same vocabulary as --targets, plus 'fc', the head's "
                         "[2d->d] input projection; bare flag = the default "
                         "attn+mlp list + fc). --targets never reaches into the "
                         "head and --mtp-targets never reaches into the trunk, "
                         "so '--targets --mtp-targets' (empty trunk list) trains "
                         "the MTP tensors ALONE on the frozen trunk. The head "
                         "learns the trunk's next-token task one step ahead "
                         "(state i-1 + token i -> token i+1, exactly the "
                         "generator's draft wiring); its adapters save into the "
                         "same adapter dir under mtp.* keys.")
    ap.add_argument("--mtp-loss-weight", type=float, default=1.0,
                    help="With --mtp-targets AND a training trunk: weight of the "
                         "head's loss in 'trunk + w * mtp'. Ignored when only the "
                         "head trains (its loss is then THE loss).")
    ap.add_argument("--freeze-trunk", action="store_true",
                    help="Stage 2 of the two-stage MTP recipe: --resume a "
                         "finished trunk adapter with the SAME --r/--targets, "
                         "freeze it (it still applies in the forward and is "
                         "re-exported), and train only --mtp-targets so the "
                         "head is fitted to the tuned trunk it will draft for. "
                         "The trunk then runs under no_grad (no checkpointing / "
                         "saved activations). Implies --reset-optimizer.")
    ap.add_argument("--mtp-device", default=None,
                    help="(--mtp-targets) device for the MTP head (default: the "
                         "trunk's output device -- --device, or the last split "
                         "device under --parallel split).")
    ap.add_argument("--expert-r", type=int, default=None,
                    help="LoRA rank for ROUTED-expert adapters (expert_* targets) "
                         "when it should differ from --r. On a many-expert model "
                         "the per-expert adapters dominate the trainable size; "
                         "PEFT's MoE recipe uses ~r/num_experts (min 1). Default: "
                         "same as --r.")
    ap.add_argument("--train-embeddings", action="store_true",
                    help="Also FULLY train the input embeddings (modules_to_save), "
                         "not just LoRA. Saved to modules_to_save.safetensors. Big "
                         "(vocab x hidden) -- raises VRAM and, under DDP, the "
                         "per-step grad all-reduce. On a tied model this also "
                         "trains the head (shared weight).")
    ap.add_argument("--train-head", action="store_true",
                    help="Also FULLY train the LM head (modules_to_save). Switches "
                         "the loss off the fused frozen-head path to a supervised-"
                         "position cross-entropy so the head gets a gradient. On a "
                         "tied model this is equivalent to --train-embeddings.")
    ap.add_argument("--lora-embed", action="store_true",
                    help="Train a rank-r LoRA on the input embedding instead of "
                         "fully (mutually exclusive with --train-embeddings). Far "
                         "cheaper: r*(vocab+hidden) params, GPU-resident, no offload "
                         "needed. A low-rank shift of the whole embedding (use "
                         "PEFT-style trainable-tokens instead if you only added new "
                         "tokens). Saved to lora_modules.safetensors (merge-path).")
    ap.add_argument("--lora-head", action="store_true",
                    help="Train a rank-r LoRA on the LM head instead of fully "
                         "(mutually exclusive with --train-head). Adds a low-rank "
                         "delta to the head logits at the supervised positions; "
                         "memory scales with supervised tokens, params are tiny. "
                         "Saved to lora_modules.safetensors (merge-path).")
    ap.add_argument("--module-lora-lr-mul", type=float, default=1.0,
                    help="LR multiplier for the embed/head LoRA adapters "
                         "(--lora-embed/--lora-head): their own optimizer param "
                         "group at lr * this, following the same schedule. S64 "
                         "found the head LoRA ~8x undertrained at the shared LR "
                         "(gain monotone to x8, worse at x16). No effect on the "
                         "per-linear adapters or fully-trained modules_to_save.")
    ap.add_argument("--offload-embed-head-optim", action="store_true",
                    help="Put the fully-trained embedding/LM-head optimizer on CPU "
                         "(torchao CPUOffloadOptimizer) with bf16 stochastic-rounding "
                         "master weights, so the embed/head Adam state never sits on "
                         "the GPU -- frees ~12 bytes/param of the (huge, untied) "
                         "embed/head matrices. Requires torchao and "
                         "--train-embeddings/--train-head. Single-process only (not "
                         "the DDP arm); these params are excluded from grad clipping "
                         "and follow the same LR schedule as the LoRA group.")
    ap.add_argument("--offload-activations", action="store_true",
                    help="Offload the grad-checkpointed block activations to CPU RAM "
                         "(pinned) to free GPU memory for longer context / bigger "
                         "batch. Needs gradient checkpointing (on by default) + CUDA. "
                         "Wraps only the decoder block loop.")
    ap.add_argument("--offload-mode", choices=["async", "sync"], default="async",
                    help="How --offload-activations moves the data: async (default; "
                         "double-buffered side-stream copies that overlap compute, "
                         "value-exact) or sync (torch save_on_cpu, the pre-S36 "
                         "behavior -- blocking copies; keep for A/B or retain_graph).")
    ap.add_argument("--vram-spillover", action="store_true",
                    help="Linux only: allocate all CUDA tensors as unified memory "
                         "(cudaMallocManaged) so a run that slightly exceeds VRAM "
                         "spills cold pages to host RAM instead of insta-OOMing -- "
                         "the behavior Windows' driver sysmem fallback provides by "
                         "default. Slowdown is proportional to the spill (a few %% "
                         "over VRAM is usable; big oversubscription crawls). Peak-"
                         "VRAM reporting is unavailable in this mode, and --parallel "
                         "split is rejected (its capacity probing is meaningless "
                         "when allocations never fail). See training/uvm_allocator.py.")
    ap.add_argument("--use-liger", action="store_true",
                    help="Route RMSNorm (2D/3D norms) and SwiGLU (silu only) through "
                         "Liger Triton kernels for lower activation memory + speed. "
                         "Needs liger-kernel + CUDA fp16/bf16; eager/fp32/CPU paths "
                         "are unchanged. Changes numerics slightly -- run "
                         "qlora_validate_native.py --use-liger to confirm parity first.")
    ap.add_argument("--compute-dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--attn-impl", choices=["auto", "eager", "flash"], default="auto",
                    help="Attention kernel: auto (FlashAttention-2 when the "
                         "flash_attn package is importable and the run is CUDA "
                         "fp16/bf16, else eager), flash (require it), or eager "
                         "(the reference; O(t^2) memory). Flash is O(t) memory -- "
                         "the lever for long-context training.")
    ap.add_argument("--ce-chunk", type=int, default=1024)
    ap.add_argument("--head-vocab-chunk", type=int, default=0,
                    help="Reconstruct + matmul the frozen LM head in vocab-column "
                         "chunks of this many columns (0 = off, single-shot). Bounds "
                         "the head's peak memory on the OUTPUT device -- the full "
                         "[hidden, vocab] reconstruction + fp32 upcast is the spike "
                         "for big-vocab models (e.g. Gemma 262k). Try 32768. Same "
                         "loss/grad as off; no extra dequant cost (vocab-outer loop).")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--sample-every", type=int, default=25,
                    help="Generate a sample completion every N steps (0 to disable)")
    ap.add_argument("--sample-prompt", default="Tell me about your day.")
    ap.add_argument("--save-every", type=int, default=0,
                    help="Overwrite the adapter at --out every N steps (0 = only "
                         "at the end). The adapter is also saved on Ctrl-C. This "
                         "keeps a single latest copy; use --checkpoint-every for a "
                         "retained history.")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="Every N steps, save a RETAINED checkpoint to "
                         "--out/checkpoint-<step> (kept; not overwritten), so you "
                         "build a history to roll back to or pick from. Independent "
                         "of --save-every (latest at --out) and --save-best (best at "
                         "--out). 0 disables.")
    ap.add_argument("--keep-checkpoints", type=int, default=0,
                    help="Cap the number of --checkpoint-every dirs to keep, "
                         "deleting the oldest (0 = keep all). Useful with "
                         "--train-embeddings/--train-head, where each checkpoint is "
                         "large.")
    ap.add_argument("--resume", default=None,
                    help="Adapter dir to resume from (continues those weights). If "
                         "the dir holds a trainer_state.pt (any --checkpoint-every / "
                         "--save-* dir from this trainer), the optimizer, LR "
                         "schedule and step counter are ALSO restored so the run "
                         "continues seamlessly; pass --reset-optimizer to skip that "
                         "(cold AdamW, schedule from step 0). --r/--targets must "
                         "match the checkpoint.")
    ap.add_argument("--reset-optimizer", action="store_true",
                    help="With --resume, load only the weights and start the "
                         "optimizer/LR-schedule/step fresh (the old resume "
                         "behavior). Use when changing LR/schedule or resuming "
                         "across a different device topology.")
    ap.add_argument("--eval-split", default=None,
                    help="Use this split of the dataset (e.g. 'test') as the "
                         "held-out eval set, instead of carving --val-frac off "
                         "train. Real held-out data; takes precedence over "
                         "--val-frac.")
    ap.add_argument("--eval-dataset", default=None,
                    help="Dataset id/path for --eval-split (defaults to --dataset).")
    ap.add_argument("--eval-config", default=None,
                    help="HF dataset config for the primary eval set (parity "
                         "with --eval2-config).")
    ap.add_argument("--eval-text-key", default=None,
                    help="If set, treat the PRIMARY eval set as PLAIN TEXT and "
                         "compute a language-modeling loss over packed --seq-len "
                         "blocks, exactly like --eval2-text-key. --save-best then "
                         "tracks that LM loss.")
    ap.add_argument("--eval-max-samples", type=int, default=0,
                    help="Cap source rows for the primary eval set (0 = all; "
                         "parity with --eval2-max-samples).")
    ap.add_argument("--eval-max-blocks", type=int, default=0,
                    help="Cap packed LM blocks for --eval-text-key (0 = all; "
                         "parity with --eval2-max-blocks).")
    ap.add_argument("--eval2-dataset", default=None,
                    help="A SECOND held-out eval set, reported alongside the "
                         "primary one each --eval-every and at the end, so you can "
                         "watch them move together (e.g. your test set vs "
                         "wikitext). --save-best stays keyed on the PRIMARY eval.")
    ap.add_argument("--eval2-split", default="test",
                    help="Split for --eval2-dataset (default 'test').")
    ap.add_argument("--eval2-config", default=None,
                    help="HF dataset config for --eval2-dataset (e.g. "
                         "'wikitext-2-raw-v1' for the 'wikitext' dataset).")
    ap.add_argument("--eval2-text-key", default=None,
                    help="If set, treat --eval2-dataset as PLAIN TEXT and compute "
                         "a language-modeling loss over packed --seq-len blocks "
                         "(every token supervised) -- e.g. 'text' for wikitext. "
                         "If unset, --eval2-dataset is built as a second SFT eval "
                         "using the same instruction/messages keys.")
    ap.add_argument("--eval2-max-samples", type=int, default=0,
                    help="Cap source rows for --eval2-dataset (0 = all).")
    ap.add_argument("--eval2-max-blocks", type=int, default=0,
                    help="Cap the number of packed LM blocks for --eval2-text-key "
                         "(0 = all). Use this to size eval2 to roughly match the "
                         "primary eval set (e.g. wikitext packs into far more "
                         "blocks than your test set has examples); --eval2-max-"
                         "samples caps source rows, which is unpredictable after "
                         "packing.")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="Hold out this fraction of train for held-out eval loss "
                         "(deterministic; the SAME split as qlora_train_bnb.py "
                         "given the same dataset/seed). Ignored if --eval-split is "
                         "set. 0 = no eval.")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="Also report held-out loss every N steps (needs "
                         "--val-frac > 0). 0 = only at the end.")
    ap.add_argument("--save-best", action="store_true",
                    help="Save the adapter only when held-out loss improves "
                         "(needs --val-frac + --eval-every), so a long run keeps "
                         "the best checkpoint instead of an overfit endpoint.")
    ap.add_argument("--run-log", default="qlora_runs.csv",
                    help="Append one metadata row per run to this CSV (model, "
                         "hyperparameters, start/end/best-val loss, timing, tok/s, "
                         "peak VRAM, ...). Written on normal finish, on Ctrl-C, AND "
                         "on failure (status=failed + error; traceback goes to "
                         "<run-log>.errors.log). Empty string disables.")
    ap.add_argument("--profile-dequant", type=int, default=0, metavar="N",
                    help="Measure frozen-weight (trellis) reconstruction time for "
                         "the first N training steps, print its share of the step "
                         "wall time, then disable. Adds a device sync around every "
                         "reconstruction while active, so expect those N steps to "
                         "run slower; the reported %% is still representative.")
    ap.add_argument("--torch-profile", type=int, default=0, metavar="N",
                    help="Run torch.profiler (CPU+CUDA, python stacks, shapes) "
                         "over N steady-state training steps (3 skipped + 2 "
                         "warmup steps first) and write a chrome trace + "
                         "key-averages tables to <out>/torch_profile/. The "
                         "stack attribution separates trellis-reconstruct / "
                         "Hadamard-transform / base-GEMM / LoRA-GEMM time "
                         "(Session 32 forward-gap split). Profiling overhead "
                         "is real; don't trust tok/s from a profiled run.")
    ap.add_argument("--ga-loss", choices=["token", "mean"], default="token",
                    help="Gradient-accumulation loss weighting. token (default): "
                         "weight each micro-batch by its supervised-token share, "
                         "so the step gradient equals one big batch (the Oct-2024 "
                         "HF/Unsloth GA fix). mean: the pre-Session-11 mean-of-"
                         "means (over-weights tokens in short micro-batches), "
                         "kept for reproducing old runs. No-op at --grad-accum 1.")
    ap.add_argument("--dequant-mode", choices=["fast", "legacy"], default="fast",
                    help="Frozen-weight dequant path (audit A1). fast (default): "
                         "reconstruct only the inner trellis weight and apply the "
                         "Hadamard/sign transforms to the activations (the "
                         "inference reconstruct_hgemm math). legacy: the original "
                         "full get_weight_tensor per reconstruction, kept for A/B "
                         "and reproducing old runs.")
    ap.add_argument("--dequant-cache", action="store_true",
                    help="Opt-in: reuse each frozen weight between the checkpoint-"
                         "recompute forward and its backward (3 -> 2 dequants per "
                         "step) at the cost of one block's frozen weights held "
                         "live at a time. Box-measured tradeoffs (Session 30): "
                         "under the default fast dequant path this is a net LOSS "
                         "-- reconstructions are so cheap that cache bookkeeping "
                         "costs ~1-4%% tok/s AND +0.5-1.5 GB peak VRAM (worst on "
                         "MoE). Only worth trying with --dequant-mode legacy, "
                         "where each avoided reconstruction is 5-7 ms.")
    ap.add_argument("--no-report", action="store_true",
                    help="Disable the local run report. By default every run "
                         "with an --out writes a self-contained report to "
                         "<out>/run_report/report.html (config, per-step metrics, "
                         "evals, summary; inline charts, no third-party account) "
                         "-- the default shareable dashboard. This turns it off "
                         "for throwaway runs.")
    ap.add_argument("--live-report", action="store_true",
                    help="Serve a LIVE run monitor from a localhost http thread "
                         "and open it in the browser at run start: the report "
                         "page redraws its charts as metrics are logged, plus a "
                         "step viewer showing the decoded text of the exact "
                         "examples any step trains on (past or future -- the "
                         "data order is deterministic, so batches are computed "
                         "on demand from memory; the dataset is never written "
                         "to disk or into report.html, which stays the "
                         "dataset-free shareable artifact). Requires the local "
                         "report (i.e. not --no-report).")
    ap.add_argument("--live-report-port", type=int, default=0,
                    help="Port for --live-report (default 0 = pick a free one).")
    ap.add_argument("--run-name", default="",
                    help="Name for the local run report (report title / compare "
                         "legend). Default: basename of --out. Independent of "
                         "wandb; --wandb-run-name still names the wandb run.")
    ap.add_argument("--wandb-project", default="",
                    help="Log the run to Weights & Biases under this project "
                         "(opt-in, OFF by default -- the local report is the "
                         "default path; empty = wandb never imported). Logs the "
                         "per-step readout (loss/ema/grad/lr/|dB|/tok-s/epoch), "
                         "eval losses as they happen, and a final summary; the "
                         "run config mirrors the run-log CSV row so runs are "
                         "comparable across both.")
    ap.add_argument("--wandb-run-name", default="",
                    help="wandb run name (default: basename of --out).")
    ap.add_argument("--wandb-entity", default="",
                    help="wandb entity (user/team; empty = account default).")
    args = ap.parse_args()

    # Set before anything can build a renderer, so every jinja_renderers() call
    # in this process (dataset build, eval build, --sample-every prompts) agrees.
    from chat_jinja import set_strip_sys_prompt_extras
    set_strip_sys_prompt_extras(args.strip_sys_prompt_extras)

    # UVM spillover must be installed before the first CUDA tensor exists, so
    # this runs before anything touches a device (model load is the first).
    if args.vram_spillover:
        if args.parallel == "split":
            raise SystemExit(
                "--vram-spillover is not compatible with --parallel split: the "
                "autosplit loader probes real device capacity (memory fractions "
                "+ OOM boundaries), which is meaningless when managed-memory "
                "allocations never fail. Use --parallel single, or split "
                "without spillover.")
        import uvm_allocator
        uvm_allocator.enable()

    from exllamav3.training import backbone as _backbone_cfg
    _backbone_cfg.set_dequant_mode(args.dequant_mode)

    # Dataset-mix continuity: a fresh run snapshots its data files into
    # <out>/dataset/ (copy + sha256); a resume verifies the config's paths
    # against that record and may substitute the snapshot, so the mix a run
    # trained on stays knowable and reloadable even after the source jsonl is
    # edited or regenerated. Done before the failure-log record below so the
    # RESOLVED paths are what gets logged.
    if args.out:
        data_roles = {"dataset": args.dataset,
                      "eval_dataset": args.eval_dataset,
                      "eval2_dataset": args.eval2_dataset}
        if args.resume:
            resolved = resolve_resumed_datasets(args.out, args.resume, data_roles)
            args.dataset = resolved["dataset"]
            args.eval_dataset = resolved["eval_dataset"]
            args.eval2_dataset = resolved["eval2_dataset"]
        else:
            snapshot_datasets(args.out, data_roles)

    # Seed the failure logger with everything knowable before work starts, so a
    # crash at ANY later point (bad dataset name, OOM, unsupported arch guard)
    # still produces a run-log row identifying the attempt. Progress fields
    # (steps_done, end_loss, peak VRAM, phase) are refreshed as the run advances.
    _FAIL_CTX["run_log"] = args.run_log
    _FAIL_CTX["phase"] = "startup"
    _FAIL_CTX["record"] = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "arm": "exl3-native", "model": args.model, "out": args.out,
        "dataset": args.dataset, "eval_split": args.eval_split or "",
        "eval_dataset": args.eval_dataset or "",
        "eval2_dataset": args.eval2_dataset or "",
        "r": args.r, "alpha": args.alpha,
        "expert_r": "" if args.expert_r is None else args.expert_r,
        "use_rslora": int(bool(args.use_rslora)), "init_lora": args.init_lora,
        "quant_aware": args.quant_aware, "quant_aware_scale": args.quant_aware_scale,
        "lr": args.lr,
        "scheduler": args.scheduler, "weight_decay": args.weight_decay,
        "batch": args.batch, "grad_accum": args.grad_accum, "world_size": 1,
        "eff_batch": args.batch * args.grad_accum, "epochs": args.epochs,
        "steps_planned": args.steps, "seq_len": args.seq_len,
        "compute_dtype": args.compute_dtype, "attn_impl": args.attn_impl,
        "parallel": args.parallel, "shuffle": int(bool(args.shuffle)),
        "pack": int(bool(args.pack)),
        "pack_algo": args.pack_algo if args.pack else "",
        "vision": int(bool(args.vision)),
        "ga_loss": args.ga_loss, "max_samples": args.max_samples,
        "train_embeddings": int(bool(args.train_embeddings)),
        "train_head": int(bool(args.train_head)),
        "lora_embed": int(bool(args.lora_embed)),
        "lora_head": int(bool(args.lora_head)),
        "module_lora_lr_mul": args.module_lora_lr_mul,
        "prompt_format": args.prompt_format,
    }

    # Text cleaning is opt-in (--clean-text). --no-clean-text is the old default
    # and now a no-op, kept so existing commands don't break.
    if args.no_clean_text:
        print(" -- note: --no-clean-text is deprecated; cleaning is now OFF by "
              "default. Drop the flag, or use --clean-text to enable cleaning.")
    clean_text = args.clean_text

    cdt = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16}[args.compute_dtype]

    # 1. Load native model + tokenizer (the forward that's correct on EXL3).
    _FAIL_CTX["phase"] = "load_model"
    config = Config.from_directory(args.model)
    model = Model.from_config(config)

    # The KV cache must be created BEFORE model.load() so each attention layer
    # allocates its cache during loading; otherwise generation asserts on a
    # missing k_cache. Only needed for the live samples.
    cache = None
    if args.sample_every:
        from exllamav3 import Cache
        cache = Cache(model, max_num_tokens=4096)

    if args.split_even is not None and args.parallel != "split":
        raise SystemExit("--split-even only applies to --parallel split.")
    if args.parallel == "split":
        load_kwargs = {}
        if args.split_even is not None:
            if args.reserve_per_device is not None or args.use_per_device is not None:
                raise SystemExit("--split-even computes its own per-device budgets; "
                                 "drop --reserve-per-device/--use-per-device.")
            budgets = compute_even_split_budgets(model, config, args.split_even)
            print(f" -- split-even: use_per_device = "
                  f"[{', '.join(f'{b:.1f}' for b in budgets)}] GB "
                  f"(even weight shares + {args.split_even:g} GB load margin)")
            load_kwargs["use_per_device"] = budgets
        if args.reserve_per_device is not None:
            load_kwargs["reserve_per_device"] = args.reserve_per_device
        if args.use_per_device is not None:
            load_kwargs["use_per_device"] = args.use_per_device
        model.load(progressbar=True, **load_kwargs)
        active_devices = list(model.active_devices)
        print(f" -- layer-autosplit: active devices {active_devices}, "
              f"output device {model.output_device}")
    else:
        model.load(device=args.device, progressbar=True)
        active_devices = [torch.device(args.device).index]
    tokenizer = Tokenizer.from_config(config)
    pad_id = tokenizer.pad_token_id
    if pad_id is None or pad_id < 0:
        pad_id = tokenizer.eos_token_id or 0

    # 1b. MTP head (--mtp-targets): the model's own draft head, a separate
    #     component model (like the vision tower) that the offline trainers
    #     otherwise never load. Loaded WHOLE on the trunk's output device, where
    #     the final-norm state it consumes is produced; it borrows the trunk's
    #     embedding + LM head (attach_to at inference; the native forward reads
    #     them from the trunk directly).
    mtp_model = None
    if args.mtp_targets is not None:
        if "mtp" not in config.model_classes:
            raise SystemExit(f"--mtp-targets: {config.architecture} defines no MTP "
                             f"head (or this checkpoint has mtp_num_hidden_layers "
                             f"= 0 / no mtp.* tensors).")
        mtp_device = args.mtp_device or (str(model.output_device)
                                         if args.parallel == "split" else args.device)
        _FAIL_CTX["phase"] = "load_mtp"
        mtp_model = Model.from_config(config, component="mtp")
        mtp_model.load(device=mtp_device, progressbar=True)
        print(f" -- MTP head loaded on {mtp_device} "
              f"({len(mtp_model.modules)} modules)")
    elif args.freeze_trunk:
        raise SystemExit("--freeze-trunk leaves nothing to train without --mtp-targets.")
    elif args.mtp_device:
        raise SystemExit("--mtp-device needs --mtp-targets.")
    if args.freeze_trunk and not args.resume:
        print(" -- note: --freeze-trunk without --resume: the trunk is the plain "
              "base model (no adapter), so this is MTP-only training on the "
              "base trunk -- same as '--targets --mtp-targets ...'.")

    # 2. Build the differentiable QLoRA model (frozen base + trainable adapters).
    if args.offload_embed_head_optim and not (args.train_embeddings or args.train_head):
        raise SystemExit("--offload-embed-head-optim has nothing to offload without "
                         "--train-embeddings and/or --train-head.")
    # bf16 embed/head master weights when the CPU-offload optimizer (bf16 stochastic
    # rounding) drives them; fp32 otherwise.
    ms_dtype = torch.bfloat16 if args.offload_embed_head_optim else torch.float32
    _FAIL_CTX["phase"] = "build_net"
    _FAIL_CTX["record"]["arch"] = getattr(config, "architecture", "")
    net = NativeLlamaQLoRA(
        model, r=args.r, alpha=args.alpha, target_modules=args.targets,
        use_rslora=args.use_rslora,
        compute_dtype=cdt, gradient_checkpointing=not args.no_grad_ckpt,
        train_embeddings=args.train_embeddings, train_head=args.train_head,
        attn_impl=args.attn_impl, head_vocab_chunk=args.head_vocab_chunk,
        modules_to_save_dtype=ms_dtype,
        lora_embed=args.lora_embed, lora_head=args.lora_head,
        offload_activations=args.offload_activations,
        offload_mode=args.offload_mode, use_liger=args.use_liger,
        expert_r=args.expert_r, lora_dropout=args.lora_dropout,
        # A bare --mtp-targets (empty list) means the default head target list;
        # the net reads None as that default ([] = load the head, adapt nothing
        # -- what the validate gate builds).
        mtp_model=mtp_model, mtp_targets=(args.mtp_targets or None),
        mtp_loss_weight=args.mtp_loss_weight, freeze_trunk=args.freeze_trunk,
    )
    net.train()
    if net.mtp is not None:
        if not net.mtp_parameters():
            raise SystemExit("--mtp-targets matched no linear in the MTP head; "
                             f"available: {sorted({w.key.split('.')[-1] for w in net._mtp_wrappers})}")
        print(f" -- MTP head: {len(net.mtp.blocks)} block(s) on {net._mtp_device}, "
              f"targets={net.mtp_target_modules}; "
              + ("trunk FROZEN -- the head's loss is the loss, trunk under no_grad"
                 if not net.trunk_trainable() else
                 f"trained jointly (loss = trunk + {args.mtp_loss_weight:g} * mtp)"))
    if args.pack and (getattr(net, "has_gdn", False)
                      or getattr(net, "has_shortconv", False)):
        raise SystemExit(
            "--pack is not supported on GatedDeltaNet (Qwen3.5/3.6) or ShortConv "
            "(LFM2) models: the recurrence / causal conv would carry state "
            "across packed document boundaries. Drop --pack and train unpacked.")
    if args.head_vocab_chunk and net._head_slice is None:
        print(" -- note: --head-vocab-chunk set but this head can't slice; using "
              "the single-shot fused head.")
    if args.init_lora in ("pissa", "qerr"):
        # SVD init from the loaded weights. On resume this is recomputed and then
        # OVERWRITTEN by load_adapter below (pissa restores its exact offsets from
        # the checkpoint sidecar) -- a few wasted seconds, kept for simplicity.
        # (eva runs after the dataset is built -- it needs an activation pre-pass.)
        _FAIL_CTX["phase"] = "init_lora"
        net.apply_init_lora(args.init_lora, ref_model_dir=args.init_ref_model,
                            svd_niter=args.init_svd_niter)
    if args.resume:
        net.load_adapter(args.resume)
    if args.quant_aware != "none":
        # Run configuration, not learned state: applied fresh every run
        # (including resumes -- nothing about it lives in checkpoints).
        _FAIL_CTX["phase"] = "quant_aware"
        net.set_quant_aware(args.quant_aware, scale=args.quant_aware_scale,
                            ref_model_dir=(args.quant_aware_ref_model
                                           or args.init_ref_model),
                            seed=args.shuffle_seed)
    ms = [n for n, p in [("embed", net.embed_weight), ("head", net.head_weight)] if p is not None]
    print(f" -- trainable params: {net.num_trainable():,} "
          f"(r={args.r}, alpha={args.alpha}, targets={net.target_modules}"
          f"{', modules_to_save=' + str(ms) if ms else ''})")
    print(f" -- {net.describe_attn()}")
    if args.parallel == "split":
        from collections import Counter
        dist = Counter(str(d) for d in net._block_devices)
        print(f" -- decoder block devices: {dict(dist)}  (final norm + head on {net.device})")

    # 2b. Vision (--vision): the model's own frozen vision component, encoding
    # the dataset's images into the features the text tower's forward splices
    # in (training/vision_data.py + exllamav3.training.vision).
    vdata = None
    if args.vision:
        _FAIL_CTX["phase"] = "load_vision"
        if args.pack:
            raise SystemExit("--vision is not combinable with --pack (image rows "
                             "are not sample-packed; drop --pack).")
        if not args.messages_key:
            raise SystemExit("--vision needs --messages-key: images ride the "
                             "content parts of OpenAI-style messages rows.")
        from vision_data import VisionData, resolve_placeholder_id
        from exllamav3.training.vision import VisionEncoder
        if args.vision_device:
            vdev = args.vision_device
        elif args.parallel == "split":
            d0 = active_devices[0]
            vdev = f"cuda:{d0}" if isinstance(d0, int) else str(d0)
        else:
            vdev = args.device
        encoder = VisionEncoder(
            config, tokenizer, device=vdev, max_pixels=args.image_max_pixels,
            cache_bytes=int(args.vision_cache_gb * (1 << 30)), progressbar=True)
        vdata = VisionData(encoder, tokenizer,
                           placeholder_id=resolve_placeholder_id(config, tokenizer),
                           mrope=net.has_mrope, images_key=args.images_key)
        print(f" -- vision tower loaded on {vdev} (frozen); mrope={net.has_mrope}, "
              f"deepstack={sorted(net._deepstack) or 'none'}, "
              f"bidirectional image spans={net._bidir_mm}, "
              f"image_max_pixels={args.image_max_pixels or 'model default'}")

    # 3. Data.
    _FAIL_CTX["phase"] = "build_dataset"
    from chat_jinja import parse_template_vars
    template_vars = parse_template_vars(args.template_vars)
    examples = build_sft_examples(
        model, tokenizer, args.dataset, args.max_samples, args.seq_len,
        instruction_key=args.instruction_key, context_key=args.context_key,
        response_key=args.response_key, split=args.dataset_split,
        clean_text=clean_text,
        min_response_words=args.min_response_words,
        uppercase_response=args.uppercase_response,
        messages_key=args.messages_key,
        prompt_format=args.prompt_format,
        shuffle=args.shuffle, shuffle_seed=args.shuffle_seed,
        chat_template_file=args.chat_template_file,
        template_vars=template_vars, vision=vdata,
    )
    print(f" -- {len(examples)} SFT examples{' (shuffled)' if args.shuffle else ''}")
    assert examples, "no usable training examples"
    if vdata is not None and not any(ex.get("images") for ex in examples):
        print(" -- WARNING: --vision is set but no training row carried an image "
              "(check --messages-key / the content-parts layout / --images-key).")

    def collate_mm(batch, input_ids):
        # The image splice for a collated batch (None on text-only batches /
        # without --vision); built on CPU, the forward moves it.
        if vdata is None:
            return None
        return vdata.collate_mm(batch, input_ids.shape[1], cdt)

    # Tokenization check: decode the prompt span (labels==-100) and the supervised
    # response span (labels!=-100) separately, so the mask boundary and any
    # --seq-len truncation are visible before committing to a run. Specials are
    # shown so the chat template / <|eot_id|> stop token can be eyeballed.
    if args.inspect:
        _, eot = format_prompt_and_eot(
            model, tokenizer, args.prompt_format,
            chat_template_file=args.chat_template_file,
            template_vars=template_vars)
        eot_id = tokenizer.encode(eot, add_bos=False,
                                  encode_special_tokens=True)[0].tolist() if eot else []
        # encode() auto-prepends BOS (see build_sft_examples); strip it so the
        # "ends with turn-end token" check compares the real eot id(s), not [BOS, eot].
        bos = tokenizer.bos_token_id
        if eot_id and bos is not None and eot_id[0] == bos:
            eot_id = eot_id[1:]
        for i, ex in enumerate(examples[:args.inspect]):
            ids, labs = ex["input_ids"], ex["labels"]
            n_masked = sum(1 for l in labs if l == -100)
            sup = [t for t, l in zip(ids, labs) if l != -100]
            dec = lambda seq: tokenizer.decode(torch.tensor([seq]),
                                               decode_special_tokens=True)
            ends_eot = bool(eot_id) and sup[-len(eot_id):] == eot_id
            img_note = ""
            if ex.get("images"):
                n_slots = sum(1 for t in ids if t == vdata.placeholder_id)
                img_note = (f", {len(ex['images'])} image(s) / {n_slots} image "
                            f"tokens at {[s for s, _ in ex['images']]}"
                            + (", 3-D mRoPE positions" if "mrope_position_ids" in ex else ""))
            print(f"\n===== example {i} | {len(ids)} tokens "
                  f"({n_masked} masked / {len(sup)} supervised{img_note}) =====")
            # Decode contiguous masked/supervised spans in order: a single-turn
            # row is one prompt span + one response span (the old two-line
            # output); a multi-turn row interleaves several, and showing each
            # span makes every mask boundary eyeball-able.
            spans = []
            for t, l in zip(ids, labs):
                is_sup = l != -100
                if spans and spans[-1][0] == is_sup:
                    spans[-1][1].append(t)
                else:
                    spans.append((is_sup, [t]))
            if len(spans) <= 2:
                print(f"  PROMPT  (masked, -100): {dec(ids[:n_masked])!r}")
                print(f"  RESPONSE(supervised)  : {dec(sup)!r}")
            else:
                for is_sup, seq in spans:
                    tag = "SUPERVISED" if is_sup else "masked    "
                    print(f"  [{tag}] {dec(seq)!r}")
            print(f"  ends with turn-end token ({eot!r})? {ends_eot}"
                  + ("" if ends_eot else
                     "   <-- WARNING: response truncated by --seq-len; "
                     "raise --seq-len so the model learns to stop"))
        if args.pack:
            # Inspect decodes per-document (the tokenization check is per doc);
            # add a one-line packing summary so the fill ratio is visible too.
            real_tokens = sum(len(ex["input_ids"]) for ex in examples)
            blocks = pack_examples(examples, args.seq_len, pad_id, algo=args.pack_algo)
            cap = max(1, len(blocks) * args.seq_len)
            print(f"\n -- packing ({args.pack_algo}): {len(examples)} docs -> "
                  f"{len(blocks)} blocks of "
                  f"{args.seq_len} tok ({100.0 * real_tokens / cap:.1f}% filled); "
                  f"block 0 holds {len(set(blocks[0]['seg_ids']))} docs")
        print(f"\n -- inspect only ({args.inspect} shown); exiting before training.")
        return

    # Held-out eval set. Prefer the dataset's own eval split (real held-out data);
    # otherwise carve a deterministic val_frac off the front of train (same rows
    # as qlora_train_bnb.py, so the arms' eval losses stay comparable).
    val_examples = []
    if args.eval_split:
        if args.eval_text_key:
            # Plain-text LM eval (parity with --eval2-text-key): every token
            # supervised, straight nats/token -- and what --save-best tracks.
            val_examples = build_lm_examples(
                tokenizer, args.eval_dataset or args.dataset, args.eval_split,
                args.seq_len, text_key=args.eval_text_key,
                max_samples=args.eval_max_samples,
                config_name=args.eval_config,
                max_blocks=args.eval_max_blocks)
            kind = f"LM blocks over '{args.eval_text_key}'"
        else:
            val_examples = build_sft_examples(
                model, tokenizer, args.eval_dataset or args.dataset,
                args.eval_max_samples, args.seq_len,
                instruction_key=args.instruction_key, context_key=args.context_key,
                response_key=args.response_key, split=args.eval_split,
                clean_text=clean_text,
                min_response_words=args.min_response_words,
                uppercase_response=args.uppercase_response,
                messages_key=args.messages_key,
                prompt_format=args.prompt_format,
                config_name=args.eval_config,
                chat_template_file=args.chat_template_file,
                template_vars=template_vars, vision=vdata,
            )
            kind = "SFT"
        print(f" -- held-out eval: {len(val_examples)} {kind} examples from "
              f"split '{args.eval_split}'; {len(examples)} for training")
    elif args.val_frac > 0:
        n_val = max(1, int(len(examples) * args.val_frac))
        val_examples, examples = examples[:n_val], examples[n_val:]
        print(f" -- held out {len(val_examples)} val examples; "
              f"{len(examples)} for training")
        assert examples, "val_frac too large; no training examples left"

    # Optional SECOND held-out eval set (task-independent monitor, e.g. wikitext).
    # Plain-text LM loss when --eval2-text-key is given, else a second SFT eval.
    val2_examples = []
    eval2_label = ""
    if args.eval2_dataset:
        eval2_label = args.eval2_dataset.split("/")[-1]
        if args.eval2_text_key:
            val2_examples = build_lm_examples(
                tokenizer, args.eval2_dataset, args.eval2_split, args.seq_len,
                text_key=args.eval2_text_key, max_samples=args.eval2_max_samples,
                config_name=args.eval2_config, max_blocks=args.eval2_max_blocks)
            kind = f"LM blocks over '{args.eval2_text_key}'"
        else:
            val2_examples = build_sft_examples(
                model, tokenizer, args.eval2_dataset, args.eval2_max_samples,
                args.seq_len, instruction_key=args.instruction_key,
                context_key=args.context_key, response_key=args.response_key,
                split=args.eval2_split, clean_text=clean_text,
                min_response_words=args.min_response_words,
                uppercase_response=args.uppercase_response,
                messages_key=args.messages_key, prompt_format=args.prompt_format,
                config_name=args.eval2_config,
                chat_template_file=args.chat_template_file,
                template_vars=template_vars)
            kind = "SFT"
        print(f" -- eval2 ({eval2_label}): {len(val2_examples)} {kind} examples "
              f"from split '{args.eval2_split}'")

    # Sample packing (training set ONLY -- eval stays per-example for comparable
    # losses). Done after the val carve so a packed block never straddles the
    # train/val boundary; resolve_steps below then counts packed blocks as the
    # training unit, so --epochs still means "passes over the data".
    if vdata is not None:
        # Every image the run can touch has now been encoded (train + eval
        # builds). If all their features fit the cache, the tower's VRAM is
        # better spent on activations -- drop it. Otherwise it stays loaded
        # to re-encode the over-budget images per batch.
        if vdata.encoder.all_cached:
            vdata.encoder.unload()
            print(" -- vision: all image features cached in RAM; vision tower "
                  "unloaded (VRAM freed).")
        else:
            print(f" -- vision: {vdata.encoder.n_evicted} image(s) exceed "
                  f"--vision-cache-gb {args.vision_cache_gb:g}; the vision tower "
                  f"stays loaded to re-encode them per batch.")

    if args.pack:
        n_docs = len(examples)
        real_tokens = sum(len(ex["input_ids"]) for ex in examples)
        examples = pack_examples(examples, args.seq_len, pad_id, algo=args.pack_algo)
        cap = max(1, len(examples) * args.seq_len)
        print(f" -- packed ({args.pack_algo}) {n_docs} docs -> {len(examples)} blocks "
              f"of {args.seq_len} tok ({100.0 * real_tokens / cap:.1f}% filled, "
              f"~{real_tokens / max(1, len(examples)):.0f} real tok/block)")
        assert examples, "no training blocks after packing"

    # eva init runs HERE (not with pissa/qerr above): it streams a no-grad
    # activation pre-pass over the actual training batches. Skipped on resume --
    # the checkpoint's A/B already carry the init, and unlike pissa there are no
    # frozen offsets to reconstruct.
    if args.init_lora == "eva" and not args.resume:
        _FAIL_CTX["phase"] = "init_lora"

        def eva_prepass():
            used, i = 0, 0
            while used < args.init_eva_tokens and i < len(examples):
                batch = examples[i:i + args.batch]
                i += len(batch)
                input_ids, _, attn, pos_ids, seg_ids = collate(batch, pad_id)
                used += int(attn.sum())
                yield dict(input_ids=input_ids, attention_mask=attn,
                           position_ids=pos_ids, seg_ids=seg_ids,
                           mm=collate_mm(batch, input_ids))

        net.apply_init_lora("eva", svd_niter=args.init_svd_niter,
                            eva_batches=eva_prepass())
    elif args.init_lora == "eva":
        print(" -- eva init skipped on --resume (the checkpoint's adapters "
              "already carry it)")

    # Finalize step count (from --epochs) and warmup before building the schedule.
    args.steps, warmup_steps = resolve_steps_and_warmup(
        args, len(examples), args.batch * args.grad_accum)
    print(f" -- {args.steps} steps, scheduler={args.scheduler}, "
          f"warmup={warmup_steps}, weight_decay={args.weight_decay}")

    # 4. Optional generator for live samples (KV-cache inference path). The cache
    #    was allocated before load() above. Use the training chat format so the
    #    preview is meaningful for a metharme-trained adapter.
    build_prompt, _ = format_prompt_and_eot(
        model, tokenizer, args.prompt_format,
        chat_template_file=args.chat_template_file,
        template_vars=template_vars)
    generator = None
    if args.sample_every:
        from exllamav3 import Generator
        generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
        net.eval()
        with torch.inference_mode():
            base = sample(model, cache, tokenizer, generator, build_prompt, args.sample_prompt)
        net.train()
        print(f"\n\U0001f3ad  baseline (step 0): {args.sample_prompt}\n     -> {base}\n")

    # 5. Optimizer over the trainable params, plus the LR schedule. Weight decay
    #    on the LoRA params only (param_groups puts embed/head in a 0-WD group).
    #    With --offload-embed-head-optim the embed/head group is split off onto a
    #    separate CPU-offload optimizer (offload_opt) and the main optimizer/scheduler
    #    drives only the LoRA group; offload_opt mirrors the schedule's LR per step.
    offload_opt = None
    if args.module_lora_lr_mul != 1.0 and not (args.lora_embed or args.lora_head):
        raise SystemExit("--module-lora-lr-mul has nothing to scale without "
                         "--lora-embed and/or --lora-head.")
    if args.offload_embed_head_optim:
        lora_groups = net.lora_param_groups(args.weight_decay, args.lr,
                                            args.module_lora_lr_mul)
        opt = build_optimizer(lora_groups, args.lr, args.optim)
        offload_opt = build_cpu_offload_optimizer(net.modules_to_save_parameters(), args.lr)
        print(f" -- embed/head optimizer offloaded to CPU (torchao, bf16 stochastic "
              f"rounding); excluded from grad clip, follows the LoRA LR schedule")
    else:
        opt = build_optimizer(net.param_groups(args.weight_decay, args.lr,
                                               args.module_lora_lr_mul),
                              args.lr, args.optim)
    if args.module_lora_lr_mul != 1.0:
        print(f" -- embed/head LoRA adapters in their own param group at "
              f"lr x {args.module_lora_lr_mul:g} = {args.lr * args.module_lora_lr_mul:.2e}")
    sched = make_lr_scheduler(opt, args.scheduler, args.steps, warmup_steps)

    # 5a. Optionally restore optimizer/schedule/step from the resumed checkpoint so
    #     the run continues instead of cold-restarting warmup/cosine. resume_state
    #     seeds best_val/ema below; resume_step shifts the loop's start.
    resume_step, resume_state = 0, None
    yaml_base_lrs = list(sched.base_lrs)  # per-group base LRs from the CONFIG
    # --freeze-trunk changes the trainable set (head only), so the resumed
    # optimizer state (trunk adapters) can't apply: always a cold start there.
    if args.resume and not args.reset_optimizer and not args.freeze_trunk:
        resume_state = load_trainer_state(args.resume)
        if resume_state is not None:
            try:
                restore_optimizer_state(opt, resume_state["optimizer"])
            except ValueError as e:
                if net.mtp is None:
                    raise
                # A trunk checkpoint resumed into a joint trunk+MTP run: the
                # param groups grew by the head's adapters, so the saved moments
                # don't line up. Continue cold (weights restored, schedule from
                # step 0) rather than die -- the same as --reset-optimizer.
                print(f" -- note: optimizer state in {args.resume} does not match "
                      f"the trunk+MTP parameter set ({e}); starting the optimizer "
                      f"and schedule cold (as --reset-optimizer).")
                groups = (net.lora_param_groups(args.weight_decay, args.lr,
                                                args.module_lora_lr_mul)
                          if args.offload_embed_head_optim else
                          net.param_groups(args.weight_decay, args.lr,
                                           args.module_lora_lr_mul))
                opt = build_optimizer(groups, args.lr, args.optim)
                sched = make_lr_scheduler(opt, args.scheduler, args.steps, warmup_steps)
                resume_state = None
        if resume_state is not None:
            # The CPU-offload optimizer manages its own (CPU) state placement, so
            # load it directly rather than through restore_optimizer_state (which would
            # move state onto the params' GPU devices). Absent in pre-offload runs.
            if offload_opt is not None and resume_state.get("offload_optimizer") is not None:
                offload_opt.load_state_dict(resume_state["offload_optimizer"])
            if resume_state.get("scheduler") is not None:
                sched.load_state_dict(resume_state["scheduler"])
            # The config is the source of truth for the base LR on resume: the
            # restored optimizer/scheduler state carries the ORIGINAL run's lr
            # in param_groups/base_lrs, which used to silently override an
            # edited lr:. Re-base the schedule at the config value -- moments,
            # step counter and schedule position all kept. A byte-identical
            # config leaves everything exactly as restored.
            if list(sched.base_lrs) != yaml_base_lrs:
                old_lr = sched.base_lrs[0]
                sched.base_lrs = list(yaml_base_lrs)
                for g, base, fn in zip(opt.param_groups, sched.base_lrs,
                                       sched.lr_lambdas):
                    g["initial_lr"] = base
                    g["lr"] = base * fn(sched.last_epoch)
                sched._last_lr = [g["lr"] for g in opt.param_groups]
                print(f" -- resume LR override: base lr {old_lr:.2e} "
                      f"(checkpoint) -> {yaml_base_lrs[0]:.2e} (config); "
                      f"optimizer moments and schedule position kept")
            resume_step = int(resume_state["step"])
            print(f" -- resumed trainer state from {args.resume}: continuing at "
                  f"step {resume_step + 1}/{args.steps} (best_val "
                  f"{resume_state['best_val']}, lr {sched.get_last_lr()[0]:.2e})")
            if resume_step >= args.steps:
                print(f" -- WARNING: resume step {resume_step} >= --steps "
                      f"{args.steps}; nothing to do. Raise --steps/--epochs.")
        else:
            print(f" -- {args.resume} has no trainer_state.pt; resuming weights "
                  f"only (cold optimizer + schedule from step 0).")

    def batches():
        order = list(range(len(examples)))
        while True:
            random.Random(0).shuffle(order)
            for i in range(0, len(order) - args.batch + 1, args.batch):
                yield [examples[j] for j in order[i:i + args.batch]]

    # |dB| telemetry baseline. With an SVD init the raw ‖B‖ is dominated by the
    # constant init component (a whole run moves it in the 4th decimal), so the
    # step line logs the distance from the init instead: B0 is zero for the
    # default init (‖B-B0‖ == ‖B‖, matching historical logs), the exact fp32
    # sidecar masters for pissa (survives --resume), or a CPU fp32 snapshot
    # taken here (qerr; on a qerr --resume this measures movement since the
    # resume, not since the original init). CPU on purpose -- the snapshot must
    # not eat VRAM, and the per-step transfer is microseconds per wrapper.
    b0_refs = []
    with torch.no_grad():
        for w in net._wrappers:
            if w.r <= 0:
                continue
            if w.init_b0_master is not None:
                b0_refs.append((w, w.init_b0_master))
            elif args.init_lora != "default" and w.lora_b.abs().max().item() > 0:
                b0_refs.append((w, w.lora_b.detach().float().cpu().clone()))
            else:
                b0_refs.append((w, None))

    def adapter_b_norm():
        # Per-wrapper sums live on each wrapper's own device (they differ under a
        # layer split), so reduce each to a Python float before summing -- adding
        # tensors across cuda:0/cuda:1 would raise a cross-device error.
        with torch.no_grad():
            tot = 0.0
            for w, b0 in b0_refs:
                if b0 is None:
                    tot += w.lora_b.float().pow(2).sum().item()
                else:
                    tot += (w.lora_b.detach().float().cpu() - b0).pow(2).sum().item()
            return tot ** 0.5

    def save(tag, directory=None):
        # Always leave net in train mode after; saving touches the adapter only.
        # ``directory`` defaults to --out (the best-val / --save-every target);
        # the final and interrupted saves pass final_dir() so they never clobber
        # a kept best-val adapter.
        directory = directory or args.out
        net.save_adapter(directory, base_model_name_or_path=args.model)
        save_trainer_state(directory, step=step, opt=opt, sched=sched,
                           best_val=best_val, best_val_step=best_val_step, ema=ema,
                           offload_opt=offload_opt)
        print(f"{tag} Adapter written to {directory}")

    def eval_loss(exs):
        # Mean per-example loss over an eval set, one example at a time (no
        # padding effects). qlora_train_bnb.py computes this identically. Works
        # for both SFT (completion-masked) and plain-LM (all-supervised) sets.
        if not exs:
            return None
        net.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for ex in exs:
                input_ids, labels, attn, pos_ids, seg_ids = collate([ex], pad_id)
                l = net.compute_loss(input_ids, labels, attention_mask=attn,
                                     chunk=args.ce_chunk,
                                     position_ids=pos_ids, seg_ids=seg_ids,
                                     mm=collate_mm([ex], input_ids))
                total += l.item()
                n += 1
        net.train()
        return total / n

    def evaluate():
        return eval_loss(val_examples)

    bgen = batches()
    if resume_step:
        # Fast-forward the data stream to where the original process stopped:
        # the sequence is fully deterministic from Random(0), so skipping
        # resume_step * grad_accum micro-batches reproduces the exact order an
        # uninterrupted run would have used. Without this a resume replayed
        # epoch 0's permutation while the step counter and LR schedule
        # continued (doc/bug_resume_data_order.md). Yields are index lists --
        # no tokenization -- so this is microseconds. Only exact when the
        # dataset matches the original run's (enforced above by
        # resolve_resumed_datasets unless 'current' was chosen on a mismatch).
        for _ in range(resume_step * args.grad_accum):
            next(bgen)
        print(f" -- data stream fast-forwarded {resume_step * args.grad_accum} "
              f"micro-batches to continue the run's batch order at step "
              f"{resume_step + 1}")
    opt.zero_grad(set_to_none=True)
    if offload_opt is not None:
        offload_opt.zero_grad(set_to_none=True)
    # Seed from the resumed state so best-tracking and the EMA continue rather than
    # reset (resume_state is None under --reset-optimizer / a weights-only dir).
    ema = resume_state["ema"] if resume_state else None
    step = resume_step
    best_val = resume_state["best_val"] if resume_state else float("inf")
    best_val_step = resume_state["best_val_step"] if resume_state else 0
    start_loss = end_loss = None
    start_val = start_eval2 = None
    last_eval_step, last_val, last_eval2 = -1, None, None
    tok_seen, tot_seen, t0 = 0, 0, time.time()
    run_started = datetime.datetime.now().isoformat(timespec="seconds")
    meter = ThroughputMeter()

    # (peak_vram_gb / log_run are defined once below, after the baseline eval --
    # an earlier duplicate pair that used to sit here was dead code and is gone.)

    # Run config: reuses the failure-logger record (the same identity/
    # hyperparameter fields as the run-log CSV), refreshed with the values
    # finalized since startup (epoch-resolved steps, warmup, targets, split
    # sizes), so the CSV row, the local report, and any wandb run all line up.
    run_config = dict(_FAIL_CTX["record"])
    run_config.update(
        steps_planned=args.steps, steps_per_epoch=args.steps_per_epoch,
        warmup_steps=warmup_steps, targets=" ".join(net.target_modules),
        mtp_targets=" ".join(net.mtp_target_modules),
        freeze_trunk=int(bool(args.freeze_trunk)),
        trainable_params=net.num_trainable(), n_train=len(examples),
        n_val=len(val_examples), n_eval2=len(val2_examples))
    run_name = (args.run_name or args.wandb_run_name
                or os.path.basename(os.path.normpath(args.out)))

    # Local run report -- the DEFAULT logging path (self-contained HTML, no
    # third-party account). On by default whenever there's an --out to write it
    # next to; --no-report opts out. Metrics stream to disk as they're logged so
    # a crash still renders whatever it got (via _finish_report in _log_failure).
    report = None
    if args.out and not args.no_report:
        report = RunLogger(args.out, run_name, config=run_config)
        _REPORT["rep"] = report

    # Live monitor (--live-report): recompute any step's batch on demand. The
    # mapping mirrors batches()/the loop exactly: optimizer step s consumes
    # micro-batches (s-1)*ga .. (s-1)*ga+ga-1 of the stream -- an ABSOLUTE
    # mapping, valid on resume too now that bgen is fast-forwarded to the
    # resume step (so pre-resume steps are browsable and correct). batches()
    # reseeds Random(0) every pass but shuffles the SAME list in place, so
    # epoch e's permutation is that seeded shuffle applied e+1 times
    # cumulatively -- replayed here lazily as steps are browsed (verified
    # against the generator in the parity test).
    if args.live_report and report is None:
        print(" -- --live-report needs the local report; ignoring (--no-report set)")
    elif args.live_report:
        live_orders = {"order": list(range(len(examples))), "perms": []}
        live_mpe = max(1, len(examples) // args.batch)  # micro-batches per pass

        def live_perm(e):
            while len(live_orders["perms"]) <= e:
                random.Random(0).shuffle(live_orders["order"])
                live_orders["perms"].append(live_orders["order"][:])
            return live_orders["perms"][e]

        def live_batch_fn(s):
            if not (0 < s <= args.steps):
                raise ValueError(f"step must be in [1, {args.steps}]")
            mbs = []
            for g in range(args.grad_accum):
                m = (s - 1) * args.grad_accum + g
                e, w = divmod(m, live_mpe)
                seqs = [{"index": j,
                         "docs": decode_example_docs(tokenizer, examples[j])}
                        for j in live_perm(e)[w * args.batch:(w + 1) * args.batch]]
                mbs.append({"micro": g, "epoch": e, "sequences": seqs})
            return {"step": s, "micro_batches": mbs}

        start_live_monitor(
            args.out, batch_fn=live_batch_fn, port=args.live_report_port,
            live_info={"total_steps": args.steps, "first_step": resume_step + 1,
                       "run_name": run_name})

    # Optional wandb run (--wandb-project; OFF by default). Same config so runs
    # are comparable across the CSV, the local report, and wandb.
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_name,
            config=run_config)
        _WANDB_RUN["run"] = wandb_run

    # Baseline eval at step 0 (the adapter is a no-op at init, B=0, so this is the
    # base model's held-out loss) -- a reference point for the trained numbers.
    _FAIL_CTX["phase"] = "baseline_eval"
    if val_examples or val2_examples:
        start_val = evaluate()
        start_eval2 = eval_loss(val2_examples) if val2_examples else None
        parts = []
        if start_val is not None:
            parts.append(f"held-out {start_val:.4f}")
        if start_eval2 is not None:
            parts.append(f"{eval2_label} {start_eval2:.4f}")
        print("    [eval] step 0 (baseline): " + " | ".join(parts))
        base_eval = {k: v for k, v in (("eval/held_out", start_val),
                                       ("eval/eval2", start_eval2))
                     if v is not None}
        if report is not None:
            report.log(base_eval, step=0)
        if wandb_run is not None:
            wandb_run.log(base_eval, step=0)

    # Start the training timer + VRAM peak AFTER the baseline eval so neither is
    # counted against training throughput.
    t0 = time.time()
    # The UVM spillover allocator (a torch pluggable allocator) doesn't support
    # memory stats -- reset/max_memory_allocated raise -- so peak VRAM is
    # reported as unavailable in that mode.
    if torch.cuda.is_available() and not args.vram_spillover:
        for d in active_devices:
            torch.cuda.reset_peak_memory_stats(d)

    def peak_vram_gb():
        if not torch.cuda.is_available() or args.vram_spillover:
            return 0.0
        return max((torch.cuda.max_memory_allocated(d) / 1e9 for d in active_devices),
                   default=0.0)

    def log_run(status, dt, final_val, final_eval2):
        # One CSV row capturing the run's identity, hyperparameters and results.
        # Called on normal finish and on Ctrl-C; crashes are covered separately
        # by _log_failure (main()'s wrapper), which this call disarms.
        # NOTE: start_val/start_eval2 were silently missing from this record for
        # several sessions (a duplicate-definition merge artifact shadowed the
        # copy that had them); restored in Session 11.
        _FAIL_CTX["logged"] = True
        rnd = lambda x, n=6: round(x, n) if isinstance(x, (int, float)) else ""
        append_run_log(args.run_log, {
            "timestamp": run_started, "arm": "exl3-native", "status": status,
            "model": args.model, "arch": getattr(config, "architecture", ""),
            "out": args.out, "dataset": args.dataset,
            "eval_split": args.eval_split or "", "eval_dataset": args.eval_dataset or "",
            "eval2_dataset": args.eval2_dataset or "",
            "r": args.r, "alpha": args.alpha,
            "expert_r": "" if args.expert_r is None else args.expert_r,
            "use_rslora": int(bool(args.use_rslora)), "init_lora": args.init_lora,
            "quant_aware": args.quant_aware, "quant_aware_scale": args.quant_aware_scale,
            "lr": args.lr,
            "scheduler": args.scheduler, "warmup_steps": warmup_steps,
            "weight_decay": args.weight_decay, "batch": args.batch,
            "grad_accum": args.grad_accum, "world_size": 1,
            "eff_batch": args.batch * args.grad_accum, "epochs": args.epochs,
            "steps_planned": args.steps, "steps_done": step, "seq_len": args.seq_len,
            "targets": " ".join(net.target_modules),
            "mtp_targets": " ".join(net.mtp_target_modules),
            "freeze_trunk": int(bool(args.freeze_trunk)),
            "compute_dtype": args.compute_dtype,
            "attn_impl": args.attn_impl, "parallel": args.parallel,
            "shuffle": int(bool(args.shuffle)), "pack": int(bool(args.pack)),
            "pack_algo": args.pack_algo if args.pack else "",
            "ga_loss": args.ga_loss,
            "max_samples": args.max_samples,
            "train_embeddings": int(bool(args.train_embeddings)),
            "train_head": int(bool(args.train_head)), "prompt_format": args.prompt_format,
            "lora_embed": int(bool(args.lora_embed)),
            "lora_head": int(bool(args.lora_head)),
            "module_lora_lr_mul": args.module_lora_lr_mul,
            "trainable_params": net.num_trainable(), "n_train": len(examples),
            "n_val": len(val_examples), "n_eval2": len(val2_examples),
            "start_loss": rnd(start_loss), "end_loss": rnd(end_loss),
            "best_val": rnd(best_val) if best_val != float("inf") else "",
            "best_val_step": best_val_step or "",
            "start_val": rnd(start_val), "start_eval2": rnd(start_eval2),
            "final_val": rnd(final_val), "final_eval2": rnd(final_eval2),
            "total_s": rnd(dt, 1), "s_per_step": rnd(dt / step, 4) if step else "",
            "sup_tok_s": round(tok_seen / dt) if dt else "",
            "tot_tok_s": round(tot_seen / dt) if dt else "",
            "peak_vram_gb": rnd(peak_vram_gb(), 3),
            "t_data_s": rnd(timer.total["data"], 1), "t_fwd_s": rnd(timer.total["fwd"], 1),
            "t_bwd_s": rnd(timer.total["bwd"], 1), "t_opt_s": rnd(timer.total["opt"], 1),
            "dequant_s_per_step": rnd(dequant_s_per_step, 3)
                if dequant_s_per_step is not None else "",
            "phase": "", "error": "", "notes": "",
        })

    # Per-step wall-clock section breakdown (data/fwd/bwd/opt). Under --parallel
    # split the step spans several devices; sync them all at each mark.
    timer = StepTimer(devices=active_devices if torch.cuda.is_available() else None)

    # Optional dequant profiling: time every frozen-weight reconstruction for the
    # first N steps to answer "how much of the step is trellis reconstruction".
    from exllamav3.training import backbone as _backbone
    # The recompute->backward weight cache only has a second use to serve when
    # gradient checkpointing recomputes the forward; without it the stores
    # would just pin every weight for the whole backward.
    dequant_cache = args.dequant_cache and not args.no_grad_ckpt
    dq_profile = None
    dequant_s_per_step = None
    if args.profile_dequant > 0:
        dq_profile = {"calls": 0, "s": 0.0}
        _backbone.profile_dequant(dq_profile)
        print(f" -- profiling dequant (trellis reconstruction) for the first "
              f"{args.profile_dequant} steps; adds sync overhead while active")

    # Optional torch.profiler window (--torch-profile N): skip 3 steps, 2
    # profiler-warmup steps, then N active steps. Stacks + shapes let the
    # analysis attribute GEMM time to source lines (base matmul vs Hadamard
    # transforms vs LoRA adapters vs trellis reconstruct) without touching
    # the hot path when profiling is off.
    prof = None
    if args.torch_profile > 0:
        from torch.profiler import profile, schedule, ProfilerActivity
        prof_dir = os.path.join(args.out, "torch_profile")
        os.makedirs(prof_dir, exist_ok=True)

        def _prof_ready(p):
            p.export_chrome_trace(os.path.join(prof_dir, "trace.json.gz"))
            avg = p.key_averages()
            for sort, name in (("self_cuda_time_total", "cuda"),
                               ("self_cpu_time_total", "cpu")):
                with open(os.path.join(prof_dir, f"key_averages_{name}.txt"), "w") as f:
                    f.write(avg.table(sort_by=sort, row_limit=100))
            with open(os.path.join(prof_dir, "key_averages_stacks.txt"), "w") as f:
                f.write(p.key_averages(group_by_stack_n=12).table(
                    sort_by="self_cuda_time_total", row_limit=150,
                    max_src_column_width=200))
            print(f"  [torch-profile] trace + key-averages tables -> {prof_dir}")

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(skip_first=3, wait=0, warmup=2,
                              active=args.torch_profile, repeat=1),
            on_trace_ready=_prof_ready, with_stack=True, record_shapes=True)
        prof.start()
        print(f" -- torch.profiler armed: 3 skip + 2 warmup steps, then "
              f"{args.torch_profile} profiled steps -> {prof_dir}")
    # Per-step epoch readout: steps_per_epoch was stashed by
    # resolve_steps_and_warmup from the same training units the loop consumes
    # (packed blocks under --pack), so in --steps mode the total shows the
    # equivalent epoch count the step budget works out to.
    epochs_total = args.epochs if args.epochs > 0 else args.steps / args.steps_per_epoch
    # Last COMPLETED optimizer step, as opposed to the in-flight one `step` holds
    # mid-iteration. Ctrl-C used to save the latter, producing a state whose
    # step counter disagreed with the scheduler's last_epoch; the resume then
    # trusted the counter, skipped that iteration's micro-batches and ran the
    # rest of the LR schedule one step late. Measured on GPU in S65.
    done_step = resume_step
    try:
        for step in range(resume_step + 1, args.steps + 1):
            _FAIL_CTX["phase"] = f"train step {step}"
            step_t0 = time.time()
            accum_loss = 0.0
            accum_parts: dict = {}
            step_sup = step_tot = 0
            timer.begin_step()
            # Draw the WHOLE accumulation window first so each micro-batch can be
            # weighted by its share of the window's supervised tokens (--ga-loss
            # token, the Oct-2024 HF/Unsloth grad-accumulation fix): compute_loss
            # returns a mean over each micro-batch's own supervised tokens, so
            # averaging those means (the old behavior, kept as --ga-loss mean)
            # over-weights tokens in short micro-batches. Weighting each loss by
            # n_sup/total_sup makes the step gradient identical to one big batch.
            # Counts use the SHIFTED labels ([:, 1:]) to match the CE denominator.
            # A no-op when grad_accum == 1 (weight = 1).
            window = []
            for _ in range(args.grad_accum):
                mb = next(bgen)
                col = collate(mb, pad_id)
                window.append(col + (collate_mm(mb, col[0]),))
            n_sups = [int((w[1][:, 1:] != -100).sum()) for w in window]
            total_sup = max(sum(n_sups), 1)
            timer.mark("data")
            for (input_ids, labels, attn, pos_ids, seg_ids, mm), n_sup in zip(window, n_sups):
                loss = net.compute_loss(input_ids, labels, attention_mask=attn,
                                        chunk=args.ce_chunk,
                                        position_ids=pos_ids, seg_ids=seg_ids,
                                        mm=mm)
                # .item() before backward (harmless to the graph) so the fwd/bwd
                # sections split cleanly at the sync.
                loss_val = loss.item()
                timer.mark("fwd")
                w_i = (n_sup / total_sup) if args.ga_loss == "token" \
                    else (1.0 / args.grad_accum)
                with _backbone.backward_dequant_cache(enable=dequant_cache):
                    (loss * w_i).backward()
                timer.mark("bwd")
                # accum_loss uses the same weights, so under "token" it is the
                # true per-token mean over the whole accumulation window.
                accum_loss += loss_val * w_i
                # Joint trunk+MTP runs: the components behind the summed loss.
                for k, v in getattr(net, "last_losses", {}).items():
                    accum_parts[k] = accum_parts.get(k, 0.0) + v * w_i
                step_sup += int((labels != -100).sum())   # supervised tokens
                step_tot += int(attn.sum())                # total (non-pad) tokens

            # grad norm BEFORE clipping is a direct check that gradients reach the
            # adapters (a flat ~0 here would mean the backward graph is broken).
            # Under --offload-embed-head-optim the embed/head params are excluded
            # (torchao's CPUOffloadOptimizer forbids grad clipping on its params);
            # the LoRA grads are what the norm/clip should reflect anyway.
            clip_params = (net.lora_parameters() if offload_opt is not None
                           else net.trainable_parameters())
            gnorm = torch.nn.utils.clip_grad_norm_(
                clip_params, args.max_grad_norm or float("inf")
            ).item()
            if offload_opt is not None:
                # Mirror the schedule's current LR (the one opt.step() is about to use)
                # onto the offload optimizer, which has no scheduler support, then step
                # both. Set before stepping so embed/head move at the same LR as LoRA.
                set_offload_lr(offload_opt, sched.get_last_lr()[0])
                offload_opt.step()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if offload_opt is not None:
                offload_opt.zero_grad(set_to_none=True)
            timer.mark("opt")
            timer.end_step()

            # Rolling tok/s over the train-step compute only (eval/sample/save
            # below are excluded so the rate reflects steady-state throughput).
            tok_seen += step_sup
            tot_seen += step_tot
            meter.update(time.time() - step_t0, step_sup, step_tot)
            sup_tps, tot_tps = meter.rates()

            if start_loss is None:
                start_loss = accum_loss
            end_loss = accum_loss
            ema = accum_loss if ema is None else 0.9 * ema + 0.1 * accum_loss
            epoch_now = step / args.steps_per_epoch
            cur_lr = sched.get_last_lr()[0]
            b_dist = adapter_b_norm()
            # Joint trunk+MTP: show the two components next to their sum. In
            # MTP-only runs the loss IS the head's loss, so nothing extra.
            parts = (f"(trunk {accum_parts['trunk']:6.4f} mtp {accum_parts['mtp']:6.4f}) "
                     if "trunk" in accum_parts else "")
            print(f"  step {step:>5}/{args.steps} | "
                  f"ep {epoch_now:.2f}/{epochs_total:.4g} | "
                  f"loss {accum_loss:6.4f} {parts}| "
                  f"ema {ema:6.4f} | grad {gnorm:7.4f} | lr {cur_lr:.2e} | "
                  f"|dB| {b_dist:7.3f} | {tot_tps:,.0f} tok/s | {timer.step_line()}")
            train_metrics = {
                "train/loss": accum_loss, "train/ema": ema,
                "train/grad_norm": gnorm, "train/lr": cur_lr,
                "train/adapter_b_dist": b_dist, "train/epoch": epoch_now,
                "perf/sup_tok_s": sup_tps, "perf/tot_tok_s": tot_tps,
                **{f"train/loss_{k}": v for k, v in accum_parts.items()},
            }
            if report is not None:
                report.log(train_metrics, step=step)
            if wandb_run is not None:
                wandb_run.log(train_metrics, step=step)

            # Keep the failure record current so a later crash (or kill -9 at
            # least leaves the errors.log short a row, see _FAIL_CTX note)
            # carries how far the run got and its memory/loss state.
            _FAIL_CTX["record"].update(
                steps_done=step, end_loss=round(accum_loss, 6),
                peak_vram_gb=round(peak_vram_gb(), 3))
            if start_loss is not None and "start_loss" not in _FAIL_CTX["record"]:
                _FAIL_CTX["record"]["start_loss"] = round(start_loss, 6)

            # End of the dequant profiling window: report reconstruction time
            # against the timed step wall clock, then disable the hook.
            if dq_profile is not None and (step - resume_step) >= args.profile_dequant:
                n_prof = step - resume_step
                wall = sum(timer.total.values())
                dequant_s_per_step = dq_profile["s"] / max(1, n_prof)
                print(f"  [profile] dequant: {dq_profile['calls']:,} reconstructions, "
                      f"{dq_profile['s']:.2f}s over {n_prof} steps = "
                      f"{dequant_s_per_step:.3f}s/step "
                      f"({100.0 * dq_profile['s'] / max(wall, 1e-9):.0f}% of step wall "
                      f"time) -- profiling off from here")
                from exllamav3.training import backbone as _backbone
                _backbone.profile_dequant(None)
                dq_profile = None

            if (args.eval_every and step % args.eval_every == 0
                    and (val_examples or val2_examples)):
                vl = evaluate() if val_examples else None
                v2 = eval_loss(val2_examples) if val2_examples else None
                last_eval_step, last_val, last_eval2 = step, vl, v2
                parts = []
                if vl is not None:
                    parts.append(f"held-out {vl:.4f}")
                    # Track best val for the run log regardless of --save-best;
                    # only write the checkpoint when --save-best is set.
                    if vl < best_val:
                        best_val = vl
                        best_val_step = step
                        if args.save_best:
                            save(f"[best step {step}, val {vl:.4f}]")
                if v2 is not None:
                    parts.append(f"{eval2_label} {v2:.4f}")
                print(f"    [eval] step {step}: " + " | ".join(parts))
                eval_metrics = {k: v for k, v in (("eval/held_out", vl),
                                                  ("eval/eval2", v2))
                                if v is not None}
                if vl is not None:
                    eval_metrics["eval/best_val"] = best_val
                if report is not None:
                    report.log(eval_metrics, step=step)
                if wandb_run is not None:
                    wandb_run.log(eval_metrics, step=step)

            if args.sample_every and step % args.sample_every == 0:
                net.eval()
                net.apply_to_native()      # make generation reflect the adapter
                with torch.inference_mode():
                    txt = sample(model, cache, tokenizer, generator, build_prompt, args.sample_prompt)
                net.remove_from_native()
                net.train()
                print(f"\n  \U0001f3ad  [step {step}] {args.sample_prompt}\n     -> {txt}\n")

            if args.save_every and step % args.save_every == 0:
                save(f"[checkpoint step {step}]")

            if args.checkpoint_every and step % args.checkpoint_every == 0:
                cdir = checkpoint_dir(args.out, step)
                net.save_adapter(cdir, base_model_name_or_path=args.model)
                save_trainer_state(cdir, step=step, opt=opt, sched=sched,
                                   best_val=best_val, best_val_step=best_val_step,
                                   ema=ema, offload_opt=offload_opt)
                print(f"  [checkpoint] step {step} -> {cdir} (resumable)")
                prune_checkpoints(args.out, args.keep_checkpoints)

            if prof is not None:
                prof.step()
                # Release the profiler once the active window has been traced
                # (on_trace_ready has fired) so the remaining steps run clean.
                if (step - resume_step) >= 5 + args.torch_profile:
                    prof.stop()
                    prof = None

            # Reached only if the whole iteration ran: this step is now durable.
            done_step = step
    except KeyboardInterrupt:
        if prof is not None:   # window incomplete: no artifacts, just release
            prof.stop()
            prof = None
        # Roll back to the last completed step so the saved counter, the
        # scheduler and the data stream all agree; save()/log_run() below close
        # over `step`. An interrupt inside the first iteration saves resume_step
        # (no progress), which is correct.
        step = done_step
        # Stopping early at the loss plateau is a normal workflow; never discard
        # the adapter trained so far. Under --save-best, --out belongs to the
        # best-val adapter, so the interrupted weights go to final_dir() instead
        # of clobbering it with later (likely overfit) weights.
        if args.save_best and val_examples:
            print(f"\nInterrupted at step {step}; best-val adapter kept in "
                  f"{args.out}.")
            if step > 0:
                save("[interrupted, final weights]", directory=final_dir(args.out))
        else:
            print(f"\nInterrupted at step {step}; saving adapter before exit.")
            if step > 0:
                save("[interrupted]")
        log_run("interrupted", time.time() - t0, None, None)
        _finish_report(exit_code=0, status="interrupted")
        _finish_wandb()
        raise SystemExit(0)

    if prof is not None:       # run shorter than the profile window
        prof.stop()
        prof = None

    # 6. Save adapter (PEFT format; loadable by exllamav3.model.lora.LoRA).
    #    With --save-best the best-val checkpoint already owns --out, so the
    #    (likely overfit) final-step weights go to final_dir() alongside it
    #    rather than being dropped.
    dt = time.time() - t0
    _FAIL_CTX["phase"] = "final_eval"
    # Final held-out numbers. Reuse the last in-loop eval when it already ran on
    # the final step (avoids a duplicate full pass that looks like a hang after
    # "Done."); otherwise compute once, announcing it so the GPU churn is expected.
    if last_eval_step == step:
        val_loss, final_eval2 = last_val, last_eval2
    elif val_examples or val2_examples:
        print(" -- computing final held-out eval (GPU busy, not hung) ...")
        val_loss = evaluate()
        final_eval2 = eval_loss(val2_examples) if val2_examples else None
    else:
        val_loss, final_eval2 = None, None
    if args.save_best and val_examples:
        # --out holds the best-val adapter; keep the final-step weights too, in
        # final_dir(), so a run whose last step misses a --checkpoint-every
        # boundary doesn't lose them.
        save("Done. [final weights]", directory=final_dir(args.out))
    else:
        save("Done.")
    if val_loss is not None:
        tag = f" (best kept: {best_val:.4f})" if args.save_best else ""
        print(f"\n[EVAL] held-out loss (EXL3 arm): {val_loss:.4f}{tag} "
              f"over {len(val_examples)} examples")
    if final_eval2 is not None:
        print(f"[EVAL] eval2 ({eval2_label}) loss: {final_eval2:.4f} "
              f"over {len(val2_examples)} examples")
    if args.vram_spillover:
        peak_str = "n/a (uvm spillover)"
    elif torch.cuda.is_available():
        peak_str = " / ".join(
            f"cuda:{d} {torch.cuda.max_memory_allocated(d) / 1e9:.2f}GB"
            for d in active_devices
        )
    else:
        peak_str = "n/a"
    print(f"[PERF] {tok_seen / dt if dt else 0:,.0f} sup tok/s, "
          f"{tot_seen / dt if dt else 0:,.0f} tot tok/s | "
          f"peak VRAM {peak_str} | {dt:.0f}s for {step} steps | "
          f"step time: {timer.summary()}")
    log_run("completed", dt, val_loss, final_eval2)
    final_summary = {k: v for k, v in {
        "end_loss": end_loss, "final_val": val_loss,
        "final_eval2": final_eval2,
        "best_val": best_val if best_val != float("inf") else None,
        "best_val_step": best_val_step or None,
        "peak_vram_gb": peak_vram_gb(),
        "sup_tok_s": tok_seen / dt if dt else 0,
        "tot_tok_s": tot_seen / dt if dt else 0,
        "total_s": dt, "steps_done": step,
    }.items() if v is not None}
    if report is not None:
        report.update_summary(final_summary)
        _finish_report()
    if wandb_run is not None:
        wandb_run.summary.update(final_summary)
        _finish_wandb()
    print("Verify with: python training/qlora_infer_native.py "
          f"--model {args.model} --adapter {args.out}"
          + (" --mtp   (loads the head + its adapters and drafts with it)"
             if net.mtp is not None else ""))


if __name__ == "__main__":
    main()
