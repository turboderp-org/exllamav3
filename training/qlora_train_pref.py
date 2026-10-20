"""
Preference optimization (DPO / KTO / SimPO / ORPO) of an EXL3 model, native
path -- no HuggingFace Transformers in the loop.

Trains LoRA adapters on a frozen EXL3 base with a preference objective instead
of SFT cross-entropy, using the same transformers-free differentiable forward
as training/qlora_train_native.py. For DPO/KTO the REFERENCE model is the
frozen base itself, obtained by running the same net under
``NativeLlamaQLoRA.adapters_disabled()`` (the PEFT disable-adapter trick) -- no
second model copy is ever loaded, so the VRAM story matches SFT plus one extra
no-grad forward per batch. SimPO and ORPO are REFERENCE-FREE: no reference
forward at all, so a step is roughly half the compute of a DPO step on the
same pairs (and exactly SFT's VRAM story).

Loss semantics follow HuggingFace TRL's stable trainers (DPOTrainer /
KTOTrainer / CPOTrainer's "simpo" loss; KTO was promoted to TRL's stable API
in huggingface/trl#6175), so hyperparameters transfer directly -- see
``exllamav3/training/preference.py`` for the formulas and the TRL
(Apache-2.0) attribution.

Methods:
  --method dpo   Paired data: each row has a prompt, a CHOSEN completion and a
                 REJECTED completion (--prompt-key/--chosen-key/--rejected-key,
                 TRL "explicit prompt" format; conversational values -- lists of
                 {role, content} -- are accepted too, and a multi-turn prompt
                 keeps its FULL history as masked context, so completions score
                 against the turn they answer. Multi-turn prompts need an
                 explicit --prompt-format; auto is single-turn only). One
                 micro-batch of ``--batch`` pairs runs as 2*batch sequences per
                 forward.
  --method kto   UNPAIRED data: each row has a prompt, one completion, and a
                 bool/int label -- True/1 = desirable (--prompt-key/
                 --completion-key/--label-key; conversational/multi-turn
                 prompts as for dpo -- only the final completion is rewarded,
                 per-step supervision is just one row per assistant turn with
                 the history-so-far as prompt). The KL reference point is
                 estimated per micro-batch from mismatched prompt/completion
                 pairs (TRL's +1-offset rotation), which needs --batch >= 2.
  --method simpo Paired data, same format/keys as dpo. Reference-free,
                 length-normalized rewards (beta * mean-per-token logp) with a
                 target margin --gamma (SimPO, arXiv:2405.14734). --beta
                 defaults to 2.0 here (the paper's 2.0-2.5 range), NOT the
                 0.1 of dpo/kto; there is no ln-2 step-0 anchor. Optional
                 --sft-weight > 0 adds TRL CPOTrainer's cpo_alpha NLL term on
                 the chosen completions (the "CPO-SimPO" mix).
  --method orpo  Paired data, same format/keys as dpo. Reference-free,
                 monolithic SFT + preference (ORPO, arXiv:2403.07691; TRL's
                 experimental ORPOTrainer / axolotl ``rl: orpo``): loss =
                 NLL(chosen) - beta * log sigmoid(log odds ratio) on
                 length-normalized probabilities, --beta = the paper's lambda
                 (default 0.1; paper 0.1-0.25). --orpo-nll picks what the NLL
                 term supervises: ``full`` (prompt + chosen, batch-token mean;
                 what TRL and axolotl compute, the default) or ``completion``
                 (chosen tokens only). Meant to replace SFT on a base model,
                 so SFT-like LRs (not DPO's 5e-6) are the usual starting
                 point; it works on instruct checkpoints too.

Usage (defaults are illustrative; DPO on a paired dataset):
    python training/qlora_train_pref.py --method dpo \
        --model /path/to/exl3_model --out out/exl3_dpo_adapter \
        --dataset /data/pairs.jsonl --beta 0.1 --lr 5e-6 \
        --scheduler cosine --warmup-ratio 0.1 --epochs 1

Notes vs the SFT trainer:
  * Single-process only (--parallel single|split). Not wired into the DDP arm
    or the YAML launcher yet (KTO's KL estimate would also need a cross-rank
    all-reduce under DDP).
  * No sample packing (per-sequence logprobs need one document per row) and no
    --train-embeddings/--train-head/--lora-embed/--lora-head (frozen head only).
  * Typical preference LRs are ~10-100x LOWER than SFT (5e-6 .. 5e-5 range).
  * --init-lora pissa/eva/default keep reference == step-0 policy exactly;
    qerr does not (its step 0 is the error-repaired model, the reference is the
    raw quantized base) -- the run prints a note. (simpo has no reference, so
    the qerr caveat does not apply to it.)

Verify the adapter afterwards on the native inference path:
    python training/qlora_infer_native.py --model <model> --adapter <out>
"""

import argparse
import datetime
import os
import random
import sys
import time
import torch

# Reuse the SFT trainer's helpers (same dir on sys.path when run as a script).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qlora_train_native import (  # noqa: E402
    ThroughputMeter, StepTimer, append_run_log,
    checkpoint_dir, prune_checkpoints,
    save_trainer_state, load_trainer_state, restore_optimizer_state,
    snapshot_datasets, resolve_resumed_datasets,
    format_prompt_and_eot, turn_end_token,
    build_optimizer, make_lr_scheduler, resolve_steps_and_warmup,
    load_dataset_split, build_sft_examples, collate,
    _FAIL_CTX, _log_failure, _REPORT, _finish_report,
)
from run_report import RunLogger, start_live_monitor  # noqa: E402
from chat_turns import (extract_turns, single_turn_shape,  # noqa: E402
                        make_segment_builder, encode_segments,
                        encode_completion, AUTO_SINGLE_TURN_HINT)

from exllamav3 import Config, Model, Tokenizer  # noqa: E402
from exllamav3.training.native_llama import NativeLlamaQLoRA  # noqa: E402
from exllamav3.training.preference import (  # noqa: E402
    dpo_loss, kto_loss, simpo_loss, orpo_loss, mismatched_kl_shift,
    ORPO_NLL_MODES,
    DPO_LOSS_TYPES, KTO_LOSS_TYPES,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_rows(dataset_name, split, config_name=None):
    """HF hub id or local json/jsonl/parquet/csv path -> datasets split.
    Same resolution logic as build_sft_examples (shared helper)."""
    return load_dataset_split(dataset_name, split, config_name)


def _prompt_turns(value, rich=False):
    """Prompt column -> normalized turns list. A plain string is a bare user
    turn; a TRL-conversational list keeps its FULL history -- the whole
    conversation becomes masked context, so the completion is scored against
    the turn it actually answers. (The old extract_single_turn collapse broke
    at the first assistant turn, silently pairing completions with a user turn
    from the middle of the history.) ``rich=True`` (the jinja path) keeps
    reasoning_content / tool_calls / tool roles / content-parts lists that
    the hardcoded formats can't render (chat_jinja.extract_rich_turns)."""
    if isinstance(value, str):
        v = value.strip()
        return [{"role": "user", "content": v}] if v else []
    if isinstance(value, list):
        if rich:
            from chat_jinja import extract_rich_turns
            return extract_rich_turns(value)
        return extract_turns(value)
    return []


def _encode_prompt_ids(tokenizer, turns, build_prompt, seg_builder,
                       extras=None):
    """Render + tokenize a prompt/history as fully-masked context ending with
    the assistant turn opener. Single-turn shapes ([system?] user) keep the
    original single-string path, bit-for-bit; multi-turn histories render
    turn-by-turn (prior assistant turns stay masked -- only the separately
    encoded completion is ever scored, the TRL semantic). Returns None for a
    history the format can't render (caller skips and counts the row).

    ``extras`` is not None on the jinja path: EVERY history (single-turn
    included) segment-renders through the model's chat template with the
    row's tools/template_vars in context -- the template is the single
    source of truth, there is no single-turn shortcut."""
    if extras is None:
        single = single_turn_shape(turns, need_assistant=False)
        if single is not None:
            sys_text, user, _ = single
            prompt_text = build_prompt(user, system=sys_text or None)
            ids = tokenizer.encode(
                prompt_text, add_bos=False, encode_special_tokens=True)[0].tolist()
            bos = tokenizer.bos_token_id
            if bos is not None:
                while len(ids) >= 2 and ids[0] == bos and ids[1] == bos:
                    ids = ids[1:]
            return ids
        if seg_builder is None:
            raise SystemExit(AUTO_SINGLE_TURN_HINT)
    try:
        segments = seg_builder(turns, add_generation_prompt=True,
                               **(extras or {}))
    except ValueError:
        return None
    ids, _ = encode_segments(tokenizer, [(t, False) for t, _ in segments])
    return ids


def _completion_text(value):
    """Completion column -> text. Accepts a plain string or a TRL-conversational
    list of messages (assistant contents joined)."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [(m.get("content") or "").strip() for m in value
                 if (m.get("role") or "").lower() == "assistant"]
        return "\n".join(p for p in parts if p).strip()
    return ""


def _fit(prompt_ids, comp_ids, seq_len):
    """Truncate a (prompt, completion) pair to seq_len, completion-tail first
    (mirrors build_sft_examples). Returns the truncated completion or None if
    nothing supervisable survives."""
    room = seq_len - len(prompt_ids)
    if room <= 0:
        return None
    out = comp_ids[:room]
    return out if out else None


def _prompt_renderers(model, tokenizer, prompt_format,
                      chat_template_file=None, template_vars=None):
    """(seg_builder, build_prompt, eot, row_extras_fn) for a prompt format.
    ``row_extras_fn(row)`` is None outside the jinja path; under jinja it
    pulls the row's tools + template_vars/chat_template_kwargs columns for
    the render context (chat_jinja.row_template_extras)."""
    if prompt_format == "jinja":
        from chat_jinja import (jinja_renderers, tokenizer_special_tokens,
                                row_template_extras)
        seg_builder, build_prompt, eot = jinja_renderers(
            tokenizer.config.directory,
            special_tokens=tokenizer_special_tokens(tokenizer),
            template_file=chat_template_file, default_vars=template_vars)
        return (seg_builder, build_prompt, eot or turn_end_token(tokenizer),
                row_template_extras)
    build_prompt, eot = format_prompt_and_eot(model, tokenizer, prompt_format)
    seg_builder = make_segment_builder(prompt_format,
                                       bos_token=tokenizer.bos_token,
                                       eos_token=tokenizer.eos_token)
    return seg_builder, build_prompt, eot, None


def build_dpo_examples(model, tokenizer, dataset_name, split, seq_len,
                       prompt_key="prompt", chosen_key="chosen",
                       rejected_key="rejected", max_samples=0,
                       shuffle=False, shuffle_seed=0, prompt_format="auto",
                       config_name=None, chat_template_file=None,
                       template_vars=None):
    """Tokenize a paired preference dataset (TRL explicit-prompt format) with
    the model's chat template. Each example keeps the prompt and the two
    completions separate so collation can mask the prompt exactly:
    ``{prompt_ids, chosen_ids, rejected_ids}`` (completions end with the
    architecture-correct turn-end token -- under --prompt-format jinja, the
    close the model's own chat template appends)."""
    ds = _load_rows(dataset_name, split, config_name)
    if shuffle or (max_samples and max_samples < len(ds)):
        ds = ds.shuffle(seed=shuffle_seed)
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    seg_builder, build_prompt, eot, row_extras = _prompt_renderers(
        model, tokenizer, prompt_format, chat_template_file, template_vars)
    examples, skipped = [], 0
    for row in ds:
        turns = _prompt_turns(row.get(prompt_key), rich=row_extras is not None)
        chosen = _completion_text(row.get(chosen_key))
        rejected = _completion_text(row.get(rejected_key))
        if not turns or not chosen or not rejected:
            skipped += 1
            continue
        prompt_ids = _encode_prompt_ids(
            tokenizer, turns, build_prompt, seg_builder,
            extras=row_extras(row) if row_extras else None)
        if prompt_ids is None:
            skipped += 1
            continue
        chosen_ids = encode_completion(tokenizer, chosen, eot)
        rejected_ids = encode_completion(tokenizer, rejected, eot)
        chosen_ids = _fit(prompt_ids, chosen_ids, seq_len)
        rejected_ids = _fit(prompt_ids, rejected_ids, seq_len)
        if chosen_ids is None or rejected_ids is None:
            skipped += 1
            continue
        examples.append({"prompt_ids": prompt_ids, "chosen_ids": chosen_ids,
                         "rejected_ids": rejected_ids})
    if skipped:
        print(f" -- skipped {skipped} rows (missing fields or over --seq-len)")
    return examples


def build_kto_examples(model, tokenizer, dataset_name, split, seq_len,
                       prompt_key="prompt", completion_key="completion",
                       label_key="label", max_samples=0,
                       shuffle=False, shuffle_seed=0, prompt_format="auto",
                       config_name=None, chat_template_file=None,
                       template_vars=None):
    """Tokenize an unpaired KTO dataset (TRL format: prompt / completion /
    bool label). Returns ``{prompt_ids, completion_ids, label}`` dicts."""
    ds = _load_rows(dataset_name, split, config_name)
    if shuffle or (max_samples and max_samples < len(ds)):
        ds = ds.shuffle(seed=shuffle_seed)
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    seg_builder, build_prompt, eot, row_extras = _prompt_renderers(
        model, tokenizer, prompt_format, chat_template_file, template_vars)
    examples, skipped = [], 0
    for row in ds:
        turns = _prompt_turns(row.get(prompt_key), rich=row_extras is not None)
        completion = _completion_text(row.get(completion_key))
        label = row.get(label_key)
        if not turns or not completion or label is None:
            skipped += 1
            continue
        prompt_ids = _encode_prompt_ids(
            tokenizer, turns, build_prompt, seg_builder,
            extras=row_extras(row) if row_extras else None)
        if prompt_ids is None:
            skipped += 1
            continue
        comp_ids = encode_completion(tokenizer, completion, eot)
        comp_ids = _fit(prompt_ids, comp_ids, seq_len)
        if comp_ids is None:
            skipped += 1
            continue
        examples.append({"prompt_ids": prompt_ids, "completion_ids": comp_ids,
                         "label": bool(label)})
    if skipped:
        print(f" -- skipped {skipped} rows (missing fields or over --seq-len)")
    n_d = sum(1 for e in examples if e["label"])
    n_u = len(examples) - n_d
    print(f" -- KTO labels: {n_d} desirable / {n_u} undesirable")
    if n_d and n_u:
        # KTO paper / TRL guidance: keep desirable_weight*n_d / undesirable_
        # weight*n_u in [1, 4/3]; suggest a weight when the data is imbalanced.
        ratio = n_d / n_u
        if ratio > 4 / 3:
            print(f"    (imbalanced: consider --desirable-weight ~{1 / ratio:.2f}"
                  f" .. {4 / (3 * ratio):.2f})")
        elif ratio < 1:
            print(f"    (imbalanced: consider --undesirable-weight ~{ratio:.2f}"
                  f" .. {4 * ratio / 3:.2f})")
    return examples


def rows_to_batch(rows, pad_id):
    """(prompt_ids, completion_ids) rows -> right-padded (input_ids, labels,
    attention_mask) with the prompt masked -100 (completion-only supervision)."""
    seqs = [(p + c, [-100] * len(p) + list(c)) for p, c in rows]
    maxlen = max(len(s[0]) for s in seqs)
    input_ids, labels, attn = [], [], []
    for ids, labs in seqs:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(labs + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long))


# ---------------------------------------------------------------------------
# Batch loss (policy + frozen-base reference through ONE net)
# ---------------------------------------------------------------------------

def dpo_batch_metrics(net, batch, pad_id, args):
    """Loss + logging metrics for one micro-batch of DPO pairs. The 2*batch
    rows (chosen block, then rejected block) share a single policy forward and
    a single no-grad reference forward."""
    rows = ([(e["prompt_ids"], e["chosen_ids"]) for e in batch]
            + [(e["prompt_ids"], e["rejected_ids"]) for e in batch])
    input_ids, labels, attn = rows_to_batch(rows, pad_id)
    pol_logps, counts = net.compute_logps(input_ids, labels, attention_mask=attn,
                                          chunk=args.ce_chunk)
    with torch.no_grad(), net.adapters_disabled():
        ref_logps, _ = net.compute_logps(input_ids, labels, attention_mask=attn,
                                         chunk=args.ce_chunk)
    b = len(batch)
    losses, cr, rr = dpo_loss(
        pol_logps[:b], pol_logps[b:], ref_logps[:b], ref_logps[b:],
        beta=args.beta, label_smoothing=args.label_smoothing,
        loss_type=args.dpo_loss,
        chosen_counts=counts[:b], rejected_counts=counts[b:])
    loss = losses.mean()
    nll = (-pol_logps[:b] / counts[:b].clamp(min=1)).mean()   # mean-token NLL on chosen
    if args.rpo_alpha > 0:                                     # RPO / DPO+SFT (Pang et al. 2024)
        loss = loss + args.rpo_alpha * nll
    metrics = {
        "margin": (cr - rr).mean().item(),
        "acc": (cr > rr).float().mean().item(),
        "nll": nll.item(),
        "sup": int((labels[:, 1:] != -100).sum()),
        "tot": int(attn.sum()),
        "n": b,
    }
    return loss, metrics


def simpo_batch_metrics(net, batch, pad_id, args):
    """Loss + logging metrics for one micro-batch of SimPO pairs. Same data
    layout as DPO (chosen block, then rejected block, ONE policy forward) but
    reference-free: there is no adapters_disabled() forward at all. With
    --sft-weight > 0 the batch-token-mean NLL of the CHOSEN completions is
    mixed in (TRL CPOTrainer's cpo_alpha term)."""
    rows = ([(e["prompt_ids"], e["chosen_ids"]) for e in batch]
            + [(e["prompt_ids"], e["rejected_ids"]) for e in batch])
    input_ids, labels, attn = rows_to_batch(rows, pad_id)
    pol_logps, counts = net.compute_logps(input_ids, labels, attention_mask=attn,
                                          chunk=args.ce_chunk)
    b = len(batch)
    losses, cr, rr = simpo_loss(
        pol_logps[:b], pol_logps[b:], counts[:b], counts[b:],
        beta=args.beta, gamma=args.gamma,
        label_smoothing=args.label_smoothing)
    loss = losses.mean()
    metrics = {
        "margin": (cr - rr).mean().item(),
        "acc": (cr > rr).float().mean().item(),
        "sup": int((labels[:, 1:] != -100).sum()),
        "tot": int(attn.sum()),
        "n": b,
    }
    if args.sft_weight > 0:
        nll = -pol_logps[:b].sum() / counts[:b].sum().clamp(min=1)
        loss = loss + args.sft_weight * nll
        metrics["nll"] = nll.item()
    return loss, metrics


def orpo_batch_metrics(net, batch, pad_id, args):
    """Loss + logging metrics for one micro-batch of ORPO pairs. Same data
    layout as DPO/SimPO (chosen block, then rejected block, ONE policy
    forward, no reference). Loss = NLL(chosen) + odds-ratio term. With
    --orpo-nll full (TRL / axolotl semantics) the forward supervises every
    non-pad token, so the per-token logps serve both the full-sequence NLL
    on the chosen rows and the completion-only means the odds need; with
    ``completion`` the NLL is the batch-token mean over chosen completion
    tokens (the same term as SimPO's --sft-weight)."""
    rows = ([(e["prompt_ids"], e["chosen_ids"]) for e in batch]
            + [(e["prompt_ids"], e["rejected_ids"]) for e in batch])
    input_ids, labels, attn = rows_to_batch(rows, pad_id)
    b = len(batch)
    cmask = labels[:, 1:] != -100                      # completion positions
    if args.orpo_nll == "full":
        full_labels = input_ids.masked_fill(attn == 0, -100)
        tok_logps, fmask = net.compute_logps(
            input_ids, full_labels, attention_mask=attn, chunk=args.ce_chunk,
            return_per_token=True)
        cmask = cmask.to(tok_logps.device)
        pol_logps = (tok_logps * cmask).sum(dim=-1)
        counts = cmask.sum(dim=-1)
        fm = fmask[:b]
        nll = -(tok_logps[:b] * fm).sum() / fm.sum().clamp(min=1)
    else:
        pol_logps, counts = net.compute_logps(input_ids, labels, attention_mask=attn,
                                              chunk=args.ce_chunk)
        nll = -pol_logps[:b].sum() / counts[:b].sum().clamp(min=1)
    losses, cr, rr, log_odds = orpo_loss(
        pol_logps[:b], pol_logps[b:], counts[:b], counts[b:], beta=args.beta)
    loss = nll + losses.mean()
    metrics = {
        "margin": (cr - rr).mean().item(),
        "acc": (cr > rr).float().mean().item(),
        "nll": nll.item(),
        "log_odds": log_odds.mean().item(),
        "sup": int(cmask.sum()),
        "tot": int(attn.sum()),
        "n": b,
    }
    return loss, metrics


def kto_batch_metrics(net, batch, pad_id, args):
    """Loss + logging metrics for one micro-batch of KTO rows. Two extra
    no-grad forwards estimate the KL reference point on mismatched pairs
    (skipped for --kto-loss apo_zero_unpaired, or a singleton batch)."""
    rows = [(e["prompt_ids"], e["completion_ids"]) for e in batch]
    input_ids, labels, attn = rows_to_batch(rows, pad_id)
    pol_logps, _ = net.compute_logps(input_ids, labels, attention_mask=attn,
                                     chunk=args.ce_chunk)
    with torch.no_grad(), net.adapters_disabled():
        ref_logps, _ = net.compute_logps(input_ids, labels, attention_mask=attn,
                                         chunk=args.ce_chunk)

    pol_kl = ref_kl = None
    if args.kto_loss == "kto" and len(batch) > 1:
        shift = mismatched_kl_shift(len(batch))
        kl_rows = []
        for i, j in enumerate(shift):
            comp = _fit(batch[i]["prompt_ids"], batch[j]["completion_ids"],
                        args.seq_len)
            if comp:
                kl_rows.append((batch[i]["prompt_ids"], comp))
        if kl_rows:
            kl_ids, kl_labels, kl_attn = rows_to_batch(kl_rows, pad_id)
            with torch.no_grad():
                pol_kl, _ = net.compute_logps(kl_ids, kl_labels,
                                              attention_mask=kl_attn,
                                              chunk=args.ce_chunk)
                with net.adapters_disabled():
                    ref_kl, _ = net.compute_logps(kl_ids, kl_labels,
                                                  attention_mask=kl_attn,
                                                  chunk=args.ce_chunk)

    des = torch.tensor([e["label"] for e in batch], dtype=torch.bool,
                       device=pol_logps.device)
    losses, cr, rr, kl = kto_loss(
        pol_logps[des], pol_logps[~des], pol_kl,
        ref_logps[des], ref_logps[~des], ref_kl,
        beta=args.beta, desirable_weight=args.desirable_weight,
        undesirable_weight=args.undesirable_weight, loss_type=args.kto_loss)
    loss = losses.mean()
    metrics = {
        "kl": kl.item(),
        "reward_d": cr.mean().item() if cr.numel() else float("nan"),
        "reward_u": rr.mean().item() if rr.numel() else float("nan"),
        "sup": int((labels[:, 1:] != -100).sum()),
        "tot": int(attn.sum()),
        "n": len(batch),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        _run_main()
    except KeyboardInterrupt as e:
        _log_failure("interrupted", e)
        raise SystemExit(130)
    except SystemExit as e:
        if e.code not in (0, None):
            _log_failure("failed", e)
        raise
    except BaseException as e:
        _log_failure("failed", e)
        raise


def _run_main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    if os.environ.get("RANK") is not None or os.environ.get("WORLD_SIZE") is not None:
        raise SystemExit(
            "qlora_train_pref.py is single-process (--parallel single|split); "
            "there is no DDP arm for preference training yet (KTO's KL estimate "
            "would need a cross-rank all-reduce).")

    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["dpo", "kto", "simpo", "orpo"],
                    required=True,
                    help="Preference objective: dpo (paired chosen/rejected), "
                         "kto (unpaired desirable/undesirable labels), "
                         "simpo (paired, reference-free, length-normalized), "
                         "or orpo (paired, reference-free SFT + odds ratio).")
    ap.add_argument("--model", required=True, help="Path to a local EXL3 model dir")
    ap.add_argument("--out", default="out/exl3_pref_adapter")
    ap.add_argument("--device", default="cuda:0",
                    help="single-device load target (ignored when --parallel split)")
    ap.add_argument("--parallel", choices=["single", "split"], default="single")
    ap.add_argument("--reserve-per-device", nargs="*", type=float, default=None, metavar="GB")
    ap.add_argument("--use-per-device", nargs="*", type=float, default=None, metavar="GB")
    # LoRA
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--use-rslora", action="store_true")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="Target module leaf names (default: attn+mlp projections)")
    ap.add_argument("--expert-r", type=int, default=None,
                    help="LoRA rank for ROUTED-expert adapters (expert_* targets); "
                         "default: same as --r. See qlora_train_native.py.")
    ap.add_argument("--init-lora", choices=["default", "pissa", "qerr", "eva"],
                    default="default",
                    help="Adapter init (see qlora_train_native.py). default/pissa/"
                         "eva keep reference == step-0 policy; qerr does not "
                         "(reference is the raw quantized base).")
    ap.add_argument("--init-svd-niter", type=int, default=16)
    ap.add_argument("--init-ref-model", default=None)
    ap.add_argument("--init-eva-tokens", type=int, default=65536)
    # Preference objective
    ap.add_argument("--beta", type=float, default=None,
                    help="Inverse temperature on the log-ratio. Default: 0.1 "
                         "for dpo/kto (TRL default); 2.0 for simpo (the SimPO "
                         "paper's 2.0-2.5 range -- note TRL's CPOConfig still "
                         "defaults to 0.1, pass it explicitly for parity); "
                         "0.1 for orpo (the paper's lambda; TRL default).")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="(dpo sigmoid / simpo) cDPO-style robust label-flip "
                         "probability.")
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="(simpo) target reward margin on the length-normalized "
                         "logp difference (TRL's simpo_gamma; paper range "
                         "0.5-1.6, scaled with beta).")
    ap.add_argument("--sft-weight", type=float, default=0.0,
                    help="(simpo) weight of an added batch-token-mean NLL loss "
                         "on the CHOSEN completions (TRL CPOTrainer's "
                         "cpo_alpha). 0 = the pure SimPO paper objective; "
                         "> 0 = the CPO-SimPO mix.")
    ap.add_argument("--orpo-nll", choices=list(ORPO_NLL_MODES), default="full",
                    help="(orpo) tokens the SFT (NLL) term supervises: full = "
                         "prompt + chosen completion, batch-token mean (what "
                         "TRL's ORPOTrainer and axolotl compute; default), "
                         "completion = chosen completion tokens only.")
    ap.add_argument("--rpo-alpha", type=float, default=0.0,
                    help="(dpo) RPO / DPO+SFT: weight on an auxiliary length-"
                         "normalized NLL loss over the CHOSEN completion (Pang "
                         "et al. 2024; TRL rpo_alpha). 0 = pure DPO; 1.0 is a "
                         "sane start. Raises the ABSOLUTE prob of chosen so it "
                         "actually generates, not just its reward vs rejected.")
    ap.add_argument("--dpo-loss", choices=list(DPO_LOSS_TYPES), default="sigmoid",
                    help="DPO variant: sigmoid (default), hinge (SLiC), ipo "
                         "(length-normalized).")
    ap.add_argument("--kto-loss", choices=list(KTO_LOSS_TYPES), default="kto",
                    help="KTO variant: kto (batch-KL reference point, needs "
                         "--batch >= 2) or apo_zero_unpaired (no KL term).")
    ap.add_argument("--desirable-weight", type=float, default=1.0,
                    help="(kto) loss weight on desirable examples (lambda_D).")
    ap.add_argument("--undesirable-weight", type=float, default=1.0,
                    help="(kto) loss weight on undesirable examples (lambda_U).")
    # Optimization
    ap.add_argument("--lr", type=float, default=5e-6,
                    help="Preference LRs run ~10-100x lower than SFT (TRL DPO "
                         "default 1e-6; 5e-6..5e-5 is a common LoRA range).")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--optim", choices=["adamw", "adamw8bit", "paged_adamw8bit"],
                    default="adamw")
    ap.add_argument("--scheduler", choices=["none", "linear", "cosine"], default="none")
    ap.add_argument("--warmup-ratio", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=2,
                    help="Examples per micro-batch. NOTE a DPO pair is TWO "
                         "sequences (chosen + rejected share one forward), and "
                         "every batch also runs no-grad reference (and, for KTO, "
                         "KL) forwards -- size accordingly. KTO's KL estimate "
                         "needs --batch >= 2.")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    # Data
    ap.add_argument("--dataset", required=True,
                    help="HF dataset id or local json/jsonl/parquet/csv path.")
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--prompt-key", default="prompt")
    ap.add_argument("--chosen-key", default="chosen", help="(dpo)")
    ap.add_argument("--rejected-key", default="rejected", help="(dpo)")
    ap.add_argument("--completion-key", default="completion", help="(kto)")
    ap.add_argument("--label-key", default="label",
                    help="(kto) bool/int column; truthy = desirable")
    ap.add_argument("--prompt-format",
                    choices=["auto", "mistral", "metharme", "gemma4-nothink",
                             "llama3", "qwen3.5", "qwen3.5-nothink", "chatml",
                             "hy3", "hy3-nothink", "jinja"],
                    default="auto",
                    help="Chat format for the prompt/history (see the SFT "
                         "trainer's help). jinja: the model directory's own "
                         "Jinja chat template -- multi-turn histories with "
                         "tool roles / tool_calls / reasoning_content render "
                         "through it, and rows may carry tools + template_vars/"
                         "chat_template_kwargs columns (chat_jinja.py).")
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
                         "strength:', 'Knowledge cutoff:' and 'Current "
                         "date:' lines from every rendered prompt (OFF by "
                         "default). Harmony-style templates synthesize "
                         "these into the system block for rows with no "
                         "system message; the date comes from "
                         "strftime_now(), so without this the rendered "
                         "corpus changes from day to day.")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all rows")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=1024,
                    help="Max prompt+completion tokens per sequence (completion "
                         "tail truncated; rows whose prompt alone exceeds this "
                         "are skipped).")
    ap.add_argument("--inspect", type=int, default=0, metavar="N",
                    help="Decode the first N built examples (prompt vs "
                         "completion spans) and exit without training.")
    # Eval / saving
    ap.add_argument("--eval-split", default=None,
                    help="Held-out split of the dataset (e.g. 'test') for eval "
                         "preference loss + reward metrics.")
    ap.add_argument("--eval-dataset", default=None,
                    help="Dataset id/path for --eval-split (defaults to --dataset).")
    ap.add_argument("--eval-max-samples", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="Carve this fraction off train for eval when no "
                         "--eval-split is given.")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--save-best", action="store_true",
                    help="Keep the checkpoint with the best held-out preference "
                         "loss (needs an eval set + --eval-every).")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--keep-checkpoints", type=int, default=0)
    # SFT-style held-out eval, alongside (or instead of) the preference eval.
    # The preference eval scores PAIRS; this one scores completion-only
    # cross-entropy on an ordinary messages/instruction set, so a preference
    # run can track "is the target voice getting more likely" on data that has
    # no chosen/rejected structure at all. Same builder + loss as the SFT
    # trainer, so the number is directly comparable to an SFT run's held_out.
    ap.add_argument("--sft-eval-dataset", default=None,
                    help="Dataset id/path for an SFT-style held-out CE eval "
                         "(messages or instruction columns -- NOT preference "
                         "pairs). Evaluated on the --eval-every schedule.")
    ap.add_argument("--sft-eval-split", default="train",
                    help="Split of --sft-eval-dataset (local jsonl loads as "
                         "'train').")
    ap.add_argument("--sft-eval-messages-key", default="messages",
                    help="Messages column of --sft-eval-dataset; set empty to "
                         "use the flat instruction/context/response columns.")
    ap.add_argument("--sft-eval-max-samples", type=int, default=0)
    # Local run report + live monitor (mirrors the SFT trainer).
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-report", action="store_true",
                    help="Skip the local HTML run report.")
    ap.add_argument("--live-report", action="store_true",
                    help="Serve the live run monitor while training. Charts "
                         "only -- the step dataset browser is SFT-only.")
    ap.add_argument("--live-report-port", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--reset-optimizer", action="store_true")
    ap.add_argument("--run-log", default="qlora_runs.csv")
    # Model/runtime knobs shared with the SFT trainer
    ap.add_argument("--compute-dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--attn-impl", choices=["auto", "eager", "flash"], default="auto")
    ap.add_argument("--ce-chunk", type=int, default=1024)
    ap.add_argument("--head-vocab-chunk", type=int, default=0)
    ap.add_argument("--offload-activations", action="store_true")
    ap.add_argument("--offload-mode", choices=["async", "sync"], default="async")
    ap.add_argument("--use-liger", action="store_true")
    ap.add_argument("--dequant-mode", choices=["fast", "legacy"], default="fast",
                    help="Frozen-weight dequant path (audit A1); see the SFT arm.")
    ap.add_argument("--dequant-cache", action="store_true",
                    help="Opt-in recompute->backward frozen-weight cache (a net "
                         "loss under the default fast path; see the SFT arm).")
    args = ap.parse_args()

    # Process-wide, before any renderer is built (see chat_jinja).
    from chat_jinja import set_strip_sys_prompt_extras
    set_strip_sys_prompt_extras(args.strip_sys_prompt_extras)

    if args.beta is None:
        args.beta = 2.0 if args.method == "simpo" else 0.1

    from exllamav3.training import backbone as _backbone
    _backbone.set_dequant_mode(args.dequant_mode)

    method = args.method

    # Dataset-mix continuity (see the SFT trainer): fresh runs snapshot their
    # data files into <out>/dataset/; resumes verify against that record and
    # may substitute the snapshot. Before the failure-log record below so the
    # RESOLVED paths get logged.
    if args.out:
        data_roles = {"dataset": args.dataset,
                      "eval_dataset": args.eval_dataset}
        if args.resume:
            resolved = resolve_resumed_datasets(args.out, args.resume, data_roles)
            args.dataset = resolved["dataset"]
            args.eval_dataset = resolved["eval_dataset"]
        else:
            snapshot_datasets(args.out, data_roles)

    _FAIL_CTX["run_log"] = args.run_log
    _FAIL_CTX["phase"] = "startup"
    _FAIL_CTX["record"] = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "arm": f"exl3-{method}", "model": args.model, "out": args.out,
        "dataset": args.dataset, "eval_split": args.eval_split or "",
        "eval_dataset": args.eval_dataset or "",
        "r": args.r, "alpha": args.alpha,
        "use_rslora": int(bool(args.use_rslora)), "init_lora": args.init_lora,
        "lr": args.lr, "scheduler": args.scheduler,
        "weight_decay": args.weight_decay,
        "batch": args.batch, "grad_accum": args.grad_accum, "world_size": 1,
        "eff_batch": args.batch * args.grad_accum, "epochs": args.epochs,
        "steps_planned": args.steps, "seq_len": args.seq_len,
        "compute_dtype": args.compute_dtype, "attn_impl": args.attn_impl,
        "parallel": args.parallel, "shuffle": int(bool(args.shuffle)),
        "max_samples": args.max_samples, "prompt_format": args.prompt_format,
        "notes": _hparam_note(args),
    }

    cdt = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16}[args.compute_dtype]

    # 1. Load the native model + tokenizer.
    _FAIL_CTX["phase"] = "load_model"
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    if args.parallel == "split":
        load_kwargs = {}
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

    # 2. Differentiable QLoRA net. Frozen head only: the fused per-token logps
    #    path has no trainable-head gradient (and preference training rarely
    #    wants one), so the embed/head trainable surfaces are not offered here.
    _FAIL_CTX["phase"] = "build_net"
    _FAIL_CTX["record"]["arch"] = getattr(config, "architecture", "")
    net = NativeLlamaQLoRA(
        model, r=args.r, alpha=args.alpha, target_modules=args.targets,
        use_rslora=args.use_rslora, compute_dtype=cdt,
        gradient_checkpointing=not args.no_grad_ckpt,
        attn_impl=args.attn_impl, head_vocab_chunk=args.head_vocab_chunk,
        offload_activations=args.offload_activations,
        offload_mode=args.offload_mode, use_liger=args.use_liger,
        expert_r=args.expert_r,
    )
    net.train()
    if args.init_lora in ("pissa", "qerr"):
        _FAIL_CTX["phase"] = "init_lora"
        net.apply_init_lora(args.init_lora, ref_model_dir=args.init_ref_model,
                            svd_niter=args.init_svd_niter)
        if args.init_lora == "qerr" and method not in ("simpo", "orpo"):
            print(" -- note: with --init-lora qerr the step-0 policy is the "
                  "error-repaired model but the DPO/KTO reference is the raw "
                  "quantized base, so rewards start slightly nonzero.")
    if args.resume:
        net.load_adapter(args.resume)
    print(f" -- trainable params: {net.num_trainable():,} "
          f"(r={args.r}, alpha={args.alpha}, targets={net.target_modules})")
    print(f" -- {net.describe_attn()}")
    if method == "dpo":
        mline = f"loss {args.dpo_loss}, label_smoothing {args.label_smoothing}"
    elif method == "kto":
        mline = (f"loss {args.kto_loss}, weights D {args.desirable_weight} / "
                 f"U {args.undesirable_weight}")
    elif method == "orpo":
        mline = f"nll {args.orpo_nll} (reference-free)"
    else:
        mline = (f"gamma {args.gamma}, sft_weight {args.sft_weight}, "
                 f"label_smoothing {args.label_smoothing} (reference-free)")
    print(f" -- method: {method} | beta {args.beta} | {mline}")
    if method == "kto" and args.kto_loss == "kto" and args.batch < 2:
        raise SystemExit("--method kto with --kto-loss kto needs --batch >= 2 "
                         "(the KL reference point is estimated from mismatched "
                         "pairs within the micro-batch). Use --batch 2+ or "
                         "--kto-loss apo_zero_unpaired.")

    # 3. Data.
    _FAIL_CTX["phase"] = "build_dataset"
    paired = method in ("dpo", "simpo", "orpo")   # simpo/orpo share the DPO data format
    build_examples = build_dpo_examples if paired else build_kto_examples
    keys = (dict(prompt_key=args.prompt_key, chosen_key=args.chosen_key,
                 rejected_key=args.rejected_key) if paired else
            dict(prompt_key=args.prompt_key, completion_key=args.completion_key,
                 label_key=args.label_key))
    from chat_jinja import parse_template_vars
    template_vars = parse_template_vars(args.template_vars)
    examples = build_examples(
        model, tokenizer, args.dataset, args.dataset_split, args.seq_len,
        max_samples=args.max_samples, shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed, prompt_format=args.prompt_format,
        config_name=args.dataset_config,
        chat_template_file=args.chat_template_file,
        template_vars=template_vars, **keys)
    print(f" -- {len(examples)} {method.upper()} examples"
          f"{' (shuffled)' if args.shuffle else ''}")
    assert examples, "no usable training examples"

    if args.inspect:
        dec = lambda seq: tokenizer.decode(torch.tensor([seq]),
                                           decode_special_tokens=True)
        for i, ex in enumerate(examples[:args.inspect]):
            print(f"\n===== example {i} =====")
            print(f"  PROMPT (masked): {dec(ex['prompt_ids'])!r}")
            if paired:
                print(f"  CHOSEN  (supervised): {dec(ex['chosen_ids'])!r}")
                print(f"  REJECTED(supervised): {dec(ex['rejected_ids'])!r}")
            else:
                tag = "desirable" if ex["label"] else "UNdesirable"
                print(f"  COMPLETION ({tag}, supervised): {dec(ex['completion_ids'])!r}")
        print(f"\n -- inspect only ({args.inspect} shown); exiting before training.")
        return

    # Held-out eval set: a real split when given, else a val_frac carve.
    val_examples = []
    if args.eval_split:
        val_examples = build_examples(
            model, tokenizer, args.eval_dataset or args.dataset,
            args.eval_split, args.seq_len, max_samples=args.eval_max_samples,
            prompt_format=args.prompt_format, config_name=args.dataset_config,
            chat_template_file=args.chat_template_file,
            template_vars=template_vars, **keys)
        print(f" -- held-out eval: {len(val_examples)} examples from split "
              f"'{args.eval_split}'")
    elif args.val_frac > 0:
        n_val = max(1, int(len(examples) * args.val_frac))
        val_examples, examples = examples[:n_val], examples[n_val:]
        print(f" -- held out {len(val_examples)} val examples; "
              f"{len(examples)} for training")
        assert examples, "val_frac too large; no training examples left"

    # SFT-style held-out set (--sft-eval-dataset): ordinary completion-masked
    # rows, built by the SFT trainer's own builder so the CE number means the
    # same thing it does there. Independent of the preference eval above --
    # either, both, or neither may be configured.
    sft_val_examples = []
    if args.sft_eval_dataset:
        _FAIL_CTX["phase"] = "build_sft_eval"
        sft_val_examples = build_sft_examples(
            model, tokenizer, args.sft_eval_dataset,
            args.sft_eval_max_samples, args.seq_len,
            split=args.sft_eval_split,
            messages_key=args.sft_eval_messages_key or None,
            prompt_format=args.prompt_format,
            config_name=args.dataset_config,
            chat_template_file=args.chat_template_file,
            template_vars=template_vars)
        print(f" -- SFT-style held-out eval: {len(sft_val_examples)} examples "
              f"from {args.sft_eval_dataset} (split '{args.sft_eval_split}')")
        assert sft_val_examples, (
            "--sft-eval-dataset produced no usable examples (check "
            "--sft-eval-split and --sft-eval-messages-key)")

    # eva init: stream the preference batches (policy view) through the no-grad
    # pre-pass, exactly like the SFT trainer feeds its training blocks.
    if args.init_lora == "eva" and not args.resume:
        _FAIL_CTX["phase"] = "init_lora"

        def eva_prepass():
            used, i = 0, 0
            while used < args.init_eva_tokens and i < len(examples):
                batch = examples[i:i + args.batch]
                i += len(batch)
                if paired:
                    rows = [(e["prompt_ids"], e["chosen_ids"]) for e in batch]
                else:
                    rows = [(e["prompt_ids"], e["completion_ids"]) for e in batch]
                input_ids, _, attn = rows_to_batch(rows, pad_id)
                used += int(attn.sum())
                yield dict(input_ids=input_ids, attention_mask=attn)

        net.apply_init_lora("eva", svd_niter=args.init_svd_niter,
                            eva_batches=eva_prepass())
    elif args.init_lora == "eva":
        print(" -- eva init skipped on --resume (the checkpoint's adapters "
              "already carry it)")

    args.steps, warmup_steps = resolve_steps_and_warmup(
        args, len(examples), args.batch * args.grad_accum)
    print(f" -- {args.steps} steps, scheduler={args.scheduler}, "
          f"warmup={warmup_steps}, weight_decay={args.weight_decay}")

    # 4. Optimizer + schedule (+ optional resumable state).
    opt = build_optimizer(net.param_groups(args.weight_decay), args.lr, args.optim)
    sched = make_lr_scheduler(opt, args.scheduler, args.steps, warmup_steps)
    resume_step, resume_state = 0, None
    yaml_base_lrs = list(sched.base_lrs)  # per-group base LRs from the CONFIG
    if args.resume and not args.reset_optimizer:
        resume_state = load_trainer_state(args.resume)
        if resume_state is not None:
            restore_optimizer_state(opt, resume_state["optimizer"])
            if resume_state.get("scheduler") is not None:
                sched.load_state_dict(resume_state["scheduler"])
            # Config lr wins on resume (see the SFT trainer): re-base the
            # restored schedule at the config value; moments/position kept.
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
                  f"step {resume_step + 1}/{args.steps}")
        else:
            print(f" -- {args.resume} has no trainer_state.pt; resuming weights "
                  f"only (cold optimizer + schedule from step 0).")

    def batches():
        order = list(range(len(examples)))
        while True:
            random.Random(0).shuffle(order)
            for i in range(0, len(order) - args.batch + 1, args.batch):
                yield [examples[j] for j in order[i:i + args.batch]]

    # |dB| telemetry baseline (same convention as the SFT trainer: distance from
    # the init, so SVD-initialized runs show the trained delta, not the init).
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
        with torch.no_grad():
            tot = 0.0
            for w, b0 in b0_refs:
                if b0 is None:
                    tot += w.lora_b.float().pow(2).sum().item()
                else:
                    tot += (w.lora_b.detach().float().cpu() - b0).pow(2).sum().item()
            return tot ** 0.5

    batch_metrics = {"dpo": dpo_batch_metrics, "kto": kto_batch_metrics,
                     "simpo": simpo_batch_metrics,
                     "orpo": orpo_batch_metrics}[method]

    def save(tag):
        net.save_adapter(args.out, base_model_name_or_path=args.model)
        save_trainer_state(args.out, step=step, opt=opt, sched=sched,
                           best_val=best_val, best_val_step=best_val_step, ema=ema)
        print(f"{tag} Adapter written to {args.out}")

    def evaluate():
        """Mean preference loss (+ reward metrics) over the eval set, in
        micro-batches of --batch so the KTO KL estimate stays defined."""
        if not val_examples:
            return None, {}
        net.eval()
        tot_loss, tot_n = 0.0, 0
        agg = {}
        with torch.no_grad():
            for i in range(0, len(val_examples), args.batch):
                vb = val_examples[i:i + args.batch]
                if method == "kto" and args.kto_loss == "kto" and len(vb) < 2:
                    continue  # a KL-less singleton tail would skew the mean
                loss, m = batch_metrics(net, vb, pad_id, args)
                tot_loss += loss.item() * m["n"]
                tot_n += m["n"]
                for k in ("margin", "acc", "kl", "reward_d", "reward_u", "nll",
                          "log_odds"):
                    if k in m and m[k] == m[k]:   # skip NaN (empty subsets)
                        s, n = agg.get(k, (0.0, 0))
                        agg[k] = (s + m[k] * m["n"], n + m["n"])
        net.train()
        if not tot_n:
            return None, {}
        return tot_loss / tot_n, {k: s / n for k, (s, n) in agg.items()}

    def sft_eval_loss():
        """Mean completion-only CE over --sft-eval-dataset, one example at a
        time (no padding effects). Deliberately identical to the SFT trainer's
        eval_loss, so this number is directly comparable to an SFT run's
        eval/held_out rather than being a preference-specific quantity."""
        if not sft_val_examples:
            return None
        net.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for ex in sft_val_examples:
                input_ids, labels, attn, pos_ids, seg_ids = collate([ex], pad_id)
                total += net.compute_loss(
                    input_ids, labels, attention_mask=attn,
                    chunk=args.ce_chunk,
                    position_ids=pos_ids, seg_ids=seg_ids).item()
                n += 1
        net.train()
        return total / n if n else None

    def fmt_eval(vl, em):
        parts = [f"pref-loss {vl:.4f}"]
        if "acc" in em:
            parts.append(f"acc {em['acc']:.3f}")
        if "margin" in em:
            parts.append(f"margin {em['margin']:.3f}")
        if "kl" in em:
            parts.append(f"kl {em['kl']:.3f}")
        if "nll" in em:
            parts.append(f"nll {em['nll']:.3f}")
        if "log_odds" in em:
            parts.append(f"log-odds {em['log_odds']:+.3f}")
        if "reward_d" in em and "reward_u" in em:
            parts.append(f"reward D {em['reward_d']:.3f} / U {em['reward_u']:.3f}")
        return " | ".join(parts)

    bgen = batches()
    if resume_step:
        # Fast-forward the data stream to where the original process stopped
        # (see the SFT trainer; doc/bug_resume_data_order.md). Deterministic
        # from Random(0), so this reproduces the exact order an uninterrupted
        # run would have used.
        for _ in range(resume_step * args.grad_accum):
            next(bgen)
        print(f" -- data stream fast-forwarded {resume_step * args.grad_accum} "
              f"micro-batches to continue the run's batch order at step "
              f"{resume_step + 1}")
    opt.zero_grad(set_to_none=True)
    ema = resume_state["ema"] if resume_state else None
    step = resume_step
    best_val = resume_state["best_val"] if resume_state else float("inf")
    best_val_step = resume_state["best_val_step"] if resume_state else 0
    start_loss = end_loss = None
    start_val = None
    start_sft_val = last_sft_val = None
    best_sft_val, best_sft_val_step = float("inf"), 0
    last_sft_eval_step = -1
    last_eval_step, last_val = -1, None
    tok_seen, tot_seen = 0, 0
    run_started = datetime.datetime.now().isoformat(timespec="seconds")
    meter = ThroughputMeter()

    # Baseline eval: with default/pissa/eva inits the policy == reference at
    # step 0, so DPO sits at -log sigmoid(0) = 0.693 and KTO at ~0.5 (weighted);
    # a different number here usually means a data/config problem (or qerr).
    # SimPO has NO reference and therefore no fixed anchor -- its step-0 loss
    # depends on the data's length-normalized logp margins.
    # Local run report -- same contract as the SFT trainer: metrics stream to
    # metrics.jsonl as they are logged, _log_failure renders whatever it got if
    # the run dies, and finish() is idempotent.
    run_config = dict(_FAIL_CTX["record"])
    run_config.update(
        method=method, steps_planned=args.steps, warmup_steps=warmup_steps,
        targets=" ".join(net.target_modules),
        trainable_params=net.num_trainable(), n_train=len(examples),
        n_val=len(val_examples), n_sft_val=len(sft_val_examples))
    run_name = args.run_name or os.path.basename(os.path.normpath(args.out))
    report = None
    if args.out and not args.no_report:
        report = RunLogger(args.out, run_name, config=run_config)
        _REPORT["rep"] = report

    # Live monitor (--live-report): charts only. batch_fn is None because the
    # step dataset browser would need this trainer's pair-shaped examples
    # decoded and its batch order replayed; start_live_monitor serves /batch
    # as 404 in that case and the rest of the page works unchanged.
    if args.live_report and report is None:
        print(" -- --live-report needs the local report; ignoring (--no-report set)")
    elif args.live_report:
        start_live_monitor(
            args.out, port=args.live_report_port,
            live_info={"total_steps": args.steps, "first_step": resume_step + 1,
                       "run_name": run_name})

    _FAIL_CTX["phase"] = "baseline_eval"
    if val_examples:
        start_val, em = evaluate()
        if start_val is not None:
            print(f"    [eval] step 0 (baseline): {fmt_eval(start_val, em)}")
    if sft_val_examples:
        _FAIL_CTX["phase"] = "baseline_sft_eval"
        start_sft_val = last_sft_val = sft_eval_loss()
        if start_sft_val is not None:
            print(f"    [eval] step 0 (baseline): sft-held-out "
                  f"{start_sft_val:.4f}")
    if report is not None:
        base_row = {k: v for k, v in (("eval/pref_loss", start_val),
                                      ("eval/sft_held_out", start_sft_val))
                    if v is not None}
        if base_row:
            report.log(base_row, step=0)

    t0 = time.time()
    if torch.cuda.is_available():
        for d in active_devices:
            torch.cuda.reset_peak_memory_stats(d)

    def peak_vram_gb():
        if not torch.cuda.is_available():
            return 0.0
        return max((torch.cuda.max_memory_allocated(d) / 1e9 for d in active_devices),
                   default=0.0)

    def _sft_eval_note(base):
        """Append the SFT-style held-out numbers to the CSV notes cell."""
        if not sft_val_examples:
            return base
        bits = [f"sft_eval={os.path.basename(args.sft_eval_dataset)}",
                f"n={len(sft_val_examples)}"]
        if start_sft_val is not None:
            bits.append(f"start={start_sft_val:.4f}")
        if best_sft_val != float("inf"):
            bits.append(f"best={best_sft_val:.4f}@{best_sft_val_step}")
        if last_sft_val is not None:
            bits.append(f"final={last_sft_val:.4f}")
        note = " ".join(bits)
        return f"{base}; {note}" if base else note

    def log_run(status, dt, final_val):
        _FAIL_CTX["logged"] = True
        rnd = lambda x, n=6: round(x, n) if isinstance(x, (int, float)) else ""
        append_run_log(args.run_log, {
            "timestamp": run_started, "arm": f"exl3-{method}", "status": status,
            "model": args.model, "arch": getattr(config, "architecture", ""),
            "out": args.out, "dataset": args.dataset,
            "eval_split": args.eval_split or "",
            "eval_dataset": args.eval_dataset or "",
            "r": args.r, "alpha": args.alpha,
            "use_rslora": int(bool(args.use_rslora)), "init_lora": args.init_lora,
            "lr": args.lr, "scheduler": args.scheduler,
            "warmup_steps": warmup_steps, "weight_decay": args.weight_decay,
            "batch": args.batch, "grad_accum": args.grad_accum, "world_size": 1,
            "eff_batch": args.batch * args.grad_accum, "epochs": args.epochs,
            "steps_planned": args.steps, "steps_done": step, "seq_len": args.seq_len,
            "targets": " ".join(net.target_modules),
            "compute_dtype": args.compute_dtype, "attn_impl": args.attn_impl,
            "parallel": args.parallel, "shuffle": int(bool(args.shuffle)),
            "max_samples": args.max_samples, "prompt_format": args.prompt_format,
            "trainable_params": net.num_trainable(), "n_train": len(examples),
            "n_val": len(val_examples),
            "start_loss": rnd(start_loss), "end_loss": rnd(end_loss),
            "best_val": rnd(best_val) if best_val != float("inf") else "",
            "best_val_step": best_val_step or "",
            "start_val": rnd(start_val), "final_val": rnd(final_val),
            "total_s": rnd(dt, 1), "s_per_step": rnd(dt / step, 4) if step else "",
            "sup_tok_s": round(tok_seen / dt) if dt else "",
            "tot_tok_s": round(tot_seen / dt) if dt else "",
            "peak_vram_gb": rnd(peak_vram_gb(), 3),
            "t_data_s": rnd(timer.total["data"], 1),
            "t_fwd_s": rnd(timer.total["fwd"], 1),
            "t_bwd_s": rnd(timer.total["bwd"], 1),
            "t_opt_s": rnd(timer.total["opt"], 1),
            # The SFT-style eval rides in `notes` rather than as new columns:
            # qlora_runs.csv has a fixed header shared with the SFT/EBFT
            # trainers and widening it would desync older rows.
            "phase": "", "error": "",
            "notes": _sft_eval_note(_hparam_note(args)),
        })

    timer = StepTimer(devices=active_devices if torch.cuda.is_available() else None)

    # Last COMPLETED optimizer step (see qlora_train_native.py): `step` holds the
    # in-flight iteration mid-loop, and saving that on Ctrl-C desyncs the step
    # counter from the scheduler, skipping an iteration's batches on resume.
    done_step = resume_step
    try:
        for step in range(resume_step + 1, args.steps + 1):
            _FAIL_CTX["phase"] = f"train step {step}"
            step_t0 = time.time()
            accum_loss = 0.0
            step_sup = step_tot = 0
            step_m = {}
            timer.begin_step()
            # Example-weighted grad accumulation: every example contributes one
            # loss term (per-sequence, unlike SFT's per-token CE), so weight each
            # micro-batch by its example share of the window.
            window = [next(bgen) for _ in range(args.grad_accum)]
            total_n = max(sum(len(b) for b in window), 1)
            timer.mark("data")
            for batch in window:
                loss, m = batch_metrics(net, batch, pad_id, args)
                loss_val = loss.item()
                timer.mark("fwd")
                w_i = m["n"] / total_n
                with _backbone.backward_dequant_cache(
                        enable=args.dequant_cache and not args.no_grad_ckpt):
                    (loss * w_i).backward()
                timer.mark("bwd")
                accum_loss += loss_val * w_i
                step_sup += m["sup"]
                step_tot += m["tot"]
                for k in ("margin", "acc", "kl", "reward_d", "reward_u", "nll",
                          "log_odds"):
                    if k in m and m[k] == m[k]:
                        s, n = step_m.get(k, (0.0, 0))
                        step_m[k] = (s + m[k] * m["n"], n + m["n"])

            gnorm = torch.nn.utils.clip_grad_norm_(
                net.trainable_parameters(), args.max_grad_norm or float("inf")
            ).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            timer.mark("opt")
            timer.end_step()

            tok_seen += step_sup
            tot_seen += step_tot
            meter.update(time.time() - step_t0, step_sup, step_tot)
            _, tot_tps = meter.rates()

            if start_loss is None:
                start_loss = accum_loss
            end_loss = accum_loss
            ema = accum_loss if ema is None else 0.9 * ema + 0.1 * accum_loss
            mm = {k: s / n for k, (s, n) in step_m.items()}
            if method in ("dpo", "simpo", "orpo"):
                extras = (f"acc {mm.get('acc', 0):.3f} | "
                          f"margin {mm.get('margin', 0):+.3f}")
                if "nll" in mm:
                    extras += f" | nll {mm['nll']:.3f}"
                if "log_odds" in mm:
                    extras += f" | log-odds {mm['log_odds']:+.3f}"
            else:
                extras = (f"kl {mm.get('kl', 0):.3f} | "
                          f"rD {mm.get('reward_d', float('nan')):+.3f} "
                          f"rU {mm.get('reward_u', float('nan')):+.3f}")
            b_norm = adapter_b_norm()
            cur_lr = sched.get_last_lr()[0]
            print(f"  step {step:>5}/{args.steps} | loss {accum_loss:6.4f} | "
                  f"ema {ema:6.4f} | {extras} | grad {gnorm:7.4f} | "
                  f"lr {cur_lr:.2e} | "
                  f"|dB| {b_norm:7.3f} | {tot_tps:,.0f} tok/s | "
                  f"{timer.step_line()}")
            if report is not None:
                row = {"train/loss": accum_loss, "train/ema": ema,
                       "train/grad_norm": gnorm, "train/lr": cur_lr,
                       "train/adapter_b_norm": b_norm,
                       "train/tok_per_s": tot_tps,
                       "train/epoch": step * args.batch * args.grad_accum
                       / max(len(examples), 1)}
                for k, v in mm.items():
                    row[f"train/{k}"] = v
                report.log(row, step=step)

            _FAIL_CTX["record"].update(
                steps_done=step, end_loss=round(accum_loss, 6),
                peak_vram_gb=round(peak_vram_gb(), 3))
            if start_loss is not None and "start_loss" not in _FAIL_CTX["record"]:
                _FAIL_CTX["record"]["start_loss"] = round(start_loss, 6)

            # Eval schedule. The preference eval and the SFT-style eval are
            # independent: either can be configured alone, so this fires on
            # --eval-every whenever at least one eval set exists.
            if args.eval_every and step % args.eval_every == 0 and (
                    val_examples or sft_val_examples):
                eval_row = {}
                if val_examples:
                    _FAIL_CTX["phase"] = f"eval step {step}"
                    vl, em = evaluate()
                    last_eval_step, last_val = step, vl
                    if vl is not None:
                        print(f"    [eval] step {step}: {fmt_eval(vl, em)}")
                        eval_row["eval/pref_loss"] = vl
                        for k, v in em.items():
                            eval_row[f"eval/{k}"] = v
                        if vl < best_val:
                            best_val = vl
                            best_val_step = step
                            if args.save_best:
                                save(f"[best step {step}, val {vl:.4f}]")
                if sft_val_examples:
                    _FAIL_CTX["phase"] = f"sft eval step {step}"
                    sv = sft_eval_loss()
                    last_sft_val = sv
                    last_sft_eval_step = step
                    if sv is not None:
                        best_tag = ""
                        if sv < best_sft_val:
                            best_sft_val, best_sft_val_step = sv, step
                            best_tag = "  (best)"
                        print(f"    [eval] step {step}: sft-held-out "
                              f"{sv:.4f}{best_tag}")
                        eval_row["eval/sft_held_out"] = sv
                if report is not None and eval_row:
                    report.log(eval_row, step=step)

            if args.save_every and step % args.save_every == 0:
                save(f"[checkpoint step {step}]")

            if args.checkpoint_every and step % args.checkpoint_every == 0:
                cdir = checkpoint_dir(args.out, step)
                net.save_adapter(cdir, base_model_name_or_path=args.model)
                save_trainer_state(cdir, step=step, opt=opt, sched=sched,
                                   best_val=best_val, best_val_step=best_val_step,
                                   ema=ema)
                print(f"  [checkpoint] step {step} -> {cdir} (resumable)")
                prune_checkpoints(args.out, args.keep_checkpoints)

            # Reached only if the whole iteration ran: this step is now durable.
            done_step = step
    except KeyboardInterrupt:
        # Roll back to the last completed step so the saved counter, the
        # scheduler and the data stream all agree.
        step = done_step
        if args.save_best and val_examples:
            print(f"\nInterrupted at step {step}; keeping best-val adapter.")
        else:
            print(f"\nInterrupted at step {step}; saving adapter before exit.")
            if step > 0:
                save("[interrupted]")
        log_run("interrupted", time.time() - t0, None)
        if report is not None:
            report.update_summary({
                "steps_done": step,
                "best_sft_held_out": (best_sft_val
                                      if best_sft_val != float("inf") else None),
                "best_sft_held_out_step": best_sft_val_step or None,
                "last_sft_held_out": last_sft_val})
        _finish_report(exit_code=0, status="interrupted")
        raise SystemExit(0)

    dt = time.time() - t0
    _FAIL_CTX["phase"] = "final_eval"
    if last_eval_step == step:
        val_loss = last_val
    elif val_examples:
        print(" -- computing final held-out eval (GPU busy, not hung) ...")
        val_loss, em = evaluate()
        if val_loss is not None:
            print(f"    [eval] final: {fmt_eval(val_loss, em)}")
    else:
        val_loss = None
    # Final SFT-style eval, unless this step already ran one.
    if sft_val_examples and last_sft_eval_step != step:
        print(" -- computing final SFT-style held-out eval "
              "(GPU busy, not hung) ...")
        sv = sft_eval_loss()
        if sv is not None:
            last_sft_val = sv
            if sv < best_sft_val:
                best_sft_val, best_sft_val_step = sv, step
            print(f"    [eval] final: sft-held-out {sv:.4f}")
            if report is not None:
                report.log({"eval/sft_held_out": sv}, step=step)
    if not (args.save_best and val_examples):
        save("Done.")
    if torch.cuda.is_available():
        peak_str = " / ".join(
            f"cuda:{d} {torch.cuda.max_memory_allocated(d) / 1e9:.2f}GB"
            for d in active_devices)
    else:
        peak_str = "n/a"
    print(f"[PERF] {tok_seen / dt if dt else 0:,.0f} sup tok/s, "
          f"{tot_seen / dt if dt else 0:,.0f} tot tok/s (policy fwd only) | "
          f"peak VRAM {peak_str} | {dt:.0f}s for {step} steps | "
          f"step time: {timer.summary()}")
    log_run("completed", dt, val_loss)
    if report is not None:
        report.update_summary({
            "steps_done": step, "start_loss": start_loss, "end_loss": end_loss,
            "best_pref_val": best_val if best_val != float("inf") else None,
            "best_pref_val_step": best_val_step or None,
            "final_pref_val": val_loss,
            "start_sft_held_out": start_sft_val,
            "best_sft_held_out": (best_sft_val
                                  if best_sft_val != float("inf") else None),
            "best_sft_held_out_step": best_sft_val_step or None,
            "final_sft_held_out": last_sft_val,
            "peak_vram_gb": round(peak_vram_gb(), 3),
            "total_s": round(dt, 1)})
    _finish_report(exit_code=0)
    print("Verify with: python training/qlora_infer_native.py "
          f"--model {args.model} --adapter {args.out}")


def _hparam_note(args):
    """Preference hyperparameters for the run-log 'notes' column (the CSV
    schema is shared with the SFT arms; no schema roll for method-specific
    knobs)."""
    if args.method == "dpo":
        return (f"method=dpo beta={args.beta} loss={args.dpo_loss} "
                f"rpo_alpha={args.rpo_alpha} "
                f"label_smoothing={args.label_smoothing}")
    if args.method == "simpo":
        return (f"method=simpo beta={args.beta} gamma={args.gamma} "
                f"sft_weight={args.sft_weight} "
                f"label_smoothing={args.label_smoothing}")
    if args.method == "orpo":
        return f"method=orpo beta={args.beta} nll={args.orpo_nll}"
    return (f"method=kto beta={args.beta} loss={args.kto_loss} "
            f"wD={args.desirable_weight} wU={args.undesirable_weight}")


if __name__ == "__main__":
    main()
