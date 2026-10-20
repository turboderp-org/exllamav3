"""
GRPO (Group Relative Policy Optimization) of an EXL3 model, native path --
no HuggingFace Transformers in the loop.

Trains LoRA adapters on a frozen EXL3 base with on-policy reinforcement
learning from a REWARD FUNCTION (DeepSeekMath, Shao et al. 2024,
arXiv:2402.03300), using the same transformers-free differentiable forward as
qlora_train_native.py. Loss semantics follow HuggingFace TRL's
``GRPOTrainer`` (see ``exllamav3/training/grpo.py``), so hyperparameters
and community reward functions transfer directly.

One optimizer step, per generation batch of --prompts-per-step prompts:
  1. ROLLOUT: sample --num-generations completions per prompt from the
     CURRENT policy. Default: the native KV-cached generator with the
     adapter pushed into the runtime LoRA slots (``apply_to_native``, the
     realtime trick) -- a paged prefill shared by a prompt's G siblings plus
     single-token decode steps. ``--rollout-sampler exact`` instead re-runs
     the differentiable forward per token (slow, no KV cache, but the exact
     policy the loss sees).
  2. REWARD: decode the completions and call every --reward function with
     TRL's signature ``f(prompts=, completions=, completion_ids=, **columns)``
     (every other dataset column is passed as a per-row list, e.g.
     ``answer=[...]``). Built-ins: exact_answer, think_format, length_target;
     or ``path/to/file.py:func`` / ``package.module:func``. Weighted sum
     (--reward-weights) -> group-relative advantages (--scale-rewards).
  3. POLICY: per-token logps of the sampled completions on the
     differentiable forward (one micro-batch of --batch rows at a time),
     the clipped surrogate (--epsilon / --epsilon-high, --loss-type), plus
     --beta * k3 KL against the REFERENCE = the frozen base, obtained by
     running the same net under ``adapters_disabled()`` (no second model
     copy; --beta 0, TRL's default, skips that forward entirely). The OLD
     policy of the ratio is the detached current logps (one optimizer step
     per generation batch) or, with --num-iterations > 1, the logps captured
     right after rollout (the batch is then reused for mu optimizer steps).

No reference model, old-policy model or reward model is ever loaded: the
reward is a function (which may itself call a judge model if you write one).

Usage (verifiable-reward demo on a GSM8K-style set with an ``answer`` column):
    python training/qlora_train_grpo.py \
        --model /path/to/exl3_model --out out/exl3_grpo_adapter \
        --dataset /data/gsm8k_train.jsonl --prompt-key question \
        --reward exact_answer --num-generations 8 --prompts-per-step 4 \
        --max-completion-length 512 --lr 1e-5 --steps 300

Notes vs the SFT trainer:
  * Single-process only (--parallel single|split); frozen LM head only (the
    logps path), no --train-head/--train-embeddings/packing.
  * The paper's full-FT LR is 1e-6; LoRA wants more (1e-5 .. 5e-5 is the
    usual range). Keep --temperature 1.0 (TRL's default): a colder sampler
    collapses the groups and most prompts give zero advantage.
  * --rollout-sampler native is refused for routed-expert LoRA (the fused
    MoE inference kernels bypass the runtime LoRA slots, so rollouts would
    come from the base experts); use --rollout-sampler exact or non-expert
    --targets.
  * Sampler/trainer numerics: the native generator's fused kernels and the
    training forward agree to inference-kernel rounding, not bit-exactly;
    the ratio uses training-forward logps on both sides (TRL's rule under
    vLLM), so this only makes the samples very slightly off-policy.

Verify afterwards on the native inference path:
    python training/qlora_infer_native.py --model <model> --adapter <out>
"""

import argparse
import datetime
import os
import random
import sys
import time
import torch

# Reuse the SFT / preference trainers' helpers (same dir on sys.path when run
# as a script).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qlora_train_native import (  # noqa: E402
    ThroughputMeter, StepTimer, append_run_log,
    checkpoint_dir, prune_checkpoints,
    save_trainer_state, load_trainer_state, restore_optimizer_state,
    snapshot_datasets, resolve_resumed_datasets,
    build_optimizer, make_lr_scheduler, resolve_steps_and_warmup,
    compute_even_split_budgets,
    _FAIL_CTX, _log_failure, _REPORT, _finish_report,
)
from qlora_train_pref import (  # noqa: E402
    _load_rows, _prompt_renderers, _prompt_turns, _encode_prompt_ids,
    rows_to_batch,
)
from run_report import RunLogger  # noqa: E402

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer  # noqa: E402
from exllamav3.generator.sampler import ComboSampler  # noqa: E402
from exllamav3.training.native_llama import NativeLlamaQLoRA  # noqa: E402
from exllamav3.training.grpo import (  # noqa: E402
    GRPO_LOSS_TYPES, SCALE_REWARDS, IS_LEVELS,
    group_advantages, combine_rewards, grpo_loss,
    sample_completions_exact, load_reward_func, call_reward_funcs,
)
from exllamav3.training.rewards import BUILTIN_REWARDS  # noqa: E402


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_prompt_examples(model, tokenizer, dataset_name, split, prompt_key,
                          max_prompt_length, max_samples=0, shuffle=False,
                          shuffle_seed=0, prompt_format="auto", config_name=None,
                          chat_template_file=None, template_vars=None):
    """Prompt-only dataset (TRL GRPO format: a ``prompt`` column that is a
    string or a conversational message list; every OTHER column is kept and
    handed to the reward functions). Each example: ``{prompt_ids, prompt,
    extras}`` with the prompt rendered through the chat template and ending
    at the assistant-turn opener. Returns ``(examples, eot_text)``."""
    ds = _load_rows(dataset_name, split, config_name)
    if shuffle or (max_samples and max_samples < len(ds)):
        ds = ds.shuffle(seed=shuffle_seed)
    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))
    seg_builder, build_prompt, eot, row_extras = _prompt_renderers(
        model, tokenizer, prompt_format, chat_template_file, template_vars)
    examples, skipped, too_long = [], 0, 0
    for row in ds:
        turns = _prompt_turns(row.get(prompt_key), rich=row_extras is not None)
        if not turns:
            skipped += 1
            continue
        ids = _encode_prompt_ids(tokenizer, turns, build_prompt, seg_builder,
                                 extras=row_extras(row) if row_extras else None)
        if ids is None:
            skipped += 1
            continue
        if len(ids) > max_prompt_length:
            too_long += 1
            continue
        examples.append({
            "prompt_ids": ids,
            "prompt": row.get(prompt_key),
            "extras": {k: v for k, v in row.items() if k != prompt_key},
        })
    if skipped or too_long:
        print(f" -- skipped {skipped} rows (no usable prompt) and {too_long} "
              f"over --max-prompt-length {max_prompt_length}")
    return examples, eot


def stop_token_ids(tokenizer, eot_text):
    """EOS plus the chat format's turn-end token (when it is one token and
    differs from EOS): the sampled completion ends at the first of these."""
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    if eot_text:
        enc = tokenizer.encode(eot_text, add_bos=False,
                               encode_special_tokens=True)[0].tolist()
        if len(enc) == 1:
            ids.add(int(enc[0]))
    return sorted(ids)


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

def sample_completions_native(gen, prompt_rows, max_new_tokens, args, stop_ids,
                              seed):
    """KV-cached on-policy completions through the native inference path.
    The caller has pushed the current adapter into the runtime LoRA slots
    (``net.apply_to_native()``). Returns ``[(completion_ids, truncated)]``
    per prompt row, the stop token included as the completion's last token
    when one ended it (so the policy learns to stop)."""
    assert args.temperature > 0.0, "GRPO needs temperature > 0"
    sampler = ComboSampler(temperature=args.temperature, top_k=args.top_k,
                           top_p=args.top_p)
    job_idx = {}
    for i, ids in enumerate(prompt_rows):
        # Job counts the prefill-position sample toward its budget (N yields
        # N-1 sampled tokens), so ask for max_new_tokens + 1.
        job = Job(input_ids=torch.tensor([list(ids)], dtype=torch.long),
                  max_new_tokens=max_new_tokens + 1, sampler=sampler,
                  seed=seed + i, stop_conditions=list(stop_ids))
        job_idx[job] = i
        gen.enqueue(job)
    out = [[] for _ in prompt_rows]
    reason = [None] * len(prompt_rows)
    stop_tok = [None] * len(prompt_rows)
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            i = job_idx.get(r["job"])
            if i is None:
                continue
            for blob in (r, r.get("held") or {}):
                t = blob.get("token_ids")
                if t is not None:
                    out[i] += t.flatten().tolist()
            if r.get("eos"):
                reason[i] = r.get("eos_reason")
                stop_tok[i] = r.get("eos_triggering_token_id")
    result = []
    stop_set = set(stop_ids)
    for i, toks in enumerate(out):
        toks = toks[:max_new_tokens]
        ended = reason[i] == "stop_token"
        if ended and (not toks or toks[-1] not in stop_set):
            t = stop_tok[i] if stop_tok[i] is not None else next(iter(stop_set))
            toks = (toks + [int(t)])[-max_new_tokens:]
        result.append((toks, not ended))
    return result


def generate_and_score(net, batch, args, gen, reward_funcs, reward_weights,
                       rng, pad_id, stop_ids, tokenizer):
    """Rollouts + rewards + advantages for one generation batch of prompts.
    Returns a dict with the ``(prompt_ids, completion_ids)`` rows (the G
    completions of a prompt consecutive), advantages, per-function rewards
    and diagnostics."""
    G = args.num_generations
    prompt_rows = [ex["prompt_ids"] for ex in batch for _ in range(G)]
    seed = rng.randrange(1 << 31)
    if gen is not None:
        net.apply_to_native()
        try:
            comps = sample_completions_native(
                gen, prompt_rows, args.max_completion_length, args, stop_ids, seed)
        finally:
            net.remove_from_native()
    else:
        comps = sample_completions_exact(
            net, prompt_rows, args.max_completion_length,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            stop_ids=stop_ids, pad_id=pad_id)
    completion_ids = [c for c, _ in comps]
    truncated = [t for _, t in comps]
    texts = [tokenizer.decode(torch.tensor([c], dtype=torch.long),
                              decode_special_tokens=False)[0] if c else ""
             for c in completion_ids]
    prompts = [ex["prompt"] for ex in batch for _ in range(G)]
    keys = sorted({k for ex in batch for k in ex["extras"]})
    extras = {k: [ex["extras"].get(k) for ex in batch for _ in range(G)]
              for k in keys}
    per_func = call_reward_funcs(reward_funcs, prompts, texts, completion_ids,
                                 extras)
    rewards, all_nan = combine_rewards(per_func, reward_weights)
    if bool(all_nan.any()):
        print(f"  !! {int(all_nan.sum())} completion(s) got no reward from any "
              f"function (reward 0)")
    adv, info = group_advantages(rewards, G, scale=args.scale_rewards)
    return {
        "rows": list(zip(prompt_rows, completion_ids)),
        "adv": adv, "truncated": truncated, "texts": texts,
        "per_func": per_func, "rewards": rewards, "info": info,
        "lengths": [len(c) for c in completion_ids],
    }


# ---------------------------------------------------------------------------
# Policy pass
# ---------------------------------------------------------------------------

def _token_logps(net, rows, pad_id, args, truncated=None, reference=False):
    """Per-token completion logps for a list of rows (policy, or the frozen
    base with adapters disabled). Truncated rows are fully masked when
    --mask-truncated-completions."""
    input_ids, labels, attn = rows_to_batch(rows, pad_id)
    if truncated is not None and args.mask_truncated_completions:
        for i, t in enumerate(truncated):
            if t:
                labels[i] = -100
    if reference:
        with torch.no_grad(), net.adapters_disabled():
            return net.compute_logps(input_ids, labels, attention_mask=attn,
                                     chunk=args.ce_chunk, return_per_token=True)
    return net.compute_logps(input_ids, labels, attention_mask=attn,
                             chunk=args.ce_chunk, return_per_token=True)


def policy_pass(net, sample, args, pad_id, old_cache=None, ref_cache=None,
                backward=True):
    """One pass over a generation batch in micro-batches of --batch rows:
    forward, GRPO loss, backward (accumulated). Returns ``(loss_value,
    metrics)``; ``old_cache`` / ``ref_cache`` are per-micro-batch lists of
    ``(token_logps, mask)`` captured earlier (or None)."""
    rows, adv, trunc = sample["rows"], sample["adv"], sample["truncated"]
    R = len(rows)
    micro = [range(i, min(i + args.batch, R)) for i in range(0, R, args.batch)]
    n_micro = len(micro)
    total_tokens = sum(len(c) for (_, c), t in zip(rows, trunc)
                       if not (t and args.mask_truncated_completions))
    total = 0.0
    agg = {"kl": 0.0, "clip_frac": 0.0, "clip_low": 0.0, "clip_high": 0.0,
           "ratio_mean": 0.0}
    sup = 0
    for mi, mb in enumerate(micro):
        idx = list(mb)
        tok_logps, mask = _token_logps(net, [rows[i] for i in idx], pad_id, args,
                                       truncated=[trunc[i] for i in idx])
        maskf = mask.to(tok_logps.dtype)
        old = old_cache[mi][0] if old_cache is not None else None
        ref = ref_cache[mi][0] if ref_cache is not None else None
        loss, m = grpo_loss(
            tok_logps, maskf, adv[idx], old_token_logps=old, ref_token_logps=ref,
            beta=args.beta, epsilon=args.epsilon, epsilon_high=args.epsilon_high,
            loss_type=args.loss_type,
            importance_sampling_level=args.importance_sampling_level,
            max_completion_length=args.max_completion_length,
            total_tokens=total_tokens)
        if args.loss_type != "dapo":
            loss = loss / n_micro          # TRL: loss / gradient_accumulation_steps
        if backward:
            loss.backward()
        total += loss.item()
        w = float(maskf.sum()) / max(total_tokens, 1)
        for k in agg:
            agg[k] += m[k] * w
        sup += int(mask.sum())
    agg["sup"] = sup
    return total, agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        _run_main()
    except SystemExit:
        raise
    except BaseException as e:
        _log_failure(e)
        raise


def _run_main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    if os.environ.get("RANK") is not None or os.environ.get("WORLD_SIZE") is not None:
        raise SystemExit("qlora_train_grpo.py is single-process (--parallel single|split).")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to a local EXL3 model dir")
    ap.add_argument("--out", default="out/exl3_grpo_adapter")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--parallel", choices=["single", "split"], default="single")
    ap.add_argument("--reserve-per-device", nargs="*", type=float, default=None, metavar="GB")
    ap.add_argument("--split-even", nargs="?", type=float, const=2.0, default=None,
                    metavar="MARGIN_GB",
                    help="(split) Balance the layer split across GPUs (see the "
                         "EBFT trainer's help).")
    ap.add_argument("--use-per-device", nargs="*", type=float, default=None, metavar="GB")
    # LoRA
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--use-rslora", action="store_true")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--expert-r", type=int, default=None)
    ap.add_argument("--init-lora", choices=["default", "pissa", "qerr", "eva"],
                    default="default",
                    help="default/pissa/eva keep reference == step-0 policy "
                         "(KL starts at 0); qerr does not.")
    ap.add_argument("--init-svd-niter", type=int, default=16)
    ap.add_argument("--init-ref-model", default=None)
    ap.add_argument("--init-eva-tokens", type=int, default=65536)
    # GRPO objective
    ap.add_argument("--reward", nargs="+", action="extend", default=None,
                    metavar="SPEC",
                    help="Reward function(s) (repeatable or space-separated; "
                         "rewards are summed with "
                         "--reward-weights). A built-in name "
                         f"({', '.join(BUILTIN_REWARDS)}), path/to/file.py:func, "
                         "or package.module:func -- TRL's reward signature. "
                         "Required.")
    ap.add_argument("--reward-weights", nargs="*", type=float, default=None,
                    help="One weight per --reward (default: all 1.0).")
    ap.add_argument("--num-generations", type=int, default=8,
                    help="Completions per prompt, G (TRL default 8; paper 64).")
    ap.add_argument("--prompts-per-step", type=int, default=4,
                    help="Prompts per generation batch = per optimizer step; "
                         "rows per step = prompts_per_step * num_generations.")
    ap.add_argument("--batch", type=int, default=8,
                    help="Completion ROWS per policy micro-forward (the memory "
                         "knob; the step's rows are accumulated in chunks of "
                         "this).")
    ap.add_argument("--num-iterations", type=int, default=1,
                    help="Optimizer steps per generation batch, mu (TRL "
                         "num_iterations, default 1 = ratio 1, clipping a "
                         "no-op). > 1 captures old-policy logps after rollout.")
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--max-prompt-length", type=int, default=512,
                    help="Prompts longer than this (tokens) are dropped.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0,
                    help="KL penalty weight (TRL default 0.0 = no reference "
                         "forward at all; paper 0.04). The reference is the "
                         "frozen base via adapters_disabled().")
    ap.add_argument("--epsilon", type=float, default=0.2,
                    help="Clip range (lower bound 1 - epsilon).")
    ap.add_argument("--epsilon-high", type=float, default=None,
                    help="Upper clip bound 1 + epsilon_high (DAPO clip-higher, "
                         "e.g. 0.28); default = --epsilon.")
    ap.add_argument("--loss-type", choices=list(GRPO_LOSS_TYPES), default="dapo",
                    help="Reduction: dapo (token sum / generation-batch tokens; "
                         "TRL default), grpo (per-sequence mean then batch mean; "
                         "the paper), bnpo (micro-batch token mean), dr_grpo "
                         "(token sum / (rows * max_completion_length)).")
    ap.add_argument("--scale-rewards", choices=list(SCALE_REWARDS), default="group",
                    help="Advantage scaling: group std (paper/TRL default), "
                         "batch std, or none (Dr. GRPO).")
    ap.add_argument("--importance-sampling-level", choices=list(IS_LEVELS),
                    default="token", help="token (GRPO) or sequence (GSPO).")
    ap.add_argument("--mask-truncated-completions", action="store_true",
                    help="Exclude completions that hit --max-completion-length "
                         "from the loss (DAPO overlong filtering).")
    ap.add_argument("--rollout-sampler", choices=["native", "exact"], default="native",
                    help="native: KV-cached generator through the runtime LoRA "
                         "slots (fast; refused for routed-expert LoRA). exact: "
                         "the differentiable forward token by token (slow, "
                         "bit-exact policy).")
    ap.add_argument("--sampler-cache-tokens", type=int, default=65536,
                    help="Paged KV cache size (tokens) for the native rollout "
                         "sampler; rows that don't fit at once run in waves.")
    # Optimization
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="Paper full-FT LR 1e-6; LoRA 1e-5 .. 5e-5.")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--optim", choices=["adamw", "adamw8bit", "paged_adamw8bit"],
                    default="adamw")
    ap.add_argument("--adam-betas", default="0.9,0.99")
    ap.add_argument("--scheduler", choices=["none", "linear", "cosine"], default="none")
    ap.add_argument("--warmup-ratio", type=float, default=0.0)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    # Data
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--prompt-key", default="prompt",
                    help="Prompt column: a string or a conversational message "
                         "list. Every OTHER column is passed to the rewards.")
    ap.add_argument("--prompt-format",
                    choices=["auto", "mistral", "metharme", "gemma4-nothink",
                             "llama3", "qwen3.5", "qwen3.5-nothink", "chatml",
                             "hy3", "hy3-nothink", "jinja"],
                    default="auto",
                    help="Chat format (see the SFT trainer's help; jinja = the "
                         "model directory's own Jinja chat template).")
    ap.add_argument("--chat-template-file", default=None)
    ap.add_argument("--template-vars", default=None,
                    help="(jinja) JSON object of extra template variables.")
    ap.add_argument("--strip-sys-prompt-extras", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    # Eval / saving
    ap.add_argument("--eval-split", default=None)
    ap.add_argument("--eval-dataset", default=None)
    ap.add_argument("--eval-max-samples", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-generations", type=int, default=1,
                    help="Completions per held-out prompt at eval (mean reward "
                         "is the eval metric; higher is better).")
    ap.add_argument("--save-best", action="store_true",
                    help="Keep the checkpoint with the best held-out mean reward.")
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--keep-checkpoints", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--reset-optimizer", action="store_true")
    ap.add_argument("--run-log", default="qlora_runs.csv")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--print-samples", type=int, default=2,
                    help="Every --sample-every steps print this many of the "
                         "step's own completions with their rewards (0 = off).")
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--self-test", action="store_true",
                    help="One generation batch + rewards + a policy pass on the "
                         "first prompts, printed; then exit without training.")
    # Runtime knobs shared with the SFT trainer
    ap.add_argument("--compute-dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--attn-impl", choices=["auto", "eager", "flash"], default="auto")
    ap.add_argument("--ce-chunk", type=int, default=1024)
    ap.add_argument("--head-vocab-chunk", type=int, default=0)
    ap.add_argument("--offload-activations", action="store_true")
    ap.add_argument("--offload-mode", choices=["async", "sync"], default="async")
    ap.add_argument("--use-liger", action="store_true")
    ap.add_argument("--dequant-mode", choices=["fast", "legacy"], default="fast")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()

    from chat_jinja import set_strip_sys_prompt_extras, parse_template_vars
    set_strip_sys_prompt_extras(args.strip_sys_prompt_extras)
    from exllamav3.training import backbone as _backbone
    _backbone.set_dequant_mode(args.dequant_mode)

    if not args.reward:
        raise SystemExit("--reward is required (a built-in name, path/to/file.py:func "
                         "or package.module:func; repeatable).")
    if args.num_generations < 2:
        raise SystemExit("--num-generations must be >= 2 (group-relative advantages).")
    if args.num_iterations < 1:
        raise SystemExit("--num-iterations must be >= 1")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0")
    reward_funcs = [load_reward_func(s) for s in args.reward]
    if args.reward_weights is not None and len(args.reward_weights) != len(reward_funcs):
        raise SystemExit(f"--reward-weights needs {len(reward_funcs)} values")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    if args.out:
        data_roles = {"dataset": args.dataset, "eval_dataset": args.eval_dataset}
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
        "arm": "exl3-grpo", "model": args.model, "out": args.out,
        "dataset": args.dataset, "eval_split": args.eval_split or "",
        "eval_dataset": args.eval_dataset or "",
        "r": args.r, "alpha": args.alpha,
        "use_rslora": int(bool(args.use_rslora)), "init_lora": args.init_lora,
        "lr": args.lr, "scheduler": args.scheduler,
        "weight_decay": args.weight_decay,
        "batch": args.batch, "grad_accum": 1, "world_size": 1,
        "eff_batch": args.prompts_per_step * args.num_generations,
        "epochs": args.epochs, "steps_planned": args.steps,
        "seq_len": args.max_prompt_length + args.max_completion_length,
        "compute_dtype": args.compute_dtype, "attn_impl": args.attn_impl,
        "parallel": args.parallel, "shuffle": int(bool(args.shuffle)),
        "max_samples": args.max_samples, "prompt_format": args.prompt_format,
        "notes": _hparam_note(args),
    }

    cdt = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16}[args.compute_dtype]

    # 1. Model + tokenizer (+ the rollout KV cache, allocated before load()).
    _FAIL_CTX["phase"] = "load_model"
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    sampler_cache = None
    if args.rollout_sampler == "native":
        sampler_cache = Cache(model, max_num_tokens=args.sampler_cache_tokens)
    if args.split_even is not None and args.parallel != "split":
        raise SystemExit("--split-even only applies to --parallel split.")
    if args.parallel == "split":
        load_kwargs = {}
        if args.split_even is not None:
            if args.reserve_per_device is not None or args.use_per_device is not None:
                raise SystemExit("--split-even computes its own per-device budgets; "
                                 "drop --reserve-per-device/--use-per-device.")
            load_kwargs["use_per_device"] = compute_even_split_budgets(
                model, config, args.split_even)
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

    # 2. Differentiable QLoRA net.
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
        if args.init_lora == "qerr" and args.beta:
            print(" -- note: with qerr the step-0 policy is the error-repaired "
                  "model but the KL reference is the raw quantized base.")
    if args.resume:
        net.load_adapter(args.resume)
    print(f" -- trainable params: {net.num_trainable():,} "
          f"(r={args.r}, alpha={args.alpha}, targets={net.target_modules})")
    print(f" -- {net.describe_attn()}")
    print(f" -- GRPO: G={args.num_generations} prompts/step={args.prompts_per_step} "
          f"(rows/step {args.prompts_per_step * args.num_generations}, micro "
          f"{args.batch}) mu={args.num_iterations} | beta {args.beta} eps "
          f"{args.epsilon}/{args.epsilon_high if args.epsilon_high is not None else args.epsilon} "
          f"| loss {args.loss_type} scale {args.scale_rewards} IS "
          f"{args.importance_sampling_level} | temp {args.temperature} "
          f"max_new {args.max_completion_length} | rewards "
          f"{', '.join(n for _, n in reward_funcs)}")

    is_moe_lora = any(w.r > 0 and ".experts." in w.key for w in net._wrappers)
    gen = None
    if args.rollout_sampler == "native":
        if is_moe_lora:
            raise SystemExit(
                "--rollout-sampler native is off-policy for routed-expert LoRA "
                "(fused MoE kernels bypass the runtime LoRA slots). Use "
                "--rollout-sampler exact or non-expert --targets.")
        gen = Generator(model=model, cache=sampler_cache, tokenizer=tokenizer)
        print(f" -- rollout sampler: native generator ({args.sampler_cache_tokens} "
              f"cache tokens); scoring on the differentiable path")
    else:
        print(" -- rollout sampler: exact (differentiable forward, no KV cache)")

    # 3. Data.
    _FAIL_CTX["phase"] = "build_dataset"
    template_vars = parse_template_vars(args.template_vars)
    data_kw = dict(prompt_key=args.prompt_key,
                   max_prompt_length=args.max_prompt_length,
                   prompt_format=args.prompt_format,
                   config_name=args.dataset_config,
                   chat_template_file=args.chat_template_file,
                   template_vars=template_vars)
    examples, eot = build_prompt_examples(
        model, tokenizer, args.dataset, args.dataset_split,
        max_samples=args.max_samples, shuffle=args.shuffle,
        shuffle_seed=args.shuffle_seed, **data_kw)
    print(f" -- {len(examples)} prompts{' (shuffled)' if args.shuffle else ''}")
    assert examples, "no usable prompts"
    stop_ids = stop_token_ids(tokenizer, eot)
    print(f" -- stop tokens: {stop_ids} (eos + turn end {eot!r})")

    val_examples = []
    if args.eval_split:
        val_examples, _ = build_prompt_examples(
            model, tokenizer, args.eval_dataset or args.dataset, args.eval_split,
            max_samples=args.eval_max_samples, **data_kw)
        print(f" -- held-out eval: {len(val_examples)} prompts from '{args.eval_split}'")
    elif args.val_frac > 0:
        n_val = max(1, int(len(examples) * args.val_frac))
        val_examples, examples = examples[:n_val], examples[n_val:]
        print(f" -- held out {len(val_examples)} val prompts; {len(examples)} for training")
        assert examples, "val_frac too large; no training prompts left"

    if args.init_lora == "eva" and not args.resume:
        _FAIL_CTX["phase"] = "init_lora"

        def eva_prepass():
            used, i = 0, 0
            while used < args.init_eva_tokens and i < len(examples):
                b = examples[i:i + args.batch]
                i += len(b)
                ids, labels, attn = rows_to_batch(
                    [(ex["prompt_ids"], []) for ex in b], pad_id)
                used += int(attn.sum())
                yield dict(input_ids=ids, attention_mask=attn)

        net.apply_init_lora("eva", svd_niter=args.init_svd_niter,
                            eva_batches=eva_prepass())
    elif args.init_lora == "eva":
        print(" -- eva init skipped on --resume")

    def show_samples(sample, n, tag):
        G = args.num_generations
        shown = 0
        for i in range(0, len(sample["texts"]), G):
            if shown >= n:
                break
            j = i + int(torch.argmax(sample["rewards"][i:i + G]))
            txt = sample["texts"][j].strip().replace("\n", "\\n")
            if len(txt) > 400:
                txt = txt[:400] + " ..."
            print(f"  \U0001f3b2 {tag} prompt {i // G}: best reward "
                  f"{float(sample['rewards'][j]):+.3f} (group mean "
                  f"{float(sample['rewards'][i:i + G].mean()):+.3f}, "
                  f"{sample['lengths'][j]} tok"
                  f"{', truncated' if sample['truncated'][j] else ''})\n     -> {txt}")
            shown += 1

    if args.self_test:
        _FAIL_CTX["phase"] = "self_test"
        batch = examples[:args.prompts_per_step]
        t0 = time.time()
        sample = generate_and_score(net, batch, args, gen, reward_funcs,
                                    args.reward_weights, rng, pad_id, stop_ids,
                                    tokenizer)
        print(f" -- self-test: {len(sample['rows'])} rollouts in {time.time() - t0:.1f}s; "
              f"reward mean {sample['info']['reward_mean']:+.3f} std "
              f"{sample['info']['reward_std']:.3f}, zero-std groups "
              f"{sample['info']['frac_zero_std']:.2f}, mean len "
              f"{sum(sample['lengths']) / len(sample['lengths']):.0f}, truncated "
              f"{sum(sample['truncated']) / len(sample['truncated']):.2f}")
        for j, name in enumerate(n for _, n in reward_funcs):
            print(f"    reward {name}: mean {float(torch.nanmean(sample['per_func'][:, j])):+.3f}")
        show_samples(sample, min(len(batch), 3), "self-test")
        ref_cache = None
        if args.beta:
            ref_cache = [_token_logps(net, [sample["rows"][i] for i in mb], pad_id, args,
                                      truncated=[sample["truncated"][i] for i in mb],
                                      reference=True)
                         for mb in [range(i, min(i + args.batch, len(sample["rows"])))
                                    for i in range(0, len(sample["rows"]), args.batch)]]
        loss, m = policy_pass(net, sample, args, pad_id, ref_cache=ref_cache)
        gn = torch.nn.utils.clip_grad_norm_(net.trainable_parameters(), float("inf")).item()
        print(f" -- self-test policy pass: loss {loss:+.5f} kl {m['kl']:.5f} clip "
              f"{m['clip_frac']:.3f} ratio {m['ratio_mean']:.4f} grad-norm {gn:.4f}")
        assert torch.isfinite(torch.tensor(loss)) and torch.isfinite(torch.tensor(gn))
        opt_probe = sum(p.grad.abs().sum().item() for p in net.trainable_parameters()
                        if p.grad is not None)
        print(f" -- self-test: gradient reached the adapter (|grad|_1 = {opt_probe:.3e}); "
              "exiting (drop --self-test to train).")
        return

    args.steps, warmup_steps = resolve_steps_and_warmup(
        args, len(examples), args.prompts_per_step)
    print(f" -- {args.steps} steps, scheduler={args.scheduler}, "
          f"warmup={warmup_steps}, weight_decay={args.weight_decay}")

    # 4. Optimizer + schedule.
    betas = tuple(float(b) for b in args.adam_betas.split(","))
    opt = build_optimizer(net.param_groups(args.weight_decay), args.lr, args.optim)
    for g in opt.param_groups:
        g["betas"] = betas
    sched = make_lr_scheduler(opt, args.scheduler, args.steps * args.num_iterations,
                              warmup_steps * args.num_iterations)
    resume_step, resume_state = 0, None
    if args.resume and not args.reset_optimizer:
        resume_state = load_trainer_state(args.resume)
        if resume_state is not None:
            restore_optimizer_state(opt, resume_state["optimizer"])
            if resume_state.get("scheduler") is not None:
                sched.load_state_dict(resume_state["scheduler"])
            resume_step = int(resume_state["step"])
            print(f" -- resumed trainer state: continuing at step "
                  f"{resume_step + 1}/{args.steps}")

    def batches():
        order = list(range(len(examples)))
        while True:
            random.Random(args.shuffle_seed).shuffle(order)
            for i in range(0, len(order) - args.prompts_per_step + 1,
                           args.prompts_per_step):
                yield [examples[j] for j in order[i:i + args.prompts_per_step]]

    def adapter_b_norm():
        with torch.no_grad():
            return sum(w.lora_b.float().pow(2).sum().item()
                       for w in net._wrappers if w.r > 0) ** 0.5

    def save(tag):
        net.save_adapter(args.out, base_model_name_or_path=args.model)
        save_trainer_state(args.out, step=step, opt=opt, sched=sched,
                           best_val=best_val, best_val_step=best_val_step, ema=ema)
        print(f"{tag} Adapter written to {args.out}")

    def evaluate():
        """Mean reward over the held-out prompts (--eval-generations
        completions each, same sampler). Higher is better."""
        if not val_examples:
            return None, {}
        net.eval()
        saved_g = args.num_generations
        args.num_generations = max(1, args.eval_generations)
        tot, n, lens, trunc = 0.0, 0, 0, 0
        per = None
        try:
            with torch.no_grad():
                for i in range(0, len(val_examples), args.prompts_per_step):
                    vb = val_examples[i:i + args.prompts_per_step]
                    s = generate_and_score(net, vb, args, gen, reward_funcs,
                                           args.reward_weights, random.Random(args.seed + i),
                                           pad_id, stop_ids, tokenizer)
                    tot += float(s["rewards"].sum())
                    n += len(s["rewards"])
                    lens += sum(s["lengths"])
                    trunc += sum(s["truncated"])
                    pf = torch.nan_to_num(s["per_func"], nan=0.0).sum(0)
                    per = pf if per is None else per + pf
        finally:
            args.num_generations = saved_g
            net.train()
        if not n:
            return None, {}
        em = {"len": lens / n, "truncated": trunc / n}
        for j, (_, name) in enumerate(reward_funcs):
            em[f"reward_{name}"] = float(per[j]) / n
        return tot / n, em

    def fmt_eval(v, em):
        parts = [f"reward {v:+.4f}", f"len {em.get('len', 0):.0f}",
                 f"truncated {em.get('truncated', 0):.2f}"]
        parts += [f"{k[7:]} {val:+.3f}" for k, val in em.items() if k.startswith("reward_")]
        return " | ".join(parts)

    bgen = batches()
    if resume_step:
        for _ in range(resume_step):
            next(bgen)
        print(f" -- data stream fast-forwarded {resume_step} generation batches")
    opt.zero_grad(set_to_none=True)
    ema = resume_state["ema"] if resume_state else None
    step = resume_step
    best_val = resume_state["best_val"] if resume_state else float("-inf")
    if best_val is None:
        best_val = float("-inf")
    best_val_step = resume_state["best_val_step"] if resume_state else 0
    start_loss = end_loss = None
    start_val = None
    start_reward = end_reward = None
    last_eval_step, last_val = -1, None
    tok_seen, tot_seen = 0, 0
    run_started = datetime.datetime.now().isoformat(timespec="seconds")
    meter = ThroughputMeter()

    report = None
    if args.out and not args.no_report:
        run_config = dict(_FAIL_CTX["record"])
        run_config.update(warmup_steps=warmup_steps, targets=" ".join(net.target_modules),
                          trainable_params=net.num_trainable(), n_train=len(examples),
                          n_val=len(val_examples))
        report = RunLogger(args.out, args.run_name or os.path.basename(os.path.normpath(args.out)),
                           config=run_config)
        _REPORT["rep"] = report

    _FAIL_CTX["phase"] = "baseline_eval"
    if val_examples:
        start_val, em = evaluate()
        if start_val is not None:
            print(f"    [eval] step 0 (baseline): {fmt_eval(start_val, em)}")
            if report is not None:
                report.log({"eval/reward": start_val}, step=0)

    t0 = time.time()
    if torch.cuda.is_available():
        for d in active_devices:
            torch.cuda.reset_peak_memory_stats(d)

    def peak_vram_gb():
        if not torch.cuda.is_available():
            return 0.0
        return max((torch.cuda.max_memory_allocated(d) / 1e9 for d in active_devices),
                   default=0.0)

    def log_run(status, dt, final_val):
        _FAIL_CTX["logged"] = True
        rnd = lambda x, n=6: round(x, n) if isinstance(x, (int, float)) else ""
        append_run_log(args.run_log, {
            "timestamp": run_started, "arm": "exl3-grpo", "status": status,
            "model": args.model, "arch": getattr(config, "architecture", ""),
            "out": args.out, "dataset": args.dataset,
            "eval_split": args.eval_split or "",
            "eval_dataset": args.eval_dataset or "",
            "r": args.r, "alpha": args.alpha,
            "use_rslora": int(bool(args.use_rslora)), "init_lora": args.init_lora,
            "lr": args.lr, "scheduler": args.scheduler,
            "warmup_steps": warmup_steps, "weight_decay": args.weight_decay,
            "batch": args.batch, "grad_accum": 1, "world_size": 1,
            "eff_batch": args.prompts_per_step * args.num_generations,
            "epochs": args.epochs,
            "steps_planned": args.steps, "steps_done": step,
            "seq_len": args.max_prompt_length + args.max_completion_length,
            "targets": " ".join(net.target_modules),
            "compute_dtype": args.compute_dtype, "attn_impl": args.attn_impl,
            "parallel": args.parallel, "shuffle": int(bool(args.shuffle)),
            "max_samples": args.max_samples, "prompt_format": args.prompt_format,
            "trainable_params": net.num_trainable(), "n_train": len(examples),
            "n_val": len(val_examples),
            "start_loss": rnd(start_loss), "end_loss": rnd(end_loss),
            "best_val": rnd(best_val) if best_val != float("-inf") else "",
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
            "phase": "", "error": "",
            "notes": _hparam_note(args) + (f" start_reward={start_reward:.4f} "
                                           f"end_reward={end_reward:.4f}"
                                           if start_reward is not None else ""),
        })

    timer = StepTimer(devices=active_devices if torch.cuda.is_available() else None)
    done_step = resume_step
    try:
        for step in range(resume_step + 1, args.steps + 1):
            _FAIL_CTX["phase"] = f"train step {step}"
            step_t0 = time.time()
            timer.begin_step()
            batch = next(bgen)
            # (1) + (2): rollouts and rewards, no grad.
            net.eval()
            with torch.no_grad():
                sample = generate_and_score(net, batch, args, gen, reward_funcs,
                                            args.reward_weights, rng, pad_id,
                                            stop_ids, tokenizer)
            net.train()
            micro_ranges = [range(i, min(i + args.batch, len(sample["rows"])))
                            for i in range(0, len(sample["rows"]), args.batch)]
            ref_cache = old_cache = None
            if args.beta:
                ref_cache = [_token_logps(net, [sample["rows"][i] for i in mb], pad_id,
                                          args, truncated=[sample["truncated"][i] for i in mb],
                                          reference=True) for mb in micro_ranges]
            if args.num_iterations > 1:
                with torch.no_grad():
                    old_cache = [_token_logps(net, [sample["rows"][i] for i in mb], pad_id,
                                              args, truncated=[sample["truncated"][i] for i in mb])
                                 for mb in micro_ranges]
            timer.mark("data")

            # (3): mu optimizer steps over the batch.
            step_loss, mm = 0.0, {}
            for it in range(args.num_iterations):
                loss_val, m = policy_pass(net, sample, args, pad_id,
                                          old_cache=old_cache, ref_cache=ref_cache)
                timer.mark("fwd")
                gnorm = torch.nn.utils.clip_grad_norm_(
                    net.trainable_parameters(), args.max_grad_norm or float("inf")).item()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                timer.mark("opt")
                step_loss += loss_val / args.num_iterations
                # Report the LAST pass: on pass 0 the ratio is exactly 1 (old == policy),
                # so kl/clip_frac from it can never show what mu > 1 actually does
                mm = m
            timer.end_step()

            info = sample["info"]
            sup = mm.get("sup", 0)
            tot = sum(len(p) + len(c) for p, c in sample["rows"])
            tok_seen += sup
            tot_seen += tot
            meter.update(time.time() - step_t0, sup, tot)
            _, tot_tps = meter.rates()
            if start_loss is None:
                start_loss = step_loss
                start_reward = info["reward_mean"]
            end_loss = step_loss
            end_reward = info["reward_mean"]
            ema = info["reward_mean"] if ema is None else 0.9 * ema + 0.1 * info["reward_mean"]
            mean_len = sum(sample["lengths"]) / len(sample["lengths"])
            trunc_frac = sum(sample["truncated"]) / len(sample["truncated"])
            print(f"  step {step:>5}/{args.steps} | reward {info['reward_mean']:+.4f} "
                  f"(ema {ema:+.4f}, std {info['reward_std']:.3f}, zero-std "
                  f"{info['frac_zero_std']:.2f}) | loss {step_loss:+.5f} | "
                  f"kl {mm.get('kl', 0):.4f} clip {mm.get('clip_frac', 0):.3f} | "
                  f"len {mean_len:.0f} trunc {trunc_frac:.2f} | grad {gnorm:7.4f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | |B| {adapter_b_norm():7.3f} | "
                  f"{tot_tps:,.0f} tok/s | {timer.step_line()}")
            if report is not None:
                rec = {
                    "train/loss": step_loss, "train/reward": info["reward_mean"],
                    "train/reward_std": info["reward_std"],
                    "train/frac_zero_std": info["frac_zero_std"],
                    "train/kl": mm.get("kl", 0.0), "train/clip_frac": mm.get("clip_frac", 0.0),
                    "train/completion_len": mean_len, "train/truncated": trunc_frac,
                    "train/grad_norm": gnorm, "train/lr": sched.get_last_lr()[0],
                    "train/adapter_b_norm": adapter_b_norm(), "perf/tot_tok_s": tot_tps,
                }
                for j, (_, name) in enumerate(reward_funcs):
                    rec[f"train/reward_{name}"] = float(torch.nanmean(sample["per_func"][:, j]))
                report.log(rec, step=step)
            _FAIL_CTX["record"].update(steps_done=step, end_loss=round(step_loss, 6),
                                       peak_vram_gb=round(peak_vram_gb(), 3))

            if args.print_samples and args.sample_every and step % args.sample_every == 0:
                show_samples(sample, args.print_samples, f"[step {step}]")

            if args.eval_every and step % args.eval_every == 0 and val_examples:
                vl, em = evaluate()
                last_eval_step, last_val = step, vl
                if vl is not None:
                    print(f"    [eval] step {step}: {fmt_eval(vl, em)}")
                    if report is not None:
                        report.log({"eval/reward": vl,
                                    "eval/best_reward": max(vl, best_val)}, step=step)
                    if vl > best_val:
                        best_val, best_val_step = vl, step
                        if args.save_best:
                            save(f"[best step {step}, reward {vl:+.4f}]")

            if args.save_every and step % args.save_every == 0:
                save(f"[checkpoint step {step}]")
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                cdir = checkpoint_dir(args.out, step)
                net.save_adapter(cdir, base_model_name_or_path=args.model)
                save_trainer_state(cdir, step=step, opt=opt, sched=sched,
                                   best_val=best_val, best_val_step=best_val_step, ema=ema)
                print(f"  [checkpoint] step {step} -> {cdir} (resumable)")
                prune_checkpoints(args.out, args.keep_checkpoints)
            done_step = step
    except KeyboardInterrupt:
        step = done_step
        if args.save_best and val_examples:
            print(f"\nInterrupted at step {step}; keeping best-reward adapter.")
        else:
            print(f"\nInterrupted at step {step}; saving adapter before exit.")
            if step > 0:
                save("[interrupted]")
        log_run("interrupted", time.time() - t0, None)
        if report is not None:
            report.update_summary({"end_loss": end_loss,
                                   "best_val": best_val if best_val != float("-inf") else None,
                                   "best_val_step": best_val_step or None,
                                   "peak_vram_gb": peak_vram_gb(), "steps_done": step})
        _finish_report(exit_code=0, status="interrupted")
        raise SystemExit(0)

    dt = time.time() - t0
    _FAIL_CTX["phase"] = "final_eval"
    if last_eval_step == step:
        val = last_val
    elif val_examples:
        print(" -- computing final held-out eval ...")
        val, em = evaluate()
        if val is not None:
            print(f"    [eval] final: {fmt_eval(val, em)}")
    else:
        val = None
    if not (args.save_best and val_examples):
        save("Done.")
    if torch.cuda.is_available():
        peak_str = " / ".join(f"cuda:{d} {torch.cuda.max_memory_allocated(d) / 1e9:.2f}GB"
                              for d in active_devices)
    else:
        peak_str = "n/a"
    print(f"[PERF] {tok_seen / dt if dt else 0:,.0f} completion tok/s, "
          f"{tot_seen / dt if dt else 0:,.0f} tot tok/s | peak VRAM {peak_str} | "
          f"{dt:.0f}s for {step} steps | step time: {timer.summary()}")
    if start_reward is not None:
        print(f" -- train reward: {start_reward:+.4f} (step 1) -> {end_reward:+.4f} "
              f"(step {step}); reward ema {ema:+.4f}")
    log_run("completed", dt, val)
    if report is not None:
        report.update_summary({k: v for k, v in {
            "end_loss": end_loss, "final_val": val,
            "best_val": best_val if best_val != float("-inf") else None,
            "best_val_step": best_val_step or None,
            "peak_vram_gb": peak_vram_gb(),
            "sup_tok_s": tok_seen / dt if dt else 0,
            "tot_tok_s": tot_seen / dt if dt else 0,
            "total_s": dt, "steps_done": step,
        }.items() if v is not None})
        _finish_report()
    print("Verify with: python training/qlora_infer_native.py "
          f"--model {args.model} --adapter {args.out}")


def _hparam_note(args):
    return (f"method=grpo rewards={','.join(args.reward or [])} G={args.num_generations} "
            f"prompts/step={args.prompts_per_step} mu={args.num_iterations} "
            f"beta={args.beta} eps={args.epsilon}/{args.epsilon_high} "
            f"loss={args.loss_type} scale={args.scale_rewards} "
            f"temp={args.temperature} max_new={args.max_completion_length} "
            f"sampler={args.rollout_sampler}")


if __name__ == "__main__":
    main()
