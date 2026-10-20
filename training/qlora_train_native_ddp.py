"""
Multi-GPU (DistributedDataParallel) QLoRA fine-tuning of an EXL3 model, native
path (no HuggingFace Transformers).

Why DDP and not FSDP for QLoRA-on-EXL3
--------------------------------------
Only the LoRA ``a``/``b`` are trainable; the EXL3 base is frozen and tiny (that's
the whole point of low-bpw quant). So we don't want to *shard* the base -- we
replicate the (small, quantized) model on every GPU, shard the *batch*, and
all-reduce only the LoRA gradients (a few MB). That is exactly DDP. FSDP would
have to learn to all-gather the packed trellis buffers for no memory benefit.

We do NOT wrap the module in ``nn.parallel.DistributedDataParallel`` -- the model
is mostly frozen buffers plus a custom ``autograd.Function`` and gradient
checkpointing, which DDP's bucketing handles awkwardly. Instead we average the
LoRA gradients by hand after backward; with so few trainable params this is
trivial and fully equivalent.

Launch with torchrun (one process per GPU):

    torchrun --standalone --nproc_per_node=4 training/qlora_train_native_ddp.py \
        --model /path/to/exl3_model \
        --out   out/exl3_qlora_adapter \
        --dataset TeeZee/dolly-15k-pirate-speech \
        --instruction-key instruction --context-key context --response-key response \
        --lora-r 64 --alpha 64 --batch 16 --steps 800
        # NB: use --lora-r, not --r -- torchrun intercepts the abbreviated "--r".

Effective batch = --batch * nproc_per_node * --grad-accum.

NOTE: this has been written to standard DDP idioms but NOT yet run on a real
multi-GPU box. Validate the first run carefully (see the checklist at the bottom).
"""

import argparse
import datetime
import os
import random
import sys
import time

import torch
import torch.distributed as dist

# Reuse the single-GPU example's data helpers (same dir on sys.path under torchrun).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qlora_train_native import (  # noqa: E402
    build_sft_examples, build_lm_examples, pack_examples, collate, make_lr_scheduler,
    resolve_steps_and_warmup, ThroughputMeter, StepTimer, append_run_log,
    checkpoint_dir, prune_checkpoints,
    save_trainer_state, load_trainer_state, restore_optimizer_state,
    snapshot_datasets, resolve_resumed_datasets,
    format_prompt_and_eot, sample,
    _FAIL_CTX, _log_failure, _REPORT, _finish_report,
)
from run_report import (RunLogger, decode_example_docs,  # noqa: E402
                        start_live_monitor)

from exllamav3 import Config, Model, Tokenizer  # noqa: E402
from exllamav3.training.native_llama import NativeLlamaQLoRA  # noqa: E402


def ddp_setup():
    """Read torchrun env, init the process group, pin this rank to its GPU."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    # Pass device_id so NCCL collectives/barriers know this rank's device --
    # without it, barrier() warns and can HANG at teardown.
    dist.init_process_group(backend="nccl",
                            device_id=torch.device("cuda", local_rank))
    return rank, local_rank, world_size


def is_main(rank):
    return rank == 0


def main():
    """Run the DDP trainer with failure capture (mirrors the single-GPU arm):
    any exception on rank 0 appends a status=failed row to the run-log CSV and
    the traceback to <run_log>.errors.log before re-raising. Non-rank-0 ranks
    never log (their _FAIL_CTX run_log stays None)."""
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
    # Line-buffer stdout/stderr so per-step progress flushes on each newline.
    # Python block-buffers stdout when it isn't a TTY -- exactly when the run is
    # redirected to a file or piped through tee -- which otherwise holds every
    # step line in an ~8KB buffer and dumps them all at once at process exit.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass  # not a TextIOWrapper (already line-buffered, or wrapped)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="out/exl3_qlora_adapter")
    # NB: not "--r" -- torchrun greedily abbreviation-matches "--r" against its
    # own --rdzv-*/--role/--run-path options and errors out before our script
    # ever sees it. "--lora-r" sidesteps that; dest stays "r" so code is unchanged.
    ap.add_argument("--lora-r", dest="r", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=64.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0,
                    help="PEFT-style dropout on each per-linear LoRA branch's "
                         "input (frozen base path never dropped; train-time "
                         "only). Per-rank RNG, so DDP ranks draw independent "
                         "masks -- same as PEFT under DDP.")
    ap.add_argument("--use-rslora", action="store_true",
                    help="Rank-stabilized LoRA scaling: scale = alpha/sqrt(r) "
                         "instead of alpha/r (see the single-GPU arm).")
    ap.add_argument("--init-lora", choices=["default", "pissa", "qerr", "eva"],
                    default="default",
                    help="Adapter initialization (see the single-GPU arm). Under "
                         "DDP the init is computed ON RANK 0 ONLY (randomized SVD "
                         "is not deterministic across processes) and broadcast; "
                         "for pissa the frozen A0/B0 offsets -- part of every "
                         "forward -- are broadcast too, so all ranks compute "
                         "identical losses.")
    ap.add_argument("--init-svd-niter", type=int, default=16,
                    help="Randomized-SVD subspace iterations for --init-lora "
                         "(0 = exact full SVD, much slower). Default 16.")
    ap.add_argument("--init-ref-model", default=None,
                    help="ORIGINAL (unquantized) HF model dir; required by "
                         "--init-lora qerr.")
    ap.add_argument("--init-eva-tokens", type=int, default=65536,
                    help="Token budget for the --init-lora eva activation "
                         "pre-pass (rank 0 only; result is broadcast).")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01,
                    help="AdamW weight decay on the LoRA params (default 0.01).")
    ap.add_argument("--scheduler", choices=["none", "linear", "cosine"],
                    default="none",
                    help="LR schedule after warmup: none/linear/cosine (to 0).")
    ap.add_argument("--warmup-ratio", type=float, default=0.0,
                    help="Fraction of total steps to warm up the LR from 0 "
                         "(e.g. 0.05-0.1). Ignored if --warmup-steps>0.")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="Absolute warmup steps; overrides --warmup-ratio when >0.")
    ap.add_argument("--epochs", type=float, default=0.0,
                    help="If >0, set --steps to cover this many passes over the "
                         "FULL training set (one step = batch*world*grad-accum "
                         "examples), so the schedule matches the epoch count.")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=16, help="Per-GPU micro-batch")
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--dataset", default="superdrew100/UwU_Alpaca_data")
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--instruction-key", default="instruction")
    ap.add_argument("--context-key", default="input")
    ap.add_argument("--response-key", default="output")
    ap.add_argument("--messages-key", default=None,
                    help="Column holding OpenAI-style single-turn messages (e.g. "
                         "'messages' for UnstableLlama/semancy). When set, the user "
                         "turn is the prompt and the assistant turn the supervised "
                         "response; the flat instruction/response keys are ignored.")
    ap.add_argument("--prompt-format",
                    choices=["auto", "mistral", "metharme", "gemma4-nothink",
                             "llama3", "qwen3.5", "qwen3.5-nothink", "chatml",
                             "hy3", "hy3-nothink", "jinja"],
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
                         "(= auto for qwen3/3.5). qwen3.5-nothink: ChatML with an "
                         "empty <think> block pre-closed in the masked prompt. "
                         "hy3 / hy3-nothink: HunYuan V3 turns (= auto for the "
                         "hy_v3 arch), -nothink with the no_think header and an "
                         "empty think span pre-closed. "
                         "jinja: the model directory's own Jinja chat template "
                         "(see training/chat_jinja.py).")
    ap.add_argument("--chat-template-file", default=None,
                    help="(jinja) Jinja template file overriding the model "
                         "directory's template.")
    ap.add_argument("--template-vars", default=None,
                    help="(jinja) JSON object of extra template variables, "
                         "e.g. '{\"enable_thinking\": false}'.")
    ap.add_argument("--strip-sys-prompt-extras", action="store_true",
                    help="(jinja) Delete template-injected 'Reasoning "
                         "strength:', 'Knowledge cutoff:' and 'Current "
                         "date:' lines from every rendered prompt (OFF by "
                         "default). Harmony-style templates synthesize "
                         "these into the system block for rows with no "
                         "system message; the date comes from "
                         "strftime_now(), so without this the rendered "
                         "corpus changes from day to day.")
    ap.add_argument("--clean-text", action="store_true",
                    help="Strip [stage directions]/*actions* + normalize whitespace "
                         "(OFF by default; leave off for reasoning/code/markdown).")
    ap.add_argument("--no-clean-text", action="store_true",
                    help=argparse.SUPPRESS)  # deprecated: cleaning is now opt-in
    ap.add_argument("--min-response-words", type=int, default=3)
    ap.add_argument("--uppercase-response", action="store_true",
                    help="Smoke test: train to RESPOND IN ALL CAPS (dense, "
                         "unambiguous proof the training path works).")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle the training rows once (deterministically, same "
                         "order on every rank) before the --val-frac carve and "
                         "sharding, so the held-out split is random and shards are "
                         "well mixed. Matched with the single-GPU/BNB arms by seed.")
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="Seed for --shuffle (also the random-subset seed when "
                         "--max-samples caps the rows). Default 0.")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--pack", action="store_true",
                    help="Sample packing: concatenate training documents into each "
                         "--seq-len sequence instead of padding (kills pad-token "
                         "waste on short-answer data). Documents stay isolated "
                         "(per-doc RoPE reset + flash-varlen / block-diagonal "
                         "attention). Training set only; eval stays per-example. "
                         "Packed once (identically on every rank) then sharded, so "
                         "the per-rank block counts and step math stay in lockstep.")
    ap.add_argument("--pack-algo", choices=["bfd", "nextfit"], default="bfd",
                    help="Bin-packing strategy for --pack (bfd = best-fit "
                         "decreasing, ~97%%+ fill; nextfit = old behavior). "
                         "Deterministic, so ranks stay identical.")
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--expert-r", type=int, default=None,
                    help="LoRA rank for ROUTED-expert adapters (expert_* targets); "
                         "default: same as --lora-r. See the single-GPU trainer.")
    ap.add_argument("--train-embeddings", action="store_true",
                    help="Also FULLY train the input embeddings (modules_to_save). "
                         "Big (vocab x hidden): raises VRAM AND the per-step LoRA-"
                         "grad all-reduce by ~that size on every rank. On a tied "
                         "model this also trains the head.")
    ap.add_argument("--train-head", action="store_true",
                    help="Also FULLY train the LM head (modules_to_save); uses a "
                         "supervised-position CE so the head gets a gradient. Tied "
                         "model => same as --train-embeddings.")
    ap.add_argument("--compute-dtype", default="bfloat16",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--attn-impl", choices=["auto", "eager", "flash"], default="auto",
                    help="Attention kernel: auto (FlashAttention-2 when available "
                         "on CUDA fp16/bf16, else eager), flash (require), or eager "
                         "(reference, O(t^2) memory). Flash is O(t) -- long context.")
    ap.add_argument("--ce-chunk", type=int, default=1024)
    ap.add_argument("--head-vocab-chunk", type=int, default=0,
                    help="Reconstruct + matmul the frozen LM head in vocab-column "
                         "chunks (0 = off). Bounds head peak memory on the output "
                         "device for big-vocab models; same loss/grad. Try 32768.")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=0,
                    help="Overwrite the adapter at --out every N steps (rank 0). "
                         "Single latest copy; use --checkpoint-every for history.")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="Every N steps, save a RETAINED checkpoint (rank 0) to "
                         "--out/checkpoint-<step>, building a history. Independent "
                         "of --save-every and --save-best. 0 disables.")
    ap.add_argument("--keep-checkpoints", type=int, default=0,
                    help="Cap --checkpoint-every dirs to keep, deleting the oldest "
                         "(0 = keep all).")
    ap.add_argument("--eval-split", default=None,
                    help="Use this split of the dataset (e.g. 'test') as the "
                         "held-out eval set, instead of carving --val-frac off "
                         "train. Real held-out data; takes precedence over "
                         "--val-frac. Built identically on every rank.")
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
                         "primary one each --eval-every and at the end (e.g. your "
                         "test set vs wikitext). --save-best stays keyed on the "
                         "PRIMARY eval. Built identically on every rank.")
    ap.add_argument("--eval2-split", default="test",
                    help="Split for --eval2-dataset (default 'test').")
    ap.add_argument("--eval2-config", default=None,
                    help="HF dataset config for --eval2-dataset (e.g. "
                         "'wikitext-2-raw-v1' for the 'wikitext' dataset).")
    ap.add_argument("--eval2-text-key", default=None,
                    help="If set, treat --eval2-dataset as PLAIN TEXT and compute "
                         "an LM loss over packed --seq-len blocks (e.g. 'text' for "
                         "wikitext). If unset, built as a second SFT eval.")
    ap.add_argument("--eval2-max-samples", type=int, default=0,
                    help="Cap source rows for --eval2-dataset (0 = all).")
    ap.add_argument("--eval2-max-blocks", type=int, default=0,
                    help="Cap packed LM blocks for --eval2-text-key (0 = all); size "
                         "eval2 to roughly match the primary eval set.")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="Hold out this fraction of train for held-out eval loss; "
                         "the SAME deterministic split as the single-GPU / BNB "
                         "arms. Held out before sharding so it never leaks into "
                         "training. Ignored if --eval-split is set.")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="Report held-out loss every N steps (needs --val-frac>0).")
    ap.add_argument("--save-best", action="store_true",
                    help="Save only when held-out loss improves (needs "
                         "--val-frac + --eval-every); keeps the best checkpoint "
                         "instead of an overfit endpoint.")
    ap.add_argument("--resume", default=None,
                    help="Adapter dir to resume from (e.g. a single-GPU checkpoint). "
                         "Loaded on every rank before the broadcast. If the dir has "
                         "a trainer_state.pt (from --checkpoint-every / --save-*), "
                         "the optimizer, LR schedule and step are ALSO restored on "
                         "every rank so the run continues seamlessly (DDP ranks "
                         "hold identical synced state). --r/--targets must match.")
    ap.add_argument("--reset-optimizer", action="store_true",
                    help="With --resume, restore weights only and start the "
                         "optimizer/schedule/step fresh (use when changing "
                         "LR/schedule or the GPU count).")
    ap.add_argument("--sample-every", type=int, default=0,
                    help="Generate a sample completion every N steps (rank 0 "
                         "generates, all ranks barrier so lockstep is kept; 0 "
                         "disables -- note the single-GPU arm defaults to 25).")
    ap.add_argument("--sample-prompt", default="Tell me about your day.")
    ap.add_argument("--no-report", action="store_true",
                    help="Disable the local run report. By default every run "
                         "with an --out writes a self-contained report to "
                         "<out>/run_report/report.html (rank 0 only; same "
                         "format as the single-GPU arm).")
    ap.add_argument("--run-name", default=None,
                    help="Name for the local run report (defaults to the "
                         "basename of --out).")
    ap.add_argument("--live-report", action="store_true",
                    help="Serve a LIVE run monitor from rank 0 (localhost http "
                         "thread, opens the browser at run start): charts "
                         "redraw as metrics log, plus a step viewer decoding "
                         "the exact examples any step trains on -- every "
                         "rank's shard, labeled by rank; past or future steps "
                         "(deterministic order, simulated on rank 0; nothing "
                         "written to disk -- report.html stays the dataset-"
                         "free shareable artifact).")
    ap.add_argument("--live-report-port", type=int, default=0,
                    help="Port for --live-report (default 0 = pick a free one).")
    ap.add_argument("--run-log", default="qlora_runs.csv",
                    help="Append one metadata row per run to this CSV (rank 0 only). "
                         "Same schema as the single-GPU arm; failures are recorded "
                         "too (status=failed + error, traceback to "
                         "<run-log>.errors.log). Empty string disables.")
    ap.add_argument("--profile-dequant", type=int, default=0, metavar="N",
                    help="Measure frozen-weight (trellis) reconstruction time for "
                         "the first N steps (every rank profiles, rank 0 reports), "
                         "then disable. Adds sync overhead while active.")
    ap.add_argument("--ga-loss", choices=["token", "mean"], default="token",
                    help="Loss weighting across micro-batches AND ranks. token "
                         "(default): weight each micro-batch by its supervised-"
                         "token share of the whole step (one small all-reduce of "
                         "the counts per step), so the step gradient equals one "
                         "big batch. mean: the pre-Session-11 mean-of-means.")
    ap.add_argument("--dequant-mode", choices=["fast", "legacy"], default="fast",
                    help="Frozen-weight dequant path (audit A1); see the "
                         "single-GPU arm. Ranks behave identically either way.")
    ap.add_argument("--dequant-cache", action="store_true",
                    help="Opt-in recompute->backward frozen-weight cache (3 -> 2 "
                         "dequants per step, costs VRAM; a net loss under the "
                         "default fast path -- see the single-GPU arm).")
    args = ap.parse_args()

    # Process-wide, before any renderer is built (see chat_jinja).
    from chat_jinja import set_strip_sys_prompt_extras
    set_strip_sys_prompt_extras(args.strip_sys_prompt_extras)

    from exllamav3.training import backbone as _backbone_cfg
    _backbone_cfg.set_dequant_mode(args.dequant_mode)

    # Cleaning is opt-in (--clean-text); --no-clean-text is now a no-op kept for
    # backward compatibility (warned once, on rank 0, below after setup).
    clean_text = args.clean_text

    rank, local_rank, world_size = ddp_setup()
    device = f"cuda:{local_rank}"

    # Dataset-mix continuity (see the single-GPU arm). Every rank resolves the
    # same paths deterministically; interactive=False because a per-rank prompt
    # under torchrun would desync the ranks -- a hash mismatch aborts with
    # instructions instead. Only rank 0 writes the fresh-run snapshot.
    if args.out:
        data_roles = {"dataset": args.dataset,
                      "eval_dataset": args.eval_dataset,
                      "eval2_dataset": args.eval2_dataset}
        if args.resume:
            resolved = resolve_resumed_datasets(
                args.out, args.resume, data_roles,
                interactive=False, verbose=is_main(rank))
            args.dataset = resolved["dataset"]
            args.eval_dataset = resolved["eval_dataset"]
            args.eval2_dataset = resolved["eval2_dataset"]
        elif is_main(rank):
            snapshot_datasets(args.out, data_roles)

    # Failure logger: rank 0 only (other ranks keep run_log=None -> no-op), so a
    # crash anywhere after this point still leaves one run-log row + traceback.
    if is_main(rank):
        _FAIL_CTX["run_log"] = args.run_log
        _FAIL_CTX["phase"] = "startup"
        _FAIL_CTX["record"] = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "arm": "exl3-native-ddp", "model": args.model, "out": args.out,
            "dataset": args.dataset, "eval_split": args.eval_split or "",
            "eval_dataset": args.eval_dataset or "",
            "eval2_dataset": args.eval2_dataset or "",
            "r": args.r, "alpha": args.alpha,
            "expert_r": "" if args.expert_r is None else args.expert_r,
            "use_rslora": int(bool(args.use_rslora)), "init_lora": args.init_lora,
            "lr": args.lr,
            "scheduler": args.scheduler, "weight_decay": args.weight_decay,
            "batch": args.batch, "grad_accum": args.grad_accum,
            "world_size": world_size,
            "eff_batch": args.batch * world_size * args.grad_accum,
            "epochs": args.epochs, "steps_planned": args.steps,
            "seq_len": args.seq_len, "compute_dtype": args.compute_dtype,
            "attn_impl": args.attn_impl, "parallel": "ddp",
            "shuffle": int(bool(args.shuffle)), "pack": int(bool(args.pack)),
            "pack_algo": args.pack_algo if args.pack else "",
            "ga_loss": args.ga_loss,
            "max_samples": args.max_samples,
            "train_embeddings": int(bool(args.train_embeddings)),
            "train_head": int(bool(args.train_head)),
            "prompt_format": args.prompt_format,
        }
    if args.no_clean_text and is_main(rank):
        print(" -- note: --no-clean-text is deprecated; cleaning is now OFF by "
              "default (use --clean-text to enable).")
    cdt = {"float32": torch.float32, "float16": torch.float16,
           "bfloat16": torch.bfloat16}[args.compute_dtype]

    # 1. Every rank loads a full copy of the (small, quantized) model on its GPU.
    _FAIL_CTX["phase"] = "load_model"
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    # KV cache for live samples: must exist BEFORE model.load() so attention
    # layers allocate their cache pages. Rank 0 only -- it's the only rank that
    # generates (the others barrier), and the cache costs VRAM.
    cache = None
    if args.sample_every and is_main(rank):
        from exllamav3 import Cache
        cache = Cache(model, max_num_tokens=4096)
    model.load(device=device, progressbar=is_main(rank))
    tokenizer = Tokenizer.from_config(config)
    pad_id = tokenizer.pad_token_id
    if pad_id is None or pad_id < 0:
        pad_id = tokenizer.eos_token_id or 0

    # 2. Build the differentiable QLoRA model (frozen base + trainable adapters).
    _FAIL_CTX["phase"] = "build_net"
    if is_main(rank):
        _FAIL_CTX["record"]["arch"] = getattr(config, "architecture", "")
    net = NativeLlamaQLoRA(
        model, r=args.r, alpha=args.alpha, target_modules=args.targets,
        use_rslora=args.use_rslora,
        compute_dtype=cdt, gradient_checkpointing=not args.no_grad_ckpt,
        train_embeddings=args.train_embeddings, train_head=args.train_head,
        attn_impl=args.attn_impl, head_vocab_chunk=args.head_vocab_chunk,
        expert_r=args.expert_r, lora_dropout=args.lora_dropout,
    )
    net.train()
    if args.pack and (getattr(net, "has_gdn", False)
                      or getattr(net, "has_shortconv", False)):
        raise SystemExit(
            "--pack is not supported on GatedDeltaNet (Qwen3.5/3.6) or ShortConv "
            "(LFM2) models: the recurrence / causal conv would carry state "
            "across packed document boundaries. Drop --pack and train unpacked.")
    if is_main(rank):
        ms = net.modules_to_save_parameters()
        print(f" -- world_size {world_size}, trainable params: "
              f"{net.num_trainable():,} (r={args.r}, alpha={args.alpha}"
              f"{', +modules_to_save (' + str(sum(p.numel() for p in ms)) + ')' if ms else ''})")
        print(f" -- {net.describe_attn()}")

    # 3a. SVD-based LoRA init (pissa/qerr). Rank 0 ONLY: the fast-SVD recipe
    #     (torch.svd_lowrank) is randomized and NOT deterministic across
    #     processes, so every rank computing its own would silently diverge.
    #     Rank 0's result reaches the other ranks via the 3c/3d broadcasts.
    #     On resume this is recomputed and then OVERWRITTEN by load_adapter
    #     below (pissa restores its exact offsets from the checkpoint sidecar)
    #     -- a few wasted seconds on rank 0, kept for parity with the
    #     single-GPU arm. (eva runs after the dataset is built -- it needs an
    #     activation pre-pass.)
    if args.init_lora in ("pissa", "qerr"):
        _FAIL_CTX["phase"] = "init_lora"
        if is_main(rank):
            net.apply_init_lora(args.init_lora, ref_model_dir=args.init_ref_model,
                                svd_niter=args.init_svd_niter)

    # 3b. Optionally resume from a checkpoint (e.g. stop a single-GPU run, then
    #     continue on N GPUs). Every rank loads the same file; the broadcast below
    #     then guarantees bit-identical starting weights regardless.
    if args.resume:
        net.load_adapter(args.resume)

    # 3c. Sync initial trainable weights from rank 0 so every rank starts
    #     identical (lora_a is random, lora_b zero; embed/head copies are
    #     identical already but broadcast for safety). Without this, ranks would
    #     diverge from step 1.
    for p in net.trainable_parameters():
        dist.broadcast(p.data, src=0)

    # 3d. PiSSA's frozen A0/B0 offsets are part of EVERY forward (the wrappers
    #     compute against the residual base W - s*A0@B0), so they must be
    #     bit-identical across ranks too -- 3c only covers the trainable A/B.
    #     Broadcast rank 0's fp32 masters (from the fresh SVD, or the resume
    #     sidecar) and install them on the other ranks. set_init_offset also
    #     refreshes the CPU masters, so a rank-0 save writes the same values.
    if args.init_lora == "pissa":
        for w in net._wrappers:
            if w.r <= 0:
                continue
            if is_main(rank):
                a0 = w.init_a0_master.to(device)
                b0 = w.init_b0_master.to(device)
            else:
                a0 = torch.empty(w.lora_a.shape, dtype=torch.float32, device=device)
                b0 = torch.empty(w.lora_b.shape, dtype=torch.float32, device=device)
            dist.broadcast(a0, src=0)
            dist.broadcast(b0, src=0)
            w.set_init_offset(a0, b0)

    # 4. Data. Build the full set identically on every rank (same seed/order), then
    #    take a disjoint stride-shard so each GPU trains on different examples.
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
        template_vars=template_vars,
    )
    # Held-out eval set, built identically on every rank. Prefer the dataset's own
    # eval split (real held-out data); otherwise carve the first val_frac off
    # train BEFORE sharding so those rows never leak into any training shard.
    val_examples = []
    if args.eval_split:
        if args.eval_text_key:
            # Plain-text LM eval (parity with --eval2-text-key); what
            # --save-best tracks. Built identically on every rank.
            val_examples = build_lm_examples(
                tokenizer, args.eval_dataset or args.dataset, args.eval_split,
                args.seq_len, text_key=args.eval_text_key,
                max_samples=args.eval_max_samples,
                config_name=args.eval_config,
                max_blocks=args.eval_max_blocks)
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
                template_vars=template_vars,
            )
    elif args.val_frac > 0:
        n_val = max(1, int(len(examples) * args.val_frac))
        val_examples, examples = examples[:n_val], examples[n_val:]

    # Optional SECOND held-out eval set (e.g. wikitext LM), built identically on
    # every rank so all ranks stay in lockstep when evaluating it.
    val2_examples = []
    eval2_label = ""
    if args.eval2_dataset:
        eval2_label = args.eval2_dataset.split("/")[-1]
        if args.eval2_text_key:
            val2_examples = build_lm_examples(
                tokenizer, args.eval2_dataset, args.eval2_split, args.seq_len,
                text_key=args.eval2_text_key, max_samples=args.eval2_max_samples,
                max_blocks=args.eval2_max_blocks,
                config_name=args.eval2_config)
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
    # Sample packing (training set only). Pack the FULL set identically on every
    # rank BEFORE sharding, so all ranks see the same block count (lockstep step
    # math) and the stride-shard stays balanced. The val carve above already
    # removed held-out docs, so no eval data leaks into a packed block.
    if args.pack:
        n_docs = len(examples)
        real_tokens = sum(len(ex["input_ids"]) for ex in examples)
        examples = pack_examples(examples, args.seq_len, pad_id, algo=args.pack_algo)
        cap = max(1, len(examples) * args.seq_len)
        if is_main(rank):
            print(f" -- packed ({args.pack_algo}) {n_docs} docs -> {len(examples)} "
                  f"blocks of {args.seq_len} tok ({100.0 * real_tokens / cap:.1f}% "
                  f"filled, ~{real_tokens / max(1, len(examples)):.0f} real tok/block)")

    # eva init runs HERE (not with pissa/qerr above): it streams a no-grad
    # activation pre-pass over the actual training batches. Rank 0 only (the
    # incremental sketch uses randomized svd_lowrank -- not deterministic across
    # processes); the examples list is identical on every rank so rank 0's
    # pre-pass sees the same data a single-GPU run would. The other ranks wait
    # in the broadcast below. Skipped on resume -- the checkpoint's A/B already
    # carry the init, and unlike pissa there are no frozen offsets.
    if args.init_lora == "eva" and not args.resume:
        _FAIL_CTX["phase"] = "init_lora"
        if is_main(rank):
            def eva_prepass():
                used, i = 0, 0
                while used < args.init_eva_tokens and i < len(examples):
                    batch = examples[i:i + args.batch]
                    i += len(batch)
                    input_ids, _, attn, pos_ids, seg_ids = collate(batch, pad_id)
                    used += int(attn.sum())
                    yield dict(input_ids=input_ids, attention_mask=attn,
                               position_ids=pos_ids, seg_ids=seg_ids)

            net.apply_init_lora("eva", svd_niter=args.init_svd_niter,
                                eva_batches=eva_prepass())
        for p in net.trainable_parameters():
            dist.broadcast(p.data, src=0)
    elif args.init_lora == "eva" and is_main(rank):
        print(" -- eva init skipped on --resume (the checkpoint's adapters "
              "already carry it)")

    shard = examples[rank::world_size]
    assert shard, "no training examples on this rank"

    # Finalize step count (from --epochs over the FULL train set) and warmup.
    eff_batch = args.batch * world_size * args.grad_accum
    args.steps, warmup_steps = resolve_steps_and_warmup(args, len(examples), eff_batch)
    # For the per-step epoch readout (steps_per_epoch stashed by the resolver).
    epochs_total = args.epochs if args.epochs > 0 else args.steps / args.steps_per_epoch
    if is_main(rank):
        print(f" -- {len(examples)} train examples total, ~{len(shard)} per rank; "
              f"{len(val_examples)} held out for eval "
              f"({'split ' + args.eval_split if args.eval_split else 'val_frac'})")
        print(f" -- {args.steps} steps, eff_batch {eff_batch}, "
              f"scheduler={args.scheduler}, warmup={warmup_steps}, "
              f"weight_decay={args.weight_decay}")

    # Local run report -- the DEFAULT logging path, rank 0 only (same format as
    # the single-GPU arm; metrics stream to disk so a crash still renders).
    # Registered in the shared _REPORT so _log_failure finishes it on a crash.
    report = None
    if is_main(rank) and args.out and not args.no_report:
        run_config = dict(_FAIL_CTX["record"])
        run_config.update(
            steps_planned=args.steps, steps_per_epoch=args.steps_per_epoch,
            warmup_steps=warmup_steps, targets=" ".join(net.target_modules),
            trainable_params=net.num_trainable(), n_train=len(examples),
            n_val=len(val_examples), n_eval2=len(val2_examples))
        run_name = args.run_name or os.path.basename(os.path.normpath(args.out))
        report = RunLogger(args.out, run_name, config=run_config)
        _REPORT["rep"] = report

    # Live-sample generator (rank 0 only; see --sample-every). The barrier keeps
    # the other ranks in lockstep while rank 0 generates.
    generator, build_prompt = None, None
    if args.sample_every:
        if is_main(rank):
            from exllamav3 import Generator
            build_prompt, _ = format_prompt_and_eot(
                model, tokenizer, args.prompt_format,
                chat_template_file=args.chat_template_file,
                template_vars=template_vars)
            generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
            net.eval()
            with torch.inference_mode():
                base = sample(model, cache, tokenizer, generator, build_prompt,
                              args.sample_prompt)
            net.train()
            print(f"\n\U0001f3ad  baseline (step 0): {args.sample_prompt}\n     -> {base}\n")
        dist.barrier()

    opt = torch.optim.AdamW(net.param_groups(args.weight_decay), lr=args.lr)
    sched = make_lr_scheduler(opt, args.scheduler, args.steps, warmup_steps)

    # Restore optimizer/schedule/step from the resumed checkpoint (every rank loads
    # the same trainer_state.pt; ranks hold identical synced state after each
    # all-reduce, so this keeps them in lockstep). --reset-optimizer skips it.
    resume_step, resume_state = 0, None
    yaml_base_lrs = list(sched.base_lrs)  # per-group base LRs from the CONFIG
    if args.resume and not args.reset_optimizer:
        resume_state = load_trainer_state(args.resume)
        if resume_state is not None:
            restore_optimizer_state(opt, resume_state["optimizer"])
            if resume_state.get("scheduler") is not None:
                sched.load_state_dict(resume_state["scheduler"])
            # Config lr wins on resume (see the single-GPU arm): re-base the
            # restored schedule at the config value; moments/position kept.
            # Deterministic, so every rank applies the identical override.
            if list(sched.base_lrs) != yaml_base_lrs:
                old_lr = sched.base_lrs[0]
                sched.base_lrs = list(yaml_base_lrs)
                for g, base, fn in zip(opt.param_groups, sched.base_lrs,
                                       sched.lr_lambdas):
                    g["initial_lr"] = base
                    g["lr"] = base * fn(sched.last_epoch)
                sched._last_lr = [g["lr"] for g in opt.param_groups]
                if is_main(rank):
                    print(f" -- resume LR override: base lr {old_lr:.2e} "
                          f"(checkpoint) -> {yaml_base_lrs[0]:.2e} (config); "
                          f"optimizer moments and schedule position kept")
            resume_step = int(resume_state["step"])
            if is_main(rank):
                print(f" -- resumed trainer state from {args.resume}: continuing at "
                      f"step {resume_step + 1}/{args.steps} (best_val "
                      f"{resume_state['best_val']}, lr {sched.get_last_lr()[0]:.2e})")
        elif is_main(rank):
            print(f" -- {args.resume} has no trainer_state.pt; resuming weights "
                  f"only (cold optimizer + schedule from step 0).")

    # Live monitor (--live-report, rank 0 only). Mirrors every rank's batches():
    # shard r = examples[r::world_size] shuffled by a STATEFUL per-rank rng
    # (Random(1234+r), cumulative shuffles across epochs), so rank 0 simulates
    # each rank's rng stream, extending the per-epoch permutations lazily as
    # future steps are browsed. Optimizer step s consumes micro-batches
    # (s-1)*ga .. (s-1)*ga+ga-1 of every rank's own stream -- an ABSOLUTE
    # mapping, valid on resume too now that each rank fast-forwards its bgen
    # to the resume step; shard sizes (and so micro-batches per pass) can
    # differ by rank, which is why the epoch index is computed per rank.
    if args.live_report and is_main(rank):
        if report is None:
            print(" -- --live-report needs the local report; ignoring "
                  "(--no-report set or no --out)")
        else:
            live_shards = [examples[r::world_size] for r in range(world_size)]
            live_orders = [{"rng": random.Random(1234 + r),
                            "order": list(range(len(live_shards[r]))),
                            "perms": []} for r in range(world_size)]

            def live_perm(r, e):
                st = live_orders[r]
                while len(st["perms"]) <= e:
                    st["rng"].shuffle(st["order"])
                    st["perms"].append(st["order"][:])
                return st["perms"][e]

            def live_batch_fn(s):
                if not (0 < s <= args.steps):
                    raise ValueError(f"step must be in [1, {args.steps}]")
                mbs = []
                for r in range(world_size):
                    mpe = max(1, len(live_shards[r]) // args.batch)
                    for g in range(args.grad_accum):
                        m = (s - 1) * args.grad_accum + g
                        e, w = divmod(m, mpe)
                        seqs = [{"index": r + j * world_size,  # pre-shard index
                                 "docs": decode_example_docs(
                                     tokenizer, live_shards[r][j])}
                                for j in live_perm(r, e)[w * args.batch:
                                                         (w + 1) * args.batch]]
                        mbs.append({"micro": g, "epoch": e, "rank": r,
                                    "sequences": seqs})
                return {"step": s, "micro_batches": mbs}

            start_live_monitor(
                args.out, batch_fn=live_batch_fn, port=args.live_report_port,
                live_info={"total_steps": args.steps,
                           "first_step": resume_step + 1,
                           "run_name": run_name})

    # |dB| telemetry baseline (rank 0 only -- it's the only rank that logs).
    # Same rationale as the single-GPU arm: with an SVD init the raw ||B|| is
    # dominated by the init component, so log the distance from init instead.
    b0_refs = []
    if is_main(rank):
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

    def batches():
        order = list(range(len(shard)))
        # Per-rank RNG so shards reshuffle independently each epoch.
        rng = random.Random(1234 + rank)
        while True:
            rng.shuffle(order)
            for i in range(0, len(order) - args.batch + 1, args.batch):
                yield [shard[j] for j in order[i:i + args.batch]]

    def allreduce_grads():
        # Average trainable grads across ranks == DDP. Once per optimizer step,
        # after accumulation, before clipping. NB: with --train-embeddings/-head
        # the embed/head grads (vocab x hidden) are reduced here too -- much
        # larger than the usual few-MB LoRA grads.
        #
        # Zero-fill missing grads first: with routed-expert adapters
        # (expert_* targets on a MoE model) an expert can receive tokens on
        # one rank but not another, leaving p.grad None on some ranks only --
        # the old per-rank `if grad is not None: all_reduce` would then pair
        # MISMATCHED collectives across ranks (hang / silent corruption).
        # Zero is the correct value for "no tokens routed here this step",
        # and every rank running every all_reduce keeps them in lockstep.
        for p in net.trainable_parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad /= world_size

    def save(tag):
        # Single writer; every rank holds identical adapters after the all-reduce.
        if is_main(rank):
            net.save_adapter(args.out, base_model_name_or_path=args.model)
            save_trainer_state(args.out, step=step, opt=opt, sched=sched,
                               best_val=best_val, best_val_step=best_val_step, ema=ema)
            print(f"{tag} adapter -> {args.out}")
        dist.barrier()

    def eval_loss(exs):
        # All ranks compute the same loss (replicated, synced adapters) so they
        # stay in lockstep; only rank 0 prints. Mean per-example loss, batch 1 --
        # identical to the single-GPU and BNB arms. Works for SFT and plain-LM
        # eval sets alike.
        if not exs:
            return None
        net.eval()
        total = 0.0
        with torch.no_grad():
            for ex in exs:
                ii, ll, aa, pp, ss = collate([ex], pad_id)
                total += net.compute_loss(ii, ll, attention_mask=aa,
                                          chunk=args.ce_chunk,
                                          position_ids=pp, seg_ids=ss).item()
        net.train()
        return total / len(exs)

    def evaluate():
        return eval_loss(val_examples)

    bgen = batches()
    if resume_step:
        # Fast-forward each rank's stream to where the original process
        # stopped (see the single-GPU arm; doc/bug_resume_data_order.md).
        # Every rank skips the same count on its OWN shard/rng, so the
        # resumed order is exactly what an uninterrupted run would have used.
        for _ in range(resume_step * args.grad_accum):
            next(bgen)
        if is_main(rank):
            print(f" -- data stream fast-forwarded "
                  f"{resume_step * args.grad_accum} micro-batches/rank to "
                  f"continue the run's batch order at step {resume_step + 1}")
    opt.zero_grad(set_to_none=True)
    # Seed from the resumed state so best-tracking / EMA continue (None under
    # --reset-optimizer or a weights-only dir).
    ema = resume_state["ema"] if resume_state else None
    step = resume_step
    best_val = resume_state["best_val"] if resume_state else float("inf")
    best_val_step = resume_state["best_val_step"] if resume_state else 0
    start_loss = end_loss = None
    start_val = start_eval2 = None
    last_eval_step, last_val, last_eval2 = -1, None, None
    tok_seen, tot_seen, t0 = 0, 0, time.time()
    run_started = datetime.datetime.now().isoformat(timespec="seconds")
    status = "completed"
    meter = ThroughputMeter()

    # (log_run is defined once below, after the baseline eval -- an earlier
    # duplicate that used to sit here was dead code; NB the duplicate pair had
    # drifted apart, silently dropping start_val/start_eval2 from the CSV.)

    # Baseline eval at step 0 (no-op adapter = base model); all ranks compute in
    # lockstep, rank 0 prints. Reference point for the trained numbers.
    _FAIL_CTX["phase"] = "baseline_eval"
    if val_examples or val2_examples:
        start_val = evaluate()
        start_eval2 = eval_loss(val2_examples)
        if is_main(rank):
            parts = []
            if start_val is not None:
                parts.append(f"held-out {start_val:.4f}")
            if start_eval2 is not None:
                parts.append(f"{eval2_label} {start_eval2:.4f}")
            print("    [eval] step 0 (baseline): " + " | ".join(parts))
            if report is not None:
                base_eval = {k: v for k, v in (("eval/held_out", start_val),
                                               ("eval/eval2", start_eval2))
                             if v is not None}
                report.log(base_eval, step=0)

    # Start the training timer + VRAM peak after the baseline eval.
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    def log_run(status, dt, final_val, final_eval2, sup_tok_s, tot_tok_s):
        # Rank 0 writes one CSV row (same schema as the single-GPU arm). tok/s are
        # passed in: the caller has the all-reduced totals at normal finish, or a
        # per-rank x world_size estimate on interrupt (no collective there).
        # Disarms the failure logger (this run is recorded).
        _FAIL_CTX["logged"] = True
        if not is_main(rank):
            return
        rnd = lambda x, n=6: round(x, n) if isinstance(x, (int, float)) else ""
        append_run_log(args.run_log, {
            "timestamp": run_started, "arm": "exl3-native-ddp", "status": status,
            "model": args.model, "arch": getattr(config, "architecture", ""),
            "out": args.out, "dataset": args.dataset,
            "eval_split": args.eval_split or "", "eval_dataset": args.eval_dataset or "",
            "eval2_dataset": args.eval2_dataset or "",
            "r": args.r, "alpha": args.alpha,
            "expert_r": "" if args.expert_r is None else args.expert_r,
            "use_rslora": int(bool(args.use_rslora)), "init_lora": args.init_lora,
            "lr": args.lr,
            "scheduler": args.scheduler, "warmup_steps": warmup_steps,
            "weight_decay": args.weight_decay, "batch": args.batch,
            "grad_accum": args.grad_accum, "world_size": world_size,
            "eff_batch": eff_batch, "epochs": args.epochs,
            "steps_planned": args.steps, "steps_done": step, "seq_len": args.seq_len,
            "targets": " ".join(net.target_modules), "compute_dtype": args.compute_dtype,
            "attn_impl": args.attn_impl, "parallel": "ddp",
            "shuffle": int(bool(args.shuffle)), "pack": int(bool(args.pack)),
            "pack_algo": args.pack_algo if args.pack else "",
            "ga_loss": args.ga_loss,
            "max_samples": args.max_samples,
            "train_embeddings": int(bool(args.train_embeddings)),
            "train_head": int(bool(args.train_head)), "prompt_format": args.prompt_format,
            "trainable_params": net.num_trainable(), "n_train": len(examples),
            "n_val": len(val_examples), "n_eval2": len(val2_examples),
            "start_loss": rnd(start_loss), "end_loss": rnd(end_loss),
            "best_val": rnd(best_val) if best_val != float("inf") else "",
            "best_val_step": best_val_step or "",
            "start_val": rnd(start_val), "start_eval2": rnd(start_eval2),
            "final_val": rnd(final_val), "final_eval2": rnd(final_eval2),
            "total_s": rnd(dt, 1), "s_per_step": rnd(dt / step, 4) if step else "",
            "sup_tok_s": round(sup_tok_s) if sup_tok_s else "",
            "tot_tok_s": round(tot_tok_s) if tot_tok_s else "",
            "peak_vram_gb": rnd(torch.cuda.max_memory_allocated(device) / 1e9, 3),
            # Section timings are rank 0's own (ranks are in lockstep, so
            # representative of all).
            "t_data_s": rnd(timer.total["data"], 1), "t_fwd_s": rnd(timer.total["fwd"], 1),
            "t_bwd_s": rnd(timer.total["bwd"], 1), "t_opt_s": rnd(timer.total["opt"], 1),
            "dequant_s_per_step": rnd(dequant_s_per_step, 3)
                if dequant_s_per_step is not None else "",
            "phase": "", "error": "", "notes": "",
        })

    # Per-step wall-clock breakdown (this rank's own; ranks run in lockstep).
    timer = StepTimer(devices=[local_rank])

    # Optional dequant profiling window (all ranks profile so timing stays in
    # lockstep; rank 0 reports).
    from exllamav3.training import backbone as _backbone
    dequant_cache = args.dequant_cache and not args.no_grad_ckpt
    dq_profile = None
    dequant_s_per_step = None
    if args.profile_dequant > 0:
        dq_profile = {"calls": 0, "s": 0.0}
        _backbone.profile_dequant(dq_profile)
        if is_main(rank):
            print(f" -- profiling dequant (trellis reconstruction) for the first "
                  f"{args.profile_dequant} steps; adds sync overhead while active")
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
            timer.begin_step()
            # --ga-loss token (see the single-GPU arm for the rationale): weight
            # each micro-batch by its supervised-token share of the WHOLE step --
            # across micro-batches AND ranks, so one all-reduce of the counts.
            # allreduce_grads() divides the summed grads by world_size, so each
            # rank scales by world_size/total to land at loss_sum/total overall.
            # Counts use the SHIFTED labels to match the CE denominator.
            window = [collate(next(bgen), pad_id) for _ in range(args.grad_accum)]
            n_sups = [int((w[1][:, 1:] != -100).sum()) for w in window]
            if args.ga_loss == "token":
                tot = torch.tensor(float(sum(n_sups)), device=device)
                dist.all_reduce(tot, op=dist.ReduceOp.SUM)
                total_sup = max(tot.item(), 1.0)
            timer.mark("data")
            for (input_ids, labels, attn, pos_ids, seg_ids), n_sup in zip(window, n_sups):
                loss = net.compute_loss(input_ids, labels, attention_mask=attn,
                                        chunk=args.ce_chunk,
                                        position_ids=pos_ids, seg_ids=seg_ids)
                # .item() before backward (harmless to the graph) so the fwd/bwd
                # sections split cleanly at the sync.
                loss_val = loss.item()
                timer.mark("fwd")
                w_i = (n_sup * world_size / total_sup) if args.ga_loss == "token" \
                    else (1.0 / args.grad_accum)
                with _backbone.backward_dequant_cache(enable=dequant_cache):
                    (loss * w_i).backward()
                timer.mark("bwd")
                # The log line below SUMs accum_loss across ranks then divides
                # by world_size; since the token weights w_i sum to world_size
                # across the whole step, that reduction yields exactly the
                # per-token mean over the step (and the old mean-of-means for
                # --ga-loss mean).
                accum_loss += loss_val * w_i
                step_sup += int((labels != -100).sum())
                step_tot += int(attn.sum())

            # "opt" section includes the grad all-reduce (the DDP comm cost) --
            # a fat opt% relative to the single-GPU arm points at the interconnect.
            allreduce_grads()
            gnorm = torch.nn.utils.clip_grad_norm_(
                net.trainable_parameters(), args.max_grad_norm or float("inf")
            ).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            timer.mark("opt")
            timer.end_step()

            # Average the loss across ranks just for a representative log line.
            lt = torch.tensor(accum_loss, device=device)
            dist.all_reduce(lt, op=dist.ReduceOp.SUM)
            global_loss = lt.item() / world_size
            if start_loss is None:
                start_loss = global_loss
            end_loss = global_loss
            # Live tok/s is this rank's only (no per-step collective); shards are
            # balanced so multiply by world_size for an aggregate estimate. The
            # final [PERF] line all-reduces the true total.
            tok_seen += step_sup
            tot_seen += step_tot
            meter.update(time.time() - step_t0, step_sup, step_tot)
            _, tot_tps = meter.rates()
            ema = global_loss if ema is None else 0.9 * ema + 0.1 * global_loss
            if is_main(rank):
                epoch_now = step / args.steps_per_epoch
                cur_lr = sched.get_last_lr()[0]
                b_dist = adapter_b_norm()
                print(f"  step {step:>5}/{args.steps} | "
                      f"ep {epoch_now:.2f}/{epochs_total:.4g} | "
                      f"loss {global_loss:6.4f} | "
                      f"ema {ema:6.4f} | grad {gnorm:7.4f} | "
                      f"lr {cur_lr:.2e} | |dB| {b_dist:7.3f} | "
                      f"~{tot_tps * world_size:,.0f} tok/s | {timer.step_line()}")
                if report is not None:
                    sup_tps, _ = meter.rates()
                    report.log({
                        "train/loss": global_loss, "train/ema": ema,
                        "train/grad_norm": gnorm, "train/lr": cur_lr,
                        "train/adapter_b_dist": b_dist, "train/epoch": epoch_now,
                        # This rank's rates x world_size -- an aggregate estimate
                        # (shards are balanced); the final [PERF] line is exact.
                        "perf/sup_tok_s": sup_tps * world_size,
                        "perf/tot_tok_s": tot_tps * world_size,
                    }, step=step)
                # Keep the failure record current (steps reached, loss, memory).
                _FAIL_CTX["record"].update(
                    steps_done=step, end_loss=round(global_loss, 6),
                    peak_vram_gb=round(torch.cuda.max_memory_allocated(device) / 1e9, 3))

            # End of the dequant profiling window: report and disable (all ranks
            # disable together so lockstep timing is preserved).
            if dq_profile is not None and (step - resume_step) >= args.profile_dequant:
                n_prof = step - resume_step
                wall = sum(timer.total.values())
                dequant_s_per_step = dq_profile["s"] / max(1, n_prof)
                if is_main(rank):
                    print(f"  [profile] dequant: {dq_profile['calls']:,} reconstructions, "
                          f"{dq_profile['s']:.2f}s over {n_prof} steps = "
                          f"{dequant_s_per_step:.3f}s/step "
                          f"({100.0 * dq_profile['s'] / max(wall, 1e-9):.0f}% of step "
                          f"wall time) -- profiling off from here")
                from exllamav3.training import backbone as _backbone
                _backbone.profile_dequant(None)
                dq_profile = None

            if (args.eval_every and step % args.eval_every == 0
                    and (val_examples or val2_examples)):
                vl = evaluate()
                v2 = eval_loss(val2_examples)
                last_eval_step, last_val, last_eval2 = step, vl, v2
                if is_main(rank):
                    parts = []
                    if vl is not None:
                        parts.append(f"held-out {vl:.4f}")
                    if v2 is not None:
                        parts.append(f"{eval2_label} {v2:.4f}")
                    print(f"    [eval] step {step}: " + " | ".join(parts))
                # Track best val for the run log regardless of --save-best; all
                # ranks evaluate identically so they branch in lockstep (save()
                # has a barrier). Only checkpoint when --save-best.
                if vl is not None and vl < best_val:
                    best_val = vl
                    best_val_step = step
                    if args.save_best:
                        save(f"[best step {step}, val {vl:.4f}]")
                if is_main(rank) and report is not None:
                    eval_metrics = {k: v for k, v in (("eval/held_out", vl),
                                                      ("eval/eval2", v2))
                                    if v is not None}
                    if vl is not None:
                        eval_metrics["eval/best_val"] = best_val
                    report.log(eval_metrics, step=step)

            if args.sample_every and step % args.sample_every == 0:
                # Rank 0 generates through the native path; everyone else waits
                # at the barrier so the next step's collectives stay lockstep.
                if is_main(rank):
                    net.eval()
                    net.apply_to_native()  # make generation reflect the adapter
                    with torch.inference_mode():
                        txt = sample(model, cache, tokenizer, generator,
                                     build_prompt, args.sample_prompt)
                    net.remove_from_native()
                    net.train()
                    print(f"\n  \U0001f3ad  [step {step}] {args.sample_prompt}\n     -> {txt}\n")
                dist.barrier()

            if args.save_every and step % args.save_every == 0:
                save(f"[checkpoint step {step}]")

            if args.checkpoint_every and step % args.checkpoint_every == 0:
                # Retained per-step checkpoint. Single writer (every rank holds
                # identical adapters after the all-reduce), barrier so no rank
                # races ahead of the write/prune.
                if is_main(rank):
                    cdir = checkpoint_dir(args.out, step)
                    net.save_adapter(cdir, base_model_name_or_path=args.model)
                    save_trainer_state(cdir, step=step, opt=opt, sched=sched,
                                       best_val=best_val, best_val_step=best_val_step,
                                       ema=ema)
                    print(f"  [checkpoint] step {step} -> {cdir} (resumable)")
                    prune_checkpoints(args.out, args.keep_checkpoints)
                dist.barrier()

            # Reached only if the whole iteration ran: this step is now durable.
            done_step = step
    except KeyboardInterrupt:
        status = "interrupted"
        # Roll back to the last completed step so the saved counter, the
        # scheduler and the data stream agree. Every rank breaks out of the same
        # iteration, so this stays consistent across ranks.
        step = done_step
        # Don't clobber the best-val checkpoint with the current (later, likely
        # worse) weights when --save-best is on; it's already saved.
        if args.save_best and val_examples:
            if is_main(rank):
                print(f"\nInterrupted at step {step}; keeping best-val adapter.")
        else:
            if is_main(rank):
                print(f"\nInterrupted at step {step}; saving.")
            if step > 0:
                save("[interrupted]")

    # With --save-best the best-val checkpoint is already saved; don't clobber it.
    if not (args.save_best and val_examples):
        save("Done.")

    # Held-out loss (all ranks compute identically; rank 0 reports) + global
    # throughput (sum supervised tokens across ranks) + this rank's peak VRAM.
    _FAIL_CTX["phase"] = "final_eval"
    dt = time.time() - t0
    # Reuse the last in-loop eval if it landed on the final step (all ranks share
    # it, computed in lockstep) instead of a duplicate full pass after the loop.
    if last_eval_step == step:
        val_loss, val2_loss = last_val, last_eval2
    else:
        if is_main(rank) and (val_examples or val2_examples):
            print(" -- computing final held-out eval (GPU busy, not hung) ...")
        val_loss = evaluate()
        val2_loss = eval_loss(val2_examples)
    tok_t = torch.tensor([float(tok_seen), float(tot_seen)], device=device)
    dist.all_reduce(tok_t, op=dist.ReduceOp.SUM)
    if is_main(rank):
        if val_loss is not None:
            tag = f" (best kept: {best_val:.4f})" if args.save_best else ""
            print(f"\n[EVAL] held-out loss (EXL3 arm, DDP): {val_loss:.4f}{tag} "
                  f"over {len(val_examples)} examples")
        if val2_loss is not None:
            print(f"[EVAL] eval2 ({eval2_label}) loss: {val2_loss:.4f} "
                  f"over {len(val2_examples)} examples")
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[PERF] {tok_t[0].item() / dt if dt else 0:,.0f} sup tok/s, "
              f"{tok_t[1].item() / dt if dt else 0:,.0f} tot tok/s (all ranks) | "
              f"peak VRAM/GPU {peak_gb:.2f} GB | {dt:.0f}s, "
              f"{step} steps, world_size {world_size} | "
              f"step time: {timer.summary()}")
    # Record the run (rank 0 only, inside log_run) with all-reduced tok/s totals.
    log_run(status, dt, val_loss, val2_loss,
            tok_t[0].item() / dt if dt else 0, tok_t[1].item() / dt if dt else 0)
    if report is not None:
        final_summary = {k: v for k, v in {
            "end_loss": end_loss, "final_val": val_loss, "final_eval2": val2_loss,
            "best_val": best_val if best_val != float("inf") else None,
            "best_val_step": best_val_step or None,
            "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1e9,
            "sup_tok_s": tok_t[0].item() / dt if dt else 0,
            "tot_tok_s": tok_t[1].item() / dt if dt else 0,
            "total_s": dt, "steps_done": step,
        }.items() if v is not None}
        report.update_summary(final_summary)
        _finish_report(status=status)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# First-run checklist on a real multi-GPU box (this script is untested on HW):
#   * `nvidia-smi` shows all N GPUs busy and at similar memory.
#   * Loss matches a single-GPU run with the same EFFECTIVE batch
#     (batch * nproc_per_node) -- if DDP loss is ~N x too small or large, the
#     grad averaging / loss reduction is off.
#   * Disjoint shards: rank logs should show ~total/N examples per rank.
#   * Adapter saved by rank 0 loads + steers in qlora_infer_native.py exactly
#     like the single-GPU adapter.
#   * If NCCL hangs at init, set NCCL_DEBUG=INFO and check the interface/port.
# ---------------------------------------------------------------------------
