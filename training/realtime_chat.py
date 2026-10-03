"""
Interactive demo of real-time (inference-time) QLoRA training.

One loaded EXL3 model both chats and learns: generation runs through the
normal exllamav3 Generator with the adapter live in the runtime LoRA slots,
and any sample you feed it trains the same adapter in place
(:class:`exllamav3.training.RealtimeQLoRA`) -- no reload, no second model
copy. The page table is nuked on every adapter update (the config-option
default; see the realtime module docstring for the adapter-free-KV
alternative), so the next generation runs with fresh state under the updated
weights.

Usage:
    python training/realtime_chat.py --model /path/to/exl3-model \
        --checkpoint-dir out/realtime_demo

Chat normally at the prompt. Commands:

    /learn <text>      Replace the assistant's LAST reply with <text> and
                       train on the corrected conversation (the whole point:
                       correct the model and watch the correction stick).
    /prefer <text>     DPO: train on one preference pair -- <text> is the
                       CHOSEN reply, the assistant's last reply the REJECTED
                       one, the conversation before it the prompt. The
                       history then continues with <text>.
    /good, /bad        KTO: label the assistant's last reply desirable /
                       undesirable and train on that single unpaired row
                       (a /bad reply stays in the history; /learn or /prefer
                       to replace it).
    /again             Re-ask the last user message (compare before/after).
    /ingest <file>     Train through a JSONL file of samples; each line is a
                       dict in any ingest form: {"messages": [...]},
                       {"prompt", "response"}, {"text"}, pre-tokenized, or
                       the TRL explicit-prompt preference forms
                       {"prompt", "chosen", "rejected"} (DPO) and
                       {"prompt", "completion", "label"} (KTO) -- prompt as
                       a message list (rendered through the chat template)
                       or a rendered string, completions as strings or
                       assistant message lists.
    /lr <value>        Set the (constant) learning rate, e.g. /lr 5e-5.
    /checkpoint        Write a timestamped adapter checkpoint now.
    /unload            Remove the adapter from generation (compare to base).
    /reload            Re-apply the adapter to generation.
    /reset             Clear the conversation history.
    /stats             Optimizer step count / samples seen / lr.
    /quit              Exit.

Prompts are rendered through the model directory's own Jinja chat template
(the same one inference servers use), and `/learn` / `/ingest` messages
samples train with exact per-turn loss masks via the same template
(training/chat_jinja.py) -- assistant turns supervised, everything else
masked. Preference samples (`/prefer`, `/good`, `/bad`, the DPO/KTO ingest
forms) render their prompt through the same template with the generation
prompt appended and score only the completion (+ the template's own
turn-close), the frozen base serving as the DPO/KTO reference model (adapters
disabled, no second copy). Preference LRs typically run 10-100x below SFT
(--lr 5e-6 .. 5e-5), and KTO's KL reference point needs --batch >= 2 (only
reachable through /ingest files; the single-row /good and /bad train with
KL = 0, i.e. the apo_zero_unpaired loss). This script is also the reference
wiring for hooking RealtimeQLoRA
into a server backend (e.g. tabbyAPI's backends/exllamav3/model.py): the
pieces marked [integration] below are exactly what a backend needs to add.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exllamav3 import Config, Model, Cache, Tokenizer, Generator  # noqa: E402
from exllamav3.training.realtime import RealtimeQLoRA, RealtimeConfig  # noqa: E402
from chat_jinja import (jinja_renderers, tokenizer_special_tokens,  # noqa: E402
                        extract_rich_turns, row_template_extras,
                        parse_template_vars)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model", required=True, help="EXL3 model directory")
    p.add_argument("--adapter", default=None,
                   help="resume from an adapter/checkpoint directory")
    p.add_argument("--cache-size", type=int, default=8192,
                   help="KV cache size in tokens (default 8192)")
    p.add_argument("--system", default=None, help="system prompt")
    p.add_argument("--template-vars", default=None,
                   help='JSON template vars, e.g. \'{"enable_thinking": false}\'')
    p.add_argument("--max-new-tokens", type=int, default=512)
    # RealtimeConfig knobs (defaults follow the dataclass)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=float, default=32.0)
    p.add_argument("--targets", nargs="*", default=None,
                   help="LoRA target modules (default: all attn+mlp proj; "
                        "exclude k_proj/v_proj for an adapter-free KV cache)")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="constant learning rate (SFT default 1e-4; preference "
                        "training usually wants 5e-6 .. 5e-5)")
    p.add_argument("--batch", type=int, default=1,
                   help="samples per micro-batch (a DPO pair is 2 sequences; "
                        "KTO's KL estimate needs >= 2)")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--head-vocab-chunk", type=int, default=None,
                   help="LM-head loss in vocab-column tiles of this size "
                        "(default: the coordinator's 32768; 0 = reconstruct "
                        "the whole head at once, which spikes ~12 GB on a "
                        "248k-vocab 27B and OOMs a serving card)")
    # preference (DPO / KTO) knobs -- same semantics as qlora_train_pref.py
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO/KTO inverse temperature (default 0.1)")
    p.add_argument("--dpo-loss", default="sigmoid",
                   choices=["sigmoid", "hinge", "ipo"],
                   help="DPO loss variant (sigmoid = standard/cDPO, hinge = "
                        "SLiC, ipo)")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="cDPO label-flip probability (sigmoid loss)")
    p.add_argument("--kto-loss", default="kto",
                   choices=["kto", "apo_zero_unpaired"],
                   help="KTO loss variant (kto = batch-KL reference point, "
                        "needs --batch >= 2; apo_zero_unpaired = no KL term)")
    p.add_argument("--desirable-weight", type=float, default=1.0,
                   help="KTO loss weight on desirable (/good) rows")
    p.add_argument("--undesirable-weight", type=float, default=1.0,
                   help="KTO loss weight on undesirable (/bad) rows")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="checkpoint every N optimizer steps (0 = manual only)")
    p.add_argument("--keep-checkpoints", type=int, default=0,
                   help="prune to newest N checkpoints (0 = keep all)")
    p.add_argument("--no-idle-offload", action="store_true",
                   help="keep the training state (fp32 masters + Adam moments) "
                        "in VRAM between ingests instead of parking it in "
                        "system RAM while only serving")
    p.add_argument("--mtp", action="store_true",
                   help="load the model's MTP head (architectures that have "
                        "one) as a draft model for speculative decoding; it "
                        "is parked out of VRAM during every ingest and "
                        "restored for serving (--no-aux-offload to keep it "
                        "resident)")
    p.add_argument("--no-aux-offload", action="store_true",
                   help="keep aux component models (the --mtp head) in VRAM "
                        "during ingests instead of parking them")
    return p.parse_args()


def main():
    args = parse_args()

    # -- ordinary exllamav3 serving setup (nothing realtime-specific yet) --
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=args.cache_size)
    model.load(progressbar=True)
    tokenizer = Tokenizer.from_config(config)
    draft_model, draft_cache = None, None
    if args.mtp:
        if "mtp" not in config.model_classes:
            raise SystemExit(f" !! --mtp: {config.architecture} does not "
                             f"define an MTP head")
        draft_model = Model.from_config(config, component="mtp")
        draft_cache = Cache(draft_model, max_num_tokens=args.cache_size)
        draft_model.load(progressbar=True)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                          draft_model=draft_model, draft_cache=draft_cache)

    # The model's own chat template drives BOTH prompt rendering for
    # generation and segment masking for training -- one source of truth.
    seg_build, _, eot = jinja_renderers(
        args.model, tokenizer_special_tokens(tokenizer),
        default_vars=parse_template_vars(args.template_vars))

    def render_segments(sample):
        turns = extract_rich_turns(sample["messages"])
        return seg_build(turns, **row_template_extras(sample))

    def render_prompt(sample):
        # A preference sample's conversational prompt: the history rendered
        # WITH the generation prompt, so the separately encoded completion
        # follows the assistant-turn opener exactly as it does at inference.
        turns = extract_rich_turns(sample["prompt"])
        return seg_build(turns, add_generation_prompt=True,
                         **row_template_extras(sample))

    # -- [integration] the realtime coordinator ------------------------------
    rt = RealtimeQLoRA(
        model, tokenizer,
        RealtimeConfig(
            r=args.r, alpha=args.alpha, target_modules=args.targets,
            lr=args.lr, batch_size=args.batch, grad_accum=args.grad_accum,
            seq_len=args.seq_len, checkpoint_dir=args.checkpoint_dir,
            **({} if args.head_vocab_chunk is None
               else {"head_vocab_chunk": args.head_vocab_chunk}),
            checkpoint_every=args.checkpoint_every,
            keep_checkpoints=args.keep_checkpoints,
            offload_when_idle=not args.no_idle_offload,
            offload_aux_when_training=not args.no_aux_offload,
            # the template's own turn-close string closes every prompt/
            # response reply and every preference completion
            eot_text=eot or "",
            beta=args.beta, dpo_loss=args.dpo_loss,
            label_smoothing=args.label_smoothing, kto_loss=args.kto_loss,
            desirable_weight=args.desirable_weight,
            undesirable_weight=args.undesirable_weight),
        adapter_dir=args.adapter,
        render_segments=render_segments,
        render_prompt=render_prompt,
        base_model_name_or_path=args.model)
    rt.attach_generator(generator)   # page table reset on every adapter update
    # [integration] serving-only components (a vision tower would go here too)
    # are parked out of VRAM while each ingest trains, restored after.
    rt.attach_aux_models(draft_model)
    print(f" -- realtime adapter: {rt.net.num_trainable():,} trainable params, "
          f"lr {rt.lr:g}, targets {' '.join(rt.net.target_modules)}")

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    # Stop on EOS and on the template's own turn-close string. The rendered
    # prompt already carries any BOS the template wants (add_bos=False below,
    # same as the offline trainer's live sampling).
    stop = [s for s in (tokenizer.eos_token_id, eot or None)
            if s is not None and s != ""]

    def generate():
        prompt = "".join(t for t, _ in
                         seg_build(extract_rich_turns(messages),
                                   add_generation_prompt=True))
        # [integration] every generator drive holds the inference read lock,
        # so an ingest can drain in-flight requests and block new ones.
        with rt.inference():
            return generator.generate(
                prompt=prompt, max_new_tokens=args.max_new_tokens,
                stop_conditions=stop, completion_only=True, add_bos=False)

    def report(stats):
        if stats.get("skipped"):
            print(f" !! skipped {stats['skipped']} sample(s) whose prompt "
                  f"alone fills --seq-len {args.seq_len}")
        if stats["steps"] == 0:
            print(" -- nothing to train on")
            return
        parts = []
        if stats["mean_loss"] is not None:
            parts.append(f"sft {stats['sft_samples']} sample(s) "
                         f"loss {stats['mean_loss']:.4f}")
        if stats["dpo"]:
            d = stats["dpo"]
            parts.append(f"dpo {d['pairs']} pair(s) loss {d['loss']:.4f} "
                         f"reward acc {d['acc']:.2f} margin {d['margin']:+.3f}")
        if stats["kto"]:
            k = stats["kto"]
            rewards = ", ".join(
                f"{name} {val:+.3f}" for name, val in
                (("reward_d", k["reward_d"]), ("reward_u", k["reward_u"]))
                if val is not None)
            parts.append(f"kto {k['samples']} row(s) loss {k['loss']:.4f} "
                         f"kl {k['kl']:.3f} {rewards}")
        print(f" -- trained: {stats['steps']} step(s), "
              f"{stats['samples']} sample(s), "
              f"{stats['supervised_tokens']} supervised tokens; "
              f"{'; '.join(parts)}; "
              f"{stats['duration_s']:.1f}s (lr {stats['lr']:g}); "
              f"adapter live, cache flushed")
        for c in stats["checkpoints"]:
            print(f" -- checkpoint: {c}")

    def last_exchange():
        """(prompt messages, last assistant reply) for the preference
        commands, or None when there is no reply to judge yet."""
        if not messages or messages[-1]["role"] != "assistant":
            return None
        return list(messages[:-1]), messages[-1]["content"]

    print("Chat away. /learn <corrected reply> trains on a correction (SFT), "
          "/prefer <better reply> on a preference pair (DPO), /good and /bad "
          "label the last reply (KTO); /quit exits; see --help for all "
          "commands.")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            rest = rest.strip()
            if cmd == "/quit":
                break
            elif cmd == "/reset":
                messages = messages[:1] if args.system else []
                print(" -- conversation cleared")
            elif cmd == "/lr":
                try:
                    rt.lr = float(rest)
                    print(f" -- lr = {rt.lr:g}")
                except ValueError:
                    print(" !! usage: /lr 5e-5")
            elif cmd == "/stats":
                print(f" -- step {rt.step}, samples seen {rt.samples_seen}, "
                      f"lr {rt.lr:g}")
            elif cmd == "/checkpoint":
                try:
                    print(f" -- checkpoint: {rt.checkpoint()}")
                except ValueError as e:
                    print(f" !! {e} (pass --checkpoint-dir)")
            elif cmd == "/unload":
                rt.unload_from_inference()
                print(" -- adapter removed from generation (base model)")
            elif cmd == "/reload":
                rt.sync_to_inference()
                print(" -- adapter re-applied to generation")
            elif cmd == "/again":
                if not messages or messages[-1]["role"] != "assistant":
                    print(" !! nothing to retry")
                    continue
                messages.pop()
                response = generate()
                print(response)
                messages.append({"role": "assistant", "content": response})
            elif cmd == "/learn":
                if not rest:
                    print(" !! usage: /learn <corrected assistant reply>")
                    continue
                if not messages or messages[-1]["role"] != "assistant":
                    print(" !! no assistant reply to correct yet")
                    continue
                messages[-1] = {"role": "assistant", "content": rest}
                report(rt.ingest([{"messages": list(messages)}]))
            elif cmd == "/prefer":
                if not rest:
                    print(" !! usage: /prefer <better assistant reply>")
                    continue
                ex = last_exchange()
                if ex is None:
                    print(" !! no assistant reply to prefer against yet")
                    continue
                prompt, rejected = ex
                # TRL explicit-prompt pair: the history is the prompt, the
                # model's own reply is rejected, the user's text is chosen.
                report(rt.ingest([{"prompt": prompt, "chosen": rest,
                                   "rejected": rejected}]))
                messages[-1] = {"role": "assistant", "content": rest}
            elif cmd in ("/good", "/bad"):
                ex = last_exchange()
                if ex is None:
                    print(" !! no assistant reply to label yet")
                    continue
                prompt, completion = ex
                report(rt.ingest([{"prompt": prompt, "completion": completion,
                                   "label": cmd == "/good"}]))
            elif cmd == "/ingest":
                if not os.path.exists(rest):
                    print(f" !! no such file: {rest}")
                    continue
                with open(rest, encoding="utf8") as f:
                    samples = [json.loads(l) for l in f if l.strip()]
                report(rt.ingest(samples))
            else:
                print(f" !! unknown command {cmd}")
            continue

        messages.append({"role": "user", "content": line})
        response = generate()
        print(response)
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
