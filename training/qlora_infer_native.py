"""
Native (no-transformers) before/after demo for a QLoRA adapter on EXL3.

The HF Transformers integration is only needed for *training* (autograd).
For inference we use exllamav3's own forward -- which works correctly on the
quantized weights -- and its native PEFT LoRA loader, sidestepping the HF
integration and any transformers version issues entirely.

If the run also trained the embedding / LM head (--lora-head, --lora-embed,
--train-head, --train-embeddings), those are saved beside the adapter in
lora_modules.safetensors / modules_to_save.safetensors, and LoRA.from_directory
applies them here automatically (head LoRA via the LM-head's runtime LoRA slot,
a fully-trained head via a full-weight override, embed deltas folded into the
embedding weight). So this before/after demo reflects a trained head/embed too,
not just the per-linear LoRA.

Usage:
    # base model only (no adapter loaded at all -- a true control)
    python training/qlora_infer_native.py --model /path/to/exl3_model

    # multi-arm comparison: one model load, base + N adapters, same seed
    python training/qlora_infer_native.py \
        --model /path/to/exl3_model \
        --adapter out/run_a/checkpoint-00000160 out/run_b/final \
        --prompt-format jinja --seed 1337
"""

import argparse
import os
from exllamav3 import Config, Model, Cache, Tokenizer, Generator
from exllamav3.model.lora import LoRA
from exllamav3.generator.sampler import ComboSampler

from qlora_train_native import format_prompt_and_eot


# General everyday prompts: a style adapter is most convincing on plain,
# open-ended questions, where the learned voice (e.g. florid Shakespearean
# English) colours an ordinary answer rather than a meta/refusal one.
PROMPTS = [
    "Tell me about your day.",
    "Give me some advice about love.",
    "Explain how the water cycle works.",
    "What should I have for dinner tonight?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mtp", action="store_true",
                    help="Also load the model's MTP (multi-token prediction) head "
                         "and generate WITH it as the self-speculative draft. Each "
                         "--adapter is loaded onto the head too (its mtp.* tensors, "
                         "if the adapter trained the head via --mtp-targets), so "
                         "the arms show the drafted output and the per-arm "
                         "acceptance rate -- the number a retrained head is meant "
                         "to move. Outputs are identical with or without a draft; "
                         "only speed / acceptance differ.")
    ap.add_argument("--num-draft-tokens", type=int, default=None,
                    help="(--mtp) draft window per verify step (default: the head's "
                         "own default, 4 for Qwen3.5).")
    ap.add_argument("--adapter", nargs="*", default=None,
                    help="Adapter directories, each generated as a separate arm off "
                         "ONE model load (unloaded between arms). Omit entirely for a "
                         "base-model-only run -- that is a genuinely untouched control, "
                         "unlike --lora-scaling 0, which still walks the embed-fold and "
                         "full-weight-override paths.")
    ap.add_argument("--prompt-format", default="llama3",
                    help="Chat template the adapter was trained with (same choices "
                         "as the trainer: jinja/auto/mistral/metharme/gemma4-nothink/"
                         "llama3/qwen3.5/qwen3.5-nothink). Default llama3 matches "
                         "this script's original hardcoded template. Use 'jinja' to "
                         "render the model's OWN chat template -- the only correct "
                         "choice for a run trained with prompt_format: jinja.")
    ap.add_argument("--chat-template-file", default=None,
                    help="Jinja template file for --prompt-format jinja. Defaults to "
                         "the model directory's own chat_template.jinja. Must match "
                         "the trainer's chat_template_file exactly.")
    ap.add_argument("--module-lora-scale", type=float, default=None,
                    help="Override the embed/head LoRA scale (alpha/sqrt(r) under "
                         "rslora, else alpha/r). Only needed for adapters saved before "
                         "save_adapter recorded 'module_lora_scale' in the config -- "
                         "for those, a pissa export leaves the config describing the "
                         "CONVERTED rank, so the loader's fallback guess is wrong.")
    ap.add_argument("--prompts-file", default=None,
                    help="File of prompts, one per line (blank lines and #-comments "
                         "ignored). Avoids shell-quoting long question sets.")
    ap.add_argument("--use-per-device", type=float, nargs="*", default=None,
                    help="Split the model across GPUs with these per-device VRAM "
                         "caps in GB (same semantics as the trainer's "
                         "use_per_device). Default: load fully on cuda:0.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--lora-scaling", type=float, default=1.0,
                    help="Extra multiplier on the adapter (on top of alpha/r). "
                         ">1 amplifies the learned style to make a subtle adapter visible.")
    ap.add_argument("--prompts", nargs="*", default=None,
                    help="Custom prompts (default: a content-rich built-in set)")
    # Sampling. The library default is temp 0.8 + min_p 0.08, which truncates the
    # low-probability tail -- if a style's markers are sparse/rare, they get cut.
    # Raise --temperature and set --min-p 0 to test whether the style is in the
    # tail (vs not learned). --temperature 0 = greedy/argmax.
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--min-p", type=float, default=0.08)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None,
                    help="Sampling seed for reproducible before/after comparison")
    ap.add_argument("--gen-out", default=None,
                    help="Write the ADAPTED generations to this jsonl ('output' "
                         "field) for experiments/score_style_density.py (symmetric with the "
                         "BNB arm's --gen-out). Use --temperature 0 for greedy.")
    args = ap.parse_args()

    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f
                       if ln.strip() and not ln.lstrip().startswith("#")]
    else:
        prompts = args.prompts or PROMPTS
    sampler = ComboSampler(
        temperature=args.temperature, min_p=args.min_p,
        top_p=args.top_p, top_k=args.top_k,
    )

    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    if args.use_per_device:
        model.load(use_per_device=args.use_per_device, progressbar=True)
    else:
        model.load(device="cuda:0", progressbar=True)
    tokenizer = Tokenizer.from_config(config)
    # --mtp: the model's own draft head as a second component model on the
    # trunk's output device (it borrows the trunk's embedding + LM head via
    # attach_to, which the Generator performs). Same wiring as the realtime
    # demo's --mtp and eval/spec_decode.py.
    draft_model, draft_cache = None, None
    if args.mtp:
        if "mtp" not in config.model_classes:
            raise SystemExit(f"--mtp: {config.architecture} defines no MTP head")
        draft_model = Model.from_config(config, component="mtp")
        draft_cache = Cache(draft_model, max_num_tokens=4096)
        draft_model.load(device=str(model.output_device) if args.use_per_device
                         else "cuda:0", progressbar=True)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                          draft_model=draft_model, draft_cache=draft_cache,
                          num_draft_tokens=args.num_draft_tokens)
    ctf = args.chat_template_file
    if args.prompt_format == "jinja" and ctf is None:
        cand = os.path.join(args.model, "chat_template.jinja")
        if os.path.exists(cand):
            ctf = cand
    build_prompt, eot = format_prompt_and_eot(model, tokenizer, args.prompt_format,
                                              chat_template_file=ctf)
    if args.prompt_format == "jinja":
        print(f" -- jinja template: {ctf or '(from model config)'}")
        print(f" -- derived eot: {eot!r}")

    # Stop at end-of-turn so the demo shows one clean answer instead of running
    # past <|eot_id|> into hallucinated new assistant turns.
    # Filter Nones: a checkpoint whose generation_config.json defines no
    # eos_token_id (Trinity-Nano) yields eos_token_id_list == [None], which
    # the Job constructor rejects.
    stop = [t for t in (getattr(config, "eos_token_id_list", None) or [])
            if t is not None]
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in stop:
        stop.append(tokenizer.eos_token_id)
    stop += ["<|eot_id|>", "<|start_header_id|>", "</s>", "<|im_end|>",
             # gpt-oss/Onyx-style channels: <|eot|> ends the turn, <|eom|> ends a
             # non-final message (a to=self reasoning block). Stopping at <|eom|>
             # is deliberate -- it makes "the model opened a thinking channel
             # instead of answering" visible instead of hiding it mid-transcript.
             "<|eot|>", "<|eom|>", "<|start|>"]
    if eot and eot not in stop:
        stop.append(eot)

    def run(label: str, dump=None):
        print("=" * 70)
        print(label)
        print("=" * 70)
        dacc = drej = 0
        for p in prompts:
            resp, last = generator.generate(
                prompt=build_prompt(p),
                max_new_tokens=args.max_new_tokens,
                sampler=sampler,
                seed=args.seed,
                add_bos=False,
                completion_only=True,
                stop_conditions=stop,
                return_last_results=True,
            )
            print(f"\n> {p}\n{resp}")
            if dump is not None:
                dump.append({"instruction": p, "input": "", "output": resp})
            # Draft bookkeeping from the job's final result (the same fields
            # eval/spec_decode.py aggregates): every drafted position is
            # counted as accepted or rejected.
            dacc += int(last.get("accepted_draft_tokens", 0) or 0)
            drej += int(last.get("rejected_draft_tokens", 0) or 0)
        if draft_model is not None:
            fed = dacc + drej
            print(f"[mtp] draft acceptance over this arm: {dacc}/{fed} = "
                  f"{(dacc / fed if fed else 0.0):.1%}")
        print()

    # Arm 1 is always the untouched base: no adapter is ever loaded before this
    # point, so it is a true control rather than a zero-scaled adapter.
    dumps = {}
    dump = [] if args.gen_out else None
    run("ARM: no adapter (base model)", dump=dump)
    if dump is not None:
        dumps["base"] = dump

    # Remaining arms: load, generate, unload. One model load for all of them, and
    # the same --seed per prompt, so any difference between arms is the adapter.
    for adapter_dir in (args.adapter or []):
        lora = LoRA.from_directory(model, adapter_dir,
                                   lora_scaling=args.lora_scaling,
                                   module_lora_scale=args.module_lora_scale)
        # The head's adapters (mtp.* keys) live in the same dir; the loader
        # matches keys by module suffix, so given the draft model it applies
        # exactly those and skips the trunk's (and vice versa above).
        lora_mtp = (LoRA.from_directory(draft_model, adapter_dir,
                                        lora_scaling=args.lora_scaling)
                    if draft_model is not None else None)
        dump = [] if args.gen_out else None
        run(f"ARM: {adapter_dir}", dump=dump)
        if dump is not None:
            dumps[adapter_dir] = dump
        lora.unload()
        if lora_mtp is not None:
            lora_mtp.unload()

    if args.gen_out:
        import json
        with open(args.gen_out, "w", encoding="utf-8") as f:
            for arm, rows in dumps.items():
                for r in rows:
                    f.write(json.dumps({"arm": arm, **r}, ensure_ascii=False) + "\n")
        print(f"Generations for {len(dumps)} arm(s) written to {args.gen_out}")


if __name__ == "__main__":
    main()
