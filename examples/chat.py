import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from exllamav3 import Generator, Job, model_init
from exllamav3.generator.sampler import ComboSampler
from chat_templates import *
from chat_util import *
from chat_io import *
import torch
from chat_console import *

@torch.inference_mode()
def main(args):

    # Prompt format
    if args.modes or args.mode is None:
        print("Available modes:")
        for k, v in prompt_formats.items():
            print(f" - {k:16} {v.description}")
        return

    user_name = args.user_name
    bot_name = args.bot_name
    prompt_format = prompt_formats[args.mode](user_name, bot_name)
    system_prompt = prompt_format.default_system_prompt(args.think) if not args.system_prompt else args.system_prompt
    add_bos = prompt_format.add_bos()
    max_response_tokens = args.max_response_tokens
    multiline = args.multiline
    show_tps = args.show_tps

    if args.basic_console:
        read_input_fn = read_input_ptk
        streamer_cm = Streamer_basic
    else:
        read_input_fn = read_input_ptk
        streamer_cm = Streamer_rich

    # Load model
    model, config, cache, tokenizer = model_init.init(args)
    context_length = cache.max_num_tokens

    # Generator
    generator = Generator(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
    )
    stop_conditions = prompt_format.stop_conditions(tokenizer)

    # Sampler
    sampler = ComboSampler(
        rep_p = args.repetition_penalty,
        pres_p = args.presence_penalty,
        freq_p = args.frequency_penalty,
        rep_sustain_range = args.penalty_range,
        rep_decay_range = args.penalty_range,
        temperature = args.temperature,
        min_p = args.min_p,
        top_k = args.top_k,
        top_p = args.top_p,
        temp_last = not args.temperature_first,
    )

    # Single prompt mode
    single_prompt = args.prompt

    # Main loop
    print("\n" + col_sysprompt + system_prompt.strip() + col_default)
    context = []
    tt = prompt_format.thinktag()
    banned_strings = [tt[0], tt[1]] if args.no_think else []
    response = ""

    while True:

        # Amnesia mode
        if args.amnesia:
            context = []

        # Get user prompt
        if single_prompt is not None:
            # This round, use provided prompt from cmdline
            user_prompt = single_prompt
            prefix = ""
            # Next round, exit
            single_prompt = "/x"
        else:
            try:
                user_prompt = read_input_fn(args, user_name, multiline)
                prefix = ""
            except KeyboardInterrupt:
                user_prompt = "/x"

        # Intercept commands
        if user_prompt.startswith("/"):
            c = user_prompt.strip().split(" ")
            match c[0]:

                # Exit app
                case "/x":
                    print_info("Exiting")
                    break

                # Copy codeblock to clipboard
                case "/cc":
                    try:
                        b = int(c[1])
                    except:
                        b = 1
                    snippet = copy_last_codeblock(response, b)
                    if not snippet:
                        print_error("No code block found in last response")
                    else:
                        num_lines = len(snippet.split("\n"))
                        print_info(f"Copied {num_lines} line{'s' if num_lines > 1 else ''} to the clipboard")
                    continue

                # Toggle multiline mode
                case "/mli":
                    multiline = not multiline
                    if multiline:
                        print_info("Enabled multiline mode")
                    else:
                        print_info("Disabled multiline mode")
                    continue

                # Clear context
                case "/clear":
                    context = []
                    print_info("Cleared context")
                    continue

                # Toggle TPS
                case "/tps":
                    multiline = not multiline
                    if multiline:
                        print_info("Enabled tokens/second output")
                    else:
                        print_info("Disabled tokens/second output")
                    continue

                # Retry last response
                case "/r":
                    user_prompt = context[-1][0]
                    context = context[:-1]

                # Edit last response
                case "/e":
                    print_info("Press Alt+Enter to submit")
                    user_prompt = context[-1][0]
                    last_reply = context[-1][-1]
                    prefix = read_input_fn(args, bot_name, True, last_reply)
                    context = context[:-1]

                # Edit system prompt
                case "/sp":
                    print_info("Press Alt+Enter to submit")
                    system_prompt = read_input_fn(args, "System prompt", True, system_prompt)
                    continue

                # Edit banned strings
                case "/ban":
                    print_info("Write each string on a new line and enclose in \"double quotes\", press Alt+Enter to submit")
                    bans = "\n".join(f"\"{b}\"" for b in banned_strings)
                    bans = read_input_fn(args, "Banned strings", True, bans)
                    bans = [b.strip() for b in bans.split("\n")]
                    bans = [b[1:-1] for b in bans if b.startswith("\"") and b.endswith("\"")]
                    d = len(bans) - len(banned_strings)
                    banned_strings = bans
                    if d < 0:
                        print_info(f"{-d} string(s) removed")
                    elif d > 0:
                        print_info(f"{d} string(s) added")
                    else:
                        print_info("Strings updated")
                    continue

                # Save conversation
                case "/save":
                    if len(c) == 1:
                        c.append("~/chat_py_session.json")
                    save_session(c[1], system_prompt, banned_strings, context)
                    print_info(f"Saved session to: {c[1]}")
                    continue

                # Load conversation
                case "/load":
                    if len(c) == 1:
                        c.append("~/chat_py_session.json")
                        try:
                            (
                                system_prompt,
                                banned_strings,
                                context
                            ) = load_session(c[1])
                            print_info(f"Loaded session from: {c[1]}")
                        except:
                            print_error(f"Error loading {c[1]}")
                        continue

                case _:
                    print_error(f"Unknown command: {c[0]}")
                    continue

        # Add to context
        context.append((user_prompt, None))

        # Tokenize context and trim from head if too long
        def get_input_ids(_prefix):
            frm_context = prompt_format.format(system_prompt, context)
            if _prefix:
                frm_context += prefix
            elif args.think:
                frm_context += prompt_format.thinktag()[0]
            ids_ = tokenizer.encode(frm_context, add_bos = add_bos, encode_special_tokens = True)
            exp_len_ = ids_.shape[-1] + max_response_tokens + 1
            return ids_, exp_len_

        ids, exp_len = get_input_ids(prefix)
        if exp_len > context_length:
            while exp_len > context_length - 2 * max_response_tokens:
                context = context[1:]
                ids, exp_len = get_input_ids(prefix)

        # Inference
        job = Job(
            input_ids = ids,
            max_new_tokens =  max_response_tokens,
            stop_conditions = stop_conditions,
            sampler = sampler,
            banned_strings = banned_strings
        )
        generator.enqueue(job)

        # Stream response
        ctx_exceeded = False
        with streamer_cm(args, bot_name) as s:
            if prefix:
                s.stream(prefix, tt[0], tt[1])
            while generator.num_remaining_jobs():
                for r in generator.iterate():
                    chunk = r.get("text", "")
                    s.stream(chunk, tt[0], tt[1])
                    if r["eos"] and r["eos_reason"] == "max_new_tokens":
                        ctx_exceeded = True

        if ctx_exceeded:
            print(
                "\n" + col_error + f" !! Response exceeded {max_response_tokens} tokens "
                "and was cut short." + col_default
            )

        if show_tps:
            prompt_tokens = ids.shape[-1]
            cached_tokens = r["cached_tokens"]
            new_ctx_tokens = prompt_tokens - cached_tokens
            prompt_tps = new_ctx_tokens / r["time_prefill"]
            new_tokens = r["new_tokens"]
            tps = new_tokens / r["time_generate"]
            print(
                "\n"
                f"Context: {col_info}{new_ctx_tokens:,}{col_default} new tokens at {col_info}{prompt_tps:.3f}{col_default} t/s - "
                f"{col_info}{cached_tokens:,}{col_default} tokens cached - "
                f"Generate: {col_info}{new_tokens:,}{col_default} tokens at {col_info}{tps:.3f}{col_default} t/s"
            )

        # Add response to context
        response = s.all_text.strip()

        context[-1] = (user_prompt, response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = True)
    parser.add_argument("-mode", "--mode", type = str, help = "Prompt mode", default = None)
    parser.add_argument("-modes", "--modes", action = "store_true", help = "List available prompt modes and exit")
    parser.add_argument("-un", "--user_name", type = str, default = "User", help = "User name (raw mode only)")
    parser.add_argument("-bn", "--bot_name", type = str, default = "Assistant", help = "Bot name (raw mode only)")
    parser.add_argument("-mli", "--multiline", action = "store_true", help = "Enable multi line input (use Alt+Enter to submit input)")
    parser.add_argument("-sp", "--system_prompt", type = str, help = "Use custom system prompt")
    parser.add_argument("-maxr", "--max_response_tokens", type = int, default = 1000, help = "Max tokens per response, default = 1000")
    parser.add_argument("-basic", "--basic_console", action = "store_true", help = "Use basic console output (no markdown and fancy prompt input")
    parser.add_argument("-think", "--think", action = "store_true", help = "Use (very simplistic) reasoning template and formatting")
    parser.add_argument("-no_think", "--no_think", action = "store_true", help = "Suppress think tags (won't necessarily stop reasoning model from reasoning anyway)")
    parser.add_argument("-amnesia", "--amnesia", action = "store_true", help = "Forget context with every new prompt")
    parser.add_argument("-temp", "--temperature", type = float, help = "Sampling temperature", default = 0.8)
    parser.add_argument("-temp_first", "--temperature_first", action = "store_true", help = "Apply temperature before truncation")
    parser.add_argument("-repp", "--repetition_penalty", type = float, help = "Repetition penalty, HF style, 1 to disable (default: disabled)", default = 1.0)
    parser.add_argument("-presp", "--presence_penalty", type = float, help = "Presence penalty, 0 to disable (default: disabled)", default = 0.0)
    parser.add_argument("-freqp", "--frequency_penalty", type = float, help = "Frequency penalty, 0 to disable (default: disabled)", default = 0.0)
    parser.add_argument("-penr", "--penalty_range", type = int, help = "Range for penalties, in tokens (default: 1024) ", default = 1024)
    parser.add_argument("-minp", "--min_p", type = float, help = "Min-P truncation, 0 to disable (default: 0.08)", default = 0.08)
    parser.add_argument("-topk", "--top_k", type = int, help = "Top-K truncation, 0 to disable (default: disabled)", default = 0)
    parser.add_argument("-topp", "--top_p", type = float, help = "Top-P truncation, 1 to disable (default: disabled)", default = 1.0)
    parser.add_argument("-tps", "--show_tps", action = "store_true", help = "Show tokens/second after every reply")
    parser.add_argument("-prompt", "--prompt", type = str, help = "Run single prompt, then exit")
    _args = parser.parse_args()
    main(_args)
