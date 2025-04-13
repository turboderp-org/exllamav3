import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from exllamav3 import Generator, Job, model_init
from chat_templates import *
import torch
from chat_console import *

thinktag = ("<think>", "</think>")

@torch.inference_mode()
def main(args):

    # Prompt format
    if args.modes:
        print("Available modes:")
        for k, v in prompt_formats.items():
            print(f" - {k:16} {v.description}")
        return

    user_name = args.user_name
    bot_name = args.bot_name
    prompt_format = prompt_formats[args.mode](user_name, bot_name)
    system_prompt = prompt_format.default_system_prompt() if not args.system_prompt else args.system_prompt
    add_bos = prompt_format.add_bos()
    max_response_tokens = args.max_response_tokens

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

    # Main loop
    print("\n" + col_sysprompt + system_prompt.strip() + col_default)
    context = []

    while True:

        # Get user prompt and add to context
        user_prompt = read_input_fn(args, user_name)
        context.append((user_prompt, None))

        # Tokenize context and trim from head if too long
        def get_input_ids():
            frm_context = prompt_format.format(system_prompt, context)
            if args.think:
                frm_context += thinktag[0]
            ids_ = tokenizer.encode(frm_context, add_bos = add_bos, encode_special_tokens = True)
            exp_len_ = ids_.shape[-1] + max_response_tokens + 1
            return ids_, exp_len_

        ids, exp_len = get_input_ids()
        if exp_len > context_length:
            while exp_len > context_length - 2 * max_response_tokens:
                context = context[1:]
                ids, exp_len = get_input_ids()

        # Inference
        job = Job(
            input_ids = ids,
            max_new_tokens =  max_response_tokens,
            stop_conditions = stop_conditions
        )
        generator.enqueue(job)

        # Stream response
        ctx_exceeded = False
        with streamer_cm(args, bot_name) as s:
            while generator.num_remaining_jobs():
                for r in generator.iterate():
                    chunk = r.get("text", "")
                    s.stream(chunk, thinktag[1])
                    if r["eos"] and r["eos_reason"] == "max_new_tokens":
                        ctx_exceeded = True

        if ctx_exceeded:
            print(
                "\n" + col_error + f" !! Response exceeded {max_response_tokens} tokens "
                "and was cut short." + col_default
            )

        # Add response to context
        response = s.all_text.strip()

        context[-1] = (user_prompt, response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = True)
    parser.add_argument("-mode", "--mode", type = str, help = "Prompt mode", required = True)
    parser.add_argument("-modes", "--modes", action = "store_true", help = "List available prompt modes and exit")
    parser.add_argument("-un", "--user_name", type = str, default = "User", help = "User name (raw mode only)")
    parser.add_argument("-bn", "--bot_name", type = str, default = "Assistant", help = "Bot name (raw mode only)")
    parser.add_argument("-mli", "--multiline", action = "store_true", help = "Enable multi line input (use Alt+Enter to submit input)")
    parser.add_argument("-sp", "--system_prompt", type = str, help = "Use custom system prompt")
    parser.add_argument("-maxr", "--max_response_tokens", type = int, default = 1000, help = "Max tokens per response, default = 1000")
    parser.add_argument("-basic", "--basic_console", action = "store_true", help = "Use basic console output (no markdown and fancy prompt input")
    parser.add_argument("-rps", "--refresh_per_second", type = int, help = "Max updates per second in Markdown mode, default = 25", default = 25)
    parser.add_argument("-think", "--think", action = "store_true", help = "Use (very simplistic) reasoning template and formatting")
    # TODO: Sampling options
    _args = parser.parse_args()
    main(_args)
