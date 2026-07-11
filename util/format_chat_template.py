import argparse
from transformers import AutoTokenizer


def add_bool_template_arg(parser, name, help_true, help_false):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest = name, action = "store_true", default = None, help = help_true)
    group.add_argument(f"--no-{name}", dest = name, action = "store_false", help = help_false)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    messages = [
        {
            "role": "system",
            "content": "SYSTEM_MESSAGE_HERE",
        },
        {
            "role": "user",
            "content": "USER_MESSAGE_HERE",
        },
    ]

    template_args = {}
    if args.enable_thinking is not None:
        template_args["enable_thinking"] = args.enable_thinking
    if args.add_generation_prompt is not None:
        template_args["add_generation_prompt"] = args.add_generation_prompt

    context = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        **template_args
    )
    token_ids = tokenizer.encode(context, add_special_tokens = False)

    print(context)
    print()
    print("Token IDs:")
    for token_id in token_ids:
        vocab_entry = tokenizer.convert_ids_to_tokens(token_id)
        print(f"{token_id:6}: {repr(vocab_entry)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type = str, help = "Path to model directory")
    add_bool_template_arg(
        parser,
        "enable_thinking",
        "Pass enable_thinking=true to the chat template",
        "Pass enable_thinking=false to the chat template",
    )
    add_bool_template_arg(
        parser,
        "add_generation_prompt",
        "Pass add_generation_prompt=true to the chat template",
        "Pass add_generation_prompt=false to the chat template",
    )
    _args = parser.parse_args()
    main(_args)
