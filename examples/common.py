
"""
Quick and dirty and probably not very accurate prompt templates for a couple of models
"""

def format_prompt(prompt_format, sp, p):

    match prompt_format:

        case "llama":
            return f"<s>[INST] <<SYS>>\n{sp}\n<</SYS>>\n\n{p} [/INST]"

        case "llama3":
            return (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{sp}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{p}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        case "mistral":
            return f"<s>[INST] {sp}\n\n n{p}[/INST]"

        case "granite":
            return (
                f"System:\n"
                f"{sp}\n\n"
                f"Question:\n"
                f"{p}\n\n"
                f"Answer:\n"
            )

        case "chatml":
            return (
                f"<|im_start|>system\n"
                f"{sp}<|im_end|>\n"
                f"<|im_start|>user\n"
                f"{p}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        case "gemma":
            return (
                f"<bos><start_of_turn>user\n"
                f"{p}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )

        case _:
            raise ValueError("Unknown prompt format")


def get_stop_conditions(prompt_format, tokenizer):

    match prompt_format:
        case "llama":
            return [tokenizer.eos_token_id]
        case "llama3":
            return [tokenizer.single_id("<|eot_id|>")]
        case "granite":
            return [tokenizer.eos_token_id, "\n\nQuestion:"]
        case "gemma":
            return [tokenizer.eos_token_id, "<end_of_turn>"]
        case "chatml":
            return [tokenizer.eos_token_id, "<|im_end|>"]
