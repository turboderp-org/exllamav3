import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3 import Config, Model, Cache, Tokenizer, DefaultSampler
from exllamav3.util import Timer
from common import format_prompt, get_stop_conditions
import torch

"""
This script demonstrates a minimal, cached generation pipeline, starting with tokenization of a prompt, prefill
and then token-by-token sampling from logits produced by iterative forward passes through the model. For most
applications the built-in generator offers more flexibility, though. 
"""

# Load model
config = Config.from_directory("/mnt/str/models/llama3.1-8b-instruct/exl3/4.0bpw/")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 2048)
model.load()

# Load tokenizer
tokenizer = Tokenizer.from_config(config)

# Prepare inputs
prompt_format = "llama3"
prompt_text = format_prompt(
    prompt_format,
    "You are a super helpful language model.",
    "List five ways in which cats are superior to dogs."
)
context_ids = tokenizer.encode(prompt_text, encode_special_tokens = True)

# Sampling and stop conditions
sampler = DefaultSampler()
stop_conditions = get_stop_conditions(prompt_format, tokenizer)

# Get model vocabulary as a list of strings, for streaming the completion
vocab = tokenizer.get_id_to_piece_list()

# Prefill the prompt, up to but not including the last token, which will be the first token forwarded in the
# generation loop. Treat the cache as a rectangular batch
model.prefill(
    input_ids = context_ids[:, :-1],
    params = {
        "attn_mode": "flash_attn",
        "cache": cache,
        "past_len": 0,
        "batch_shape": (1, 2048),
    }
)

# Generation loop
max_new_tokens = 500
generated_tokens = 0
response = ""

torch.cuda.synchronize()
with Timer() as t:
    while generated_tokens < max_new_tokens:

        # Get logits for current position
        logits = model.forward(
            input_ids = context_ids[:, -1:],
            params = {
                "attn_mode": "flash_attn",
                "cache": cache,
                "past_len": context_ids.shape[-1] - 1,
                "batch_shape": (1, 2048),
            }
        )

        # Sample from logits
        sample = sampler.forward(logits, tokenizer = tokenizer)
        token_id = sample.item()

        # Detect end of stream
        if token_id in stop_conditions:
            break

        # Append sampled token to context
        context_ids = torch.cat((context_ids, sample.cpu()), dim = -1)
        token = vocab[token_id]
        response += token
        generated_tokens += 1

        # Stream to the console
        print(token, end = "", flush = True)

print()
print("---")
print(f"{generated_tokens} tokens at {generated_tokens/t.interval:.3f} tokens/second")