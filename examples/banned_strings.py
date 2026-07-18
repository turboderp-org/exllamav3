import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job, GreedySampler
from common import format_prompt, get_stop_conditions

"""
A simple showcase of the banned strings feature of the generator, which prevents the model from sampling any of a
predefined set of phrases.   
"""

# Initialize model, tokenizer etc.
config = Config.from_directory("/mnt/str/models/qwen3.5-9b/exl3/4.00bpw_mul1/")
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = 8192)
model.load()
tokenizer = Tokenizer.from_config(config)
generator = Generator(model = model, cache = cache, tokenizer = tokenizer)

# Prompt
prompt_format = "chatml"
prompt = format_prompt(
    prompt_format,
    "You are a helpful AI assistant.",
    "Tell me a story about cats.",
    no_think = True
)
stop_conditions = get_stop_conditions(prompt_format, tokenizer)

# List of some common refusals
banned_strings = [
    "Once upon a time",
    "Barnaby",
    "Whiskers",
    "Luna",
    "Mittens",
    "no ordinary cat",
    "like the other cats"
]

# Generate with and without banned strings
def generate(bs):

    input_ids = tokenizer.encode(prompt, add_bos = False, encode_special_tokens = True)
    job = Job(
        input_ids = input_ids,
        sampler = GreedySampler(),
        min_new_tokens = 100 if bs else 0,  # Prevent model from ending stream too early
        max_new_tokens = 300,
        banned_strings = bs,
        stop_conditions = stop_conditions
    )
    generator.enqueue(job)

    # Stream output to console. Banned strings will not be included in the output stream, but every time a string
    # is suppressed the offending text is returned in the results packet, so we can illustrate what's going on
    col_banned = "\u001b[9m\u001b[31;1m"  # Magenta, strikethrough
    col_default = "\u001b[0m"

    while generator.num_remaining_jobs():
        results = generator.iterate()
        for result in results:
            if "text" in result:
                print(result["text"], end = "", flush = True)
            if "suppressed_text" in result:
                print(col_banned + result["suppressed_text"] + col_default, end = "", flush = True)
    print()

print("--------------------------------------------------------------------------------------")
print("Without banned strings")
print("--------------------------------------------------------------------------------------")

generate(bs = None)
print()

print("--------------------------------------------------------------------------------------")
print("With banned strings")
print("--------------------------------------------------------------------------------------")

generate(bs = banned_strings)
print()
