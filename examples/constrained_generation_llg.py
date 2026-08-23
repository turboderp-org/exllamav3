import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, Filter, LLGuidanceFilter

def get_superhero_filter(tokenizer) -> list[Filter]:

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "secret_identity": {"type": "string"},
            "gender": {"enum": ["male", "female"]},
            "superpowers": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            "first_appearance": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "issue_number": {"type": "integer"},
                    "year": {"type": "integer"},
                },
                "required": ["title", "issue_number", "year"],
                "additionalProperties": False,
            },
        },
        "required": ["name", "secret_identity", "gender", "superpowers", "first_appearance"],
        "additionalProperties": False,
    }
    filters = [LLGuidanceFilter(tokenizer, eos_after_completed = True, json_schema = schema)]

    # Additional constraint to force leading { and pretty-printing style
    filters += [LLGuidanceFilter(tokenizer, lark_grammar = 'start: "{\\n"')]

    # Test triggered filter, triggers on "Bruce" (ID 79579 in Llama3.1 vocab)
    filters += [LLGuidanceFilter(tokenizer, trigger_token = 79579, lark_grammar = 'start: " Thomas"')]

    return filters


def get_arithmetic_filter(tokenizer) -> list[Filter]:

    arithmetic_grammar = r"""
start: expression " = " expression "\n"
expression: term (("+" | "-") term)*
term: factor (("*" | "/") factor)*
factor: NUMBER | "(" expression ")"
NUMBER: /[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?/
"""
    return [LLGuidanceFilter(tokenizer, eos_after_completed = True, lark_grammar = arithmetic_grammar)]


def stream_gen(generator, tokenizer, prompt, filters):

    # Create job
    job = Job(
        input_ids = tokenizer.encode(prompt, add_bos = True),
        filters = filters,
        max_new_tokens = 400,
    )
    generator.enqueue(job)

    print("----------------------")
    print(prompt, end = "")

    while generator.num_remaining_jobs():
        results = generator.iterate()
        for result in results:
            text = result.get("text", "")
            print(text, end = "", flush = True)
    print()


def main():

    # Load model etc.
    model_dir = "/mnt/str/models/llama3.1-8b-instruct/exl3/4.0bpw/"
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = 8192)
    model.load()
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer)

    # Single gens can reuse filters
    sh_filter = get_superhero_filter(tokenizer)
    stream_gen(generator, tokenizer, "Here is vital information about Superman, in JSON format:\n\n", sh_filter)
    stream_gen(generator, tokenizer, "Here is vital information about Batman, in JSON format:\n\n", sh_filter)
    ar_filter = get_arithmetic_filter(tokenizer)
    stream_gen(generator, tokenizer, "Number of seconds in a century: 100*", ar_filter)  # (Llama3.1 can't math)
    stream_gen(generator, tokenizer, "Two plus two: 2+", ar_filter)

    # But filters are stateful, so we need multiple instances for batched gen
    sh_filter2 = get_superhero_filter(tokenizer)
    ar_filter2 = get_arithmetic_filter(tokenizer)
    batched_gens = generator.generate(
        prompt = [
            "Here is vital information about Superman, in JSON format:\n\n",
            "Here is vital information about Batman, in JSON format:\n\n",
            "Number of seconds in a century: 100*",
            "Two plus two: 2+",
        ],
        filters = [
            sh_filter,
            sh_filter2,
            ar_filter,
            ar_filter2,
        ],
        max_new_tokens = 400,
        add_bos = True,
    )
    for g in batched_gens:
        print("----------------------")
        print(g)


if __name__ == "__main__":
    main()
