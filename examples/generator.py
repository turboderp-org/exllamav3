import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, TopPSampler
from common import format_prompt, get_stop_conditions

"""
A couple of examples showing uses of the generator
"""

prompt_format = "llama3"  # see common.py
model_dir = "/mnt/str/eval_models/llama3.1-8b-instruct/exl3/4.0bpw/"
cache_size = 16384

system_prompt = "You are a very nice language model."

instructions = [
    "Write a short story beginning with the words 'Once in a while, when you least expect it'.",
    "Why are cats so awesome?",
    "Who was the tallest president of the United States?",
    "Why are there so many different kinds of screws?",
    "oinnvdoehwemnascnawwui8dh2",
    "Write a haiku about catnip."
]

# Generate a single completion to a single prompt
def generate_single(generator, tokenizer):
    instruction = instructions[0]
    print("------------------")
    print("Prompt: " + instruction)
    print()
    response = generator.generate(
        prompt = format_prompt(prompt_format, system_prompt, instruction),
        stop_conditions = get_stop_conditions(prompt_format, tokenizer),
        max_new_tokens = 500,
        completion_only = True,
        add_bos = True
    )
    print("Response: " + response)
    print()


# Generate multiple batched completions
def generate_batched(generator, tokenizer):
    print("------------------")
    responses = generator.generate(
        prompt = [format_prompt(prompt_format, system_prompt, instruction) for instruction in instructions],
        stop_conditions = get_stop_conditions(prompt_format, tokenizer),
        max_new_tokens = 100,
        completion_only = True,
        add_bos = True
    )
    for idx, response in enumerate(responses):
        print(f"#{idx + 1}: {response}")
        print("------------------")


# Create a job and generate a stream of tokens
def generate_streaming(generator, tokenizer):
    instruction = instructions[0]
    print("------------------")
    print("Prompt: " + instruction)
    print()
    print("Response: ", end = "", flush = True)

    # Create the job and enqueue it
    formatted_prompt = format_prompt(prompt_format, system_prompt, instruction)
    job = Job(
        input_ids = tokenizer.encode(formatted_prompt, add_bos = True),
        max_new_tokens = 400,
        stop_conditions = get_stop_conditions(prompt_format, tokenizer),
    )
    generator.enqueue(job)

    # Keep iterating until the generator has no more jobs
    while generator.num_remaining_jobs():
        results = generator.iterate()

        # Each iteration returns a list of results, each of which may contain output tokens for a running job. We
        # only care about the "text" field here.
        for result in results:
            text = result.get("text", "")
            print(text, end = "", flush = True)

    print()


# Create a batch of jobs and stream the results
def generate_streaming_batched(generator, tokenizer):

    # Some buffers for collecting results
    responses = [""] * len(instructions)

    for idx, instruction in enumerate(instructions):

        # Only print the second job to the console
        if idx == 1:
            print("------------------")
            print("Prompt: " + instruction)
            print()
            print("Streamed response: ", end = "", flush = True)

        # Create each job and enqueue it. Since one iteration of the generator can return multiple results, adding
        # an identifier argument lets us track which sequence each chunk of output pertains to. The identifier can
        # be any object, but a simple index will work here
        formatted_prompt = format_prompt(prompt_format, system_prompt, instruction)
        job = Job(
            input_ids = tokenizer.encode(formatted_prompt, add_bos = True),
            max_new_tokens = 400,
            stop_conditions = get_stop_conditions(prompt_format, tokenizer),
            identifier = idx,
        )
        generator.enqueue(job)

    # Keep iterating until the generator has no more jobs
    while generator.num_remaining_jobs():
        results = generator.iterate()

        for result in results:
            text = result.get("text", "")
            idx = result["identifier"]

            # If this result is from the first job, stream to the console
            if idx == 1:
                print(text, end = "", flush = True)

            # Collect results
            responses[idx] += text

    print()
    print("--------------")

    # Finally print all the collected results
    for idx, response in enumerate(responses):
        print(f"#{idx + 1}: {response}")
        print("------------------")


# Generate a series of completions with increasing temperature
def generate_temperature(generator, tokenizer):
    instruction = instructions[5]
    print("------------------")
    print("Prompt: " + instruction)
    print()
    temperature = 0.0
    while temperature <= 3.01:
        print(f"Temperature = {temperature:.2f}: ", end = "", flush = True)
        response = generator.generate(
            prompt = format_prompt(prompt_format, system_prompt, instruction),
            stop_conditions = get_stop_conditions(prompt_format, tokenizer),
            sampler = TopPSampler(temperature = temperature, top_p = 0.95, temperature_last = True),
            max_new_tokens = 100,
            completion_only = True,
            add_bos = True
        )
        print(response)
        print()
        temperature += 0.25


def main():

    # Load a model with cache
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = cache_size)
    model.load(progressbar = True)
    tokenizer = Tokenizer.from_config(config)

    # Create generator
    generator = Generator(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
    )

    # Do some things
    generate_single(generator, tokenizer)
    generate_batched(generator, tokenizer)
    generate_streaming(generator, tokenizer)
    generate_streaming_batched(generator, tokenizer)
    generate_temperature(generator, tokenizer)


if __name__ == "__main__":
    main()

