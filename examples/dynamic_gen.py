import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Model, Config, Cache, Tokenizer, Generator, Job, Sampler
from exllamav3.util import Timer
from blessed import Terminal
from common import format_prompt, get_stop_conditions
import pprint

"""
This is a demo and small showcase some of the features of the dynamic batching generator

Display modes for this demo:
1: One line per job, updated continuously
2: Print completions as jobs finish
3: Step over output iteration by iteration
4: Space heater mode (no output)
"""
display_mode = 1

# Show graphical visualization of the paged cache (adds some overhead)
show_visualization = False

# Where to find our model
model_dir = "/mnt/str/eval_models/llama3.1-8b-instruct/exl3/4.0bpw/"

# Total number of tokens to allocate space for in the cache.
total_context = 16384

# Max number of batches to run at once, assuming the sequences will fit within total_context.
max_batch_size = 16

# Max chunk size. Determines the size of prefill operations. Can be reduced to reduce pauses whenever a
# new job is started, but at the expense of overall prompt ingestion speed.
max_chunk_size = 2048

# Max new tokens per completion. For this example applies to all jobs.
max_new_tokens = 500

# Some prompts to feed the generator
prompt_format = "llama3"
system_prompt = "You are an AI assistant"
prompts = [
    "What is 2+2 and why?",
    "Can you guess the next number in this sequence: " + ", ".join(str(n) for n in range(500)),
    "Can you guess the next number in this sequence: " + ", ".join(str(n) for n in range(400)),
    "Can you guess the next number in this sequence: " + ", ".join(str(n) for n in range(200)),
    "Can you write a C++ quicksort implementation pretty please?",
    "Hello!",
    "What's the difference smoke and vapor?",
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 123 else 69) for n in range(200)),
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 42 else 111) for n in range(200)),
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 42 else 111) for n in range(200)),
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 42 else 111) for n in range(200)),
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 42 else 111) for n in range(200)),
    "Please guess the next 20 numbers in this sequence: " + ", ".join(str(n) for n in range(700)),
    "Write a short essay about cell membranes.",
    "How do I open a can of beans?",
    "How do I open a can of soup?",
    "How do I open a can of strawberry jam?",
    "How do I open a can of raspberry jam?",
    "What's the tallest building in Paris?",
    "What's the most populous nation on Earth?",
    "What's the most populous nation on Mars?",
    "What do the Mole People actually want and how can we best appease them?",
    "Why is the sky blue?",
    "Where is Waldo?",
    "Who is Waldo?",
    "Why is Waldo?",
    "Is it legal to base jump off the Eiffel Tower?",
    "Is it legal to base jump into a volcano?",
    "Why are cats better than dogs?",
    "Why is the Hulk so angry all the time?",
    "How do I build a time machine?",
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 123 else 69) for n in range(200)),
    "Is it legal to grow your own catnip?",
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 360 else 420) for n in range(400)),
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 361 else 421) for n in range(400)),
    "What's inside a black hole?",
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 360 else 420) for n in range(400)),
    "What seems out of place in this sequence: " + ", ".join(str(n if n != 363 else 421) for n in range(400)),
    "What do the numbers 2, 4, 8, 16, 32 and 64 have in common?",
    "What do the numbers 2, 3, 5, 7, 11 and 13 have in common?",
    "Is there life on Mars?",
    "Why are cats better than dogs?",
    "Write a parable about why cats are better than dogs.",
    "Can you guess the next number in this sequence: " + ", ".join(str(n) for n in range(999)),
    "Can you guess the next number in this sequence: " + ", ".join(str(n) for n in range(999)),
    "Can you guess the next number in this sequence: " + ", ".join(str(n) for n in range(999)),
    "Can you guess the next number in this sequence: " + ", ".join(str(n) for n in range(999)),
    "Can you guess the next number in this sequence: " + ", ".join(str(n) for n in range(999)),
]

term = Terminal()

def main():

    # Load the model config
    config = Config.from_directory("/mnt/str/models/llama3.1-8b-instruct/exl3/4.0bpw/")

    # Create the model from the config
    model = Model.from_config(config)

    # Create the cache before loading the model, so cache tensors are accounted for in the split
    cache = Cache(model, max_num_tokens = total_context)

    # Finally load the model. The default mode is autosplit.
    model.load()

    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_config(config)

    # Initialize the generator
    generator = Generator(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
        max_batch_size = max_batch_size,
        max_chunk_size = max_chunk_size,
        show_visualizer = show_visualization
    )

    # Create jobs
    jobs = []
    for prompt in prompts:
        fprompt =  format_prompt(prompt_format, system_prompt, prompt)
        input_ids = tokenizer.encode(fprompt, encode_special_tokens = True)
        job = Job(
            input_ids = input_ids,
            max_new_tokens = max_new_tokens,
            stop_conditions = get_stop_conditions(prompt_format, tokenizer)
        )
        jobs.append(job)

    # Enqueue all the jobs at once
    generator.enqueue(jobs)

    # Go
    match display_mode:

        # Mode 1
        case 1:
            class JobStatusDisplay:
                def __init__(self, job, console_line):
                    self.console_line = console_line
                    self.job = job
                    self.prefill = 0
                    self.max_prefill = 0
                    self.collected_output = ""
                    self.tokens = 0
                    self.spaces = " " * 80
                    text = term.darkgray(f"{self.console_line:3}:")
                    text += term.blue("enqueued")
                    print(term.move_xy(0, self.console_line) + text)

                def update(self, r):
                    stage = r["stage"]
                    stage = r.get("eos_reason", stage)
                    self.collected_output += r.get("text", "").replace("\n", "\\n")
                    token_ids = r.get("token_ids", None)
                    if token_ids is not None: self.tokens += token_ids.shape[-1]
                    self.prefill = r.get("curr_progress", self.prefill)
                    self.max_prefill = r.get("max_progress", self.max_prefill)
                    text = term.darkgray(f"{self.console_line:3}:")
                    text += term.blue(f"{stage:16}")
                    text += "prefill [ " + term.yellow(f"{self.prefill: 5} / {self.max_prefill: 5}") + " ]"
                    text += "   "
                    text += term.green(f"{self.tokens: 5} t")
                    text += term.darkgray(" -> ")
                    text += (self.spaces + self.collected_output)[-80:].replace("\t", " ")
                    if "accepted_draft_tokens" in r:
                        acc = r["accepted_draft_tokens"]
                        rej = r["rejected_draft_tokens"]
                        eff = acc / (acc + rej) * 100.0
                        text += term.bright_magenta(f"   SD eff.: {eff:6.2f}%")
                    print(term.move_xy(0, self.console_line) + text)

            print(term.enter_fullscreen())
            displays = { job: JobStatusDisplay(job, line) for line, job in enumerate(jobs) }
            while generator.num_remaining_jobs():
                results = generator.iterate()
                for r in results:
                    job = r["job"]
                    displays[job].update(r)
            print(term.move_xy(0, len(displays) + 1) + "Press any key to continue...")
            with term.cbreak():
                term.inkey()

        # Mode 2
        case 2:
            total_tokens = 0
            total_time = 0
            while generator.num_remaining_jobs():
                with Timer() as t:
                    results = generator.iterate()
                total_time += t.interval
                for r in results:
                    if r["stage"] == "streaming" and not r["eos"]:
                        total_tokens += r["token_ids"].shape[-1]
                for r in results:
                    if r["stage"] == "streaming" and r["eos"]:
                        job = r["job"]
                        in_prompt = \
                        tokenizer.decode(job.sequences[0].input_ids.torch(), decode_special_tokens = True)[0]
                        print("\n")
                        print(term.darkgray("Input: "))
                        print(term.yellow(in_prompt))
                        print()
                        print(term.darkgray("Output:"))
                        print(r["full_completion"])
                        print()
                        print(term.darkgray("New tokens:        ") + term.green(f"{r['new_tokens']:9} t"))
                        print(term.darkgray("Cached tokens:     ") + term.green(
                            f"{r['cached_tokens']:7} t / {r['prompt_tokens']:7} t"))
                        print(term.darkgray("Enqueued:          ") + term.blue(f"{r['time_enqueued']:9.2f} s"))
                        print(term.darkgray("Prefill:           ") + term.blue(f"{r['time_prefill']:9.2f} s"))
                        print(term.darkgray("Generation:        ") + term.blue(f"{r['new_tokens']:9.2f} s"))
                        speed_input = r['prompt_tokens'] / (r['time_prefill'] + 1e-10)
                        speed_output = r['new_tokens'] / (r['time_generate'] + 1e-10)
                        speed_total = total_tokens / total_time
                        print(term.darkgray("Job input          ") + term.cyan(f"{speed_input:9.2f} t/s"))
                        print(term.darkgray("Job output         ") + term.cyan(f"{speed_output:9.2f} t/s"))
                        print(term.darkgray("Overall output     ") + term.cyan(f"{speed_total:9.2f} t/s"))
                        if "accepted_draft_tokens" in r:
                            acc = r["accepted_draft_tokens"]
                            rej = r["rejected_draft_tokens"]
                            eff = acc / (acc + rej) * 100.0
                            print(term.darkgray("SD efficiency:     ") + term.bright_magenta(f"{eff:9.2f}%"))

        # Mode 3
        case 3:
            while generator.num_remaining_jobs():
                results = generator.iterate()
                print()
                pprint.pprint(results, indent = 4)
                print()
                print("Press any key to continue...")
                with term.cbreak():
                    term.inkey()

        case 4:
            while generator.num_remaining_jobs():
                generator.iterate()


if __name__ == "__main__":
    try:
        main()
    finally:
        pass
        if display_mode == 1:
            print(term.exit_fullscreen())