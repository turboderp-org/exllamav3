import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler
import random

"""
This script creates a generator and runs varying queue depths of completion requests forever, to verify that
the paging and caching logic is sound. Each prompt is an increasing sequence of integers and every completion
is verified as a correct continuation of the sequence.   
"""

model_dir = "/mnt/str/eval_models/llama3.1-8b-instruct/exl3/4.0bpw/"
cache_size = 16384
draft_model_dir = None
prompt_len = (50, 8192)
completion_len = (50, 2048)
target_q_depth = (0, 20)
prefixes = ["All the numbers: ", "It never ends: ", "Counting forever: "]
suffix = ", ".join([str(i) for i in range(prompt_len[1])])
random.seed(0)

if draft_model_dir:
    draft_config = Config.from_directory(draft_model_dir)
    draft_model = Model.from_config(draft_config)
    draft_cache = Cache(draft_model, max_num_tokens = cache_size)
    draft_model.load("cuda:2")
else:
    draft_model, draft_cache = None, None

config = Config.from_directory(model_dir)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens = cache_size)
model.load("cuda:1")

tokenizer = Tokenizer.from_config(config)

generator = Generator(
    model = model,
    cache = cache,
    draft_model = draft_model,
    draft_cache = draft_cache,
    tokenizer = tokenizer,
)

def start_new_job():
    prefix = prefixes[random.randint(0, len(prefixes) - 1)]
    prompt = (prefix + suffix)[:random.randint(prompt_len[0], prompt_len[1])]
    prompt = prompt[:prompt.rfind(",") + 1]
    input_ids = tokenizer.encode(prompt, add_bos = True)
    job = Job(
        input_ids = input_ids,
        max_new_tokens = random.randint(completion_len[0], completion_len[1]),
        sampler = ArgmaxSampler(),
        identifier = prompt
    )
    generator.enqueue(job)


def is_consecutive_integers(s: str) -> bool:
    nums = [int(x.strip()) for x in s.split(',')]
    return all(nums[i + 1] == nums[i] + 1 for i in range(len(nums) - 1))


def iterate():
    num_active = generator.num_active_jobs()
    num_pending = generator.num_pending_jobs()
    results = generator.iterate()
    for result in results:
        if result["eos"]:
            cached_tokens = result["cached_tokens"]
            cached_pages = result["cached_pages"]
            print(
                f"{str(result['job'])}  pending: {num_pending}  active: {num_active}  "
                f"cached_p: {cached_pages}  cached_t: {cached_tokens}  -  ",
                end = ""
            )
            full = result["identifier"] + result["full_completion"]
            full = full[full.find(": ") + 2:]
            full = full[:full.rfind(",")]
            try:
                ok = is_consecutive_integers(full)
            except:
                ok = False

            if ok:
                print("OK!")
            else:
                print("Sus!")
                print("--------")
                print(full)
                print("--------")

# Main loop
next_target_q_depth = 0
while True:

    # Iterate until target q depth is reached
    if generator.num_remaining_jobs() > next_target_q_depth:
        print(f" - Generating, target depth {next_target_q_depth}")
    while generator.num_remaining_jobs() > next_target_q_depth:
        iterate()
        # print ("iter:", generator.num_remaining_jobs())
    next_target_q_depth = random.randint(target_q_depth[0], target_q_depth[1])

    # Start new jobs until target queue depth is achieved
    if generator.num_remaining_jobs() < next_target_q_depth:
        print(f" - Creating jobs, target depth {next_target_q_depth}")
    while generator.num_remaining_jobs() < next_target_q_depth:
        start_new_job()
        # print ("add:", generator.num_remaining_jobs())
    next_target_q_depth = random.randint(target_q_depth[0], generator.num_remaining_jobs() - 1)
