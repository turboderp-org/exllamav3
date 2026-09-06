"""
Full VRAM accounting for a loaded model + generator (exllamav3.util.memory.vram_accounting),
at three points: after load, after the Generator is built, and after a long prompt has been
prefilled and a few tokens generated (steady state). Reports every device the model occupies
(layer splits / TP), then a total. Same model arguments as chat.py; in chat.py the same report
is available as /vra.

    python tests/vram_accounting_.py -m /mnt/str/models/glm5.3-flash/exl3/2.05bpw -cs 131072 -p 40000
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))
import torch
from exllamav3 import model_init, Generator, Job
from exllamav3.util.memory import vram_accounting, format_vram_report
from exllamav3.generator.sampler import ArgmaxSampler
from chat_util import make_haystack_prompt

parser = argparse.ArgumentParser()
model_init.add_args(parser, default_cache_size = 32768)
parser.add_argument("-p", "--prompt_tokens", type = int, default = 40000)
parser.add_argument("-n", "--new_tokens", type = int, default = 32)
parser.add_argument("--chunk", type = int, default = 2048)
parser.add_argument("--recurrent_cache_gb", type = float, default = 4.0)
parser.add_argument("--no_color", action = "store_true")
parser.add_argument("--json", action = "store_true", help = "also dump the steady-state report as JSON")
args = parser.parse_args()
GiB = 1024 ** 3

def report(tag, model, cache, gen):
    print(f"\n===== {tag} =====")
    reps = vram_accounting(model, cache, gen)
    print(format_vram_report(reps, color = not args.no_color))
    return reps

for i in range(torch.cuda.device_count()):
    torch.cuda.reset_peak_memory_stats(i)
model, config, cache, tokenizer, *_ = model_init.init(args)
report("after load", model, cache, None)
gen = Generator(model = model, cache = cache, tokenizer = tokenizer, max_chunk_size = args.chunk,
                recurrent_cache_size = int(args.recurrent_cache_gb * GiB), cpu_cache_size = 0)
report("after generator", model, cache, gen)

prompt = make_haystack_prompt(args.prompt_tokens, tokenizer)[0]
ids = tokenizer.encode(prompt, add_bos = True, encode_special_tokens = True)
for i in range(torch.cuda.device_count()):
    torch.cuda.reset_peak_memory_stats(i)
t0 = time.time()
job = Job(input_ids = ids, max_new_tokens = args.new_tokens, sampler = ArgmaxSampler(), decode_special_tokens = True)
gen.enqueue(job)
while gen.num_remaining_jobs():
    gen.iterate()
torch.cuda.synchronize()
print(f"\nprompt {ids.shape[1]} tokens + {args.new_tokens} generated in {time.time() - t0:.1f}s")
reps = report(f"steady state after {ids.shape[1]}-token prompt", model, cache, gen)
if args.json:
    import json
    print(json.dumps([r.as_dict() for r in reps], indent = 2, default = str))
