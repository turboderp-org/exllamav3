"""
Prompt-processing (pp) and token-generation (tg) throughput on the sm_75 port.

Reports the two numbers separately because they are bound by different things on Turing:
prefill is compute/shared-memory bound in the GEMM and attention kernels, decode is bound by
weight bandwidth (and, with MoE experts offloaded, by the PCIe/CPU handoff). A single
"tok/s" figure hides which one the port actually costs.

pp is measured from the generator's own prefill timing over a synthetic prompt of the
requested length, tg from the streaming phase only, so neither includes load or tokenization.
Greedy sampling throughout, so runs are comparable.

    python tests/sm75_ppttg.py <model_dir> [--pp 512,2048] [--tg 64] [--moe-cpu N]
"""

import argparse
import sys
import time

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dir")
    p.add_argument("--pp", default="512,2048", help="prompt lengths to measure, comma separated")
    p.add_argument("--tg", type=int, default=64, help="tokens to generate per run")
    p.add_argument("--moe-cpu", type=int, default=0, help="MoE layers to offload to system RAM")
    p.add_argument("--cache", type=int, default=8192, help="cache size in tokens")
    p.add_argument("--warmup", action="store_true", default=True)
    args = p.parse_args()

    pp_lens = [int(x) for x in args.pp.split(",") if x]

    props = torch.cuda.get_device_properties(0)
    from exllamav3.ext import exllamav3_ext as ext
    print(f"device: {props.name} sm_{props.major}{props.minor} x{torch.cuda.device_count()}  "
          f"smem_optin={ext.g_get_smem_max(0)}", flush=True)

    config = Config.from_directory(args.model_dir)
    if args.moe_cpu:
        config.infer_params.moe_cpu_offload = args.moe_cpu
        print(f"moe_cpu_offload: {args.moe_cpu} layers", flush=True)

    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=args.cache)

    t0 = time.time()
    model.load(progressbar=False)
    print(f"load: {time.time() - t0:.0f}s", flush=True)

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    def run(prompt_len, new_tokens):
        # Distinct token ids so nothing dedupes or hits the prefix cache across runs
        ids = torch.randint(
            1000, 20000, (1, prompt_len), dtype=torch.long
        )
        job = Job(input_ids=ids, max_new_tokens=new_tokens, sampler=GreedySampler())
        generator.enqueue(job)
        res = None
        n_stream = 0
        t_first = None
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                if r.get("stage") == "streaming":
                    if t_first is None:
                        t_first = time.time()
                    n_stream += 1
                if r.get("eos"):
                    res = r
        return res, n_stream, t_first

    if args.warmup:
        # First call pays kernel autotuning and Triton JIT; excluded from every figure below
        run(128, 8)

    print(f"\n{'prompt':>8}  {'pp tok/s':>10}  {'tg tok/s':>10}")
    print("-" * 32)
    for n in pp_lens:
        if n + args.tg > args.cache:
            print(f"{n:>8}  {'skipped (cache too small)':>22}")
            continue
        res, n_stream, t_first = run(n, args.tg)
        if res is None:
            print(f"{n:>8}  {'no result':>10}")
            continue
        # The generator reports prefill separately from generation, and discounts whatever
        # the page table already had cached
        new_ctx = res["prompt_tokens"] - res["cached_tokens"]
        pp = new_ctx / res["time_prefill"] if res["time_prefill"] > 0 else float("nan")
        tg = res["new_tokens"] / res["time_generate"] if res["time_generate"] > 0 else float("nan")
        print(f"{n:>8}  {pp:>10.1f}  {tg:>10.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
