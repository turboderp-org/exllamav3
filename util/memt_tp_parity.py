"""
MemT under tensor parallelism: commit a fixed turn from the zero state and generate with it, single-device
or TP, and save the committed state so two runs can be compared bit for bit (or near it).

    python util/memt_tp_parity.py MODEL_DIR --out /tmp/single.safetensors
    python util/memt_tp_parity.py MODEL_DIR --tp --out /tmp/tp.safetensors
    python util/memt_tp_parity.py --compare /tmp/single.safetensors /tmp/tp.safetensors
"""

import sys, argparse, time
import torch

if __name__ == "__main__":   # TP workers spawn and re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument("memt_dir", nargs = "?")
    ap.add_argument("--split", default = "20,23")
    ap.add_argument("--tp", action = "store_true")
    ap.add_argument("--out", default = None)
    ap.add_argument("--compare", nargs = 2, default = None)
    args = ap.parse_args()

    if args.compare:
        from safetensors.torch import load_file
        a, b = load_file(args.compare[0]), load_file(args.compare[1])
        worst = 0.0
        for k in sorted(a):
            d = (a[k].float() - b[k].float()).abs().max().item()
            ref = a[k].float().abs().max().item()
            print(f"{k}: max|a|={ref:.4f} max|a-b|={d:.2e}")
            worst = max(worst, d / max(ref, 1e-12))
        print(f"worst relative max diff: {worst:.2e}")
        print("STATES MATCH" if worst < 1e-3 else "STATES DIFFER")
        sys.exit(0 if worst < 1e-3 else 1)

    from exllamav3 import Config, Model, Tokenizer, Cache, Generator, Job
    from exllamav3.generator.sampler import ArgmaxSampler

    cfg = Config.from_directory(args.memt_dir)
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens = 8192)
    t0 = time.time()
    model.load(progressbar = False, use_per_device = [float(v) for v in args.split.split(",")], tensor_p = args.tp)
    print(f"loaded (tp={args.tp}) in {time.time() - t0:.1f}s", flush = True)
    tok = Tokenizer.from_config(cfg)
    gen = Generator(model, cache, tok, memt_memory = False)
    enc = lambda s: tok.encode(s, add_bos = False, encode_special_tokens = True)

    facts = ("<|im_start|>user\nRemember: my cat is Quillon, I live in Tromsø, my sister is Beatrix and my locker code is 4817."
             "<|im_end|>\n<|im_start|>assistant\nNoted.<|im_end|>\n")
    t0 = time.time()
    state = model.memt_commit(enc(facts), model.memt_zero_state())
    print(f"commit of {enc(facts).shape[1]} tokens: {time.time() - t0:.2f}s", flush = True)
    if args.out:
        model.memt_save_state(state, args.out)
        print(f"state saved to {args.out}", flush = True)

    prompt = ("<|im_start|>system\nAnswer briefly.<|im_end|>\n<|im_start|>user\nList everything you remember about me."
              "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    def run(st, n = 120):
        job = Job(input_ids = enc(prompt), max_new_tokens = n, sampler = ArgmaxSampler(),
                  stop_conditions = [tok.eos_token_id, "<|im_end|>"], memt_state = st,
                  memt_key = (b"\x07" * 16 if st is not None else None))
        gen.enqueue(job); text = ""; nt = 0; t1 = None
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r["stage"] == "streaming":
                    if t1 is None: t1 = time.time()
                    text += r.get("text", ""); nt += 1
        return text.strip(), nt / max(time.time() - t1, 1e-9)
    run(state, 10)
    txt, tps = run(state)
    print(f"with memory: {tps:.1f} tok/s | {txt[:200]!r}", flush = True)
    ok = all(w in txt for w in ("Quillon", "Tromsø", "Beatrix", "4817"))
    txt0, tps0 = run(None)
    print(f"no memory:   {tps0:.1f} tok/s | {txt0[:120]!r}", flush = True)
    print("RECALL PASS" if ok else "RECALL FAIL", flush = True)
    if args.tp:
        model.unload_tp()
    sys.exit(0 if ok else 1)
