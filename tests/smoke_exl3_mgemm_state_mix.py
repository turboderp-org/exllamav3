import argparse
import json
import random
import torch

from PIL import Image

from exllamav3 import Cache, Config, Job, Model, Tokenizer
from exllamav3.generator import Generator
from exllamav3.generator.sampler import GreedySampler


TEXT_PROMPTS = {
    "text_short": "Answer with one short word: hello.",
    "text_repeat": "Repeat the word alpha exactly three times, separated by spaces.",
    "text_long": "Summarize this sentence in one word: " + " ".join(["curious"] * 256),
}

KINDS = [
    "text_short",
    "text_repeat",
    "text_long",
    "mm_single_cat",
    "mm_single_fruit",
    "mm_dup_same",
    "mm_pair_animal",
]


def run_job(generator, tokenizer, prompt_ids, embeddings=None):
    job = Job(
        input_ids=prompt_ids.cpu(),
        max_new_tokens=16,
        stop_conditions=[tokenizer.eos_token_id, "<turn|>"],
        decode_special_tokens=True,
        embeddings=embeddings,
        sampler=GreedySampler(),
    )
    generator.enqueue(job)
    output = ""
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            if result.get("stage") == "streaming":
                output += result.get("text", "")
    return output


def run_text(generator, tokenizer, text):
    prompt_ids = tokenizer.hf_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": text}]}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return run_job(generator, tokenizer, prompt_ids)


def run_mm(generator, tokenizer, embeddings, question):
    prompt_ids = tokenizer.hf_chat_template(
        [
            {
                "role": "user",
                "content": ([{"type": "image"} for _ in embeddings] + [{"type": "text", "text": question}]),
            }
        ],
        add_generation_prompt=True,
        enable_thinking=False,
        embeddings=embeddings,
    )
    return run_job(generator, tokenizer, prompt_ids, embeddings=embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--reserve_per_device", default="0,0")
    parser.add_argument("--seed", type=int, default=20260407)
    parser.add_argument("--nreq", type=int, default=64)
    args = parser.parse_args()

    reserve = [float(x) for x in args.reserve_per_device.split(",")]
    rng = random.Random(args.seed)
    seq = [rng.choice(KINDS) for _ in range(args.nreq)]

    cfg = Config.from_directory(args.model_dir)
    model = Model.from_config(cfg)
    vision_model = Model.from_config(cfg, component="vision")
    tokenizer = Tokenizer.from_config(cfg)
    cache = Cache(
        model,
        max_num_tokens=4096,
        layer_type=model.caps.get("quantized_kv_cache_layer"),
        k_bits=4,
        v_bits=4,
    )

    cat_image = "/nvme512g/util/exllamav3/doc/cat.png"
    fruit_image = "/nvme512g/util/exllamav3/examples/media/strawberry.png"

    vision_model.load(torch.device("cuda:0"))
    cat = vision_model.get_image_embeddings(tokenizer, Image.open(cat_image).convert("RGB"))
    cat_dup = vision_model.get_image_embeddings(tokenizer, Image.open(cat_image).convert("RGB"))
    fruit = vision_model.get_image_embeddings(tokenizer, Image.open(fruit_image).convert("RGB"))
    vision_model.unload()

    model.load(reserve_per_device=reserve)
    generator = Generator(model, cache, tokenizer)

    mm_map = {
        "mm_single_cat": ([cat], "What animal is shown? Answer with one word."),
        "mm_single_fruit": ([fruit], "What fruit is shown? Answer with one word."),
        "mm_dup_same": ([cat, cat_dup], "Do the two images show the same subject? Answer yes or no."),
        "mm_pair_animal": ([cat, fruit], "Which image shows an animal? Answer first or second."),
    }

    results = []
    for i, kind in enumerate(seq, start=1):
        if kind.startswith("text_"):
            out = run_text(generator, tokenizer, TEXT_PROMPTS[kind])
        else:
            emb, q = mm_map[kind]
            out = run_mm(generator, tokenizer, emb, q)
        record = {"idx": i, "kind": kind, "out": out[:80]}
        results.append(record)
        print(json.dumps(record), flush=True)

    print(json.dumps({"status": "OK", "nreq": len(results)}))


if __name__ == "__main__":
    main()
