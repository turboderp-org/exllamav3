import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
from exllamav3 import Model, model_init, Generator, Job
from PIL import Image
import glob
from pathlib import (Path)

# ANSI codes
col_default = "\u001b[0m"
col_yellow = "\u001b[33;1m"
col_green = "\u001b[32;1m"


def resolve_files(input_path):
    input_path = Path(input_path)
    if input_path.is_dir():
        return [str(p) for p in input_path.rglob("*") if p.is_file()]
    elif input_path.is_file():
        return [str(input_path)]
    else:
        return [str(p) for p in glob.glob(str(input_path), recursive = True) if Path(p).is_file()]


@torch.inference_mode()
def main(args):

    # Resolve filenames
    input_files = []
    for arg in args.input:
        input_files += resolve_files(arg)

    # Prepare model etc.
    model, config, cache, tokenizer, draft_model, draft_config, draft_cache = model_init.init(args)
    generator = Generator(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
        draft_model = draft_model,
        draft_cache = draft_cache,
        num_draft_tokens = args.num_draft_tokens,
    )

    # Load the image component model
    vision_model = Model.from_config(config, component = "vision")
    vision_model.load(progressbar = True)

    # Process images
    for idx in range(len(input_files)):
        try:
            img = Image.open(input_files[idx])
        except (IOError, SyntaxError):
            # Skip non-image files and ignore other errors
            print(f"{col_yellow}Skipping: {input_files[idx]}{col_default}")
            continue

        embed = vision_model.get_image_embeddings(tokenizer, img)
        prompt = model.default_chat_prompt(f"{embed.text_alias}\n{args.prompt.strip()}")
        input_ids = tokenizer.encode(prompt, embeddings = [embed])

        job = Job(
            input_ids = input_ids,
            max_new_tokens = 2048,
            decode_special_tokens = True,
            stop_conditions = config.eos_token_id_list,
            embeddings = [embed],
        )

        generator.enqueue(job)

        print(f"{col_green}Image: {input_files[idx]}{col_default}")
        while generator.num_remaining_jobs():
            results = generator.iterate()
            for result in results:
                text = result.get("text", "")
                print(text, end = "", flush = True)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(parser, cache = True, default_cache_size = 16384, add_draft_model_args = True)
    parser.add_argument("-p", "--prompt", type = str, help = "Text prompt (default: Describe this image.)", default = "Describe this image.")
    parser.add_argument("input", nargs = "+", type = str, help = "Input files")
    _args = parser.parse_args()
    main(_args)
