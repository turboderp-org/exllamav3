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

MAX_DIM = 64  # maximum image dimension, in half-block pixels
UPPER_HALF = "▀"
LOWER_HALF = "▄"
RESET = "\x1b[0m"

def target_size(width, height, max_dim, max_cols):
    limit = min(max_dim, max_cols) if max_cols else max_dim
    scale = min(limit / width, max_dim / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def checkerboard(size, square = 4, light = (120, 120, 120, 255), dark = (80, 80, 80, 255)):
    """Backdrop for transparent images, so alpha reads as transparency, not black."""
    width, height = size
    board = Image.new("RGBA", size)
    board.putdata([
        light if ((x // square) + (y // square)) % 2 else dark
        for y in range(height) for x in range(width)
    ])
    return board


def img_load(path, max_dim, max_cols):
    img = Image.open(path)
    has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
    img = img.convert("RGBA" if has_alpha else "RGB")
    img = img.resize(target_size(*img.size, max_dim, max_cols), Image.LANCZOS)

    if has_alpha:
        img = Image.alpha_composite(checkerboard(img.size), img).convert("RGB")
    return img


def render(img):
    px = img.load()
    width, height = img.size

    for y in range(0, height, 2):
        parts = []
        last_fg = last_bg = None
        for x in range(width):
            top = px[x, y]
            if y + 1 < height:
                bottom = px[x, y + 1]
                if top != last_fg:
                    parts.append("\x1b[38;2;%d;%d;%dm" % top)
                    last_fg = top
                if bottom != last_bg:
                    parts.append("\x1b[48;2;%d;%d;%dm" % bottom)
                    last_bg = bottom
                parts.append(UPPER_HALF)
            else:
                # Odd height: draw the final row as a lower block on the
                # terminal's own background so nothing is invented below it.
                if last_bg is not None:
                    parts.append("\x1b[49m")
                    last_bg = None
                if top != last_fg:
                    parts.append("\x1b[38;2;%d;%d;%dm" % top)
                    last_fg = top
                parts.append(LOWER_HALF)
        parts.append(RESET)
        yield "".join(parts)


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

    print()

    # Process images
    for idx in range(len(input_files)):
        try:
            image_file = input_files[idx]
            img = Image.open(image_file)
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

        # Render the image
        print(f"{col_green}Image: {input_files[idx]}{col_default}")
        if not args.no_render:
            r_img = img_load(image_file, MAX_DIM, os.get_terminal_size().columns if sys.stdout.isatty() else 0)
            for line in render(r_img):
                print(line)

        # Generate
        while generator.num_remaining_jobs():
            results = generator.iterate()
            for result in results:
                text = result.get("text", "")
                print(text, end = "", flush = True)
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(parser, cache = True, default_cache_size = 16384, add_draft_model_args = True)
    parser.add_argument("-p", "--prompt", type = str, help = "Text prompt (default: Describe this image.)", default = "Describe this image.")
    parser.add_argument("-nr", "--no_render", action = "store_true", help = "Don't render images in the terminal")
    parser.add_argument("input", nargs = "+", type = str, help = "Input files")
    _args = parser.parse_args()
    main(_args)
