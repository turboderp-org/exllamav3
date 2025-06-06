import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import argparse
from exllamav3 import Model, model_init
from exllamav3.util.progress import ProgressBar
from PIL import Image
import glob
from pathlib import (Path)
from imgsearch_gallery import gallery

"""
Small example of using a VLM to query multiple images with the same yes/no question and collect all the positive
results. Note that only Gemma3 currently benefits from batch sizes higher than 1. The model is only sampled for one
token per image and only the relative probabilities of 'Yes' and 'No' are considered. A more advanced pipeline is
probably required to take full advantage of most models.

Example:

python examples/imgsearch.py /path/to/images -m /path/to/vlm -v -p "Are there any cats in this image?" 
"""

def resolve_files(input_path):
    input_path = Path(input_path)
    if input_path.is_dir():
        return [str(p) for p in input_path.rglob("*") if p.is_file()]
    elif input_path.is_file():
        return [str(input_path)]
    else:
        return [str(p) for p in glob.glob(str(input_path), recursive = True) if Path(p).is_file()]

def get_token_mask(tokenizer, substr):
    vocab = tokenizer.get_id_to_piece_list()
    substr1 = substr.upper()
    substr2 = substr.upper()
    mask = torch.tensor([(token.upper().startswith(substr1) or token.upper().startswith(substr2)) for token in vocab])
    return mask

@torch.inference_mode()
def main(args):

    # Resolve filenames
    input_files = []
    for arg in args.input:
        input_files += resolve_files(arg)

    # Prepare model etc.
    model, config, _, tokenizer = model_init.init(args)

    # Load the image component model
    vision_model = Model.from_config(config, component = "vision")
    vision_model.load(progressbar = True)

    batchsize = args.batchsize
    if batchsize > 1 and not vision_model.caps.get("fixed_size_image_embeddings"):
        print(" !! Cannot do batched image ingestion with this model, falling back to batch size 1")
        batchsize = 1

    # Output masks
    yes_mask = get_token_mask(tokenizer, "yes")
    no_mask = get_token_mask(tokenizer, "no")
    vocab_size = tokenizer.actual_vocab_size  # To account for padded logits

    # Results
    skipped_files = 0
    total_files = 0
    all_matches = []

    # Process images
    with ProgressBar(" -- Inference", count = len(input_files)) as pb:
        idx = 0
        batch_files = []
        batch_images = []

        while idx < len(input_files):

            try:
                img = Image.open(input_files[idx])
                batch_files.append(input_files[idx])
                batch_images.append(img)
                total_files += 1
            except (IOError, SyntaxError):
                # Skip non-image files and ignore other errors
                skipped_files += 1
            idx += 1

            if len(batch_images) == batchsize or idx == len(input_files):
                batch_embed = vision_model.get_image_embeddings(tokenizer, batch_images)
                batch_prompt = [
                    model.default_chat_prompt(f"{be.text_alias}\n{args.prompt.strip()}")
                    for be in batch_embed
                ]
                input_ids = tokenizer.encode(batch_prompt, embeddings = batch_embed)
                params = {
                    "last_tokens_only": 1,
                    "indexed_embeddings": batch_embed
                }
                logits = model.forward(input_ids, params)[:, :, :vocab_size]

                probs = logits.softmax(dim = -1)
                probs = probs.cpu()
                yes = torch.sum(probs * yes_mask, dim = -1)
                no = torch.sum(probs * no_mask, dim = -1)

                for m, filename in enumerate(batch_files):
                    if (not args.no and yes[m] > no[m]) or (args.no and yes[m] < no[m]):
                        print(f" -- Match: {filename}")
                        all_matches.append(filename)

                batch_files = []
                batch_images = []
                pb.update(idx)

    # Results
    print(f" -- Total files checked: {total_files:,}")
    print(f" -- Skipped files: {skipped_files:,}")
    if args.view:
        gallery(all_matches, args.prompt + (" (No)" if args.no else " (Yes)"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = False)
    parser.add_argument("-p", "--prompt", type = str, help = "Per-image prompt (yes/no question)", required = True)
    parser.add_argument("-n", "--no", action = "store_true", help = "Match images on 'No' instead of 'Yes'")
    parser.add_argument("-bsz", "--batchsize", type = int, help = "Batch size", default = 1)
    parser.add_argument("-v", "--view", action = "store_true", help = "View results after search")
    parser.add_argument("input", nargs = "+", type = str, help = "Input files")
    _args = parser.parse_args()
    main(_args)
