import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from PIL import Image
from common import format_prompt, get_stop_conditions
import requests
import torch
torch.set_printoptions(precision = 5, sci_mode = False, linewidth=200)

mode = "mistral3"
cache_size = 8192
streaming = True

match mode:
    case "gemma3":
        prompt_format = "gemma"
        model_dir = "/mnt/str/models/gemma3-4b-it/exl3/5.0bpw/"
    case "mistral3":
        prompt_format = "mistral"
        model_dir = "/mnt/str/models/mistral-small-3.1-24b-instruct-2503/exl3/4.0bpw/"

images = [
    # Cat
    {"file": "media/cat.png"},

    # Line drawing
    # {"file": "media/strawberry.png"},

    # Random photo from picsum
    # {"url": "https://picsum.photos/800/600"},

    # Unrandom photo from picsum
    # {"url": "https://fastly.picsum.photos/id/451/800/600.jpg?hmac=B0-st7nsgJ0F8ufKM5HjVwP-1y_vIL60R-PpNFLITiQ"}
]

system_prompt = "You are a very nice language model."
instruction = "Describe the image."

def get_image(file = None, url = None):
    assert (file or url) and not (file and url)
    if file:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, file)
        return Image.open(file_path)
    elif url:
        return Image.open(requests.get(url, stream = True).raw)

def main():

    # Load model with cache
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = cache_size)
    model.load(progressbar = True)
    tokenizer = Tokenizer.from_config(config)

    # Load the image component model
    vision_model = Model.from_config(config, component = "vision")
    vision_model.load(progressbar = True)

    # Create generator
    generator = Generator(
        model = model,
        cache = cache,
        tokenizer = tokenizer,
    )

    # Process images
    image_embeddings = [
        vision_model.get_image_embeddings(tokenizer = tokenizer, image = get_image(**img_args))
        for img_args in images
    ]

    # Build string of placeholder symbols that will be tokenized as image embeddings
    placeholders = "\n".join([ie.text_alias for ie in image_embeddings]) + "\n"

    # Format prompt
    prompt = format_prompt(prompt_format, system_prompt, placeholders + instruction)

    # Streaming response
    if streaming:
        input_ids = tokenizer.encode(
            prompt,
            encode_special_tokens = True,
            embeddings = image_embeddings,
        )

        job = Job(
            input_ids = input_ids,
            max_new_tokens = 500,
            decode_special_tokens = True,
            stop_conditions = get_stop_conditions(prompt_format, tokenizer),
            embeddings = image_embeddings,
        )

        generator.enqueue(job)

        print()
        print(prompt, end = "", flush = True)
        while generator.num_remaining_jobs():
            results = generator.iterate()
            for result in results:
                text = result.get("text", "")
                print(text, end = "", flush = True)
        print()

    # Non-streaming
    else:
        output = generator.generate(
            prompt = prompt,
            max_new_tokens = 500,
            encode_special_tokens = True,
            decode_special_tokens = True,
            stop_conditions = get_stop_conditions(prompt_format, tokenizer),
            embeddings = image_embeddings,
        )
        print(output)

if __name__ == "__main__":
    main()