import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from PIL import Image
from common import format_prompt, get_stop_conditions
import requests
import torch
torch.set_printoptions(precision = 5, sci_mode = False, linewidth=200)

mode = "glm"
cache_size = 8192
streaming = True

match mode:
    case "gemma3":
        prompt_format = "gemma"
        model_dir = "/mnt/str/models/gemma3-4b-it/exl3/5.0bpw/"
    case "mistral3":
        prompt_format = "mistral"
        model_dir = "/mnt/str/models/mistral-small-3.1-24b-instruct-2503/exl3/4.0bpw/"
    case "qwen3":
        prompt_format = "chatml"
        model_dir = "/mnt/str/models/qwen3-vl-30b-a3b-instruct/exl3/5.00bpw"
    case "glm":
        prompt_format = "glmv"
        model_dir = "/mnt/str/models/glm4.5v/exl3/4.00bpw"

images = [
    # Cat
    {"file": "media/cat.png"},

    # Line drawing
    # {"file": "media/strawberry.png"},

    # Random photo from picsum
    # {"url": "https://picsum.photos/800/600"},

    # Unrandom photo from picsum
    # {"url": "https://fastly.picsum.photos/id/451/800/600.jpg?hmac=B0-st7nsgJ0F8ufKM5HjVwP-1y_vIL60R-PpNFLITiQ"}

    # Qwen3 demo image
    # {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
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

    # Config for text and vision model
    config = Config.from_directory(model_dir)

    # Load the image component model (can also be loaded after main model)
    vision_model = Model.from_config(config, component = "vision")
    vision_model.load(progressbar = True)

    # Load the text model
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = cache_size)
    model.load(progressbar = True)
    tokenizer = Tokenizer.from_config(config)

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
            max_new_tokens = 1000,
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