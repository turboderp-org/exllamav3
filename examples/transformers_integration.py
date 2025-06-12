import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForConditionalGeneration
from exllamav3.integration.transformers import patch_transformers

# At the moment, ExLlamaV3 integrates into Transformers by injecting a couple of classes into Transformers' lists
# of recognized quantization formats. Expect this method to change
patch_transformers()

@torch.inference_mode
def main():

    # Model ID. Currently, this needs to point to a local directory and models can't be loaded directly from the HF
    # Hub. All models supported by ExLlamaV3 _should_ work here, except for:
    #
    # Models with fused q/k/v or up/gate projections (e.g. Phi4) are currently not handled correctly. ExLlamaV3
    # un-fuses those layers during quantization.
    #
    # Nemotron-Ultra specifically can only be quantized by splitting a couple of extremely wide MLP layers into slices,
    # which breaks compatibility with the model implementation in Transformers
    model_id = "/mnt/str/models/llama3.1-70b-instruct/exl3/1.6bpw_H3/"

    # Create the AutoModel
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map = "auto")

    # Format and tokenize a prompt
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a very nice assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        tokenize = True,
        return_tensors = "pt",
        add_generation_prompt = True
    ).to(model.device)

    # Generate a response
    output_ids = model.generate(input_ids = input_ids, max_new_tokens = 100, do_sample = True, top_p = 0.8)
    output = tokenizer.decode(output_ids[0].tolist())
    print(output)

if __name__ == "__main__":
    main()