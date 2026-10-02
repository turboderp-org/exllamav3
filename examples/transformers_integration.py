import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from exllamav3.integration.transformers import register

# Registers the EXL3 quantizer with Transformers (its own registration API; nothing else is patched). The
# "quant_method": "exl3" entry the converter writes into config.json then selects it in from_pretrained.
# Requires transformers >= 5
register()

# Any EXL3 model whose quantized tensors map one-to-one onto nn.Linear modules of the Transformers
# implementation, or onto a v5 experts container (per-expert gate/up/down). Not loadable: models whose HF
# implementation fuses projections the converter splits (fused qkv or gate_up), or wide layers the converter
# slices. Multimodal checkpoints with legacy key prefixes are mapped through the model's own renaming rules.
model_id = "/mnt/str/models/gemma4-12b-it/exl3/4.00bpw_mul1/"


def main():
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map = "auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    input_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a very nice assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        tokenize = True,
        return_tensors = "pt",
        return_dict = True,
        add_generation_prompt = True
    )["input_ids"].to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(input_ids = input_ids, max_new_tokens = 100, do_sample = True, top_p = 0.8)
    print(tokenizer.decode(output_ids[0].tolist()))

    # The EXL3 layers are differentiable with respect to their input (the quantized weights are frozen), so
    # gradients reach everything upstream: here the input embeddings, but equally norms, adapters or other
    # trainable modules added to the model. The loss is the model's own response scored under teacher forcing
    # (the prompt's template tokens are never training targets and would dominate a whole-sequence loss)
    prompt_len = input_ids.shape[1]
    output_ids = output_ids.clone()   # (generated under inference_mode; autograd needs a regular tensor)
    embeds = model.get_input_embeddings()(output_ids).detach().requires_grad_(True)
    logits = model(inputs_embeds = embeds).logits
    loss = torch.nn.functional.cross_entropy(logits[0, prompt_len - 1 : -1].float(), output_ids[0, prompt_len:])
    loss.backward()
    print(f"response loss {loss.item():.4f} over {output_ids.shape[1] - prompt_len} tokens, "
          f"gradient norm at the embeddings {embeds.grad.norm().item():.4f}")


if __name__ == "__main__":
    main()
