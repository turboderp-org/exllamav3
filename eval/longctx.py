import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from transformers import AutoTokenizer
from exllamav3.util.progress import ProgressBar
from exllamav3 import Config, Model, Cache, Tokenizer, model_init, Generator, Job, GreedySampler
import torch

# ANSI codes
ESC = "\u001b"
col_default = "\u001b[0m"
col_yellow = "\u001b[33;1m"
col_blue = "\u001b[34;1m"
col_green = "\u001b[32;1m"
col_red = "\u001b[31;1m"
col_gray = "\u001b[37;1m"

@torch.inference_mode()
def main(args):

    # Load model
    model, config, cache, tokenizer = model_init.init(args)
    generator = Generator(model, cache, tokenizer, show_visualizer = args.visualize_cache)
    bpw_layer, bpw_head, vram_bits = model.get_storage_info()

    print(f" -- Model: {args.model_dir}")
    print(f" -- Bitrate: {bpw_layer:.2f} bpw / {bpw_head:.2f} bpw (head)")

    # Load Transformers tokenizers
    t_tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # Get
    texts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_texts")
    with open(os.path.join(texts_dir, "illustrious_client.txt"), "r") as file:
        text_ic_orig = file.read()
    with open(os.path.join(texts_dir, "illustrious_client_c1.txt"), "r") as file:
        text_ic_french = file.read()
    with open(os.path.join(texts_dir, "illustrious_client_c2.txt"), "r") as file:
        text_ic_zoomer = file.read()
    with open(os.path.join(texts_dir, "illustrious_client_sum.txt"), "r") as file:
        text_ic_sum = file.read()
    with open(os.path.join(texts_dir, "variable_man_mod.txt"), "r") as file:
        text_vm_mod = file.read()
    with open(os.path.join(texts_dir, "variable_man_mod_c1.txt"), "r") as file:
        text_vm_pony = file.read()
    with open(os.path.join(texts_dir, "variable_man_sum.txt"), "r") as file:
        text_vm_sum = file.read()
    with open(os.path.join(texts_dir, "variable_man_char.txt"), "r") as file:
        text_vm_char = file.read()

    # Template
    def make_job(instruction):
        chat = [{
            "role": "user",
            "content": instruction
        }]
        input_ids = t_tokenizer.apply_chat_template(chat, add_generation_prompt = True)
        input_ids = torch.tensor(input_ids, dtype = torch.long).unsqueeze(0)
        job = Job(
            input_ids = input_ids,
            max_new_tokens = 768,
            stop_conditions = config.eos_token_id_list,
            sampler = GreedySampler()
        )
        return job, input_ids.shape[-1]

    # Tests
    # TODO: Find some original source material that models are sure to be entirely unfamiliar with
    job_ic_sum, len_ic_sum = make_job(text_ic_orig + "\n\n---\n\nProvide an extremely short summary of the story.")
    job_ic_french, _ = make_job(text_ic_french + "\n\n---\n\nOne paragraph in this story has been translated to a different language. Translate it back.")
    job_ic_zoomer, _ = make_job(text_ic_zoomer + "\n\n---\n\nTwo paragraphs have been rewritten in a zoomer slang style. Identify them.")
    job_vm_sum, len_vm_sum = make_job(text_vm_mod + "\n\n---\n\nProvide an extremely short summary of the story.")
    vm_q1 = "Why do the SRB computers stop giving reliable war-odds after Edward Milsom arrives in the 22nd century?"
    vm_a1 = "Milsom’s behavior is unpredictable to the machines because he comes from a different era and doesn’t fit their statistical patterns, so his presence introduces a “variable” they cannot factor."
    vm_q2 = "What does Edward Milsom secretly do to the Icarus bomb’s control turret?"
    vm_a2 = "Instead of wiring it to trigger an explosion, he rewires it so the craft can decelerate safely from faster-than-light speed, turning it from a bomb into a workable FTL drive."
    vm_q3 = "How does humanity ultimately benefit from Milsom’s interference, even though Earth loses the war against Jorblax?"
    vm_a3 = "Milsom’s solution delivers a practical faster-than-light return method, giving Earth true interstellar travel and opening the entire universe for exploration and colonization, making the war’s outcome irrelevant."
    job_vm_q1, _ = make_job(text_vm_mod + f"\n\n---\n\nAnswer in one paragraph: {vm_q1}")
    job_vm_q2, _ = make_job(text_vm_mod + f"\n\n---\n\nAnswer in one paragraph: {vm_q2}")
    job_vm_q3, _ = make_job(text_vm_mod + f"\n\n---\n\nAnswer in one paragraph: {vm_q3}")
    job_vm_char, _ = make_job(text_vm_mod + "\n\n---\n\nList all the named characters in the story.")
    job_vm_pony, _ = make_job(text_vm_pony+ "\n\n---\n\nA passage from an unrelated story is inserted in the middle of the text. Can you find it?")

    # Inference
    jobs = [
        job_ic_sum,
        job_ic_french,
        job_ic_zoomer,
        job_vm_sum,
        job_vm_q1,
        job_vm_q2,
        job_vm_q3,
        job_vm_pony,
        job_vm_char,
    ]
    generator.enqueue(jobs)

    with ProgressBar("Inference", len(jobs)) as pb:
        while j := generator.num_remaining_jobs():
            generator.iterate()
            pb.update(len(jobs) - j)

    # Results
    print()
    print(f"{col_green}------------{col_default}")
    print(f"{col_green}SUMMARY TEST{col_default}")
    print(f"{col_green}------------{col_default}")
    print(f"{col_blue}Short summary of 'The Illustrious Client', {len_ic_sum} tokens.\nReference summary:{col_default}")
    print(f"{col_gray}{text_ic_sum.strip()}{col_default}")
    print()
    print(job_ic_sum.full_completion.strip())
    print()

    print()
    print(f"{col_green}-----------{col_default}")
    print(f"{col_green}FRENCH TEST{col_default}")
    print(f"{col_green}-----------{col_default}")
    print(f"{col_blue}One paragraph in this story has been translated to a different language. Translate it back.{col_default}")
    print()
    print(job_ic_french.full_completion.strip())
    print()

    print()
    print(f"{col_green}-----------{col_default}")
    print(f"{col_green}ZOOMER TEST{col_default}")
    print(f"{col_green}-----------{col_default}")
    print(f"{col_blue}A zoomer has edited the text. Identify the edited passages.{col_default}")
    print()
    print(job_ic_zoomer.full_completion.strip())
    print()

    print()
    print(f"{col_green}------------{col_default}")
    print(f"{col_green}SUMMARY TEST{col_default}")
    print(f"{col_green}------------{col_default}")
    print(f"{col_blue}Short summary of a version of 'The Variable Man' with some names replaced, {len_vm_sum} tokens.\nReference summary:{col_default}")
    print(f"{col_gray}{text_vm_sum.strip()}{col_default}")
    print()
    print(job_vm_sum.full_completion.strip())
    print()

    print()
    print(f"{col_green}--------{col_default}")
    print(f"{col_green}Q&A TEST{col_default}")
    print(f"{col_green}--------{col_default}")
    print(f"{col_blue}{vm_q1} Reference answer:{col_default}")
    print(f"{col_gray}{vm_a1}{col_default}")
    print()
    print(job_vm_q1.full_completion.strip())
    print()
    print(f"{col_blue}{vm_q2} Reference answer:{col_default}")
    print(f"{col_gray}{vm_a2}{col_default}")
    print()
    print(job_vm_q2.full_completion.strip())
    print()
    print(f"{col_blue}{vm_q3} Reference answer:{col_default}")
    print(f"{col_gray}{vm_a3}{col_default}")
    print()
    print(job_vm_q3.full_completion.strip())
    print()

    print()
    print(f"{col_green}---------------{col_default}")
    print(f"{col_green}CORRUPTION TEST{col_default}")
    print(f"{col_green}---------------{col_default}")
    print(f"{col_blue}Some MLP fan fiction has made it into the story. Can we detect it?{col_default}")
    print()
    print(job_vm_pony.full_completion.strip())
    print()

    print()
    print(f"{col_green}--------------------{col_default}")
    print(f"{col_green}NAME EXTRACTION TEST{col_default}")
    print(f"{col_green}--------------------{col_default}")
    print(f"{col_blue}List all the named characters in the story.\nReference:{col_default}")
    print(f"{col_gray}{text_vm_char}{col_default}")
    print()
    print(job_vm_char.full_completion.strip())
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, default_cache_size = 65536)
    parser.add_argument("-vis", "--visualize_cache", action = "store_true", help = "Show cache visualizer (slow)")
    _args = parser.parse_args()
    main(_args)
