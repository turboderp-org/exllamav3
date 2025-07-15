import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from exllamav3.util.file import disk_lru_cache, disk_lru_cache_clear
from exllamav3.util.progress import ProgressBar
from exllamav3.util.memory import free_mem
from exllamav3 import Config, Model, Cache, Tokenizer, model_init
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import torch.nn.functional as F
import math


@disk_lru_cache("get_dataset_text")
def get_dataset_text(spec: dict):
    assert spec["dataset"] == "wiki2", "Only wiki2 implemented atm"
    dataset_text = "\n\n".join(
        load_dataset("wikitext", "wikitext-2-raw-v1", split = "test")
        ["text"]
    )
    return dataset_text


def get_test_tokens(tokenizer, rows, eval_len = 2048, eval_stride = 512):
    with ProgressBar("Tokenizing", rows) as pb:
        dataset_spec = { "dataset": "wiki2" }
        eval_tokens = tokenizer.encode(get_dataset_text(dataset_spec))
        num_tokens = eval_tokens.shape[-1]
        seqs = []
        for a in range(0, num_tokens - eval_len, eval_stride):
            b = a + eval_len
            seqs.append(eval_tokens[:, a:b])
            pb.update(len(seqs))
            if len(seqs) >= rows:
                break
    return torch.cat(seqs, dim = 0)[:, :]


@torch.inference_mode()
def main(args):

    # Load model
    config = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer.from_config(config)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map = "auto",
        torch_dtype = torch.half if args.tight else None,
    )
    if args.tight:
        free_mem()
        model.half()
        free_mem()

    vocab_size = tokenizer.actual_vocab_size

    # Dataset
    eval_ids = get_test_tokens(tokenizer, args.rows, eval_len = args.length).cuda()

    # Test
    logprob_sum = 0.0
    logprob_count = 0
    with ProgressBar("Evaluating", args.rows) as pb:
        for row in range(eval_ids.shape[0]):
            pb.update(row)
            input_ids = eval_ids[row:row + 1, :]
            logits = model.forward(input_ids)["logits"]
            logits = logits[:, :-1, :vocab_size].float() + 1e-10
            log_probs = F.log_softmax(logits, dim = -1)
            del logits
            target_ids = input_ids[:, 1:].to(log_probs.device)
            del input_ids
            target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            logprob_sum += target_log_probs.sum().item()
            logprob_count += target_ids.numel()
            del target_log_probs
            del target_ids
            torch.cuda.empty_cache()
        pb.update(args.rows)
        mean_log_prob = logprob_sum / logprob_count
        perplexity = math.exp(-mean_log_prob)

    print(f" -- Model: {args.model_dir}")
    print(f" -- Loaded with Transformers")
    print(f" -- Evaluated: {eval_ids.shape[0]} rows of {eval_ids.shape[1]} tokens")
    print(f" -- Perplexity: {perplexity:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = False)
    parser.add_argument("-r", "--rows", type = int, help = "Number of rows", default = 100)
    parser.add_argument("-t", "--tight", action = "store_true", help = "Force FP16 dtype to save memory")
    parser.add_argument("-l", "--length", type = int, help = "Length", default = 2048)
    _args = parser.parse_args()
    main(_args)
