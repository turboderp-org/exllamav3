import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from exllamav3.util.file import disk_lru_cache, disk_lru_cache_clear
from exllamav3.util.progress import ProgressBar
from exllamav3.util.memory import free_mem
from exllamav3 import Config, Model, Cache, Tokenizer, model_init
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
    # TODO: inplace softmax, reduce max_output_factor to 3
    model, config, _, tokenizer = model_init.init(
        args,
        override_dynamic_seq_len = 2048,
        max_output_size = 2048,
        max_output_factor = 5,
    )

    vocab_size = tokenizer.actual_vocab_size
    bpw_layer, bpw_head, vram_bits = model.get_storage_info()

    # Dataset
    eval_ids = get_test_tokens(tokenizer, args.rows)

    # Test
    logprob_sum = 0.0
    logprob_count = 0
    with ProgressBar("Evaluating", args.rows) as pb:
        for row in range(eval_ids.shape[0]):
            pb.update(row)
            input_ids = eval_ids[row:row + 1, :]
            logits = model.forward(input_ids, {"attn_mode": "flash_attn_nc"})
            logits = logits[:, :-1, :vocab_size].float()
            logits += 1e-10
            log_probs = F.log_softmax(logits, dim = -1)
            del logits
            target_ids = input_ids[:, 1:].to(log_probs.device)
            del input_ids
            target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            logprob_sum += target_log_probs.sum().item()
            logprob_count += target_ids.numel()
            del log_probs
            del target_log_probs
            del target_ids
            torch.cuda.empty_cache()
        pb.update(args.rows)
        mean_log_prob = logprob_sum / logprob_count
        perplexity = math.exp(-mean_log_prob)

    print(f" -- Model: {args.model_dir}")
    print(f" -- Bitrate: {bpw_layer:.2f} bpw / {bpw_head:.2f} bpw (head)")
    print(f" -- Evaluated: {eval_ids.shape[0]} rows of {eval_ids.shape[1]} tokens")
    print(f" -- Perplexity: {perplexity:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = False)
    parser.add_argument("-r", "--rows", type = int, help = "Number of rows", default = 100)
    _args = parser.parse_args()
    main(_args)
