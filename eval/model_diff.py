import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from exllamav3.util.file import disk_lru_cache, disk_lru_cache_clear
from exllamav3.util.progress import ProgressBar
from exllamav3.util.memory import free_mem
from exllamav3 import Config, Model, Cache, Tokenizer
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

    config_a = Config.from_directory(args.model_a)
    config_a.override_dynamic_seq_len(2048)
    tokenizer = Tokenizer.from_config(config_a)
    model_a = Model.from_config(config_a)

    config_b = Config.from_directory(args.model_b)
    config_b.override_dynamic_seq_len(2048)
    model_b = Model.from_config(config_b)

    # Dataset
    eval_ids = get_test_tokens(tokenizer, args.rows)
    state_a = eval_ids
    state_b = eval_ids

    for idx, (module_a, module_b) in enumerate(zip(model_a.modules, model_b.modules)):

        module_a.load("cuda:0" if not module_a.caps.get("prefer_cpu") else "cpu")
        params_a = {}
        state_a = module_a.prepare_for_device(state_a, params_a)
        state_a = module_a.forward(state_a, params_a)
        module_a.unload()
        free_mem()

        module_b.load("cuda:0" if not module_b.caps.get("prefer_cpu") else "cpu")
        params_b = {}
        state_b = module_b.prepare_for_device(state_b, params_b)
        state_b = module_b.forward(state_b, params_b)
        module_b.unload()
        free_mem()

        if idx < args.keep_b:
            state_a = state_b.clone()

        max_diff = 0
        rfn_error_sum = 0
        rows = state_a.shape[0]
        for i in range(rows):
            sa = state_a[i].to(float, copy = True)
            sb = state_b[i].to(float)
            sa -= sb
            rfn_error_sum += (torch.linalg.norm(sa, 'fro') / torch.linalg.norm(sb, 'fro').mean()).item()
            sa.abs_()
            md = ((sa.max().item()) / torch.linalg.norm(sb, 'fro').mean()).item()
            max_diff = max(max_diff, md)

        rfn_error = rfn_error_sum / rows
        print(f" -- {module_a.key:40}   error: {rfn_error:.6f}   max_diff/norm: {max_diff:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ma", "--model_a", type = str, help = "Model A", required = True)
    parser.add_argument("-mb", "--model_b", type = str, help = "Model B", required = True)
    parser.add_argument("-r", "--rows", type = int, help = "Number of rows", default = 100)
    parser.add_argument("-kb", "--keep_b", type = int, help = "Maintain B state for number of modules", default = 0)
    _args = parser.parse_args()
    main(_args)
